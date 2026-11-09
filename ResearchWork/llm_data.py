"""
Data for the frozen-LLM experiments: jsonl records → (context, question, answer) examples → segment tensors.

Fixes vs the legacy trainer:  P6 shuffled+seeded split (not glob-order tail)  ·  P7 prompt and answer are tokenised
SEPARATELY and joined as ids, so the answer boundary is exact (no BPE merge across it).

Batch layout (all segments are exactly `seg_len` tokens so they can be run as independent sequences):
    ids    (B, n+1, S)   n context segments, then 1 question segment (prompt, answer, pad)
    write  (B, (n+1)·S)  1 where a token may write to memory (real tokens), 0 on padding
    The context is LEFT-padded to n·S with the pad id (state is still zero there, so padding costs nothing),
    the question segment is right-padded. No attention mask is used anywhere: pads sit before real tokens
    (left) or after them (right), so causal attention never lets a real token see a pad that matters.
    ans_pos (B,A)  positions in the question segment whose hidden state predicts the A answer tokens
    ans_tok (B,A), ans_valid (B,A)
"""

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from llm_config import LLMConfig

PROMPT = "\n\nQuestion: {q}\nAnswer:"
DEVICES = ["flow meter", "pressure sensor", "compressor", "cooling pump", "control valve", "generator", "transformer", "gearbox"]


def _span_tokens(offsets, a, b, limit):
    return [i for i, (s_, e_) in enumerate(offsets) if e_ > a and s_ < b and i < limit]


def annotation_tokens(tok, ex, ctx_ids, prompt_ids):
    """Programmatic documents only: token indices of entity mentions in the context and the question, and of the first token of each
    candidate value ([answer] + distractors). Used ONLY by the oracle arm D (pool mask) and the auxiliary key loss (training signal);
    never needed at inference by the pooled arms."""
    import re as _re
    q = ex["question"]
    devs = [d_ for d_ in DEVICES if d_ in q]
    if not devs or not ex["meta"].get("annot"):
        return None
    dev = devs[0]
    enc = tok(ex["context"], add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"][:len(ctx_ids)] != ctx_ids:
        return None
    off, L = enc["offset_mapping"], len(ctx_ids)
    pat = _re.escape(dev) + r" at ([A-Z][a-z]+ [0-9A-Z]+)"
    ent_ctx = sorted({t for m in _re.finditer(pat, ex["context"]) for t in _span_tokens(off, m.start(), m.end(), L)})
    p_text = PROMPT.format(q=q)
    penc = tok(p_text, add_special_tokens=False, return_offsets_mapping=True)
    mq = _re.search(pat, p_text)
    ent_q = _span_tokens(penc["offset_mapping"], mq.start(), mq.end(), len(prompt_ids)) if mq else []
    a = ex["meta"]["annot"]
    val_first = []
    for v in [ex["answer"]] + list(a.get("distractor_values", [])):
        c = ex["context"].find(v)
        vt = _span_tokens(off, c, c + len(v), L) if c >= 0 else []
        val_first.append(vt[0] if vt else None)
    return {"ent_ctx": ent_ctx, "ent_q": ent_q, "val_first": val_first}


# ── record → example (pure python; no tokenizer needed) ───────────────────────

def to_example(rec: dict) -> Optional[dict]:
    """None for malformed records (the real data has 1 long_context record with only `context`)."""
    try:
        return _to_example(rec)
    except (KeyError, TypeError, IndexError):
        return None


def _to_example(rec: dict) -> Optional[dict]:
    t = rec.get("_type")
    fact_char = None
    if t == "long_context":
        ctx, q, a = rec["context"].strip(), rec["question"], rec["answer"].strip()
        meta = {"type": t, "fact_position": rec.get("fact_position"), "distractors": rec.get("distractor_count"), "annot": rec.get("annot")}
        wrong = None
        i = ctx.find(a)
        fact_char = i if i >= 0 else None
    elif t == "dialogue":
        lines = [f"{x['role'].capitalize()}: {x['content']}" for x in rec["turns"]]
        ctx = "\n".join(lines)
        q, a = rec["test_question"], rec["correct_answer"]
        anchors = rec.get("memory_anchors") or [{}]
        est = anchors[0].get("established_at_turn")
        fact_char = sum(len(l) + 1 for l in lines[:est]) if isinstance(est, int) and 0 <= est < len(lines) else None
        gap = (anchors[0].get("recalled_at_turn", 0) or 0) - (anchors[0].get("established_at_turn", 0) or 0)
        meta = {"type": t, "gap": gap}
        wrong = None
    elif t == "agentic":
        dp = rec["decision_point"]
        steps = [s for s in rec["steps"] if s["step"] < dp["at_step"]]        # only what the agent has seen so far
        header = f"Task: {rec['task']}\n"
        step_lines = [f"Step {s['step']} [{s['action']}] input: {s['input']} output: {s['output']}" for s in steps]
        ctx = header + "\n".join(step_lines)
        idx = next((k for k, s in enumerate(steps) if s["step"] == dp["requires_memory_of_step"]), None)
        fact_char = len(header) + sum(len(l) + 1 for l in step_lines[:idx]) if idx is not None else None
        q = f"At step {dp['at_step']}, what is the correct decision?"
        a, wrong = dp["correct_decision"], dp.get("wrong_decision_if_forgotten")
        meta = {"type": t, "distance": dp["at_step"] - dp["requires_memory_of_step"]}
    else:
        return None
    if not (ctx and q and a):
        return None
    ctx = ctx.strip()                                # (strip only trims trailing/leading blanks; fact_char computed on stripped text above)
    return {"context": ctx, "question": q.strip(), "answer": a.strip(), "wrong": wrong, "meta": meta,
            "fact_char": fact_char}


def load_split(cfg: LLMConfig) -> Dict[str, Dict[str, List[dict]]]:
    """{type: {"train": [...], "val": [...], "gen": [...]}} — shuffled with cfg.seed, deterministic."""
    out = {}
    for t in cfg.types.split(","):
        path = os.path.join(cfg.data_dir, f"minititan_{t}.jsonl")
        exs, skipped = [], 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                ex = to_example(json.loads(line))
                if ex:
                    exs.append(ex)
                else:
                    skipped += 1
        if skipped:
            print(f"[data] {t}: skipped {skipped} malformed record(s) of {len(exs) + skipped}")
        random.Random(cfg.split_seed).shuffle(exs)
        val = exs[:cfg.val_per_type]
        out[t] = {"val": val, "gen": val[:cfg.gen_eval_per_type], "train": exs[cfg.val_per_type:]}
    return out


# ── tokenisation ──────────────────────────────────────────────────────────────

def tokenize_example(tok, ex: dict, cfg: LLMConfig, progress=None) -> dict:
    """progress ∈ [0,1] (training only) enables the crop-around-the-fact curriculum; None = full document (eval)."""
    S = cfg.seg_len
    ids = tok(ex["context"], add_special_tokens=False)["input_ids"]
    fact_tok, dist = None, None
    fc = ex.get("fact_char")
    if fc is not None:
        fact_tok = len(tok(ex["context"][:fc], add_special_tokens=False)["input_ids"]) if fc > 0 else 0
        dist = len(ids) - fact_tok
    if progress is not None and cfg.crop and fact_tok is not None and (cfg.crop_annotated or not ex["meta"].get("annot")):
        dmax = len(ids) - fact_tok
        dmin = min(cfg.crop_d_min, dmax)
        dcur = dmin + (dmax - dmin) * min(1.0, progress / max(cfg.crop_curriculum, 1e-6))
        d = random.randint(dmin, max(dmin, int(dcur)))
        ids = ids[max(0, fact_tok - cfg.crop_pre_tokens): fact_tok + d]
    ctx = ids[:cfg.max_ctx_segments * S]
    prompt = tok(PROMPT.format(q=ex["question"]), add_special_tokens=False)["input_ids"]
    eos = tok.eos_token_id

    def ans_ids(text):
        return tok(" " + text, add_special_tokens=False)["input_ids"] + [eos]

    if len(prompt) > S - 8:                                   # keep the end of an over-long question
        prompt = prompt[-(S - 8):]
    a = ans_ids(ex["answer"])[:S - len(prompt)]
    w = ans_ids(ex["wrong"])[:S - len(prompt)] if ex.get("wrong") else None
    ann = None
    cropped = progress is not None and cfg.crop and fact_tok is not None and (cfg.crop_annotated or not ex["meta"].get("annot"))
    if (cfg.pool == "oracle" or cfg.key_loss_weight > 0) and ex["meta"].get("annot") and not cropped:
        ann = annotation_tokens(tok, ex, ctx, prompt)
    return {"ctx": ctx, "prompt": prompt, "ans": a, "wrong": w, "meta": ex["meta"], "text_answer": ex["answer"],
            "fact_tok": fact_tok, "dist_tokens": dist, "ann": ann}


class QADataset:
    def __init__(self, examples: List[dict], tok, cfg: LLMConfig, progress=None):
        self.examples, self.tok, self.cfg = examples, tok, cfg
        self.progress = progress                     # shared multiprocessing.Value, updated by the training loop

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        p = self.progress.value if self.progress is not None else None
        return tokenize_example(self.tok, self.examples[i], self.cfg, progress=p)


# ── collation ─────────────────────────────────────────────────────────────────

def collate(items: List[dict], cfg: LLMConfig, pad_id: int, use_wrong: bool = False, with_answer: bool = True):
    """Segmented batch. with_answer=False → question segment holds only the prompt (for generation)."""
    import torch
    S, B = cfg.seg_len, len(items)
    n = max(1, max(math.ceil(len(it["ctx"]) / S) for it in items))
    ids = torch.full((B, n + 1, S), pad_id, dtype=torch.long)
    write = torch.zeros(B, (n + 1) * S)
    ctx_flat = torch.full((B, n * S), pad_id, dtype=torch.long)
    A = max(len(it["wrong"] if use_wrong and it["wrong"] else it["ans"]) for it in items)
    ans_pos = torch.zeros(B, A, dtype=torch.long)
    ans_tok = torch.zeros(B, A, dtype=torch.long)
    ans_valid = torch.zeros(B, A)
    prompt_len = torch.zeros(B, dtype=torch.long)
    for i, it in enumerate(items):
        c = it["ctx"]
        if c:
            ctx_flat[i, n * S - len(c):] = torch.tensor(c)
            write[i, n * S - len(c):n * S] = 1
        p = it["prompt"]
        ans = it["wrong"] if use_wrong and it["wrong"] else it["ans"]
        seq = p + ans if with_answer else p
        ids[i, n, :len(seq)] = torch.tensor(seq)
        write[i, n * S:n * S + len(seq)] = 1
        prompt_len[i] = len(p)
        L = len(ans)
        ans_pos[i, :L] = torch.arange(len(p) - 1, len(p) - 1 + L)
        ans_tok[i, :L] = torch.tensor(ans)
        ans_valid[i, :L] = 1
    ids[:, :n] = ctx_flat.view(B, n, S)
    out = {"ids": ids, "write": write, "ans_pos": ans_pos, "ans_tok": ans_tok, "ans_valid": ans_valid,
           "prompt_len": prompt_len, "n_ctx_seg": n, "meta": [it["meta"] for it in items],
           "text_answer": [it["text_answer"] for it in items]}
    if any(it.get("ann") for it in items):                 # oracle pool mask + key-loss positions (programmatic docs only)
        pm = torch.zeros(B, (n + 1) * S)
        K = max((len(it["ann"]["val_first"]) for it in items if it.get("ann")), default=1)
        kl_q = torch.zeros(B, dtype=torch.long)
        kl_c = torch.zeros(B, K, dtype=torch.long)
        kl_m = torch.zeros(B, K, dtype=torch.bool)
        kl_h = torch.zeros(B, dtype=torch.bool)
        for i, it in enumerate(items):
            an = it.get("ann")
            if not an:
                continue
            base = n * S - len(it["ctx"])
            for t in an["ent_ctx"]:
                pm[i, base + t] = 1
            for t in an["ent_q"]:
                pm[i, n * S + t] = 1
            vf = an["val_first"]
            if len(vf) >= 2 and all(v_ is not None and v_ >= 1 for v_ in vf):
                kl_q[i] = n * S + len(it["prompt"]) - 1
                kl_c[i, :len(vf)] = torch.tensor([base + v_ - 1 for v_ in vf])
                kl_m[i, :len(vf)] = True
                kl_h[i] = True
        out.update({"pool_mask": pm, "kl_q": kl_q, "kl_cands": kl_c, "kl_cmask": kl_m, "kl_has": kl_h})
    return out


def collate_oracle(items: List[dict], cfg: LLMConfig, pad_id: int, use_wrong: bool = False):
    """Full-context reference batch (NOT segmented): context + prompt + answer in one sequence, right-padded."""
    import torch
    B = len(items)
    seqs, spans = [], []
    for it in items:
        ans = it["wrong"] if use_wrong and it["wrong"] else it["ans"]
        seq = it["ctx"] + it["prompt"] + ans
        spans.append((len(it["ctx"]) + len(it["prompt"]), len(ans)))
        seqs.append(seq)
    T = max(len(s) for s in seqs)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.long)
    A = max(a for _, a in spans)
    ans_pos = torch.zeros(B, A, dtype=torch.long)
    ans_tok = torch.zeros(B, A, dtype=torch.long)
    ans_valid = torch.zeros(B, A)
    for i, (s, (start, L)) in enumerate(zip(seqs, spans)):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
        ans_pos[i, :L] = torch.arange(start - 1, start - 1 + L)
        ans_tok[i, :L] = ids[i, start:start + L]
        ans_valid[i, :L] = 1
    return {"ids": ids, "mask": mask, "ans_pos": ans_pos, "ans_tok": ans_tok, "ans_valid": ans_valid}


def collate_oracle_prompt(items: List[dict], cfg: LLMConfig, pad_id: int):
    """Left-padded context+prompt for HF generate()."""
    import torch
    seqs = [it["ctx"] + it["prompt"] for it in items]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(items), T), pad_id, dtype=torch.long)
    mask = torch.zeros(len(items), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):] = torch.tensor(s)
        mask[i, T - len(s):] = 1
    return ids, mask


# ── pure-python self-check (no tokenizer / torch needed for the formatting part) ─────

def _check_formatting(data_dir: str):
    cfg = LLMConfig(data_dir=data_dir)
    sp = load_split(cfg)
    for t, d in sp.items():
        assert len(d["val"]) == cfg.val_per_type and len(d["train"]) > 1000, t
        ex = d["train"][0]
        print(f"{t}: train={len(d['train'])} val={len(d['val'])}  Q: {ex['question'][:70]!r}  A: {ex['answer'][:50]!r}")
        assert set(map(id, d["val"])).isdisjoint(map(id, d["train"]))
    ag = [e for e in sp["agentic"]["train"][:50]]
    assert all(e["wrong"] for e in ag), "agentic examples need a wrong decision for preference scoring"
    print("formatting OK")


if __name__ == "__main__":
    import sys
    _check_formatting(sys.argv[1] if len(sys.argv) > 1 else "../data_generation/miniTitan_data")
