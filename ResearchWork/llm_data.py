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


# ── record → example (pure python; no tokenizer needed) ───────────────────────

def to_example(rec: dict) -> Optional[dict]:
    """None for malformed records (the real data has 1 long_context record with only `context`)."""
    try:
        return _to_example(rec)
    except (KeyError, TypeError, IndexError):
        return None


def _to_example(rec: dict) -> Optional[dict]:
    t = rec.get("_type")
    if t == "long_context":
        ctx, q, a = rec["context"], rec["question"], rec["answer"]
        meta = {"type": t, "fact_position": rec.get("fact_position"), "distractors": rec.get("distractor_count")}
        wrong = None
    elif t == "dialogue":
        ctx = "\n".join(f"{x['role'].capitalize()}: {x['content']}" for x in rec["turns"])
        q, a = rec["test_question"], rec["correct_answer"]
        anchors = rec.get("memory_anchors") or [{}]
        gap = (anchors[0].get("recalled_at_turn", 0) or 0) - (anchors[0].get("established_at_turn", 0) or 0)
        meta = {"type": t, "gap": gap}
        wrong = None
    elif t == "agentic":
        dp = rec["decision_point"]
        steps = [s for s in rec["steps"] if s["step"] < dp["at_step"]]        # only what the agent has seen so far
        ctx = f"Task: {rec['task']}\n" + "\n".join(
            f"Step {s['step']} [{s['action']}] input: {s['input']} output: {s['output']}" for s in steps)
        q = f"At step {dp['at_step']}, what is the correct decision?"
        a, wrong = dp["correct_decision"], dp.get("wrong_decision_if_forgotten")
        meta = {"type": t, "distance": dp["at_step"] - dp["requires_memory_of_step"]}
    else:
        return None
    if not (ctx and q and a):
        return None
    return {"context": ctx.strip(), "question": q.strip(), "answer": a.strip(), "wrong": wrong, "meta": meta}


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
        random.Random(cfg.seed).shuffle(exs)
        val = exs[:cfg.val_per_type]
        out[t] = {"val": val, "gen": val[:cfg.gen_eval_per_type], "train": exs[cfg.val_per_type:]}
    return out


# ── tokenisation ──────────────────────────────────────────────────────────────

def tokenize_example(tok, ex: dict, cfg: LLMConfig) -> dict:
    S = cfg.seg_len
    ctx = tok(ex["context"], add_special_tokens=False)["input_ids"][:cfg.max_ctx_segments * S]
    prompt = tok(PROMPT.format(q=ex["question"]), add_special_tokens=False)["input_ids"]
    eos = tok.eos_token_id

    def ans_ids(text):
        return tok(" " + text, add_special_tokens=False)["input_ids"] + [eos]

    if len(prompt) > S - 8:                                   # keep the end of an over-long question
        prompt = prompt[-(S - 8):]
    a = ans_ids(ex["answer"])[:S - len(prompt)]
    w = ans_ids(ex["wrong"])[:S - len(prompt)] if ex.get("wrong") else None
    return {"ctx": ctx, "prompt": prompt, "ans": a, "wrong": w, "meta": ex["meta"], "text_answer": ex["answer"]}


class QADataset:
    def __init__(self, examples: List[dict], tok, cfg: LLMConfig):
        self.examples, self.tok, self.cfg = examples, tok, cfg

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return tokenize_example(self.tok, self.examples[i], self.cfg)


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
    return {"ids": ids, "write": write, "ans_pos": ans_pos, "ans_tok": ans_tok, "ans_valid": ans_valid,
            "prompt_len": prompt_len, "n_ctx_seg": n, "meta": [it["meta"] for it in items],
            "text_answer": [it["text_answer"] for it in items]}


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
