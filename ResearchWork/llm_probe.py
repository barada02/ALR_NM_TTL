"""
Localisation probes for the binding failure (E12). Eval-only. For colliding-fact documents (k same-type facts, one asked):

  P1 storage     — query the memory with the DOCUMENT's own write key of each fact (a perfect address) and rank the stored values:
                   is the right value retrievable at all from the final state ΔW?
  P2 addressing  — at the question position (the token that predicts the first answer token), does the key match the asked
                   fact's write key better than the colliding facts' write keys?  Measured in key space (after W_k) and in raw
                   normalised hidden space (before W_k): entity missing from the frozen representation vs discarded by W_k.
  P3 read        — does the ACTUAL memory read at the question position rank the asked fact's written value first?
  (P4 readout    — likelihood ranking after the frozen upper layers; measured by llm_eval, ~chance in E11/E12.)

Positions (shift_v pairs k_t → v_{t+1}): the first value token of fact i sits at flat position P_i; its write key and the value
written for it are at position p_i = P_i − 1. The question read is at q = n·S + prompt_len − 1.
Every metric is a top-1 among the k facts of the document (chance 1/k), reported with a Wilson 95% interval and by k.
"""

import math
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F

from llm_config import LLMConfig
from llm_data import collate, tokenize_example
from llm_eval import load_examples
from llm_model import SegmentedMemoryLM
from llm_train import to_dev


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


@torch.no_grad()
def capture(model: SegmentedMemoryLM, b) -> Dict[str, torch.Tensor]:
    model.memory.capture = {}
    try:
        model.question_hidden(b["ids"], b["write"], "on", extra=b)
        cap = model.memory.capture
    finally:
        model.memory.capture = None
    cap["delta"] = [d.detach() for d in model.memory._delta]          # final fast-weight state of each sequence
    return cap


def value_token_index(tok, context: str, ctx_ids: List[int], value: str):
    enc = tok(context, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"][:len(ctx_ids)] == ctx_ids, "tokenisation mismatch"
    c = context.find(value)
    if c < 0:
        return None
    for i, (a, e) in enumerate(enc["offset_mapping"]):
        if e > c:
            return i if i < len(ctx_ids) else None
    return None


@torch.no_grad()
def probe_file(model, tok, cfg: LLMConfig, path: str, pad_id: int, device, bs: int = 4) -> dict:
    exs = load_examples(path)
    items = [tokenize_example(tok, ex, cfg) for ex in exs]
    S = cfg.seg_len
    rows = []
    for i0 in range(0, len(items), bs):
        chunk_items, chunk_exs = items[i0:i0 + bs], exs[i0:i0 + bs]
        b = to_dev(collate(chunk_items, cfg, pad_id), device)
        n = b["ids"].shape[1] - 1
        all_pos = []
        for j, (it, ex) in enumerate(zip(chunk_items, chunk_exs)):
            pos = []
            for v in [ex["answer"]] + list(ex["annot"].get("distractor_values", [])):
                ti = value_token_index(tok, ex["context"], it["ctx"], v)
                pos.append(None if ti is None else n * S - len(it["ctx"]) + ti - 1)
            all_pos.append(pos)
        if cfg.force_gate in ("pos", "sent"):                            # E16: force the write gate at every fact's write position
            fp = torch.zeros(b["ids"].shape[0], (n + 1) * S, device=device)
            for j, pos in enumerate(all_pos):
                for p in pos:
                    if p is not None and p >= 0:
                        fp[j, p] = 1
            model.memory.force_pos = fp
        try:
            cap = capture(model, b)
        finally:
            model.memory.force_pos = None
        for j, (it, ex) in enumerate(zip(chunk_items, chunk_exs)):
            a = ex["annot"]
            cands = [ex["answer"]] + list(a.get("distractor_values", []))
            pos = all_pos[j]
            if any(p is None or p < 0 for p in pos):
                continue
            q = n * S + int(b["prompt_len"][j]) - 1
            K = cap.get("kw", cap["k"])[j, pos]                          # (m, d) WRITE keys of the m facts (v5: k^write)
            V = cap["v"][j, pos]                                         # (m, d) values written for their first value token
            H = cap["h"][j, pos]                                         # (m, d) normalised hidden at the write positions
            kq, hq, rq = cap["k"][j, q], cap["h"][j, q], cap["read"][j, q]
            cosk = F.cosine_similarity(kq.unsqueeze(0), K, dim=-1)
            cosh = F.cosine_similarity(hq.unsqueeze(0), H, dim=-1)
            cosr = F.cosine_similarity(rq.unsqueeze(0), V, dim=-1)
            # P1 storage: read the memory with each fact's OWN write key against the final state; does fact 0's key return value 0?
            delta_j = [d[j:j + 1] for d in cap["delta"]]
            R_oracle = model.memory.memory(K.unsqueeze(0), delta_j)[0]   # (m, d)
            cos_store = F.cosine_similarity(R_oracle[0].unsqueeze(0), V, dim=-1)
            vv = F.cosine_similarity(V.unsqueeze(1), V.unsqueeze(0), dim=-1)
            off = vv[~torch.eye(len(cands), dtype=torch.bool, device=vv.device)]
            rows.append({
                "k": len(cands), "attr": a.get("attr"),
                "storage_top1": int(cos_store.argmax() == 0), "storage_margin": float(cos_store[0] - cos_store[1:].max()),
                "key_top1": int(cosk.argmax() == 0), "key_margin": float(cosk[0] - cosk[1:].max()),
                "hidden_top1": int(cosh.argmax() == 0), "hidden_margin": float(cosh[0] - cosh[1:].max()),
                "read_top1": int(cosr.argmax() == 0), "read_margin": float(cosr[0] - cosr[1:].max()),
                "read_cos_target": float(cosr[0]),
                "value_max_offdiag_cos": float(off.max()),               # ~1 → candidate values indistinguishable in v-space
                "write_gate_at_fact": float(cap["gate"][j, pos].mean()),
                "write_gate_doc_mean": float(cap["gate"][j, n * S - len(it["ctx"]):n * S].mean()),
            })

    def summarise(rs):
        out = {"n": len(rs), "chance": sum(1 / r["k"] for r in rs) / max(len(rs), 1)}
        for m in ("storage", "key", "hidden", "read"):
            hits = sum(r[f"{m}_top1"] for r in rs)
            out[m] = {"top1": hits / max(len(rs), 1), "ci95": wilson(hits, len(rs)),
                      "margin": sum(r[f"{m}_margin"] for r in rs) / max(len(rs), 1)}
        out["value_max_offdiag_cos"] = sum(r["value_max_offdiag_cos"] for r in rs) / max(len(rs), 1)
        out["write_gate_at_fact"] = sum(r["write_gate_at_fact"] for r in rs) / max(len(rs), 1)
        out["write_gate_doc_mean"] = sum(r["write_gate_doc_mean"] for r in rs) / max(len(rs), 1)
        return out

    res = {"overall": summarise(rows), "skipped": len(items) - len(rows)}
    by = defaultdict(list)
    for r in rows:
        by[f"n_colliding={r['k'] - 1}"].append(r)
        by[f"attr={r['attr']}"].append(r)
    res["by_group"] = {g: summarise(rs) for g, rs in sorted(by.items())}
    return res


def run_probe(cfg: LLMConfig, ckpt_path: str, files: List[str], sha: str = "") -> dict:
    from dataclasses import asdict
    from transformers import AutoTokenizer
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = SegmentedMemoryLM(cfg).to(device)
    model.lm.to(device)
    model.memory.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    out = {}
    for path in files:
        name = path.rsplit("/", 1)[-1].replace(".jsonl", "")
        out[name] = probe_file(model, tok, cfg, path, pad_id, device)
    return {"config": asdict(cfg), "git_sha": sha, "ckpt": ckpt_path, "results": out}


def summarize_probe(res: dict) -> str:
    lines = [f"== probe of {res['ckpt'].rsplit('/', 1)[-1]}"]
    for name, r in res["results"].items():
        o = r["overall"]
        lines.append(f"  {name:24s} n={o['n']} chance={o['chance']:.2f}  " + "  ".join(
            f"{m} {o[m]['top1']:.2f} [{o[m]['ci95'][0]:.2f},{o[m]['ci95'][1]:.2f}]" for m in ("storage", "key", "hidden", "read"))
            + f"  | value off-diag cos {o['value_max_offdiag_cos']:.2f}  gate@fact {o['write_gate_at_fact']:.3f} vs doc {o['write_gate_doc_mean']:.3f}")
    return "\n".join(lines)
