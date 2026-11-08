"""
Eval-only runner: load a trained memory checkpoint and score it on annotated evaluation sets (no training).

  python ResearchWork/launch.py eval <run_name> [--files all|eval_overwrite,eval_haystack] [--gen 100] [--tag zeroshot]

For every set and every annotation group (n_colliding, n_versions, gap_kind, n_docs, fact_position, ...):
  * answer NLL under none / swap / on / oracle (frozen base alone · wrong document · trained memory · full attention)
  * greedy generation classified as   correct | stale (the OLD value) | confused (a DISTRACTOR's value) | other
    and, for absent-fact questions, hallucinated (a value-like string) vs not.
The classification is what lets a failure say which mechanism broke.
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, replace
from typing import Dict, List

import torch

from llm_config import LLMConfig
from llm_data import collate, collate_oracle, collate_oracle_prompt, tokenize_example, to_example
from llm_model import SegmentedMemoryLM
from llm_train import eval_nll, norm_text, to_dev

GROUP_KEYS = ("n_colliding", "n_versions", "gap_kind", "n_docs", "fact_position", "attr")


def load_examples(path: str, limit: int = 0):
    exs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ex = to_example(rec)
            if ex is None:
                continue
            ex["annot"] = rec.get("annot", {})
            ex["meta"] = {**ex["meta"], "annot": ex["annot"]}
            exs.append(ex)
            if limit and len(exs) >= limit:
                break
    return exs


def classify(text: str, annot: dict, gold: str) -> str:
    first = norm_text(text.strip().split("\n")[0])
    if annot.get("absent"):
        values = [norm_text(v) for v in annot.get("all_values", [])]
        return "hallucinated" if (any(v and v in first for v in values) or re.search(r"\d", first)) else "no_value"
    if norm_text(gold) in first:
        return "correct"
    if any(norm_text(v) in first for v in annot.get("old_values", [])):
        return "stale"
    if any(norm_text(v) in first for v in annot.get("distractor_values", [])):
        return "confused"
    return "other"


@torch.no_grad()
def generate_texts(model, tok, items: List[dict], cfg: LLMConfig, pad_id: int, device, mode: str, max_new: int, bs: int):
    out = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        if mode == "oracle":
            ids, mask = collate_oracle_prompt(chunk, cfg, pad_id)
            toks = model.oracle_generate(ids.to(device), mask.to(device), max_new, pad_id)
        else:
            b = to_dev(collate(chunk, cfg, pad_id, with_answer=False), device)
            toks = model.generate(b, mode, max_new)
        out += [tok.decode(t, skip_special_tokens=True) for t in toks]
    return out


def rank_candidates(model, tok, items: List[dict], cfg: LLMConfig, pad_id: int, device, modes, bs: int, max_cand: int = 4):
    """Likelihood ranking — works even when exact generation is at the floor. For every item score the CURRENT value against
    the OLD values (overwrite) and the DISTRACTOR values (collisions) by teacher-forced NLL; the model 'prefers' the candidate
    with the lowest NLL. Returns per-mode counts of  current / stale / confused  and the chance level (mean 1/#candidates).
    Up to `max_cand` candidates per item (gold + the first others) to bound cost on long haystacks."""
    eos = tok.eos_token_id
    flat, owner, kind = [], [], []
    for i, it in enumerate(items):
        a = it["meta"]["annot"]
        cands = [(it["text_answer"], "current")] + [(v, "stale") for v in a.get("old_values", [])] + \
                [(v, "confused") for v in a.get("distractor_values", [])]
        cands = cands[:max_cand]
        if len(cands) < 2:
            continue
        for text, k in cands:
            c = dict(it)
            c["ans"] = tok(" " + text, add_special_tokens=False)["input_ids"] + [eos]
            c["wrong"] = None
            flat.append(c)
            owner.append(i)
            kind.append(k)
    nll = eval_nll(model, flat, cfg, pad_id, device, modes, bs=bs)
    out = {m: [] for m in modes}
    chance = []
    by_item = defaultdict(list)
    for j, i in enumerate(owner):
        by_item[i].append(j)
    res = {m: {} for m in modes}
    for i, js in by_item.items():
        chance.append(1.0 / len(js))
        for m in modes:
            best = min(js, key=lambda j: nll[m][j])
            res[m][i] = kind[best]
    summary = {m: {k: sum(1 for v in res[m].values() if v == k) / max(len(res[m]), 1) for k in ("current", "stale", "confused")}
               for m in modes}
    return summary, res, sum(chance) / max(len(chance), 1), len(by_item)


def group_of(item: dict) -> Dict[str, str]:
    a = item["meta"].get("annot", {})
    g = {}
    for k in GROUP_KEYS:
        v = a.get(k, item["meta"].get(k))
        if v is not None:
            g[k] = f"{k}={v}"
    return g


def run_eval(cfg: LLMConfig, ckpt_path: str, files: List[str], gen_n: int = 100, max_new: int = 24, limit: int = 0,
             sha: str = "") -> dict:
    from transformers import AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = SegmentedMemoryLM(cfg).to(device)
    model.lm.to(device)
    model.memory.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    results = {}
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        exs = load_examples(path, limit)
        items = [tokenize_example(tok, ex, cfg) for ex in exs]
        is_absent = bool(items[0]["meta"]["annot"].get("absent"))
        is_hay = name.endswith("haystack")
        bs = 1 if is_hay else 4
        r = {"n": len(items), "avg_ctx_tokens": sum(len(it["ctx"]) for it in items) / len(items)}

        if not is_absent:                                       # NLL of an "unknown" answer is meaningless
            modes = ("none", "on", "oracle") if is_hay else ("none", "swap", "on", "oracle")
            nll = eval_nll(model, items, cfg, pad_id, device, modes, bs=bs)
            r["nll"] = {m: sum(v) / len(v) for m, v in nll.items()}
            groups = defaultdict(lambda: defaultdict(list))
            for i, it in enumerate(items):
                for gk, gv in group_of(it).items():
                    for m in modes:
                        groups[gv][m].append(nll[m][i])
            r["nll_by_group"] = {g: {m: sum(v) / len(v) for m, v in ms.items()} | {"count": len(ms["on"])} for g, ms in sorted(groups.items())}

        if not is_absent:
            rmodes = ("none", "on", "oracle")
            summ, per_item, chance, n_rank = rank_candidates(model, tok, items, cfg, pad_id, device, rmodes, bs=bs)
            if n_rank:
                r["rank"] = {"n": n_rank, "chance": chance, "share": summ}
                rg = defaultdict(lambda: defaultdict(list))
                for i, it in enumerate(items):
                    if i in per_item["on"]:
                        for gk, gv in group_of(it).items():
                            for m in rmodes:
                                rg[gv][m].append(per_item[m][i])
                r["rank_by_group"] = {g: {m: {k: sum(1 for x in v if x == k) / len(v) for k in ("current", "stale", "confused")} for m, v in ms.items()}
                                      | {"count": len(ms["on"])} for g, ms in sorted(rg.items())}
        sel = items[:gen_n]
        if not sel:
            results[name] = r
            continue
        gen = {}
        for mode in ("none", "on", "oracle"):
            texts = generate_texts(model, tok, sel, cfg, pad_id, device, mode, max_new, bs=bs if is_hay else 8)
            gen[mode] = [classify(t, it["meta"]["annot"], it["text_answer"]) for t, it in zip(texts, sel)]
            if mode == "on":
                r["examples_on"] = [{"gold": it["text_answer"][:40], "pred": t.strip().split("\n")[0][:60], "class": c}
                                    for t, it, c in list(zip(texts, sel, gen[mode]))[:5]]
        labels = sorted({c for v in gen.values() for c in v})
        r["generation"] = {m: {c: sum(1 for x in v if x == c) / len(v) for c in labels} for m, v in gen.items()}
        gg = defaultdict(lambda: defaultdict(list))
        for i, it in enumerate(sel):
            for gk, gv in group_of(it).items():
                for m in gen:
                    gg[gv][m].append(gen[m][i])
        r["generation_by_group"] = {g: {m: {c: sum(1 for x in v if x == c) / len(v) for c in labels} for m, v in ms.items()} | {"count": len(ms["on"])}
                                    for g, ms in sorted(gg.items())}
        results[name] = r
    return {"config": asdict(cfg), "git_sha": sha, "ckpt": ckpt_path, "results": results}


def summarize_eval(res: dict) -> str:
    lines = [f"== eval of {os.path.basename(res['ckpt'])}"]
    for name, r in res["results"].items():
        g = r.get("generation")
        row = " | ".join(f"{m}: " + " ".join(f"{c} {v:.2f}" for c, v in g[m].items()) for m in ("none", "on", "oracle")) if g else ""
        nll = ""
        if "nll" in r:
            nll = "  NLL " + " ".join(f"{m} {v:.2f}" for m, v in r["nll"].items())
        gen_line = f"\n  {'':24s} gen → {row}" if "generation" in r else ""
        rk = ""
        if "rank" in r:
            rk = f"\n  {'':24s} rank (chance {r['rank']['chance']:.2f}) → " + " | ".join(
                f"{m}: " + " ".join(f"{k} {v:.2f}" for k, v in r['rank']['share'][m].items()) for m in ("none", "on", "oracle"))
        lines.append(f"  {name:24s} n={r['n']:<4d} ctx≈{r['avg_ctx_tokens']:.0f}tok{nll}{gen_line}{rk}")
    return "\n".join(lines)
