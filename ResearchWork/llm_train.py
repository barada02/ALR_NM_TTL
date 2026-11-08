"""
Train + evaluate the memory module on top of a frozen LLM (Phase 2). Entry point: run_llm(cfg) -> dict
(called by modal_llm.py on a GPU container). Only the memory module trains.

Fixes vs the legacy trainer: P1 segmented bottleneck (llm_model) · P2 frozen lm_head, no fresh vocab head ·
P3 optimizer steps every iteration · P4 autoregressive generation eval · P5 honest baselines
("none" = frozen base alone, "off" = memory present but starved, "oracle" = full-context upper bound) ·
P6 shuffled seeded split · P7 exact answer boundary.
"""

import math
import os
import re
import string
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List

import torch

from llm_config import LLMConfig
from llm_data import QADataset, collate, collate_oracle, collate_oracle_prompt, load_split, tokenize_example
from llm_model import SegmentedMemoryLM

MODES = ("none", "off", "swap", "on", "oracle")


def modes_for(cfg: LLMConfig):
    """The adapter-oracle control run additionally reports the adapter-matched oracle NLL."""
    return MODES + (("oracle_adapter",) if cfg.train_mode == "adapter_oracle" else ())


# ── helpers ───────────────────────────────────────────────────────────────────

def to_dev(b, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in set(string.punctuation) - {"-"})
    return re.sub(r"\s+", " ", s).strip()


def token_f1(pred: str, gold: str) -> float:
    p, g = norm_text(pred).split(), norm_text(gold).split()
    if not p or not g:
        return float(p == g)
    common = defaultdict(int)
    for w in g:
        common[w] += 1
    overlap = 0
    for w in p:
        if common[w] > 0:
            overlap += 1
            common[w] -= 1
    if overlap == 0:
        return 0.0
    pr, rc = overlap / len(p), overlap / len(g)
    return 2 * pr * rc / (pr + rc)


def sample_nll(logp: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per-sample mean negative log-likelihood over answer tokens."""
    return -(logp.sum(1) / valid.sum(1).clamp(min=1))


def git_sha(root: str) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_nll(model, items: List[dict], cfg: LLMConfig, pad_id: int, device, modes, bs: int = 4) -> Dict[str, List[float]]:
    out = {m: [] for m in modes}
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        seg = to_dev(collate(chunk, cfg, pad_id), device) if any(not m.startswith("oracle") for m in modes) else None
        for m in modes:
            if m.startswith("oracle"):
                ob = to_dev(collate_oracle(chunk, cfg, pad_id), device)
                lp, valid = model.oracle_logp(ob, adapter=(m == "oracle_adapter")), ob["ans_valid"]
            else:
                lp, valid = model.answer_logp(seg, m), seg["ans_valid"]
            out[m] += sample_nll(lp, valid).tolist()
    return out


@torch.no_grad()
def eval_pref(model, items: List[dict], cfg: LLMConfig, pad_id: int, device, modes, bs: int = 4) -> Dict[str, float]:
    """Agentic: does the model rate the correct decision above the 'forgot the step' decision? (mean token logp)"""
    items = [it for it in items if it["wrong"]]
    wins = {m: [] for m in modes}
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        for m in modes:
            if m.startswith("oracle"):
                c, w = collate_oracle(chunk, cfg, pad_id), collate_oracle(chunk, cfg, pad_id, use_wrong=True)
                c, w = to_dev(c, device), to_dev(w, device)
                ad = (m == "oracle_adapter")
                sc = -sample_nll(model.oracle_logp(c, adapter=ad), c["ans_valid"])
                sw = -sample_nll(model.oracle_logp(w, adapter=ad), w["ans_valid"])
            else:
                c, w = to_dev(collate(chunk, cfg, pad_id), device), to_dev(collate(chunk, cfg, pad_id, use_wrong=True), device)
                sc = -sample_nll(model.answer_logp(c, m), c["ans_valid"])
                sw = -sample_nll(model.answer_logp(w, m), w["ans_valid"])
            wins[m] += (sc > sw).float().tolist()
    return {m: (sum(v) / len(v) if v else float("nan")) for m, v in wins.items()}


@torch.no_grad()
def eval_generation(model, tok, items: List[dict], cfg: LLMConfig, pad_id: int, device, modes, bs: int = 8):
    """Greedy autoregressive answers. Reports exact match / contains (normalised, first line) and token F1."""
    stats = {m: {"em": [], "contains": [], "f1": [], "digit_recall": []} for m in modes}
    examples = {m: [] for m in modes}
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        for m in modes:
            if m == "oracle":
                ids, mask = collate_oracle_prompt(chunk, cfg, pad_id)
                toks = model.oracle_generate(ids.to(device), mask.to(device), cfg.gen_max_new, pad_id)
            else:
                b = to_dev(collate(chunk, cfg, pad_id, with_answer=False), device)
                toks = model.generate(b, m, cfg.gen_max_new)
            for it, t in zip(chunk, toks):
                text = tok.decode(t, skip_special_tokens=True)
                first = text.strip().split("\n")[0]
                gold = it["text_answer"]
                stats[m]["em"].append(float(norm_text(first) == norm_text(gold)))
                stats[m]["contains"].append(float(norm_text(gold) in norm_text(text)))
                stats[m]["f1"].append(token_f1(first, gold))
                gd = set(re.findall(r"\d+", gold))                    # key details: numbers / IDs in the gold answer
                if gd:
                    stats[m]["digit_recall"].append(len(gd & set(re.findall(r"\d+", first))) / len(gd))
                if len(examples[m]) < 3:
                    examples[m].append({"gold": gold[:80], "pred": first[:80]})
    mean = lambda v: (sum(v) / len(v)) if v else float("nan")
    return {m: {k: mean(v) for k, v in s.items()} for m, s in stats.items()}, examples


def full_eval(model, tok, splits, cfg: LLMConfig, pad_id: int, device) -> dict:
    res = {}
    for t, d in splits.items():
        val = [tokenize_example(tok, ex, cfg) for ex in d["val"]]
        gen = [tokenize_example(tok, ex, cfg) for ex in d["gen"]]
        MODES_ = modes_for(cfg)
        nll = eval_nll(model, val, cfg, pad_id, device, MODES_)
        r = {"n_val": len(val), "nll": {m: sum(v) / len(v) for m, v in nll.items()},
             "ppl": {m: math.exp(sum(v) / len(v)) for m, v in nll.items()}}
        r["nll_gain_on_vs_none"] = r["nll"]["none"] - r["nll"]["on"]       # >0 ⇒ memory helps
        r["nll_gain_on_vs_off"] = r["nll"]["off"] - r["nll"]["on"]         # inflated: a starved memory injects junk
        r["nll_gain_on_vs_swap"] = r["nll"]["swap"] - r["nll"]["on"]       # >0 ⇒ document-SPECIFIC recall (the honest control)
        r["frac_gain_recovered"] = (r["nll"]["none"] - r["nll"]["on"]) / max(r["nll"]["none"] - r["nll"]["oracle"], 1e-6)
        tot = r["nll"]["none"] - r["nll"]["on"]
        r["doc_specific_share"] = (r["nll"]["swap"] - r["nll"]["on"]) / tot if tot > 1e-6 else float("nan")
        nseg = [max(1, math.ceil(len(it["ctx"]) / cfg.seg_len)) for it in val]
        by_seg = defaultdict(lambda: defaultdict(list))
        for i, n in enumerate(nseg):
            for m in MODES_:
                by_seg[n][m].append(nll[m][i])
        r["nll_by_n_segments"] = {str(n): {m: sum(v) / len(v) for m, v in ms.items()} | {"count": len(ms["on"])}
                                  for n, ms in sorted(by_seg.items())}
        by_dist = defaultdict(lambda: defaultdict(list))                   # fact→question distance in tokens
        for i, it in enumerate(val):
            dd = it.get("dist_tokens")
            if dd is None:
                continue
            lab = "<256" if dd < 256 else "256-512" if dd < 512 else "512-1024" if dd < 1024 else "1024+"
            for m in MODES_:
                by_dist[lab][m].append(nll[m][i])
        r["nll_by_fact_distance"] = {lab: {m: sum(v) / len(v) for m, v in ms.items()} | {"count": len(ms["on"])}
                                     for lab, ms in by_dist.items()}
        if t == "long_context":
            by_pos = defaultdict(lambda: defaultdict(list))
            for i, it in enumerate(val):
                for m in MODES_:
                    by_pos[it["meta"].get("fact_position")][m].append(nll[m][i])
            r["nll_by_fact_position"] = {str(p): {m: sum(v) / len(v) for m, v in ms.items()} | {"count": len(ms["on"])}
                                         for p, ms in by_pos.items()}
        if t == "agentic":
            r["pref_acc"] = eval_pref(model, val, cfg, pad_id, device, MODES_)
        g, ex = eval_generation(model, tok, gen, cfg, pad_id, device, ("none", "on", "oracle"))
        r["generation"], r["generation_examples"] = g, ex
        res[t] = r
    return res


# ── training ──────────────────────────────────────────────────────────────────

def run_llm(cfg: LLMConfig, sha: str = "", verbose: bool = True) -> dict:
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    import transformers

    device = "cuda"
    torch.manual_seed(cfg.seed)
    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    splits = load_split(cfg)
    train_ex = [ex for t in splits for ex in splits[t]["train"]]
    quick_val = [tokenize_example(tok, ex, cfg) for t in splits for ex in splits[t]["val"][:cfg.eval_batches * 4 // 3 + 1]]
    if verbose:
        print({t: {k: len(v) for k, v in d.items()} for t, d in splits.items()})

    model = SegmentedMemoryLM(cfg).to(device)
    model.lm.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.memory.parameters())
    opt = torch.optim.AdamW(model.memory_parameters(), lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(cfg.warmup, 1)) *
        (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, cfg.steps) / max(cfg.steps, 1)))))
    import torch.multiprocessing as mp
    progress = mp.Value("d", 0.0)
    ds = QADataset(train_ex, tok, cfg, progress=progress)
    g = torch.Generator().manual_seed(cfg.seed)
    train_collate = (lambda items: collate_oracle(items, cfg, pad_id)) if cfg.train_mode == "adapter_oracle" \
        else (lambda items: collate(items, cfg, pad_id))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=2,
                        collate_fn=train_collate, generator=g)
    if verbose:
        print(f"[llm] memory params {n_params/1e6:.2f}M  insert={cfg.insert}  mode={cfg.mem_mode}  "
              f"seg={cfg.seg_len} chunk={cfg.chunk_size} tbptt={cfg.tbptt_chunks}")

    curve, quick, skipped = [], [], 0
    run_loss, run_met, step, t0 = [], {}, 0, time.time()
    it = iter(loader)
    while step < cfg.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = to_dev(batch, device)
        progress.value = step / max(cfg.steps, 1)
        if cfg.train_mode == "adapter_oracle":
            lp = model.oracle_logp(batch, adapter=True)
        elif cfg.train_mode == "adapter_seg":
            lp = model.answer_logp(batch, "off")
        else:
            lp = model.answer_logp(batch, "on")
        nll_on = sample_nll(lp, batch["ans_valid"])
        loss = nll_on.mean()
        contrast_gap = None
        if cfg.contrast_weight > 0 and cfg.train_mode == "memory" and batch["ids"].shape[0] > 1:
            nll_sw = sample_nll(model.answer_logp(batch, "swap"), batch["ans_valid"])
            gap = nll_sw - nll_on
            loss = loss + cfg.contrast_weight * torch.relu(cfg.contrast_margin - gap).mean()
            contrast_gap = float(gap.detach().mean())
        if cfg.mem_mode == "A" and model.last_aux is not None:
            loss = loss + cfg.aux_weight * model.last_aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.memory_parameters(), cfg.grad_clip)
        step += 1
        if not torch.isfinite(gn):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        sched.step()
        run_loss.append(loss.item())
        run_met = {k: model.last_met[k] for k in ("mean_write_gate", "retention_halflife_tokens", "state_norm_ratio",
                                                  "mean_blend", "mean_surprise") if k in model.last_met}
        if step % cfg.log_every == 0:
            row = {"step": step, "nll": sum(run_loss) / len(run_loss), "grad_norm": float(gn),
                   "nll_on_batch": float(nll_on.mean()), "swap_gap_batch": contrast_gap,
                   "time_s": time.time() - t0, "mem_gb": torch.cuda.max_memory_allocated() / 1e9, **run_met}
            curve.append(row)
            run_loss = []
            if verbose:
                print(f"step {step:5d}  nll {row['nll']:.3f}  gn {row['grad_norm']:.2f}  "
                      f"{row['time_s']/step:.2f}s/step  gpu {row['mem_gb']:.1f}GB  "
                      + "  ".join(f"{k.replace('mean_', '')}={v:.3f}" for k, v in run_met.items()))
        if step % cfg.eval_every == 0 and step < cfg.steps:
            model.eval()
            q = eval_nll(model, quick_val, cfg, pad_id, device, ("none", "on"))
            quick.append({"step": step, **{m: sum(v) / len(v) for m, v in q.items()}})
            if verbose:
                print(f"  quick-eval@{step}: none {quick[-1]['none']:.3f}  on {quick[-1]['on']:.3f}")
            model.train()

    train_time = time.time() - t0
    model.eval()
    final = full_eval(model, tok, splits, cfg, pad_id, device)
    if cfg.save_ckpt:
        os.makedirs(cfg.out_dir, exist_ok=True)
        torch.save(model.memory.state_dict(), os.path.join(cfg.out_dir, f"{cfg.run_name or 'run'}_memory.pt"))
    state_floats = sum(o * i for o, i in model.memory.memory.shapes())          # per-sequence fast state (the fair-comparison axis)
    return {"config": asdict(cfg), "git_sha": sha, "memory_params": n_params, "state_floats": state_floats,
            "curve": curve, "quick_eval": quick,
            "final": final, "skipped_nonfinite_steps": skipped, "train_time_s": train_time,
            "wall_time_s": time.time() - t_start, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__}}
