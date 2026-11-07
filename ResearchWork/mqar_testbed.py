"""
MiniTitan synthetic testbed: hosts + training + evaluation. (Tasks live in synth_tasks.py, config in
exp_config.py, sweeps in sweeps.py, Modal wrapper in modal_app.py.)

Hosts (cfg.model) — no attention, no convolution, so the only cross-token path is the thing under test:
    none     embedding → head                       (chance; the null for the whole testbed)
    gru      embedding → GRU over all tokens → head (Branch C: plain recurrent state, "is TTL buying anything?";
                                                     width set by gru_hidden or matched to a Titan's core params)
    titan_a  embedding → MiniTitan v4 Mode A → head (learned update rule, TBPTT + aux loss)
    titan_b  embedding → MiniTitan v4 Mode B → head (gradient update + momentum, Titans-faithful)

Programmatic entry point (what Modal calls):  run_experiment(cfg: ExpConfig) -> dict
Output dict: config, git SHA, params, curves, eval grid  →  results_io.save_result
"""

import argparse
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, fields
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_config import ExpConfig  # noqa: E402
from memory_module_v4 import MemoryConfig, MiniTitanMemoryModule  # noqa: E402
from results_io import save_result  # noqa: E402,F401  (re-exported for convenience)
from synth_tasks import AGE_BUCKETS, FILL, Task, build_task  # noqa: E402


# ── Parameter matching ────────────────────────────────────────────────────────

def _memory_config(cfg: ExpConfig, model: str) -> MemoryConfig:
    return MemoryConfig(
        dim=cfg.dim, mem_hidden=cfg.mem_hidden, mem_layers=cfg.mem_layers, rank=cfg.rank,
        chunk_size=cfg.chunk_size, mode="A" if model == "titan_a" else "B",
        tbptt_chunks=cfg.tbptt_chunks or None, inner_lr=cfg.inner_lr, inner_clip=cfg.inner_clip,
        use_momentum=cfg.use_momentum, second_order=cfg.second_order, aux_lag_chunks=cfg.aux_lag_chunks,
        retention_bias=cfg.retention_bias, retention_learned=cfg.retention_learned)


def titan_core_params(cfg: ExpConfig, model: str) -> int:
    return sum(p.numel() for p in MiniTitanMemoryModule(_memory_config(cfg, model)).parameters())


def gru_core_params(d: int, h: int) -> int:
    n = 3 * (d * h + h * h + 2 * h)                       # nn.GRU(d, h): 3 gates, w_ih + w_hh + 2 biases
    return n + (h * d + d if h != d else 0)               # output projection back to d


def match_gru_hidden(d: int, target: int) -> int:
    return min(range(4, 2049), key=lambda h: abs(gru_core_params(d, h) - target))


# ── Hosts ─────────────────────────────────────────────────────────────────────

class Host(nn.Module):
    def __init__(self, cfg: ExpConfig, task: Task):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.E_type = nn.Embedding(3, d)
        self.E_key = nn.Embedding(task.n_keys + 1, d, padding_idx=task.n_keys)
        self.E_val = nn.Embedding(task.n_vals + 1, d, padding_idx=task.n_vals)
        self.E_fil = nn.Embedding(task.n_filler + 1, d, padding_idx=task.n_filler)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, task.n_vals)
        self.mem = self.gru = self.gru_out = None
        if cfg.model in ("titan_a", "titan_b"):
            self.mem = MiniTitanMemoryModule(_memory_config(cfg, cfg.model))
        elif cfg.model == "gru":
            h = cfg.gru_hidden or d
            if cfg.gru_match:
                h = match_gru_hidden(d, titan_core_params(cfg, cfg.gru_match))
            self.gru_hidden = h
            self.gru = nn.GRU(d, h, batch_first=True)
            self.gru_out = nn.Linear(h, d) if h != d else nn.Identity()
        else:
            assert cfg.model == "none", f"unknown model {cfg.model!r}"

    def write_mask(self, b):
        cfg = self.cfg
        if not (cfg.oracle_write_mask or cfg.oracle_mask_filler):
            return None
        read_only = b["qmask"].clone()
        if cfg.oracle_mask_filler:
            read_only |= b["typ"] == FILL
        return (~read_only).float()

    def forward(self, b):
        h = (self.E_type(b["typ"]) + self.E_key(b["key_id"]) +
             self.E_val(b["val_id"]) + self.E_fil(b["fil_id"]))
        aux, met = h.new_zeros(()), {}
        if self.mem is not None:
            self.mem.reset_memory()
            h, met, aux = self.mem(h, write_mask=self.write_mask(b))
        elif self.gru is not None:
            g, _ = self.gru(h)
            h = h + self.gru_out(g)
        return self.head(self.norm(h)), aux, met


def count_params(model: nn.Module, skip=("E_", "head", "norm")) -> dict:
    total = sum(p.numel() for p in model.parameters())
    core = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(skip))
    return {"total": total, "memory_core": core}


# ── Train / eval ──────────────────────────────────────────────────────────────

def query_acc_loss(logits, b):
    q = b["qmask"]
    lg, tg = logits[q], b["target"][q]
    return F.cross_entropy(lg, tg), (lg.argmax(-1) == tg).float().mean()


@torch.no_grad()
def evaluate(model: Host, cfg: ExpConfig, task: Task, device):
    """Returns (grid, diag). grid: accuracy per eval label (stream task: also per age bucket '<label>|age=<bucket>').
    diag: per eval label, mean memory diagnostics (write gate, median-chunk half-life, state/weight norm ratio)."""
    model.eval()
    diag_acc = defaultdict(lambda: defaultdict(list))
    gen = torch.Generator().manual_seed(12345)           # fixed eval set across runs/models
    tally = defaultdict(lambda: [0, 0])                  # key -> [correct, total]
    for label, b in task.eval_items(cfg.batch_size, gen, device, cfg.eval_batches):
        logits, _, met = model(b)
        for k_ in ("mean_surprise", "mean_write_gate", "retention_halflife_tokens", "state_norm_ratio"):
            if k_ in met:
                diag_acc[label][k_].append(met[k_])
        q = b["qmask"]
        correct = (logits[q].argmax(-1) == b["target"][q]).float()
        tally[label][0] += correct.sum().item()
        tally[label][1] += correct.numel()
        if task.name == "stream":
            age = b["age"][q]
            for lo, hi, name in AGE_BUCKETS:
                m = (age >= lo) & (age <= hi)
                if m.any():
                    t = tally[f"{label}|age={name}"]
                    t[0] += correct[m].sum().item()
                    t[1] += int(m.sum())
    model.train()
    diag = {lab: {k_: sum(v) / len(v) for k_, v in d.items()} for lab, d in diag_acc.items()}
    return {k: c / t for k, (c, t) in tally.items()}, diag


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(__file__),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def run_experiment(cfg: ExpConfig, device: Optional[str] = None, verbose: bool = True,
                   return_model: bool = False, sha: str = ""):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    task = build_task(cfg)
    model = Host(cfg, task).to(device)
    params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * (0.5 * (1 + math.cos(math.pi * s / max(cfg.steps, 1))) * 0.9 + 0.1))
    chance = 1.0 / task.n_vals
    if verbose:
        extra = f" gru_hidden={model.gru_hidden}" if cfg.model == "gru" else ""
        print(f"[{cfg.model}/{cfg.task}] params: {params}{extra}  chance={chance:.3f}  device={device}")

    curve, evals, t0 = [], [], time.time()
    run_loss, run_acc, run_met, skipped = [], [], {}, 0
    for step in range(1, cfg.steps + 1):
        b = task.train_batch(cfg.batch_size, gen, device, progress=(step - 1) / max(cfg.steps, 1))
        logits, aux, met = model(b)
        ce, acc = query_acc_loss(logits, b)
        loss = ce + cfg.aux_weight * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if not torch.isfinite(gn):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        sched.step()
        run_loss.append(ce.item())
        run_acc.append(acc.item())
        if met:
            run_met = {k: met[k] for k in ("mean_surprise", "mean_write_gate", "mean_retention",
                                           "retention_halflife_tokens", "state_norm_ratio", "mean_blend")}
        if step % 50 == 0 and run_loss:
            row = {"step": step, "ce": sum(run_loss) / len(run_loss), "acc": sum(run_acc) / len(run_acc),
                   "aux": float(aux.detach()), "time_s": time.time() - t0, **run_met}
            curve.append(row)
            run_loss, run_acc = [], []
            if verbose and step % 100 == 0:
                ex = "  ".join(f"{k.replace('mean_', '')}={v:.3f}" for k, v in run_met.items())
                print(f"step {step:5d}  ce {row['ce']:.3f}  acc {row['acc']:.3f}  aux {row['aux']:.3f}  {ex}")
        if step % cfg.eval_every == 0 or step == cfg.steps:
            ev, dg = evaluate(model, cfg, task, device)
            evals.append({"step": step, "grid": ev, "diag": dg})
            if verbose:
                print(f"  eval@{step}: " + "  ".join(f"{k}:{v:.2f}" for k, v in ev.items() if "|" not in k))

    result = {"config": asdict(cfg), "git_sha": sha or git_sha(), "params": params, "chance": chance,
              "device": device, "wall_time_s": time.time() - t0, "skipped_nonfinite_steps": skipped,
              "curve": curve, "evals": evals, "final_grid": evals[-1]["grid"] if evals else {},
              "final_diag": evals[-1]["diag"] if evals else {}}
    if cfg.model == "gru":
        result["gru_hidden"] = model.gru_hidden
    return (result, model) if return_model else result


def parse_args() -> ExpConfig:
    p = argparse.ArgumentParser()
    for f in fields(ExpConfig):
        if f.type is bool:
            p.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            p.add_argument(f"--{f.name}", type=f.type, default=f.default)
    return ExpConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    cfg = parse_args()
    print("saved", save_result(run_experiment(cfg)))
