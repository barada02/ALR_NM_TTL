"""
MiniTitan Phase 0/1 testbed — multi-query associative recall (MQAR) with a cross-chunk bottleneck.

Task (one sequence):
    [ pair tokens ... ][ filler ... ][ query tokens ... ]
    pair token   = E_key[k] + E_val[v] + E_type[PAIR]     (binding is done by the embedding, so the test
    filler token = E_fill[f]            + E_type[FILL]     isolates the *memory*, not the host's ability to
    query token  = E_key[k]             + E_type[QUERY]    bind adjacent tokens)
Every query must be answered with the v bound to k. Pairs and queries are always in different chunks, so
the only route from pair to query is the memory state. There is no attention and no convolution in the
host, hence the no-memory baseline is exactly at chance — that is the point of Gate 0.

Hosts (--model):
    none     embedding → head                       (chance; the null for the whole testbed)
    gru      embedding → GRU over all tokens → head (Branch C: plain recurrent state, "is TTL buying anything?")
    titan_a  embedding → MiniTitan v4 Mode A → head (learned update rule, TBPTT + aux loss)
    titan_b  embedding → MiniTitan v4 Mode B → head (gradient update + momentum, Titans-faithful)

Run:   python mqar_testbed.py --model titan_b --steps 1500
Output: results/<run_id>.json   (config + git SHA + curves + eval grid)  — see ROADMAP "Infrastructure".
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, fields
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_module_v4 import MemoryConfig, MiniTitanMemoryModule  # noqa: E402

PAIR, FILL, QUERY = 0, 1, 2


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ExpConfig:
    model: str = "titan_a"           # none | gru | titan_a | titan_b
    # host / memory
    dim: int = 64
    mem_hidden: int = 128
    mem_layers: int = 2
    rank: int = 8
    chunk_size: int = 16
    tbptt_chunks: int = 4            # 0 = full BPTT
    inner_lr: float = 0.2
    inner_clip: float = 0.1
    use_momentum: bool = True
    second_order: bool = True
    aux_lag_chunks: int = 4
    # task
    n_keys: int = 64
    n_vals: int = 32
    n_filler_vocab: int = 32
    n_pairs: int = 8
    train_filler_max: int = 48       # train filler length ~ U[0, this]
    eval_fillers: str = "0,48,128,512"
    eval_pairs: str = "4,8,16"
    oracle_write_mask: bool = False  # force read-only on query tokens (removes the write-gate learning problem)
    # optimisation
    batch_size: int = 32
    steps: int = 1500
    lr: float = 3e-3
    aux_weight: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 4
    seed: int = 0
    out_dir: str = "results"
    run_name: str = ""


# ── Data ──────────────────────────────────────────────────────────────────────

def make_batch(cfg: ExpConfig, B: int, n_pairs: int, filler_len: int, gen: torch.Generator, device):
    """Returns dict of (B,T) long tensors; `target`/`qmask` are only meaningful at query positions."""
    C = cfg.chunk_size
    NK, NV, NF = cfg.n_keys, cfg.n_vals, cfg.n_filler_vocab
    keys = torch.rand(B, NK, generator=gen).argsort(-1)[:, :n_pairs]                # distinct keys per sequence
    vals = torch.randint(0, NV, (B, n_pairs), generator=gen)
    n_fill = math.ceil((n_pairs + filler_len) / C) * C - n_pairs                    # queries start on a chunk boundary
    fill = torch.randint(0, NF, (B, n_fill), generator=gen)
    perm = torch.rand(B, n_pairs, generator=gen).argsort(-1)                        # query order
    qkeys = keys.gather(1, perm)
    qvals = vals.gather(1, perm)

    T = n_pairs + n_fill + n_pairs
    typ = torch.full((B, T), FILL, dtype=torch.long)
    typ[:, :n_pairs] = PAIR
    typ[:, n_pairs + n_fill:] = QUERY
    key_id = torch.full((B, T), NK, dtype=torch.long)                               # NK = null (zero) embedding
    val_id = torch.full((B, T), NV, dtype=torch.long)
    fil_id = torch.full((B, T), NF, dtype=torch.long)
    key_id[:, :n_pairs] = keys
    val_id[:, :n_pairs] = vals
    fil_id[:, n_pairs:n_pairs + n_fill] = fill
    key_id[:, n_pairs + n_fill:] = qkeys
    target = torch.zeros(B, T, dtype=torch.long)
    target[:, n_pairs + n_fill:] = qvals
    qmask = typ == QUERY
    return {k: v.to(device) for k, v in dict(typ=typ, key_id=key_id, val_id=val_id, fil_id=fil_id,
                                              target=target, qmask=qmask).items()}


# ── Hosts ─────────────────────────────────────────────────────────────────────

class Host(nn.Module):
    def __init__(self, cfg: ExpConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.E_type = nn.Embedding(3, d)
        self.E_key = nn.Embedding(cfg.n_keys + 1, d, padding_idx=cfg.n_keys)
        self.E_val = nn.Embedding(cfg.n_vals + 1, d, padding_idx=cfg.n_vals)
        self.E_fil = nn.Embedding(cfg.n_filler_vocab + 1, d, padding_idx=cfg.n_filler_vocab)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.n_vals)
        self.mem = None
        self.gru = None
        if cfg.model in ("titan_a", "titan_b"):
            self.mem = MiniTitanMemoryModule(MemoryConfig(
                dim=d, mem_hidden=cfg.mem_hidden, mem_layers=cfg.mem_layers, rank=cfg.rank,
                chunk_size=cfg.chunk_size, mode="A" if cfg.model == "titan_a" else "B",
                tbptt_chunks=cfg.tbptt_chunks or None, inner_lr=cfg.inner_lr,
                inner_clip=cfg.inner_clip,
                use_momentum=cfg.use_momentum, second_order=cfg.second_order,
                aux_lag_chunks=cfg.aux_lag_chunks))
        elif cfg.model == "gru":
            self.gru = nn.GRU(d, d, batch_first=True)
        else:
            assert cfg.model == "none"

    def forward(self, b):
        h = (self.E_type(b["typ"]) + self.E_key(b["key_id"]) +
             self.E_val(b["val_id"]) + self.E_fil(b["fil_id"]))
        aux, met = h.new_zeros(()), {}
        if self.mem is not None:
            self.mem.reset_memory()
            wm = (~b["qmask"]).float() if self.cfg.oracle_write_mask else None
            h, met, aux = self.mem(h, write_mask=wm)
        elif self.gru is not None:
            g, _ = self.gru(h)
            h = h + g
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
def evaluate(model: Host, cfg: ExpConfig, device) -> dict:
    model.eval()
    gen = torch.Generator().manual_seed(12345)           # fixed eval set across runs/models
    grid = {}
    for n_pairs in [int(x) for x in cfg.eval_pairs.split(",")]:
        for fl in [int(x) for x in cfg.eval_fillers.split(",")]:
            accs = []
            for _ in range(cfg.eval_batches):
                b = make_batch(cfg, cfg.batch_size, n_pairs, fl, gen, device)
                logits, _, _ = model(b)
                accs.append(query_acc_loss(logits, b)[1].item())
            grid[f"pairs={n_pairs},filler={fl}"] = sum(accs) / len(accs)
    model.train()
    return grid


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(__file__),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def run_experiment(cfg: ExpConfig, device: Optional[str] = None, verbose: bool = True,
                   return_model: bool = False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = Host(cfg).to(device)
    params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * (0.5 * (1 + math.cos(math.pi * s / max(cfg.steps, 1))) * 0.9 + 0.1))
    chance = 1.0 / cfg.n_vals
    if verbose:
        print(f"[{cfg.model}] params: {params}  chance={chance:.3f}  device={device}")

    curve, evals, t0 = [], [], time.time()
    run_loss, run_acc, run_met = [], [], {}
    for step in range(1, cfg.steps + 1):
        fl = int(torch.randint(0, cfg.train_filler_max + 1, (1,), generator=gen))
        b = make_batch(cfg, cfg.batch_size, cfg.n_pairs, fl, gen, device)
        logits, aux, met = model(b)
        ce, acc = query_acc_loss(logits, b)
        loss = ce + cfg.aux_weight * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if not torch.isfinite(gn):
            if verbose:
                print(f"step {step}: non-finite grad norm, skipping update")
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        sched.step()
        run_loss.append(ce.item())
        run_acc.append(acc.item())
        if met:
            run_met = {k: met[k] for k in ("mean_surprise", "mean_write_gate", "mean_retention",
                                           "retention_halflife_tokens", "state_norm_ratio", "mean_blend")}
        if step % 50 == 0:
            row = {"step": step, "ce": sum(run_loss) / len(run_loss), "acc": sum(run_acc) / len(run_acc),
                   "aux": float(aux.detach()), "time_s": time.time() - t0, **run_met}
            curve.append(row)
            run_loss, run_acc = [], []
            if verbose and step % 100 == 0:
                extra = "  ".join(f"{k.replace('mean_', '')}={v:.3f}" for k, v in run_met.items())
                print(f"step {step:5d}  ce {row['ce']:.3f}  acc {row['acc']:.3f}  aux {row['aux']:.3f}  {extra}")
        if step % cfg.eval_every == 0 or step == cfg.steps:
            ev = evaluate(model, cfg, device)
            evals.append({"step": step, "grid": ev})
            if verbose:
                print(f"  eval@{step}: " + "  ".join(f"{k}:{v:.2f}" for k, v in ev.items()))

    result = {"config": asdict(cfg), "git_sha": git_sha(), "params": params, "chance": chance,
              "device": device, "wall_time_s": time.time() - t0, "curve": curve, "evals": evals,
              "final_grid": evals[-1]["grid"] if evals else {}}
    return (result, model) if return_model else result


def save_result(result: dict) -> str:
    cfg = result["config"]
    name = cfg["run_name"] or f"{cfg['model']}_s{cfg['seed']}_{int(time.time())}"
    os.makedirs(cfg["out_dir"], exist_ok=True)
    path = os.path.join(cfg["out_dir"], f"{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    return path


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
    res = run_experiment(cfg)
    print("saved", save_result(res))
