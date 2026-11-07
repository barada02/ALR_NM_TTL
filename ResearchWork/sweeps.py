"""
Named experiment sweeps. A sweep is just a list of ExpConfig; the Modal launcher fans them out.

To add an experiment: write a function returning list[ExpConfig] and register it in SWEEPS.
Run names are  <sweep>__<tag>__s<seed>  and are unique, so re-launching skips finished runs.

Torch-free on purpose (imports exp_config only).
"""

from dataclasses import replace
from typing import Callable, Dict, Iterable, List, Optional

from exp_config import ExpConfig

DEFAULT_SEEDS = (0, 1, 2)

# Phase 1 host line-up. The two GRUs are parameter-matched to Mode B / Mode A's core (Branch C, the null).
HOSTS: Dict[str, dict] = {
    "none":      dict(model="none"),
    "gru_small": dict(model="gru", gru_match="titan_b"),
    "gru_large": dict(model="gru", gru_match="titan_a"),
    "titan_a":   dict(model="titan_a"),
    "titan_b":   dict(model="titan_b"),
}

TYPED = dict(task="typed", steps=1500, eval_every=500)
STREAM = dict(task="stream", steps=3000, eval_every=1000)


def _cfg(sweep: str, tag: str, seed: int, **kw) -> ExpConfig:
    return ExpConfig(**{**kw, "seed": seed, "run_name": f"{sweep}__{tag}__s{seed}"})


def _grid(sweep: str, variants: Dict[str, dict], seeds: Iterable[int], base: dict) -> List[ExpConfig]:
    return [_cfg(sweep, tag, s, **{**base, **kw}) for tag, kw in variants.items() for s in seeds]


# ── sweeps ────────────────────────────────────────────────────────────────────

def smoke(seeds):
    """Tiny end-to-end check of the Modal pipeline and both tasks (a minute or two, pennies)."""
    fast = dict(steps=60, eval_every=60, eval_batches=1, eval_facts="16,64", eval_fillers="0,48", eval_pairs="8",
                batch_size=16)
    v = {f"stream_{h}": {**HOSTS[h], "task": "stream"} for h in ("none", "gru_small", "titan_a", "titan_b")}
    v["typed_titan_b"] = {**HOSTS["titan_b"], "task": "typed"}
    return _grid("smoke", v, seeds[:1], fast)


def p1_typed(seeds):
    """Phase 0 task, all hosts, multiple seeds, matched GRUs (error bars for the Gate-0 table)."""
    return _grid("p1_typed", HOSTS, seeds, TYPED)


def p1_stream(seeds):
    """Hard task: distractor facts look like facts, one token type, queries uniform over all facts."""
    return _grid("p1_stream", HOSTS, seeds, STREAM)


def abl_b_mask(seeds):
    """Why does Mode B degrade at long distance? Hypothesis: it keeps writing junk during filler."""
    v = {"base": {}, "mask_filler": dict(oracle_mask_filler=True)}
    return _grid("abl_b_mask", v, seeds, {**TYPED, **HOSTS["titan_b"]})


def abl_momentum(seeds):
    v = {"momentum": dict(use_momentum=True), "no_momentum": dict(use_momentum=False)}
    return _grid("abl_momentum", v, seeds, {**STREAM, **HOSTS["titan_b"]})


def abl_second_order(seeds):
    """First-order meta-gradients are much cheaper — does Mode B need the second-order path?"""
    v = {"second_order": dict(second_order=True), "first_order": dict(second_order=False)}
    return _grid("abl_second_order", v, seeds, {**STREAM, **HOSTS["titan_b"]})


def abl_rank(seeds):
    v = {f"rank{r}": dict(rank=r) for r in (2, 4, 8, 16)}
    return _grid("abl_rank", v, seeds, {**STREAM, **HOSTS["titan_a"]})


def abl_tbptt(seeds):
    out = []
    for h in ("titan_a", "titan_b"):
        v = {f"{h}_tbptt{w}": dict(tbptt_chunks=w) for w in (1, 2, 4, 8)}
        out += _grid("abl_tbptt", v, seeds, {**STREAM, **HOSTS[h]})
    return out


def abl_aux(seeds):
    """Mode A aux loss: none / immediate-only / immediate+delayed / delayed at 10x weight."""
    v = {"no_aux": dict(aux_weight=0.0, aux_lag_chunks=0),
         "imm_only": dict(aux_weight=0.1, aux_lag_chunks=0),
         "imm+delayed": dict(aux_weight=0.1, aux_lag_chunks=4),
         "imm+delayed_w1": dict(aux_weight=1.0, aux_lag_chunks=4)}
    return _grid("abl_aux", v, seeds, {**STREAM, **HOSTS["titan_a"]})


def abl_inner_lr(seeds):
    v = {f"lr{lr}": dict(inner_lr=lr) for lr in (0.05, 0.1, 0.2, 0.4)}
    return _grid("abl_inner_lr", v, seeds, {**STREAM, **HOSTS["titan_b"]})


ABLATIONS = [abl_b_mask, abl_momentum, abl_second_order, abl_rank, abl_tbptt, abl_aux, abl_inner_lr]


def abl_all(seeds):
    return [c for f in ABLATIONS for c in f(seeds)]


def all_p1(seeds):
    return p1_typed(seeds) + p1_stream(seeds)


SWEEPS: Dict[str, Callable] = {f.__name__: f for f in
                               [smoke, p1_typed, p1_stream, all_p1, *ABLATIONS, abl_all]}


def build_sweep(name: str, seeds: Optional[Iterable[int]] = None) -> List[ExpConfig]:
    if name not in SWEEPS:
        raise ValueError(f"unknown sweep {name!r}; available: {sorted(SWEEPS)}")
    cfgs = SWEEPS[name](tuple(seeds) if seeds else DEFAULT_SEEDS)
    names = [c.run_name for c in cfgs]
    assert len(names) == len(set(names)), "duplicate run names in sweep"
    return [replace(c, out_dir=f"results/{name}") for c in cfgs]


if __name__ == "__main__":
    for n in SWEEPS:
        print(f"{n:18s} {len(build_sweep(n)):3d} runs")
