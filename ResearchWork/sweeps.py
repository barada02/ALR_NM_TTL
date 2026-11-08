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

# Linear associative memory = 1-layer Mode B (a single d×d map updated by the delta rule). NOT a DeltaNet (no per-channel gating,
# different normalisation) — it isolates whether MLP depth/nonlinearity matters. lin_b_d128 matches the 2-layer MLP's state size
# (128×128 = 16,384 floats = 2 × 128×64 at dim 64); lin_b at dim 64 has 4× less state.
HOSTS_EXT: Dict[str, dict] = {**HOSTS,
    "lin_b":      dict(model="titan_b", mem_layers=1),
    "lin_b_d128": dict(model="titan_b", mem_layers=1, dim=128)}

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


# ── interference: overwrite / colliding keys, and the linear-memory ablation ─────────────────────────────────

# Mode B recipe from E6 (full BPTT + curriculum + small clip, no momentum) for every memory host, so only the host differs.
UPD = dict(task="update", steps=3000, eval_every=1000, tbptt_chunks=0, use_momentum=False, curriculum=0.5, inner_clip=0.05)


def upd_main(seeds):
    """Same key re-written with new values; query = latest value. Reports accuracy and the stale-answer rate."""
    return _grid("upd_main", HOSTS_EXT, seeds, UPD)


def upd_similar(seeds):
    """Colliding keys: 4 attributes per entity, so keys of one entity share part of their representation."""
    return _grid("upd_similar", HOSTS_EXT, seeds, {**UPD, "n_attrs": 4})


def upd_diag(seeds):
    """Why are ~80% of overwrite errors STALE values? H1: writes inside one chunk are simultaneous, so same-key order is lost
    (smaller chunks should help). H2: writes are too gentle (step clip 0.05) to fully replace the old value (looser clip should help)."""
    v = {}
    for h in ("titan_b", "lin_b_d128"):
        for tag, kw in {"c8": dict(chunk_size=8), "c4": dict(chunk_size=4), "clip0.3": dict(inner_clip=0.3),
                        "clip1": dict(inner_clip=1.0)}.items():
            v[f"{h}_{tag}"] = {**HOSTS_EXT[h], **kw}
    return _grid("upd_diag", v, seeds, UPD)


def lin_stream(seeds):
    """Linear associative memory on the stream task with the E6 recipe; compare with ret_combo__full256_cur_clip05 (2-layer)."""
    base = {**STREAM, "use_momentum": False, "train_facts_max": 256, "tbptt_chunks": 0, "curriculum": 0.5, "inner_clip": 0.05}
    v = {k: HOSTS_EXT[k] for k in ("lin_b", "lin_b_d128")}
    return _grid("lin_stream", v, seeds, base)


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


# ── retention round: why does Mode B lose facts written long ago? ─────────────
# Baseline = p1_stream/titan_b (0.99 / 0.70 / 0.43 at 64 / 256 / 512 facts; accuracy by age 0.73 → 0.23).
# Hypotheses: (a) never trained beyond ~4 chunks, (b) memory too small / interference, (c) inner_clip throttles
# writes (16 facts share one clipped step), (d) chunk too coarse, (e) decay too aggressive.

def ret_dist(seeds):
    """(a) train on longer streams, with and without a longer TBPTT window."""
    v = {"base": {},
         "train128": dict(train_facts_max=128), "train256": dict(train_facts_max=256),
         "train128_w8": dict(train_facts_max=128, tbptt_chunks=8), "train256_w8": dict(train_facts_max=256, tbptt_chunks=8)}
    return _grid("ret_dist", v, seeds, {**STREAM, **HOSTS["titan_b"]})


def ret_capacity(seeds):
    """(b) bigger memory MLP."""
    v = {"hid256": dict(mem_hidden=256), "hid512": dict(mem_hidden=512), "hid256_L3": dict(mem_hidden=256, mem_layers=3)}
    return _grid("ret_capacity", v, seeds, {**STREAM, **HOSTS["titan_b"]})


def ret_clip(seeds):
    """(c) loosen / tighten the Mode B step clip (stability vs write strength)."""
    v = {f"clip{c}": dict(inner_clip=c) for c in (0.05, 0.3, 1.0)}
    return _grid("ret_clip", v, seeds, {**STREAM, **HOSTS["titan_b"]})


def ret_chunk(seeds):
    """(d) finer / coarser write granularity (TBPTT window kept ≈ 64 tokens)."""
    v = {"c8_w8": dict(chunk_size=8, tbptt_chunks=8), "c32_w2": dict(chunk_size=32, tbptt_chunks=2)}
    return _grid("ret_chunk", v, seeds, {**STREAM, **HOSTS["titan_b"]})


def ret_decay(seeds):
    """(e) retention: higher init, or frozen at a near-1 constant (no learned forgetting)."""
    v = {"bias6": dict(retention_bias=6.0), "bias8": dict(retention_bias=8.0),
         "fixed_bias8": dict(retention_bias=8.0, retention_learned=False),
         "fixed_bias4": dict(retention_bias=4.0, retention_learned=False)}
    return _grid("ret_decay", v, seeds, {**STREAM, **HOSTS["titan_b"]})


def ret_bptt(seeds):
    """Full BPTT (no detach) and coarse chunks. Decides the LLM training recipe: a long-context fact sits 80+
    chunks before its question, far beyond any TBPTT window, so gradient can only reach the write if BPTT is long."""
    v = {"train256_full": dict(train_facts_max=256, tbptt_chunks=0),
         "c64_w4": dict(chunk_size=64, tbptt_chunks=4, train_facts_min=64, train_facts_max=256),
         "c64_full": dict(chunk_size=64, tbptt_chunks=0, train_facts_min=64, train_facts_max=256)}
    return _grid("ret_bptt", v, seeds, {**STREAM, **HOSTS["titan_b"]})


def ret_combo(seeds):
    """Confirmation: combine what helped (long horizon, small step clip, no momentum) and cure take-off with a curriculum."""
    base = {**STREAM, **HOSTS["titan_b"], "use_momentum": False, "train_facts_max": 256}
    v = {"full256_nocur": dict(tbptt_chunks=0),
         "full256_cur": dict(tbptt_chunks=0, curriculum=0.5),
         "full256_cur_clip05": dict(tbptt_chunks=0, curriculum=0.5, inner_clip=0.05),
         "w8_256_cur": dict(tbptt_chunks=8, curriculum=0.5),
         "w8_256_cur_clip05": dict(tbptt_chunks=8, curriculum=0.5, inner_clip=0.05)}
    return _grid("ret_combo", v, seeds, base)


RETENTION = [ret_dist, ret_capacity, ret_clip, ret_chunk, ret_decay]   # ret_bptt is standalone (launched separately)


def ret_all(seeds):
    return [c for f in RETENTION for c in f(seeds)]


ABLATIONS = [abl_b_mask, abl_momentum, abl_second_order, abl_rank, abl_tbptt, abl_aux, abl_inner_lr]


def abl_all(seeds):
    return [c for f in ABLATIONS for c in f(seeds)]


def all_p1(seeds):
    return p1_typed(seeds) + p1_stream(seeds)


SWEEPS: Dict[str, Callable] = {f.__name__: f for f in
                               [smoke, p1_typed, p1_stream, all_p1, *ABLATIONS, abl_all, *RETENTION, ret_all, ret_bptt, ret_combo,
                                upd_main, upd_similar, lin_stream, upd_diag]}


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
