"""Named LLM run sets (Phase 2) + a compact result summary. Torch-free."""

from dataclasses import replace
from typing import List

from llm_config import LLMConfig


def smoke() -> List[LLMConfig]:
    """Tiny runs: verify the pipeline end to end and measure s/step + GPU memory for each insertion point."""
    base = LLMConfig(steps=12, log_every=2, eval_every=10 ** 9, val_per_type=6, gen_eval_per_type=3,
                     batch_size=2, gen_max_new=8, save_ckpt=False)
    return [replace(base, run_name="smoke_mid", insert="mid"), replace(base, run_name="smoke_last", insert="last")]


def gate2a() -> List[LLMConfig]:
    """Gate 2a: where to inject — mid-layer (frozen upper layers decode it) vs last (straight into lm_head)."""
    base = LLMConfig(steps=800, eval_every=400, warmup=50)
    return [replace(base, run_name="g2a_mid", insert="mid"), replace(base, run_name="g2a_last", insert="last"),
            replace(base, run_name="g2a_mid_w8", insert="mid", tbptt_chunks=8)]   # smoke showed huge grad norms with full BPTT


def full_a() -> List[LLMConfig]:
    """Stage A of the full run: mid insertion + short window won Gate 2a (800 steps). Longer training, one factor at a time."""
    base = LLMConfig(steps=4000, eval_every=1000, warmup=100, insert="mid", tbptt_chunks=8)
    v = {"a_base": {}, "a_w16": dict(tbptt_chunks=16), "a_clip05": dict(inner_clip=0.05),
         "a_L8": dict(insert_layer=8), "a_L16": dict(insert_layer=16), "a_hid512": dict(mem_hidden=512),
         "a_lr3e-4": dict(lr=3e-4), "a_lr3e-3": dict(lr=3e-3)}
    return [replace(base, run_name=k, **kw) for k, kw in v.items()]


# Stage-A best config (lr 3e-4) is the reference: long_context NLL none 3.70 | swap 2.23 | on 2.15 (generic adapter only).
FIX_BASE = dict(steps=3000, eval_every=1000, warmup=100, insert="mid", tbptt_chunks=8, lr=3e-4)


def fix_smoke() -> List[LLMConfig]:
    """12 steps of crop + contrastive loss: checks the new code paths and the memory cost of the extra swap pass."""
    base = LLMConfig(steps=12, log_every=2, eval_every=10 ** 9, val_per_type=6, gen_eval_per_type=3, batch_size=4,
                     gen_max_new=8, save_ckpt=False, insert="mid", tbptt_chunks=8, crop=True, contrast_weight=1.0)
    return [replace(base, run_name="fixsmoke_both")]


def fix_a() -> List[LLMConfig]:
    """E7 fixes: make the document-independent shortcut unprofitable. Judge by swap→on and by `on` NLL vs the generic baseline."""
    base = LLMConfig(**FIX_BASE)
    v = {"f1_contrast": dict(contrast_weight=1.0),
         "f2_crop": dict(crop=True),
         "f3_both": dict(crop=True, contrast_weight=1.0),
         "f4_both_w16": dict(crop=True, contrast_weight=1.0, tbptt_chunks=16),
         "f5_both_m2": dict(crop=True, contrast_weight=1.0, contrast_margin=2.0)}
    return [replace(base, run_name=k, **kw) for k, kw in v.items()]


def fix_c() -> List[LLMConfig]:
    """Replicate and stress the one positive signal (shift_v + crop + window 16): control without shift_v, 2 more seeds,
    wider window, full BPTT, longer training."""
    base = LLMConfig(**{**FIX_BASE, "tbptt_chunks": 16}, shift_v=True, crop=True)
    v = {"c_ctl_noshift": dict(shift_v=False),
         "c_s3_seed1": dict(seed=1), "c_s3_seed2": dict(seed=2),
         "c_w32": dict(tbptt_chunks=32), "c_full": dict(tbptt_chunks=0), "c_long6k": dict(steps=6000, eval_every=2000)}
    return [replace(base, run_name=k, **kw) for k, kw in v.items()]


def final_b() -> List[LLMConfig]:
    """Stage B: the recipe that passed the controls (next-token values + crop curriculum), long BPTT, 6000 steps,
    3 seeds each, larger validation set. Seeds share one validation split (split_seed)."""
    base = LLMConfig(**{**FIX_BASE, "steps": 6000, "eval_every": 2000}, shift_v=True, crop=True,
                     val_per_type=300, gen_eval_per_type=100)
    out = []
    for tag, w in (("full", 0), ("w32", 32)):
        for s in (0, 1, 2):
            out.append(replace(base, run_name=f"b_{tag}_s{s}", tbptt_chunks=w, seed=s))
    return out


def paired_ctl() -> List[LLMConfig]:
    """Paired controls on the SAME val set and training budget as Stage B (6000 steps, lr 3e-4, 300 val/type):
    ctl_adapter_seg    = same module, writes disabled, segmented base  → document-free adapter (fair 'generic' reference)
    ctl_adapter_oracle = same module, writes disabled, full attention  → adapter-matched oracle (fair ceiling)"""
    base = LLMConfig(**{**FIX_BASE, "steps": 6000, "eval_every": 2000}, val_per_type=300, gen_eval_per_type=100)
    return [replace(base, run_name="ctl_adapter_seg", train_mode="adapter_seg"),
            replace(base, run_name="ctl_adapter_oracle", train_mode="adapter_oracle")]


def fix_d() -> List[LLMConfig]:
    """Screen (1 seed, 3000 steps, 300 val/type) the two levers that may change 'model v1':
    chunk size (E8: stale overwrite values fall 0.24 → 0.05 from chunk 16 → 4) and memory width (measurement, not an expected fix).
    The gradient window is kept at ~1024 tokens so ONLY the chunk size changes (32→w32, 16→w64, 8→w128).
    Report accuracy per unit of fast state (state_floats = 2·hidden·896) and the wall-time cost of smaller chunks."""
    base = LLMConfig(**{**FIX_BASE, "steps": 3000, "eval_every": 1000, "tbptt_chunks": 32}, shift_v=True, crop=True,
                     val_per_type=300, gen_eval_per_type=100, chunk_size=32)
    v = {"d_c32": dict(), "d_c16": dict(chunk_size=16, tbptt_chunks=64), "d_c8": dict(chunk_size=8, tbptt_chunks=128),
         "d_h512": dict(mem_hidden=512), "d_h1024": dict(mem_hidden=1024)}
    return [replace(base, run_name=k, **kw) for k, kw in v.items()]


GEM = ("long_context", "dialogue", "agentic")
PROG = ("prog_colliding", "prog_overwrite", "prog_single")


def _mix(gemini: float, colliding: float, overwrite: float, single: float) -> str:
    parts = [f"{t}:{gemini / 3:.5f}" for t in GEM] + [f"prog_colliding:{colliding}", f"prog_overwrite:{overwrite}", f"prog_single:{single}"]
    return ",".join(parts)


def stage2() -> List[LLMConfig]:
    """Stage 2 (see EXPERIMENT_LOG 'Pre-registration'): can the memory learn entity-value BINDING when the data demands it?
    Stage B recipe (6000 steps, shift_v, full BPTT, lr 3e-4, hidden 256, chunk 32); ONLY the mixture changes.
      s2_mix50      50% Gemini · 25% colliding (2-4 questions per document) · 15% overwrite · 10% clean single-fact   (3 seeds)
      s2_mix25      75% Gemini · 12.5 / 7.5 / 5                                                                      dose arm
      s2_single50   50% Gemini · 50% clean single-fact programmatic                                                  control: more programmatic data alone
      s2_mix50_c8   as s2_mix50 with chunk 8 (gradient window 1024 tokens)                                           chunk size after overwrite training
    Programmatic documents are not cropped (crop_annotated=False) so the colliders stay in the document."""
    base = LLMConfig(**{**FIX_BASE, "steps": 6000, "eval_every": 2000, "tbptt_chunks": 0}, shift_v=True, crop=True, val_per_type=300,
                     gen_eval_per_type=100, types=",".join(GEM + PROG), eval_types=",".join(GEM))
    out = [replace(base, run_name=f"s2_mix50_s{s}", seed=s, mix=_mix(0.50, 0.25, 0.15, 0.10)) for s in (0, 1, 2)]
    out += [replace(base, run_name="s2_mix25_s0", mix=_mix(0.75, 0.125, 0.075, 0.05)),
            replace(base, run_name="s2_single50_s0", mix=_mix(0.50, 0.0, 0.0, 0.50)),
            replace(base, run_name="s2_mix50_c8_s0", chunk_size=8, tbptt_chunks=128, mix=_mix(0.50, 0.25, 0.15, 0.10))]
    return out


V5_INIT = "/vol/llm/b_full_s0/b_full_s0_memory.pt"          # model v1 = Stage B seed 0


def _v5_base(**kw) -> LLMConfig:
    """E15 arms (DESIGN_ENTITY_KEYS.md §7): warm start from model v1, Stage 2 mixture, 3000 steps, lr 3e-4, full BPTT, chunk 32."""
    base = dict(steps=3000, eval_every=1000, warmup=100, insert="mid", tbptt_chunks=0, lr=3e-4, shift_v=True, crop=True,
                val_per_type=300, gen_eval_per_type=100, types=",".join(GEM + PROG), eval_types=",".join(GEM),
                mix=_mix(0.50, 0.25, 0.15, 0.10), init_ckpt=V5_INIT, abort_step=1000, abort_min_gain=1.0, ckpt_steps="2000")
    base.update(kw)
    return LLMConfig(**base)


def v5_screen() -> List[LLMConfig]:
    return [_v5_base(run_name="v5_A1_s0", key_mode="composite", pool="mean"),
            _v5_base(run_name="v5_A2_s0", key_mode="composite", pool="attn"),
            _v5_base(run_name="v5_A3_s0", key_mode="composite", pool="attn", key_loss_weight=0.1),
            _v5_base(run_name="v5_D_s0", key_mode="composite", pool="oracle"),
            _v5_base(run_name="v5_R_s0", key_mode="token")]


def v5_confirm() -> List[LLMConfig]:
    """E15 confirmation of the LOCKED arm A3: 6000 steps, seeds 0 / 1 / 2, identical configs (seed 0 re-run fresh)."""
    return [_v5_base(run_name=f"v5c_A3_s{s}", seed=s, key_mode="composite", pool="attn", key_loss_weight=0.1,
                     steps=6000, eval_every=2000, ckpt_steps="3000") for s in (0, 1, 2)]


def v5_smoke() -> List[LLMConfig]:
    """Plumbing only: a few steps of each composite variant (incl. oracle masks and the key loss) and the token reference."""
    tiny = dict(steps=12, log_every=2, eval_every=10 ** 9, val_per_type=6, gen_eval_per_type=3, batch_size=2, gen_max_new=8,
                save_ckpt=False, abort_step=6, abort_min_gain=-100.0, ckpt_steps="")
    return [_v5_base(run_name="v5smoke_A3", key_mode="composite", pool="attn", key_loss_weight=0.1, **tiny),
            _v5_base(run_name="v5smoke_D", key_mode="composite", pool="oracle", **tiny),
            _v5_base(run_name="v5smoke_R", key_mode="token", **tiny)]


def fix_b() -> List[LLMConfig]:
    """Root-cause test: same-position (k_t, v_t) cannot store a multi-token string → next-token values (shift_v).
    Success = `on` NLL clearly BELOW the generic adapter (long_context 2.15) and exact-match recall > 0, not just swap→on."""
    base = LLMConfig(**FIX_BASE, shift_v=True)
    v = {"s1_shift": dict(),
         "s2_shift_crop": dict(crop=True),
         "s3_shift_crop_w16": dict(crop=True, tbptt_chunks=16),
         "s4_shift_crop_contrast_w16": dict(crop=True, contrast_weight=1.0, tbptt_chunks=16)}
    return [replace(base, run_name=k, **kw) for k, kw in v.items()]


RUNS = {"smoke": smoke, "gate2a": gate2a, "full_a": full_a, "fix_smoke": fix_smoke, "fix_a": fix_a, "fix_b": fix_b,
        "fix_c": fix_c, "final_b": final_b, "paired_ctl": paired_ctl, "fix_d": fix_d, "stage2": stage2,
        "v5_screen": v5_screen, "v5_smoke": v5_smoke, "v5_confirm": v5_confirm}


def build_runs(name: str) -> List[LLMConfig]:
    if name not in RUNS:
        raise ValueError(f"unknown run set {name!r}; available: {sorted(RUNS)}")
    return RUNS[name]()


def summarize(res: dict) -> str:
    c, f = res["config"], res["final"]
    lines = [f"== {c['run_name']}  insert={c['insert']} mode={c['mem_mode']}  mem={res['memory_params']/1e6:.2f}M  "
             f"train {res['train_time_s']/60:.1f}min  peak {res['peak_gpu_gb']:.1f}GB  skipped={res['skipped_nonfinite_steps']}"]
    for t, r in f.items():
        n = r["nll"]
        sw = f" | swap {n['swap']:.3f}" if "swap" in n else ""
        if "oracle_adapter" in n:
            sw += f" | ORACLE+ADAPTER {n['oracle_adapter']:.3f}"
        rec = f"  recovered {r['frac_gain_recovered']*100:.0f}% of oracle gap" if "frac_gain_recovered" in r else ""
        gen_on = {"long_context": 2.15, "dialogue": 2.05, "agentic": 2.97}.get(t)          # Stage A best (generic adapter)
        vg = f" vs-generic {n['on'] - gen_on:+.2f}" if gen_on else ""
        ds = f"  DOC-SPECIFIC {r['nll_gain_on_vs_swap']:+.3f} ({r['doc_specific_share']*100:.0f}% of gain)" if "doc_specific_share" in r else ""
        lines.append(f"  {t:13s} NLL none {n['none']:.3f} | off {n['off']:.3f}{sw} | ON {n['on']:.3f} | oracle {n['oracle']:.3f}"
                     f"   gain(on-vs-none) {r['nll_gain_on_vs_none']:+.3f}{vg}{ds}")
        if r.get("nll_by_fact_distance"):
            lines.append("  " + " " * 13 + "by fact distance (tokens), swap→on: " + "  ".join(
                f"{k}: {v['swap']:.2f}→{v['on']:.2f} (n={v['count']})" for k, v in r["nll_by_fact_distance"].items()))
        g = r["generation"]
        dr = f"  detail-recall none/on/oracle {g['none'].get('digit_recall', float('nan')):.2f}/{g['on'].get('digit_recall', float('nan')):.2f}/{g['oracle'].get('digit_recall', float('nan')):.2f}"
        pref = ""
        if "pref_acc" in r:
            pref = "   pref-acc none/off/on/oracle " + "/".join(f"{r['pref_acc'][m]:.2f}" for m in ("none", "off", "on", "oracle"))
        lines.append(f"  {'':13s} gen contains none/on/oracle {g['none']['contains']:.2f}/{g['on']['contains']:.2f}/"
                     f"{g['oracle']['contains']:.2f}   F1 {g['none']['f1']:.2f}/{g['on']['f1']:.2f}/{g['oracle']['f1']:.2f}" + dr + pref)
    return "\n".join(lines)


if __name__ == "__main__":
    for n in RUNS:
        print(n, [c.run_name for c in build_runs(n)])
