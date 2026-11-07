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


RUNS = {"smoke": smoke, "gate2a": gate2a, "full_a": full_a}


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
        rec = f"  recovered {r['frac_gain_recovered']*100:.0f}% of oracle gap" if "frac_gain_recovered" in r else ""
        lines.append(f"  {t:13s} NLL none {n['none']:.3f} | off {n['off']:.3f}{sw} | ON {n['on']:.3f} | oracle {n['oracle']:.3f}"
                     f"   gain(on-vs-none) {r['nll_gain_on_vs_none']:+.3f}{rec}")
        g = r["generation"]
        pref = ""
        if "pref_acc" in r:
            pref = "   pref-acc none/off/on/oracle " + "/".join(f"{r['pref_acc'][m]:.2f}" for m in ("none", "off", "on", "oracle"))
        lines.append(f"  {'':13s} gen contains none/on/oracle {g['none']['contains']:.2f}/{g['on']['contains']:.2f}/"
                     f"{g['oracle']['contains']:.2f}   F1 {g['none']['f1']:.2f}/{g['on']['f1']:.2f}/{g['oracle']['f1']:.2f}" + pref)
    return "\n".join(lines)


if __name__ == "__main__":
    for n in RUNS:
        print(n, [c.run_name for c in build_runs(n)])
