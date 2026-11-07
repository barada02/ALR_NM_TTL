"""
Aggregate result JSONs: mean ± std over seeds per (sweep, tag). Torch-free; run locally after a sweep.

  python ResearchWork/analyze_results.py results/p1_stream
  python ResearchWork/analyze_results.py results/p1_stream --ages facts=512      # accuracy vs age at 512 facts
  python ResearchWork/analyze_results.py results --keys "facts=64,facts=512" --out results/summary.md
"""

import argparse
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_io import load_results  # noqa: E402


def group_name(r: dict) -> str:
    name = r["config"].get("run_name", "")
    parts = name.split("__")
    return "__".join(parts[:-1]) if len(parts) >= 3 else (name or r["config"]["model"])


def short_summary(r: dict, max_keys: int = 6) -> str:
    g = {k: v for k, v in r["final_grid"].items() if "|" not in k}
    keys = list(g)[:max_keys] if len(g) <= max_keys else [list(g)[0], *list(g)[-(max_keys - 1):]]
    body = " ".join(f"{k.split(',')[-1] if ',' in k else k}:{g[k]:.2f}" for k in keys)
    return f"{r['config']['run_name'] or r['config']['model']:44s} {body}  ({r['wall_time_s']:.0f}s)"


def _fmt(vals):
    if not vals:
        return "—"
    m = statistics.mean(vals)
    return f"{m:.2f}±{statistics.stdev(vals):.2f}" if len(vals) > 1 else f"{m:.2f}"


def table(results, keys=None, ages=None) -> str:
    groups = defaultdict(list)
    for r in results:
        groups[group_name(r)].append(r)
    if ages:   # columns = age buckets of one eval label
        cols = [k for k in sorted({k for r in results for k in r["final_grid"] if k.startswith(ages + "|age=")},
                                  key=lambda s: [int(x) for x in s.split("age=")[1].replace("+", "").split("-")])]
    elif keys:
        cols = keys
    else:
        cols = sorted({k for r in results for k in r["final_grid"] if "|" not in k})
    head = ["run", "n", "core params"] + [c.replace(ages + "|", "") if ages else c for c in cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for g, rs in sorted(groups.items()):
        core = rs[0]["params"]["memory_core"]
        cells = [_fmt([r["final_grid"][c] for r in rs if c in r["final_grid"]]) for c in cols]
        lines.append("| " + " | ".join([g, str(len(rs)), f"{core:,}"] + cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--keys", default="", help="comma-separated eval labels (default: all overall labels)")
    p.add_argument("--ages", default="", help="eval label whose age-bucket breakdown to show, e.g. facts=512")
    p.add_argument("--out", default="")
    a = p.parse_args()
    res = load_results(*a.dirs)
    if not res:
        sys.exit("no results found")
    out = table(res, keys=[k for k in a.keys.split(",") if k] or None, ages=a.ages or None)
    print(out)
    if a.out:
        with open(a.out, "w") as f:
            f.write(out + "\n")
