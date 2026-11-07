"""
Modal wrapper for the synthetic experiments. Thin on purpose: all logic lives in mqar_testbed / sweeps.

  python ResearchWork/launch.py sweep smoke                 # loads credentials from .env, runs the sweep on Modal
  python ResearchWork/launch.py sweep p1_stream --seeds 0,1
  python ResearchWork/launch.py sweep abl_all --dry-run     # list the runs without launching

How it works
  * `run_one` runs ONE config in its own container and writes the result JSON to the Modal Volume
    `minititan-results` (/vol/<sweep>/<run>.json), then returns it.
  * `main` (runs on your machine) builds the sweep, fans it out with `starmap` (parallel containers) and
    saves every result to results/<sweep>/ locally as soon as it arrives. Runs whose JSON already exists
    locally are skipped (use --force to redo). Failed runs are listed at the end and are retried by
    simply launching the sweep again.

Compute: these hosts are tiny (<0.3M params) and dominated by a Python loop over chunks, so CPU containers
are the right default. Set MINITITAN_GPU=T4 (or L4/A10G/...) in the environment to run on a GPU instead.
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REMOTE_SRC = "/root/ResearchWork"

GPU = os.environ.get("MINITITAN_GPU") or None
CPU_CORES = 2.0

app = modal.App("minititan")
results_vol = modal.Volume.from_name("minititan-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.15.0", "numpy")
    .add_local_dir(HERE, remote_path=REMOTE_SRC,
                   ignore=lambda p: "__pycache__" in p.parts or p.suffix == ".pyc")
)


@app.function(image=image, cpu=CPU_CORES, memory=4096, gpu=GPU, timeout=2 * 60 * 60,
              retries=modal.Retries(max_retries=2, initial_delay=5.0),
              volumes={"/vol": results_vol})
def run_one(cfg_dict: dict, sweep: str, sha: str = "") -> dict:
    sys.path.insert(0, REMOTE_SRC)
    import torch
    from exp_config import ExpConfig
    from mqar_testbed import run_experiment

    torch.set_num_threads(int(CPU_CORES))
    cfg = ExpConfig(**cfg_dict)
    res = run_experiment(cfg, verbose=False, sha=sha)
    res["host"] = {"gpu": GPU, "cpu": CPU_CORES, "torch": torch.__version__}

    out = f"/vol/{sweep}"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/{cfg.run_name}.json", "w") as f:
        json.dump(res, f)
    results_vol.commit()
    return res


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "ResearchWork"], cwd=ROOT,
                                        stderr=subprocess.DEVNULL).decode().strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


@app.local_entrypoint()
def main(sweep: str = "smoke", seeds: str = "", dry_run: bool = False, force: bool = False, limit: int = 0):
    sys.path.insert(0, HERE)
    from analyze_results import short_summary
    from results_io import save_result
    from sweeps import build_sweep

    cfgs = build_sweep(sweep, [int(s) for s in seeds.split(",")] if seeds else None)
    local_dir = os.path.join(ROOT, "results", sweep)
    todo = [c for c in cfgs if force or not os.path.exists(os.path.join(local_dir, c.run_name + ".json"))]
    if limit:
        todo = todo[:limit]
    print(f"sweep {sweep}: {len(cfgs)} runs, {len(cfgs) - len(todo)} already done locally, launching {len(todo)}")
    if dry_run or not todo:
        for c in todo:
            print("  ", c.run_name)
        return

    sha, t0, failed, done = _git_sha(), time.time(), [], 0
    args = [(asdict(c), sweep, sha) for c in todo]
    for res in run_one.starmap(args, order_outputs=False, return_exceptions=True):
        if isinstance(res, BaseException):
            failed.append(repr(res)[:300])
            print(f"  FAILED: {failed[-1]}")
            continue
        save_result(res, out_dir=local_dir)
        done += 1
        print(f"[{done}/{len(todo)} {time.time() - t0:5.0f}s] {short_summary(res)}")
    print(f"\n{done} saved to {local_dir}" + (f", {len(failed)} FAILED — re-launch to retry" if failed else ""))
