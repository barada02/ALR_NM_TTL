"""
Modal wrapper for the frozen-LLM experiments (Phase 2): one GPU container per run.

  python ResearchWork/launch.py llm smoke              # tiny end-to-end run: checks plumbing + measures s/step and GPU memory
  python ResearchWork/launch.py llm gate2a             # insertion-point comparison (mid vs last)
  python ResearchWork/launch.py llm <name> --dry-run

One-time data upload (the jsonl files are gitignored, 152 MB):
  python ResearchWork/launch.py volume put minititan-data data_generation/miniTitan_data /miniTitan_data

Volumes:  minititan-data (/data, the jsonl)  ·  minititan-hf-cache (/cache, HF downloads)  ·
          minititan-results (/vol, result JSON + memory checkpoints under /vol/llm/<run>/)
GPU: A10G by default; set MINITITAN_LLM_GPU=L4 / A100 / ... to change.
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
GPU = os.environ.get("MINITITAN_LLM_GPU") or "A10G"

app = modal.App("minititan-llm")
data_vol = modal.Volume.from_name("minititan-data", create_if_missing=True)
cache_vol = modal.Volume.from_name("minititan-hf-cache", create_if_missing=True)
res_vol = modal.Volume.from_name("minititan-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.15.0", "transformers", "accelerate", "numpy")
    .env({"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(HERE, remote_path=REMOTE_SRC,
                   ignore=lambda p: "__pycache__" in p.parts or p.suffix == ".pyc")
)


@app.function(image=image, gpu=GPU, cpu=4.0, memory=32768, timeout=8 * 60 * 60,
              volumes={"/data": data_vol, "/cache": cache_vol, "/vol": res_vol})
def train_llm(cfg_dict: dict, sha: str = "") -> dict:
    sys.path.insert(0, REMOTE_SRC)
    from llm_config import LLMConfig
    from llm_train import run_llm

    cfg = LLMConfig(**cfg_dict)
    if not os.path.exists(os.path.join(cfg.data_dir, "minititan_long_context.jsonl")):
        raise FileNotFoundError(
            f"{cfg.data_dir} missing on the data volume. Upload once with:\n"
            "  python ResearchWork/launch.py volume put minititan-data data_generation/miniTitan_data /miniTitan_data")
    cfg.out_dir = f"/vol/llm/{cfg.run_name}"
    res = run_llm(cfg, sha=sha)
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "result.json"), "w") as f:
        json.dump(res, f)
    res_vol.commit()
    cache_vol.commit()
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
def main(run: str = "smoke", force: bool = False, dry_run: bool = False):
    sys.path.insert(0, HERE)
    from llm_runs import build_runs, summarize

    cfgs = build_runs(run)
    out_dir = os.path.join(ROOT, "results", "llm")
    todo = [c for c in cfgs if force or not os.path.exists(os.path.join(out_dir, c.run_name + ".json"))]
    print(f"llm run set {run}: {len(cfgs)} runs, launching {len(todo)} on {GPU}")
    if dry_run or not todo:
        for c in todo:
            print("  ", c.run_name)
        return
    os.makedirs(out_dir, exist_ok=True)
    sha, t0 = _git_sha(), time.time()
    for res in train_llm.starmap([(asdict(c), sha) for c in todo], order_outputs=False, return_exceptions=True):
        if isinstance(res, BaseException):
            print(f"FAILED: {repr(res)[:600]}")
            continue
        path = os.path.join(out_dir, res["config"]["run_name"] + ".json")
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
        print(f"[{time.time() - t0:5.0f}s] saved {path}\n{summarize(res)}")
