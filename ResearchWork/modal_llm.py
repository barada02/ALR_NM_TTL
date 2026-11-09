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


EVAL_SETS = ["eval_single", "eval_overwrite", "eval_colliding", "eval_colliding_overwrite", "eval_haystack", "eval_absent"]


@app.function(image=image, gpu=GPU, cpu=4.0, memory=32768, timeout=4 * 60 * 60,
              volumes={"/data": data_vol, "/cache": cache_vol, "/vol": res_vol})
def eval_llm(run: str, files: list, gen_n: int = 100, max_new: int = 24, limit: int = 0, tag: str = "zeroshot", sha: str = "") -> dict:
    """Score a trained checkpoint (/vol/llm/<run>/) on annotated eval sets from the data volume (/data/miniTitan_eval/)."""
    sys.path.insert(0, REMOTE_SRC)
    from llm_config import LLMConfig
    from llm_eval import run_eval

    saved = json.load(open(f"/vol/llm/{run}/result.json"))
    cfg = LLMConfig(**saved["config"])
    cfg.max_ctx_segments = 64                          # haystacks are far longer than the 8-segment training documents
    paths = [f"/data/miniTitan_eval/{f}.jsonl" for f in files]
    res = run_eval(cfg, f"/vol/llm/{run}/{run}_memory.pt", paths, gen_n=gen_n, max_new=max_new, limit=limit, sha=sha)
    res["run"], res["tag"] = run, tag
    os.makedirs("/vol/llm_eval", exist_ok=True)
    with open(f"/vol/llm_eval/{run}__{tag}.json", "w") as f:
        json.dump(res, f)
    res_vol.commit()
    cache_vol.commit()
    return res


@app.function(image=image, gpu=GPU, cpu=4.0, memory=32768, timeout=2 * 60 * 60,
              volumes={"/data": data_vol, "/cache": cache_vol, "/vol": res_vol})
def probe_llm(run: str, files: list, tag: str = "probe", sha: str = "") -> dict:
    """Localisation probes (storage / addressing / read) on a trained checkpoint. Eval-only."""
    sys.path.insert(0, REMOTE_SRC)
    from llm_config import LLMConfig
    from llm_probe import run_probe

    saved = json.load(open(f"/vol/llm/{run}/result.json"))
    cfg = LLMConfig(**saved["config"])
    cfg.max_ctx_segments = 64
    res = run_probe(cfg, f"/vol/llm/{run}/{run}_memory.pt", [f"/data/miniTitan_eval/{f}.jsonl" for f in files], sha=sha)
    res["run"], res["tag"] = run, tag
    os.makedirs("/vol/llm_probe", exist_ok=True)
    with open(f"/vol/llm_probe/{run}__{tag}.json", "w") as f:
        json.dump(res, f)
    res_vol.commit()
    return res


@app.function(image=image, gpu=GPU, cpu=4.0, memory=32768, timeout=2 * 60 * 60,
              volumes={"/data": data_vol, "/cache": cache_vol, "/vol": res_vol})
def keyprobe_llm(n_train: int = 2000, epochs: int = 40, sha: str = "") -> dict:
    """Supervised linear key probe on the FROZEN model (no memory checkpoint involved)."""
    sys.path.insert(0, REMOTE_SRC)
    from llm_config import LLMConfig
    from llm_keyprobe import run_keyprobe

    cfg = LLMConfig()
    rep = run_keyprobe(cfg, "/data/miniTitan_data/minititan_prog_colliding.jsonl",
                       {"seen": "/data/miniTitan_eval/eval_seen_colliding.jsonl",
                        "heldout": "/data/miniTitan_eval/eval_heldout_colliding.jsonl"},
                       n_train=n_train, epochs=epochs)
    rep["git_sha"] = sha
    os.makedirs("/vol/llm_probe", exist_ok=True)
    with open("/vol/llm_probe/keyprobe.json", "w") as f:
        json.dump(rep, f)
    res_vol.commit()
    cache_vol.commit()
    return rep


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
def probe_main(run: str, files: str = "eval_seen_colliding,eval_heldout_colliding", tag: str = "probe"):
    sys.path.insert(0, HERE)
    from llm_probe import summarize_probe

    res = probe_llm.remote(run, files.split(","), tag, _git_sha())
    out_dir = os.path.join(ROOT, "results", "llm_probe")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run}__{tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"saved {path}\n{summarize_probe(res)}")


@app.local_entrypoint()
def keyprobe_main(n_train: int = 2000, epochs: int = 40):
    sys.path.insert(0, HERE)
    from llm_keyprobe import summarize_keyprobe

    rep = keyprobe_llm.remote(n_train, epochs, _git_sha())
    out_dir = os.path.join(ROOT, "results", "llm_probe")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "keyprobe.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=1)
    print("saved", path)
    print(summarize_keyprobe(rep))


@app.local_entrypoint()
def eval_main(run: str, files: str = "all", gen: int = 100, max_new: int = 24, limit: int = 0, tag: str = "zeroshot"):
    sys.path.insert(0, HERE)
    from llm_eval import summarize_eval

    names = EVAL_SETS if files == "all" else files.split(",")
    res = eval_llm.remote(run, names, gen, max_new, limit, tag, _git_sha())
    out_dir = os.path.join(ROOT, "results", "llm_eval")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run}__{tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"saved {path}\n{summarize_eval(res)}")


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
