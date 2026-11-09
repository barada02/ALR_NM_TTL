"""
Launcher: loads Modal credentials from the repo's .env into the environment of the Modal CLI, then runs it.
The values are never printed or logged; only MODAL_TOKEN_ID / MODAL_TOKEN_SECRET are forwarded.

  python ResearchWork/launch.py sweep smoke                  # = modal run ResearchWork/modal_app.py --sweep smoke
  python ResearchWork/launch.py sweep p1_stream --seeds 0,1 --force
  python ResearchWork/launch.py llm smoke                     # Phase 2: frozen LLM + memory on a Modal GPU (see modal_llm.py)
  python ResearchWork/launch.py eval <run> [--files all] [--gen 100]   # score a trained checkpoint on the annotated eval sets
  python ResearchWork/launch.py check                         # run memory_module_v4 sanity suite on Modal
  python ResearchWork/launch.py profile                       # check that the credentials work (modal profile current)
  python ResearchWork/launch.py <any modal CLI args>          # e.g.  volume ls minititan-results
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORWARD = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")


def read_env(path: str) -> dict:
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.replace("export ", "").strip()
            out[k] = v.strip().strip('"').strip("'")
    return out


def main(argv):
    vals = read_env(os.path.join(ROOT, ".env"))
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    for k in FORWARD:
        if vals.get(k):
            env[k] = vals[k]
    missing = [k for k in FORWARD if not env.get(k)]
    if missing:
        sys.exit(f"missing {', '.join(missing)} (set in .env or the environment)")

    if argv and argv[0] == "sweep":
        args = ["run", os.path.join("ResearchWork", "modal_app.py") + "::main", "--sweep", *argv[1:]]
    elif argv and argv[0] == "llm":
        args = ["run", os.path.join("ResearchWork", "modal_llm.py") + "::main", "--run", *argv[1:]]
    elif argv and argv[0] == "probe":
        args = ["run", os.path.join("ResearchWork", "modal_llm.py") + "::probe_main", "--run", *argv[1:]]
    elif argv and argv[0] == "eval":
        args = ["run", os.path.join("ResearchWork", "modal_llm.py") + "::eval_main", "--run", *argv[1:]]
    elif argv and argv[0] == "check":
        args = ["run", os.path.join("ResearchWork", "modal_app.py") + "::checks"]
    elif argv and argv[0] == "profile":
        args = ["profile", "current"]
    else:
        args = argv
    cmd = [sys.executable, "-W", "ignore", "-m", "modal", *args]
    return subprocess.call(cmd, cwd=ROOT, env=env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
