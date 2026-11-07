"""Reading/writing experiment result JSONs. Torch-free."""

import glob
import json
import os
import time
from typing import List


def save_result(result: dict, out_dir: str = "") -> str:
    cfg = result["config"]
    out_dir = out_dir or cfg["out_dir"]
    name = cfg["run_name"] or f"{cfg['model']}_s{cfg['seed']}_{int(time.time())}"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    return path


def load_results(*dirs: str) -> List[dict]:
    out = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            try:
                with open(p) as f:
                    r = json.load(f)
                if "config" in r and "final_grid" in r:
                    r["_path"] = p
                    out.append(r)
            except Exception:
                pass
    return out
