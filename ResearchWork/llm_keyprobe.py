"""
Supervised linear KEY probe (E14). Question: can a FIXED LINEAR MAP extract "which entity is being asked about" from the frozen
model's hidden states, for entities and templates it never saw during probe training?

Frozen Qwen only (no memory): the segmented pipeline is run once with output_hidden_states=True, and features are read at
several layers and positions:
  question side   answer  = the token that predicts the first answer token (what the memory reads with)
                  entity  = mean over the entity tokens in the question ("flow meter at Ridge 4")
                  span    = mean over the whole question prompt
  document side   write   = the token before the fact's first value token (where the memory WRITES the key, with shift_v)
                  entity  = mean over the entity tokens in that fact's sentence
  layers          0 (embeddings) · 6 · 12 (= the memory's insertion layer) · 18 · 23

A contrastive probe scores fact i of a document as cos(A·q, B·d_i)/τ and is trained (cross-entropy over the k facts) to pick the
asked fact. Variants: separate maps A, B (the most permissive linear test) and a shared map A = B (closest to the memory's single
W_k). Trained on programmatic TRAINING-pool documents; tested on the seen eval set (same entity pools, other documents) and on the
HELD-OUT set (new devices, new locations, new templates). Also reported: raw cosine without any probe.
Positive control: layer 0, question entity tokens vs fact entity tokens is a lexical match and must succeed.
"""

import math
import re
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_config import LLMConfig
from llm_data import PROMPT, collate, tokenize_example
from llm_eval import load_examples

LAYERS = (0, 6, 12, 18, 23)
Q_VARIANTS = ("answer", "entity", "span")
D_VARIANTS = ("write", "entity")
DEVICES = ["flow meter", "pressure sensor", "compressor", "cooling pump", "control valve", "generator", "transformer", "gearbox"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def tok_span(offsets, a: int, b: int, limit: int) -> List[int]:
    """token indices whose character span overlaps [a, b)"""
    return [i for i, (s, e) in enumerate(offsets) if e > a and s < b and i < limit]


@torch.no_grad()
def extract(lm, tok, exs: List[dict], cfg: LLMConfig, pad_id: int, device, bs: int = 4):
    """Returns a list of per-example feature dicts (CPU float16) and the number skipped."""
    out, skipped = [], 0
    S = cfg.seg_len
    for i0 in range(0, len(exs), bs):
        chunk = exs[i0:i0 + bs]
        items = [tokenize_example(tok, ex, cfg) for ex in chunk]
        b = collate(items, cfg, pad_id)
        B, n1, _ = b["ids"].shape
        flat = b["ids"].reshape(B * n1, S).to(device)
        pos = torch.arange(S, device=device).unsqueeze(0).expand(B * n1, S)
        hs = lm.model(input_ids=flat, position_ids=pos, output_hidden_states=True, use_cache=False).hidden_states
        H = torch.stack([hs[l].reshape(B, n1 * S, -1) for l in LAYERS], dim=2)          # (B, T, nL, d)
        n = n1 - 1
        for j, (it, ex) in enumerate(zip(items, chunk)):
            try:
                f = positions_and_features(tok, it, ex, H[j], n, S)
            except (AssertionError, ValueError, IndexError):
                f = None
            if f is None:
                skipped += 1
            else:
                out.append(f)
    return out, skipped


def positions_and_features(tok, it, ex, Hj, n, S):
    ctx_text, q_text = ex["context"], ex["question"]
    device_name = [d for d in DEVICES if d in q_text][0]
    q_loc = re.search(r" at (.+?)[?.]$", q_text).group(1)
    a = ex["annot"]
    values = [ex["answer"]] + list(a.get("distractor_values", []))
    enc = tok(ctx_text, add_special_tokens=False, return_offsets_mapping=True)
    L = len(it["ctx"])
    assert enc["input_ids"][:L] == it["ctx"]
    off = enc["offset_mapping"]
    base = n * S - L
    write_pos, ent_pos = [], []
    for v in values:
        c = ctx_text.find(v)
        assert c >= 0
        vt = tok_span(off, c, c + len(v), L)
        assert vt and vt[0] >= 1
        write_pos.append(base + vt[0] - 1)
        # entity of THIS fact: "<device> at <Location N>" inside the sentence that holds the value
        s0 = max(ctx_text.rfind(". ", 0, c), ctx_text.rfind("\n", 0, c)) + 1
        s1 = ctx_text.find(".", c + len(v))
        s1 = len(ctx_text) if s1 < 0 else s1
        m = re.search(re.escape(device_name) + r" at ([A-Z][a-z]+ \d+)", ctx_text[s0:s1 + 1])
        assert m is not None
        et = tok_span(off, s0 + m.start(), s0 + m.end(), L)
        assert et
        ent_pos.append([base + t for t in et])
    p_text = PROMPT.format(q=q_text)
    penc = tok(p_text, add_special_tokens=False, return_offsets_mapping=True)
    assert penc["input_ids"] == it["prompt"]
    plen = len(it["prompt"])
    qe = p_text.find(f"{device_name} at {q_loc}")
    assert qe >= 0
    q_ent = tok_span(penc["offset_mapping"], qe, qe + len(f"{device_name} at {q_loc}"), plen)
    assert q_ent
    qb = n * S
    feats_q = {"answer": Hj[qb + plen - 1], "entity": Hj[[qb + t for t in q_ent]].mean(0), "span": Hj[qb:qb + plen].mean(0)}
    feats_d = {"write": Hj[write_pos], "entity": torch.stack([Hj[p].mean(0) for p in ent_pos])}
    return {"q": {k: v.float().cpu().half() for k, v in feats_q.items()},          # (nL, d)
            "d": {k: v.float().cpu().half() for k, v in feats_d.items()},          # (k, nL, d)
            "k": len(values)}


class Probe(nn.Module):
    def __init__(self, d: int, r: int = 128, shared: bool = False):
        super().__init__()
        self.A = nn.Linear(d, r, bias=False)
        self.B = self.A if shared else nn.Linear(d, r, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.1)))

    def forward(self, q, D, mask):
        """q (N, d), D (N, K, d), mask (N, K) → logits (N, K)"""
        a = F.normalize(self.A(q), dim=-1)
        bvec = F.normalize(self.B(D), dim=-1)
        logits = (bvec @ a.unsqueeze(-1)).squeeze(-1) / self.log_tau.exp()
        return logits.masked_fill(~mask, -1e9)


def stack(feats, side_q, side_d, li, kmax, stats=None):
    N = len(feats)
    d = feats[0]["q"][side_q].shape[-1]
    Q = torch.stack([f["q"][side_q][li] for f in feats]).float()
    D = torch.zeros(N, kmax, d)
    M = torch.zeros(N, kmax, dtype=torch.bool)
    for i, f in enumerate(feats):
        k = f["k"]
        D[i, :k] = f["d"][side_d][:, li].float()
        M[i, :k] = True
    if stats is None:                                    # z-score with TRAINING statistics (Qwen has huge outlier dims)
        allv = torch.cat([Q, D[M]], 0)
        stats = (allv.mean(0), allv.std(0) + 1e-4)
    mu, sd = stats
    return (Q - mu) / sd, (D - mu) / sd * M.unsqueeze(-1), M, stats


def top1(logits, mask):
    """Tie-aware top-1. The asked fact is stored at index 0, so a plain argmax would credit every exact tie to it (this happened at
    layer 0, where identical tokens give identical vectors). A tie among m facts counts as 1/m (= random tie-breaking)."""
    mx = logits.max(-1, keepdim=True).values
    ties = (logits >= mx - 1e-6) & mask
    return ties[:, 0].float() / ties.sum(-1).clamp(min=1).float()


def run_keyprobe(cfg: LLMConfig, train_path: str, test_paths: Dict[str, str], n_train: int = 2000, epochs: int = 40,
                 device: str = "cuda") -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    try:
        lm = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=torch.bfloat16).to(device).eval()
    except TypeError:
        lm = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16).to(device).eval()

    tr_ex = load_examples(train_path, limit=n_train)
    feats = {"train": extract(lm, tok, tr_ex, cfg, pad_id, device)}
    for name, p in test_paths.items():
        feats[name] = extract(lm, tok, load_examples(p), cfg, pad_id, device)
    kmax = max(f["k"] for fs, _ in feats.values() for f in fs)
    report = {"n": {k: len(v[0]) for k, v in feats.items()}, "skipped": {k: v[1] for k, v in feats.items()},
              "layers": list(LAYERS), "results": {}}
    tr = feats["train"][0]
    for li, layer in enumerate(LAYERS):
        for qv in Q_VARIANTS:
            for dv in D_VARIANTS:
                Qtr, Dtr, Mtr, stats = stack(tr, qv, dv, li, kmax)
                cell = {}
                # raw cosine, no probe (z-scored with training statistics)
                for name in test_paths:
                    Q, D, M, _ = stack(feats[name][0], qv, dv, li, kmax, stats)
                    raw = F.cosine_similarity(Q.unsqueeze(1), D, dim=-1).masked_fill(~M, -1e9)
                    hit = top1(raw, M)
                    cell[f"{name}/raw"] = hit.mean().item()
                for shared in (False, True):
                    torch.manual_seed(0)
                    probe = Probe(Qtr.shape[-1], shared=shared).to(device)
                    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)
                    Qd, Dd, Md = Qtr.to(device), Dtr.to(device), Mtr.to(device)
                    tgt = torch.zeros(len(Qd), dtype=torch.long, device=device)
                    for ep in range(epochs):
                        perm = torch.randperm(len(Qd), device=device)
                        for s in range(0, len(Qd), 128):
                            ix = perm[s:s + 128]
                            loss = F.cross_entropy(probe(Qd[ix], Dd[ix], Md[ix]), tgt[ix])
                            opt.zero_grad()
                            loss.backward()
                            opt.step()
                    tag = "shared" if shared else "separate"
                    with torch.no_grad():
                        cell[f"train/{tag}"] = top1(probe(Qd, Dd, Md), Md).mean().item()
                        for name in test_paths:
                            Q, D, M, _ = stack(feats[name][0], qv, dv, li, kmax, stats)
                            hit = top1(probe(Q.to(device), D.to(device), M.to(device)), M.to(device)).cpu()
                            ks = torch.tensor([f["k"] for f in feats[name][0]])
                            cell[f"{name}/{tag}"] = hit.mean().item()
                            cell[f"{name}/{tag}/ci95"] = wilson(int(round(hit.sum().item())), len(hit))
                            cell[f"{name}/{tag}/by_k"] = {int(k): hit[ks == k].mean().item() for k in ks.unique()}
                for name in test_paths:
                    ks = [f["k"] for f in feats[name][0]]
                    cell[f"{name}/chance"] = sum(1 / k for k in ks) / len(ks)
                report["results"][f"L{layer}|q={qv}|d={dv}"] = cell
    return report


def summarize_keyprobe(rep: dict) -> str:
    lines = [f"key probe: n = {rep['n']}  skipped = {rep['skipped']}",
             f"{'layer|question|document':28s} {'chance':>6s} | {'HELD-OUT separate':>17s} {'shared':>7s} {'raw':>5s} | {'seen sep':>8s} {'train sep':>9s}"]
    for key, c in rep["results"].items():
        lines.append(f"{key:28s} {c['heldout/chance']:6.2f} | {c['heldout/separate']:9.2f} "
                     f"[{c['heldout/separate/ci95'][0]:.2f},{c['heldout/separate/ci95'][1]:.2f}] {c['heldout/shared']:7.2f} "
                     f"{c['heldout/raw']:5.2f} | {c['seen/separate']:8.2f} {c['train/separate']:9.2f}")
    return "\n".join(lines)
