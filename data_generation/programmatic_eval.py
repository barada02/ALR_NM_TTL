"""
Programmatic evaluation sets with EXACT ground truth (Stage 0). No API, no model: entity slots + value pools + templates,
seeded and reproducible. Records use the existing `long_context` schema (so llm_data.to_example reads them unchanged) plus an
`annot` dict that lets a failure say WHICH mechanism broke.

Categories (default sizes in parentheses):
  single             (300)  one fact, nothing else about that entity — control; fact position balanced early/middle/late
  overwrite          (300)  the fact is stated, then restated with a NEW value later (2 or 3 versions, short/medium/long gap);
                            the question asks for the current value; annot has the old values
  colliding          (300)  k ∈ {2,4,8} facts of the SAME device type and attribute at different locations, different values;
                            the question targets one; annot has the distractor values
  colliding_overwrite(200)  k = 4 colliding facts AND the target is overwritten (hardest)
  haystack           (400)  target document wrapped in n_docs ∈ {2,4,8,16} other documents that contain a COLLIDING fact
                            (same device + attribute, other location, other value) — interference, not just longer text
  absent             (200)  the asked-for fact is not in the document (answer "unknown"): measures confident hallucination

Prose is template text on purpose: a style nobody trained on (the training data came from Gemini), so these sets also act as a
held-out style. They test the mechanism (recall under overwrite / interference), not language naturalness.

Usage:  python programmatic_eval.py --out_dir miniTitan_eval            # writes eval_<category>.jsonl + manifest.json
        python programmatic_eval.py --check                             # generate a small set and verify every invariant
"""

import argparse
import json
import os
import random
import re
import string
from typing import Dict, List, Tuple

VERSION = "programmatic_eval_v1"

DEVICES = ["flow meter", "pressure sensor", "compressor", "cooling pump", "control valve", "generator", "transformer", "gearbox"]
LOCS = ["Deck 3", "Deck 4", "Bay 7", "Bay 2", "Tower B", "Annex 2", "Level 5", "Zone D", "Hall 4", "Dock 9", "Wing A", "Unit 12",
        "Platform 6", "Block C", "Station 8", "Sector 5", "Gallery 1", "Pier 11", "Hangar 3", "Vault 6"]
FIRST = ["Elias", "Marta", "Jonas", "Priya", "Tomas", "Ingrid", "Rafael", "Noor", "Lukas", "Amara", "Felix", "Sana", "Mateo", "Hana"]
LAST = ["Vogt", "Okafor", "Lindqvist", "Banerjee", "Moreau", "Castillo", "Haddad", "Novak", "Sorensen", "Adeyemi", "Kovacs", "Tanaka"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
LET = string.ascii_uppercase


def v_serial(r): return f"{r.choice(LET)}{r.choice(LET)}-{r.randint(1000, 9999)}-{r.choice(LET)}"
def v_cost(r): return f"€{r.randint(1, 9)},{r.randint(100, 999)},{r.randint(100, 999)}"
def v_date(r): return f"{r.choice(MONTHS)} {r.randint(1, 28)}, {r.randint(2018, 2025)}"
def v_port(r): return f"{r.randint(1024, 65000)}"
def v_eng(r): return f"{r.choice(FIRST)} {r.choice(LAST)}"


ATTRS = {"serial": ("serial number", v_serial), "cost": ("replacement cost", v_cost), "date": ("last inspection date", v_date),
         "port": ("service port", v_port), "engineer": ("responsible engineer", v_eng)}
FACT_T = ["The {noun} of the {ent} is {v}.", "Records state that the {noun} of the {ent} is {v}.", "For the {ent}, the {noun} is {v}."]
RESTATE_T = ["Following a recent update, the {noun} of the {ent} is now {v}.", "A correction was filed: the {noun} of the {ent} has changed to {v}.",
             "After the changeover, the {ent} now has the {noun} {v}."]

TEAMS = ["operations team", "audit team", "safety committee", "maintenance crew", "planning group", "logistics office", "review board"]
DOCS = ["maintenance schedule", "incident report", "budget proposal", "training plan", "logistics summary", "shift roster", "supplier contract"]
PERIODS = ["weekly", "monthly", "quarterly", "morning", "evening", "annual", "emergency"]
VERBS = ["reviewed", "updated", "archived", "discussed", "circulated", "approved", "postponed"]
ADJ = ["routine", "detailed", "brief", "thorough", "preliminary", "revised"]
FILLER_T = [
    "The {team} {verb} the {doc} during the {period} meeting.",
    "A {adj} walkthrough covered the {doc} and several open questions.",
    "Technicians walked past the {dev} at {loc} without noting any irregularities.",
    "The {team} asked for a {adj} summary of the {doc} before the next {period} cycle.",
    "Weather delays pushed the {period} inspection of {loc} back by a day.",
    "Several staff members {verb} the {doc} and sent comments to the {team}.",
    "The {dev} at {loc} was mentioned in passing during the {period} briefing.",
    "No changes were requested to the {doc} after the {adj} review.",
    "The {team} noted that the {period} schedule for {loc} remained unchanged.",
    "A {adj} note about the {doc} was added to the shared folder.",
    "Visitors to {loc} were reminded to follow the usual safety procedures.",
    "The {team} {verb} the {doc} and thanked everyone for their patience.",
]


def filler(r: random.Random) -> str:
    return r.choice(FILLER_T).format(team=r.choice(TEAMS), doc=r.choice(DOCS), period=r.choice(PERIODS), verb=r.choice(VERBS),
                                     adj=r.choice(ADJ), dev=r.choice(DEVICES), loc=r.choice(LOCS))


def build_doc(r: random.Random, n_words: int, facts: List[Tuple[float, str]]) -> Tuple[str, List[float]]:
    """facts: (target position fraction in [0,1], sentence). Returns (text, realised word-position fraction of each fact, input order)."""
    sents, words = [], 0
    while words < n_words:
        s = filler(r)
        sents.append(("f", s))
        words += len(s.split())
    n = len(sents)
    order = sorted(range(len(facts)), key=lambda i: facts[i][0])
    for rank, i in enumerate(order):                         # insert in position order; each insert shifts later indices by one
        idx = min(int(facts[i][0] * n), n) + rank
        sents.insert(idx, ("fact", facts[i][1], i))
    total = sum(len(x[1].split()) for x in sents)
    pos, acc = [0.0] * len(facts), 0
    for x in sents:
        if x[0] == "fact":
            pos[x[2]] = acc / total
        acc += len(x[1].split())
    paras = [" ".join(x[1] for x in sents[k:k + 6]) for k in range(0, len(sents), 6)]
    return "\n\n".join(paras), pos


def bucket(frac: float) -> str:
    return "early" if frac < 1 / 3 else "middle" if frac < 2 / 3 else "late"


class Gen:
    def __init__(self, seed: int):
        self.r = random.Random(seed)

    def value(self, attr: str, used: set) -> str:
        while True:
            v = ATTRS[attr][1](self.r)
            if v not in used:
                used.add(v)
                return v

    def ent(self, device: str, loc: str) -> str:
        return f"{device} at {loc}"

    def fact_sentence(self, attr, device, loc, v, restate=False):
        t = self.r.choice(RESTATE_T if restate else FACT_T)
        return t.format(noun=ATTRS[attr][0], ent=self.ent(device, loc), v=v)

    def question(self, attr, device, loc, current=False):
        return f"What is the {'current ' if current else ''}{ATTRS[attr][0]} of the {self.ent(device, loc)}?"

    def record(self, text, q, a, frac, cat, **annot):
        annot.update(category=category_name(cat), fact_pos_frac=round(frac, 3), doc_words=len(text.split()))
        return {"_type": "long_context", "context": text, "question": q, "answer": a, "fact_position": bucket(frac),
                "distractor_count": annot.get("n_colliding", 0), "annot": annot}

    # ── categories ──
    def single(self):
        r = self.r
        attr, device, loc = r.choice(list(ATTRS)), r.choice(DEVICES), r.choice(LOCS)
        used = set()
        v = self.value(attr, used)
        frac = r.uniform(0.03, 0.97)
        text, pos = build_doc(r, r.randint(850, 1150), [(frac, self.fact_sentence(attr, device, loc, v))])
        return self.record(text, self.question(attr, device, loc), v, pos[0], "single", attr=attr, overwritten=False, n_versions=1,
                           n_colliding=0, absent=False, old_values=[], distractor_values=[], all_values=[v])

    def overwrite(self, n_colliding=0):
        r = self.r
        attr, device = r.choice(list(ATTRS)), r.choice(DEVICES)
        locs = r.sample(LOCS, 1 + n_colliding)
        used = set()
        n_ver = 2 if r.random() < 0.7 else 3
        gap_kind = r.choice(["short", "medium", "long"])
        gap = {"short": (0.05, 0.15), "medium": (0.2, 0.4), "long": (0.5, 0.75)}[gap_kind]
        vals = [self.value(attr, used) for _ in range(n_ver)]
        total_span = (n_ver - 1) * r.uniform(*gap)
        p0 = r.uniform(0.03, max(0.04, 0.95 - total_span))
        facts, targets = [], []
        for k in range(n_ver):
            pk = min(p0 + k * (total_span / (n_ver - 1)), 0.97)
            facts.append((pk, self.fact_sentence(attr, device, locs[0], vals[k], restate=(k > 0))))
        dist = []
        for j in range(n_colliding):
            dv = self.value(attr, used)
            dist.append(dv)
            facts.append((r.uniform(0.03, 0.97), self.fact_sentence(attr, device, locs[1 + j], dv)))
        text, pos = build_doc(r, r.randint(850, 1150), facts)
        gap_words = int((pos[n_ver - 1] - pos[0]) * len(text.split()))
        cat = "colliding_overwrite" if n_colliding else "overwrite"
        return self.record(text, self.question(attr, device, locs[0], current=True), vals[-1], pos[n_ver - 1], cat, attr=attr, overwritten=True,
                           n_versions=n_ver, n_colliding=n_colliding, absent=False, old_values=vals[:-1], distractor_values=dist,
                           all_values=vals + dist, gap_words=gap_words, gap_kind=gap_kind)

    def colliding(self, k):
        r = self.r
        attr, device = r.choice(list(ATTRS)), r.choice(DEVICES)
        locs = r.sample(LOCS, k)
        used = set()
        vals = [self.value(attr, used) for _ in range(k)]
        fracs = [r.uniform(0.03, 0.97) for _ in range(k)]
        t = r.randrange(k)                                   # which one the question targets (uniform → position balanced)
        facts = [(fracs[i], self.fact_sentence(attr, device, locs[i], vals[i])) for i in range(k)]
        text, pos = build_doc(r, r.randint(850, 1150), facts)
        return self.record(text, self.question(attr, device, locs[t]), vals[t], pos[t], "colliding", attr=attr, overwritten=False, n_versions=1,
                           n_colliding=k - 1, absent=False, old_values=[], distractor_values=[v for i, v in enumerate(vals) if i != t], all_values=vals)

    def haystack(self, n_docs):
        r = self.r
        attr, device = r.choice(list(ATTRS)), r.choice(DEVICES)
        locs = r.sample(LOCS, min(n_docs, len(LOCS)))
        if n_docs > len(LOCS):
            locs = [r.choice(LOCS) for _ in range(n_docs)]       # locations may repeat across docs; values never do
        used = set()
        vals = [self.value(attr, used) for _ in range(n_docs)]
        t = r.randrange(n_docs)
        docs, fr_t = [], 0.0
        for i in range(n_docs):
            text, pos = build_doc(r, 350, [(r.uniform(0.05, 0.95), self.fact_sentence(attr, device, locs[i], vals[i]))])
            docs.append(text)
            if i == t:
                fr_t = (sum(len(d.split()) for d in docs[:i]) + pos[0] * len(text.split()))
        full = "\n\n".join(docs)
        frac = fr_t / max(len(full.split()), 1)
        q_loc = locs[t]
        same_loc = sum(1 for l in locs if l == q_loc)
        assert same_loc == 1 or n_docs > len(LOCS), "ambiguous question"
        return self.record(full, self.question(attr, device, q_loc), vals[t], frac, "haystack", attr=attr, overwritten=False, n_versions=1,
                           n_colliding=n_docs - 1, n_docs=n_docs, absent=False, old_values=[],
                           distractor_values=[v for i, v in enumerate(vals) if i != t], all_values=vals)

    def absent(self):
        r = self.r
        attr, device = r.choice(list(ATTRS)), r.choice(DEVICES)
        locs = r.sample(LOCS, 3)
        used = set()
        k = r.randint(1, 2)
        vals = [self.value(attr, used) for _ in range(k)]
        facts = [(r.uniform(0.05, 0.95), self.fact_sentence(attr, device, locs[i], vals[i])) for i in range(k)]
        text, pos = build_doc(r, r.randint(850, 1150), facts)
        # ask about a location that has NO fact (but is a plausible, similar entity)
        return self.record(text, self.question(attr, device, locs[2]), "unknown", 0.5, "absent", attr=attr, overwritten=False, n_versions=0,
                           n_colliding=k, absent=True, old_values=[], distractor_values=vals, all_values=vals)


def category_name(c):
    return c


def generate(seed: int, sizes: Dict[str, int]) -> Dict[str, List[dict]]:
    g = Gen(seed)
    out = {c: [] for c in ("single", "overwrite", "colliding", "colliding_overwrite", "haystack", "absent")}
    for _ in range(sizes["single"]):
        out["single"].append(g.single())
    for _ in range(sizes["overwrite"]):
        out["overwrite"].append(g.overwrite())
    for i in range(sizes["colliding"]):
        out["colliding"].append(g.colliding([2, 4, 8][i % 3]))
    for _ in range(sizes["colliding_overwrite"]):
        out["colliding_overwrite"].append(g.overwrite(n_colliding=4))
    for i in range(sizes["haystack"]):
        out["haystack"].append(g.haystack([2, 4, 8, 16][i % 4]))
    for _ in range(sizes["absent"]):
        out["absent"].append(g.absent())
    return out


def verify(rec: dict):
    """Every invariant the eval relies on. Raises AssertionError with the reason."""
    a, ctx, an = rec["answer"], rec["context"], rec["annot"]
    q = rec["question"]
    if an["absent"]:
        assert a == "unknown"
        loc = re.search(r" at (.+?)\?", q).group(1)
        dev = [d for d in DEVICES if d in q][0]
        # no sentence states a value for the asked entity
        for s in re.split(r"(?<=\.)\s+", ctx):
            assert not (f"{dev} at {loc}" in s and any(v in s for v in an["all_values"])), "absent fact is present"
        return
    assert ctx.count(a) == 1, f"answer must appear exactly once ({ctx.count(a)})"
    vals = an["old_values"] + [a] + an["distractor_values"]
    assert len(set(vals)) == len(vals), "values must be distinct"
    assert all(v in ctx for v in vals), "every stated value must be in the document"
    assert (rec["annot"]["n_versions"] == 1) == (not an["old_values"])
    if an["old_values"]:
        pos = [ctx.find(v) for v in an["old_values"]] + [ctx.find(a)]
        assert pos == sorted(pos) and len(set(pos)) == len(pos), "versions must appear in order: old first, current last"
    loc = re.search(r" at (.+?)\?", q).group(1)
    sent = [s for s in re.split(r"(?<=\.)\s+", ctx) if a in s][0]
    assert f" at {loc}" in sent or f"{loc}" in sent, "the answer sentence must be about the asked entity"
    for dv in an["distractor_values"]:
        s2 = [s for s in re.split(r"(?<=\.)\s+", ctx) if dv in s][0]
        assert f"at {loc}" not in s2, "a distractor value is attached to the asked entity"
    assert an["fact_pos_frac"] >= 0 and rec["fact_position"] in ("early", "middle", "late")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="miniTitan_eval")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the default category sizes")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    base = {"single": 300, "overwrite": 300, "colliding": 300, "colliding_overwrite": 200, "haystack": 400, "absent": 200}
    sizes = {k: (24 if a.check else max(4, int(v * a.scale))) for k, v in base.items()}
    data = generate(a.seed, sizes)
    n_ok = 0
    for cat, recs in data.items():
        for rec in recs:
            verify(rec)
            n_ok += 1
    print(f"verified {n_ok} records across {len(data)} categories (all invariants hold)")
    if a.check:
        pos = {"early": 0, "middle": 0, "late": 0}
        for rec in data["single"] + data["overwrite"]:
            pos[rec["fact_position"]] += 1
        print("fact position balance (single+overwrite):", pos)
        for cat in data:
            r0 = data[cat][0]
            print(f"\n[{cat}] words={r0['annot']['doc_words']}  Q: {r0['question']}  A: {r0['answer']}  annot: "
                  f"{ {k: v for k, v in r0['annot'].items() if k in ('n_versions', 'n_colliding', 'n_docs', 'gap_kind', 'old_values', 'fact_pos_frac')} }")
        return
    os.makedirs(a.out_dir, exist_ok=True)
    manifest = {"version": VERSION, "seed": a.seed, "sizes": sizes, "files": {}}
    for cat, recs in data.items():
        path = os.path.join(a.out_dir, f"eval_{cat}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        manifest["files"][cat] = {"path": os.path.basename(path), "n": len(recs),
                                  "avg_words": sum(r["annot"]["doc_words"] for r in recs) // len(recs)}
    with open(os.path.join(a.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest["files"], indent=1))


if __name__ == "__main__":
    main()
