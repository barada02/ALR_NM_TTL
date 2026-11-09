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

VERSION = "programmatic_eval_v2"   # v2 adds train/heldout pools; pools="all" reproduces v1 byte for byte

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

# Held-out machinery (v2). pools="all" is the v1 behaviour (all 8 devices, 20 locations, 3 fact / 3 restate templates, ONE question
# template, no extra rng draws). "train" / "heldout" use DISJOINT entity types, locations and surface templates, so ranking on
# held-out sets cannot be won by template-fitting or by memorising entity names.
LOCS2 = ["Annex 7", "Ridge 4", "Basin 2", "Cell 9", "Depot 5", "Field 8", "Gate 3", "Harbor 12", "Island 6", "Junction 4", "Keep 2", "Lab 11",
         "Mill 7", "Nest 3", "Outpost 9", "Plant 5", "Quay 8", "Rig 14", "Silo 6", "Terminal 2", "Unit 31", "Vent 4", "Wharf 10", "Yard 13",
         "Arch 5", "Bluff 3", "Camp 8", "Dome 2", "Edge 9", "Fort 7", "Grid 12", "Hub 6", "Inlet 4", "Jetty 11", "Kiln 3", "Lock 8",
         "Mast 5", "Node 10", "Orbit 2", "Pad 9"]
FACT_T_HELD = ["According to the asset register, the {noun} of the {ent} reads {v}.", "The register lists {v} as the {noun} of the {ent}."]
RESTATE_T_HELD = ["The {noun} of the {ent} was revised and now reads {v}.", "Per the latest amendment, {v} is the new {noun} of the {ent}."]
Q_T = {"all": ["What is the {noun} of the {ent}?"],
       "train": ["What is the {noun} of the {ent}?", "Which {noun} is listed for the {ent}?"],
       "heldout": ["Can you tell me the {noun} of the {ent}?", "Please state the {noun} of the {ent}."]}
POOLS = {
    "all": dict(devices=DEVICES, locs=LOCS, fact=FACT_T, restate=RESTATE_T),
    "train": dict(devices=[DEVICES[i] for i in (0, 2, 4, 6)], locs=LOCS2[:24], fact=FACT_T, restate=RESTATE_T),
    "heldout": dict(devices=[DEVICES[i] for i in (1, 3, 5, 7)], locs=LOCS2[24:], fact=FACT_T_HELD, restate=RESTATE_T_HELD),
}

PRONOUN_EVENTS = ["was moved last week", "was serviced on Monday", "was inspected by the night shift", "was recalibrated after the audit",
                  "was relabelled during the review"]

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
    def __init__(self, seed: int, pools: str = "all"):
        self.r = random.Random(seed)
        self.pools = pools
        P = POOLS[pools]
        self.devices, self.locs, self.fact_t, self.restate_t = P["devices"], P["locs"], P["fact"], P["restate"]

    def value(self, attr: str, used: set) -> str:
        while True:
            v = ATTRS[attr][1](self.r)
            if v not in used:
                used.add(v)
                return v

    def ent(self, device: str, loc: str) -> str:
        return f"{device} at {loc}"

    def fact_sentence(self, attr, device, loc, v, restate=False):
        t = self.r.choice(self.restate_t if restate else self.fact_t)
        return t.format(noun=ATTRS[attr][0], ent=self.ent(device, loc), v=v)

    def question(self, attr, device, loc, current=False):
        qs = Q_T[self.pools]
        t = qs[0] if len(qs) == 1 else self.r.choice(qs)          # no rng draw for pools="all" (keeps v1 reproducible)
        return t.format(noun=("current " if current else "") + ATTRS[attr][0], ent=self.ent(device, loc))

    def record(self, text, q, a, frac, cat, **annot):
        annot.update(category=category_name(cat), fact_pos_frac=round(frac, 3), doc_words=len(text.split()))
        return {"_type": "long_context", "context": text, "question": q, "answer": a, "fact_position": bucket(frac),
                "distractor_count": annot.get("n_colliding", 0), "annot": annot}

    # ── categories ──
    def single(self):
        r = self.r
        attr, device, loc = r.choice(list(ATTRS)), r.choice(self.devices), r.choice(self.locs)
        used = set()
        v = self.value(attr, used)
        frac = r.uniform(0.03, 0.97)
        text, pos = build_doc(r, r.randint(850, 1150), [(frac, self.fact_sentence(attr, device, loc, v))])
        return self.record(text, self.question(attr, device, loc), v, pos[0], "single", attr=attr, overwritten=False, n_versions=1,
                           n_colliding=0, absent=False, old_values=[], distractor_values=[], all_values=[v])

    def overwrite(self, n_colliding=0):
        r = self.r
        attr, device = r.choice(list(ATTRS)), r.choice(self.devices)
        locs = r.sample(self.locs, 1 + n_colliding)
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
        attr, device = r.choice(list(ATTRS)), r.choice(self.devices)
        locs = r.sample(self.locs, k)
        used = set()
        vals = [self.value(attr, used) for _ in range(k)]
        fracs = [r.uniform(0.03, 0.97) for _ in range(k)]
        t = r.randrange(k)                                   # which one the question targets (uniform → position balanced)
        facts = [(fracs[i], self.fact_sentence(attr, device, locs[i], vals[i])) for i in range(k)]
        text, pos = build_doc(r, r.randint(850, 1150), facts)
        return self.record(text, self.question(attr, device, locs[t]), vals[t], pos[t], "colliding", attr=attr, overwritten=False, n_versions=1,
                           n_colliding=k - 1, absent=False, old_values=[], distractor_values=[v for i, v in enumerate(vals) if i != t], all_values=vals)

    def colliding_multi(self, k, m):
        """ONE colliding document, m (2-4) separate examples that each query a different entity — multiplies the binding signal
        per document (a single question gives one binding gradient per ~1000 words)."""
        r = self.r
        attr, device = r.choice(list(ATTRS)), r.choice(self.devices)
        locs = r.sample(self.locs, k)
        used = set()
        vals = [self.value(attr, used) for _ in range(k)]
        facts = [(r.uniform(0.03, 0.97), self.fact_sentence(attr, device, locs[i], vals[i])) for i in range(k)]
        text, pos = build_doc(r, r.randint(850, 1150), facts)
        doc_id = r.getrandbits(40)
        out = []
        for t in r.sample(range(k), m):
            out.append(self.record(text, self.question(attr, device, locs[t]), vals[t], pos[t], "colliding", attr=attr, overwritten=False,
                                   n_versions=1, n_colliding=k - 1, absent=False, old_values=[],
                                   distractor_values=[v for i, v in enumerate(vals) if i != t], all_values=vals, doc_id=doc_id, n_questions=m))
        return out

    def overwrite_cue(self, variant: str):
        """Cue probe (two statements of ONE entity). variant:
             orig      neutral statement first, cue-bearing restatement ("now", "changed to") LATER   (as in training)
             cuefree   both statements neutral (no now / new / changed); the LATER one is current by convention
             cuefirst  the cue-bearing restatement comes FIRST, a neutral statement LATER
           gold = the LATER value in every variant (for bookkeeping); annot records which value carries the cue, so the eval
           can report 'prefers later' and 'prefers the cue' separately."""
        r = self.r
        attr, device, loc = r.choice(list(ATTRS)), r.choice(self.devices), r.choice(self.locs)
        used = set()
        v_early, v_late = self.value(attr, used), self.value(attr, used)
        gap = r.choice([(0.05, 0.15), (0.2, 0.4), (0.5, 0.75)])
        span = r.uniform(*gap)
        p0 = r.uniform(0.03, max(0.04, 0.95 - span))
        cue_early = variant == "cuefirst"
        cue_late = variant == "orig"
        s_early = self.fact_sentence(attr, device, loc, v_early, restate=cue_early)
        s_late = self.fact_sentence(attr, device, loc, v_late, restate=cue_late)
        text, pos = build_doc(r, r.randint(850, 1150), [(p0, s_early), (min(p0 + span, 0.97), s_late)])
        cue_value = v_early if cue_early else (v_late if cue_late else None)
        return self.record(text, self.question(attr, device, loc, current=True), v_late, pos[1], "overwrite_cue", attr=attr,
                           overwritten=True, n_versions=2, n_colliding=0, absent=False, old_values=[v_early], distractor_values=[],
                           all_values=[v_early, v_late], variant=variant, cue_value=cue_value, later_value=v_late)

    def paraphrase(self, kind: str, k: int):
        """P1 / P2 sets of DESIGN_ENTITY_KEYS.md §8 (colliding document, k same-type facts, question targets one).
          reorder  the fact and the question name the entity in DIFFERENT forms ("flow meter at Ridge 4" vs "Ridge 4 flow meter" /
                   "Ridge 4's flow meter"), in either direction; all facts of the document use the same form
          partial  facts use "flow meter at Ridge 4", the question drops the device word: "the unit at Ridge 4"
          pronoun  the entity is in the PREVIOUS sentence: "The flow meter at Ridge 4 was moved last week. Its serial number is X." """
        r = self.r
        attr, device = r.choice(list(ATTRS)), r.choice(self.devices)
        noun = ATTRS[attr][0]
        locs = r.sample(self.locs, k)
        used = set()
        vals = [self.value(attr, used) for _ in range(k)]
        t = r.randrange(k)
        canon = lambda l: f"{device} at {l}"
        alt = r.choice([lambda l: f"{l} {device}", lambda l: f"{l}'s {device}"])
        if kind == "reorder":
            fact_form, q_form = (canon, alt) if r.random() < 0.5 else (alt, canon)
        elif kind == "partial":
            fact_form, q_form = canon, (lambda l: f"unit at {l}")
        else:
            fact_form, q_form = canon, canon
        facts = []
        for i in range(k):
            if kind == "pronoun":
                sent = f"The {canon(locs[i])} {r.choice(PRONOUN_EVENTS)}. Its {noun} is {vals[i]}."
            else:
                sent = f"The {noun} of the {fact_form(locs[i])} is {vals[i]}."
            facts.append((r.uniform(0.03, 0.97), sent))
        text, pos = build_doc(r, r.randint(850, 1150), facts)
        q = f"What is the {noun} of the {q_form(locs[t])}?"
        cat = {"reorder": "p1_reorder", "partial": "p1_partial", "pronoun": "p2_pronoun"}[kind]
        return self.record(text, q, vals[t], pos[t], cat, attr=attr, overwritten=False, n_versions=1, n_colliding=k - 1, absent=False,
                           old_values=[], distractor_values=[v for i, v in enumerate(vals) if i != t], all_values=vals,
                           fact_mention=fact_form(locs[t]), question_mention=q_form(locs[t]), device=device, location=locs[t])

    def haystack(self, n_docs):
        r = self.r
        attr, device = r.choice(list(ATTRS)), r.choice(self.devices)
        locs = r.sample(self.locs, min(n_docs, len(self.locs)))
        if n_docs > len(self.locs):
            locs = [r.choice(self.locs) for _ in range(n_docs)]       # locations may repeat across docs; values never do
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
        assert same_loc == 1 or n_docs > len(self.locs), "ambiguous question"
        return self.record(full, self.question(attr, device, q_loc), vals[t], frac, "haystack", attr=attr, overwritten=False, n_versions=1,
                           n_colliding=n_docs - 1, n_docs=n_docs, absent=False, old_values=[],
                           distractor_values=[v for i, v in enumerate(vals) if i != t], all_values=vals)

    def absent(self):
        r = self.r
        attr, device = r.choice(list(ATTRS)), r.choice(self.devices)
        locs = r.sample(self.locs, 3)
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


def generate(seed: int, sizes: Dict[str, int], pools: str = "all") -> Dict[str, List[dict]]:
    g = Gen(seed, pools)
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
        loc = re.search(r" at (.+?)[?.]$", q).group(1)
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
    loc = re.search(r" at (.+?)[?.]$", q).group(1)
    sent = [s for s in re.split(r"(?<=\.)\s+", ctx) if a in s][0]
    assert f" at {loc}" in sent or f"{loc}" in sent, "the answer sentence must be about the asked entity"
    for dv in an["distractor_values"]:
        s2 = [s for s in re.split(r"(?<=\.)\s+", ctx) if dv in s][0]
        assert f"at {loc}" not in s2, "a distractor value is attached to the asked entity"
    assert an["fact_pos_frac"] >= 0 and rec["fact_position"] in ("early", "middle", "late")


def verify_para(rec: dict):
    """Invariants of the P1 / P2 sets."""
    a, ctx, an = rec["answer"], rec["context"], rec["annot"]
    assert ctx.count(a) == 1, "answer must appear exactly once"
    vals = [a] + an["distractor_values"]
    assert len(set(vals)) == len(vals) and all(v in ctx for v in vals)
    sents = re.split(r"(?<=[.])\s+", ctx)            # any whitespace, incl. paragraph breaks
    s_ans = [x for x in sents if a in x][0]
    if an["category"] == "p2_pronoun":
        assert s_ans.startswith("Its ") and an["location"] not in s_ans and an["device"] not in s_ans, "pronoun sentence must not name the entity"
        prev = sents[sents.index(s_ans) - 1]
        assert an["fact_mention"] in prev, "the entity must be in the previous sentence"
    else:
        assert an["fact_mention"] in s_ans, "the fact sentence must use the fact form"
        assert an["question_mention"] != an["fact_mention"] and an["question_mention"] in rec["question"]
        assert an["fact_mention"] not in rec["question"], "the question must not repeat the fact's exact mention"
    for dv in an["distractor_values"]:
        s2 = [x for x in sents if dv in x][0]
        assert an["location"] not in s2, "a distractor value is attached to the asked location"


def generate_train(seed: int, n_colliding_docs: int, n_overwrite: int, n_single: int) -> Dict[str, List[dict]]:
    """Programmatic TRAINING data (pools='train'): colliding docs with 2-4 questions each, overwrite docs, clean single-fact docs."""
    g = Gen(seed, "train")
    out = {"prog_colliding": [], "prog_overwrite": [], "prog_single": []}
    for i in range(n_colliding_docs):
        k = [2, 4, 8][i % 3]
        m = g.r.randint(2, min(4, k))
        out["prog_colliding"] += g.colliding_multi(k, m)
    for i in range(n_overwrite):
        out["prog_overwrite"].append(g.overwrite(n_colliding=0))
    for _ in range(n_single):
        out["prog_single"].append(g.single())
    return out


def write_jsonl(path: str, recs: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="miniTitan_eval")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the default category sizes")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--splits", action="store_true", help="v2: write eval_seen_* and eval_heldout_* (disjoint entity types / locations / templates)")
    ap.add_argument("--para", action="store_true", help="write the P1 / P2 paraphrase and pronoun sets (DESIGN_ENTITY_KEYS.md 8)")
    ap.add_argument("--cue", action="store_true", help="write the overwrite cue-probe sets eval_cue_{orig,cuefree,cuefirst}.jsonl (train pools)")
    ap.add_argument("--train_out", default="", help="v2: write programmatic TRAINING files minititan_prog_*.jsonl into this directory")
    a = ap.parse_args()
    if a.para:
        os.makedirs(a.out_dir, exist_ok=True)
        for tag, pools, seed in (("seen", "train", 6060), ("heldout", "heldout", 6161)):
            g = Gen(seed, pools)
            p1 = [g.paraphrase("reorder" if i % 3 else "partial", [2, 4][i % 2]) for i in range(300)]
            p2 = [g.paraphrase("pronoun", [2, 4][i % 2]) for i in range(300)]
            for name, recs in (("p1", p1), ("p2", p2)):
                for rec in recs:
                    verify_para(rec)
                write_jsonl(os.path.join(a.out_dir, f"eval_{tag}_{name}.jsonl"), recs)
                print(f"eval_{tag}_{name}: {len(recs)} records (verified)")
        return
    if a.cue:
        os.makedirs(a.out_dir, exist_ok=True)
        for i, variant in enumerate(("orig", "cuefree", "cuefirst")):
            g = Gen(5150 + i, "train")
            recs = [g.overwrite_cue(variant) for _ in range(300)]
            cue_re = re.compile(r"\b(now|new|changed|update|correction|changeover|revised|amendment)\b", re.I)
            for rec in recs:
                verify(rec)
                sents = re.split(r"(?<=\.)\s+", rec["context"])
                early = [x for x in sents if rec["annot"]["old_values"][0] in x][0]
                late = [x for x in sents if rec["annot"]["later_value"] in x][0]
                want = {"orig": (False, True), "cuefree": (False, False), "cuefirst": (True, False)}[variant]
                assert (bool(cue_re.search(early)), bool(cue_re.search(late))) == want, (variant, early, late)
            write_jsonl(os.path.join(a.out_dir, f"eval_cue_{variant}.jsonl"), recs)
            print(f"eval_cue_{variant}: {len(recs)} records (verified)")
        return
    if a.train_out:
        data = generate_train(a.seed + 1000, n_colliding_docs=4000, n_overwrite=6000, n_single=4000)
        os.makedirs(a.train_out, exist_ok=True)
        for name, recs in data.items():
            for rec in recs:
                verify(rec)
            write_jsonl(os.path.join(a.train_out, f"minititan_{name}.jsonl"), recs)
            docs = len({r["context"] for r in recs})
            print(f"{name}: {len(recs)} examples from {docs} documents (verified)")
        return
    if a.splits:
        sizes = {"single": 300, "overwrite": 300, "colliding": 300, "colliding_overwrite": 200, "haystack": 200, "absent": 0}
        os.makedirs(a.out_dir, exist_ok=True)
        manifest = {"version": VERSION, "files": {}}
        for tag, pools, seed in (("seen", "train", 4242), ("heldout", "heldout", 4343)):
            data = generate(seed, sizes, pools)
            for cat, recs in data.items():
                if not recs:
                    continue
                for rec in recs:
                    verify(rec)
                write_jsonl(os.path.join(a.out_dir, f"eval_{tag}_{cat}.jsonl"), recs)
                manifest["files"][f"eval_{tag}_{cat}"] = {"n": len(recs), "pools": pools, "seed": seed}
        with open(os.path.join(a.out_dir, "manifest_splits.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        print("wrote", len(manifest["files"]), "split eval files (all verified)")
        return
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
