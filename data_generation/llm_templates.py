"""
E17: sentence TEMPLATES from an open-weight LLM on Fireworks (serverless), for the diversified training mix (family A) and for the
untouched v3 held-out set (family B, used nowhere in training).

The LLM never writes facts or answers. It writes templates with placeholders — fact templates use {ENT}, {ATTR}, {VALUE} exactly once
each, question templates use {ENT}, {ATTR} — and the generator fills them programmatically, so every answer stays exact and
verifiable. Every raw response, every accepted template and every rejection reason is saved (templates/<tag>.json) for reproducibility.

  python data_generation/llm_templates.py --model accounts/fireworks/models/glm-5p3 --tag train_glm
  python data_generation/llm_templates.py --model accounts/fireworks/models/kimi-k3 --tag v3_kimi --exclude templates/train_glm.json

The API key is read ONLY from the environment variable FIREWORKS_API_KEY and never printed.
"""

import argparse
import json
import os
import re
import time
import urllib.request

URL = "https://api.fireworks.ai/inference/v1/chat/completions"

FACT_PROMPT = """You are writing reusable sentence TEMPLATES for a synthetic dataset about industrial equipment records.
Each template states ONE attribute value of ONE piece of equipment and uses exactly these three placeholders, each exactly once:
  {{ENT}}   a bare noun phrase naming the equipment item, WITHOUT an article (e.g. "boiler at Mesa 4"), so write "the {{ENT}}"
  {{ATTR}}  a bare attribute name WITHOUT an article (e.g. "serial number", "replacement cost"), so write "the {{ATTR}}" or "a {{ATTR}}"
  {{VALUE}} the value itself (e.g. "QY-1234-B", "March 3, 2021", "Hana Vogt")
Rules: exactly one sentence ending with a period; {{VALUE}} must NOT be the first word of the sentence; no other placeholders,
no digits, no quotation marks, no line breaks; 8 to 25 words; the sentence must make sense for any attribute and any equipment.
Style for this batch: {style}
Write {n} templates that differ from each other in structure, not just in wording. Return ONLY a JSON array of strings."""

QUESTION_PROMPT = """You are writing reusable QUESTION templates for a synthetic dataset about industrial equipment records.
Each template asks for ONE attribute of ONE piece of equipment and uses exactly these two placeholders, each exactly once:
  {{ENT}}   a bare noun phrase naming the equipment item, WITHOUT an article (e.g. "boiler at Mesa 4"), so write "the {{ENT}}"
  {{ATTR}}  a bare attribute name WITHOUT an article (e.g. "serial number"), so write "the {{ATTR}}"
Rules: exactly one sentence, ending with "?" (a question) or "." (a request); no other placeholders, no digits, no quotation marks;
5 to 18 words; it must make sense for any attribute and any equipment.
Style for this batch: {style}
Write {n} templates that differ from each other in structure, not just in wording. Return ONLY a JSON array of strings."""

FACT_STYLES = [
    "formal maintenance-log register; vary the verb (lists, records, shows, gives, notes, specifies)",
    "passive voice, e.g. something was assigned / was recorded / was entered; put {VALUE} before {ENT} in about half",
    "start with a prepositional or adverbial opener (According to ..., In the ..., Following ..., As of ...)",
    "short plain statements by a technician or inspector; vary who reports it",
    "put {ENT} at the very start or the very end of the sentence; mix both",
    "documents and systems as the source (the asset database, the handover sheet, the audit file, the label plate)",
    "{VALUE} in the middle of the sentence, followed by a clause (e.g. '..., which ...', '... as of the last audit')",
    "varied clause order: subordinate clause first, main clause second",
]
QUESTION_STYLES = [
    "direct wh-questions with different openers (What, Which, Who/When only if it fits any attribute)",
    "polite requests and indirect questions (Could you ..., I need ..., Please look up ...)",
    "questions mentioning a record or document as the source (the register, the log, the file)",
    "questions that put {ENT} first, e.g. 'For the {ENT}, ...'",
]

PH_FACT = ("{ENT}", "{ATTR}", "{VALUE}")
PH_Q = ("{ENT}", "{ATTR}")


def call(model: str, prompt: str, seed: int, temperature: float = 0.9, retries: int = 4) -> str:
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        raise SystemExit("FIREWORKS_API_KEY is not set in the environment")
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature,
                       "max_tokens": 24000, "seed": seed, "reasoning_effort": "low"}).encode()   # high effort overthinks past 24k tokens
    for i in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                                                  "User-Agent": "minititan-research"})
            with urllib.request.urlopen(req, timeout=600) as r:
                c = json.load(r)["choices"][0]
            if c["message"].get("content"):
                return c["message"]["content"]
            print(f"  empty content (finish_reason {c.get('finish_reason')}), retry {i + 1}/{retries}")
        except Exception as e:                      # noqa: BLE001 — report the type only (never the request, which holds the key)
            print(f"  request failed ({type(e).__name__}), retry {i + 1}/{retries}")
            time.sleep(5 * (i + 1))
    print("  giving up on this batch (recorded as empty)")
    return ""


def parse_list(text: str):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return []
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [x.strip() for x in out if isinstance(x, str)]


def check_fact(t: str):
    """None if valid, else the rejection reason."""
    for ph in PH_FACT:
        if t.count(ph) != 1:
            return f"{ph} count {t.count(ph)}"
    if re.search(r"\{(?!ENT\}|ATTR\}|VALUE\})", t):
        return "other placeholder"
    if not t.endswith("."):
        return "does not end with a period"
    if t.startswith("{VALUE}"):
        return "starts with the value (sentence-initial values are never written by design)"
    body = t[:-1]
    if re.search(r"[.!?;]\s", body) or "?" in t or "!" in t or "\n" in t:
        return "more than one sentence"
    if re.search(r"[0-9\"“”]", t):
        return "digit or quote"
    if not re.search(r"\{VALUE\}(?=[ .,])", t):
        return "value not followed by space / comma / period"
    n = len(t.split())
    if not 6 <= n <= 30:
        return f"length {n}"
    if re.search(r"\b(it|its|they|their)\b", t, flags=re.I):
        return "pronoun (would need cross-sentence reference)"
    return None


def check_question(t: str):
    for ph in PH_Q:
        if t.count(ph) != 1:
            return f"{ph} count {t.count(ph)}"
    if "{VALUE}" in t or re.search(r"\{(?!ENT\}|ATTR\})", t):
        return "other placeholder"
    if not (t.endswith("?") or t.endswith(".")):
        return "bad ending"
    if re.search(r"[.!?;]\s", t[:-1]) or "\n" in t:
        return "more than one sentence"
    if re.search(r"[0-9\"“”]", t):
        return "digit or quote"
    n = len(t.split())
    if not 4 <= n <= 20:
        return f"length {n}"
    low = t.lower()
    if re.match(r"(how|is|are|does|do|did|has|have|where|why|was|were)\b", low) or re.search(r"\b(where|anywhere|whether|source)\b", low):
        return "does not ask for the value itself (yes/no, where, how)"
    return None


def _norm_words(t: str):
    """Lower-cased word sequence with the placeholders (either style: {ENT}/{ATTR}/{VALUE} or {ent}/{noun}/{v}) as tokens."""
    for a_, b_ in (("{ENT}", " ENT "), ("{ATTR}", " ATTR "), ("{VALUE}", " VALUE "), ("{ent}", " ENT "), ("{noun}", " ATTR "), ("{v}", " VALUE ")):
        t = t.replace(a_, b_)
    return [w if w in ("ENT", "ATTR", "VALUE") else w.lower() for w in re.findall(r"[A-Za-z']+", t)]


def trigrams(t: str):
    w = _norm_words(t)
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def too_close(t: str, others, thr: float = 0.3) -> bool:
    a = trigrams(t)
    for o in others:
        b = trigrams(o)
        if a and b and len(a & b) / len(a | b) >= thr:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_per_call", type=int, default=20)
    ap.add_argument("--exclude", default="", help="comma-separated template json files / builtin sets whose templates must not be approached")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    import programmatic_eval as pe
    # Training templates must stay away from the v2 development set (it selects between arms). Facts and questions are checked
    # against BOTH kinds of v2 sentence, because phrasing learned from one transfers to the other.
    v2_all = list(pe.FACT_T_V2) + list(pe.RESTATE_T_V2) + list(pe.Q_T["heldout_v2"])
    excl_f, excl_q = list(v2_all), list(v2_all)
    for path in [p for p in a.exclude.split(",") if p]:       # v3: also stay away from EVERYTHING training uses
        d = json.load(open(os.path.join(here, path), encoding="utf-8"))
        train_all = d["fact"]["accepted"] + d["question"]["accepted"] + pe.FACT_T + pe.FACT_T_TRAIN_DIV + pe.RESTATE_T + \
            pe.RESTATE_T_DIV + pe.Q_T["train"] + pe.Q_T_TRAIN_DIV
        excl_f += train_all
        excl_q += train_all
    out = {"model": a.model, "seed": a.seed, "excluded_against": "v2 templates" + (f" + all training templates ({a.exclude})" if a.exclude else ""),
           "prompts": {"fact": FACT_PROMPT, "question": QUESTION_PROMPT}}
    for kind, prompt, styles, check, excl in (("fact", FACT_PROMPT, FACT_STYLES, check_fact, excl_f),
                                              ("question", QUESTION_PROMPT, QUESTION_STYLES, check_question, excl_q)):
        raw, accepted, rejected = [], [], []
        for i, style in enumerate(styles):
            text = call(a.model, prompt.format(style=style, n=a.n_per_call), seed=a.seed + i)
            items = parse_list(text)
            raw.append({"style": style, "n_parsed": len(items), "response": text})
            for t in items:
                why = check(t)
                if why is None and any(_norm_words(t) == _norm_words(x) for x in accepted):
                    why = "duplicate"
                if why is None and too_close(t, accepted, thr=0.6):
                    why = "near-duplicate of an accepted template"
                if why is None and too_close(t, excl):
                    why = "too close to an excluded (held-out / training) template"
                (rejected if why else accepted).append({"t": t, "why": why} if why else t)
            print(f"  {kind} style {i}: parsed {len(items)}, accepted so far {len(accepted)}")
        out[kind] = {"accepted": accepted, "rejected": rejected, "raw": raw}
    os.makedirs(os.path.join(here, "templates"), exist_ok=True)
    path = os.path.join(here, "templates", f"{a.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"saved {path}: {len(out['fact']['accepted'])} fact / {len(out['question']['accepted'])} question templates accepted")


if __name__ == "__main__":
    main()
