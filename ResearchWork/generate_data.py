"""
MiniTitan Data Generation via Gemini API
Generates three dataset types for training the memory module.

Run this LOCALLY or in CPU-only Kaggle session (no GPU needed).
Uses batch processing to stay within free tier limits.

pip install google-generativeai datasets tqdm
"""

import os
import json
import time
import random
import hashlib
from pathlib import Path
from typing import Generator
from tqdm import tqdm
import google.generativeai as genai

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_KEY_HERE")
MODEL_NAME     = "gemini-1.5-flash"   # cheapest, fast, sufficient
OUTPUT_DIR     = Path("./minititan_data")
SAMPLES_PER_TYPE = 10_000             # 10K per type = 30K total to start
                                      # scale to 50K later if budget allows

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

OUTPUT_DIR.mkdir(exist_ok=True)

# ── Prompt Templates ───────────────────────────────────────────────────────────

LONG_CONTEXT_PROMPT = """Generate a long-context recall example for training an AI memory system.

Format as JSON with these exact fields:
{{
  "context": "<a story or document, 800-1200 words, that contains 3 specific facts>",
  "facts": ["fact1", "fact2", "fact3"],
  "question": "<a question that requires recalling one of the buried facts>",
  "answer": "<the exact answer from the context>",
  "fact_position": "<early|middle|late — where in context the answer fact appears>",
  "distractor_count": <number of plausible-sounding wrong facts in the context>
}}

Rules:
- The answer must be unambiguous and verbatim extractable from context
- Include at least 2 distractors (similar-sounding wrong facts) to make it hard
- The question should require the model to remember something from far back
- Topic: {topic}

Return ONLY valid JSON. No markdown. No explanation."""

DIALOGUE_PROMPT = """Generate a multi-turn stateful dialogue example for training AI memory.

Format as JSON:
{{
  "turns": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    ...
  ],
  "memory_anchors": [
    {{
      "established_at_turn": <turn index where key info is introduced>,
      "recalled_at_turn": <turn index where it must be recalled>,
      "info": "<the specific info that must be remembered>"
    }}
  ],
  "test_question": "<a question asked after all turns that requires early-turn memory>",
  "correct_answer": "<what a model with perfect memory would say>"
}}

Rules:
- 12-20 turns total
- At least 2 memory anchors (info established early, tested late)
- The conversation should feel NATURAL — not obviously a memory test
- Intervening turns should introduce noise/distractors
- Topic: {topic}

Return ONLY valid JSON. No markdown."""

AGENTIC_PROMPT = """Generate an agentic tool-call sequence example for training AI memory.

Format as JSON:
{{
  "task": "<high-level task the agent is trying to accomplish>",
  "steps": [
    {{
      "step": <int>,
      "action": "<tool_name>",
      "input": "<what was passed to tool>",
      "output": "<what tool returned>",
      "key_result": "<null or the critical piece of info this step produced>"
    }}
  ],
  "decision_point": {{
    "at_step": <step where agent must use earlier result>,
    "requires_memory_of_step": <which earlier step's output is needed>,
    "correct_decision": "<what the agent should do, based on that earlier result>",
    "wrong_decision_if_forgotten": "<what a memoryless agent would do instead>"
  }},
  "final_answer": "<correct task completion answer>"
}}

Rules:
- 8-15 steps total
- The critical result from an early step must influence a decision 4+ steps later
- Include at least one "red herring" intermediate result that might confuse
- Make it realistic: file operations, API calls, database queries, calculations
- Domain: {domain}

Return ONLY valid JSON. No markdown."""

# ── Topics & Domains ──────────────────────────────────────────────────────────

LONG_CONTEXT_TOPICS = [
    "scientific expedition", "historical mystery", "company acquisition",
    "medical case study", "archaeological discovery", "legal investigation",
    "engineering project", "financial audit", "environmental study",
    "biographical profile", "travel journal", "product development timeline",
]

DIALOGUE_TOPICS = [
    "planning a home renovation", "job interview preparation", "trip planning",
    "learning a new skill", "medical consultation", "software debugging session",
    "investment planning", "recipe development", "book club discussion",
    "startup pitch preparation", "research collaboration", "apartment hunting",
]

AGENTIC_DOMAINS = [
    "data pipeline debugging", "cloud infrastructure setup",
    "multi-step file processing", "API integration workflow",
    "database migration", "security audit", "deployment automation",
    "document processing pipeline", "web scraping and analysis",
    "ML experiment tracking", "codebase refactoring", "report generation",
]

# ── Generation Helpers ─────────────────────────────────────────────────────────

def call_gemini_with_retry(prompt: str, max_retries: int = 3) -> str | None:
    """Call Gemini with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.9,
                    max_output_tokens=2048,
                )
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = (2 ** attempt) * 5
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Error: {e}")
                return None
    return None


def parse_json_response(text: str) -> dict | None:
    """Safely parse Gemini's JSON response."""
    if not text:
        return None
    # Strip any accidental markdown fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def dedup_key(sample: dict) -> str:
    """Generate a deduplication hash for a sample."""
    content = json.dumps(sample, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def generate_dataset(
    dataset_type: str,
    n_samples: int,
    output_file: Path,
) -> int:
    """
    Generate a dataset of given type.
    Resumes from existing file if interrupted (checkpoint-friendly).
    Returns number of samples generated.
    """
    # Load existing samples for dedup and resume
    existing = []
    seen_hashes = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    existing.append(sample)
                    seen_hashes.add(dedup_key(sample))
                except:
                    pass
        print(f"  Resuming: {len(existing)} samples already exist.")

    needed = n_samples - len(existing)
    if needed <= 0:
        print(f"  Already complete ({len(existing)} samples).")
        return len(existing)

    generated = 0
    failed = 0

    with open(output_file, "a") as f:
        pbar = tqdm(total=needed, desc=f"Generating {dataset_type}")
        while generated < needed:
            # Pick random topic/domain
            if dataset_type == "long_context":
                topic = random.choice(LONG_CONTEXT_TOPICS)
                prompt = LONG_CONTEXT_PROMPT.format(topic=topic)
            elif dataset_type == "dialogue":
                topic = random.choice(DIALOGUE_TOPICS)
                prompt = DIALOGUE_PROMPT.format(topic=topic)
            elif dataset_type == "agentic":
                domain = random.choice(AGENTIC_DOMAINS)
                prompt = AGENTIC_PROMPT.format(domain=domain)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

            raw = call_gemini_with_retry(prompt)
            sample = parse_json_response(raw)

            if sample is None:
                failed += 1
                if failed > 50:
                    print("\n  Too many failures. Check API key / quota.")
                    break
                continue

            # Deduplicate
            h = dedup_key(sample)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # Add metadata
            sample["_type"] = dataset_type
            sample["_id"] = len(existing) + generated

            f.write(json.dumps(sample) + "\n")
            f.flush()  # important: don't lose progress on crash

            generated += 1
            pbar.update(1)

            # Throttle: ~1 req/sec to stay within free tier
            time.sleep(1.1)

        pbar.close()

    total = len(existing) + generated
    print(f"  Done. {generated} new samples. Total: {total}. Failed: {failed}.")
    return total


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MiniTitan Data Generation")
    print("=" * 60)

    datasets = [
        ("long_context", OUTPUT_DIR / "long_context_train.jsonl"),
        ("dialogue",     OUTPUT_DIR / "dialogue_train.jsonl"),
        ("agentic",      OUTPUT_DIR / "agentic_train.jsonl"),
    ]

    for dtype, outfile in datasets:
        print(f"\n[{dtype.upper()}] → {outfile}")
        n = generate_dataset(dtype, SAMPLES_PER_TYPE, outfile)
        print(f"  Total: {n} samples")

    # Print stats
    print("\n" + "=" * 60)
    print("Dataset Summary:")
    total = 0
    for _, outfile in datasets:
        if outfile.exists():
            count = sum(1 for _ in open(outfile))
            size_mb = outfile.stat().st_size / 1e6
            print(f"  {outfile.name}: {count} samples ({size_mb:.1f} MB)")
            total += count
    print(f"  TOTAL: {total} samples")
    print("\nNext: Upload ./minititan_data/ as a Kaggle Dataset.")


if __name__ == "__main__":
    main()
