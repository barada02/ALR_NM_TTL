import os

# Use a specific model version supported by Google GenAI Batch API
MODEL_NAME = "gemini-3.1-pro-preview"  # Keep costs low for generation
BATCH_SIZE = 10000               # Number of samples per phase

# ── Prompt Templates ───────────────────────────────────────────────────────────

DATAFORGE_SYSTEM_INSTRUCTION = """You are an expert dataset creator specialized in generating high-quality Supervised Fine-Tuning (SFT) data for a neural long-term memory module.
Your task is to generate complex sequences where information is established early and required later to make a correct decision or answer a question.
STRICT RULES:
- Output ONLY valid JSON matching the requested schema.
- Do not output Markdown code blocks (e.g., no ```json).
- Do not provide explanations.
"""

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
- The story or document must be realistic, highly detailed, and logically consistent. Avoid sci-fi jargon, fantasy elements, or absurd situations.
- The answer must be unambiguous and verbatim extractable from the context.
- Include at least 2 distractors (similar-sounding wrong facts, e.g. alternate numbers, dates, or specifications) to make it challenging.
- The question should require the model to remember something from far back.
- Topic: {topic}
- Setting/Location: {setting}
- Perspective/Role: Written from the perspective of {role}
- Structural Constraint: {constraint}

Return ONLY valid JSON."""

DIALOGUE_PROMPT = """Generate a multi-turn stateful dialogue example for training AI memory.

Format as JSON:
{{
  "turns": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
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
- 12-20 turns total.
- At least 2 memory anchors (info established early, tested late).
- The conversation should feel NATURAL — not obviously a memory test.
- Speakers should not speak in an absurd or robotic manner; the dialogue must sound like actual human conversation in the given setting.
- Intervening turns should introduce noise, topic changes, or distractors.
- Topic/Purpose: {topic}
- Medium/Setting: {setting}
- Speakers: {persona}
- Dialogue Constraint: {constraint}

Return ONLY valid JSON."""

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
- 8-15 steps total.
- The critical result from an early step must influence a decision 4+ steps later.
- Include at least one "red herring" intermediate result that might confuse a memoryless agent.
- Make it realistic: actual file paths, command names, API query structures, database tables, and exit codes.
- Do NOT use generic placeholder names like 'tool_1', 'input_a', or 'result_x'. Use real tools from the list below.
- Domain: {domain}
- Infrastructure/Environment: {environment}
- Allowed Tools: {tools}
- Operational Constraint: {constraint}

Return ONLY valid JSON."""
