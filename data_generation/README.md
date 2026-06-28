# MiniTitan SFT Dataset Generation Pipeline

This directory contains the pipeline for generating high-quality Supervised Fine-Tuning (SFT) datasets using the Google GenAI Batch API. It is designed to train a neural long-term memory module. 

The pipeline generates three distinct memory-heavy SFT task categories:
1. **Long Context:** 800–1200 word narratives with buried facts and distractors.
2. **Dialogue:** 12–20 turn natural multi-turn conversations with memory anchors.
3. **Agentic:** 8–15 step tool-calling sequences requiring historical state recall.

---

## Directory & File Structure

Due to the large file sizes and local tracking requirements, several directories are untracked by Git (`.gitignore`). Below is the mapping of all folders and files:

```
data_generation/
│
├── shared/                             # [TRACKED] Shared modules and configuration
│   ├── config.py                       # Prompt templates and global variables
│   ├── topics.py                       # Combinatorial seed vocabulary lists
│   ├── generator_base.py               # Core state machine logic
│   ├── extract_utils.py                # JSON parsing and deduplication utilities
│   └── batch_utils.py                  # Gemini Files/Batch API wrappers
│
├── generate_long_context.py            # [TRACKED] Long Context generation entrypoint
├── generate_dialogue.py                # [TRACKED] Dialogue generation entrypoint
├── generate_agentic.py                 # [TRACKED] Agentic generation entrypoint
├── test_batch.py                       # [TRACKED] Test entrypoint for 6-sample runs
│
├── .env.example                        # [TRACKED] Template for environment variables
├── .env                                # [GITIGNORED] Holds local GEMINI_API_KEY
│
├── state/                              # [GITIGNORED] Local execution checkpoint folders
│   ├── <dataset_type>_state.json       # Current status and Batch Job metadata
│   ├── <dataset_type>_requests.jsonl   # Raw request prompts payload submitted to Gemini
│   └── dedup_db.json                   # md5 hash registry of all parsed samples
│
├── raw_results/                        # [GITIGNORED] Downloaded raw Gemini responses
│   └── <dataset_type>_raw.jsonl        # Unprocessed raw Batch outputs from Gemini
│
└── processed/                          # [GITIGNORED] Cleaned final datasets
    └── minititan_<dataset_type>.jsonl  # Extracted, deduped, valid JSON SFT datasets
```

---

## How It Works: The State Machine

Each generator script implements an idempotent state machine to handle long-running Batch API jobs safely across interruptions:

```
[NOT_STARTED] ──(Generates requests & submits Batch Job)──> [SUBMITTED]
                                                               │
[DOWNLOADED]  <──(Checks status & downloads raw results)───────┘
     │
     └──(Parses JSON, deduplicates, & saves dataset)─────────> [COMPLETED]
```

To run a script again for a new phase of generation:
1. Archive the processed file under `processed/`.
2. Delete the corresponding state file under `state/`.
3. Run the script.

---

## Configuration

* **Model:** `gemini-3.1-pro-preview` (supports advanced multi-step reasoning).
* **Thinking Mode:** Enabled. This allows the model to map out logically consistent contexts, personas, and distractors prior to writing the final SFT JSON.
* **Token Budget:** `max_output_tokens` is set to `20000` to prevent truncation from thinking tokens.
* **Combinatorial Seeds:** In `topics.py`, we define extensive collections of settings, domains, roles, personas, and tools. When generating 10,000 samples, it randomly combines these elements ($>150,000$ unique prompts per category), guaranteeing **zero duplicates** and maximum diversity.

---

## Setup & Execution

### 1. Configure the Environment
Copy the example environment file and add your API key from Google AI Studio:
```bash
cp data_generation/.env.example data_generation/.env
# Open data_generation/.env and set your GEMINI_API_KEY
```

### 2. Install Dependencies
```bash
pip install google-genai python-dotenv
```

### 3. Run a Quick Test Batch (6 Samples)
This runs a 6-sample batch across all three categories to verify API connection, token budget, and extraction parsing:
```bash
python data_generation/test_batch.py
```

### 4. Trigger Production Generation (10,000 samples per type)
Submit batch jobs to the Google GenAI Batch API:
```bash
# Long Context
python data_generation/generate_long_context.py

# Dialogue
python data_generation/generate_dialogue.py

# Agentic
python data_generation/generate_agentic.py
```

*Note: Since these jobs are processed in the cloud, the script will submit the job and exit. Rerun the commands later to check status and automatically download and parse the results.*
