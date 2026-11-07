"""
Experiment config — one dataclass shared by every synthetic experiment ("branch = config, not code fork").

Deliberately torch-free so launchers / sweep builders / analysis can import it without a GPU stack.
"""

from dataclasses import dataclass


@dataclass
class ExpConfig:
    # ── which host ───────────────────────────────────────────────────────────
    model: str = "titan_a"           # none | gru | titan_a | titan_b
    gru_hidden: int = 0              # 0 → dim
    gru_match: str = ""              # "titan_a"/"titan_b": pick GRU width so core params ≈ that model's core params

    # ── host / memory ────────────────────────────────────────────────────────
    dim: int = 64
    mem_hidden: int = 128
    mem_layers: int = 2
    rank: int = 8
    chunk_size: int = 16
    tbptt_chunks: int = 4            # 0 = full BPTT
    inner_lr: float = 0.2
    inner_clip: float = 0.1
    use_momentum: bool = True
    second_order: bool = True
    aux_lag_chunks: int = 4
    aux_weight: float = 0.1

    # ── task ─────────────────────────────────────────────────────────────────
    task: str = "typed"              # typed (easy, Phase 0) | stream (hard, Phase 1)
    n_keys: int = 0                  # 0 → task default (typed 64, stream 1024)
    n_vals: int = 32
    # typed task
    n_filler_vocab: int = 32
    n_pairs: int = 8
    train_filler_max: int = 48       # train filler length ~ U[0, this]
    eval_fillers: str = "0,48,128,512"
    eval_pairs: str = "4,8,16"
    # stream task: facts and distractors are the same kind of token; queries sample uniformly over ALL facts
    n_queries: int = 8
    train_facts_min: int = 16
    train_facts_max: int = 64
    eval_facts: str = "16,64,256,512"
    # oracle ablations (typed task): force read-only positions
    oracle_write_mask: bool = False  # queries are read-only
    oracle_mask_filler: bool = False # filler (and queries) are read-only

    # ── optimisation ─────────────────────────────────────────────────────────
    batch_size: int = 32
    steps: int = 1500
    lr: float = 3e-3
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 4
    seed: int = 0
    out_dir: str = "results"
    run_name: str = ""
