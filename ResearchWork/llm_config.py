"""
Config for the frozen-LLM + memory experiments (Phase 2). Torch-free (importable by launchers/analysis).

Pipeline in one picture:

    document ──► [seg 1][seg 2] ... [seg n][question+answer seg]       every segment = seg_len tokens
                   │      │            │            │
                   └──────┴── frozen Qwen, run on each segment INDEPENDENTLY (no attention across segments) ──┐
                                                                                                              │
         memory hook at one layer ("mid") or after the final norm ("last") sees ALL segments in order:       │
         writes while reading the context, reads while processing the question  ◄───────────────────────────┘
    loss = CE on the answer tokens only, through the frozen lm_head. Only the memory module trains.
"""

from dataclasses import dataclass


@dataclass
class LLMConfig:
    run_name: str = ""
    base_model: str = "Qwen/Qwen2-0.5B"
    insert: str = "mid"              # mid: after layer `insert_layer` (re-run frozen upper layers) | last: after final norm
    insert_layer: int = -1           # number of layers BELOW the memory; -1 → num_layers // 2
    seg_len: int = 256
    max_ctx_segments: int = 8        # longer documents are truncated, keeping the START (most facts are early)

    # memory (Mode B was chosen on the synthetic tasks: second-order, no momentum)
    mem_mode: str = "B"
    mem_hidden: int = 256
    mem_layers: int = 2
    rank: int = 8
    chunk_size: int = 32             # must divide seg_len
    tbptt_chunks: int = 0            # 0 = full BPTT; long-context facts sit 10-60 chunks before the question
    shift_v: bool = False            # next-token values (induction-style) — see MemoryConfig.shift_v
    inner_lr: float = 0.2
    inner_clip: float = 0.1
    use_momentum: bool = False
    second_order: bool = True
    retention_bias: float = 4.0
    retention_learned: bool = True
    aux_weight: float = 0.1          # Mode A only

    # paired controls (the fair reference for the headline numbers)
    train_mode: str = "memory"       # memory: the real model
                                     # adapter_seg: SAME module with writes disabled on the segmented model = document-free adapter
                                     # adapter_oracle: SAME module with writes disabled on the FULL-attention model = adapter-matched oracle

    # anti-shortcut training (E7: 95-99% of the Stage A gain was document-independent)
    contrast_weight: float = 0.0     # >0: also run the question with ANOTHER document and add
                                     #   w · relu(margin − (nll_swap − nll_on))  → only document-specific info lowers the loss
    contrast_margin: float = 1.0
    crop: bool = False               # crop-around-the-fact distance curriculum (early: short context right after the fact)
    crop_curriculum: float = 0.5     # fraction of training over which the max fact→question distance ramps up
    crop_pre_tokens: int = 48        # context kept before the fact
    crop_d_min: int = 32             # shortest distance (tokens after the fact starts)

    # mixture (Stage 2): "type:weight,..." sampled with replacement; empty = uniform over all training examples
    mix: str = ""
    eval_types: str = ""             # restrict the end-of-run full evaluation to these types (empty = all); programmatic types are
                                     # evaluated afterwards with the eval-only runner (seen / held-out sets)
    crop_annotated: bool = False     # False: programmatic (annotated) documents are NOT cropped — cropping would cut the colliders out

    # data
    data_dir: str = "/data/miniTitan_data"
    types: str = "long_context,dialogue,agentic"
    val_per_type: int = 150
    gen_eval_per_type: int = 48
    seed: int = 0                    # training randomness (init, batch order)
    split_seed: int = 0              # train/val split — fixed so every seed is evaluated on the same validation set

    # optimisation
    batch_size: int = 4
    steps: int = 3000
    lr: float = 1e-3
    warmup: int = 100
    grad_clip: float = 1.0
    log_every: int = 25
    eval_every: int = 500
    eval_batches: int = 12           # quick in-training eval (CE only)
    gen_max_new: int = 32
    save_ckpt: bool = True
    out_dir: str = "results/llm"
