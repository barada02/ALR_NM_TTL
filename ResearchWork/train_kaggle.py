"""
MiniTitan Training Loop — Kaggle Notebook
=========================================
Designed for: T4x2 or P100, 16GB VRAM, 12hr session cap
Checkpoint-aware: resumes from any interruption point
Multi-session: friend's account can fork and continue

Usage:
  1. Upload memory_module.py as a Kaggle Dataset
  2. Upload minititan_data/ as a Kaggle Dataset
  3. Run this notebook with GPU T4x2 enabled
  4. Outputs checkpoint to /kaggle/working/ — save to Dataset after each session

Install (first cell in Kaggle):
  !pip install transformers bitsandbytes accelerate peft datasets -q
"""

import os
import json
import time
import math
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

# Import our memory module
# In Kaggle: sys.path.insert(0, '/kaggle/input/minititan-code/')
from memory_module import MiniTitanMemoryModule, MiniTitanOutputHead


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    base_model: str         = "Qwen/Qwen2-1.5B"   # frozen, 4-bit quantized
    memory_hidden_dim: int  = 256
    memory_layers: int      = 3

    # Data
    data_dir: str           = "/kaggle/input/minititan-data/"
    max_seq_len: int        = 1024     # tokens per sample
    train_split: float      = 0.95

    # Training
    batch_size: int         = 4        # safe for 16GB VRAM with 1.5B base
    grad_accum_steps: int   = 8        # effective batch = 32
    learning_rate: float    = 3e-4
    weight_decay: float     = 0.01
    warmup_steps: int       = 100
    max_steps: int          = 5000     # per session
    save_every: int         = 250      # checkpoint every N steps
    log_every: int          = 25

    # Session management
    checkpoint_dir: str     = "/kaggle/working/checkpoints/"
    # After training: copy to /kaggle/working/ and save as Dataset output

    # Safety: stop N minutes before Kaggle kills the session
    session_budget_hours: float = 10.5   # leave 1.5hr buffer from 12hr limit
    start_time: float           = field(default_factory=time.time)

    def time_remaining_hours(self) -> float:
        elapsed = (time.time() - self.start_time) / 3600
        return self.session_budget_hours - elapsed

    def should_stop(self) -> bool:
        return self.time_remaining_hours() < 0.25  # stop if <15min left


cfg = TrainConfig()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Dataset ────────────────────────────────────────────────────────────────────

class MiniTitanDataset(Dataset):
    """
    Loads all three dataset types and formats them for memory training.
    
    For each sample, we format as:
      [CONTEXT TOKENS] [SEP] [QUESTION TOKENS] [SEP] [ANSWER TOKENS]
    
    The memory module sees the full context, then must recall relevant
    info when generating the answer.
    """

    def __init__(self, tokenizer, data_dir: str, split: str = "train",
                 train_split: float = 0.95, max_seq_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        data_path = Path(data_dir)
        files = list(data_path.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files found in {data_dir}")

        print(f"Loading datasets from {data_dir}...")
        raw_samples = []
        for fpath in files:
            with open(fpath) as f:
                for line in f:
                    try:
                        raw_samples.append(json.loads(line))
                    except:
                        pass
        print(f"  Loaded {len(raw_samples)} raw samples")

        # Train/val split (deterministic)
        split_idx = int(len(raw_samples) * train_split)
        if split == "train":
            raw_samples = raw_samples[:split_idx]
        else:
            raw_samples = raw_samples[split_idx:]

        # Format each sample type
        for sample in raw_samples:
            formatted = self._format_sample(sample)
            if formatted:
                self.samples.append(formatted)

        print(f"  {split} set: {len(self.samples)} formatted samples")

    def _format_sample(self, sample: dict) -> Optional[dict]:
        """Convert raw sample to (input_ids, labels, context_len) tuple."""
        dtype = sample.get("_type", "unknown")

        if dtype == "long_context":
            context = sample.get("context", "")
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            text = f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}"
            # Context ends before question
            context_text = f"Context:\n{context}\n\nQuestion: {question}\nAnswer: "

        elif dtype == "dialogue":
            turns = sample.get("turns", [])
            test_q = sample.get("test_question", "")
            correct = sample.get("correct_answer", "")
            dialogue = "\n".join(
                f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}"
                for t in turns
            )
            text = f"{dialogue}\n\nFinal question: {test_q}\nAnswer: {correct}"
            context_text = f"{dialogue}\n\nFinal question: {test_q}\nAnswer: "

        elif dtype == "agentic":
            task = sample.get("task", "")
            steps = sample.get("steps", [])
            dp = sample.get("decision_point", {})
            final = sample.get("final_answer", "")
            steps_text = "\n".join(
                f"Step {s['step']}: [{s['action']}] Input: {s['input']} → Output: {s['output']}"
                for s in steps
            )
            text = f"Task: {task}\n\n{steps_text}\n\nFinal answer: {final}"
            context_text = f"Task: {task}\n\n{steps_text}\n\nFinal answer: "

        else:
            return None

        # Tokenize
        full_enc = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )
        ctx_enc = self.tokenizer(
            context_text,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = full_enc["input_ids"].squeeze(0)
        context_len = min(ctx_enc["input_ids"].shape[1], len(input_ids))

        # Labels: -100 for context tokens (don't train on them), real ids for answer
        labels = input_ids.clone()
        labels[:context_len] = -100

        if labels[context_len:].eq(-100).all():
            return None  # no answer tokens — skip

        return {
            "input_ids": input_ids,
            "labels": labels,
            "context_len": context_len,
            "dtype": dtype,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(s["input_ids"].shape[0] for s in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels    = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, s in enumerate(batch):
        L = s["input_ids"].shape[0]
        input_ids[i, :L] = s["input_ids"]
        labels[i, :L]    = s["labels"]
        attn_mask[i, :L] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn_mask,
        "context_lens": [s["context_len"] for s in batch],
    }


# ── Model Setup ────────────────────────────────────────────────────────────────

def load_frozen_base(model_name: str, device: str):
    """Load base LLM in 4-bit quantization — frozen, never trained."""
    print(f"Loading base model: {model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_dim = model.config.hidden_size
    vocab_size  = model.config.vocab_size
    print(f"  Hidden dim: {hidden_dim}, Vocab: {vocab_size}")
    print(f"  Base model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M (frozen)")
    return model, tokenizer, hidden_dim, vocab_size


def get_base_hidden_states(
    base_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hidden states from frozen base model (midpoint layer)."""
    with torch.no_grad():
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    # Use middle layer — empirically better than last layer for memory training
    n_layers = len(outputs.hidden_states)
    mid_layer = n_layers // 2
    return outputs.hidden_states[mid_layer]  # (B, seq_len, hidden_dim)


# ── Checkpoint Management ──────────────────────────────────────────────────────

class CheckpointManager:
    """
    Handles save/load across Kaggle sessions.
    Saves: memory module weights, optimizer state, step count, metrics.
    
    After session ends: go to Kaggle sidebar → Output → Save to Dataset
    Before next session: load that Dataset as input.
    """
    def __init__(self, checkpoint_dir: str):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "training_log.jsonl"

    def save(
        self,
        step: int,
        memory_module: nn.Module,
        output_head: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        metrics: dict,
    ):
        """Save checkpoint. Keeps last 3 to save space."""
        ckpt = {
            "step": step,
            "memory_module_state": memory_module.state_dict(),
            "output_head_state":   output_head.state_dict(),
            "optimizer_state":     optimizer.state_dict(),
            "scheduler_state":     scheduler.state_dict(),
            "metrics":             metrics,
        }
        path = self.dir / f"step_{step:06d}.pt"
        torch.save(ckpt, path)
        print(f"  ✓ Checkpoint saved: {path.name}")

        # Keep only last 3 checkpoints
        all_ckpts = sorted(self.dir.glob("step_*.pt"))
        for old in all_ckpts[:-3]:
            old.unlink()

        # Log metrics
        with open(self.log_path, "a") as f:
            entry = {"step": step, "timestamp": time.time(), **metrics}
            f.write(json.dumps(entry) + "\n")

    def load_latest(
        self,
        memory_module: nn.Module,
        output_head: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
    ) -> int:
        """Load most recent checkpoint. Returns step number (0 if none)."""
        all_ckpts = sorted(self.dir.glob("step_*.pt"))
        if not all_ckpts:
            # Also check input datasets (from previous session)
            input_ckpts = sorted(Path("/kaggle/input/").rglob("step_*.pt"))
            if not input_ckpts:
                print("  No checkpoint found. Starting fresh.")
                return 0
            all_ckpts = input_ckpts

        latest = all_ckpts[-1]
        print(f"  Loading checkpoint: {latest}")
        ckpt = torch.load(latest, map_location=DEVICE)

        memory_module.load_state_dict(ckpt["memory_module_state"])
        output_head.load_state_dict(ckpt["output_head_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])

        step = ckpt["step"]
        print(f"  Resumed from step {step}")
        return step


# ── Training Loop ─────────────────────────────────────────────────────────────

def train():
    cfg.start_time = time.time()

    # 1. Load frozen base
    base_model, tokenizer, hidden_dim, vocab_size = load_frozen_base(
        cfg.base_model, DEVICE
    )

    # 2. Initialize trainable modules
    memory_module = MiniTitanMemoryModule(
        input_dim=hidden_dim,
        memory_hidden_dim=cfg.memory_hidden_dim,
        memory_layers=cfg.memory_layers,
    ).to(DEVICE).float()

    output_head = MiniTitanOutputHead(hidden_dim, vocab_size).to(DEVICE).float()

    trainable_params = (
        list(memory_module.parameters()) +
        list(output_head.parameters())
    )
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"\nTrainable params: {n_trainable / 1e6:.1f}M (memory module + head)")

    # 3. Dataset
    train_dataset = MiniTitanDataset(
        tokenizer, cfg.data_dir, split="train",
        train_split=cfg.train_split, max_seq_len=cfg.max_seq_len
    )
    val_dataset = MiniTitanDataset(
        tokenizer, cfg.data_dir, split="val",
        train_split=cfg.train_split, max_seq_len=cfg.max_seq_len
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=2
    )

    # 4. Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    total_steps = cfg.max_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = GradScaler()

    # 5. Load checkpoint (resume if exists)
    ckpt_manager = CheckpointManager(cfg.checkpoint_dir)
    global_step = ckpt_manager.load_latest(
        memory_module, output_head, optimizer, scheduler
    )

    # 6. Training loop
    print(f"\n{'='*60}")
    print(f"Training from step {global_step} to {total_steps}")
    print(f"Session budget: {cfg.session_budget_hours}h")
    print(f"{'='*60}\n")

    memory_module.train()
    output_head.train()

    running_loss = 0.0
    running_surprise = 0.0
    optimizer.zero_grad()

    data_iter = iter(train_loader)
    step_in_session = 0

    while global_step < total_steps:
        # Time safety check
        if cfg.should_stop():
            print(f"\n⚠️  Time budget reached at step {global_step}. Saving final checkpoint.")
            ckpt_manager.save(
                global_step, memory_module, output_head, optimizer, scheduler,
                {"loss": running_loss / max(step_in_session, 1), "status": "time_limit"}
            )
            break

        # Get next batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids   = batch["input_ids"].to(DEVICE)
        labels      = batch["labels"].to(DEVICE)
        attn_mask   = batch["attention_mask"].to(DEVICE)

        with autocast(dtype=torch.float16):
            # Get frozen base model hidden states
            hidden_states = get_base_hidden_states(base_model, input_ids, attn_mask)

            # Reset memory for new sequence
            memory_module.reset_memory()

            # Run through memory module
            augmented, mem_metrics = memory_module(hidden_states, update_memory=True)

            # Project to logits
            logits = output_head(augmented)  # (B, seq_len, vocab_size)

            # Compute loss only on answer tokens (labels != -100)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = loss / cfg.grad_accum_steps

        scaler.scale(loss).backward()

        running_loss += loss.item() * cfg.grad_accum_steps
        running_surprise += mem_metrics["mean_surprise"]

        # Gradient accumulation
        if (step_in_session + 1) % cfg.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            step_in_session += 1

            # Logging
            if global_step % cfg.log_every == 0:
                avg_loss = running_loss / cfg.log_every
                avg_surprise = running_surprise / cfg.log_every
                lr = scheduler.get_last_lr()[0]
                time_left = cfg.time_remaining_hours()
                print(
                    f"Step {global_step:5d} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Surprise: {avg_surprise:.3f} | "
                    f"Blend: {mem_metrics['memory_blend']:.3f} | "
                    f"LR: {lr:.2e} | "
                    f"Time left: {time_left:.1f}h"
                )
                running_loss = 0.0
                running_surprise = 0.0

            # Checkpoint
            if global_step % cfg.save_every == 0:
                val_loss = evaluate(base_model, memory_module, output_head,
                                    val_loader, vocab_size, n_batches=20)
                print(f"  Val loss: {val_loss:.4f}")
                ckpt_manager.save(
                    global_step, memory_module, output_head, optimizer, scheduler,
                    {"train_loss": avg_loss, "val_loss": val_loss,
                     "mean_surprise": avg_surprise}
                )

    print(f"\nSession complete. Final step: {global_step}")
    print(f"Checkpoints saved to: {cfg.checkpoint_dir}")
    print("\nNext steps:")
    print("  1. Go to Kaggle sidebar → Data → Output")
    print("  2. Save checkpoints/ folder as a new Dataset version")
    print("  3. Next session: load that Dataset as input and re-run")


@torch.no_grad()
def evaluate(base_model, memory_module, output_head, val_loader,
             vocab_size: int, n_batches: int = 20) -> float:
    """Quick validation pass."""
    memory_module.eval()
    output_head.eval()
    total_loss = 0.0
    count = 0

    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        input_ids = batch["input_ids"].to(DEVICE)
        labels    = batch["labels"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)

        with autocast(dtype=torch.float16):
            hidden = get_base_hidden_states(base_model, input_ids, attn_mask)
            memory_module.reset_memory()
            # At eval: update_memory=False for pure retrieval test
            augmented, _ = memory_module(hidden, update_memory=False)
            logits = output_head(augmented)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        total_loss += loss.item()
        count += 1

    memory_module.train()
    output_head.train()
    return total_loss / max(count, 1)


# ── Evaluation Script (run after training) ────────────────────────────────────

@torch.no_grad()
def run_needle_in_haystack_eval(
    base_model,
    tokenizer,
    memory_module: nn.Module,
    output_head: nn.Module,
    test_samples: List[dict],
    vocab_size: int,
) -> dict:
    """
    Needle-in-haystack evaluation.
    Tests: can memory recall a fact buried N tokens back?
    
    Returns accuracy broken down by fact_position (early/middle/late).
    """
    memory_module.eval()
    output_head.eval()

    results = {"early": [], "middle": [], "late": [], "all": []}

    for sample in test_samples:
        if sample.get("_type") != "long_context":
            continue

        context = sample["context"]
        question = sample["question"]
        answer = sample["answer"].strip().lower()
        position = sample.get("fact_position", "unknown")

        # Tokenize full context + question
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        enc = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        input_ids = enc["input_ids"].to(DEVICE)
        attn_mask = enc["attention_mask"].to(DEVICE)

        # Get hidden states
        hidden = get_base_hidden_states(base_model, input_ids, attn_mask)

        # Run WITH memory
        memory_module.reset_memory()
        augmented_with, _ = memory_module(hidden, update_memory=True)
        logits_with = output_head(augmented_with)

        # Run WITHOUT memory (baseline)
        memory_module.reset_memory()
        augmented_without, _ = memory_module(hidden, update_memory=False)
        logits_without = output_head(augmented_without)

        # Greedy decode 20 tokens for each
        def greedy_decode(logits, n=20):
            tokens = []
            for t in range(min(n, logits.shape[1])):
                next_tok = logits[0, t, :].argmax().item()
                tokens.append(next_tok)
            return tokenizer.decode(tokens, skip_special_tokens=True).strip().lower()

        pred_with    = greedy_decode(logits_with)
        pred_without = greedy_decode(logits_without)

        correct_with    = answer in pred_with
        correct_without = answer in pred_without

        results["all"].append({
            "correct_with_memory":    correct_with,
            "correct_without_memory": correct_without,
            "position": position,
        })
        if position in results:
            results[position].append({
                "with": correct_with,
                "without": correct_without,
            })

    # Compute summary
    def acc(lst, key): 
        return sum(s[key] for s in lst) / len(lst) if lst else 0.0

    summary = {
        "overall_with_memory":    acc(results["all"], "correct_with_memory"),
        "overall_without_memory": acc(results["all"], "correct_without_memory"),
        "n_samples": len(results["all"]),
    }
    for pos in ["early", "middle", "late"]:
        lst = results[pos]
        if lst:
            summary[f"{pos}_with"]    = acc(lst, "with")
            summary[f"{pos}_without"] = acc(lst, "without")

    print("\nNeedle-in-Haystack Results:")
    print(f"  With memory:    {summary['overall_with_memory']:.1%}")
    print(f"  Without memory: {summary['overall_without_memory']:.1%}")
    print(f"  Delta:          {summary['overall_with_memory'] - summary['overall_without_memory']:+.1%}")

    return summary


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
