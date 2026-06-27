"""
MiniTitan: Neural Long-Term Memory Module
Inspired by: "Titans: Learning to Memorize at Test Time" (Google, 2025)

Key idea: The memory IS a small neural network whose weights update
at inference time based on prediction surprise (gradient signal).
This sits on top of a frozen base LLM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import copy


class SurpriseGate(nn.Module):
    """
    Computes how 'surprising' a new input is relative to memory's expectation.
    High surprise = update memory more aggressively.
    Low surprise = current memory is already handling this well.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, 1)

    def forward(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        error = actual - predicted                        # prediction error
        error_proj = self.proj(error)
        surprise = torch.sigmoid(self.gate(error_proj))  # scalar in [0, 1]
        return surprise                                   # shape: (B, 1)


class NeuralMemoryMLP(nn.Module):
    """
    The memory itself — a small MLP whose WEIGHTS update at test time.
    
    At training time: learns how to update efficiently.
    At inference time: its weights adapt to incoming context.
    
    Think of it as a key-value store implemented as a neural network,
    where the 'write' operation is a gradient step on the weights.
    """
    def __init__(self, dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x) + x)  # residual


class MemoryUpdateRule(nn.Module):
    """
    Learns HOW to update memory weights, not just that they should update.
    This is the meta-learning component — trained during Phase 1/2.
    
    At test time, given a surprise signal and current hidden state,
    it produces a delta that gets applied to the memory MLP weights.
    """
    def __init__(self, dim: int, memory_param_count: int):
        super().__init__()
        # Compress context into an update signal
        self.context_encoder = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
        )
        # Project to actual weight-space delta (low-rank approximation)
        # We use rank-4 decomposition to keep this tractable
        self.rank = 4
        self.to_update = nn.Linear(dim // 2, self.rank * 2)

    def forward(
        self,
        hidden: torch.Tensor,      # current token embedding (B, dim)
        surprise: torch.Tensor,    # surprise gate value (B, 1)
        memory_output: torch.Tensor # what memory currently outputs (B, dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (U, V) for a low-rank weight update: delta_W = surprise * U @ V^T"""
        ctx = torch.cat([hidden, memory_output], dim=-1)
        ctx_enc = self.context_encoder(ctx)
        uv = self.to_update(ctx_enc)  # (B, rank*2)
        U, V = uv.chunk(2, dim=-1)    # each (B, rank)
        # Scale by surprise — more surprising = larger update
        U = U * surprise
        return U, V


class RetentionGate(nn.Module):
    """
    Controls what to FORGET from memory.
    Prevents memory overflow as sequence grows long.
    Similar to the forget gate in LSTMs but operating on weight-space.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Returns retention score in [0,1]. Low = forget more."""
        return self.gate(hidden)


class MiniTitanMemoryModule(nn.Module):
    """
    Full MiniTitan memory module.
    
    At each token position:
    1. Read from memory (what does memory think about this input?)
    2. Compute surprise (how wrong was memory?)
    3. Update memory weights (learn from the surprise)
    4. Apply retention gate (controlled forgetting)
    5. Output memory-augmented representation

    This entire module is what gets trained on Kaggle.
    The base LLM is frozen — we only train this.
    """
    def __init__(
        self,
        input_dim: int,        # matches base LLM hidden dim
        memory_hidden_dim: int = 256,
        memory_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_hidden_dim = memory_hidden_dim

        # Core memory network (weights adapt at test time)
        self.memory = NeuralMemoryMLP(input_dim, memory_hidden_dim, memory_layers)

        # Components
        self.surprise_gate = SurpriseGate(input_dim)
        self.retention_gate = RetentionGate(input_dim)

        # Count memory parameters for update rule
        memory_params = sum(p.numel() for p in self.memory.parameters())
        self.update_rule = MemoryUpdateRule(input_dim, memory_params)

        # Output projection: combines base LLM repr + memory repr
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(input_dim),
        )

        # Learnable blending weight (how much to trust memory vs base)
        self.memory_blend = nn.Parameter(torch.tensor(0.3))

        # Store fast weights (memory state between tokens)
        self._fast_weights: Optional[dict] = None

    def reset_memory(self):
        """Call this between sequences / at start of new context."""
        self._fast_weights = None

    def _get_memory_output(self, x: torch.Tensor) -> torch.Tensor:
        """Run memory with current fast weights (if any)."""
        if self._fast_weights is None:
            return self.memory(x)

        # Temporarily apply fast weights as weight deltas
        # Fast weights are stored as low-rank updates: delta_W = U @ V^T
        # We apply them additively to the first linear layer of memory
        original_weight = self.memory.net[0].weight.data.clone()
        U_acc, V_acc = self._fast_weights['U'], self._fast_weights['V']

        # Low-rank update: W_new = W_orig + sum_i(u_i @ v_i^T)
        # U_acc: (rank, out_dim), V_acc: (rank, in_dim)
        delta = U_acc.T @ V_acc  # (out_dim, in_dim)
        self.memory.net[0].weight.data += delta

        out = self.memory(x)

        # Restore original weights
        self.memory.net[0].weight.data = original_weight
        return out

    def _update_fast_weights(
        self,
        U: torch.Tensor,  # (B, rank)
        V: torch.Tensor,  # (B, rank)
        retention: torch.Tensor,  # (B, 1)
    ):
        """Accumulate low-rank fast weight updates with retention gating."""
        # For simplicity in batch training, use mean across batch
        U_mean = U.mean(0, keepdim=True)  # (1, rank)
        V_mean = V.mean(0, keepdim=True)

        if self._fast_weights is None:
            self._fast_weights = {
                'U': U_mean,
                'V': V_mean,
            }
        else:
            # Retention gate decays old memory, adds new
            ret = retention.mean().item()
            self._fast_weights['U'] = ret * self._fast_weights['U'] + U_mean
            self._fast_weights['V'] = ret * self._fast_weights['V'] + V_mean

    def forward(
        self,
        hidden_states: torch.Tensor,   # (B, seq_len, dim) from frozen base LLM
        update_memory: bool = True,    # False during pure inference (no memory update)
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            hidden_states: output from frozen base LLM encoder layers
            update_memory: whether to update fast weights (True during training/inference,
                           False if you want pure read-only memory query)
        Returns:
            augmented_hidden: memory-augmented hidden states (B, seq_len, dim)
            metrics: dict with surprise scores and retention values for logging
        """
        B, seq_len, dim = hidden_states.shape
        outputs = []
        surprise_scores = []
        retention_scores = []

        for t in range(seq_len):
            h_t = hidden_states[:, t, :]  # (B, dim) — current token

            # 1. Read from memory
            mem_out = self._get_memory_output(h_t)  # (B, dim)

            # 2. Compute surprise: how different is memory's output from actual?
            surprise = self.surprise_gate(mem_out, h_t)  # (B, 1)
            surprise_scores.append(surprise.mean().item())

            # 3. Compute retention (what to keep from past memory)
            retention = self.retention_gate(h_t)  # (B, 1)
            retention_scores.append(retention.mean().item())

            # 4. Compute update (if training or doing test-time adaptation)
            if update_memory:
                U, V = self.update_rule(h_t, surprise, mem_out)  # (B, rank) each
                self._update_fast_weights(U, V, retention)

            # 5. Blend memory output with base hidden state
            blend = torch.sigmoid(self.memory_blend)
            augmented = self.output_proj(
                torch.cat([h_t, blend * mem_out + (1 - blend) * h_t], dim=-1)
            )
            outputs.append(augmented)

        augmented_hidden = torch.stack(outputs, dim=1)  # (B, seq_len, dim)

        metrics = {
            'mean_surprise': sum(surprise_scores) / len(surprise_scores),
            'mean_retention': sum(retention_scores) / len(retention_scores),
            'memory_blend': torch.sigmoid(self.memory_blend).item(),
        }

        return augmented_hidden, metrics


class MiniTitanOutputHead(nn.Module):
    """
    Lightweight output head that maps memory-augmented repr → logits.
    Only used if you want MiniTitan to produce token predictions directly.
    For most use cases, feed augmented_hidden back into the base LLM's
    remaining layers instead.
    """
    def __init__(self, input_dim: int, vocab_size: int):
        super().__init__()
        self.head = nn.Linear(input_dim, vocab_size, bias=False)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, augmented_hidden: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(augmented_hidden))
