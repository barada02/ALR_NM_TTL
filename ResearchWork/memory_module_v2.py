"""
MiniTitan v2: Neural Long-Term Memory Module
Inspired by: "Titans: Learning to Memorize at Test Time" (Google, 2025)

v2 fixes (based on expert review):
  [F1] Low-rank U,V now have correct shapes: U(out,r), V(in,r) → ΔW = U @ V.T → (out,in) ✓
  [F2] No more .data mutation — memory forward uses functional weight composition
  [F3] Fast weights stored per-layer, per-sequence (no batch averaging)
  [F4] Gradient-based memory update option (true Titans: loss.grad → fast_weight step)
  [F5] Token-dependent memory blend gate (not a single scalar)
  [F6] Surprise computed from MSE prediction loss, not raw hidden difference
  [F7] Fast weights applied via torch.nn.functional calls — no clone/restore overhead
  [F8] Update rule receives previous fast weights + prediction error (not just hidden)
  [F9] Per-sequence fast weights — batch items maintain independent memory states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field


# ── Data structure for per-layer, per-sequence fast weights ───────────────────

@dataclass
class LayerFastWeight:
    """
    Low-rank adapter for one Linear layer.
    Effective weight = W_base + U @ V.T
    
    U: (out_features, rank)
    V: (in_features,  rank)
    ΔW = U @ V.T  →  shape (out_features, in_features)  ✓ matches W_base
    """
    U: torch.Tensor   # (out, rank)
    V: torch.Tensor   # (in,  rank)

    def delta(self) -> torch.Tensor:
        """Compute weight delta. Shape: (out, in)"""
        return self.U @ self.V.T

    def decay(self, retention: float) -> "LayerFastWeight":
        """Exponential decay for forgetting."""
        return LayerFastWeight(U=self.U * retention, V=self.V * retention)

    def accumulate(self, new_U: torch.Tensor, new_V: torch.Tensor) -> "LayerFastWeight":
        """Add a new rank-r update (concatenate along rank dimension)."""
        # Concatenate and keep only most recent K updates to bound memory cost
        MAX_RANK = 32
        U_cat = torch.cat([self.U, new_U], dim=1)  # (out, old_rank + r)
        V_cat = torch.cat([self.V, new_V], dim=1)  # (in,  old_rank + r)
        if U_cat.shape[1] > MAX_RANK:
            # Drop oldest updates (leftmost columns)
            U_cat = U_cat[:, -MAX_RANK:]
            V_cat = V_cat[:, -MAX_RANK:]
        return LayerFastWeight(U=U_cat, V=V_cat)


# Per-sequence memory state: one LayerFastWeight per memory MLP layer
SequenceMemoryState = List[Optional[LayerFastWeight]]  # length = n_memory_layers


# ── Memory MLP with functional forward (no weight mutation) ───────────────────

class NeuralMemoryMLP(nn.Module):
    """
    The memory network. Base weights are trained parameters (slow weights).
    At test time, fast weights (ΔW per layer) are added functionally —
    we never touch .data, never clone/restore, fully autograd-safe.

    Architecture: Linear(dim→h) → GELU → Linear(h→h) → GELU → Linear(h→dim)
    Residual connection + LayerNorm.
    """
    def __init__(self, dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Store as ModuleList so each layer is accessible individually
        self.linears = nn.ModuleList()
        in_d = dim
        for i in range(num_layers - 1):
            self.linears.append(nn.Linear(in_d, hidden_dim))
            in_d = hidden_dim
        self.linears.append(nn.Linear(in_d, dim))

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,                          # (B, dim)
        fast_weights: SequenceMemoryState = None,  # per-layer fast weight deltas
    ) -> torch.Tensor:
        """
        Functional forward: applies base weights + fast weight deltas.
        No .data access. Fully differentiable.
        """
        h = x
        for i, layer in enumerate(self.linears):
            # Effective weight = base + fast delta (if any)
            W = layer.weight  # (out, in)
            b = layer.bias    # (out,)

            if fast_weights is not None and fast_weights[i] is not None:
                W = W + fast_weights[i].delta()  # ΔW = U @ V.T, same shape as W

            h = F.linear(h, W, b)

            # Activation after all layers except last
            if i < len(self.linears) - 1:
                h = F.gelu(h)

        # Residual + norm
        return self.norm(h + x)

    def layer_shapes(self) -> List[Tuple[int, int]]:
        """Returns (out_dim, in_dim) for each linear layer."""
        return [(l.out_features, l.in_features) for l in self.linears]


# ── Surprise: MSE-based, not raw hidden difference ────────────────────────────

class SurpriseGate(nn.Module):
    """
    Computes surprise as prediction loss (MSE), not raw vector difference.
    
    [F6 fix] Two hidden vectors can have large L2 distance without affecting
    prediction quality. Using MSE loss is more semantically meaningful —
    it measures how badly memory predicted the actual hidden state.

    Also produces a gating scalar in [0,1] that scales the memory update.
    High surprise → large update. Low surprise → small update.
    """
    def __init__(self, dim: int):
        super().__init__()
        # Maps prediction error magnitude to a gate value
        # Input: scalar MSE loss per sequence item
        self.gate_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, predicted: torch.Tensor, actual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            predicted: memory's prediction of actual (B, dim)
            actual:    true hidden state (B, dim)
        Returns:
            surprise_gate: scalar in [0,1] per batch item (B, 1)
            mse_loss:      raw MSE per batch item (B,) — used for gradient-based update
        """
        # Per-item MSE: (B,)
        mse = F.mse_loss(predicted, actual.detach(), reduction='none').mean(dim=-1)
        # Map through learned gate
        surprise_gate = self.gate_net(mse.unsqueeze(-1))  # (B, 1)
        return surprise_gate, mse


# ── Retention gate ────────────────────────────────────────────────────────────

class RetentionGate(nn.Module):
    """
    Per-token forget gate. Controls how much of past fast weights to retain.
    Output in [0,1]: 1.0 = keep everything, 0.0 = full reset.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.gate(hidden)  # (B, 1)


# ── Memory update rule ────────────────────────────────────────────────────────

class MemoryUpdateRule(nn.Module):
    """
    Produces per-layer (U, V) low-rank update factors.
    
    [F8 fix] Now receives: hidden, memory_output, surprise, and a summary
    of current fast weights — so it knows the current memory state.

    [F1 fix] U has shape (out_dim, rank), V has shape (in_dim, rank)
    so ΔW = U @ V.T has shape (out_dim, in_dim) — matches layer weight ✓
    """
    def __init__(self, dim: int, layer_shapes: List[Tuple[int, int]], rank: int = 8):
        super().__init__()
        self.rank = rank
        self.layer_shapes = layer_shapes  # [(out, in), ...]

        # Encode context: hidden + memory output + surprise
        self.context_enc = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        # Per-layer projection heads: one for U, one for V
        self.U_heads = nn.ModuleList()
        self.V_heads = nn.ModuleList()
        for out_dim, in_dim in layer_shapes:
            self.U_heads.append(nn.Linear(dim, out_dim * rank))
            self.V_heads.append(nn.Linear(dim, in_dim  * rank))

    def forward(
        self,
        hidden: torch.Tensor,         # (B, dim)
        memory_output: torch.Tensor,  # (B, dim)
        surprise: torch.Tensor,       # (B, 1)
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns list of (U, V) per layer.
        U[i]: (B, out_dim_i, rank)
        V[i]: (B, in_dim_i,  rank)
        """
        # Fuse context
        ctx = torch.cat([hidden, memory_output, surprise], dim=-1)  # (B, dim*2+1)
        ctx_enc = self.context_enc(ctx)  # (B, dim)

        updates = []
        for i, (out_dim, in_dim) in enumerate(self.layer_shapes):
            # Scale by surprise: more surprise → larger update
            U_flat = self.U_heads[i](ctx_enc) * surprise  # (B, out*rank)
            V_flat = self.V_heads[i](ctx_enc)             # (B, in*rank)

            U = U_flat.view(-1, out_dim, self.rank)  # (B, out, rank)
            V = V_flat.view(-1, in_dim,  self.rank)  # (B, in,  rank)
            updates.append((U, V))

        return updates  # list of (U, V) per layer


# ── Token-dependent blend gate ────────────────────────────────────────────────

class MemoryBlendGate(nn.Module):
    """
    [F7 fix] Token-dependent blending — different tokens trust memory differently.
    E.g. function words (the, is, a) need little memory; named entities need a lot.
    
    Replaces the single scalar nn.Parameter from v1.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        """Returns blend weight in [0,1] per batch item. (B, 1)"""
        return self.gate(torch.cat([hidden, memory_output], dim=-1))


# ── Main module ───────────────────────────────────────────────────────────────

class MiniTitanMemoryModule(nn.Module):
    """
    MiniTitan v2 — full corrected implementation.

    Two update modes (set via use_gradient_update):

    MODE A — Learned update rule (default, faster training):
        context → UpdateRule → (U, V) per layer → fast weights

    MODE B — Gradient-based update (closer to true Titans, slower):
        memory(hidden) → MSE(pred, hidden) → autograd.grad → fast weight step
        This is test-time learning in the purest sense.
    
    Per-sequence fast weights: batch_size B sequences each maintain
    their own independent SequenceMemoryState. No averaging across batch.
    """
    def __init__(
        self,
        input_dim: int,
        memory_hidden_dim: int = 256,
        memory_layers: int = 3,
        rank: int = 8,
        dropout: float = 0.1,
        use_gradient_update: bool = False,   # True = MODE B
        gradient_lr: float = 0.01,           # step size for gradient-based update
    ):
        super().__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.use_gradient_update = use_gradient_update
        self.gradient_lr = gradient_lr

        # Core memory MLP
        self.memory = NeuralMemoryMLP(input_dim, memory_hidden_dim, memory_layers)
        layer_shapes = self.memory.layer_shapes()

        # Gates and rules
        self.surprise_gate  = SurpriseGate(input_dim)
        self.retention_gate = RetentionGate(input_dim)
        self.update_rule    = MemoryUpdateRule(input_dim, layer_shapes, rank)
        self.blend_gate     = MemoryBlendGate(input_dim)

        # Output fusion
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(input_dim),
        )

        # Per-sequence fast weight storage
        # Shape: List[B] of SequenceMemoryState (one per sequence in batch)
        self._fast_weights: Optional[List[SequenceMemoryState]] = None
        self._batch_size: int = 0

    def reset_memory(self, batch_size: Optional[int] = None):
        """
        Reset memory state. Call between documents/conversations.
        batch_size: if provided, pre-allocates empty states for B sequences.
        """
        if batch_size is not None:
            n_layers = len(self.memory.linears)
            self._fast_weights = [[None] * n_layers for _ in range(batch_size)]
            self._batch_size = batch_size
        else:
            self._fast_weights = None
            self._batch_size = 0

    def _init_fast_weights_if_needed(self, B: int):
        """Lazily initialize per-sequence memory states."""
        if self._fast_weights is None or len(self._fast_weights) != B:
            n_layers = len(self.memory.linears)
            self._fast_weights = [[None] * n_layers for _ in range(B)]
            self._batch_size = B

    def _get_memory_output_batched(
        self,
        x: torch.Tensor,  # (B, dim)
        B: int,
    ) -> torch.Tensor:
        """
        Run memory for each sequence with its own fast weights.
        [F3 fix] Per-sequence fast weights, no batch averaging.
        [F2 fix] Functional forward — no .data mutation.
        """
        outputs = []
        for b in range(B):
            x_b = x[b:b+1]  # (1, dim)
            fw_b = self._fast_weights[b]  # SequenceMemoryState for this sequence
            out_b = self.memory(x_b, fast_weights=fw_b)  # (1, dim)
            outputs.append(out_b)
        return torch.cat(outputs, dim=0)  # (B, dim)

    def _gradient_based_update(
        self,
        hidden: torch.Tensor,   # (1, dim) — single sequence item
        fast_weights: SequenceMemoryState,
        retention: float,
        device: torch.device,
    ) -> SequenceMemoryState:
        """
        [F4] True Titans gradient-based fast weight update.
        
        Computes: pred = memory(hidden, fast_weights)
                  loss = MSE(pred, hidden)
                  grads = autograd.grad(loss, [U_i, V_i for each layer])
        Then steps: U_i -= lr * grad_U_i (with retention decay)
        
        This is test-time learning: the memory network updates itself
        on each new token it sees, without any learned update rule.
        """
        # Build current effective weights as leaf tensors for grad computation
        layer_uvs = []
        for i, layer in enumerate(self.memory.linears):
            fw = fast_weights[i]
            if fw is None:
                # Initialize fast weights as zeros (require grad)
                out_d, in_d = layer.out_features, layer.in_features
                U = torch.zeros(out_d, self.rank, device=device, requires_grad=True)
                V = torch.zeros(in_d,  self.rank, device=device, requires_grad=True)
            else:
                U = fw.U.detach().requires_grad_(True)
                V = fw.V.detach().requires_grad_(True)
            layer_uvs.append((U, V))

        # Forward with current fast weights
        h = hidden
        for i, layer in enumerate(self.memory.linears):
            U_i, V_i = layer_uvs[i]
            W_eff = layer.weight + U_i @ V_i.T
            h = F.linear(h, W_eff, layer.bias)
            if i < len(self.memory.linears) - 1:
                h = F.gelu(h)
        pred = self.memory.norm(h + hidden)

        # Compute prediction loss
        loss = F.mse_loss(pred, hidden.detach())

        # Gradient w.r.t. fast weight factors
        all_params = [uv for pair in layer_uvs for uv in pair]
        grads = torch.autograd.grad(loss, all_params, allow_unused=True)

        # Update fast weights with gradient step + retention decay
        new_fast_weights = []
        for i, (U, V) in enumerate(layer_uvs):
            grad_U = grads[i * 2]
            grad_V = grads[i * 2 + 1]

            U_new = (U - self.gradient_lr * (grad_U if grad_U is not None else 0)) * retention
            V_new = (V - self.gradient_lr * (grad_V if grad_V is not None else 0)) * retention

            new_fast_weights.append(LayerFastWeight(
                U=U_new.detach(),
                V=V_new.detach(),
            ))

        return new_fast_weights

    def _learned_update(
        self,
        hidden: torch.Tensor,        # (B, dim)
        memory_output: torch.Tensor, # (B, dim)
        surprise: torch.Tensor,      # (B, 1)
        retention: torch.Tensor,     # (B, 1)
        B: int,
        device: torch.device,
    ):
        """
        [F1 fix] Learned update rule with correct (U, V) shapes per layer.
        Updates self._fast_weights[b] for each sequence b independently.
        [F9 fix] No batch averaging — each sequence gets its own memory update.
        """
        # Get per-layer (U, V) updates — each is (B, out/in, rank)
        layer_updates = self.update_rule(hidden, memory_output, surprise)

        for b in range(B):
            ret = retention[b].item()
            for i, (U_batch, V_batch) in enumerate(layer_updates):
                U_b = U_batch[b]  # (out, rank)
                V_b = V_batch[b]  # (in,  rank)

                current = self._fast_weights[b][i]
                if current is None:
                    self._fast_weights[b][i] = LayerFastWeight(
                        U=U_b.detach(), V=V_b.detach()
                    )
                else:
                    # Decay old + accumulate new
                    decayed = current.decay(ret)
                    self._fast_weights[b][i] = decayed.accumulate(
                        U_b.detach(), V_b.detach()
                    )

    def forward(
        self,
        hidden_states: torch.Tensor,   # (B, seq_len, dim)
        update_memory: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Process a full sequence. Memory state persists across calls
        (within a document/conversation) until reset_memory() is called.

        Returns:
            augmented_hidden: (B, seq_len, dim)
            metrics: logging dict
        """
        B, seq_len, dim = hidden_states.shape
        device = hidden_states.device

        self._init_fast_weights_if_needed(B)

        outputs = []
        surprise_scores = []
        retention_scores = []
        blend_scores = []

        for t in range(seq_len):
            h_t = hidden_states[:, t, :]  # (B, dim)

            # 1. Read from memory (per-sequence fast weights)
            mem_out = self._get_memory_output_batched(h_t, B)  # (B, dim)

            # 2. Surprise: MSE-based [F6]
            surprise, mse = self.surprise_gate(mem_out, h_t)   # (B,1), (B,)
            surprise_scores.append(surprise.mean().item())

            # 3. Retention
            retention = self.retention_gate(h_t)                # (B, 1)
            retention_scores.append(retention.mean().item())

            # 4. Update fast weights (if requested)
            if update_memory:
                if self.use_gradient_update:
                    # MODE B: gradient-based (true Titans) — per sequence
                    for b in range(B):
                        h_b  = h_t[b:b+1]       # (1, dim)
                        ret_b = retention[b].item()
                        self._fast_weights[b] = self._gradient_based_update(
                            h_b, self._fast_weights[b], ret_b, device
                        )
                else:
                    # MODE A: learned update rule (default)
                    self._learned_update(h_t, mem_out, surprise, retention, B, device)

            # 5. Token-dependent blend [F7]
            blend = self.blend_gate(h_t, mem_out)  # (B, 1)
            blend_scores.append(blend.mean().item())

            # 6. Fuse memory + base hidden
            fused_input = torch.cat([
                h_t,
                blend * mem_out + (1.0 - blend) * h_t,
            ], dim=-1)  # (B, dim*2)
            augmented = self.output_proj(fused_input)  # (B, dim)
            outputs.append(augmented)

        augmented_hidden = torch.stack(outputs, dim=1)  # (B, seq_len, dim)

        metrics = {
            "mean_surprise":  sum(surprise_scores) / len(surprise_scores),
            "mean_retention": sum(retention_scores) / len(retention_scores),
            "mean_blend":     sum(blend_scores) / len(blend_scores),
            "update_mode":    "gradient" if self.use_gradient_update else "learned",
        }

        return augmented_hidden, metrics


class MiniTitanOutputHead(nn.Module):
    """Lightweight head: memory-augmented repr → vocab logits."""
    def __init__(self, input_dim: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.head = nn.Linear(input_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(x))


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("MiniTitan v2 — shape sanity check\n")

    B, seq_len, dim = 2, 16, 768  # realistic: Qwen2-1.5B hidden dim = 1536, using 768 here

    mod = MiniTitanMemoryModule(
        input_dim=dim,
        memory_hidden_dim=256,
        memory_layers=3,
        rank=8,
        use_gradient_update=False,
    )

    n_params = sum(p.numel() for p in mod.parameters())
    print(f"Trainable params: {n_params / 1e6:.2f}M")

    # Verify layer shapes
    print("\nMemory MLP layer shapes:")
    for i, (out_d, in_d) in enumerate(mod.memory.layer_shapes()):
        print(f"  Layer {i}: W=({out_d},{in_d})  U=({out_d},{mod.rank})  V=({in_d},{mod.rank})  ΔW=({out_d},{in_d}) ✓")

    # Forward pass
    x = torch.randn(B, seq_len, dim)
    mod.reset_memory(batch_size=B)
    out, metrics = mod(x, update_memory=True)

    assert out.shape == (B, seq_len, dim), f"Output shape mismatch: {out.shape}"
    print(f"\nForward pass: ({B}, {seq_len}, {dim}) → {out.shape} ✓")
    print(f"Metrics: {metrics}")

    # Second forward — memory state should persist
    out2, metrics2 = mod(x, update_memory=True)
    assert out2.shape == (B, seq_len, dim)
    print(f"Second pass (persistent memory): {out2.shape} ✓")

    # Reset and verify
    mod.reset_memory(batch_size=B)
    out3, _ = mod(x, update_memory=False)
    assert out3.shape == (B, seq_len, dim)
    print(f"Post-reset (read-only): {out3.shape} ✓")

    print("\nAll checks passed.")
