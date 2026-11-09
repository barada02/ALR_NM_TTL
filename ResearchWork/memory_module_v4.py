"""
MiniTitan v4 — dense fast-weight memory, chunked + vectorized.

Fixes vs v3 (see research_archive/ARCHITECTURE_LOG.md, section 2 / v3 flaws H1-H6):

  [H1] Dense ΔW storage.        ΔW ← ρ·ΔW + Σ_t U_t V_tᵀ     (no MAX_RANK truncation, no 4-token horizon)
  [H2] Mode B retention.        ρ is a tensor produced by a gate whose bias starts "remember-by-default";
                                it participates in autograd (no .item()).
  [H3] Output projection.       Memory read lives in value space → W_o maps it to hidden space.
                                Fusion is residual:  out = h + blend · W_o(read), W_o zero-init, so a fresh
                                module is exactly the identity (safe to bolt onto a frozen LLM).
  [H4] Scale mismatch.          Keys/values are RMS-normalised; the memory MLP has no output LayerNorm and no
                                input residual, so surprise MSE is O(1) and comparable across tokens.
  [H5] ρ² decay.                Retention multiplies the dense ΔW once (one factor, not two).
  [H6] Throughput.              Chunked: read/write once per chunk of C tokens, per-sequence weights handled
                                with bmm/einsum — no Python loop over batch or tokens.

Also new in v4:
  * Mode B is faithful Titans at chunk granularity:  S ← η·S − θ·∇ℓ ;  ΔW ← ρ·ΔW + S, with ℓ = ‖M(k)−v‖²
    and (optionally) second-order meta-gradients into the outer parameters.
    Dense ΔW starts at 0 — no saddle (∂ℓ/∂ΔW ≠ 0 because the base weights are non-zero).
  * Retention is *chunk-level* (one ρ per sequence per chunk) so half-life is meaningful in chunks, not tokens.
  * Write gate sees the hidden state as well as the surprise, so it can learn "don't write on query tokens".
    An optional write_mask lets an experiment force read-only positions.
  * Mode A aux loss has an immediate term and a delayed term (re-read a chunk written `aux_lag_chunks` ago),
    which trains retention rather than just recency.

Causality: the read for token t uses the state *before* its chunk is written, so a chunk never sees itself.
The effective memory delay is therefore one chunk.
"""

import math
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class MemoryConfig:
    dim: int = 64                    # hidden size of the host model
    mem_hidden: int = 128            # hidden width of the memory MLP
    mem_layers: int = 2              # number of linear layers in the memory MLP
    rank: int = 8                    # rank of each per-token write (Mode A)
    chunk_size: int = 16             # tokens per read/write step
    mode: str = "A"                  # "A" = learned update rule, "B" = gradient update (Titans)
    tbptt_chunks: Optional[int] = 4  # detach state every N chunks; None = full BPTT
    retention_bias: float = 4.0      # σ(4)=0.982 per chunk at init (remember-by-default)
    retention_learned: bool = True   # False → ρ = σ(retention_bias) constant, gate frozen (ablation)
    shift_v: bool = False            # value comes from the NEXT token: pair (k_t → v_{t+1}) = induction-style next-token
                                     # memory. Same-position (k_t, v_t) cannot store a multi-token string. Causal: the
                                     # state a read sees only ever contains tokens at or before the reading position.
    input_norm: bool = False         # RMS-normalise the hidden state the memory *sees* (k/v/gates); the residual
                                     # stream it adds into is untouched. Needed for LLM hidden states (huge outlier dims).
    # Mode B
    inner_lr: float = 0.2            # θ scale (multiplied by learned write gate ∈ (0,1)); see stability note
    inner_clip: float = 0.1          # per-sequence, per-layer: ‖step‖ ≤ inner_clip·‖W_base‖  (0 disables)
    use_momentum: bool = True
    momentum_bias: float = 1.0       # σ(1)=0.73 at init
    second_order: bool = True        # meta-gradients through the inner update
    # v5 entity-keyed memory (research_archive/DESIGN_ENTITY_KEYS.md). key_mode="token" is the unchanged v4 path.
    key_mode: str = "token"          # "token" | "composite" (local token part + sentence-pooled context part; Mode B only)
    pool: str = "attn"               # composite context pool: "mean" | "attn" | "oracle" (gold spans via pool_mask; diagnostic only)
    read_cap: int = 64               # read context = previous + current sentence, at most this many tokens IN TOTAL
    mask_cross_pair: bool = True     # do not write the pair (last token of a sentence -> first token of the next sentence)
    sentence_close_writes: bool = True   # False is ONLY for the W_c = 0 key-path unit test (it would leak future tokens otherwise)
    key_loss_tau: float = 0.1        # temperature of the optional auxiliary key loss
    # Mode A
    aux_lag_chunks: int = 4          # delayed aux reads the oldest of the last N chunks; 0 disables
    head_init: float = 0.03          # target std of a single token's U/V entries at init

    def to_dict(self) -> dict:
        return asdict(self)


# ── Helpers ───────────────────────────────────────────────────────────────────

def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


State = List[torch.Tensor]  # one dense ΔW per memory layer, each (B, out, in)


# ── Memory network (per-sequence fast weights via bmm) ────────────────────────

class BatchedMemoryMLP(nn.Module):
    """
    y = M(x; ΔW).  Base weights are shared and meta-learned; ΔW is per-sequence and dense.
    No output norm, no input residual (v3's shortcut) — the net has to store key→value associations.
    """

    def __init__(self, dim: int, hidden: int, layers: int):
        super().__init__()
        dims = [dim] + [hidden] * (layers - 1) + [dim]
        self.linears = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(layers))

    def shapes(self) -> List[Tuple[int, int]]:
        return [(l.out_features, l.in_features) for l in self.linears]

    def forward(self, x: torch.Tensor, delta: Optional[State] = None) -> torch.Tensor:
        h = x                                                    # (B, C, in)
        n = len(self.linears)
        for i, lin in enumerate(self.linears):
            if delta is None:
                h = F.linear(h, lin.weight, lin.bias)
            else:
                W = lin.weight.unsqueeze(0) + delta[i]           # (B, out, in)
                h = torch.bmm(h, W.transpose(1, 2)) + lin.bias
            if i < n - 1:
                h = F.gelu(h)
        return h


class KeyValueProjection(nn.Module):
    """k = RMSNorm(W_k h), v = RMSNorm(W_v h). Distinct spaces, unit scale (fixes H4)."""

    def __init__(self, dim: int):
        super().__init__()
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

    def forward(self, h: torch.Tensor, h_v: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return rms_norm(self.W_k(h)), rms_norm(self.W_v(h if h_v is None else h_v))


class SurpriseGate(nn.Module):
    """
    Write strength in (0,1) per token from (hidden, surprise MSE).
    Surprise is detached: the gate is trained through what the writes do, not through the read error itself.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + 1, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, h: torch.Tensor, mse: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([h, mse.detach().unsqueeze(-1)], dim=-1)))  # (B,C,1)


class RetentionGate(nn.Module):
    """Chunk-level ρ ∈ (0,1) from the chunk's mean hidden state. Bias init → remember by default."""

    def __init__(self, dim: int, bias_init: float, learned: bool = True):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, max(dim // 4, 8)), nn.GELU(),
                                 nn.Linear(max(dim // 4, 8), 1))
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.constant_(self.net[-1].bias, bias_init)
        if not learned:                       # constant ρ = σ(bias): zero last weight + freeze everything
            nn.init.zeros_(self.net[-1].weight)
            for p in self.net.parameters():
                p.requires_grad_(False)

    def forward(self, h_mean: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(h_mean))  # (B,1)


class MemoryUpdateRule(nn.Module):
    """
    Mode A hypernetwork: per-token rank-r factors (U_t, V_t) for every memory layer.
    Context = [hidden, key, value, current read, write gate].
    Head init is small so ‖ΔW‖/‖W‖ ≈ O(0.1) after the first chunk instead of swamping the base weights.
    """

    def __init__(self, dim: int, shapes: List[Tuple[int, int]], rank: int, head_init: float):
        super().__init__()
        self.rank = rank
        self.shapes = shapes
        self.enc = nn.Sequential(nn.Linear(4 * dim + 1, dim), nn.GELU(), nn.LayerNorm(dim))
        self.U_heads = nn.ModuleList(nn.Linear(dim, o * rank) for o, _ in shapes)
        self.V_heads = nn.ModuleList(nn.Linear(dim, i * rank) for _, i in shapes)
        for head in list(self.U_heads) + list(self.V_heads):
            nn.init.normal_(head.weight, std=head_init / math.sqrt(dim))
            nn.init.zeros_(head.bias)

    def forward(self, h, k, v, read, gate) -> List[torch.Tensor]:
        """Returns one dense write ΔW_i = Σ_t g_t U_t V_tᵀ per layer, each (B, out, in)."""
        B, C, _ = h.shape
        enc = self.enc(torch.cat([h, k, v, read, gate], dim=-1))          # (B,C,dim)
        writes = []
        for i, (o, inp) in enumerate(self.shapes):
            U = self.U_heads[i](enc).view(B, C, o, self.rank) * gate.unsqueeze(-1)
            V = self.V_heads[i](enc).view(B, C, inp, self.rank)
            writes.append(torch.einsum("bcor,bcir->boi", U, V))
        return writes


class MemoryBlendGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * dim, max(dim // 2, 8)), nn.GELU(),
                                 nn.Linear(max(dim // 2, 8), 1))

    def forward(self, h: torch.Tensor, read: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([h, read], dim=-1)))


# ── Main module ───────────────────────────────────────────────────────────────

class MiniTitanMemoryModule(nn.Module):
    """
    forward(hidden (B,T,dim), write_mask=None, update_memory=True) -> (out, metrics, aux_loss)

    out        (B,T,dim)  hidden + gated memory read (identity at init)
    metrics    dict of python floats / lists, for logging
    aux_loss   0-dim tensor (Mode A only; zero in Mode B). Add as  total = task_loss + λ·aux_loss.

    State persists between calls until reset_memory(); it is detached at the start of every call, so a
    pipeline may call forward() once per segment and backward() per segment.
    """

    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        assert cfg.mode in ("A", "B")
        self.cfg = cfg
        d = cfg.dim

        self.memory = BatchedMemoryMLP(d, cfg.mem_hidden, cfg.mem_layers)
        self.kv = KeyValueProjection(d)
        self.surprise_gate = SurpriseGate(d)
        self.retention_gate = RetentionGate(d, cfg.retention_bias, cfg.retention_learned)
        self.blend_gate = MemoryBlendGate(d)
        self.W_o = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.W_o.weight)

        if cfg.mode == "A":
            self.update_rule = MemoryUpdateRule(d, self.memory.shapes(), cfg.rank, cfg.head_init)
        else:
            self.momentum_gate = nn.Linear(d, 1)
            nn.init.normal_(self.momentum_gate.weight, std=1e-3)
            nn.init.constant_(self.momentum_gate.bias, cfg.momentum_bias)

        if cfg.key_mode == "composite":
            assert cfg.mode == "B" and cfg.shift_v and cfg.input_norm, "composite keys are defined for Mode B with shift_v and input_norm"
            self.W_c = nn.Linear(d, d, bias=False)            # context part of BOTH keys; zero-init → keys start as v4's
            nn.init.zeros_(self.W_c.weight)
            if cfg.pool == "attn":
                self.pool_proj = nn.Linear(d, 64, bias=False)
                self.pool_u = nn.Linear(64, 1, bias=False)
                nn.init.normal_(self.pool_u.weight, std=0.02)
        else:
            assert cfg.key_mode == "token"

        self._delta: Optional[State] = None
        self._mom: Optional[State] = None

    # ── State management ──────────────────────────────────────────────────────

    def reset_memory(self):
        self._delta, self._mom = None, None

    def detach_state(self):
        if self._delta is not None:
            self._delta = [x.detach() for x in self._delta]
        if self._mom is not None:
            self._mom = [x.detach() for x in self._mom]

    def _ensure_state(self, B: int, ref: torch.Tensor):
        if self._delta is not None and self._delta[0].shape[0] == B:
            return
        self._delta = [ref.new_zeros(B, o, i) for o, i in self.memory.shapes()]
        self._mom = [ref.new_zeros(B, o, i) for o, i in self.memory.shapes()]

    def state_norm_ratio(self) -> float:
        """‖ΔW‖_F / ‖W‖_F averaged over layers (diagnostic)."""
        if self._delta is None:
            return 0.0
        r = [d.norm() / (l.weight.norm() * math.sqrt(d.shape[0]) + 1e-8)
             for d, l in zip(self._delta, self.memory.linears)]
        return torch.stack(r).mean().item()

    # ── Mode A chunk update ───────────────────────────────────────────────────

    def _chunk_A(self, h, k, v, m, rho, update_memory):
        read = self.memory(k, self._delta)                                   # (B,c,d)
        mse = (read - v).pow(2).mean(-1)                                     # (B,c)
        gate = self.surprise_gate(h, mse)
        if m is not None:
            gate = gate * m
        aux_imm, new_delta = None, self._delta
        if update_memory:
            writes = self.update_rule(h, k, v, read, gate)
            r = rho.unsqueeze(-1)                                            # (B,1,1)
            new_delta = [r * d + w for d, w in zip(self._delta, writes)]
        return read, mse, gate, new_delta

    # ── Mode B chunk update (Titans, chunk-level) ─────────────────────────────

    def _chunk_B(self, h, k, v, m, rho, h_mean, update_memory):
        outer_grad = torch.is_grad_enabled() and self.training
        create = outer_grad and self.cfg.second_order
        with torch.enable_grad():
            leaves = [d if d.requires_grad else d.detach().requires_grad_(True) for d in self._delta]
            read = self.memory(k, leaves)
            mse = (read - v).pow(2).mean(-1)
            gate = self.surprise_gate(h, mse)
            if m is not None:
                gate = gate * m
            if update_memory:
                inner = (gate.squeeze(-1) * self.cfg.inner_lr * mse).sum()
                # retain_graph=True is required even when create_graph=False: `read` (and the graph behind it)
                # is still needed by the outer loss. Without it first-order mode dies on backward().
                grads = torch.autograd.grad(inner, leaves, create_graph=create, retain_graph=True)
                grads = [self._clip_step(g, lin.weight) for g, lin in zip(grads, self.memory.linears)]
            else:
                grads = None
        if not outer_grad:
            read, mse, gate = read.detach(), mse.detach(), gate.detach()
        if not update_memory:
            return read, mse, gate, self._delta, self._mom

        r = rho.unsqueeze(-1)
        if self.cfg.use_momentum:
            eta = torch.sigmoid(self.momentum_gate(h_mean)).unsqueeze(-1)    # (B,1,1)
            new_mom = [eta * s - g for s, g in zip(self._mom, grads)]
        else:
            new_mom = [-g for g in grads]
        new_delta = [r * d + s for d, s in zip(self._delta, new_mom)]
        return read, mse, gate, new_delta, new_mom

    def _clip_step(self, g: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Stability safeguard for Mode B. Chunk-sum delta-rule steps diverge once the effective step
        (inner_lr · write_gate) passes ≈0.15 with momentum — and the gate is free to learn up to 1.
        Clamp each sequence's per-layer step to a fraction of the base weight norm (differentiable).
        """
        if self.cfg.inner_clip <= 0:
            return g
        gn = g.flatten(1).norm(dim=1).view(-1, 1, 1)                         # (B,1,1)
        cap = self.cfg.inner_clip * W.detach().norm()
        return g * torch.clamp(cap / (gn + 1e-8), max=1.0)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, hidden: torch.Tensor, write_mask: Optional[torch.Tensor] = None,
                update_memory: bool = True, sent_ids: Optional[torch.Tensor] = None,
                pool_mask: Optional[torch.Tensor] = None, keyloss: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        if self.cfg.key_mode == "composite":
            return self._forward_composite(hidden, write_mask, update_memory, sent_ids, pool_mask, keyloss)
        cfg = self.cfg
        B, T, d = hidden.shape
        self._ensure_state(B, hidden)
        self.detach_state()

        hn_all = rms_norm(hidden) if cfg.input_norm else hidden             # only used by shift_v
        # Optional probe capture: set `module.capture = {}` before a forward pass to receive, per position, exactly what the
        # memory used — normalised hidden h, key k, value v (with shift_v: the NEXT token's value), read, write gate, blend.
        cap = getattr(self, "capture", None)
        cap_lists = {n_: [] for n_ in ("h", "k", "v", "read", "gate", "blend")} if cap is not None else None
        outs, aux_terms = [], []
        stats = {"surprise": [], "write_gate": [], "retention": [], "blend": []}
        hist: deque = deque(maxlen=max(cfg.aux_lag_chunks, 1))
        since_detach = 0

        for s in range(0, T, cfg.chunk_size):
            h_raw = hidden[:, s:s + cfg.chunk_size]                          # (B,c,d)  residual stream
            h = rms_norm(h_raw) if cfg.input_norm else h_raw                 # what the memory sees
            m = None if write_mask is None else \
                write_mask[:, s:s + cfg.chunk_size].to(h.dtype).unsqueeze(-1)  # (B,c,1)
            if cfg.shift_v:
                idx = torch.arange(s + 1, s + 1 + h_raw.shape[1], device=hidden.device)
                k, v = self.kv(h, hn_all[:, idx.clamp(max=T - 1)])
                vm = (idx < T).to(h.dtype).view(1, -1, 1)                    # the very last token has no next token
                m = vm if m is None else m * vm
            else:
                k, v = self.kv(h)
            h_mean = h.mean(dim=1)
            rho = self.retention_gate(h_mean)                                # (B,1)

            if cfg.mode == "B":
                read, mse, gate, new_delta, new_mom = self._chunk_B(h, k, v, m, rho, h_mean, update_memory)
            else:
                read, mse, gate, new_delta = self._chunk_A(h, k, v, m, rho, update_memory)
                new_mom = self._mom
                if update_memory:
                    # immediate: did the write take?   delayed: is an older write still there?
                    terms = [self._masked_mse(self.memory(k, new_delta), v.detach(), m)]
                    if cfg.aux_lag_chunks > 0 and len(hist) > 0:
                        ko, vo, mo = hist[0]
                        terms.append(self._masked_mse(self.memory(ko, new_delta), vo, mo))
                    aux_terms.append(sum(terms) / len(terms))
                    hist.append((k, v.detach(), m))

            blend = self.blend_gate(h, read)
            outs.append(h_raw + blend * self.W_o(read))
            if cap_lists is not None:
                for n_, t_ in (("h", h), ("k", k), ("v", v), ("read", read), ("gate", gate), ("blend", blend)):
                    cap_lists[n_].append(t_.detach())

            if update_memory:
                self._delta, self._mom = new_delta, new_mom
                since_detach += 1
                if cfg.tbptt_chunks and since_detach >= cfg.tbptt_chunks:
                    self.detach_state()
                    since_detach = 0

            stats["surprise"].append(mse.detach().mean())
            stats["write_gate"].append(gate.detach().mean())
            stats["retention"].append(rho.detach().mean())
            stats["blend"].append(blend.detach().mean())

        out = torch.cat(outs, dim=1)
        aux = torch.stack(aux_terms).mean() if aux_terms else hidden.new_zeros(())
        if cap_lists is not None:
            cap.update({n_: torch.cat(t_, dim=1) for n_, t_ in cap_lists.items()})

        per_chunk = {k_: torch.stack(v_).tolist() for k_, v_ in stats.items()}
        mean_ret = sum(per_chunk["retention"]) / len(per_chunk["retention"])
        # Half-life uses the MEDIAN chunk: a learned gate may wipe on some chunk types (e.g. query chunks → ρ≈0),
        # which drags the mean far below the ρ that governs long-range retention.
        med_ret = sorted(per_chunk["retention"])[len(per_chunk["retention"]) // 2]
        metrics = {
            "mode": cfg.mode,
            "mean_surprise": sum(per_chunk["surprise"]) / len(per_chunk["surprise"]),
            "mean_write_gate": sum(per_chunk["write_gate"]) / len(per_chunk["write_gate"]),
            "mean_retention": mean_ret,
            "mean_blend": sum(per_chunk["blend"]) / len(per_chunk["blend"]),
            "retention_halflife_tokens": cfg.chunk_size * math.log(0.5) / math.log(max(min(med_ret, 1 - 1e-6), 1e-6)),
            "state_norm_ratio": self.state_norm_ratio(),
            "per_chunk": per_chunk,
        }
        return out, metrics, aux

    # ── v5: composite (entity-keyed) path ─────────────────────────────────────
    def _pool_logits(self, hn: torch.Tensor) -> torch.Tensor:
        if self.cfg.pool == "attn":
            return self.pool_u(torch.tanh(self.pool_proj(hn))).squeeze(-1)          # (B,T)
        return hn.new_zeros(hn.shape[:2])                                         # mean / oracle: uniform weights

    @staticmethod
    def _masked_pool(logits: torch.Tensor, hb: torch.Tensor, allow: torch.Tensor) -> torch.Tensor:
        """logits (B,L), hb (B,L,d), allow (B,Q,L) bool → softmax-weighted pool (B,Q,d); a row with nothing allowed → 0."""
        lg = logits.unsqueeze(1).expand(-1, allow.shape[1], -1)
        lg = lg.masked_fill(~allow, -1e4)
        w = torch.softmax(lg, dim=-1) * allow.to(hb.dtype)
        w = w / w.sum(-1, keepdim=True).clamp(min=1e-9)
        return torch.bmm(w, hb)

    @staticmethod
    def _restrict(allow: torch.Tensor, om: Optional[torch.Tensor]) -> torch.Tensor:
        """oracle: keep only gold-span tokens when the span contains any, otherwise fall back to the full span."""
        if om is None:
            return allow
        ra = allow & om.unsqueeze(1)
        return torch.where(ra.any(-1, keepdim=True), ra, allow)

    def _chunk_B2(self, hw, k_r, k_w, v_w, m_w, rho, h_mean, update_memory):
        """Mode B with TWO keys: the output read uses k_r; the inner loss / ΔW update uses the write pairs (k_w → v_w)."""
        outer_grad = torch.is_grad_enabled() and self.training
        create = outer_grad and self.cfg.second_order
        with torch.enable_grad():
            leaves = [d if d.requires_grad else d.detach().requires_grad_(True) for d in self._delta]
            read = self.memory(k_r, leaves)
            read_w = self.memory(k_w, leaves)
            mse = (read_w - v_w).pow(2).mean(-1)
            gate = self.surprise_gate(hw, mse) * m_w
            if update_memory:
                inner = (gate.squeeze(-1) * self.cfg.inner_lr * mse).sum()
                grads = torch.autograd.grad(inner, leaves, create_graph=create, retain_graph=True)
                grads = [self._clip_step(g, lin.weight) for g, lin in zip(grads, self.memory.linears)]
            else:
                grads = None
        if not outer_grad:
            read, mse, gate = read.detach(), mse.detach(), gate.detach()
        if not update_memory:
            return read, mse, gate, self._delta, self._mom
        r = rho.unsqueeze(-1)
        if self.cfg.use_momentum:
            eta = torch.sigmoid(self.momentum_gate(h_mean)).unsqueeze(-1)
            new_mom = [eta * s_ - g for s_, g in zip(self._mom, grads)]
        else:
            new_mom = [-g for g in grads]
        new_delta = [r * d + s_ for d, s_ in zip(self._delta, new_mom)]
        return read, mse, gate, new_delta, new_mom

    def _forward_composite(self, hidden, write_mask, update_memory, sent_ids, pool_mask, keyloss):
        cfg = self.cfg
        assert sent_ids is not None, "composite keys need per-position sentence ids"
        if not cfg.sentence_close_writes:
            assert float(self.W_c.weight.detach().abs().sum()) == 0.0, \
                "buffering-disabled mode is a W_c = 0 unit test only; with W_c != 0 it would leak future tokens"
        B, T, d = hidden.shape
        C = cfg.chunk_size
        dev = hidden.device
        self._ensure_state(B, hidden)
        self.detach_state()

        hn = rms_norm(hidden)
        valid = torch.ones(B, T, dtype=torch.bool, device=dev) if write_mask is None else (write_mask > 0)
        sid = sent_ids.to(dev).long()
        pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
        nS = int(sid.max().item()) + 1
        end_of = torch.full((B, nS), -1, dtype=torch.long, device=dev).scatter_reduce(1, sid, pos, "amax")
        start_of = torch.full((B, nS), T, dtype=torch.long, device=dev).scatter_reduce(1, sid, pos, "amin")
        tok_end, tok_start = end_of.gather(1, sid), start_of.gather(1, sid)
        prev_start = start_of.gather(1, (sid - 1).clamp(min=0))
        read_lo = torch.maximum(prev_start, pos - (cfg.read_cap - 1)).clamp(min=0)   # ≤ read_cap tokens in total
        valid_next = torch.cat([valid[:, 1:], torch.zeros_like(valid[:, :1])], 1)
        w_ok = valid & valid_next
        if cfg.mask_cross_pair:
            w_ok = w_ok & (pos != tok_end)
        close_at = torch.where(tok_end + 1 < T, tok_end + 1, tok_end)                 # boundary-confirming lookahead token
        write_chunk = (close_at // C) if cfg.sentence_close_writes else (pos // C)
        om = None if (pool_mask is None or cfg.pool != "oracle") else (pool_mask.to(dev) > 0)

        logits = self._pool_logits(hn)
        l_all = self.kv.W_k(hn)                                                       # local part (v4's key before RMSNorm)
        hn_next = torch.cat([hn[:, 1:], hn[:, -1:]], 1)
        v_all = rms_norm(self.kv.W_v(hn_next))                                        # next-token values (shift_v)
        back = max(cfg.read_cap, 64)                                                  # sentences are ≤ 64 tokens (segmentation cap)

        cap = getattr(self, "capture", None)
        need_full = cap is not None or keyloss is not None
        outs, kr_list, kw_rows = [], [], []
        stats = {"surprise": [], "write_gate": [], "retention": [], "blend": []}
        cap_lists = {n_: [] for n_ in ("h", "k", "v", "read", "blend")} if cap is not None else None
        written = torch.zeros(B, T, device=dev) if cap is not None else None
        gate_full = torch.zeros(B, T, 1, device=dev) if cap is not None else None
        since_detach = 0
        for ci, s0 in enumerate(range(0, T, C)):
            e0 = min(s0 + C, T)
            lo = max(0, s0 - back)
            L = e0 - lo
            jpos = torch.arange(lo, e0, device=dev)
            hb, lb, vb = hn[:, lo:e0], logits[:, lo:e0], valid[:, lo:e0]
            om_b = None if om is None else om[:, lo:e0]
            # read key: pool over [read_lo(p), p] — never a token > p
            pr = torch.arange(s0, e0, device=dev)
            allow_r = (jpos.view(1, 1, L) >= read_lo[:, s0:e0].unsqueeze(-1)) & (jpos.view(1, 1, L) <= pr.view(1, -1, 1)) & vb.unsqueeze(1)
            c_r = rms_norm(self._masked_pool(lb, hb, self._restrict(allow_r, om_b)))
            k_r = rms_norm(l_all[:, s0:e0] + self.W_c(c_r))
            # write keys: whole-sentence pool, for the tokens whose sentence closes in this chunk
            allow_w = (jpos.view(1, 1, L) >= tok_start[:, lo:e0].unsqueeze(-1)) & (jpos.view(1, 1, L) <= tok_end[:, lo:e0].unsqueeze(-1)) \
                & vb.unsqueeze(1)
            c_w = rms_norm(self._masked_pool(lb, hb, self._restrict(allow_w, om_b)))
            k_w = rms_norm(l_all[:, lo:e0] + self.W_c(c_w))
            wsel = (write_chunk[:, lo:e0] == ci) & w_ok[:, lo:e0]
            if not cfg.sentence_close_writes:
                wsel = wsel & (jpos.view(1, L) >= s0)
            m_w = wsel.to(hn.dtype).unsqueeze(-1)
            h_mean = hn[:, s0:e0].mean(dim=1)
            rho = self.retention_gate(h_mean)
            read, mse_w, gate_w, new_delta, new_mom = self._chunk_B2(hb, k_r, k_w, v_all[:, lo:e0], m_w,
                                                                     rho, h_mean, update_memory)
            blend = self.blend_gate(hn[:, s0:e0], read)
            outs.append(hidden[:, s0:e0] + blend * self.W_o(read))
            if update_memory:
                self._delta, self._mom = new_delta, new_mom
                since_detach += 1
                if cfg.tbptt_chunks and since_detach >= cfg.tbptt_chunks:
                    self.detach_state()
                    since_detach = 0
            nw = m_w.sum().clamp(min=1)
            stats["surprise"].append((mse_w.detach() * m_w.squeeze(-1)).sum() / nw)
            stats["write_gate"].append((gate_w.detach()).sum() / nw)
            stats["retention"].append(rho.detach().mean())
            stats["blend"].append(blend.detach().mean())
            if need_full:
                kr_list.append(k_r)
                bi, ti = torch.nonzero(wsel, as_tuple=True)
                kw_rows.append((bi, ti + lo, k_w[bi, ti]))
            if cap is not None:
                for n_, t_ in (("h", hn[:, s0:e0]), ("k", k_r), ("v", v_all[:, s0:e0]), ("read", read), ("blend", blend)):
                    cap_lists[n_].append(t_.detach())
                written[:, lo:e0] += wsel.to(written.dtype)
                gate_full[:, lo:e0] += gate_w.detach()

        out = torch.cat(outs, dim=1)
        aux = hidden.new_zeros(())
        if need_full:
            Kr = torch.cat(kr_list, dim=1)
            bi = torch.cat([r[0] for r in kw_rows]) if kw_rows else torch.zeros(0, dtype=torch.long, device=dev)
            ti = torch.cat([r[1] for r in kw_rows]) if kw_rows else torch.zeros(0, dtype=torch.long, device=dev)
            vals = torch.cat([r[2] for r in kw_rows]) if kw_rows else hidden.new_zeros(0, d)
            Kw = hidden.new_zeros(B, T, d).index_put((bi, ti), vals)
            if keyloss is not None and bool(keyloss["has"].any()):
                q, cands, cm, has = keyloss["q"].to(dev), keyloss["cands"].to(dev), keyloss["cmask"].to(dev), keyloss["has"].to(dev)
                kq = Kr[torch.arange(B, device=dev), q]                                        # (B,d)
                kc = Kw[torch.arange(B, device=dev).unsqueeze(1), cands]                       # (B,K,d)
                lg = F.cosine_similarity(kq.unsqueeze(1), kc, dim=-1) / cfg.key_loss_tau
                lg = lg.masked_fill(~cm, -1e4)
                ce = F.cross_entropy(lg[has], torch.zeros(int(has.sum()), dtype=torch.long, device=dev))
                aux = ce
            if cap is not None:
                cap.update({n_: torch.cat(t_, dim=1) for n_, t_ in cap_lists.items()})
                cap.update({"kw": Kw.detach(), "written": written, "gate": gate_full, "writable": w_ok.to(written.dtype)})

        per_chunk = {k_: torch.stack(v_).tolist() for k_, v_ in stats.items()}
        mean_ret = sum(per_chunk["retention"]) / len(per_chunk["retention"])
        med_ret = sorted(per_chunk["retention"])[len(per_chunk["retention"]) // 2]
        metrics = {
            "mode": "B-composite",
            "mean_surprise": sum(per_chunk["surprise"]) / len(per_chunk["surprise"]),
            "mean_write_gate": sum(per_chunk["write_gate"]) / len(per_chunk["write_gate"]),
            "mean_retention": mean_ret,
            "mean_blend": sum(per_chunk["blend"]) / len(per_chunk["blend"]),
            "retention_halflife_tokens": cfg.chunk_size * math.log(0.5) / math.log(max(min(med_ret, 1 - 1e-6), 1e-6)),
            "state_norm_ratio": self.state_norm_ratio(),
            "W_c_norm": float(self.W_c.weight.detach().norm()),
            "per_chunk": per_chunk,
        }
        return out, metrics, aux

    @staticmethod
    def _masked_mse(pred, target, m):
        e = (pred - target).pow(2).mean(-1)                                  # (B,c)
        if m is None:
            return e.mean()
        w = m.squeeze(-1)
        return (e * w).sum() / (w.sum() + 1e-6)


def sentence_ids_from_closes(close: torch.Tensor, cap: int = 64, valid: Optional[torch.Tensor] = None):
    """close (B,T) bool: True where a token ENDS a sentence. Returns (sent_ids (B,T) long, capped_share float).
    Runs longer than `cap` tokens are closed every `cap` tokens (the share of such forced closes is reported)."""
    close = close.clone()
    B, T = close.shape
    sid = torch.cumsum(close.long(), 1) - close.long()
    pos = torch.arange(T, device=close.device).unsqueeze(0).expand(B, T)
    big = torch.full_like(sid, T)
    start = big.scatter_reduce(1, sid, pos, "amin").gather(1, sid)
    forced = ((pos - start) % cap == cap - 1) & ~close
    close2 = close | forced
    sid2 = torch.cumsum(close2.long(), 1) - close2.long()
    if valid is None:
        n_sent = int(close2.sum()) + B
        return sid2, float(forced.sum()) / max(n_sent, 1)
    # share over REAL text only: padding must not count (each padding token is its own sentence upstream)
    n_sent = sum(int(torch.unique(sid2[b][valid[b]]).numel()) for b in range(B))
    return sid2, float((forced & valid).sum()) / max(n_sent, 1)


# ── Sanity checks (preserve + extend these when changing the module) ──────────

def _naive_read(mod: MiniTitanMemoryModule, x: torch.Tensor, delta: State) -> torch.Tensor:
    outs = []
    for b in range(x.shape[0]):
        h = x[b]
        for i, lin in enumerate(mod.memory.linears):
            h = F.linear(h, lin.weight + delta[i][b], lin.bias)
            if i < len(mod.memory.linears) - 1:
                h = F.gelu(h)
        outs.append(h)
    return torch.stack(outs)


def _run_checks():
    import time
    torch.manual_seed(0)
    B, T, d, C = 3, 64, 32, 8
    print("MiniTitan v4 — sanity checks\n")

    def mk(mode, **kw):
        cfg = MemoryConfig(dim=d, mem_hidden=64, chunk_size=C, mode=mode, **kw)
        return MiniTitanMemoryModule(cfg)

    # 1. batched read == naive per-sequence loop
    mod = mk("A")
    delta = [torch.randn(B, o, i) * 0.05 for o, i in mod.memory.shapes()]
    x = torch.randn(B, C, d)
    assert torch.allclose(mod.memory(x, delta), _naive_read(mod, x, delta), atol=1e-5)
    print("1. bmm read == per-sequence loop                         OK")

    # 2. identity at init (W_o = 0) — safe to attach to a frozen LLM
    for mode in "AB":
        mod = mk(mode)
        mod.reset_memory()
        h = torch.randn(B, T, d)
        out, met, aux = mod(h)
        assert torch.equal(out, h), "module must be identity at init"
    print("2. identity at init (both modes)                         OK")
    mod = mk("B", input_norm=True)
    h = torch.randn(B, T, d) * 50 + 7            # large-scale, offset input like LLM hidden states
    mod.reset_memory()
    out, met, _ = mod(h)
    assert torch.equal(out, h) and math.isfinite(met["mean_surprise"])
    print("2b. input_norm: identity at init, scale-robust            OK")

    # 3. Mode A: gradient paths. Main-loss path (future read) AND aux path.
    mod = mk("A")
    nn.init.normal_(mod.W_o.weight, std=0.1)
    mod.reset_memory()
    h = torch.randn(B, T, d, requires_grad=True)
    out, met, aux = mod(h)
    assert aux.grad_fn is not None
    (out[:, C:].pow(2).mean()).backward(retain_graph=True)
    for name, p in [("update_rule", mod.update_rule.enc[0].weight),
                    ("U_head0", mod.update_rule.U_heads[0].weight),
                    ("retention_gate", mod.retention_gate.net[0].weight),
                    ("surprise_gate", mod.surprise_gate.net[0].weight),
                    ("kv.W_k", mod.kv.W_k.weight), ("kv.W_v", mod.kv.W_v.weight)]:
        assert p.grad is not None and p.grad.abs().sum() > 0, f"Mode A: no main-loss grad to {name}"
    mod.zero_grad()
    aux.backward()
    assert mod.update_rule.U_heads[0].weight.grad.abs().sum() > 0
    print(f"3. Mode A gradient paths (main + aux)                    OK   aux={aux.item():.4f}")

    # 4. Mode B: first-step inner gradient is nonzero at ΔW=0, and meta-gradients reach outer params
    mod = mk("B")
    nn.init.normal_(mod.W_o.weight, std=0.1)
    mod.reset_memory()
    h = torch.randn(B, T, d)
    out, met, aux = mod(h)
    assert met["state_norm_ratio"] > 0, "Mode B state never moved (saddle?)"
    (out[:, C:].pow(2).mean()).backward()
    for name, p in [("retention_gate", mod.retention_gate.net[0].weight),
                    ("momentum_gate", mod.momentum_gate.weight),
                    ("surprise_gate", mod.surprise_gate.net[0].weight),
                    ("kv.W_k", mod.kv.W_k.weight), ("memory.W0", mod.memory.linears[0].weight)]:
        assert p.grad is not None and p.grad.abs().sum() > 0, f"Mode B: no meta-gradient to {name}"
    print(f"4. Mode B inner grad ≠ 0 at init, meta-grads reach outer  OK   ‖ΔW‖/‖W‖={met['state_norm_ratio']:.3f}")

    # 4b. first-order Mode B (second_order=False) must backward cleanly too. Found on Modal: the inner
    #     autograd.grad freed the graph the outer loss still needed. NB: in this mode the write gate gets no
    #     gradient (θ only enters through the detached inner step) — a property of first-order, not a bug.
    mod = mk("B", second_order=False)
    nn.init.normal_(mod.W_o.weight, std=0.1)
    mod.reset_memory()
    out, met, _ = mod(torch.randn(B, T, d))
    (out[:, C:].pow(2).mean()).backward()
    for name, p in [("retention_gate", mod.retention_gate.net[0].weight), ("W_o", mod.W_o.weight),
                    ("kv.W_k", mod.kv.W_k.weight), ("memory.W0", mod.memory.linears[0].weight)]:
        assert p.grad is not None and p.grad.abs().sum() > 0, f"first-order Mode B: no grad to {name}"
    print("4b. Mode B first-order backward + grads                  OK")

    # 5. write_mask = 0 → state never moves (both modes)
    for mode in "AB":
        mod = mk(mode)
        mod.reset_memory()
        mod(torch.randn(B, T, d), write_mask=torch.zeros(B, T))
        assert mod.state_norm_ratio() == 0.0, f"mode {mode}: masked writes leaked into state"
    print("5. write_mask=0 freezes the state (both modes)           OK")

    # 6. Dense horizon test (H1): inject ΔW, run 100+ tokens with no writes, state decays by exactly Πρ.
    mod = mk("A")
    mod.reset_memory()
    mod._ensure_state(1, torch.zeros(1))
    mod._delta = [torch.randn_like(x) * 0.05 for x in mod._delta]
    n0 = [x.norm().item() for x in mod._delta]
    steps = 100
    _, met, _ = mod(torch.randn(1, steps * C, d), write_mask=torch.zeros(1, steps * C))
    expect = math.prod(met["per_chunk"]["retention"])
    # mean over a batch of 1 → exact ρ per chunk; state is overwritten only by ρ·ΔW
    got = mod._delta[0].norm().item() / n0[0]
    assert abs(got - expect) / expect < 1e-3, (got, expect)
    print(f"6. dense horizon: {steps} chunks ({steps*C} tokens) → ‖ΔW‖ ratio {got:.3f} "
          f"== Πρ {expect:.3f}   OK  (v3 would be fully evicted after 4)")

    # 6b. fixed retention: ρ is exactly σ(bias) on every chunk and the gate has no trainable parameters
    mod = mk("B", retention_bias=8.0, retention_learned=False)
    _, met, _ = mod(torch.randn(B, T, d))
    assert all(abs(r - torch.sigmoid(torch.tensor(8.0)).item()) < 1e-5 for r in met["per_chunk"]["retention"])
    assert not any(p.requires_grad for p in mod.retention_gate.parameters())
    print("6b. fixed retention ρ = σ(bias), gate frozen             OK")

    # 6c. causality (incl. shift_v): the output at position p must not depend on hidden states at positions > p
    for mode in "AB":
        for shift in (False, True):
            mod = mk(mode, shift_v=shift, input_norm=True)
            nn.init.normal_(mod.W_o.weight, std=0.1)
            mod.eval()
            h0 = torch.randn(B, T, d)
            q = 3 * C + 2
            h1 = h0.clone()
            h1[:, q] += 5.0 * torch.randn(B, d)
            mod.reset_memory()
            o0, _, _ = mod(h0)
            mod.reset_memory()
            o1, _, _ = mod(h1)
            assert torch.allclose(o0[:, :q], o1[:, :q], atol=1e-5), f"mode {mode} shift={shift}: output leaks future tokens"
            assert not torch.allclose(o0[:, q:], o1[:, q:], atol=1e-5), "perturbation had no effect downstream"
    print("6c. causal: outputs at p independent of tokens > p        OK  (Mode A/B, shift_v off/on)")

    # 6d. probe capture: returns per-position tensors and does NOT change the output
    for mode in "AB":
        mod = mk(mode, shift_v=True, input_norm=True)
        nn.init.normal_(mod.W_o.weight, std=0.1)
        mod.eval()
        h0 = torch.randn(B, T, d)
        mod.reset_memory()
        o_ref, _, _ = mod(h0)
        mod.capture = {}
        mod.reset_memory()
        o_cap, _, _ = mod(h0)
        cap = mod.capture
        mod.capture = None
        assert torch.equal(o_ref, o_cap), "capture changed the output"
        assert all(cap[n_].shape[:2] == (B, T) for n_ in ("h", "k", "v", "read", "gate", "blend"))
    print("6d. probe capture: shapes OK, output unchanged            OK")

    # 10. v5 entity-keyed (composite) path — each check verifies a mechanism stated in DESIGN_ENTITY_KEYS.md
    Tc = 96
    close = torch.rand(B, Tc) < 0.12
    sids, _ = sentence_ids_from_closes(close, cap=64)
    ends = [(sids[b] == k_).nonzero().flatten() for b in range(B) for k_ in sids[b].unique()]
    assert any(int(e_[0]) // C != int(e_[-1]) // C for e_ in ends), "test needs a sentence that crosses a chunk boundary"
    # 10a. key path (Correction 1, scoped): W_c = 0, buffering disabled, cross mask off == token-keyed Mode B
    tokm = mk("B", shift_v=True, input_norm=True)
    nn.init.normal_(tokm.W_o.weight, std=0.1)
    comp = mk("B", shift_v=True, input_norm=True, key_mode="composite", pool="attn", sentence_close_writes=False, mask_cross_pair=False)
    missing, unexpected = comp.load_state_dict(tokm.state_dict(), strict=False)
    assert not unexpected and all(k_.startswith(("W_c", "pool_")) for k_ in missing), (missing, unexpected)
    tokm.eval(); comp.eval()
    h0 = torch.randn(B, Tc, d)
    tokm.reset_memory(); o_tok, _, _ = tokm(h0)
    comp.reset_memory(); o_cmp, _, _ = comp(h0, sent_ids=sids)
    assert torch.allclose(o_tok, o_cmp, atol=1e-5), float((o_tok - o_cmp).abs().max())
    try:                                                    # the unit-test mode must refuse W_c != 0
        nn.init.normal_(comp.W_c.weight, std=0.1); comp.reset_memory(); comp(h0, sent_ids=sids)
        raise RuntimeError("buffering-disabled mode accepted W_c != 0")
    except AssertionError:
        pass
    print("10a. v5 key path: W_c=0 + no buffering == token-keyed Mode B; refuses W_c!=0   OK")
    # 10b. causality with W_c != 0, attention pool, sentence-close writes, cross mask on
    comp2 = mk("B", shift_v=True, input_norm=True, key_mode="composite", pool="attn")
    nn.init.normal_(comp2.W_o.weight, std=0.1); nn.init.normal_(comp2.W_c.weight, std=0.1)
    comp2.eval()
    q = 3 * C + 2
    h1 = h0.clone(); h1[:, q] += 5.0 * torch.randn(B, d)
    comp2.reset_memory(); o0, _, _ = comp2(h0, sent_ids=sids)
    comp2.reset_memory(); o1, _, _ = comp2(h1, sent_ids=sids)
    assert torch.allclose(o0[:, :q], o1[:, :q], atol=1e-5), "v5 output leaks a future hidden state"
    assert not torch.allclose(o0[:, q:], o1[:, q:], atol=1e-5)
    close2 = close.clone(); close2[:, q + 1:] = ~close2[:, q + 1:]
    sids2, _ = sentence_ids_from_closes(close2, cap=64)
    comp2.reset_memory(); o2, _, _ = comp2(h0, sent_ids=sids2)
    assert torch.allclose(o0[:, :q + 1], o2[:, :q + 1], atol=1e-5), "v5 output depends on a sentence boundary after p"
    print("10b. v5 causality: outputs at p unaffected by hidden > p and by boundaries > p   OK")
    # 10c. write-once (incl. sentences crossing chunk boundaries); masked tokens never written
    comp2.capture = {}
    comp2.reset_memory(); comp2(h0, sent_ids=sids)
    cp = comp2.capture
    comp2.capture = None
    assert torch.equal(cp["written"], cp["writable"]), "a writable token was written != 1 times, or a masked one was written"
    assert cp["kw"].shape == (B, Tc, d) and cp["k"].shape == (B, Tc, d)
    print(f"10c. v5 write-once: {int(cp['written'].sum())} pairs written exactly once, masked pairs never   OK")
    # 10d. gradient reaches W_c from the task loss at W_c = 0; auxiliary key loss is differentiable
    comp3 = mk("B", shift_v=True, input_norm=True, key_mode="composite", pool="attn")
    nn.init.normal_(comp3.W_o.weight, std=0.1)
    comp3.train(); comp3.reset_memory()
    kl = {"q": torch.full((B,), Tc - 2), "cands": torch.tensor([[20, 40, 60]] * B), "cmask": torch.ones(B, 3, dtype=torch.bool),
          "has": torch.ones(B, dtype=torch.bool)}
    o3, _, aux3 = comp3(torch.randn(B, Tc, d), sent_ids=sids, keyloss=kl)
    assert aux3.grad_fn is not None and math.isfinite(float(aux3))
    (o3[:, C:].pow(2).mean() + 0.1 * aux3).backward()
    assert comp3.W_c.weight.grad is not None and comp3.W_c.weight.grad.abs().sum() > 0, "no gradient to W_c at W_c = 0"
    print(f"10d. v5 gradient reaches W_c at W_c=0; key loss differentiable ({float(aux3):.3f})   OK")
    # 10e. oracle pool = masked mean over gold tokens; falls back to the full span when the span has none
    lg = torch.zeros(1, 6); hb = torch.randn(1, 6, d)
    om_ = torch.tensor([[False, True, True, False, False, False]])
    allow = torch.ones(1, 1, 6, dtype=torch.bool)
    pooled = MiniTitanMemoryModule._masked_pool(lg, hb, MiniTitanMemoryModule._restrict(allow, om_))
    assert torch.allclose(pooled[0, 0], hb[0, 1:3].mean(0), atol=1e-6)
    allow2 = torch.zeros(1, 1, 6, dtype=torch.bool); allow2[0, 0, 3:] = True
    pooled2 = MiniTitanMemoryModule._masked_pool(lg, hb, MiniTitanMemoryModule._restrict(allow2, om_))
    assert torch.allclose(pooled2[0, 0], hb[0, 3:].mean(0), atol=1e-6)
    print("10e. v5 oracle pool: masked mean over gold tokens, full-span fallback   OK")
    # 10f. segmentation cap: no sentence longer than the cap; forced-close share reported
    sids3, share = sentence_ids_from_closes(torch.zeros(1, 200, dtype=torch.bool), cap=64)
    lens = torch.bincount(sids3[0])
    assert int(lens.max()) <= 64 and share > 0
    sids4, share4 = sentence_ids_from_closes(close, cap=64)
    print(f"10f. v5 segmentation: max sentence {int(lens.max())} tokens (cap 64); forced-close share {share:.2f} (no-punctuation run), "
          f"{share4:.2f} (normal text)   OK")

    # 7. Mode B one-shot write: re-reading identical keys after a write lowers surprise
    mod = mk("B")
    mod.eval()
    mod.reset_memory()
    h1 = torch.randn(B, C, d)
    with torch.no_grad():
        _, met, _ = mod(torch.cat([h1, h1], dim=1))
    s1, s2 = met["per_chunk"]["surprise"]
    assert s2 < 0.9 * s1, f"Mode B write did not reduce surprise: {s1:.4f} → {s2:.4f}"
    print(f"7. Mode B one-shot write: surprise {s1:.3f} → {s2:.3f}         OK")

    # 8. Stability: worst case = write gate saturated at 1, big inner_lr, momentum, junk stream, 100 chunks.
    #    Without inner_clip this diverges to NaN (verified); with it the state must stay bounded.
    mod = mk("B", inner_lr=1.0)
    nn.init.constant_(mod.surprise_gate.net[-1].bias, 20.0)
    mod.reset_memory()
    out, met, _ = mod(torch.randn(B, 100 * C, d))
    assert torch.isfinite(out).all() and met["state_norm_ratio"] < 5, met["state_norm_ratio"]
    print(f"8. Mode B worst-case stability (gate=1, lr=1, 100 chunks) OK   ratio={met['state_norm_ratio']:.3f}")

    # 9. Retention half-life probe
    mod = mk("A")
    _, met, _ = mod(torch.randn(B, T, d))
    print(f"9. retention half-life at init ≈ {met['retention_halflife_tokens']:.0f} tokens "
          f"(ρ/chunk={met['mean_retention']:.3f})")

    # Param / throughput report at both toy and LLM-scale dims
    for dim, hid in [(64, 128), (768, 256)]:
        for mode in "AB":
            cfg = MemoryConfig(dim=dim, mem_hidden=hid, mode=mode, chunk_size=16)
            mm = MiniTitanMemoryModule(cfg)
            n = sum(p.numel() for p in mm.parameters()) / 1e6
            mm.reset_memory()
            x = torch.randn(4, 128, dim)
            t0 = time.time()
            mm(x)
            dt = time.time() - t0
            print(f"   dim={dim:<4} mode={mode}: {n:6.2f}M params, "
                  f"{4*128/dt:7.0f} tok/s (CPU, B=4, T=128)")

    print("\nAll checks passed. v4 is ready.")


if __name__ == "__main__":
    _run_checks()
