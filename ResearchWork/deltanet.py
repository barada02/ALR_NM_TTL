"""
Gated DeltaNet baseline (Yang et al. 2024) for the synthetic testbed — the state-matched external competitor.

    S_t = α_t · S_{t-1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ        S ∈ R^{dv×dk}   (the "delta rule" with a scalar decay gate)
    o_t = S_t q_t                                                 (read AFTER the write — the standard formulation)

k, q are L2-normalised; β_t = σ(w_β·x_t) ∈ (0,1) is the write strength, α_t = σ(w_α·x_t + b) ∈ (0,1) the decay gate
(bias b = 4 → remember by default). dk = dv = 128 gives a fast state of 128×128 = 16,384 floats = the same as the 2-layer
MLP memory at dim 64, and about the same parameter count (~33K), so it is matched on BOTH axes.

`read_first=True` (variant "gdn_rf") reads the state BEFORE writing token t — the causal order our memory module uses
(the current token's own key/value never answers its own query). Reported separately because in our tasks a query token
carries an empty value slot, so with read-after-write the query token first overwrites the stored value with junk
unless β_t learns to switch off on query tokens.

This is a plain sequential PyTorch loop (exact, no chunking, hence no within-chunk order loss) — slow but faithful.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDeltaNet(nn.Module):
    def __init__(self, dim: int, d_state: int = 128, read_first: bool = False, decay_bias: float = 4.0):
        super().__init__()
        self.d_state, self.read_first = d_state, read_first
        self.q = nn.Linear(dim, d_state, bias=False)
        self.k = nn.Linear(dim, d_state, bias=False)
        self.v = nn.Linear(dim, d_state, bias=False)
        self.beta = nn.Linear(dim, 1)
        self.alpha = nn.Linear(dim, 1)
        nn.init.constant_(self.alpha.bias, decay_bias)
        self.o = nn.Linear(d_state, dim, bias=False)
        self.state_floats = d_state * d_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B,T,dim) → x + W_o(o)   (residual, like the memory module)."""
        B, T, _ = x.shape
        q = F.normalize(self.q(x), dim=-1)
        k = F.normalize(self.k(x), dim=-1)
        v = self.v(x)
        beta = torch.sigmoid(self.beta(x))                       # (B,T,1)
        alpha = torch.sigmoid(self.alpha(x))                     # (B,T,1)
        S = x.new_zeros(B, self.d_state, self.d_state)           # (B, dv, dk)
        outs = []
        for t in range(T):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]               # (B,d)
            if self.read_first:
                outs.append(torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1))
            Sk = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)      # S k  (B,dv)
            b, a = beta[:, t].unsqueeze(-1), alpha[:, t].unsqueeze(-1)       # (B,1,1)
            S = a * (S - b * Sk.unsqueeze(-1) * kt.unsqueeze(1)) + b * vt.unsqueeze(-1) * kt.unsqueeze(1)
            if not self.read_first:
                outs.append(torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1))
        return x + self.o(torch.stack(outs, dim=1))


def _check():
    torch.manual_seed(0)
    d, ds = 16, 32
    for rf in (False, True):
        m = GatedDeltaNet(d, ds, read_first=rf)
        # 1. exact overwrite: with β=1, α=1 and unit keys the delta rule replaces the stored value for the SAME key
        k = F.normalize(torch.randn(1, ds), dim=-1)
        v1, v2 = torch.randn(1, ds), torch.randn(1, ds)
        S = torch.zeros(1, ds, ds)
        for v in (v1, v2):
            Sk = torch.bmm(S, k.unsqueeze(-1)).squeeze(-1)
            S = S - Sk.unsqueeze(-1) * k.unsqueeze(1) + v.unsqueeze(-1) * k.unsqueeze(1)
        assert torch.allclose(torch.bmm(S, k.unsqueeze(-1)).squeeze(-1), v2, atol=1e-5), "delta rule must overwrite exactly"
        # 2. causality: perturbing token q must not change outputs before q
        x0 = torch.randn(2, 20, d)
        x1 = x0.clone()
        x1[:, 11] += 3 * torch.randn(2, d)
        o0, o1 = m(x0), m(x1)
        assert torch.allclose(o0[:, :11], o1[:, :11], atol=1e-6), f"read_first={rf}: leaks future tokens"
        assert not torch.allclose(o0[:, 11:], o1[:, 11:], atol=1e-6)
        # 3. gradient reaches every parameter
        m.zero_grad()
        m(x0).pow(2).mean().backward()
        for n, p in m.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, f"no gradient to {n}"
    n = sum(p.numel() for p in GatedDeltaNet(64, 128).parameters())
    print(f"gated DeltaNet: exact delta-rule overwrite OK · causal OK (both read orders) · gradients reach all params · "
          f"{n:,} params, state {128 * 128:,} floats at dim 64 / d_state 128")


if __name__ == "__main__":
    _check()
