"""
SegmentedMemoryLM — a frozen causal LM whose segments are processed independently, with the MiniTitan memory
hooked in as the only cross-segment channel.

How the memory is inserted (no re-implementation of transformer internals — a forward hook, so it works across
`transformers` versions):
    all (B·n) segments run through the frozen model as ONE batch of independent sequences. At the hook point the
    hidden states (B·n, S, d) are reshaped to (B, n·S, d) — i.e. the segments of a document back in order — and
    run through the memory module, whose output replaces them. Upper frozen layers then continue per segment.
      insert="mid"  : hook on the output of layer `insert_layer-1`; the frozen upper layers decode the injected info
      insert="last" : hook on the final norm; the delta goes straight into the frozen lm_head
Modes (what the question segment gets to use):
      "on"     trained memory, fed the whole document
      "off"    memory module present but starved: no token may write → empty state (control: reliance on content)
      "swap"   trained memory fed ANOTHER sample's document with this sample's question — the control that separates
               document-specific recall from a generic "answer-style" adaptation (swap ≈ none ⇒ the gain is real recall)
      "none"   frozen base only; the question segment sees nothing of the document (the honest no-memory baseline)
      oracle   (separate path) frozen base with FULL attention over context+question — an upper bound, not segmented
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_config import LLMConfig
from memory_module_v4 import MemoryConfig, MiniTitanMemoryModule


class SegmentedMemoryLM(nn.Module):
    def __init__(self, cfg: LLMConfig):
        super().__init__()
        from transformers import AutoModelForCausalLM
        assert cfg.seg_len % cfg.chunk_size == 0, "chunk_size must divide seg_len"
        self.cfg = cfg
        try:
            lm = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=torch.bfloat16)
        except TypeError:                                   # older transformers
            lm = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16)
        for p in lm.parameters():
            p.requires_grad_(False)
        lm.eval()
        self.lm = lm
        d, L = lm.config.hidden_size, lm.config.num_hidden_layers
        self.dim, self.n_layers = d, L

        self.memory = MiniTitanMemoryModule(MemoryConfig(
            dim=d, mem_hidden=cfg.mem_hidden, mem_layers=cfg.mem_layers, rank=cfg.rank, chunk_size=cfg.chunk_size,
            mode=cfg.mem_mode, tbptt_chunks=cfg.tbptt_chunks or None, inner_lr=cfg.inner_lr,
            inner_clip=cfg.inner_clip, use_momentum=cfg.use_momentum, second_order=cfg.second_order,
            retention_bias=cfg.retention_bias, retention_learned=cfg.retention_learned, input_norm=True,
            shift_v=cfg.shift_v)).float()

        if cfg.insert == "last":
            target = lm.model.norm
        else:
            k = cfg.insert_layer if cfg.insert_layer >= 0 else L // 2     # layers below the memory
            assert 1 <= k < L
            target = lm.model.layers[k - 1]
            self.insert_after = k
        target.register_forward_hook(self._hook)
        self._ctx = None
        self.last_met, self.last_aux = {}, None

    # keep the frozen LM in eval mode no matter what
    def train(self, mode: bool = True):
        super().train(mode)
        self.lm.eval()
        return self

    def memory_parameters(self):
        return self.memory.parameters()

    # ── the hook ──────────────────────────────────────────────────────────────

    def _hook(self, module, inputs, output):
        ctx = self._ctx
        if ctx is None or ctx["mode"] == "none":
            return None
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output                       # (B·n, S, d), bf16
        BN, S, d = h.shape
        B = ctx["B"]
        x = h.float().reshape(B, (BN // B) * S, d)
        write = ctx["write"] if ctx["mode"] == "on" else torch.zeros_like(ctx["write"])
        self.memory.reset_memory()
        out, met, aux = self.memory(x, write_mask=write)
        self.last_met, self.last_aux = met, aux
        out = out.to(h.dtype).reshape(BN, S, d)
        return (out,) + tuple(output[1:]) if is_tuple else out

    # ── segmented forward ─────────────────────────────────────────────────────

    def question_hidden(self, ids: torch.Tensor, write: torch.Tensor, mode: str) -> torch.Tensor:
        """ids (B, n+1, S) → final-layer hidden of the question segment (B, S, d)."""
        B, n1, S = ids.shape
        flat = ids.reshape(B * n1, S)
        pos = torch.arange(S, device=ids.device).unsqueeze(0).expand(B * n1, S)   # each segment restarts at 0
        self._ctx = {"mode": mode, "B": B, "write": write}
        try:
            h = self.lm.model(input_ids=flat, position_ids=pos, use_cache=False).last_hidden_state
        finally:
            self._ctx = None
        return h.reshape(B, n1, S, -1)[:, -1]

    def _token_logp(self, hidden: torch.Tensor, b) -> torch.Tensor:
        """hidden (B, T, d); returns (B, A) log-probs of the answer tokens (0 where invalid)."""
        pos = b["ans_pos"].to(hidden.device)
        sel = hidden.gather(1, pos.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        logp = F.log_softmax(self.lm.lm_head(sel).float(), dim=-1)
        tok = b["ans_tok"].to(hidden.device)
        return logp.gather(-1, tok.unsqueeze(-1)).squeeze(-1) * b["ans_valid"].to(hidden.device)

    @staticmethod
    def swap_context(b):
        """Roll the context segments (and their write masks) by one position along the batch: sample i keeps its
        question but gets sample i-1's document. A batch of 1 is a no-op, so evaluate with batches ≥ 2."""
        ids, write = b["ids"].clone(), b["write"].clone()
        n, S = ids.shape[1] - 1, ids.shape[2]
        ids[:, :n] = ids[:, :n].roll(1, dims=0)
        write[:, :n * S] = write[:, :n * S].roll(1, dims=0)
        return {**b, "ids": ids, "write": write}

    def answer_logp(self, b, mode: str) -> torch.Tensor:
        if mode == "swap":
            b, mode = self.swap_context(b), "on"
        h = self.question_hidden(b["ids"], b["write"], mode)
        return self._token_logp(h, b)

    def oracle_logp(self, b) -> torch.Tensor:
        h = self.lm.model(input_ids=b["ids"], attention_mask=b["mask"], use_cache=False).last_hidden_state
        return self._token_logp(h, b)

    # ── generation (greedy, autoregressive; fixes P4) ─────────────────────────

    @torch.no_grad()
    def generate(self, b, mode: str, max_new: int):
        """Segmented greedy decoding: each step re-runs the whole pipeline on the growing question segment."""
        eos = self.lm.config.eos_token_id if isinstance(self.lm.config.eos_token_id, int) else \
            (self.lm.config.eos_token_id or [None])[0]
        ids, write = b["ids"].clone(), b["write"].clone()
        B, n1, S = ids.shape
        base = (n1 - 1) * S
        cur = b["prompt_len"].clone().to(ids.device)
        done = torch.zeros(B, dtype=torch.bool, device=ids.device)
        toks = [[] for _ in range(B)]
        for _ in range(max_new):
            hq = self.question_hidden(ids, write, mode)
            last = hq[torch.arange(B, device=ids.device), (cur - 1).clamp(min=0)]
            nxt = self.lm.lm_head(last).argmax(-1)
            for i in range(B):
                if done[i]:
                    continue
                t = int(nxt[i])
                if t == eos or cur[i] >= S:
                    done[i] = True
                    continue
                toks[i].append(t)
                ids[i, -1, cur[i]] = t
                write[i, base + cur[i]] = 1
                cur[i] += 1
            if bool(done.all()):
                break
        return toks

    @torch.no_grad()
    def oracle_generate(self, ids, mask, max_new: int, pad_id: int):
        out = self.lm.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=pad_id)
        return [o[ids.shape[1]:].tolist() for o in out]
