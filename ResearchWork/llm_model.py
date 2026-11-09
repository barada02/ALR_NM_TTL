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
from memory_module_v4 import MemoryConfig, MiniTitanMemoryModule, sentence_ids_from_closes


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
            shift_v=cfg.shift_v, key_mode=cfg.key_mode, pool=cfg.pool, read_cap=cfg.read_cap,
            mask_cross_pair=cfg.mask_cross_pair,
            ablate_key_part="local" if cfg.context_only_key else cfg.ablate_key_part,
            force_gate=cfg.force_gate, force_gate_value=cfg.force_gate_value)).float()
        self.last_seg_share = 0.0
        if cfg.key_mode == "composite":
            self._build_boundary_tables(lm.config.vocab_size)

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

    def _build_boundary_tables(self, vocab_size: int):
        """Sentence segmentation with NO annotations (DESIGN_ENTITY_KEYS.md §3): a token closes a sentence if its text contains a
        newline, or ends in . ! ? and the NEXT token starts with whitespace / newline (keeps "4.2", "v1.2" whole)."""
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.cfg.base_model)
        V = max(vocab_size, len(tok))
        is_end = torch.zeros(V, dtype=torch.bool)
        is_nl = torch.zeros(V, dtype=torch.bool)
        starts_ws = torch.zeros(V, dtype=torch.bool)
        for i in range(len(tok)):
            t = tok.decode([i])
            if not t:
                continue
            is_nl[i] = "\n" in t
            st = t.rstrip(" \t")
            is_end[i] = len(st) > 0 and st[-1] in ".!?"
            starts_ws[i] = t[0] in " \n\t"
        self.register_buffer("_b_end", is_end, persistent=False)
        self.register_buffer("_b_nl", is_nl, persistent=False)
        self.register_buffer("_b_ws", starts_ws, persistent=False)

    def sentence_ids(self, flat_ids: torch.Tensor, valid: torch.Tensor = None):
        nxt = torch.cat([flat_ids[:, 1:], flat_ids[:, -1:]], 1)
        close = self._b_nl[flat_ids] | (self._b_end[flat_ids] & self._b_ws[nxt])
        close[:, -1] = close[:, -1] | self._b_end[flat_ids[:, -1]]
        if valid is not None:
            # padding (left of the context, right of the question) is never part of a real sentence: every padding token
            # is its own sentence, and the token before real text starts closes the padding run
            close = close | ~valid
            close[:, :-1] = close[:, :-1] | (~valid[:, :-1] & valid[:, 1:])
        return sentence_ids_from_closes(close, cap=64, valid=valid)

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
        kw = {}
        if "sent_ids" in ctx:
            kw = {"sent_ids": ctx["sent_ids"], "pool_mask": ctx.get("pool_mask"), "keyloss": ctx.get("keyloss")}
        out, met, aux = self.memory(x, write_mask=write, **kw)
        self.last_met, self.last_aux = met, aux
        out = out.to(h.dtype).reshape(BN, S, d)
        return (out,) + tuple(output[1:]) if is_tuple else out

    # ── segmented forward ─────────────────────────────────────────────────────

    def question_hidden(self, ids: torch.Tensor, write: torch.Tensor, mode: str, extra=None) -> torch.Tensor:
        """ids (B, n+1, S) → final-layer hidden of the question segment (B, S, d).
        extra (composite keys only): the batch dict, for the oracle pool mask and the auxiliary key-loss positions."""
        B, n1, S = ids.shape
        flat = ids.reshape(B * n1, S)
        pos = torch.arange(S, device=ids.device).unsqueeze(0).expand(B * n1, S)   # each segment restarts at 0
        self._ctx = {"mode": mode, "B": B, "write": write}
        if self.cfg.key_mode == "composite":
            sid, share = self.sentence_ids(ids.reshape(B, n1 * S), valid=(write > 0))
            self.last_seg_share = share
            self._ctx["sent_ids"] = sid
            if extra is not None and "pool_mask" in extra:
                self._ctx["pool_mask"] = extra["pool_mask"]
                if mode == "on" and self.cfg.key_loss_weight > 0 and "kl_q" in extra:
                    self._ctx["keyloss"] = {"q": extra["kl_q"], "cands": extra["kl_cands"], "cmask": extra["kl_cmask"],
                                            "has": extra["kl_has"]}
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
        out = {**b, "ids": ids, "write": write}
        if "pool_mask" in b:
            pm = b["pool_mask"].clone()
            pm[:, :n * S] = pm[:, :n * S].roll(1, dims=0)
            out["pool_mask"] = pm
        for k_ in ("kl_q", "kl_cands", "kl_cmask", "kl_has"):     # the key loss never runs on a swapped document
            out.pop(k_, None)
        return out

    def answer_logp(self, b, mode: str) -> torch.Tensor:
        if mode == "swap":
            b, mode = self.swap_context(b), "on"
        h = self.question_hidden(b["ids"], b["write"], mode, extra=b)
        return self._token_logp(h, b)

    def oracle_logp(self, b, adapter: bool = False) -> torch.Tensor:
        """Full-attention reference. adapter=True also runs the memory module with writes disabled (a generic adapter with the same
        capacity as the real model but no document memory) → the adapter-matched oracle."""
        if adapter:
            B, T = b["ids"].shape
            self._ctx = {"mode": "off", "B": B, "write": torch.zeros(B, T, device=b["ids"].device)}
        try:
            h = self.lm.model(input_ids=b["ids"], attention_mask=b["mask"], use_cache=False).last_hidden_state
        finally:
            self._ctx = None
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
            hq = self.question_hidden(ids, write, mode, extra=b)
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
