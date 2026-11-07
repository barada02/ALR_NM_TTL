"""
Synthetic recall tasks for the memory module. Each task produces batches of the same dict format:

    typ, key_id, val_id, fil_id : (B,T) long   — embedding indices (the "null" id = last row = zero embedding)
    target, qmask               : (B,T)        — answer id and which positions are queries
    age                         : (B,T) long   — chunks between the fact being written and the query (queries only)

The host sums E_type + E_key + E_val + E_fil, so a "fact token" carries a key and a value in ONE token
(binding is done by the embedding — these tasks isolate the memory, not the host's ability to bind
adjacent tokens). A query is a key with an empty value slot. Queries always start on a chunk boundary
after every fact, so the memory state is the only route from fact to query.

typed  (Phase 0, easy):   [facts][filler][queries]
    Filler / fact / query tokens have different token types → "write on facts, freeze on filler, wipe on
    queries" is trivially learnable by a content-dependent gate.

stream (Phase 1, hard):   [facts ... facts][queries]
    There is no filler: distractor facts are indistinguishable from the facts that get queried, all tokens
    share one type (a query differs only by its empty value slot), and queries sample uniformly over ALL
    facts in the stream. So the gate cannot know what will be asked, retention must trade off against
    interference, and accuracy as a function of age (chunks since written) is read off one pass.
"""

import math
from typing import Dict, Iterator, List, Tuple

import torch

from exp_config import ExpConfig

PAIR, FILL, QUERY = 0, 1, 2
AGE_BUCKETS: List[Tuple[int, int, str]] = [(1, 1, "1"), (2, 4, "2-4"), (5, 16, "5-16"), (17, 10 ** 9, "17+")]


class Task:
    name = "base"
    n_keys = 0
    n_vals = 0
    n_filler = 0

    def train_batch(self, B: int, gen: torch.Generator, device) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def eval_items(self, B: int, gen: torch.Generator, device, n_batches: int
                   ) -> Iterator[Tuple[str, Dict[str, torch.Tensor]]]:
        raise NotImplementedError


def _pack(device, **kw):
    return {k: v.to(device) for k, v in kw.items()}


class TypedTask(Task):
    name = "typed"

    def __init__(self, cfg: ExpConfig):
        self.C = cfg.chunk_size
        self.n_keys = cfg.n_keys or 64
        self.n_vals = cfg.n_vals
        self.n_filler = cfg.n_filler_vocab
        self.n_pairs = cfg.n_pairs
        self.train_filler_max = cfg.train_filler_max
        self.eval_fillers = [int(x) for x in cfg.eval_fillers.split(",")]
        self.eval_pairs = [int(x) for x in cfg.eval_pairs.split(",")]

    def batch(self, B, n_pairs, filler_len, gen, device):
        C, NK, NV, NF = self.C, self.n_keys, self.n_vals, self.n_filler
        keys = torch.rand(B, NK, generator=gen).argsort(-1)[:, :n_pairs]            # distinct keys per sequence
        vals = torch.randint(0, NV, (B, n_pairs), generator=gen)
        n_fill = math.ceil((n_pairs + filler_len) / C) * C - n_pairs                # queries start on a chunk boundary
        fill = torch.randint(0, NF, (B, n_fill), generator=gen)
        perm = torch.rand(B, n_pairs, generator=gen).argsort(-1)                    # query order
        qkeys, qvals = keys.gather(1, perm), vals.gather(1, perm)

        T = n_pairs + n_fill + n_pairs
        typ = torch.full((B, T), FILL, dtype=torch.long)
        typ[:, :n_pairs] = PAIR
        typ[:, n_pairs + n_fill:] = QUERY
        key_id = torch.full((B, T), NK, dtype=torch.long)                           # NK = null (zero) embedding
        val_id = torch.full((B, T), NV, dtype=torch.long)
        fil_id = torch.full((B, T), NF, dtype=torch.long)
        key_id[:, :n_pairs], val_id[:, :n_pairs] = keys, vals
        fil_id[:, n_pairs:n_pairs + n_fill] = fill
        key_id[:, n_pairs + n_fill:] = qkeys
        target = torch.zeros(B, T, dtype=torch.long)
        target[:, n_pairs + n_fill:] = qvals
        age = torch.zeros(B, T, dtype=torch.long)
        age[:, n_pairs + n_fill:] = (n_pairs + n_fill) // C                         # all pairs sit in chunk 0
        return _pack(device, typ=typ, key_id=key_id, val_id=val_id, fil_id=fil_id,
                     target=target, qmask=typ == QUERY, age=age)

    def train_batch(self, B, gen, device):
        fl = int(torch.randint(0, self.train_filler_max + 1, (1,), generator=gen))
        return self.batch(B, self.n_pairs, fl, gen, device)

    def eval_items(self, B, gen, device, n_batches):
        for n_pairs in self.eval_pairs:
            for fl in self.eval_fillers:
                for _ in range(n_batches):
                    yield f"pairs={n_pairs},filler={fl}", self.batch(B, n_pairs, fl, gen, device)


class StreamTask(Task):
    name = "stream"

    def __init__(self, cfg: ExpConfig):
        self.C = cfg.chunk_size
        self.n_keys = cfg.n_keys or 1024
        self.n_vals = cfg.n_vals
        self.n_filler = 1                                    # unused (null row only)
        self.n_queries = cfg.n_queries
        self.train_min, self.train_max = cfg.train_facts_min, cfg.train_facts_max
        self.eval_facts = [int(x) for x in cfg.eval_facts.split(",")]
        assert max(self.eval_facts + [self.train_max]) + self.C <= self.n_keys, "need n_keys ≥ max facts (+ pad)"

    def batch(self, B, n_facts, gen, device):
        C, NK, NV, Q = self.C, self.n_keys, self.n_vals, self.n_queries
        n_pad = math.ceil(n_facts / C) * C                   # extra distractors so queries start on a boundary
        keys = torch.rand(B, NK, generator=gen).argsort(-1)[:, :n_pad]              # all keys distinct in a sequence
        vals = torch.randint(0, NV, (B, n_pad), generator=gen)
        qidx = torch.rand(B, n_pad, generator=gen).argsort(-1)[:, :Q]               # queries: uniform over ALL facts
        qkeys, qvals = keys.gather(1, qidx), vals.gather(1, qidx)

        T = n_pad + Q
        typ = torch.full((B, T), PAIR, dtype=torch.long)                            # one shared type: no shortcut
        key_id = torch.full((B, T), NK, dtype=torch.long)
        val_id = torch.full((B, T), NV, dtype=torch.long)
        fil_id = torch.full((B, T), self.n_filler, dtype=torch.long)
        key_id[:, :n_pad], val_id[:, :n_pad] = keys, vals
        key_id[:, n_pad:] = qkeys                                                   # query = key, empty value slot
        target = torch.zeros(B, T, dtype=torch.long)
        target[:, n_pad:] = qvals
        qmask = torch.zeros(B, T, dtype=torch.bool)
        qmask[:, n_pad:] = True
        age = torch.zeros(B, T, dtype=torch.long)
        qpos_chunk = (n_pad + torch.arange(Q)) // C                                 # chunk of each query token
        age[:, n_pad:] = qpos_chunk.unsqueeze(0) - qidx // C
        return _pack(device, typ=typ, key_id=key_id, val_id=val_id, fil_id=fil_id,
                     target=target, qmask=qmask, age=age)

    def train_batch(self, B, gen, device):
        n = int(torch.randint(self.train_min, self.train_max + 1, (1,), generator=gen))
        return self.batch(B, n, gen, device)

    def eval_items(self, B, gen, device, n_batches):
        for n in self.eval_facts:
            for _ in range(n_batches):
                yield f"facts={n}", self.batch(B, n, gen, device)


TASKS = {"typed": TypedTask, "stream": StreamTask}


def build_task(cfg: ExpConfig) -> Task:
    if cfg.task not in TASKS:
        raise ValueError(f"unknown task {cfg.task!r}; choose from {sorted(TASKS)}")
    return TASKS[cfg.task](cfg)


# ── task-generator self-check: pure data, no model (safe to run anywhere) ────

def _check():
    gen = torch.Generator().manual_seed(0)
    for task_name in TASKS:
        cfg = ExpConfig(task=task_name)
        t = build_task(cfg)
        for label, b in list(t.eval_items(4, gen, "cpu", 1)) + [("train", t.train_batch(4, gen, "cpu"))]:
            C, T = cfg.chunk_size, b["typ"].shape[1]
            q = b["qmask"]
            first_q = q.float().argmax(1)
            assert (first_q % C == 0).all(), f"{task_name}/{label}: queries must start on a chunk boundary"
            fact_pos = (b["val_id"] != t.n_vals) & ~q                              # positions carrying a value
            assert (first_q.unsqueeze(1) > torch.arange(T).unsqueeze(0))[fact_pos].all(), "fact after first query"
            assert (b["val_id"][q] == t.n_vals).all(), "query must have an empty value slot"
            assert (b["age"][q] >= 1).all(), f"{task_name}/{label}: query age must be ≥ 1 chunk"
            # every query's answer must equal the value bound to its key in the same sequence
            for i in range(4):
                bound = {int(k): int(v) for k, v in zip(b["key_id"][i][fact_pos[i]], b["val_id"][i][fact_pos[i]])}
                for k, tgt in zip(b["key_id"][i][q[i]], b["target"][i][q[i]]):
                    assert bound[int(k)] == int(tgt), "query target does not match bound value"
        print(f"{task_name}: batches well-formed, answers recoverable, queries after all facts, age ≥ 1")
    # the stream task must NOT be separable by token type
    b = StreamTask(ExpConfig(task="stream")).batch(4, 64, gen, "cpu")
    assert (b["typ"] == PAIR).all()
    print("stream: single shared token type (no type shortcut)")


if __name__ == "__main__":
    _check()
