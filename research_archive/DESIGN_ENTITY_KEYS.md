# Design draft — entity-keyed memory (v5 keys)

Status: **APPROVED with three corrections (applied below, marked "Correction n").** Implementation follows this document.
Motivation is measured, not intuited: E13 (storage works, addressing fails) and E14 (pooled entity keys are linearly addressable on
held-out entities at the current layer; today's key positions are not). See `EXPERIMENT_LOG.md` E12–E14.

## 1. What has to change, and what must not

| Keep (evidence) | Change (evidence) |
|---|---|
| Layer-12 insertion — pooled entity keys already work there (E14: 0.82 held-out, shared map 0.99) | **What the key is built from.** Today: the single token before the value (write) and the answer token (read). Neither carries a generalising entity code (E14: 0.33 held-out; the write position never above 0.37 at any layer) |
| Next-token values (`shift_v`) — load-bearing for exact multi-token recall (E7b: 2.18 → 1.79 NLL with it) | **When the key is written** — after the sentence is complete, because the entity can follow the value ("The register lists {v} as the … of the {ent}") |
| Mode B, dense ΔW, chunk 32, write gate, full BPTT, crop curriculum for Gemini documents | Nothing about storage: own-key storage retrieval is already 0.50–0.63 (E13) |
| One shared key projection on both sides (E14: shared map 0.99 ≥ separate 0.82) | |

## 2. The trap: no gold entity spans at inference

The E14 probe pooled over **annotated** entity mentions. A deployed memory gets none. If the design used gold spans at inference it would be an
oracle, and its numbers would be the project's third false positive (after Stage A's generic adapter and the inflated oracle gap). So the design
**does not detect entities at all — it pools**, using only punctuation and learned weights:

* **Write side:** pool over the whole sentence that contains the token. The entity is in there, before or after the value.
* **Read side:** pool over the previous sentence plus the current partial sentence, at most **64 tokens in total** (the most recent 64 of
  that span, not 64 + 64). At the answer position that is "Question: What is the … of the flow meter at Ridge 4?" + "Answer:".
* E14 says this is plausible, not certain: whole-question pooling scored 0.67 held-out against 0.82 for true entity tokens. Learned attention pooling,
  trained only through the task loss (and optionally the key loss of §6), is the candidate for closing that gap without ever being told what an entity is.
* Gold spans appear in exactly one place: the **oracle-span diagnostic arm** (§7). It is a separator, not a result.

## 3. Sentence segmentation (no annotations)

Defined on the **concatenated token stream** the memory already processes (all context segments, then the question segment, as one sequence).
The artificial 256-token segment boundaries are ignored: a sentence may start in one segment and end in the next.

* A token **closes a sentence** if its decoded text contains a newline, or ends in `.` `!` `?` **and** the next token starts with whitespace or a
  newline. The one-token lookahead keeps "4.2 ppm", "v1.2" and "€2,415.80" from splitting. It costs a one-token write delay, which stays causal (§5).
* **Length cap:** 64 tokens. A longer run (lists, logs without punctuation) is closed at the cap. The share of capped sentences is reported per
  data type, because a high share means the rule is not doing its job on that type.
* Left padding never writes (as now). A boundary table over the vocabulary is computed once from the tokenizer, so there's no per-batch string work.
* Pooling windows are capped at 64 tokens on both sides.

## 4. Keys and values (the under-specified half, decided)

Notation: h̃_t = RMSNorm(layer-12 hidden at t) — what the memory already sees.

**Correction 2 — there are TWO keys per position, and they are never interchanged.**
```
local part        ℓ_t        = W_k h̃_t                                  (the existing projection, warm-started)
write context     c^w_t      = RMSNorm( Pool(h̃ over the WHOLE sentence containing t) )        may include tokens after t
read context      c^r_p      = RMSNorm( Pool(h̃ over [previous-sentence start … p], last 64) )   ONLY tokens ≤ p
write key         k^write_t  = RMSNorm( ℓ_t + W_c c^w_t )
read key          k^read_p   = RMSNorm( ℓ_p + W_c c^r_p )
value             v_t        = RMSNorm( W_v h̃_{t+1} )                  unchanged: next-token value (shift_v)
W_c: d×d, ZERO-initialised, shared by both keys.   Pool: mean (A1) | attention α_i = softmax_i(u·W_p h̃_i), W_p: d→64, shared (A2/A3) | gold spans (D)
```
* **k^write** is used ONLY for storage: the inner Mode B loss ‖M(k^write_t) − v_t‖² and the ΔW update, together with the write gate's surprise input
  (which is computed on the write pairs).
* **k^read** is used ONLY for the output read r_p = M(k^read_p; ΔW) that is blended into the hidden stream, and for the blend gate.
* k^read is never computed from a pool that contains a token > p. Reusing a buffered k^write for the output read would leak the rest of the sentence
  into earlier positions. The causality check in §5 verifies this stated mechanism.
* **The cross-sentence pair is masked.** The last token of a sentence (usually ".") would store the first token of the NEXT sentence under the
  PREVIOUS sentence's context: one systematically wrong association per sentence. That pair is not written. (In the current token-keyed design
  the same pair is harmless, because its key has no sentence context.)

**Decision: composite key, per-token next-token values. Neither "one pooled value per sentence" nor "one sentence key for many values".**
* *One pooled value per sentence* would hand the frozen upper layers a single bag-of-sentence vector at every answer step. That loses token order,
  and we already know order matters: same-position values could not reproduce multi-token strings until `shift_v` (E7b).
* *One shared sentence key for many next-token values* would superimpose all of the sentence's next tokens under one key. The read would return
  their average: the right sentence, but no order and no exact string.
* *Composite key* splits the two jobs that E14 showed are separate. The **context part** answers "which fact" (entity binding, from pooling). The
  **local part** answers "where in the value sequence" (after "is" / "lists" → first value token; after the last generated token → the next one,
  induction-style). The local part's job is now entity-independent ("the slot before a value"), which is learnable across entities. E14's failure was
  that the local position had to carry the entity itself.
* **Warm start (Correction 1):** the new model is **parameter-warm-started from model v1 with a changed write schedule**. It does NOT start as
  model v1. With W_c = 0 both keys reduce to model v1's key, but §5 also moves WHEN pairs are written (in the chunk where their sentence closes,
  not where the token occurs). For a sentence that crosses a chunk boundary, the state trajectory differs from model v1 even at W_c = 0. So two
  variables change at once: key content and write schedule. The exactness check is therefore scoped to the key path only: "W_c = 0 with
  sentence-close buffering DISABLED reproduces model v1". The buffering-disabled mode exists only for that unit test (with W_c ≠ 0 it would leak
  future tokens), and the code refuses it unless W_c = 0. Arm R (§7) separates "new write schedule + mixture" from "new keys".

## 5. Write timing and causality

* Pairs (k^write_t → v_t) of a sentence are written to ΔW in the chunk that contains the **lookahead token** right after the sentence's last token
  (the token that confirms the boundary): all of the sentence's pairs at once, under keys that use the full-sentence pool. Open sentences are
  carried across chunk boundaries; since a sentence is at most 64 tokens, a 64-token look-back buffer per chunk holds every token it needs.
* Reads at position p use the state from before p's chunk, as now, with k^read_p. They see only sentences closed by a token in an earlier chunk.
  That is causal, and the delay is at most one sentence plus one chunk.
* The inner Mode B loss, write gate and retention are unchanged in form. Only which pairs enter a chunk's update, and their keys (k^write), change.
* Sanity checks, run on Modal before any training (each verifies a stated mechanism):
  * Causality: with W_c ≠ 0, attention pooling and sentence-close writes, outputs at p are unaffected by perturbing any hidden state > p. Also
    unaffected by changing a sentence boundary that lies after p.
  * Key path (scoped, Correction 1): W_c = 0 with buffering disabled and the cross-sentence mask off reproduces token-keyed Mode B (up to float
    summation order). The full model at W_c = 0 is NOT required to match model v1.
  * Write-once: every writable token is written exactly once, including sentences that cross chunk and segment boundaries. Masked tokens (padding,
    the cross-sentence pair, the final token) are never written.
  * Gradients reach W_c from the task loss at W_c = 0, and the oracle pool reduces to the masked mean / attention.
  * The capped-sentence share is reported per data type.

## 6. Optional auxiliary key loss (training-time only)

On programmatic colliding documents, the generator knows each fact's value position. Add a contrastive term on the composite keys:
* Pull the read key at the answer position toward the write key of the token before the asked fact's first value token.
* Push it away from the same key of the colliding facts.

It uses annotations **only as a training signal**. Inference uses none. It is an arm, not a default, so its effect is measurable.

## 7. Arms

All arms warm-start from model v1 (Stage B seed 0 checkpoint, new parameters zero-initialised). All use the Stage 2 mixture (50% Gemini, 25%
colliding, 15% overwrite, 10% single), 3000 steps at lr 3e-4, with programmatic documents uncropped. Warm-starting avoids the from-scratch recall
collapse that broke 2 of 3 Stage 2 seeds.

| Arm | Context pool | Key loss | Role |
|---|---|---|---|
| A1 | mean | – | baseline pooling |
| A2 | attention | – | **primary candidate** |
| A3 | attention | yes | does a direct key signal help? |
| D | **gold entity spans** (programmatic docs only; Gemini docs fall back to A2's pooling) | – | **diagnostic separator, never a claim** |
| R | current keys (model v1 architecture), same mixture, same warm start | – | reference: what this mixture does without the new keys |

* **Screening:** 1 seed per arm, 3000 steps. **Selection** of the arm to confirm uses the **seen** set only.
* **Extension rule (pre-registered):** if the selected arm at 3000 steps is above chance on seen ranking AND its seen ranking was still rising
  between the 2000- and 3000-step probes, it is extended to 6000 steps before any gate is declared failed. A negative at 3000 steps is otherwise
  ambiguous between "design wrong" and "undertrained".
* **Confirmation:** 3 seeds of the selected arm (at the length the extension rule fixes), judged on the gates in §9.
* **Arm R is kept.** It is the only arm that separates "warm start + new mixture" from "new keys". Stage 2's at-chance result came from training
  from scratch.
* **Arm D's scope (Correction 3):** D is an oracle only on programmatic documents. Its read ranking and every decision that uses it are computed on
  the programmatic sets only (`eval_*_colliding`, overwrite, haystack, P1/P2). On Gemini documents D is just A2 and is not reported as D.
* **Reading arm D (programmatic documents only):**
  * D passes and A1/A2 do not → pooling is the remaining problem.
  * D also fails end-to-end → something downstream of the key is broken. Stop and investigate before building more pooling.
* **Named fallback, not an arm now:** a 32-token causal read window, used only if A1/A2 fail and D passes (decision rule c).
* **Abort rule:** if the Gemini document-specific gain (wrong-document NLL minus right-document NLL, long-context) falls below 1.0 at step 1000,
  the run is recorded as a recall collapse and stopped.

## 8. Paraphrase / pronoun set (defined now, before implementation)

Programmatic, exact ground truth, built from the **held-out** pools (and seen pools, for reference). Colliding documents with k ∈ {2, 4}.
300 per subset per pool.

* **P1 — reordered / partial mentions:**
  * The fact says "the flow meter at Ridge 4" and the question says "the Ridge 4 flow meter" or "Ridge 4's flow meter", and the reverse.
  * Or the question drops the device word ("the unit at Ridge 4").
  * The names still overlap, but order and function words differ.
* **P2 — pronoun in the value sentence:** "The flow meter at Ridge 4 was moved last week. **Its** serial number is X." The entity sits in the
  *previous* sentence, so a sentence-pooled write key does not contain it.
  * Predicted: weaker than P1, because the write pool is single-sentence by design.
  * This subset sets the **scope of the claim**: binding of entity *mentions* versus binding through *reference*.
* Template paraphrase is a lower bar than free paraphrase. A Fireworks-paraphrased version (LLM rewrite of these documents, with the programmatic
  skeleton keeping the answer exact) comes only after the gates in §9 pass.

## 9. Pre-registered test (to be copied into EXPERIMENT_LOG before any training run)

Thresholds are set against the storage ceiling measured in E13 (own-key storage 0.50–0.63). End-to-end read ranking cannot credibly be required to
reach probe-level numbers (0.82).

**Primary gates** (selected arm, mean of 3 seeds):
* Held-out memory read ranking ≥ 0.50, with the Wilson lower bound > 0.29 (chance). This is E13's READ metric: the actual read at the answer position
  against the values written for each colliding fact.
* Seen memory read ranking ≥ 0.60.
* Gemini long-context NLL ≤ 1.15 (model v1 1.10 + 0.05), on the same 300-sample validation split.

**Secondary gates:**
* Output-level likelihood ranking (E12's metric, after the frozen upper layers) held-out ≥ 0.45. Without this, the improvement never reaches the
  model's answers.
* P1 held-out read ranking ≥ 0.45.
* P2 read ranking with the Wilson lower bound above chance. If not, the claim is restricted to same-sentence (mention-level) binding, and the paper
  says so.

**Always reported:** E13's storage / key / hidden metrics for every arm, with Wilson intervals and per-k sample sizes; the overwrite and haystack
ranking; the capped-sentence share; ‖W_c‖ over training (a sign of whether the context path is used at all).

**Decision rules:**
* (a) All primary gates pass → the Fireworks paraphrase set, then the held-out generator family, then RULER / BABILong in two tiers, then 1.5B.
* (b) Seen passes, held-out fails → template / entity fitting again; diversify before any scale-up.
* (c) Arm D passes on the programmatic sets, the pooled arms fail → improve pooling (span definition, attention pool; then the named 32-token
  window fallback) before anything else.
* (d) Arm D fails on the programmatic sets → downstream of the key (readout, or W_o / upper layers). Stop building pooling.
* Before any gate is declared failed, apply the §7 extension rule.
* (e) Gemini gate fails → reduce the programmatic share or lower the lr. The binding result is reported, but not as a free improvement.

## 10. Cost and implementation outline (after review only)

* **Compute:** 5 screening runs plus 2 extra confirmation seeds, about 25 min each on an A10G, plus eval-only probes. Similar to Stage 2.
  Programmatic paraphrase sets are free; Fireworks is used only after (a).
* **Code:**
  * MemoryConfig gains `key_mode = token | composite`, `pool = mean | attn | oracle`, `read_pool = prev_sentence`.
  * The hook receives per-position sentence ids, computed from token ids with the boundary table.
  * The memory buffers open sentences across chunks.
  * The probe tooling (E13/E14) is reused unchanged on the composite keys.
* **Sanity checks (§5) must pass on Modal before the first training run.**

## 11. Review answers (resolved)

1. **Read pool:** sentence pool for the primary arms; total read cap 64 tokens; a 32-token causal window is a named fallback for decision rule (c),
   not an arm now.
2. **Arm R:** kept (it separates training regime from keys).
3. **Steps:** 3000 to screen, not to conclude; the pre-registered extension rule in §7 moves the selected arm to 6000 before any gate fails.
   ‖W_c‖ is the early signal for whether the context path is used.
