# MiniTitan — Architecture Log & Design Rationale

> Research archive: every architectural decision, discovered flaw, and fix across module versions,
> recorded with enough technical detail to reconstruct the narrative for a paper later.
>
> Project: neural long-term memory module on top of a frozen base LLM, inspired by
> **"Titans: Learning to Memorize at Test Time"** (Behrouz et al., Google Research, 2025).
> Last updated: 2026-11-07 (v4 + Phase 0 results).

---

## 1. Research premise

**Hypothesis.** A small, trainable memory module whose *weights* update at inference time
(test-time learning, driven by a surprise signal) can give a frozen base LLM usable long-term
memory across context segments — at a parameter/compute budget several orders of magnitude
below fine-tuning the base model.

**Constraints that shape the design (also a paper-worthy angle):**
- Base LLM is frozen (Qwen2-0.5B/1.5B, 4-bit quantized). Only the memory module trains.
- Training budget: Kaggle free tier (T4×2/P100, 16 GB VRAM, 12 h sessions, checkpoint-resume).
- Local debugging budget: 2 GB GPU → all mechanism validation must run at toy scale.
- Data: ~30K synthetic samples (long-context recall, stateful dialogue, agentic tool-call
  traces), generated via Gemini batch API (`data_generation/`).

**Core loop (all versions).** Per input chunk: (1) read memory, (2) measure surprise
(prediction error), (3) update memory fast weights scaled by surprise, (4) decay old memory via
a retention gate, (5) blend memory output into the hidden stream.

---

## 2. Version history

### v1 — `ResearchWork/memory_module.py` (first prototype; superseded, kept for the record)

**Design.** Memory = 3-layer MLP. Fast weights = a single accumulated low-rank pair (U, V),
applied to the first linear layer by mutating `weight.data`. Surprise = sigmoid gate over a
projection of the raw hidden difference. Single scalar blend parameter. Batch-averaged updates.

**Flaws found (all fatal):**

| ID | Flaw | Detail | Consequence |
|----|------|--------|-------------|
| F1 | Fast-weight shape error | U, V stored as (1, r); `U.T @ V` gives (r, r), added to a (out, in) weight | Runtime crash on the 2nd token of any sequence |
| F2 | `.data` mutation | Fast-weight application bypasses autograd | No gradient to any test-time-learning component; also not thread/AMP safe |
| F3 | Batch-averaged fast weights | One shared memory state for the whole batch | Cross-sequence contamination; memory is not per-document |
| F5 | Scalar blend | Single learned scalar for memory-vs-base mixing | No token selectivity (function words vs entities treated equally) |
| F6 | Surprise on raw hidden difference | L2 distance in hidden space ≠ prediction quality | Uninformative gate signal |

**Lesson (paper material).** Applying fast weights must be *functional*
(`F.linear(x, W + ΔW)`), never by parameter mutation, or the meta-learning objective is
silently disconnected from the loss.

### v2 — `ResearchWork/memory_module_v2.py` (correct mechanics, dead training signal)

**Changes.** Per-layer, per-sequence fast weights as explicit low-rank factors
U ∈ ℝ^(out×r), V ∈ ℝ^(in×r), ΔW = UVᵀ; functional forward; MSE-based surprise; token-dependent
blend gate; two update modes introduced:
- **Mode A** — learned update rule: a hypernetwork maps (hidden, memory output, surprise) →
  per-layer (U, V).
- **Mode B** — gradient update (faithful Titans): inner-loop
  `loss = MSE(memory(x), x)`, fast weights stepped by `autograd.grad`.

**Flaws found:**

| ID | Flaw | Detail | Consequence |
|----|------|--------|-------------|
| G1 | Detached fast weights | `_learned_update` stores `U.detach()`, `V.detach()` every token | Update rule, surprise gate get **zero gradient** — the meta-learning premise has no training signal |
| G2 | Retention via `.item()` | Retention used as a Python float | Retention gate untrainable by construction |
| G3 | Autoencoding objective | Memory input = target = hₜ, with a residual: `memory(x)=norm(net(x)+x)` | Trivial solution (`net→0`) drives surprise to ~0; nothing forces memorization |
| G4 | Saddle-point init (Mode B) | Fast weights start U=V=0; ∂loss/∂U = G·V = 0 and vice versa | Gradient updates are exactly zero forever; Mode B is frozen at origin |
| G5 | ret² decay | Retention multiplies both U and V; ΔW = UVᵀ decays as ret² | Forgetting twice as aggressive as intended |
| G6 | Sequential compute | Python loop over seq_len × batch; ΔW materialized per token | Infeasible throughput for any real training run |

**Lessons (paper material).**
1. *Test-time learning modules die silently.* Every learnable gate needs an explicit,
   verified gradient path (e.g., assert `aux_loss.grad_fn is not None` in CI).
2. *Associative memory needs an asymmetric objective.* Reconstructing the input the memory just
   saw is trivially solvable; the objective must be a key→value mapping in distinct
   projected spaces (as in Titans' `ℓ = ‖M(kₜ) − vₜ‖²`).
3. *Zero-initialized bilinear factors are a saddle point* — the same reason LoRA initializes
   one factor at zero and the other randomly (Hu et al., 2021).

### v3 — `ResearchWork/memory_module_v3.py` (trainable, but capacity-broken) — superseded by v4

**Changes.**
- [V3-F2/F3] Key/value projections W_k, W_v; objective `MSE(memory(k), v)`; surprise from that
  MSE (fixes G3).
- [V3-F4/F5] Truncated BPTT with window K=32: fast weights stay attached inside the window,
  detached at boundaries → update rule / retention / surprise gates receive gradients (fixes
  G1, G2).
- [V3-F6] Auxiliary loss: after each update, re-read memory and penalize
  `MSE(memory_after(k), v)` — a direct, dense signal on the update rule.
- [V3-F1] Mode B LoRA-style init (U=0, V~N(0, 0.02²)) — escapes the saddle (fixes G4).

**Verified working:** Mode A gradient flow (aux loss has `grad_fn`; main-loss path exists via
future-token reads inside the TBPTT window); Mode B first-step gradients nonzero.

**Flaws found (open — to be fixed in v4):**

| ID | Severity | Flaw | Detail | Consequence |
|----|----------|------|--------|-------------|
| H1 | **Critical** | 4-token memory horizon (Mode A) | `accumulate()` concatenates rank-8 factors and truncates to MAX_RANK=32 → only the last ⌊32/8⌋=4 updates survive | Long-term memory structurally impossible; retention gate decays columns that get hard-evicted anyway |
| H2 | **Critical** | Mode B memory evaporates | Retention passed as `.item()` (untrained, ≈0.5 at init) and applied to both U and V → ΔW decays ≈0.25/token | Memory half-life < 1 token; no training signal ever corrects it — Mode B always measures as "memory doesn't help" |
| H3 | Moderate | Missing output projection W_o | `mem_pred` lives in value space but is blended directly with hidden-space hₜ (code comment claims a mapping that doesn't exist) | Incoherent mixture at init; output_proj must waste capacity compensating |
| H4 | Moderate | Init scale mismatch | values `v = W_v h` with std 0.02 (tiny) vs memory read exiting LayerNorm (unit scale) | Surprise MSE dominated by prediction magnitude → gate saturated early in training |
| H5 | Minor | ret² decay persists in Mode A | `decay()` scales both factors | Learnable retention can compensate, but semantics are off by a square |
| H6 | Perf | Throughput unchanged; aux loss doubles reads | Per-token × per-sequence Python loops; 2 memory forwards/token in Mode A | Blocks all realistic training; chunking required |

**Lesson (paper material).** *Low-rank storage ≠ low-rank updates.* Writing rank-r updates is
cheap and fine; **storing** memory as a truncated concatenation of factors imposes a hard
eviction horizon of MAX_RANK/r steps. The fix is asymmetric: keep low-rank *writes*, accumulate
into a **dense** state: `ΔW ← ρₜ·ΔW + Uₜ Vₜᵀ` (≈0.8 MB/layer/sequence at dim 768 — trivial).
This mirrors Titans, which treats the full weight tensor as the memory state.

### v4 — `ResearchWork/memory_module_v4.py` (dense state, chunked, vectorized) — current

**Changes (each maps to a v3 flaw):**

| Fix | Change | Notes |
|-----|--------|-------|
| H1 | Dense ΔW state: `ΔW ← ρ·ΔW + Σₜ Uₜ Vₜᵀ` per layer, per sequence, `(B, out, in)` | Writes stay low-rank, storage is dense. No eviction horizon. |
| H2 | Retention is a gate output (tensor) in both modes, no `.item()` | Bias init → remember-by-default |
| H3 | `W_o` (value→hidden) and **residual** fusion: `out = h + blend·W_o(read)`, `W_o` zero-init | Module is exactly the identity at init → safe to bolt onto a frozen LLM (also helps P2) |
| H4 | Keys/values RMS-normalised; memory MLP has no output LayerNorm and no input residual | Surprise MSE is O(1) and comparable across tokens |
| H5 | Retention multiplies ΔW once | One factor, not two |
| H6 | Chunked forward (C=16 default): one read + one write per chunk; per-sequence weights via `bmm`/`einsum` | No Python loop over batch or tokens. Read for token t uses state *before* its chunk → effective memory delay = 1 chunk |

**Design decisions made in v4 (not in the v3 flaw list):**
- **Retention is chunk-level**, not per-token. With per-token ρ=0.95 and C=16, a chunk would retain 0.44; half-life would be ~1 chunk
  regardless of the gate. Bias init +4 (σ=0.982/chunk, ~300-token half-life) instead of the roadmap's +3.
- **Mode B is Titans at chunk granularity**: `S ← η·S − θ·∇ℓ`, `ΔW ← ρ·ΔW + S`, ℓ = ‖M(k)−v‖², per-token θ from the write gate,
  optional second-order meta-gradients (`create_graph`). Dense ΔW starts at 0 and does **not** hit the v2 saddle (G4): the base
  weights are non-zero, so ∂ℓ/∂ΔW ≠ 0.
- **Write gate sees the hidden state**, not only the surprise, so it can learn "don't write on query tokens". An optional
  `write_mask` forces read-only positions for experiments that want to remove that learning problem.
- **Mode A aux loss has two terms**: immediate (did the write take?) and delayed (re-read the oldest of the last `aux_lag_chunks`
  chunks) — addresses the "aux loss optimises recency" risk noted for v3.
- **Mode A update-rule context includes the key explicitly**; heads initialised so a single token's U/V entries have std ≈ 0.03
  (‖ΔW‖/‖W‖ ≈ O(0.1) after one chunk).

**Flaws found while building v4:**

| ID | Severity | Flaw | Detail | Fix |
|----|----------|------|--------|-----|
| I1 | **Critical** | Mode B inner loop diverges | Chunk-summed delta-rule steps with momentum blow up once the effective step `inner_lr · write_gate` ≳ 0.15: state norm 190× the base weights at lr 0.5, NaN at lr 1.0 (no momentum) or lr 0.2 with a saturated gate. The write gate is free to learn up to 1, so a smaller default is not a guarantee. | Per-sequence, per-layer step clip `‖step‖ ≤ inner_clip·‖W_base‖` (default 0.1, differentiable) + `inner_lr` 0.2. Worst case (gate=1, lr=1, 100 chunks) now bounded at ≈1.4×‖W‖. Permanent sanity check. |
| I2 | Moderate | Mean retention is a misleading diagnostic | A trained gate wipes memory on some chunk types (query chunk ρ≈0) and holds on others (filler ρ≈0.98). The mean (0.65) suggested a ~26-token half-life while the real retention across filler was 0.60 over 32 chunks. | Half-life now computed from the median chunk. Always inspect per-chunk traces, not means. |
| I4 | Moderate | First-order Mode B crashed in training | `autograd.grad(..., create_graph=False)` frees the graph that the outer loss still needs ("backward through the graph a second time"). Surfaced on Modal in the `second_order=False` ablation; the sanity suite only exercised the default second-order path. | `retain_graph=True`; sanity check 4b added; the suite now also runs on Modal (`launch.py check`). |
| I3 | Moderate | Mode A parameter count at LLM scale | Update-rule heads are `dim → out·rank` and `dim → in·rank` per layer: 0.25M params at dim 64, **17.9M at dim 768** (Mode B: 2.9M). | Open. Consider a bottleneck/shared head before Phase 2; matters for Branch A vs B parameter fairness. |

**Lessons (paper material).**
1. *Inner-loop learning rate is a stability constraint, not just a hyperparameter.* A learned write gate can push the effective step
   past the divergence threshold; clip the step relative to the weight norm rather than trusting the init.
2. *Report distributions, not means, for learned gates.* Content-dependent gates make the mean unrepresentative of the regime that matters.
3. *Zero-init output projection + residual fusion* gives a module that is exactly the identity at init — a clean way to attach a trainable
   memory to a frozen model without perturbing it.

---

## 3. Pipeline-level flaws (trainer/eval — independent of module version)

Recorded from review of `ResearchWork/train_kaggle.py` (built for v1, not yet ported):

| ID | Flaw | Detail | Consequence |
|----|------|--------|-------------|
| P1 | No information bottleneck | Frozen base gets full attention over the whole (≤1024-token) context before memory sees anything | Memory is redundant by construction; with/without-memory delta ≈ 0 regardless of memory quality. **Fix: segmented processing — base attends within ~256-token segments; fast weights are the only cross-segment channel** |
| P2 | Fresh output head swamps experiment | New `Linear(hidden→vocab)` = ~233M params for Qwen2-1.5B vs ~few-M memory module | Measured learning is mostly the head, not memory. Fix: frozen `lm_head` + memory writes a residual-stream delta (MAL-style) or mid-layer injection re-running frozen top layers (MAC-style) |
| P3 | Optimizer never steps | `step_in_session` only incremented inside the grad-accum branch it gates | Silent infinite loop; a full session produces an untrained checkpoint |
| P4 | Eval decodes wrong positions | "Greedy decode" reads next-token logits at prompt positions 0–19, never generates after "Answer:" | Needle-in-haystack numbers are meaningless |
| P5 | Confounded baseline | "Without memory" = trained pipeline with emptied memory (`update_memory=False` after reset) | Distribution shift, not an ablation. Fix: frozen-base baseline through the same decode path |
| P6 | Unshuffled split | Train/val split by index over glob-ordered files | Val set = tail of last file (single data type) |
| P7 | Label boundary drift | Context length from tokenizing the prefix separately | BPE boundary merges can misalign the answer-token mask by ±1 token |

---

## 4. Design decisions & their rationale (current stance)

| Decision | Choice | Rationale | Status |
|----------|--------|-----------|--------|
| Memory substrate | MLP whose weights are the memory state | Titans result: weight-space memory > vector-state (KV/RNN) for capacity | Test vs Branch C (recurrent null hypothesis) |
| Write mechanism | Low-rank (U, V) per chunk | Tractable hypernetwork output size | Keep; store dense ΔW (H1 fix) |
| Update driver | Mode A (learned rule) vs Mode B (inner-loop gradient) | A: cheap, one forward; B: principled, Titans-faithful | **Open — Phase 1 branch battle** |
| Surprise | Learned gate over key→value MSE | Dense, semantically meaningful novelty signal | Keep; monitor saturation (H4) |
| Forgetting | Learned retention gate, exponential decay of ΔW | Titans' α_t weight-decay analogue | Fix ret² (H5); bias init +3 (remember-by-default) |
| Meta-training | TBPTT (window 32) + auxiliary write-quality loss | Full BPTT through all inner updates is memory-prohibitive | Keep; sweep window & aux weight. Risk: aux loss optimizes *recency* — consider delayed-recall aux (re-query an earlier key) to directly train retention |
| Integration | Undecided: last-layer residual delta (MAL-style) vs mid-layer injection (MAC-style) | Maps to Titans variants | **Open — Phase 2 branches I1/I2** |
| Long-context evaluation | Segmented attention bottleneck | Only honest way to attribute recall to the memory module | Committed (P1 fix) |
| Known deltas from Titans | Chunk-level (not per-token) updates and retention; Mode B writes a dense ΔW with a stability clip (not in Titans); learned-rule mode (Mode A) is our addition | Budget-driven simplifications; each is an ablation, not an assumption | Momentum implemented in v4 Mode B (`use_momentum`); on/off ablation still to run |
| State storage | Dense ΔW per layer per sequence | Fixes H1; Mode A writes stay low-rank | Decided (v4). Dense-vs-low-rank is still a Phase 1 ablation |
| Retention granularity | One ρ per sequence per chunk | Per-token ρ makes half-life ≪ 1 chunk for any sensible C | Decided (v4) |

---

## 5. Experimental plan (summary — full tree in `/ROADMAP.md`)

- **Phase 0 (v4 rebuild) — DONE, Gate 0 passed:** H1–H6 fixed; chunked+vectorized forward; MQAR testbed built (segmented-recall
  task not yet built). *Gate: beat no-memory baseline on MQAR or stop.* Results below.
- **Phase 1 (mechanism battle, synthetic, cheap) — DONE on Modal (provisional):** see "Phase 1 results" below. Planned scope: Branch A (learned rule) vs B (gradient +
  momentum) vs C (recurrent baseline, matched params). Winner-only ablations: dense vs low-rank
  storage, rank, TBPTT window, inner LR, aux-loss weight/variant. Metrics: recall vs distance,
  vs #stored pairs, length extrapolation (train 4 segments → test 16).
- **Phase 2 (frozen-LLM integration, Kaggle):** segmented pipeline; I1 vs I2 insertion; trainer
  rewrite fixing P1–P7; smoke test on Qwen2-0.5B. *Gate: beat frozen base on segmented needle
  recall at 2–4 segments.*
- **Phase 3 (scale + characterization):** Qwen2-1.5B, 30K samples, 8–16 segments (4K+ tokens
  effective), accuracy-vs-distance curves, fact-position breakdown, gate-trajectory analysis.

### Phase 0 results (2026-11-07) — Gate 0 PASSED, single seed

Testbed: `ResearchWork/mqar_testbed.py`. Layout `[pairs][filler][queries]`; a pair token is `E_key[k]+E_val[v]` (binding done by the
embedding so the test isolates the memory), queries are in a later chunk than all pairs, host has no attention/conv → the no-memory
baseline is exactly at chance. Train: 8 pairs, filler ~ U[0,48], 1500 steps, TBPTT 4 chunks, d=64, CPU. Eval: pairs ∈ {4,8,16} ×
filler ∈ {0,48,128,512} (filler 512 = 10× the training maximum). Chance = 1/32 = 0.031. Results in `results/p0_*_s0.json`.

| Host | Params (total / core) | pairs=4 | pairs=8 | pairs=16 | Notes |
|------|-----------------------|---------|---------|----------|-------|
| none | 11K / 0 | 0.02–0.04 | 0.03–0.04 | 0.03–0.04 | chance |
| GRU (Branch C) | 36K / 25K | 0.24–0.29 | 0.14–0.17 | 0.09–0.10 | flat in distance, falls with #pairs; **not parameter-matched** |
| Mode A | 263K / 253K | 1.00 | 1.00 | 0.98–0.99 | flat out to filler 512 |
| Mode B | 47K / 36K | 1.00 → 0.96 @512 | 1.00 → 0.86 @512 | 1.00 → 0.65 @512 | degrades past training distance, worse with more pairs |

**Leak check (Mode A).** The near-perfect score was verified before being believed: (1) `update_memory=False` → 0.035 (chance);
(2) pair values resampled while keys/queries fixed → accuracy 1.000 against the *new* answers; (3) per-chunk traces show write gate ≈0 on
filler and non-zero on the pair chunk, retention 0.984 on filler chunks (product 0.60 over 32 chunks), ρ≈0 on the query chunk. The
memory is the only cross-token path in the host, so the information flows through ΔW.

**Interpretation and caveats.**
- Gate 0 is passed in the sense defined (memory ≫ no-memory), and the GRU null is clearly beaten *in this testbed* — but the GRU is
  smaller and not tuned, so Gate 1 needs a parameter-matched Branch C.
- The testbed is easy in a specific way: filler, pair and query tokens are separable by token type, so "write on pairs, freeze on filler,
  wipe on queries" is trivially learnable by a content-dependent gate. Phase 1 needs a harder variant (filler/queries not separable by
  type, interleaved pairs and distractor facts) before any branch ranking is believed.
- Mode B's degradation at 512 filler is consistent with state pollution: its write gate stays ≈0.23–0.30 on filler tokens, so junk is
  written throughout. Mode A learned to close the gate. (Hypothesis, not yet tested — ablate with `oracle_write_mask` on filler.)
- One seed. No error bars. Do not rank A vs B from this table.

### Phase 1 results (2026-11-07) — Gate 1 (provisional): Mode B wins the hard task

Run on Modal (`ResearchWork/modal_app.py`, CPU containers), 3 seeds per cell, results in `results/all_p1/` and `results/abl_all/`,
aggregate with `analyze_results.py`. Two tasks (`synth_tasks.py`): **typed** = the Phase 0 task; **stream** = no filler, distractor facts
look exactly like target facts, one shared token type (a query differs only by an empty value slot), queries uniform over *all* facts,
accuracy reported by age in chunks. Stream training: 16–64 facts (1–4 chunks), 3000 steps; eval up to 512 facts (8× the training maximum).
GRUs are parameter-matched (hidden 75 ≈ Mode B core 36K; hidden 250 ≈ Mode A core 253K) but untuned. Chance = 1/32 = 0.031.

| Host | core params | typed: 8 pairs, filler 512 | typed: 16 pairs, filler 512 | stream 16 facts | stream 64 | stream 256 | stream 512 |
|------|------------|---|---|---|---|---|---|
| none | 0 | 0.04 | 0.03 | 0.04 | 0.03 | 0.03 | 0.03 |
| GRU small | 36.6K | 0.13±0.05 | 0.09±0.03 | 0.10 | 0.06 | 0.04 | 0.03 |
| GRU large | 253K | 0.18±0.01 | 0.12±0.00 | 0.13 | 0.07 | 0.04 | 0.04 |
| Mode A | 253K | 1.00±0.00 | 0.99±0.00 | 0.99±0.00 | 0.70±0.03 | 0.07±0.01 | 0.04±0.01 |
| Mode B | 36K | 0.85±0.02 | 0.63±0.02 | 1.00±0.00 | 0.99±0.01 | 0.70±0.04 | 0.43±0.01 |

Mode B on stream @512 facts, accuracy by age of the queried fact: 0.73 (1 chunk) / 0.83 (2–4) / 0.57 (5–16) / 0.23 (17+ chunks).
Mode A at 512 facts is at chance for *every* age, including age 1.

**The ranking flips between tasks.** On the typed task Mode A is perfect and Mode B degrades; on the stream task Mode B wins by a wide
margin with ~7× fewer parameters. The typed result was real but uninformative about memory quality: a content-dependent gate can learn
"write on pairs, freeze on filler" without ever solving the retention/interference trade-off. This confirms the Phase 0 caveat and is the
reason Gate 1 is decided on the stream task.

**Ablations** (all on stream unless noted, 3 seeds; table = accuracy at 64 / 256 / 512 facts):

| Question | Result |
|---|---|
| Is Mode B's second-order meta-gradient needed? | **Yes.** First-order → chance at every length (0.03 / 0.03 / 0.04 @64 / 256 / 512). Hypothesis (untested): without it nothing trains the write path (W_v, write gate, base weights' "writability"). |
| TBPTT window (B) | 1 or 2 chunks → chance; 4 → 0.99 / 0.71 / 0.43; 8 → 1.00 / 0.74 / 0.46. Needs a window at least as long as the training distances. |
| TBPTT window (A) | 1 / 2 / 4 / 8 → 0.08 / 0.38 / 0.70 / 0.86 @64, but @256: 0.04 / 0.06 / 0.07 / 0.02 — longer windows help in-distribution and hurt extrapolation. |
| Momentum (B) | No benefit: 0.99 / 0.76 / 0.51 without vs 0.99 / 0.70 / 0.44 with. |
| Inner LR (B), 0.05–0.4 | Flat (0.98–0.99 @64), except 1 of 3 seeds at 0.1 never took off (CE pinned at ln 32 = 3.47) → ~1 in 15 Mode B runs fails to start. |
| Rank (A), 2 / 4 / 8 / 16 | No effect (0.67–0.70 @64, ≤0.08 @256). Rank is not A's bottleneck. |
| Aux loss (A) | No aux → chance everywhere: A trains *only* through the aux signal. Immediate-only = immediate+delayed (0.70 @64): the delayed term adds nothing. Weight 1.0 → 0.82 @64 but still chance @256. |
| Does Mode B degrade at long distance because it writes junk during filler? (typed) | **No.** Forcing filler read-only gives 0.83 / 0.66 vs 0.85 / 0.63 at 512 filler (8 / 16 pairs). The pollution hypothesis from Phase 0 is refuted; the cause is open (candidates: retention vs. capacity, never training beyond ~4 chunks). |

**Gate 1 outcome (provisional).** Mode B is the winner (not within noise); Branch C (matched GRU) does not win, so the premise —
test-time weight updates buy something over a plain recurrent state at matched parameters — survives on synthetic data. Mode A is kept in
the repo but dropped from the LLM phase unless its length failure is fixed.

**Caveats.** Three seeds; synthetic tasks where binding is done by the embedding; GRU untuned (same step budget and learning rate as the
others); Mode A may be undertrained (TBPTT 8 reaches 0.86 @64) but fails beyond the training length for every variant tried; Mode B's
long-range accuracy (0.43 @512) is far from solved, and its age curve (0.23 at 17+ chunks) shows that retention over long distances is the
open problem for Phase 2.

**Negative results worth publishing regardless of outcome:** Branch C matching A/B (test-time
weight updates buy nothing at this scale); memory failing only past the bottleneck (attention
already solves ≤1-window recall); Mode A failing length extrapolation where Mode B succeeds.

---

## 6. Reproducibility notes

- Every Kaggle run should commit `results/<run_id>.json` back to this repo: full config, git
  SHA, metrics, gate trajectories. Analysis happens locally from those artifacts.
- Sanity checks live in each module's `__main__` block; the v3 additions (grad_fn assertion,
  LoRA-init check, nonzero-gradient check) should be preserved and extended in v4 —
  specifically add: dense-ΔW horizon test (store pair at t, recall at t+100) and a retention
  half-life probe.
- File lineage: `memory_module.py` (v1, archived) → `memory_module_v2.py` (archived) →
  `memory_module_v3.py` (superseded) → `memory_module_v4.py` (current).
  `generate_data.py` deleted 2026-07-31 (superseded by `data_generation/`).
- v4 sanity suite lives in `memory_module_v4.py::_run_checks` (`python memory_module_v4.py`): bmm-vs-loop equivalence, identity at init,
  gradient paths (Mode A main+aux, Mode B meta-gradients), `write_mask=0` freezes state, dense horizon (100 chunks, ‖ΔW‖ ratio == Πρ),
  Mode B one-shot write lowers surprise, Mode B worst-case stability, retention half-life probe. Extend, don't weaken.
- Experiment runs write `results/<run_name>.json` (config, git SHA, params, curves, eval grid) via `mqar_testbed.save_result`. Phase 0
  runs (`p0_*_s0`) were produced from the working tree before the commit that introduced them, so their `git_sha` is the parent commit.
- Experiments run on Modal only (`ResearchWork/launch.py sweep <name>`; `launch.py check` runs the sanity suite remotely). Config → `exp_config.py`,
  sweeps → `sweeps.py`, tasks → `synth_tasks.py`, aggregation → `analyze_results.py`. Each run writes `results/<sweep>/<sweep>__<tag>__s<seed>.json`
  (also stored on the Modal Volume `minititan-results`); re-launching a sweep skips finished runs. `git_sha` in a result is the commit at launch
  plus `-dirty` if `ResearchWork/` had uncommitted changes.
- Secrets: Modal credentials live in `.env` (gitignored). `launch.py` is the only code that reads it; it forwards only MODAL_TOKEN_ID /
  MODAL_TOKEN_SECRET to the Modal CLI process and never prints them. The local `.venv` holds the Modal client only for launching.

## 7. Reference anchors

- Behrouz, Zhong, Mirrokni. *Titans: Learning to Memorize at Test Time.* 2025. (memory as
  gradient-updated weights; surprise + momentum + weight-decay forgetting; MAC/MAG/MAL variants)
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* 2021. (U=0/V=random init)
- MQAR-style synthetic recall as mechanism validation (used across Titans/Mamba/DeltaNet
  literature) — basis of the Phase 0 testbed.
