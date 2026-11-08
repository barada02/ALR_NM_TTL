---
title: MiniTitan Research Roadmap
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: 2
  maxWidth: 340
---

# MiniTitan Roadmap

## Legend
- ⬜ todo · 🔄 in progress · ✅ done · ❌ killed branch
- 🖥️ local (2GB GPU / CPU) · ☁️ Kaggle GPU
- 🚪 GATE = decision point — results decide which branches survive

## Phase 0 — Core rebuild (v4) 🖥️
- Fix v3 critical findings
  - ✅ Dense ΔW storage: `ΔW ← ret·ΔW + U@Vᵀ` (kills 4-token horizon from MAX_RANK truncation)
  - ✅ Retention decay on one factor only (ret, not ret²)
  - ✅ Retention gate bias init (remember-by-default) — now chunk-level, bias +4 (σ=0.982/chunk)
  - ✅ Mode B: retention as tensor or fixed schedule (no dead `.item()` gate)
- Fix v3 moderate findings
  - ✅ Add W_o output projection (zero-init residual → module is identity at init) (value space → hidden space before blend)
  - ✅ Fix value/read scale mismatch (RMS-normed k/v, no output LayerNorm) (normalize values or drop LayerNorm on read)
- Chunked + vectorized forward
  - ✅ Process tokens in chunks (update memory per chunk, not per token)
  - ✅ Vectorize per-sequence reads across batch (einsum/bmm, no Python B-loop)
  - ✅ Mode B: chunk-level gradient step + momentum term (+ inner_clip stability safeguard)
- Synthetic testbed
  - ✅ MQAR task (ResearchWork/mqar_testbed.py): random key→value pairs early, queried late, with distractors
  - ⬜ Segmented-recall task: fact always in an earlier segment
  - ✅ Metrics: recall vs distance, recall vs #pairs stored, length extrapolation
- ✅ 🚪 GATE 0 PASSED (1 seed): none=chance 0.03, Mode A ≈1.00, Mode B ≈1.00 — ablation-verified, no leak
  - pass → Phase 1
  - fail → debug memory core before anything else

## Phase 1 — Branch battle on synthetics 🖥️/☁️
- Branch A — learned update rule (TBPTT + aux loss)
  - hypothesis: trains fast, may extrapolate poorly to longer sequences
  - ✅ typed MQAR: ≈1.00 everywhere (3 seeds). ❌ stream MQAR (hard): 0.99 @16 facts, 0.70 @64, chance @≥256 — fails past training length for every rank/aux variant
  - ⬜ segmented recall task still todo
- Branch B — gradient update (faithful Titans)
  - hypothesis: slower, principled, better length generalization
  - ✅ momentum on/off: momentum does not help (no-momentum 0.51 vs 0.44 @512 facts, 3 seeds)
  - ✅ stream MQAR (hard): 0.99 @64 facts, 0.70 @256, 0.43 @512 (3 seeds); typed: 1.00 → 0.85/0.63 @512 filler for 8/16 pairs
  - ⬜ segmented recall task still todo
- Branch C — plain recurrent baseline (GRU state, same param budget)
  - the null hypothesis: is test-time weight update buying anything?
  - ✅ param-matched (75 / 250 hidden ≈ Mode B / Mode A core): stream 0.10–0.13 @16 facts → 0.03–0.04 @512; typed 0.09–0.19. Untuned, same budget as the others.
- ✅ 🚪 GATE 1 (provisional, 3 seeds, synthetic): **Mode B wins** the hard task by a wide margin (not within noise); C does not win → premise survives
  - Mode A wins only the easy typed task, where a content-dependent gate can shortcut; it is dropped from the LLM phase unless its length failure is fixed
  - caveat: Mode A may be undertrained (TBPTT 8 lifts it to 0.86 @64) — but it still fails beyond the training length
- Winner-only ablations
  - ⬜ dense ΔW vs low-rank storage (Mode B state is dense by construction; Mode A state is dense too — only A's writes are low-rank)
  - ✅ update rank sweep (Mode A, 2/4/8/16): no effect on anything → rank is not the bottleneck
  - ✅ TBPTT sweep: B needs ≥4 chunks (1–2 → chance); A improves in-distribution with window but extrapolates worse
  - ✅ inner-LR sweep (B, 0.05–0.4): flat, except one seed at 0.1 never took off (CE stuck at ln 32)
  - ✅ aux ablation (A): no aux → chance; delayed term adds nothing over immediate; weight 1.0 better in-distribution (0.82 vs 0.70)
  - ✅ second-order vs first-order (B): first-order → chance. Second-order meta-gradients are required
  - ✅ filler-write mask (B, typed): masking junk writes does NOT fix the long-distance degradation → pollution hypothesis refuted, cause open
  - ⬜ longer training distances / curriculum for B (does the 512-fact degradation close?)

## Phase 2 — Frozen-LLM integration ☁️
- Segmented pipeline (the information bottleneck)
  - ✅ Base model attends only within current segment (256 tok; segments run as independent sequences)
  - ✅ Memory = only cross-segment channel (forward hook on one layer)
  - ✅ Map data to segments: context chunks → segments, question+answer → final segment (dialogue/agentic rendered to text)
- Insertion-point branches
  - Branch I1 — last-layer residual delta + frozen lm_head (MAL-style): ❌ no gain
  - Branch I2 — mid-layer inject, re-run remaining frozen layers (MAC-style): ✅ better NLL (layer 8/12/16 all equal)
  - ✅ 🚪 GATE 2a: mid + TBPTT 8 chunks (full BPTT: gradient norms up to 1e9, worse)
- Trainer rewrite (llm_train.py)
  - ✅ v4 module, optimizer steps, shuffled seeded split, exact answer boundary, autoregressive generation eval, honest baselines
  - ✅ eval modes none / off / swap / on / oracle; agentic preference scoring; breakdown by fact position and #segments
- ✅ 🚪 GATE 2b PASSED (provisional until Stage B): first attempt = generic adapter only (swap ≈ on). Root cause: same-position (k_t, v_t) cannot store strings →
  `shift_v` (next-token values) + crop-around-fact curriculum + window ≥16: long-context NLL 3.70 → 1.8–1.9 (3 seeds; generic adapter 2.15; oracle 0.63), full BPTT 1.42;
  wrong document → 3.6 (none); control without `shift_v` 2.18; detail recall 0.40 vs oracle 0.74
- ❌ contrastive swap loss: gameable (module learns mismatch detection), dropped
- ✅ Stage B (`final_b`, 3 seeds, 300 val/type): full BPTT / window 32 — long-context NLL 3.67 → 1.10±0.03 / 1.15±0.01 (generic adapter 2.15, oracle 0.62), wrong document 3.82;
  agentic ON 2.41 ≈ oracle 2.42, preference acc 0.91 (swap 0.81); detail recall 0.50 (oracle 0.83); no decay with distance up to 1024+ tokens
- ⬜ Next: adapter-matched oracle (fair upper bound) · Qwen2-1.5B · 16 segments · multi-fact / other-domain tests · Phase 3 write-up

## Phase 3 — Scale & characterize ☁️ multi-session
- ⬜ Qwen2-1.5B + full 30K dataset (data_generation/ output)
- ⬜ Long effective context: 8–16 segments (4K+ tokens)
- Analysis
  - ⬜ accuracy vs fact-distance curves
  - ⬜ early/middle/late position breakdown (data already annotated)
  - ⬜ surprise/retention/blend trajectories over documents
- ⬜ Checkpoint-resume across sessions (CheckpointManager)
- ⬜ Write-up: method, branch results, ablations, negative results included

## Infrastructure (parallel track)
- Kaggle ↔ repo loop
  - ⬜ experiment notebook pulls repo, runs config, commits results JSON back
  - ⬜ results/ dir: one JSON per run (config + metrics + git SHA)
  - ⬜ analyze results locally here after notebook sync
- Hygiene
  - ✅ delete generate_data.py (superseded by data_generation/)
  - ⬜ archive memory_module.py (v1) + v2 or mark deprecated
  - ⬜ one config dataclass shared by all experiments (branch = config, not code fork)

## Findings log
- v1: fast-weight shape crash; .data mutation kills autograd
- v2: all gates detached → update rule untrainable; autoencoding objective trivial
- v3 open: MAX_RANK truncation = 4-token horizon (Mode A)
- v3 open: Mode B retention untrained → ret² ≈ 0.25/token decay, memory gone in ~10 tokens
- v3 open: value-space blended into hidden-space without W_o
- pipeline: full attention over context makes memory redundant → segmented bottleneck required
- pipeline: fresh 233M output head swamps experiment → reuse frozen lm_head
- v4 (Phase 0, 1 seed, MQAR, train filler ≤48, eval up to 512): none/GRU/Mode A/Mode B = 0.03 / 0.09–0.29 / 0.98–1.00 / 0.65–1.00
- v4: Mode B diverges (NaN / 190× state norm) when effective step inner_lr·gate ≳ 0.15 with momentum → inner_clip safeguard
- v4: logged mean retention is misleading (learned gate wipes on query chunks, ρ≈0, ≈0.98 on filler) → use median
- v4 caveat: testbed filler/pair/query tokens are trivially separable, so "write on pairs, freeze on filler" is easy — Phase 2 must not assume this
- phase1 stream (3 seeds): Mode B 0.99/0.70/0.43 @64/256/512 facts, age-decay 0.73→0.23 @512; Mode A 0.70/0.07/0.04; GRU ≤0.13; none = chance
- phase1: ranking FLIPS between typed (A ≈1.00 > B) and stream (B ≫ A) → the typed testbed rewarded type-based gating, not memory quality
- phase1: Mode B needs second-order grads and TBPTT ≥ 4 chunks; ~1 in 15 runs never takes off (CE pinned at ln 32 = 3.47)
- bug found on Modal: first-order Mode B crashed (autograd.grad without retain_graph) — sanity suite missed it; now covered (check 4b)
- LLM (Qwen2-0.5B, mid-layer memory, 8 variants): NLL 3.70→2.3 but swap≈on → generic adaptation, not recall; lr 3e-3 never takes off; layer/width/clip/window don't matter
- LLM eval lesson: `off` (starved memory) is worse than none; `swap` (wrong document) is the control that separates recall from adaptation; agentic preference accuracy is a style artifact
- LLM: same-position (k_t,v_t) memory cannot recall strings; next-token values (`shift_v`) + crop curriculum + long window: long-context NLL 1.42-1.9 vs generic 2.15 (3 seeds), swap ≈ none
- LLM: contrastive swap loss is gameable (swap→on +1.2 with no ON gain); judge by ON vs generic reference + detail recall
- LLM Stage B (3 seeds): long-context NLL 3.67 → 1.10±0.03 (generic adapter 2.15, oracle 0.62), wrong doc 3.82, no decay with distance; oracle is not adapter-matched → % of gap over-credits
- E8 interference: overwrite errors are STALE values (75-94%) from lost within-chunk write order (chunk 16→4: stale 0.24→0.05); MLP ≈ linear on distinct-key overwrite but MLP 0.72 vs linear 0.49 on colliding keys and 0.69 vs 0.41 on long streams at matched state
- External review → open items: standalone gated DeltaNet · paired LLM controls (doc-free adapter + adapter-matched oracle) · colliding-key haystack · held-out generator + RULER/BABILong in two tiers · 1.5B only after these
- E9 paired LLM controls (same val set): doc-free adapter 2.06 / 1.98 / 2.88, memory 1.10 / 1.72 / 2.41, adapter-matched oracle 0.06 / 0.86 / 1.46 (long_context / dialogue / agentic) → fair gap closed 48% / 23% / 33%; old '84%/53%/101%' superseded
- E10: LLM chunk 32/16/8 → long-context NLL 1.46/1.40/1.43 (no effect, 1.6-2.8× cost); width 256/512/1024 → 1.46/1.32/1.28 (diminishing); gated DeltaNet (state+param matched): best at plain overwrite (0.97 vs 0.73), far worse at capacity (0.08-0.19 vs 0.69 @512) and colliding keys
- E11 Stage 0 (model v1, zero-shot, new style): NLL gain is document-specific (wrong doc worse than none) but entity-value BINDING is at chance (ranking 0.33 vs 0.33 chance; haystack 0.31); exact generation ≈ 0; frozen oracle itself stale 37% / confused 22-45% → train on programmatic overwrite/colliding data next
