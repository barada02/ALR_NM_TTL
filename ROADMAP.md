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
  - ⬜ Base model attends only within current segment (~256 tok)
  - ⬜ Memory = only cross-segment channel
  - ⬜ Map data to segments: dialogue turns, agentic steps, context chunks
- Insertion-point branches
  - Branch I1 — last-layer residual delta + frozen lm_head (MAL-style)
  - Branch I2 — mid-layer inject, re-run remaining frozen layers (MAC-style)
  - 🚪 GATE 2a: two short runs on Qwen2-0.5B decide
- Trainer rewrite (train_kaggle v2)
  - ⬜ port to v4 module (metric keys, aux loss term)
  - ⬜ fix step_in_session counter (optimizer never steps)
  - ⬜ shuffled train/val split (seeded)
  - ⬜ single-tokenization label boundary
  - ⬜ autoregressive needle eval (generate after "Answer:", not prompt logits)
  - ⬜ clean baseline: frozen base w/o memory module (not update_memory=False confound)
- ⬜ Local smoke test: Qwen2-0.5B, seq 128, batch 1 🖥️
- 🚪 GATE 2b: with-memory beats frozen base on segmented needle @ 2–4 segments

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
