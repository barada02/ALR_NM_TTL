# MiniTitan — Experiment Log

Short, append-only, written for the next coding agent. Deep rationale and flaw tables: `ARCHITECTURE_LOG.md`. Plan: `/ROADMAP.md`.
Each entry = question → setup → result → decision. Keep entries to ~6 lines; link result files instead of pasting data.

## State (read this first)

- **Goal:** a small memory module whose *weights update at inference* (Titans-style test-time learning) gives a frozen LLM memory across segments.
- **Module:** `ResearchWork/memory_module_v4.py` — dense ΔW per sequence, chunked (C=16), `out = h + gate·W_o(read)` (identity at init).
  - **Mode B (gradient update, momentum optional) is the chosen branch.** Mode A (learned update rule) is parked.
- **Synthetic evidence (3 seeds):** Mode B ≫ matched GRU ≫ no memory on the hard `stream` task. The "retention problem" was really a
  *training-horizon* problem: long BPTT + a distance **curriculum** + small step clip lifts acc@512 facts 0.43 → 0.69 (E6).
- **LLM evidence (1 seed, 800 steps):** memory injected at the **middle layer** of frozen Qwen2-0.5B, TBPTT window 8 chunks, cuts
  long-context answer NLL 3.70 → 2.46 and lifts agentic preference accuracy 0.39 → 0.76 (full-context oracle 0.81). Exact-match recall not yet there (E7).
- **Where things run:** everything on **Modal** (`launch.py`). Do not run models locally.
- **Next:** E7 stage A (8 LLM variants, running) → stage B (best config × 3 seeds, longer). Open idea: document-length curriculum for the LLM.

## Run things

```
python ResearchWork/launch.py check                  # sanity suite on Modal (always run after editing memory_module_v4.py)
python ResearchWork/launch.py sweep <name> [--seeds 0,1] [--force] [--dry-run]     # names: see sweeps.py
python ResearchWork/analyze_results.py results/<sweep> [--ages facts=512]          # mean±std over seeds
```
Results → `results/<sweep>/<sweep>__<tag>__s<seed>.json` (config, git_sha, curves, `final_grid`, `final_diag`). Re-launch skips finished runs.
Credentials: `.env` (gitignored), read only by `launch.py`. Add an experiment = add a function to `sweeps.py`.

## Repo map

| File | Role |
|---|---|
| `ResearchWork/memory_module_v4.py` | the memory module + sanity suite (`_run_checks`) |
| `ResearchWork/mqar_testbed.py` | hosts (none / gru / titan_a / titan_b), train + eval loop, `run_experiment(cfg)` |
| `ResearchWork/synth_tasks.py` | `typed` (easy) and `stream` (hard) recall tasks |
| `ResearchWork/exp_config.py`, `sweeps.py` | one config dataclass; named sweeps |
| `ResearchWork/modal_app.py`, `launch.py` | Modal fan-out; credential-safe launcher |
| `ResearchWork/memory_module_v1..v3.py`, `train_kaggle.py` | **legacy**, do not use (v1 API, known flaws P1–P7) |
| `data_generation/miniTitan_data/*.jsonl` | 30K LLM samples (gitignored; schema in its README) |

## How the pieces fit

```
tokens ─► embedding ─► h (B,T,d) ─► [ MiniTitan memory ] ─► h' ─► head
                                      │ per chunk of C=16 tokens
                                      │  read : r = M(k; W+ΔW)            (k = RMSNorm(W_k h))
                                      │  write: ΔW ← ρ·ΔW + S ,  S ← η·S − θ·∇‖M(k)−v‖²   (Mode B)
                                      └─ out = h + blend·W_o(r)   (W_o zero-init ⇒ identity at start)
 causal at chunk level: a chunk reads the state from BEFORE its own write.
```
```
stream task:  [ fact fact fact ... fact ][ query query ... ]      fact  = key+value in ONE token
              ◄──── 1..32 chunks ───────►                         query = key, empty value slot
              queries sample ALL facts uniformly; no filler, one token type ⇒ accuracy-vs-age is read off one pass
```

## Experiments

| ID | Question | Result | Decision |
|---|---|---|---|
| E0 | Why did v1–v3 fail? | Static review: detached fast weights, trivial objective, saddle init, 4-token horizon (MAX_RANK truncation), dead retention gate | Rebuild as v4 |
| E1 | Does v4 beat no-memory on MQAR (Gate 0)? | none 0.03, GRU 0.09–0.29, Mode A ≈1.00, Mode B ≈1.00 (typed task) | Gate 0 passed |
| E2 | Is E1 Mode A a leak? | No: memory off → chance; resampled values → answers follow; ρ≈0.98 on filler, ρ≈0 on query chunk (so *mean* ρ misleads) | Report median-chunk half-life |
| E3 | Is Mode B stable? | Diverges (190× / NaN) once inner_lr·gate ≳ 0.15 with momentum | `inner_clip=0.1`, `inner_lr=0.2` |
| E4 | Phase 1: A vs B vs matched GRU (`p1_typed`, `p1_stream`) | typed: A 1.00 > B 0.63–0.85 @512 filler. **stream: B 0.99/0.70/0.43 @64/256/512 facts; A 0.70/0.07/0.04; GRU ≤0.13** | **Gate 1: Mode B.** Ranking flips ⇒ typed task rewarded token-type gating |
| E5 | Ablations (`abl_*`, 78 runs) | B needs 2nd-order grads (1st-order = chance) and TBPTT ≥4 chunks; momentum no help; A needs aux loss, rank/delayed-aux irrelevant; masking filler writes does *not* fix B (hypothesis refuted); ~1/15 B runs never take off (CE stuck at ln 32) | Keep B: 2nd-order, TBPTT 4, no-momentum slightly better |
| E6 | Retention round (`ret_*`, 75 runs): why does B lose old facts (acc by age 0.73 → 0.23)? | **Not decay, not capacity.** Gradient horizon + curriculum: acc@512 0.43 → **0.69**, 15/15 seeds take off | Recipe: long BPTT + curriculum + clip 0.05 |
| E7 | LLM integration (Phase 2): frozen Qwen2-0.5B + memory, segmented | Gate 2a: **mid** insertion + window 8 ≫ last / full-BPTT. Memory carries gist/format, not yet exact tokens | Stage A/B running |

### E6 — retention round (done)
Baseline = `p1_stream/titan_b`: acc 0.99 / 0.70 / 0.43 at 64 / 256 / 512 facts. Per-seed view matters: many variants have seeds that *never take off*.
```
hypothesis                         sweep          acc@512     verdict
(e) decay too aggressive           ret_decay      0.43        NO  — fixing ρ≈1 changes nothing; half-lives already 10^5+ tokens
(b) memory too small               ret_capacity   0.46-0.47   weak (+0.03); 3-layer memory fails to start
(c) step clip throttles writes     ret_clip       0.51@0.05   YES, small: gentler writes interfere less (0.3/1.0 → 0.46)
(d) chunk too coarse               ret_chunk/bptt no gain     c=8 same as 16; c≥32 needs a long window or fails to start
(a) never trained at long horizon  ret_dist/bptt  0.53-0.67   YES — but 13/24 such runs never take off (vs 7/69 at default distance)
fix: curriculum on distance        ret_combo      0.60-0.69   15/15 seeds take off; best = full BPTT + curriculum + clip 0.05
```
`ret_combo` (3 seeds, no momentum, train ≤256 facts, curriculum ramps 16→256 over first 50%): full BPTT + curriculum + clip 0.05 → **0.88 @256, 0.69 @512**;
window-8 + curriculum → 0.85 / 0.60-0.63; full BPTT without curriculum → 2/3 seeds take off. Take-away: the bottleneck is how far the gradient can reach
into the past during training (plus write interference), not what the memory does to old content. Remaining gap at 512 facts is interference/capacity, not decay.

### E7 — LLM integration (in progress)
Setup (`llm_*.py`, run via `launch.py llm <set>`): frozen Qwen2-0.5B (bf16), 256-token segments run as independent sequences (attention never crosses
segments), memory hooked in as the only cross-segment channel, loss = CE on answer tokens through the frozen lm_head, only the memory (3.9M params) trains.
Data: 30K samples (`long_context` / `dialogue` / `agentic`), 150 val per type, seeded shuffle. Modes at eval: `none` (frozen base alone), `off` (memory starved —
*inflated* control, injects junk), `swap` (memory fed another sample's document — the honest recall control, added after Gate 2a), `on`, `oracle` (full-context attention, upper bound).
```
Gate 2a (800 steps, 1 seed, A10G ~0.4 s/step)      long_context NLL     agentic NLL    agentic pref-acc
  none (frozen base)                                    3.70               4.43           0.39
  last  (after final norm)                              3.73  (no gain)    4.38           0.41
  mid, full BPTT                                        3.06               3.73           0.61   gradient norms up to 1e9
  mid, TBPTT 8 chunks                                   2.46               3.31           0.76   stable (gn ≈ 1-3);  oracle: 0.63 / 2.43 / 0.81
```
Reading it: memory helps; gain is spread over early/middle/late fact positions; generated answers have the right *type* ("April 14th, 1944" for a date) but not
the exact tokens (exact-match 0 vs oracle 0.69) → gist, not yet precise recall. Stage A (`full_a`, 4000 steps): window, step clip, insertion depth, memory width, lr.


## Gotchas (cost us time)

- `modal volume put/ls/rm` from Git Bash rewrites `/path` into `C:/Program Files/Git/path` → prefix with `MSYS_NO_PATHCONV=1`.
- Data: ~47 of 30K jsonl records are malformed (missing fields) — the loader skips and reports them.
- LLM eval: `off` (starved memory) is *worse* than no memory because it injects junk; never quote on-vs-off as the gain. Use `none` and `swap`.
- LLM full BPTT through ~70 chunks → gradient norms 1e4-1e9 (clipped to 1.0, but worse results). Use a TBPTT window (8) or a curriculum.
- Runs that "never take off" sit at CE = ln(vocab) for the whole run; always read per-seed numbers, means hide them.

- `second_order=False` needs `retain_graph=True` in the inner `autograd.grad` (fixed; sanity check 4b).
- Never trust mean retention/gate values — gates are content-dependent; read `per_chunk` traces.
- The `typed` task is gate-shortcut-able; judge memory on `stream`.
- Modal: two local entrypoints ⇒ `modal run file.py::main`; launcher already does this.
- ~1/15 Mode B runs plateau at CE = ln(32) = 3.47 (never take off): run ≥3 seeds, report per-seed failures.
