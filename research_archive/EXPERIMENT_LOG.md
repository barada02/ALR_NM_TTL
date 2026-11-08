# MiniTitan — Experiment Log

Short, append-only, written for the next coding agent. Deep rationale and flaw tables: `ARCHITECTURE_LOG.md`. Plan: `/ROADMAP.md`.
Each entry = question → setup → result → decision. Keep entries to ~6 lines; link result files instead of pasting data.

## State (read this first)

- **Goal:** a small memory module whose *weights update at inference* (Titans-style test-time learning) gives a frozen LLM memory across segments.
- **Module:** `ResearchWork/memory_module_v4.py` — dense ΔW per sequence, chunked (C=16), `out = h + gate·W_o(read)` (identity at init).
  - **Mode B (gradient update, momentum optional) is the chosen branch.** Mode A (learned update rule) is parked.
- **Synthetic evidence (3 seeds):** Mode B ≫ matched GRU ≫ no memory on the hard `stream` task. The "retention problem" was really a
  *training-horizon* problem: long BPTT + a distance **curriculum** + small step clip lifts acc@512 facts 0.43 → 0.69 (E6).
- **LLM evidence (frozen Qwen2-0.5B): the memory now recalls documents, after a root-cause fix.** First attempt learned only a generic answer-style adapter
  (swap ≈ on, E7a). The fix: **next-token values (`shift_v`) + crop-around-the-fact curriculum + a long gradient window** (E7b). Replicated over 3 seeds,
  long-context answer NLL 3.70 (none) → 1.8–1.9 (generic adapter alone: 2.15), full BPTT 1.42; with the wrong document it falls back to none (3.6). Oracle 0.63.
- **Where things run:** everything on **Modal** (`launch.py`). Do not run models locally.
- **Next:** Stage B (`final_b`: full BPTT / window 32, 6000 steps, 3 seeds, 300 val/type) confirms the recipe; then scale (Qwen2-1.5B, more segments) and write-up.

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
| E7a | LLM integration (Phase 2): frozen Qwen2-0.5B + memory, segmented | **mid** insertion + window 8 best. But **swap control: 95–99% of the gain is a generic adapter** (no recall) | Gate 2b not passed; diagnose |
| E7b | Fixes: crop curriculum, contrastive swap loss, next-token values (`shift_v`) | crop / contrast alone: no recall (contrast is gameable). **`shift_v` + crop + window ≥16: ON 1.79 vs generic 2.15, 3 seeds; full BPTT 1.42; control without `shift_v` 2.18** | Recipe found; Gate 2b passes |

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

### E7 — LLM integration (done for now: honest negative)
Setup (`llm_*.py`, run via `launch.py llm <set>`): frozen Qwen2-0.5B (bf16), 256-token segments run as independent sequences (attention never crosses
segments), memory hooked in as the only cross-segment channel, loss = CE on answer tokens through the frozen lm_head, only the memory (3.9M params) trains.
Data: 30K samples (`long_context` / `dialogue` / `agentic`), 150 val per type, seeded shuffle. Eval modes: `none` (frozen base alone), `off` (memory starved —
*inflated*, injects junk), **`swap` (memory fed another sample's document, same question — the control that matters)**, `on`, `oracle` (full-context attention).
```
Gate 2a (800 steps, 1 seed)               long_context NLL   agentic NLL   verdict
  none                                         3.70            4.43
  last (after final norm)                      3.73            4.38         no gain
  mid, full BPTT                               3.06            3.73         unstable (grad norm up to 1e9)
  mid, TBPTT 8 chunks                          2.46            3.31         best  → used for stage A
Stage A (4000 steps, 8 variants: window 16, clip 0.05, layer 8/16, width 512, lr 3e-4 / 3e-3)
  all ≈ long_context NLL 2.15-2.30 (lr 3e-4 best, 2.15); lr 3e-3 never takes off (stays at 3.70); L8/L16/width/clip/window: no effect
```
**The control that changes the conclusion** (best variant lr 3e-4; default `a_base` is similar):
```
type           none  swap   ON   oracle | generic gain (none→swap)  document-specific gain (swap→on)  share doc-specific
long_context   3.70  2.23  2.15  0.63   |  +1.48                     +0.08                            4.9%
dialogue       2.44  2.06  2.05  1.14   |  +0.38                     +0.01                            1.8%
agentic        4.43  3.02  2.97  2.43   |  +1.41                     +0.05                            3.1%
agentic preference accuracy: none 0.39 · swap 0.84 · on 0.84 · oracle 0.81   → a dataset STYLE artifact, not memory
long_context exact-match recall (contains): none 0.00 · on 0.02 · oracle 0.69
```
Reading: the memory module (3.9M params, mid-layer, with W_o) learned a **document-independent adapter** that teaches the frozen model the answer style/domain prior
(dates look like dates; correct decisions sound different from "forgot the step" decisions). That shortcut explains almost all of the loss drop, so the gradient toward
real retrieval is tiny. By fact position (early/middle/late) the swap→on gap is ~0 everywhere. The earlier Gate 2a readings ("NLL 3.70 → 2.46", "preference accuracy
0.76 vs oracle 0.81") were mostly that shortcut; the swap control (added after Gate 2a) exposed it.
Why retrieval is hard here (hypotheses, untested): (1) shortcut learning — the generic path cuts the loss first and leaves no signal; (2) answers are rare multi-token strings
(serial numbers, PO codes) that need token-level binding from mid-layer states, unlike the synthetic tasks whose embeddings did the binding; (3) keys written at fact tokens and
queries read from the question segment are semantically related but lexically different, and a 3.9M-param module trained 4000 steps may not learn that match.

### E7b — why recall was missing, and the fix (done; Stage B confirming)
Diagnosis. The memory wrote pairs `(k_t, v_t)` from the **same position**. At layer 12 `h_t` mostly encodes the *current* token, so a same-position value cannot store a multi-token
string (`NX-774-B9`): there is no next-token association (induction). Fix: `shift_v` — pair `(k_t → v_{t+1})`; sanity check 6c proves the output at position p never depends on tokens > p.
Second ingredient: the fact sits far before the question, so training needs a gradient window long enough to connect write and read, and an easy start: the **crop curriculum**
(context = 48 tokens before the fact + D tokens after it, D ramping 32 → full document over the first 50% of training; eval always uses full documents).
```
1 seed unless noted, 3000 steps, lr 3e-4. Reference "generic adapter" (no document info at all) = long_context ON 2.15 | dialogue 2.05 | agentic 2.97
variant                                       long_ctx ON   vs generic   swap    note
crop only                                        2.23        +0.08       2.28    no
contrastive swap loss (full docs)                2.29        +0.14       2.31    no
crop + contrastive, w8 / margin 2                2.36        +0.21       2.39    no
crop + contrastive, window 16                    2.23        +0.08       3.43    GAMED: swap→on = +1.20 but ON no better; module learned "doc mismatches question → switch off"
shift_v, window 8 (± crop)                    2.16 / 2.20    ~0          2.3     no — window too short for the distance
shift_v + crop + window 16                       1.79        −0.36       3.20    YES   (seeds 1, 2: 1.88, 1.87)
  + contrastive on top                           2.05        −0.10       4.28    worse (swap pushed above "none")
control: crop + window 16, NO shift_v            2.18        +0.03       2.30    shift_v is necessary
shift_v + crop + window 32                       1.46        −0.69       3.54
shift_v + crop + full BPTT                       1.42        −0.73       3.57    best; detail recall 0.40 (oracle 0.74, none 0.04)
shift_v + crop + window 16, 6000 steps           1.53        −0.62       3.68
none 3.70 · oracle 0.63.  Agentic: best 2.69 vs generic 2.97, oracle 2.43 (≈87% of the none→oracle gap).  Dialogue: 1.93 vs 2.05, oracle 1.14.
```
Reading. (1) With `shift_v` the wrong document gives back ≈ the no-memory NLL (swap ≈ 3.6 ≈ none): the dependence on the document is now total. (2) The NLL stays flat with
fact→question distance (<256 tokens: 1.74, >1024 tokens: 1.72). (3) Generated answers carry the key details: gold `October 14, 1939` → `October 14th` (no memory: `1809`),
gold `…concentration of 4.2 ppm` → `4.2 ppm`. Exact-match is still ~0 because gold answers are long phrases and the memory returns the key span — judge by NLL and **detail recall**
(fraction of the gold's numbers/IDs present in the answer). (4) Wider windows / full BPTT help and no longer explode with the curriculum (earlier full-BPTT runs without crop did).
Lessons: judge recall by `ON` vs the generic reference, never by swap→on alone (contrastive swap is gameable); a structural cause (what the value *is*) beat all loss-engineering.

## Gotchas (cost us time)

- `swap→on` can be inflated by a *mismatch detector* (module switches off for the wrong document without recalling anything). Always also compare `ON` to the generic-adapter reference and check detail recall.
- Same-position `(k_t, v_t)` memory cannot recall multi-token strings from a frozen LM: use `shift_v` (next-token values).
- LLM seeds share one validation split (`split_seed`); `seed` only changes init/batch order.

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
