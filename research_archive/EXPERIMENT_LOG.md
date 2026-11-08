# MiniTitan — Experiment Log

Short, append-only, written for the next coding agent. Deep rationale and flaw tables: `ARCHITECTURE_LOG.md`. Plan: `/ROADMAP.md`.
Each entry = question → setup → result → decision. Keep entries to ~6 lines; link result files instead of pasting data.

## State (read this first)

- **Goal:** a small memory module whose *weights update at inference* (Titans-style test-time learning) gives a frozen LLM memory across segments.
- **Module:** `ResearchWork/memory_module_v4.py` — dense ΔW per sequence, chunked (C=16), `out = h + gate·W_o(read)` (identity at init).
  - **Mode B (gradient update, momentum optional) is the chosen branch.** Mode A (learned update rule) is parked.
- **Synthetic evidence (3 seeds):** Mode B ≫ matched GRU ≫ no memory on the hard `stream` task. The "retention problem" was really a
  *training-horizon* problem: long BPTT + a distance **curriculum** + small step clip lifts acc@512 facts 0.43 → 0.69 (E6).
- **LLM evidence (frozen Qwen2-0.5B, 3 seeds, 300 val/type): the memory recalls documents.** First attempt learned only a generic answer-style adapter (swap ≈ on, E7a).
  The fix — **next-token values (`shift_v`) + crop-around-the-fact curriculum + long gradient window** (E7b) — gives, after 6000 steps (E7c): long-context answer NLL
  **3.67 (no memory) → 1.10±0.03** (generic adapter alone: 2.15; full-context oracle 0.62); wrong document → 3.82; key-detail recall 0.50 (oracle 0.83, none 0.04).
- **Where things run:** everything on **Modal** (`launch.py`). Do not run models locally.
- **Stage 0 verdict (E11) — a refinement of the main claim:** zero-shot on a NEW generator style, model v1 still gains document-specific NLL (single 3.54 → 1.88, wrong document 4.02) but it is **at chance at entity-value
  binding**: current-value ranking vs colliding values 0.33 (chance 0.33), haystack 0.31 (0.31), overwrite 0.51 (0.45); exact generation ≈ 0 even on the single-fact control. So the memory carries *which values the document contains*
  (content / format), not *which entity owns which value*. Training data had almost no collisions or overwrites, so binding was never required; the next step is TRAINING on programmatic overwrite / colliding data (Stage 2).
- **Goal (decided):** a workshop-style *mechanism* paper (not a deployable context-extension method). External review (2 independent engineers) ranked the open risks — see "Open items".
- **Interference (E8):** overwrite works but ~75-94% of its errors are STALE values, caused by lost write order inside a chunk (smaller chunks fix it). At matched state size the 2-layer
  MLP memory ties a linear memory on distinct-key overwrite but beats it on colliding keys (0.72 vs 0.49) and on long streams (0.69 vs 0.41).
- **Fair LLM numbers (E9, paired controls, same val set/budget):** long-context NLL — none 3.67 · document-free adapter 2.06 · **memory 1.10±0.03** · adapter-matched oracle 0.06. The memory beats the
  document-free adapter by 0.96 / 0.26 / 0.48 NLL (long-context / dialogue / agentic) and closes **48% / 23% / 33% of the fair gap** (the earlier 84% / 53% / 101% used an unfair oracle and is superseded).
- **Next (agreed order):** (3) haystack with COLLIDING keys (same question template, different value — pick distractor docs by question similarity); (4) held-out generator + public benchmark in two tiers
  (zero-shot and after adaptation, report both); (5) Qwen2-1.5B only if the matched-state margin holds; standalone gated DeltaNet before submission. Also: the 50-77% remaining gap is probably state capacity — sweep memory width.

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
| E9 | LLM paired controls: document-free adapter + adapter-matched oracle (`paired_ctl`) | Fair ceiling is far lower than the old oracle (long-context 0.06 vs 0.62). Memory beats the doc-free adapter by +0.96 / +0.26 / +0.48 NLL; closes 48% / 23% / 33% of the fair gap | Report ON vs doc-free adapter and the fair gap; drop the old "% of oracle gap" |
| E11 | Stage 0: zero-shot eval of model v1 on programmatic overwrite / colliding / haystack / absent sets (new style) — NLL, generation, likelihood ranking | NLL gains document-specific (wrong doc worse than none) but entity binding at chance (0.33 vs 0.33; haystack 0.31); exact generation ≈ 0; the frozen oracle itself is stale 37% / confused 22-45% | Train on overwrite/colliding data (Stage 2); report binding by ranking, not generation |
| E10 | LLM chunk/width screen (`fix_d`) + standalone gated DeltaNet at matched state | chunk 32/16/8: long-ctx NLL 1.46/1.40/1.43 (no effect, 1.6-2.8× cost). Width 256/512/1024: 1.46/1.32/1.28 (diminishing). Gated DeltaNet: best at plain overwrite (0.97 vs 0.73), far worse on capacity (0.08-0.19 vs 0.69 @512 facts) and colliding keys | Keep model v1 = hidden 256, chunk 32; DeltaNet baseline done |
| E8 | Overwrite / colliding keys (`upd_*`), linear-memory ablation (`lin_stream`), why errors are stale (`upd_diag`) | Memory survives interference; errors are stale values from lost within-chunk order (C 16→4: stale 0.24→0.05). MLP ≈ linear on distinct keys, MLP ≫ linear on colliding keys / long streams at matched state | Use small chunks for update-heavy data; DeltaNet baseline still owed |
| E7c | Stage B: recipe at 6000 steps, full BPTT vs window 32, 3 seeds | **long-context NLL 1.10±0.03 / 1.15±0.01** (none 3.67, swap 3.82, oracle 0.62); agentic ON 2.41 ≈ oracle 2.42; pref-acc 0.91 vs swap 0.81 | **Gate 2b passed**; use full BPTT + crop + shift_v |
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

### E7c — Stage B (done): the recipe holds, tight error bars
`final_b`: shift_v + crop curriculum, 6000 steps, lr 3e-4, 3 seeds, 300 val/type (one shared split). Full BPTT vs TBPTT window 32 — **no difference**, both stable (8.7 GB, 22 min on an A10G).
```
mean±std over 3 seeds        none    swap    ON          oracle   vs generic(2.15/2.05/2.97)   oracle-gap recovered   detail-recall none/ON/oracle
long_context  full BPTT      3.67    3.82    1.10±0.03   0.62     −1.05                        84%                    0.04 / 0.50 / 0.83
              window 32      3.67    3.92    1.15±0.01   0.62     −1.00                        83%                    0.04 / 0.47 / 0.83
dialogue      full BPTT      2.40    2.11    1.72±0.01   1.13     −0.33                        53%                    0.06 / 0.44 / 0.48
agentic       full BPTT      4.40    3.23    2.41±0.01   2.42     −0.56                        101%                   preference acc none/swap/ON/oracle 0.39/0.81/0.91/0.80
```
Long-context ON NLL by fact→question distance: <256 tokens 1.14 · 256–512 1.24 · 512–1024 1.14 · 1024+ 0.96 → no decay with distance. By fact position: early 1.02, middle 1.38, late 1.32
(80% of training facts are early; middle/late are rarer — report, don't hide). Example generations (long_context): gold `October 14, 1939` → `October 14, 1939` (exact, no-memory base: `1809`);
other samples still wrong (`titanium flow-valve assembly TX-7704` → `4,200 kilograms`).
Caveats: (1) **superseded by E9**: the "oracle" here is the frozen base with full attention and **no** answer-style adaptation, so the "% of oracle gap" figures (84% / 53% / 101%) over-credit the memory;
the fair numbers are in E9 (48% / 23% / 33%). (2) The generic-adapter reference (2.15/2.05/2.97) comes from Stage A's 150-sample val sets — indicative, not paired. (3) One model (0.5B), one data
generator (3 facts / 1000-word docs), up to 8 segments. (4) Exact-match is still low; the memory returns the key details more often than whole phrases. (5) Generic style adaptation is still part of the
none→ON gain (agentic swap 3.23 vs none 4.40); only the swap→ON / ON-vs-generic parts are document-specific.

### E8 — interference: overwrite, colliding keys, linear-memory ablation (done)
Triggered by external review: the synthetic tasks never re-wrote a key or collided keys. New `update` task (`synth_tasks.UpdateTask`): K keys, each written R times with a DIFFERENT value, interleaved;
query = LATEST value; `stale` = share of answers that equal an OLDER value of that key. `n_attrs=4` makes keys collide (key = entity+attribute, keys of one entity share the entity embedding).
All memory hosts use the E6 recipe (full BPTT, curriculum, clip 0.05, no momentum); 3 seeds; "linear" = 1-layer Mode B = ONE d×d map + delta rule — a *linear associative memory*, NOT a DeltaNet
(no per-channel gating/normalisation); `lin_b_d128` has the same state as the 2-layer MLP at dim 64 (16,384 floats), `lin_b` has 4× less (4,096). Chance 0.03.
```
accuracy of the LATEST value (successful seeds)         K16 R1   K16 R2   K16 R4   K16 R8   K64 R8 | colliding keys: K16R4  K64R8
GRU 75 / 250 floats of state                             0.10-0.11 (flat, ≤0.11 everywhere)                | 0.10 / 0.11   0.05
Mode A (state 16,384)                                    1.00     0.72     0.73     0.67     0.37   |  0.61          0.27
Mode B 2-layer MLP (16,384)                              1.00     0.72     0.73     0.71     0.49   |  0.72          0.46
linear d=128 (16,384, same state)                        1.00     0.74     0.72     0.70     0.42   |  0.49          0.22
linear d=64 (4,096, 4× less state)                       0.75     –        0.42     0.39     0.21   |  0.28          0.13
```
1. **Overwrite works but is imperfect, and the errors are STALE values.** One write per key: 1.00. One overwrite (R=2): 0.72, and more overwrites barely hurt. At moderate load 75-94% of the wrong
   answers are the previous value (a random guess would be stale 3-22%). At high load (K=64, R=8) errors turn random (≈25% stale) → capacity, not staleness.
2. **Cause: lost write order inside a chunk** (all tokens of a chunk are written simultaneously, so two writes to one key cannot say which is later). `upd_diag` (Mode B, successful seeds):
   chunk 16 → 8 → 4: stale 0.24 → 0.11 → 0.04-0.06, accuracy at K16R2 0.72 → 0.88 → 0.95; a looser step clip (0.3, 1.0) changes nothing (hypothesis "writes too gentle" refuted). Smaller chunks also
   cure take-off (3/3 seeds, vs 2/3 at chunk 16). Cost: more chunk steps. Chunk 1 would be exact; a recency-aware within-chunk write (position-weighted / cumulative decay) is the untested fix.
3. **Does the nonlinearity earn its keep?** At MATCHED state it depends on the task. Distinct keys, overwrite: tie (0.72 vs 0.73). **Colliding keys: MLP 0.72 vs linear 0.49** (Mode B loses nothing from collisions:
   0.73 → 0.72; linear drops 0.72 → 0.49). **Long stream (E6 recipe): MLP 0.69 vs linear 0.41 at 512 facts** (2.5× fewer parameters: 36K vs 91K). State size matters a lot: linear with 4× less state is far worse.
**Parameters × fast state × accuracy** (successful seeds; Mode B took off in 2/3 seeds at chunk 16 — every other row 3/3):
```
host                           core params   fast-state floats   overwrite K16R4   K64R8   colliding K16R4   K64R8
GRU (small / large)            36,589 / 253K        75 / 250     0.10 / 0.11       0.05    0.10 / 0.11       0.05
Mode A (learned rule)             252,547          16,384        0.73              0.37    0.61              0.27
Mode B, 2-layer MLP                36,292          16,384        0.73              0.49    0.72              0.46
linear assoc. memory d=64          23,876           4,096        0.42              0.21    0.28              0.13
linear assoc. memory d=128         90,660          16,384        0.72              0.42    0.49              0.22
stream task, E6 recipe:   acc@64 / @256 / @512  — MLP (36,292 params, 16,384 state) 1.00 / 0.88 / 0.69 · linear d=128 (90,660, 16,384) 0.99 / 0.67 / 0.41 · linear d=64 (23,876, 4,096) 0.37 / 0.18 / 0.12
```
The claim, stated precisely: **at equal fast-state size (16,384 floats) the MLP memory matches a linear memory on distinct-key overwrite and beats it under interference (colliding keys 0.72 vs 0.49; long stream 0.69 vs 0.41)
while using 2.5× fewer parameters.** The d=64 linear memory has 4× less state and is not part of that claim. "Nonlinearity earns its keep" is supported only under interference, not on plain overwrite.
4. Mode B still has ~1/3 seeds that never take off at chunk 16 in this recipe (`titan_b` 2/3 on `upd_main` and `upd_similar`); chunk ≤8 gave 3/3.
Caveats: GRU still not state-matched in the matrix sense; no gated DeltaNet / GLA / Mamba baseline yet; keys/values are still embedding-bound single tokens; LLM data has almost no overwrites, so (1)-(2) are a
limitation to report, not a blocker for the document-recall claim.

### E9 — paired LLM controls (done): the fair numbers
`paired_ctl` (6000 steps, lr 3e-4, 300 val/type — the same validation set and budget as Stage B). `ctl_adapter_seg` = the SAME memory module with writes disabled on the segmented model (a document-free
adapter, same capacity). `ctl_adapter_oracle` = the same module with writes disabled on the FULL-attention model (adapter-matched oracle = fair ceiling).
```
NLL                          none   doc-free adapter   MEMORY (3 seeds, full BPTT)   frozen oracle (old)   oracle + adapter (fair ceiling)   memory − adapter    fair gap closed
long_context                 3.67   2.06               1.10 ± 0.03                   0.62                  0.06                              −0.96               48%   (gap 2.01)
dialogue                     2.40   1.98               1.72 ± 0.01                   1.13                  0.86                              −0.26               23%   (gap 1.12)
agentic                      4.40   2.88               2.41 ± 0.01                   2.42                  1.46                              −0.48               33%   (gap 1.43)
agentic preference accuracy: doc-free adapter 0.86 · memory 0.91 · frozen oracle 0.80 · oracle+adapter 0.96   (the style artifact alone gives 0.86)
long_context detail recall (key numbers/IDs of the gold answer): none 0.04 · memory 0.50 · frozen oracle 0.83
```
**Protocol (read before quoting any percentage).** The three runs share: the same memory module (3.9M params), 6000 steps, lr 3e-4, batch 4, one validation split (`split_seed` 0, 300 per type) and the
same frozen Qwen2-0.5B. Differences that remain and matter: the oracle adapter trains on FULL documents with full attention (no crop curriculum, no segmentation), and it sees all context directly, so its NLL is a
ceiling for "document available + same adaptation", not a competing method. The primary result is the three raw NLLs (document-free adapter, memory, adapter-matched oracle). Any percentage depends entirely on the
denominator: measured from the document-free adapter the gap closed is 48% / 23% / 33% (long-context / dialogue / agentic); measured from the no-memory baseline it is 71% / 44% / 68%. Both are "true"; quote the raw NLLs
and treat percentages as illustrative.

Reading. (1) The document-specific gain is real and large relative to seed noise (0.01-0.03). (2) The old oracle was unfair: with the same adaptation, full attention gets long-context NLL 0.06 (copying is
near-deterministic), so the memory closes about half — not 84% — of the fair gap on long-context, and a quarter to a third on dialogue / agentic. (3) The earlier generic-adapter reference (2.15 / 2.05 / 2.97) was
close to the paired one (2.06 / 1.98 / 2.88), so the E7 conclusions stand. (4) The remaining gap is the headline for improvement; capacity of the 459K-float state is the first suspect (E8: accuracy tracks state size).

### E10 — LLM chunk/width screen and the standalone gated DeltaNet (done)
**LLM screen** (`fix_d`: 1 seed, 3000 steps, 300 val/type; gradient window held at ~1024 tokens so only chunk size changes; reference on the same val set: document-free adapter 2.06 / 1.98 / 2.88, adapter-matched oracle 0.06 / 0.86 / 1.46):
```
run      chunk window hidden  fast-state floats  params   s/step  peak GB | long_context  dialogue  agentic | detail-recall
d_c32      32    32    256       458,752          3.90M     0.22     8.1   |    1.46         1.92      2.70    |   0.41
d_c16      16    64    256       458,752          3.90M     0.35    10.0   |    1.40         1.90      2.63    |   0.41
d_c8        8   128    256       458,752          3.90M     0.61    13.8   |    1.43         1.90      2.65    |   0.40
d_h512     32    32    512       917,504          4.36M     0.24    10.0   |    1.32         1.86      2.59    |   0.47
d_h1024    32    32   1024     1,835,008          5.28M     0.29    13.8   |    1.28         1.84      2.56    |   0.47
(for scale: Stage B, same recipe, 6000 steps, 3 seeds: 1.15 ± 0.01 → training length mattered more than either lever)
```
Chunk size: no effect on document recall (differences 0.03-0.06 ≈ 1-2 seed-noise widths) at 1.6× / 2.8× the time → keep 32; also no clear effect on zero-shot overwrite (see E11). Width: −0.14 NLL for 2× state, −0.18 for 4× state — a modest, diminishing lever,
not what closes the gap to the oracle. **Model v1 = Stage B (hidden 256, chunk 32, full BPTT, 6000 steps; 3 seeds).**

**Standalone gated DeltaNet** (`deltanet.py`; S_t = α_t S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ, one 128×128 state; 32,898 params, 16,384 state floats — matched to Mode B's 36,292 / 16,384 on BOTH axes; sanity: exact delta-rule overwrite, causal in both
read orders, gradients reach all parameters). Two read orders (standard read-after-write; read-before-write as in our module) and two decay inits (per-token bias 4 → half-life 38 tokens; bias 7 ≈ Mode B's per-chunk retention): results identical across both decay inits,
so decay is not what limits it. 3 seeds each, same E6 recipe:
```
                                  overwrite K16R4  K64R8 | colliding K16R1  K16R4  K64R8 | stream acc@64 / @256 / @512
Mode B 2-layer MLP (chunk 16)         0.73         0.49   |      1.00        0.72   0.46  |  1.00 / 0.88 / 0.69
linear memory d=128 (same state)      0.72         0.42   |      0.82        0.49   0.22  |  0.99 / 0.67 / 0.41
gated DeltaNet (read-first)           0.98         0.20   |      0.79        0.61   0.07  |  0.91 / 0.42 / 0.19
gated DeltaNet (read-after)           0.97         0.11   |      0.80        0.63   0.10  |  0.90 / 0.33 / 0.08
```
At matched state and parameters the two are strong in different places: the DeltaNet is far better at plain overwrite (token-level updates, stale rate 0.01 — it beats even Mode B at chunk 4, 0.93, and confirms E8: chunking causes the stale values), while Mode B is far better on
capacity (512-fact stream 0.69 vs 0.08-0.19) and on similar keys (colliding K16R1 1.00 vs 0.80). So the claim "nonlinear deep memory buys something over a linear delta-rule state" is supported under interference and at capacity, at the same state size and fewer parameters.
Caveat: one head, one state size, trained at short lengths.

### E11 — Stage 0: zero-shot evaluation of model v1 on programmatic sets (done)
Sets (`data_generation/programmatic_eval.py`, seed 2024, 1,700 records, every invariant verified; template prose = a style nobody trained on): single (300), overwrite (300), colliding (300; 2/4/8 same-type facts), colliding_overwrite (200), haystack (400; target wrapped in 2/4/8/16
documents that contain a COLLIDING fact), absent (200). Model v1, 3 seeds, zero-shot (the model never saw an overwrite, a collision or this prose). Runner: `launch.py eval <run>`.
```
answer NLL                  none         wrong doc    MEMORY       frozen oracle
single                      3.54         4.02±0.14    1.88±0.04    0.15
overwrite                   3.27         3.76±0.14    1.69±0.04    0.24
colliding                   3.75         4.31±0.07    2.33±0.08    0.25      (NLL by #colliders 1/3/7: 2.14 / 2.28 / 2.56)
colliding_overwrite         3.41         4.06±0.19    2.14±0.07    0.36
haystack (2/4/8/16 docs)    3.67         —            2.37±0.03    0.34      (2.05 / 2.58 / 2.41 / 2.46)

likelihood ranking: share of items whose lowest-NLL candidate is the CURRENT value (candidates: current + old values + distractor values, ≤4)
                       chance   none        MEMORY       frozen oracle
overwrite               0.45    0.45        0.51±0.01    0.67
colliding               0.33    0.33        0.33±0.02    0.78      (by #colliders 1/3/7: 0.50 / 0.27 / 0.23 — exactly chance 0.50 / 0.25 / 0.12)
colliding_overwrite     0.25    0.22        0.26±0.02    0.53
haystack                0.31    0.29        0.31±0.02    0.68
greedy generation, memory: correct 0.00-0.02 on every set (single-fact control included); 95-100% "other" — plausible-format values, never the real one
  e.g. gold `December 18, 2023` → `September 12th`;  gold `Lukas Kovacs` → `Dr. Aris Thorne`.   Frozen oracle: single 0.95 correct; overwrite 0.63 correct / 0.37 STALE; colliding 0.78 / 0.22 confused; haystack 0.55 / 0.45 confused.
absent-fact questions: answers containing a value-like string — none 0.95, memory 0.80, oracle 0.99 (nobody was trained to abstain; this is a baseline, not a result)
```
Reading. (1) The document-specific NLL gain transfers to a new generator style (wrong document is WORSE than no memory). (2) But the memory does **not bind values to entities**: it cannot tell the asked entity's value from a colliding one (ranking = chance at every
collision level) and prefers the current value over old ones only barely above chance. The NLL gain therefore reflects *which values/formats the document contains*. (3) Exact recall does not transfer across generators (in-distribution detail recall 0.50 → zero-shot ≈ 0). (4) The training data
had almost no collisions or overwrites, so binding was never required — this is a training-signal gap, consistent with the data-diversity argument, not evidence against the architecture. (5) Chunk size 32 / 16 / 8 (3000-step models): current-value ranking on overwrite 0.55 / 0.52 / 0.54 and on
colliding_overwrite 0.24 / 0.28 / 0.30 (n=200, SE ≈ 0.03) — a suggestive trend, not significant; the real test needs overwrite TRAINING data. (6) The frozen full-attention oracle is itself stale 37% of the time and confused 22-45%: these sets are hard even with full context.
**Correction to earlier claims:** the swap control (E7/E9) proves document identity, not entity-value binding; "document-specific recall" should be read as "document-specific content".

## Decisions since the second review
- Stage 0 is done on model v1 (E11); the next stage is Stage 2 — TRAIN on programmatic overwrite / colliding data (cheap, exact ground truth, no API) mixed with the Gemini data, hold out attributes / templates, re-evaluate with ranking; chunk size becomes testable then.
- Generator diversity: held-out family via Fireworks serverless open-weight models (key `FIREWORKS_API_KEY` in `.env`, never printed), after Stage 2.

## Pre-registration — Stage 2 (written BEFORE any Stage 2 training run; do not edit after results arrive)
**Question.** Model v1 gains document-specific NLL but is at chance at entity-value binding (E11). Is that a *training-signal gap* (H_data) or an *architecture / representation limit* (H_arch)?
**Test.** Train the memory on data that demands binding and measure binding on documents never seen, on both seen and held-out surface forms.

*Training data (programmatic, exact ground truth, `data_generation/programmatic_eval.py --train_out`, pools="train"):* `prog_colliding` 4,000 documents → 10,635 examples (each document reused with 2-4 different target
entities as separate examples: 2/3/4 questions in 2232/901/867 documents; k ∈ {2,4,8} same-type facts per document), `prog_overwrite` 6,000 documents, `prog_single` 4,000 documents. Train pools use 4 of 8 device types, 24 locations,
the 3 original fact / 3 restate templates and 2 question templates. Held-out pools use the OTHER 4 devices, 16 NEW locations, 2 NEW fact + 2 NEW restate templates and 2 NEW question templates. Disjointness is machine-checked
(no held-out location in any training fact sentence; no eval document in the training data). Caveat: held-out device names still appear in training *filler* sentences (mentioned in passing, no fact attached).
*Eval sets:* `eval_seen_*` (train pools, seed 4242) and `eval_heldout_*` (held-out pools, seed 4343), each: single 300, overwrite 300, colliding 300, colliding_overwrite 200, haystack 200; one question per document.
*Recipe:* Stage B (6000 steps, batch 4, lr 3e-4, hidden 256, chunk 32, full BPTT, shift_v; Gemini documents use the crop curriculum, programmatic documents are NOT cropped so the colliders stay in the document). ONLY the mixture changes.
*Arms:* `s2_mix50` = 50% Gemini (⅓ each type) + 25% colliding + 15% overwrite + 10% clean single (3 seeds) · `s2_mix25` = 75% Gemini + 12.5 / 7.5 / 5 (dose arm, 1 seed) · `s2_single50` = 50% Gemini + 50% clean single-fact
programmatic, NO collisions / overwrites (control for "more programmatic data", 1 seed) · `s2_mix50_c8` = `s2_mix50` with chunk 8 (chunk size after overwrite training, 1 seed).
*Metric:* likelihood ranking = share of items whose lowest-NLL candidate is the CURRENT value among {current, old values, distractor values} (≤4 candidates; chance: colliding 0.33, overwrite 0.45, colliding_overwrite 0.25).
Baseline = model v1 (Stage B, 3 seeds), measured on the same seen / held-out sets before judging.

**Success criteria (fixed now).**
1. *Binding learned:* `s2_mix50` mean over 3 seeds, `colliding` ranking ≥ **0.55 on seen** AND ≥ **0.45 on held-out** (chance 0.33; frozen oracle 0.78).
2. *No forgetting:* Gemini long-context NLL on the 300-sample validation set (same split as Stage B) ≤ **1.15** (Stage B 1.10 ± 0.03, + 0.05 tolerance), mean over 3 seeds.
3. *Control:* `s2_single50` must NOT reach criterion 1 (otherwise the gain is not from collision training).
4. Reported without a threshold: overwrite / colliding_overwrite / haystack ranking, dose (`s2_mix25`), chunk 8 vs 32, seed spread.
**Decision rules.** (a) 1 and 2 hold → H_data supported, binding generalises across held-out templates / entities → proceed to the Fireworks held-out generator family and a public benchmark. (b) Seen passes, held-out fails
→ binding is learned but template-specific → more template / generator diversity is the lever. (c) Seen colliding ranking ≤ 0.40 → H_data rejected: the limit is architectural / representational (e.g. keys derived from a single mid-layer token
position cannot carry the entity), and the next work is the key/value construction, not data. (d) Criterion 2 fails → the programmatic data is hurting the original distribution; lower the programmatic share.

## Decisions between the second review and Stage 0
- Chunk size is a first-class variable before "model v1" is frozen (E8: chunk 16 → 4 lifts overwrite accuracy 0.72 → 0.95); screened at LLM scale together with memory width in `fix_d`
  (chunk 32/16/8 with a constant ~1024-token gradient window; hidden 256/512/1024) — 1 seed to screen, 3 to confirm; width is a MEASUREMENT (the earlier synthetic width sweep moved acc@512 only 0.43 → ~0.46) and is reported per unit
  of fast state (2·hidden·896 floats: 0.46M / 0.92M / 1.8M).
- Generator for the held-out evaluation family: open-weight models on Fireworks serverless (a different family from Gemini). The key goes in `.env` as `FIREWORKS_API_KEY`; code reads it only through a launcher-style loader and never prints it.
- Gated DeltaNet runs in parallel on the synthetic side; it gates the Qwen2-1.5B decision, not the LLM evals. Stage 0 evals (programmatic overwrite / colliding facts / colliding haystacks / absent-fact) run after "model v1" is locked.

## Open items (from two external reviews, ranked)
1. ~~Standalone gated DeltaNet~~ — DONE (E10): state- and parameter-matched; better at plain overwrite, far worse at capacity / similar keys.
2. ~~LLM paired controls~~ — DONE (E9). Report ON vs the document-free adapter and the fair gap closed (48% / 23% / 33%); never the old "% of oracle gap".
3. ~~Haystack with colliding keys~~ — built and evaluated zero-shot (E11, up to 16 documents ≈ 3K tokens); longer (32-64 segments) after Stage 2. Frame as "context storage stops growing"; never quote a KV-cache break-even number.
4. Held-out generator / topics, then a public benchmark (RULER / BABILong) in TWO tiers, both reported: zero-shot transfer and after a short adaptation with the same budget given to baselines.
5. Code nits: with `shift_v` the last token of a segment pairs with the first of the next (mask the write at segment ends); retention is one scalar per chunk shared by all layers; generation re-runs the whole
   pipeline per token (eval-code limitation); context truncation keeps the start (affects ~1% of docs).
6. Cost/regime table vs full attention (only meaningful at many segments); Qwen2-1.5B only after 1-4.

## Gotchas (cost us time)

- Chunk-parallel writes lose within-chunk order: same-key overwrites inside one chunk return the OLD value ~half the time (E8). Use chunk ≤ 8 for update-heavy data.
- `lin_b` / `mem_layers=1` is a linear associative memory, not a DeltaNet; never label it as one.

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
