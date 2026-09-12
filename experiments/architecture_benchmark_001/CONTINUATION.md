# RAB-001 continuation file

Compact resume state. `evidence/COMPREHENSIVE_ANALYSIS.md` holds the full argument;
`evidence/REQUALIFICATION_REVIEW.md` records which results survived the bug fixes;
`PROGRESS_LOG.md` is the append-only history. Local files and verified artifacts are the
source of truth — prior chat summaries are not.

## Identity

| | |
|---|---|
| Worktree | `C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification` |
| Branch | `research/ced-rab001-hotfix001-qualification-v1`, pushed to `LUNCY579/ChatBubble` |
| Branched from | `2f65f9e` |
| `implementation_sha256` | **`8b47b395`** (HOTFIX-004), 31 manifest files |
| Tests | **191 passed, 0 skipped** |
| Qualification runtime | `C:\Users\clclc\BusinessProject\.venv-rab001` |
| Source weights | `Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze`, `audit_digest 97cd76af…` |
| Current receipts | `evidence/ladder-8b47b395-v1/`, `evidence/quality-8b47b395/` |

Runtime: python 3.11.15 / Windows-10-10.0.26200 / torch 2.10.0+cpu / transformers 4.51.3 /
safetensors 0.7.0 / numpy 2.3.5. Hardware: RTX 4070 **Laptop** 8 GB (**unused** — every run
is CPU), 15.2 GB RAM, 32 logical CPUs.

**`run.py:117` hard-fails unless `transformers == 4.51.3`.** The shared `Chat-Bubble/.venv`
has drifted to 2.12.1/5.14.1 and will not run this experiment. Never put a venv under
`experiments/architecture_benchmark_001/` — `implementation_manifest()` globs
`ROOT.rglob('*.py')` and would absorb it into the digest.

**Five digests have been qualified end to end** — `eab80e9f`, `8d54f540`, `fc44701e`,
`7d48de21`, `8b47b395` — and every one reproduces the same numbers to the element. Each
`.py`/`protocol.json` edit forces this cycle: change → run suite → requalify → verify inert.

## Four-arm state

| arm | what it is | construction | quality | training |
|---|---|---|---|---|
| **B28** | frozen official source | `G0` PASS | **the only working model**: CE 5.55, 6/24 pilot, `G1_QUALITY` PASS 8/8 | see below |
| **D40** | ordinary width 1024→896 | `G1_WIDTH` PASS 7/7 | **degenerate** `</think>` loop, CE 5.36 (better than source), `G1_QUALITY` **FAIL** on 3 | not run |
| **C28** | CED, 14 enc + 14 shared-memory dec | `G1_CED_C28` **FAIL** 12/15 | CE 12.59, **worse than uniform** (11.93), 0/24 pilot | loses the matched comparison |
| **C40** | CED from D40 | `G1_CED_C40` **FAIL** 12/15 | CE 12.96, worse than uniform, 0/24 | blocked by failed parent |

C28/C40 optimizer lanes are closed by `gates.py` unaided: their `G2_CACHED_*` stages declare
the failed `G1_CED_*` receipts as parents.

## What is established

1. **Neither conversion is usable, and no construction gate tested for it.** `G1_WIDTH`,
   `G1_CED_*` and `G2_CACHED_*` are all satisfied by a model emitting token salad. Parity is
   self-consistency, not quality — D40 passes the full parity grid with **zero** violations
   while degenerate. `G1_QUALITY` (HOTFIX-003/004) is the fix and catches D40 on 3 checks.
2. **A KV-sharing law.** Every removed bank displaces exactly one layer *by construction*,
   each displaced layer costs ~0.5 nats, and CE tracks the displaced **count**, not the
   topology — confirmed by CED (13 displaced → 12.5854) and local cross-layer sharing
   (14 → 12.9136) landing on one curve. Local sharing was hypothesised to win and is
   **falsified**. Usable budget: 2–3 banks, ~10% KV. No clever topology exists.
3. **CED is FP32-ill-conditioned, not buggy.** Clean at FP64 (4.1e-12); divergence starts at
   encoder layer 0 and the shared bank is already 4.6e-04 out of tolerance before any decoder
   read. CED forfeits the final-RMSNorm suppression a plain model gets. The read-side
   precision fix (HOTFIX-002) is **falsified**; full-context bank reconstruction is ruled out
   analytically (it needs an O(T²) encoder re-run).
4. **Training: CED loses at both doses.** Matched recipe, same rows, same token budget:
   +0.66 nats at lr 1e-4, **+1.21 nats at lr 1e-5**, with CED at 0/6 against the source's
   6/6. It learns the answer *format* and never the arithmetic.
5. **Dose, not data.** lr 1e-4 destroys the source's arithmetic (6/6 → 0/6) while improving
   CE; lr 1e-5 preserves 6/6 **and** reaches better CE. So CE alone cannot certify a run —
   and neither can KL, which rises even when capability is fully preserved.

## What is NOT established

- No efficacy, latency, memory, throughput, 8-GB-fit or deployment claim. GPU unused.
- No result beyond 40 training steps. No replay/regularisation recipe tried.
- ~~The gated variant is never verified on real weights.~~ **DONE** — `gated_memory.py` is
  verified function-preserving on real weights: bitwise identical logits (max delta 0.000e+00),
  CE identical, branch live under perturbation. See `evidence/arm-quality/GATED_VERIFY_split14.json`.
- **Three adversarial-review workflows all died on usage limits and produced nothing.** No
  conclusion here has survived an external adversarial pass.

## Bugs found in my own harness — and how

Four defects, **none found by code review**:

| defect | how it surfaced | impact |
|---|---|---|
| seed consumed nothing | two seeds gave bit-identical output | would have faked "replicated across 3 seeds" |
| leading-integer scoring | log showed correct answers while hits read 0 | ~21% of task set inflated |
| full-string scoring (earlier) | raw samples showed `3.` vs `3` | **inverted a published conclusion** |
| a print could kill a run | runs died with compute spent | lost 3 C28 runs |
| eval chosen without checking the teacher | smoke test: source 0/20 on `arithmetic_chain` | would have produced a confidently empty result |

**Keep the exact controls and cheap smoke tests.** `ced_last1` and `local group 1` must be
bitwise identity; inert-default requalification must reproduce numbers to the element;
step-0 metrics must match across seeds. Every bug above was caught by one of those or by
reading raw samples.

The scorer now lives in exactly one place, `experiments/rab001_exploration/scoring.py`, with
an 11-case self-test that **runs at import**. All 12 exploration scripts reconfigure
stdout/stderr to UTF-8 so output formatting cannot destroy a measurement.

## Data and evals

Local fixtures: `pilot.jsonl` 24 rows (open diagnostic, never training), `calibration.jsonl`
156, `train-smoke.jsonl` 25. Corpus:
`Chat-Bubble/Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0`, 110k rows,
16,504 families, leakage/near-duplicate reports PASS, re-verified against current sealed/dev
with **0 collisions** across 124,000 rows.

Rigorous splits in `outputs/splits/` (manifest committed): **TRAIN 800** across 36 classes,
**DEV 150**, **TASK 150** balanced over the probe classes. Zero family and zero id overlaps,
asserted at build *and* run time. Rendering verified to reproduce all 25 existing prepared
rows byte-for-byte.

**Probe classes** `percentage_reasoning`, `ratio_proportion`, `time_duration` — chosen from
the measured frozen-source baseline (65% / 50% / 30%) and held out of TRAIN/DEV so the task
measures **retention**. `arithmetic_chain` is excluded: the source scores **0/20**, so it
would measure nothing. The selection was deliberately **not re-fitted** after the scorer
correction moved `time_duration` to exactly 30.0%.

Human review is **pending for every corpus row**; `data_tools.py:67` accepts
`review_status: "approved"` verbatim with no attestation, so that field is a declaration.

## Single next action

**Finish the `source_control` 500k arm.** Its matched `C28` counterpart is COMPLETE
(`outputs/longbudget/LONGBUDGET_C28_seed17_b500000_lr1e-05.json`, 500,099 supervised target
tokens, 5.14 h) but the control has been interrupted three times and stands at 250,016 of
500,000. **The paired 500k comparison must not be presented as finished until it lands.**
Resume by re-invoking with identical arguments; the harness refuses to resume if
`train.jsonl` has changed and drives its LR schedule from the token count, so a resume lands
on the same schedule.

Then, in order:

1. **Build the training-side capability-retention gate.** `G1_QUALITY` covers initialization;
   nothing yet stops an optimiser trading away capability, and both candidate likelihood
   endpoints are measured to decouple from it. Highest-value protocol work left.
2. **Ridge-sensitivity sweep for the latent result.** `latent_kv.RIDGE = 1e-6` was chosen for
   Cholesky well-posedness and never tuned, against a covariance with condition number 1.08e8.
   It is the largest open hole in the only deployable finding here. `latent_ridge_sweep.py`
   exists; no real sweep has completed.
3. Do **not** pursue another sharing topology, read-side precision, or full-context bank —
   all falsified or ruled out.
4. Budget is **no longer** the open question, and the old "~31 h/arm, do not try it" advice was
   wrong by ~3x: the measured rate is **27.0 supervised targets/s** (500,099 targets in
   18,496 s), i.e. ~10.3 h per 1M per arm. The run was done. It did **not** restore capability
   — C28 went 0/60 → 2/60 (peak 4/60 at 50k) while reaching dev CE 0.4727. The advice now
   rests on that outcome, not on an infeasibility estimate.

## What is already finished (do not re-run)

- **Multi-seed matched comparison** — all twelve `outputs/rigorous/RIGOROUS_*.json` exist
  (4 arms × seeds 17/29/43), aggregated in `AGGREGATE_MULTISEED.json` (`runs_loaded: 12`,
  `missing: []`). Written up in `evidence/trainability/MULTISEED_RESULT.md` and
  `RECOVERABILITY_FRONTIER.md`. ~4 h of compute; do not repeat it.
- **Latent KV sweep** — `evidence/latent/LATENT_KV_RESULT.md`, 4× reduction at no measurable
  capability cost, zero training.
- **Device parity** — `evidence/device-parity/DEVICE_PARITY.md`.

## Exact safe resume commands

```bash
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
SRC=C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze
E=experiments/architecture_benchmark_001

# 1. Confirm identity before trusting anything (expect 191 passed, 0 skipped)
$PY -m pytest $E/tests -q
$PY -c "from experiments.architecture_benchmark_001.gates import implementation_digest as d; print(d())"
# expect 8b47b39595dcdd7ee82512de16e4ea30abee4642063de7f5119be76ed6d6621f

# 2. Re-read existing verdicts without recomputing anything
$PY -m experiments.architecture_benchmark_001 audit-evidence --root $E/outputs/ladder-8b47b395-v1
$PY experiments/rab001_exploration/scoring.py        # scorer self-test

# 3. Aggregate the multi-seed comparison
$PY experiments/rab001_exploration/aggregate_rigorous.py
```

Re-running the ladder is unnecessary while the digest is unchanged; if you do, use a **new**
`--out` root (`execute` refuses a root whose session context changed). Never delete an
`ACTIVE.lock` without confirming no process is running.
