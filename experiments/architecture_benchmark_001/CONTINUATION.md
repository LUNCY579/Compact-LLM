# RAB-001 continuation file

Compact resume state. `evidence/COMPREHENSIVE_ANALYSIS.md` holds the full argument;
`evidence/REQUALIFICATION_REVIEW.md` records which results survived the bug fixes;
`PROGRESS_LOG.md` is the append-only history. Local files and verified artifacts are the
source of truth — prior chat summaries are not.

**Current design checkpoint:** read `RETENTION_GATE_DESIGN.md` before implementation. Its
read-only reconciliation found final-update/resume defects in the 500k artifacts, additional
scorer gaps, and overstated endpoint/review claims. Fresh tests are **pending due to host
memory pressure**, explicitly accepted by the user; 191/0 below is the previously recorded
result. No gate implementation or protocol edit has been made at this checkpoint.

## Identity

| | |
|---|---|
| Worktree | `C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification` |
| Branch | `research/ced-rab001-hotfix001-qualification-v1`, pushed to `LUNCY579/ChatBubble` |
| Branched from | `2f65f9e` |
| `implementation_sha256` | **`8b47b395`** (HOTFIX-004), 31 manifest files |
| Tests | Previously **191 passed, 0 skipped**; fresh verification pending |
| Qualification runtime | `C:\Users\clclc\BusinessProject\.venv-rab001` |
| Source weights | `Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze`, `audit_digest 97cd76af…` |
| Current receipts | `evidence/ladder-8b47b395-v1/`, `evidence/quality-8b47b395/` |

The digest and receipt/artifact hashes were rechecked. The archived `evidence/` copies omit
model binaries and retain parent paths from the live layout; use the original `outputs/`
trees for recursive authorization. Live PASS DAGs validate against their recorded contexts.

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
2. **An initialization trend in the tested sharing layouts.** Every removed bank displaces
   one layer in these constructions. The measured initialization cost is roughly 0.5 nats
   per displaced layer: CED (13 displaced → 12.5854), local sharing (14 → 12.9136).
   Local sharing did not beat CED. This does not prove that every topology has the same CE
   cost. Later small-displacement training and latent-KV results qualify the original
   blanket conclusions; see the supersession header in `COMPREHENSIVE_ANALYSIS.md`.
3. **CED has a measured FP32 conditioning problem on the tested grid.** Clean at FP64
   (4.1e-12); divergence starts at
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

6. **Recorded likelihood and task scores diverge, with new audit limitations.** The last
   reported 500k evaluations have CE gap **0.1158** (98.4% closure), reply gap **0.4948**
   (4.27x concentration), and task C28 **2/60** versus control **26/60**. These arithmetic
   summaries reproduce, but C28's evaluation is at step 467 before its final step 468.
   Resume also drops pending accumulated gradients; update boundaries first disagree at
   step 117. Only eight truncated final samples survive. Thus the exact final matched-pair
   claim is not verified, and the legacy task counts cannot be fully rescored.
   `protocol.json` already names exact/executable capability as the G3 primary endpoint;
   its executable retention criterion, margins and budgets are missing. CE/KL alone must
   not certify training. See `RETENTION_GATE_DESIGN.md` and the preserved historical JSONs.

## What is NOT established

- No efficacy, latency, memory, throughput, 8-GB-fit or deployment claim. GPU unused.
- No result beyond the recorded 500k attempts (C28 468 optimizer updates, control 467).
  Reported cursors indicate 0.376 epochs; applied work and exact final matching have the
  resume/evaluation limitations above. No replay/regularisation recipe tried.
- ~~The gated variant is never verified on real weights.~~ **DONE** — `gated_memory.py` is
  verified function-preserving on real weights: bitwise identical logits (max delta 0.000e+00),
  CE identical, branch live under perturbation. See `evidence/arm-quality/GATED_VERIFY_split14.json`.
- Earlier review workflows failed, but commit `c8b9e28` records a completed 28-agent internal
  review. Two read-only internal reviews also completed at this design checkpoint.
  Independent external review is still not established.

## Bugs found in my own harness — and how

Five earlier defects, **none found by code review**:

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

The legacy scorer lives in `experiments/rab001_exploration/scoring.py`; its 11-case self-test
is explicitly called by consumers or under `__main__`, not at import. New cheap tests show
AM/PM, unit and negation false positives and equivalent-duration false negatives. A TASK
reference also names the wrong counted object. Typed scoring and reviewed labels are required
before certification. Keep UTF-8-safe logging and complete, untruncated prediction artifacts.

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

The 500k runs use `outputs/splits_large/`: **TRAIN 10,000**, with the original DEV/TASK.
TRAIN/DEV/TASK are mutually disjoint, but TRAIN overlaps calibration on 12 families and one
source row; DEV and TASK each overlap two calibration families. The repeatedly inspected
TASK panel is suitable for an open engineering monitor, not untouched confirmation data.

**Probe classes** `percentage_reasoning`, `ratio_proportion`, `time_duration` — chosen from
the measured frozen-source baseline (65% / 50% / 30%) and held out of TRAIN/DEV so the task
measures **retention**. `arithmetic_chain` is excluded: the source scores **0/20**, so it
would measure nothing. The selection was deliberately **not re-fitted** after the scorer
correction moved `time_duration` to exactly 30.0%.

Human review is **pending for every corpus row**; `data_tools.py:67` accepts
`review_status: "approved"` verbatim with no attestation, so that field is a declaration.

## Single next action

**Review the preimplementation design in `RETENTION_GATE_DESIGN.md`.** The user requested
this design checkpoint before writing the gate and accepted leaving fresh tests pending
while host memory is constrained. Before implementation, complete the pinned suite and
reconcile any disagreement. The proposed monitor permits zero loss of source/init-correct
items by class; separate prospective confirmation targets 90% source-success retention with
simultaneous bounds. Existing historical scores cannot be called preregistered validation.

`G1_QUALITY` currently diagnoses initialization only and is a leaf in the receipt DAG:
optimizer engineering stages do not require it. New supported training must require quality
and retention parents. Keep G3-G5 closed, preserve CED parity failures, and treat CE, reply CE
and KL as training diagnostics. The first implementation work includes typed scoring,
complete paired predictions, and exact optimizer-boundary/resume binding.

Then, in order:

1. No new topology or precision experiment is planned. Focus on retention measurement and
   enforcement; do not turn the failed local-sharing intervention into a universal theorem.
2. Budget is **closed** as an open question, and the old "~31 h/arm, do not try it" advice was
   wrong by ~3x: the measured rate is **27.0 supervised targets/s** (500,099 targets in
   18,496 s), i.e. ~10.3 h per 1M for C28 (control extrapolates to ~13.0 h). The recorded
   evaluation did **not** show restored capability
   — C28 went 0/60 → 2/60 (peak 4/60 at 50k) while reaching dev CE 0.4727. The advice now
   rests on the recorded exploratory observations and their audit limitations, not on an
   infeasibility estimate. Those timers omit initialization/cache and discarded crash work.
3. Independent external adversarial review remains unestablished; completed internal review
   must be distinguished from that missing process step.

## What is already finished (do not re-run)

- **Multi-seed matched comparison** — all twelve `outputs/rigorous/RIGOROUS_*.json` exist
  (4 arms × seeds 17/29/43), aggregated in `AGGREGATE_MULTISEED.json` (`runs_loaded: 12`,
  `missing: []`). Written up in `evidence/trainability/MULTISEED_RESULT.md` and
  `RECOVERABILITY_FRONTIER.md`. ~4 h of compute; do not repeat it. All twelve per-seed
  artifacts are now **committed** under `evidence/rigorous/` (`e7285e5`); six previously
  existed only under gitignored `outputs/`.
- **500k recorded attempts** — `outputs/longbudget/LONGBUDGET_{C28,source_control}_seed17_b500000_lr1e-05.json`,
  both `completed: true` with counters at 500,099 tokens and 3,743 rows. Recorded wall time
  is **5.14 h C28 / 6.50 h control**, not 10.3 h each. Preserve both; do not repeat them for
  this design. Final-update/matching limits are documented above; `ffc52a0`'s unqualified
  matched-final conclusion is superseded by the new reconciliation.
- **Ridge-sensitivity sweep** — `evidence/latent/LATENT_RIDGE_SWEEP_cpu.json`
  (`completed: true`, `is_smoke_run: false`, `aborted_reason: null`). Ridge 1e-8 to 1e-2 at
  three ranks: rank 512 holds **19/60 at every ridge**, dev CE spread 0.0059 nats; the
  rank-1024 exactness control reproduces the source at every ridge. This demonstrates
  insensitivity within the tested ridge range (`279277c`), not certified retention.
- **Latent KV sweep** — `evidence/latent/LATENT_KV_RESULT.md`, zero-training 4× KV result
  with recorded legacy task scores 19/60 versus source 20/60. Exploratory, not a capability
  retention certificate under the new typed-scoring design.
- **Device parity** — `evidence/device-parity/DEVICE_PARITY.md`.
- **Host-memory budget** — `experiments/rab001_exploration/hostbudget.py` (`42eb390`).
  `require(min_free_gib, label)` at startup, `guard(...)` between units of work; secondary
  jobs stand down instead of thrashing. Measured: a second model job on this 15.2 GiB host
  drove the pagefile to 13.9 GiB and 62 s → 670 s per unit of work. **Idle cores are not
  headroom; free physical memory is.**

## Exact safe resume commands

Tests are pending by user choice. Use these only when memory is available, from this
worktree root; do not start another model job beside them.

```powershell
$PY = 'C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe'
@'
from experiments.rab001_exploration.hostbudget import require, guard
require(4.0, 'RAB-001 baseline verification')
from experiments.architecture_benchmark_001.gates import implementation_digest
assert implementation_digest() == '8b47b39595dcdd7ee82512de16e4ea30abee4642063de7f5119be76ed6d6621f'
import pytest
class MemoryGuard:
    def pytest_runtest_setup(self, item):
        guard(1.0, 'RAB-001 baseline test ' + item.nodeid)
raise SystemExit(pytest.main(['experiments/architecture_benchmark_001/tests', '-q'],
                            plugins=[MemoryGuard()]))
'@ | & $PY -
if ($LASTEXITCODE -ne 0) { throw 'Baseline verification incomplete; reconcile before editing code.' }
# Expect 191 passed, 0 skipped. Do not substitute the old documented result.
```

Re-running the ladder is unnecessary while the digest is unchanged; if you do, use a **new**
`--out` root (`execute` refuses a root whose session context changed). Never delete an
`ACTIVE.lock` without confirming no process is running.
