# RAB-001 continuation file

Compact resume state. Written 2026-09-11. Read `PROGRESS_LOG.md` for the evidence behind
every claim here. Local files plus verified artifacts are the source of truth.

## Identity

| | |
|---|---|
| Worktree | `C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification` |
| Branch | `research/ced-rab001-hotfix001-qualification-v1` |
| Branched from | `2f65f9e` (`research/ced-native-loader-qualification-v1`) |
| Commits | `8b042d4` package baseline → `e0ec051` fresh ladder → (engineering commit) |
| `implementation_sha256` | `eab80e9f99494268a113fba381f4178818aab759e3acbc6193ea936d33560fe6` |
| Qualification runtime | `C:\Users\clclc\BusinessProject\.venv-rab001` |
| Source weights | `C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze` |
| `audit_digest` | `97cd76af0fb2d6064c5cb954906ac493d21e40429cebe55f0bed0000b23d9ef8` |

Runtime: python 3.11.15 / Windows-10-10.0.26200-SP0 / torch 2.10.0+cpu /
transformers 4.51.3 / safetensors 0.7.0 / numpy 2.3.5 / tokenizers 0.21.4.
Tests: **181 passed, 0 skipped**. Hardware: RTX 4070 **Laptop** 8 GB (unused — CPU runs
only), 15.2 GB RAM, 113 GB free disk.

**`run.py:117` hard-fails any gate unless `transformers == 4.51.3`.** The shared
`Chat-Bubble/.venv` is `2.12.1+cu126 / 5.14.1` and will not run this experiment. Always use
`.venv-rab001`. Never create a venv under `experiments/architecture_benchmark_001/` —
`implementation_manifest()` globs `ROOT.rglob('*.py')` and would absorb it into the digest.

## Four-arm state

| arm | checkpoint / lineage | qualification at `eab80e9f` | numerics | engineering | next blocker |
|---|---|---|---|---|---|
| **B28** | frozen official source, no checkpoint | **G0 fresh PASS** 8/8 | FP32 4 violating elements in 1 row of 8; FP64 clean (3.8e-13) | `G2_CACHED_B28` executed — see engineering section | none; it is the frozen teacher and the C28 control |
| **D40** | `outputs/ladder-eab80e9f-v1/G1_WIDTH/attempt-0001/` (from G1_IDENTITY, bitwise exact) | **G1_WIDTH fresh PASS** 7/7 | FP32 **0 violations on every path and row** | `G2_CACHED_D40` executed — see engineering section | not a *recovered* donor; recovery recipe unfrozen |
| **C28** | `outputs/ladder-eab80e9f-v1/G1_CED_C28/attempt-0001/` (donor B28, layer 14, 596,311,424 params) | **G1_CED_C28 fresh FAIL** 12/15 | FP32 up to 63,801 violations / 8.2e-4; FP64 clean (4.1e-12) | **BLOCKED** by failed parent | strict FP32 parity |
| **C40** | `outputs/ladder-eab80e9f-v1/G1_CED_C40/attempt-0001/` (donor D40, layer 20, 594,279,808 params) | **G1_CED_C40 fresh FAIL** 12/15 | FP32 up to 134,197 violations / 5.9e-3 | **BLOCKED** by failed parent | strict FP32 parity |

Status vocabulary: every PASS/FAIL above is a **fresh** result at the current implementation
digest, produced this session. The older `evidence/g*-official-supa-1.7.0/` receipts are
**historical** (schema 1, digest `dc212c9f`) and cannot qualify current code.

## The one finding that drives everything

CED is **mathematically correct but numerically ill-conditioned in FP32**.

- FP64: C28 reproduces cache-order invariance to 4.1e-12, passes every path. No logic,
  causality, masking or cache bug. `causal_suffix_isolation_exact` is True for every row of
  every arm at every precision.
- FP32: the matched ordinary control D40 passes the identical `compare_paths` grid with
  **zero** violations; C28/C40 fail by three to five orders of magnitude more violations.
- Therefore the failure is CED-specific, not generic fp32 noise, and the pre-hotfix
  `parity_note` excusing it as ordinary accumulation is empirically falsified.

`decode_choices_exact` passes on both CED arms — which is exactly why the rejected
argmax-based rule recorded them as PASS. Argmax agreement does not establish harmlessness.

## What is NOT established

- No quality, recovery, efficacy, latency, memory, throughput or deployment claim.
- No arm has been compared to any other on any task.
- No GPU/VRAM measurement at all. CUDA would be a different runtime identity and would
  require requalifying the whole ladder; devices were not silently switched.
- G3/G4/G5 remain closed. `recovery-readiness` still reports BLOCKED with 18 missing items.
- The workflow audit of the package was **not a clean review**: 60 of 123 agents died on a
  usage limit. Its `receipt-integrity` and `data-tooling` dimensions completed verification;
  `ced-parity-gate`, `numerics`, `teacher-cache` and `cli-tests-g3` findings are unverified
  second-pass observations only.

## Data state

Local fixtures: `pilot.jsonl` 24 rows (open diagnostic, never training),
`calibration.jsonl` 156 rows, `train-smoke.jsonl` 25 rows. **No scientific recovery data.**

Candidate pool (local, no download needed):
`Chat-Bubble/Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0` — 110,000 SFT
rows, 16,504 template families (none >8 rows), validation/leakage/near-duplicate reports all
PASS (0 near-dup hits at threshold 90, zero exact overlap with sealed/dev/RAG splits).

Open data questions, all of which must be settled before any budget is frozen:
1. **Supervision policy is unregistered.** Assistant-only gives ~454k shifted targets from
   the whole pool (1M unreachable); the delivered `prepare_data` emits no label mask and
   counts every window token as a shifted target, giving ~14.9M (1M needs ~7,400 docs).
   These are different experiments. Measured: mean 135.6 tokens/row rendered, mean 4.13
   final-assistant target tokens, 0/2000 rows exceed the 512 context.
2. **Human review is pending for every row.** A 520-row audit packet and 40,000-ID pool
   exist. `data_tools.py:67` accepts `review_status: "approved"` verbatim with no reviewer
   identity, timestamp or attestation — so that field is a declaration, not evidence.
3. The corpus leakage scan ran against a build-time snapshot; its README requires re-running
   against the current local repo. Current assets exist at
   `short-response-model-phase-0-1-2/data/{sealed,dev}/`. Use hashes so sealed contents never
   reach a training process.
4. v3 rows carry `messages`, not the flat `text` field `prepare-data` requires — a rendering
   step is needed either way.

## Queued code fixes (each changes the digest — fix, then requalify, then proceed)

1. `review.py:28` reads strict-parity violations only from the legacy
   `metrics.cache_order_parity` key, so `audit-evidence` reports
   `recorded_strict_parity_failures: {}` for the new schema-2 CED **FAIL** receipts.
   Independently reproduced this session. Highest priority: the audit tool currently hides
   the most important failures in the experiment.
2. `ladder.py:101` derives the next attempt number from attempt *directories* only, while
   the attempt *log* is created with exclusive open — a child failing before `_directory`
   leaves a log that collides on retry.
3. `qualify_identity` (`run.py:276`) uses one `assert_close` on a single halves split over
   second-half positions; `G1_WIDTH` tests no cache parity at all. The ordinary arms are
   held to a materially weaker numerical gate than the CED arms. Unequal gates across arms
   is an experimental-design defect independent of the CED result.
4. `engineering.py:213` `matched_control_verified` is tautologically True for non-C arms.

Do **not** apply these while relying on the receipts in
`evidence/ladder-eab80e9f-v1/` — they bind `eab80e9f`.

## Single next action

Decide the CED disposition. The measured result does not authorize optimization of C28/C40,
and no permissive substitute may be implemented because this run failed. The two legitimate
routes, in order of scientific value:

- **(a) Reduce CED's FP32 sensitivity as an implementation change**, then requalify. The
  shared encoder-final K/V bank feeds every decoder layer, so one rounding difference
  propagates everywhere instead of being re-derived per layer (hypothesis consistent with
  the ~10x FP64 error ratio; the receipt still records `cause: UNDETERMINED`). Candidate:
  higher-precision accumulation for the shared memory only. This is "fix a reproducible
  defect and rerun the affected qualification", which is in scope.
- **(b) Prospectively amend the equivalence policy** — a separately reviewed, forward-looking
  amendment defining parity in a precision-aware way, written *without* consulting this
  run's outcome as justification. Must not be done merely because this run failed.

Ordinary B28/D40 work is independently eligible and may continue under either route.

## Exact safe resume commands

```bash
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
SRC=C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze
E=experiments/architecture_benchmark_001

# 1. Confirm identity before trusting anything
$PY -m pytest $E/tests -q                      # expect 181 passed, 0 skipped
$PY -c "from experiments.architecture_benchmark_001.gates import implementation_digest as d; print(d())"
# expect eab80e9f99494268a113fba381f4178818aab759e3acbc6193ea936d33560fe6

# 2. Re-read the existing verdicts WITHOUT recomputing (nothing here re-runs a gate)
$PY -m experiments.architecture_benchmark_001 audit-evidence --root $E/outputs/ladder-eab80e9f-v1
$PY -m experiments.architecture_benchmark_001 status --source $SRC
```

Re-running the ladder is **not** needed while the digest is unchanged; use a **new** `--out`
root if you do (`execute` refuses to reuse a root whose session context changed, and
`--resume` only continues an identical session). Never delete `ACTIVE.lock` without first
confirming no process is running.
