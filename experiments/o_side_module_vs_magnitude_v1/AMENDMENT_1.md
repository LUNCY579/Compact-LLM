# OSIDE-001 Amendment 1 (pre-run, 2026-09-25)

Written before any arm was scored into `scores/`. It corrects the registered code only where an adversarial pre-run
review found a defect. **Unchanged:**
- the readout, the row sets and the arms;
- the denominator gate's limbs and thresholds;
- the comparison D and its three outcomes;
- the calibration rule, the MDE and the predictions.

Where this file and PREREG.md differ, this file governs. PREREG.md itself (962abee4) is left byte-unchanged.

## How the defects were found

A read-only, CPU-only review workflow went over the registered commit 962abee4.
- It had five independent lenses: statistics, readout/canary, arm builds and deletion, operations, and an
  independent re-implementation from the PREREG text alone.
- A triage step merged duplicates. Each finding was then given to its own refute-by-default verifier.
- 22 raw findings became 12 after triage (8 dropped as duplicates or non-defects). Of those, 10 were confirmed and
  2 refuted. **All 10 confirmed findings are minor.**

Only F1 and F2 could corrupt the registered result, and only if their trigger fires. Neither trigger was found
reachable in practice. The other eight are resume, provenance or labelling gaps. The independent
re-implementation reproduced every registered point estimate. Its only discrepancies were the two descriptive
items fixed below (F10, F11).

## Changes

| # | Finding (verified) | Change |
|---|---|---|
| F1 | A non-finite per-token NLL would become a calibrated **NULL** verdict. `np.quantile` gives [nan, nan], the null-cell exclusion test is then never true, and `outcome()` falls through. The canary would miss a NaN because Python's `max()` skips NaN unless it comes first. | The scorer stops on a non-finite NLL before writing, so it counts as a crash. `row_arrays` refuses non-finite input. The canary uses `np.max`, counts non-finite rows, and requires finite maxima. `outcome()` returns `INDETERMINATE_NONFINITE` for a non-finite interval. |
| F2 | A rebuilt arm was scored against the hash in its **own** `BUILD.json`. If the unlink in the mismatch branch failed, the rerun would score an unregistered arm as H1 or ADD-O50. | ADD-O50, H1 and H1c are scored against their **registered** hashes (`--expect-sha`). The driver refuses any arm with `REBUILD_MISMATCH.json`, or with a `BUILD.json` whose `match` is not true or whose hash differs from the registered one. The analysis checks every score file's weight hash (next row). |
| F5, F6 | Nothing enforced the prompt source or the runtime at run time. A peer's edit of `o_reply_nll.py`, `train.jsonl` or the ARM_DOSE file, or a venv change, would silently mix inputs or runtimes across arms, including on a resume. | The scorer refuses to run unless the imported `o_reply_nll.py`, `train.jsonl` and ARM_DOSE file match the manifest's recorded hashes. It also refuses unless the runtime is exactly python 3.11.15 / torch 2.6.0+cu124 / transformers 5.14.1 (PREREG §10). Every score file records the three input hashes. **New provenance check:** before anything is combined, the analysis checks every score file for the expected weight hash, the cast flag (true only for OPALLO-BF16), no row limit, 6,741 rows, finite NLLs, the registered runtime, one device, and the manifest's input hashes. A failure is `HARNESS_FAILURE`. The stage canaries run this check too. |
| F3 | The rerun that PREREG §0 allows after a canary FAIL could never pass. The resume skipped the existing score files, and the canary re-read the same failing ones. | On a canary or provenance failure the driver moves that stage's score files to `scores_failed/attempt_N_<time>/` with a `WHY.txt`. It never deletes them. It then stops. The permitted rerun rescores exactly those arms. |
| F9 | Score files were renamed without an fsync, and "already scored" meant only "the file exists". | The scorer fsyncs before the rename. A score file is **complete** only if it parses and passes the provenance check. An incomplete file (unreadable, truncated, or failing provenance) is moved to `scores_failed/` and rescored. A complete file still in `scores/` is never rescored. |
| F7 | A crash during a build left `arms/<ARM>/` without `BUILD.json`, and every rerun then stopped until someone deleted it by hand. | An arm directory without `BUILD.json` (and without `REBUILD_MISMATCH.json`) is an unverified, unscored partial build. The driver removes it, only inside `arms/` and with the path asserted, logs the removal, and rebuilds. |
| F8 | The driver imported torch in-process (seed-control → `compose.py`) and held it for the whole run, against the shared RAM budget. | Seed-control runs as a subprocess (`build_o_arms.py seed-control`). The driver refuses to take the lease if torch is in its `sys.modules`. Measured: the driver imports no torch. |
| F10 | The P4 label was issued even when the denominator gate failed. FOLLOWUP_PROPOSAL says P4 is undefined in that case. | The label is `UNDEFINED_DENOMINATOR_GATE_FAILED` unless the gate passes. The number P_O^content(H1) is still reported. |
| F11 | By-token P_O^content covered 5 of the 9 non-reference arms. PREREG §7 says "per arm". | It now covers every non-reference arm. |
| — | A final-analysis HARNESS_FAILURE would have occupied `O_SIDE_RESULT.json`, which preflight treats as "run complete", so the permitted rerun would be refused. Found while applying the fixes. | A harness-failure result is written to `O_SIDE_HARNESS_FAILURE_<time>.json` instead. |

**Refuted.**
- F4: `compose.py`'s working-tree edit does not touch `qwen_to_ced`, and the positive control ran against that
  same file.
- F12: every non-crash stop already reads nothing, or is terminal by design.

## PREREG §0, as amended

A rerun is permitted only after a **harness failure**, which is one of:
- a crash, including a non-finite NLL, which the scorer now turns into a crash;
- a canary FAIL;
- a provenance failure.

**Rules for a rerun:**
- A rerun resumes after the last **complete** score file, as defined above.
- **It never rescores a complete file still in `scores/`.** The only files ever rescored are ones the driver itself
  moved to `scores_failed/` (never deleted):
  - an incomplete file;
  - every file of a stage whose canary or provenance check failed. The stage-1 canary moves BASE and OPALLO
    together, even if only one of them failed. The permitted rerun rescores exactly the moved files.
- **A built arm's fresh score file is checked before its weights are deleted.** A failure stops the attempt with
  the weights kept, so the rerun moves the file aside and rescores it.
- **A final-analysis HARNESS_FAILURE ends the attempt**, and `O_SIDE_HARNESS_FAILURE_<time>.json` is kept.
  - Every per-file limb is enforced before a file is accepted, and the device is pinned per file. So this failure
    is reachable only by editing a file after it passed those checks.
  - Recovery then needs an operator decision, recorded in `ATTEMPT_LOG.md`: move the offending score files, and
    for a built arm also its `arms/<ARM>` directory, to `scores_failed/`. The driver never makes this choice
    itself.
- Files moved aside by the driver are kept under `scores_failed/` and committed with the run.
- Every attempt, and every move or removal the driver logs, is recorded in `ATTEMPT_LOG.md`.

## Second review: of this amendment's diff, before the run

A smaller workflow reviewed the diff between 962abee4 and this amendment. It had three lenses:
- **Regressions:** on synthetic inputs, the amended and registered analyses give identical values on every shared
  key, including D, the intervals, the calibration, the MDE, the gate and the verdicts. The only differences are
  the items this file names.
- **Fix closure:** every F1–F11 scenario was re-created against the amended code, and all are closed.
- **Failure injection:** a crash was injected at every step boundary of the driver.

It confirmed 6 findings and refuted none. All 6 are minor, and none changes a verdict or the run's integrity.
They collapse into three changes:

| Finding (verified) | Change |
|---|---|
| A final-analysis or cross-attempt provenance failure could not be cleared by the permitted rerun. The cross-file "one device" limb was invisible to the per-file resume check. A built arm's score file was first checked only after its weights were deleted. (Three lenses found this independently.) | **The GPU name is pinned per file** to `NVIDIA GeForce RTX 4070 Laptop GPU`, the device recorded in `O_REPLY_NLL.json`. The scorer refuses to run on any other device, and the provenance check tests it per file (the one-device limb stays as a redundant check). **The driver checks each built arm's fresh score before deleting its weights.** §0 above now states what the driver does and does not recover. |
| A torn BUILD.json (empty or zero-filled after an OS crash; it was not fsynced) crashed every rerun. For a seed arm the crash hit every stage canary, through `expected_hashes()`. | A BUILD.json that is missing, unparseable, or lacks a weight hash counts as an unverified partial build: it is removed (strictly inside `arms/`) and rebuilt. `expected_hashes()` and `expected_sha()` return None for it, which makes it a provenance failure instead of a crash. The driver fsyncs every file of a finished build. `build_o_arms.py` is unchanged. |
| The text said "a complete file is never rescored", which contradicts the canary move-aside rule that the code implements. | Wording only: see §0 above, the F9 row, `ATTEMPT_LOG.md` and the driver docstring. |

## Verification of this amendment, before the run

- **`test_o_side_offline.py` passes 23/23:** the original 12 plus 11 new tests. Each guard above is shown to fail
  on a defective input and to pass on a clean one:
  - a NaN in a contrast arm, and anywhere in a canary arm;
  - six provenance mutations, including the device, which is now a per-file failure;
  - the P4 label with the gate failing and passing;
  - by-token coverage;
  - an incomplete score moved aside and rescored to completion;
  - a canary failure moving its stage's files;
  - rebuilt-arm guards: a partial build rebuilt, scored with the registered `--expect-sha` and then deleted; a
    complete arm left untouched; an uncertified BUILD.json refused; `REBUILD_MISMATCH` refused;
  - torn BUILD.json files (a 0-byte one for H1, a zero-filled one for SEED23) rebuilt rather than crashing;
  - a failing fresh score (a wrong row count, or a wrong device) stopping before the delete with the weights kept,
    then moved aside and rescored on the rerun;
  - no torch in the driver.
- The driver tests run in a temporary sandbox with a stub runner that really creates build directories, score
  files and deletions. The sandbox is removed afterwards.
- **GPU smoke of the amended scorer** (BASE, first 5 rows, to scratch, under the lease):
  - max |diff| against `O_REPLY_NLL.json` is 0.0;
  - the input hashes were recorded (e4b2b77f / cd57e650 / 65e79776);
  - the runtime pin passed;
  - the file's `torch.cuda.get_device_name(0)` is exactly `NVIDIA GeForce RTX 4070 Laptop GPU`, the string the new
    device pin requires;
  - the provenance check rejected the smoke file only for its row limit and row count, as designed.
  - The device pin was added after this smoke. It compares the same call's output with that same string, and
    stage 1 of the run exercises it first, on an exposed arm (BASE), before any contrast arm exists.
