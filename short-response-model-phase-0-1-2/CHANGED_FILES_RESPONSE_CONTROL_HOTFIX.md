# Changed Files — Response-Control Hotfix

## New files

- `pytest.ini` — makes both `pytest -q` and `python -m pytest -q` resolve repository imports consistently.
- `scripts/__init__.py` — explicit package marker for IDEs and direct tooling.
- `scripts/common/response_control.py` — shared EOS/validated-boundary controller, stopping criterion, finalizer, stop-reason telemetry, and protected punctuation contexts.
- `configs/response_control.json` — release-time response-control policy contract.
- `scripts/phase_7_sparse_runtime/RESPONSE_CONTROL_CONTRACT.md` — evidence boundaries and deployment rules.
- `scripts/phase_7_sparse_runtime/verify_response_control.py` — lightweight deterministic contract verification.
- `tests/test_response_control.py` — regression tests for sentence-boundary behavior.
- `RESPONSE_CONTROL_AND_EARLY_EXIT_AUDIT.md` — audit of all three proposed methods.
- `RELEASE_VERIFICATION_v1.1.md` — completed checks and explicit target-machine limitations.
- `CHANGED_FILES_RESPONSE_CONTROL_HOTFIX.md` — this file.

## Modified Python files

- `scripts/phase_1_dense_sft/evaluate_model.py`
  - Adds `eos_only` versus `one_sentence` modes.
  - Applies a Hugging Face stopping criterion in strict mode.
  - Finalizes visible output at the first validated boundary.
  - Separates actual generated tokens from visible reply tokens.
  - Records stop reasons, trimming, sentence count, and violation rate.
  - Tightens the complete-utterance heuristic so punctuation-only output is not accepted.
- `scripts/phase_1_dense_sft/smoke_inference.py`
  - Uses the strict controller by default and records stop telemetry.
- `scripts/phase_5_context_system/evaluate_context_system.py`
  - Uses identical response control for compact/full context comparisons and records generation metadata.
- `scripts/phase_6_sparse_simulation/simulate_sparsity.py`
  - Uses identical response control for dense/masked comparisons so sparsity is not favored by a different stopping rule.
- `scripts/phase_7_sparse_runtime/benchmark_runtime.py`
  - Adds incremental validated-boundary stopping to the manual KV-cache decode loop.
  - Measures visible/actual tokens, stop reasons, trim rate, and one-sentence violations for dense and sparse paths.
  - Adds a response-control pass target.
- `scripts/phase_7_sparse_runtime/check_phase_7_gate.py`
  - Blocks Phase 7 approval when the production response-control contract was not measured or passed.
- `scripts/phase_8_release_candidate/build_release_audit.py`
  - Requires sealed `one_sentence` evaluation with zero one-sentence violations.
- `scripts/phase_8_release_candidate/package_release_candidate.py`
  - Packages the response policy and contract and refuses a release audit that did not validate strict mode.
  - Recreates the dedicated package directory to prevent stale model/router files from a previous package mode.

## Modified PowerShell runners

- `verify_post_phase_3.ps1` — now runs deterministic response-control contract verification in addition to compilation, regression tests, and runner-reference checks.

- `scripts/phase_1_dense_sft/01_smoke_inference.ps1`
- `scripts/phase_1_dense_sft/02_evaluate_base.ps1`
- `scripts/phase_1_dense_sft/04_evaluate_dense_sft.ps1`
- `scripts/phase_5_context_system/run_phase_5.ps1`
- `scripts/phase_6_sparse_simulation/run_phase_6.ps1`
- `scripts/phase_7_sparse_runtime/run_phase_7.ps1`
- `scripts/phase_8_release_candidate/run_phase_8.ps1`

Phase 1 research comparisons are pinned to `eos_only`; production-oriented phases are pinned/defaulted to `one_sentence`. Phase 7 now runs the deterministic controller verification before runtime benchmarking.

## Modified documentation

- `README.md`
- `POST_PHASE_3_RUNNABILITY_VERIFICATION.md`
- `Phases.md`
- `scripts/phase_1_dense_sft/README.md`
- `scripts/phase_5_context_system/README.md`
- `scripts/phase_6_sparse_simulation/README.md`
- `scripts/phase_7_sparse_runtime/README.md`
- `scripts/phase_7_sparse_runtime/EXECUTION_MANUAL.md`
- `scripts/phase_7_sparse_runtime/FILE_RESPONSIBILITIES.md`
- `scripts/phase_8_release_candidate/README.md`
- `scripts/phase_8_release_candidate/EXECUTION_MANUAL.md`
- `scripts/phase_8_release_candidate/FILE_RESPONSIBILITIES.md`

## Intentionally not changed

- No auxiliary intermediate exit head was added.
- No transformer layer was taught to act as a punctuation detector.
- No claim was made that selective-layer training reduces dense inference compute.
- Phase 4B long-form calibration remains EOS/cap controlled because it intentionally asks the model to continue for component profiling.
