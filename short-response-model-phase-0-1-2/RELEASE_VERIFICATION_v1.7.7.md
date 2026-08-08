# Release Verification — v1.7.7 Phase 9 SFT Selection Gate Hotfix

## Scope

This release fixes the Phase 9 grounded-recovery Stage-B/Stage-D policy mismatch without changing training data, SFT sweep hyperparameters, sealed-test boundaries, or trained-candidate quality requirements.

## Verified behavior

- Stage-B initialization prompt/decode integrity issues are recorded in `initialization_integrity_warnings` and are explicitly non-blocking for Stage-D selection.
- Initialization behavior-coverage failures remain in `initialization_blocking_issues` and still invalidate the comparison reference.
- Candidate-owned prompt/decode integrity remains fatal to that candidate.
- Candidate-owned behavior coverage, non-degeneracy, behavior success, and positive Oracle capability improvement remain required.
- A synthetic reproduction of the reported shape (dirty Stage-B decode integrity, candidate scores 0.66 / 0.85 / 1.0) selects `lr1e4_s1200`.
- The training-free `reselect_existing_grounded_sft.py` utility overwrites a stale `NO_VALID_QUALITY_FRONTIER` selection from existing candidate summaries and selects the strongest valid candidate.
- `run_phase_9_grounded_recovery.ps1 -ReuseExisting` recomputes derived Stage-D and final DEV selections instead of reusing stale selector outputs.
- The Stage-D ladder checker already evaluates the SFT summary on its own integrity and therefore required no weakening.

## Executed verification

```text
python -m compileall -q scripts/phase_9_student_distillation tests/test_phase_9_sft_selection_gate_v177.py
python -m pytest -q
```

Result: **231 passed**.

The environment used for package verification did not provide PowerShell (`pwsh`), so the PowerShell wrapper was verified by repository-wide static tests rather than executed end-to-end here. The actual model/GPU artifacts from the user's existing run were not included in the uploaded source package, so the stated `lr1e4_s1200` on-disk run result is not independently re-evaluated in this release package; the selector behavior is covered by a faithful synthetic regression test.
