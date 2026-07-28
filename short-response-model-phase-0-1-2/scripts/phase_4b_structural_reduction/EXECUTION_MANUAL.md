# Phase 4B execution manual

## 1. Prerequisites

- Phase 4 has an explicit human `PASS`.
- The accepted Phase 4 model and tokenizer can reload in a fresh process.
- The long-form calibration set contains both compact assistant targets and genuinely longer assistant targets.
- Calibration and development data are used; sealed Phase 8 evaluation remains untouched.
- Enough disk space exists for the dense reference plus multiple full checkpoints.

## 2. Preserve the source reference

Record the Phase 4 model path and hashes. Do not modify or save into that directory. Every Phase 4B candidate is written under:

```text
artifacts/phase_4b_structural_reduction/candidates/
```

## 3. Run the pilot first

```powershell
.\scripts\phase_4b_structural_reduction\run_phase_4b.ps1 `
  -Model <accepted-phase-4-model> `
  -Calibration data\phase_4b_structural_reduction\long_form_calibration.jsonl `
  -Mode Pilot
```

Use `-ReuseExisting` only when the existing candidate, evaluation, and footprint files were produced by the same code, model, benchmark, and settings.

## 4. Inspect evidence in this order

1. `long_form_profile.pt` and its summary: confirm short and late observations exist.
2. Candidate manifests: confirm smoke and reload checks passed.
3. Footprints: confirm parameter and serialized-byte reduction are real.
4. `summary.json` for each candidate: compare to dense under identical settings.
5. `results.jsonl`: manually inspect the lowest-scoring and safety-sensitive replies.
6. Recommendation report: treat it as a filter, not an approval.

## 5. Optional expansion

Run `-Mode Full` only after the pilot has at least one promising candidate. Enable attention pruning separately:

```powershell
.\scripts\phase_4b_structural_reduction\run_phase_4b.ps1 `
  -Model <accepted-phase-4-model> `
  -Calibration <calibration-jsonl> `
  -Mode Full `
  -EnableAttentionPruning
```

Attention pruning is more architecture-sensitive and should remain opt-in.

## 6. Recovery tuning

A passing reduced candidate may be recovery-tuned or distilled, but preserve the untouched reduced checkpoint as a control. Any recovery-trained model must be registered as a new candidate and reevaluated through `evaluate_candidates.py`.

## 7. Human decision

The decision report must name:

- selected candidate;
- dense source model;
- quality and category deltas;
- parameter, disk, VRAM, and latency changes;
- known regressions and rollback path;
- reviewer and date.

Then add exactly `- Human decision: PASS` and run `finalize_reference_selection.py`.

## 8. Downstream validation

Before Phase 5, verify:

```powershell
python .\scripts\common\check_human_gate.py --phase 4B
python .\scripts\common\resolve_reference_model.py
```

The second command must print the model chosen in the Phase 4B report.

## Stop conditions

Stop increasing structural reduction when any hard quality floor fails, when reload compatibility becomes fragile, or when measured user-visible benefit is too small to justify a custom checkpoint architecture.
