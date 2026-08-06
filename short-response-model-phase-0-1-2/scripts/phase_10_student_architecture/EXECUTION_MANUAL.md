# Phase 10 execution manual

## Preconditions

- `reports/PHASE_9_STOP_GATE.md` contains explicit human `PASS`.
- `artifacts/references/phase_9_student_reference.json` exists.
- Phase 9 train/dev/test, knowledge index, and gate calibration exist.
- The original accepted teacher is available locally.

## Phase 9 handoff

For a hybrid Phase 9 assessment, materialize the exact approved key before running Phase 10:

```powershell
python .\scripts\phase_10_student_architecture\finalize_phase_9_selection.py `
  --candidate kd_warm_344m `
  --role intermediate_teacher `
  --reason "Approved warm-profile KD student."
```

Legacy names such as `kd` remain valid when the single-student suite was used. A key listed in the assessment's `research_controls` must use `--role research_control`, unless the reviewer deliberately supplies `--allow-research-control-promotion`.

## Commands

Pilot:

```powershell
.\scripts\phase_10_student_architecture\run_phase_10.ps1 -Mode Pilot -TeacherModel <teacher-path>
```

Full matrix:

```powershell
.\scripts\phase_10_student_architecture\run_phase_10.ps1 -Mode Full -TeacherModel <teacher-path> -MaxSteps 1000 -RecoveryTopK 4
```

Resume:

```powershell
.\scripts\phase_10_student_architecture\run_phase_10.ps1 -Mode Full -TeacherModel <teacher-path> -ReuseExisting
```

Optional LoRA recovery trades lower training state for a base+adapter handoff:

```powershell
.\scripts\phase_10_student_architecture\run_phase_10.ps1 -Mode Pilot -TeacherModel <teacher-path> -UseLora
```

## Review

Inspect candidate manifests, raw/recovered outputs, all four RAG conditions, actual parameter/byte reductions, and the shared opportunity assessment. A smoke reload is technical validity, not promotion.

## Finalize

After human `PASS`:

```powershell
python .\scripts\phase_10_student_architecture\finalize_phase_10_selection.py --candidate <registry-name> --role deployment_candidate --reason "<decision>"
```
