# Short-Response Model — Phase 1 Dense SFT Package

This package preserves the completed Phase 0 data tooling and adds the complete
Phase 1 workflow required by the research blueprint.

## Current stage

- **Phase 0:** implementation prepared; run validation, leakage checks, and human review.
- **Phase 1:** implementation prepared; run the untouched baseline, train the
  all-layer QLoRA reference, evaluate it, and review Stop Gate 1.
- **Phase 2:** present but blocked until a human records `PASS` in the Phase 1 gate report.

Artifacts are intentionally excluded from this ZIP and will be recreated locally.

## Organized script layout

```text
scripts/
├── phase_0_data/
├── phase_1_dense_sft/
└── phase_2_layer_selection/
```

## Run Phase 0

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\phase_0_data\run_phase_0.ps1
```

Review `reports\HUMAN_DATA_AUDIT_PACKET.md`.

## Run Phase 1

```powershell
.\scripts\phase_1_dense_sft\run_phase_1.ps1
```

This performs:

1. CUDA/model smoke inference.
2. Untouched base-model evaluation.
3. All-layer QLoRA SFT training.
4. Adapter evaluation on the same benchmark.
5. Automated Stop Gate 1 comparison.

Then review `reports\PHASE_1_STOP_GATE.md` and change the human decision from
`PENDING` to `PASS` only when the generated replies are clearly better, naturally
brief, complete, and varied.

## Start Phase 2 later

```powershell
.\scripts\phase_2_layer_selection\run_layer_scan.ps1
```

The script blocks execution until Phase 1 has an explicit human `PASS`.


## Phase 2

The complete static layer-selection workflow is in `scripts/phase_2_layer_selection/`. Run `run_phase_2.ps1` only after recording the Phase 1 human PASS. See `PHASE_2_COMPLETION_NOTES.md`.
