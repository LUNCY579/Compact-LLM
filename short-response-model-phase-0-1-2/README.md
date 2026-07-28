# Short-Response Model — Mature Phase Pipeline

This repository implements the ordered short-response model program and integrates permanent architecture reduction as **Phase 4B**.

The production goal is not to force every experimental technique into the shipped model. The goal is to select the simplest checkpoint that proves useful quality, latency, memory, and maintenance characteristics on the target hardware.

## Phase folders

- `phase_0_data`: schema validation, leakage checks, and human audit.
- `phase_1_dense_sft`: dense baseline, concise-behavior SFT, and evaluation.
- `phase_2_layer_selection`: one/two/four-layer training and module-group controls.
- `phase_3_preference_alignment`: DPO for concise-correct behavior.
- `phase_4_model_soups`: compatible adapter validation and delta-space soup construction.
- `phase_4b_structural_reduction`: permanent MLP/head/layer reduction, common candidate evaluation, and downstream checkpoint selection.
- `phase_5_context_system`: retrieval, context budgeting, and prefix/KV reuse.
- `phase_6_sparse_simulation`: activation profiling and masked quality frontier.
- `phase_7_sparse_runtime`: confidence-calibrated routing and measured sparse-decode reference.
- `phase_8_release_candidate`: final ablations, hashes, safety review, and release decision.

Every phase is gated. Phase 4B may select the untouched dense Phase 4 model when reduced candidates do not create enough practical value.

## Canonical model handoff

Phase 4B writes:

```text
artifacts/references/selected_inference_reference.json
```

Phases 5–8 resolve this file by default. This prevents activation traces, routers, benchmarks, and release hashes from accidentally using different checkpoints.

## Typical execution

```powershell
.\.venv\Scripts\Activate.ps1

# Continue normal training/alignment.
.\scripts\phase_3_preference_alignment\run_phase_3.ps1
.\scripts\phase_4_model_soups\run_phase_4.ps1

# Build and evaluate structural candidates after Phase 4 approval.
.\scripts\phase_4b_structural_reduction\run_phase_4b.ps1 `
  -Model artifacts\phase_4_soup\merged_model `
  -Calibration data\phase_4b_structural_reduction\long_form_calibration.jsonl `
  -Mode Pilot

# After reviewing and approving Phase 4B, finalize the selected model.
python .\scripts\phase_4b_structural_reduction\finalize_reference_selection.py `
  --registry artifacts\phase_4b_structural_reduction\candidate_registry.json `
  --candidate dense_reference `
  --reason "Dense reference remains the best practical deployment checkpoint."

# Later phases automatically use the selected model.
.\scripts\phase_5_context_system\run_phase_5.ps1
.\scripts\phase_6_sparse_simulation\run_phase_6.ps1 `
  -Calibration data\calibration\calibration.jsonl
.\scripts\phase_7_sparse_runtime\run_phase_7.ps1
.\scripts\phase_8_release_candidate\run_phase_8.ps1
```

## Documentation map

Each phase contains a README and execution manual. Phase 4B additionally includes `FILE_RESPONSIBILITIES.md`, which documents the owner, inputs, outputs, and prohibited claims for every file.

Generated model checkpoints and benchmark artifacts are intentionally not included in this ZIP. They must be produced on your machine so measurements reflect your hardware and accepted prior-phase artifacts.
