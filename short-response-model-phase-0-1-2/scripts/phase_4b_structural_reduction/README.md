# Phase 4B — Structural Reduction and Reference Selection

Phase 4B is an integrated production-oriented decision point between model construction (Phases 1–4) and runtime optimization (Phases 5–7).

Its job is simple:

> Test whether the accepted Phase 4 model can be made physically smaller without losing the short-response behavior needed by the product, then select exactly one checkpoint for all downstream work.

A reduced model is **not required** to pass Phase 4B. Keeping the accepted dense Phase 4 checkpoint is the correct result when pruning or depth reduction does not create enough measured value.

## Why this phase is after Phase 4

- DPO and soup merging can change weights and activation patterns.
- Soup ingredients require compatible shapes, so pruning them independently would break merging.
- Phase 6 activation traces must be collected from the exact model later used by the router.

Therefore the order is:

```text
Phase 4 accepted checkpoint
        ↓
Phase 4B structural candidates and selection
        ↓
selected_inference_reference.json
        ↓
Phases 5, 6, 7, and 8
```

## What Phase 4B tests

1. MLP channels that appear disproportionately useful for long continuation but less useful for compact answers.
2. Equal-size random MLP pruning as a control.
3. Removing the final 1/2/4 decoder layers.
4. Uniformly distributed layer removal with the same retained layer count.
5. Optional grouped-query-attention head-group pruning.

Selected-layer SFT remains in Phase 2. Dynamic MLP block routing remains in Phases 6–7.

## Quick pilot

```powershell
.\.venv\Scripts\Activate.ps1

.\scripts\phase_4b_structural_reduction\run_phase_4b.ps1 `
  -Model artifacts\phase_4_soup\merged_model `
  -Calibration data\phase_4b_structural_reduction\long_form_calibration.jsonl `
  -Mode Pilot
```

Pilot mode builds:

- 10% attribution-guided MLP pruning;
- 10% random MLP pruning control;
- drop-last-2 depth reduction;
- uniform-drop-2 depth control;
- untouched dense reference.

Full mode expands MLP ratios to 5%, 10%, and 20%, and depth removal to 1, 2, and 4 layers.

## Candidate review and finalization

The runner writes:

```text
artifacts/phase_4b_structural_reduction/candidate_registry.json
artifacts/phase_4b_structural_reduction/candidate_recommendation.json
reports/PHASE_4B_CANDIDATE_RECOMMENDATION.md
```

After reviewing raw replies and real hardware measurements, update:

```text
reports/PHASE_4B_STOP_GATE.md
```

Add exactly:

```text
- Human decision: `PASS`
```

Then finalize the chosen checkpoint:

```powershell
python .\scripts\phase_4b_structural_reduction\finalize_reference_selection.py `
  --registry artifacts\phase_4b_structural_reduction\candidate_registry.json `
  --candidate mlp_long_specific_10pct `
  --reason "Passed quality floors and reduced parameters enough to justify deployment complexity."
```

Choosing `dense_reference` is supported:

```powershell
python .\scripts\phase_4b_structural_reduction\finalize_reference_selection.py `
  --registry artifacts\phase_4b_structural_reduction\candidate_registry.json `
  --candidate dense_reference `
  --reason "No reduced candidate produced sufficient practical benefit."
```

The final command creates:

```text
artifacts/references/selected_inference_reference.json
```

Phases 5–8 use that manifest by default.

## Safety and product rules

- Never overwrite the accepted Phase 4 checkpoint.
- A smoke forward verifies tensor compatibility, not user-facing quality.
- Zero critical semantic failures are required.
- Inspect factual, context, safety, ambiguity, and multilingual cases manually.
- Do not select a smaller model merely because the checkpoint is smaller.
- Stop when practical latency/VRAM/size gain does not justify maintenance risk.
- Any weight change after Phase 6 invalidates the collected activation traces and trained routers.

See `FILE_RESPONSIBILITIES.md` for each file's exact role and `EXECUTION_MANUAL.md` for the operating procedure.
