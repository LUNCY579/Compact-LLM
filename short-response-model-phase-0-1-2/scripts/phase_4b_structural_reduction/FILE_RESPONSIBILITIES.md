# Phase 4B file responsibilities

This document answers two questions for every file: **what it owns** and **what it must not claim**.

| File | Responsibility | Inputs | Outputs | Must not do |
|---|---|---|---|---|
| `run_phase_4b.ps1` | Orchestrate the pilot/full candidate matrix after checking Phase 4 approval. | Accepted Phase 4 model, calibration JSONL, benchmark. | Candidate checkpoints, common evaluations, recommendation. | Approve Phase 4B or overwrite the Phase 4 checkpoint. |
| `profile_long_form_components.py` | Teacher-force short and long targets and record early/late MLP-channel and attention-head contribution proxies. | Model and Phase 4B calibration JSONL. | `long_form_profile.pt` plus summary metadata. | Assert that a component has one human-readable purpose. |
| `build_pruning_plan.py` | Convert profile statistics into equal-width per-layer keep/prune indices. | Profile, ratios, selection method. | JSON pruning plan. | Change model weights. |
| `apply_structured_pruning.py` | Physically slice gated-MLP channels and optional complete GQA groups; smoke-test and reload. | Source model and pruning plan. | Smaller checkpoint and manifest. | Claim quality, latency, or VRAM improvement from a smoke test. |
| `reduce_depth.py` | Remove complete decoder layers using suffix, uniform, or explicit policies; smoke-test and reload. | Source model and reduction policy. | Shallower checkpoint and manifest. | Treat selected training layers as removable inference layers without evaluation. |
| `measure_checkpoint.py` | Measure parameter count, tensor bytes, dtypes, and local serialized bytes. | Any saved candidate. | Footprint JSON. | Estimate runtime speed from parameter count alone. |
| `evaluate_candidates.py` | Reuse the official evaluator and footprint tool for every candidate under identical settings. | Dense reference, candidate specs, benchmark. | Evaluations and `candidate_registry.json`. | Introduce a separate hidden scoring system. |
| `recommend_reference_candidate.py` | Apply configurable practical quality floors and recommend the best eligible reduced model, or dense fallback. | Candidate registry. | JSON and Markdown recommendation. | Grant a human gate or force a reduced model when none is worthwhile. |
| `finalize_reference_selection.py` | Verify the Phase 4B human PASS and write the canonical downstream model manifest. | Registry, chosen name, decision report. | `artifacts/references/selected_inference_reference.json`. | Select an unmeasured candidate or run before approval. |

## Supporting shared files

| File | Why Phase 4B uses it |
|---|---|
| `scripts/common/model_introspection.py` | Finds decoder/MLP/attention modules, slices linear layers, and updates model configuration safely. |
| `scripts/common/reference_selection.py` | Defines and validates the selected-reference manifest consumed by Phases 5–8. |
| `scripts/common/resolve_reference_model.py` | Lets PowerShell runners resolve the selected model without duplicating JSON parsing. |
| `scripts/phase_1_dense_sft/evaluate_model.py` | Provides the common behavioral evaluator, ensuring Phase 4B candidates are comparable with earlier models. |

## Ownership rule

Phase 4B owns **permanent architecture changes**. Phase 2 owns **which layers are trained**. Phase 6 owns **temporary activation masking**. Phase 7 owns **dynamic routed execution**. Keeping these responsibilities separate prevents a failure from being attributed to the wrong mechanism.
