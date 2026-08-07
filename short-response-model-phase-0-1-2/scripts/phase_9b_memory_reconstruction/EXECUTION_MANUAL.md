## v1.7.4 quick path

The dedicated reconstruction corpus is now the default. A normal research run no longer requires manually supplying a reviewed reconstruction file.

```powershell
.\scripts\phase_9b_memory_reconstruction\run_phase_9b.ps1 `
  -Mode Full `
  -UseLora `
  -Teacher4Bit `
  -OutputRoot .\artifacts\phase_9b_memory_reconstruction_v1_7_4
```

Optional curated reconstruction preference alignment is enabled only after SFT/KD:

```powershell
.\scripts\phase_9b_memory_reconstruction\run_phase_9b.ps1 `
  -Mode Full `
  -UseLora `
  -Teacher4Bit `
  -RunPreferenceAlignment `
  -ReuseExisting `
  -OutputRoot .\artifacts\phase_9b_memory_reconstruction_v1_7_4
```

DEV selects the strategy. `phase_9b_dev_selection.json` freezes it before the runner touches the sealed TEST split. `Prepare` remains available when you want to create additional project-specific human-review candidates from Phase 9 RAG memories; it no longer replaces the integrated curriculum.

# Phase 9B execution manual


## v1.7.3 RAG-reasoning source corpus

Phase 9B now defaults to the Phase 9 `rag_reasoning_v2` prompt/memory corpus. The candidate builder preserves authored `intended_split` metadata and performs applicability checks in this order: reject untrusted/injection memories, invalidate superseded memories for the current scope, apply explicit validity conditions, then detect conflicts among the remaining applicable memories. This prevents stale/current evidence from being mislabeled as an equal-authority conflict.

The corpus is a reconstruction seed rather than an automatic ground-truth reconstruction set: generated ACCEPT/ADAPT/REJECT/CONFLICT/INSUFFICIENT targets still require human review before Phase 9B training.

## Prerequisites

1. Phase 9 has explicit human PASS and `artifacts/references/phase_9_student_reference.json` exists.
2. The selected Phase 9 student is materialized as a complete local checkpoint.
3. The authoritative teacher checkpoint remains available for candidate generation and optional KD.
4. Phase 9B rows are reviewed by a human before training.
5. On an RTX 4070 8 GB, use LoRA and teacher 4-bit loading.

## B0 — Build candidates and teacher review packet

```powershell
.\scripts\phase_9b_memory_reconstruction\run_phase_9b.ps1 `
  -Mode Prepare `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model
```

This writes both a conservative metadata candidate and a teacher candidate. Neither is a training label yet. For every row, inspect:

- whether the historical source state matches the current state;
- whether a memory should be accepted, adapted, or rejected;
- whether multiple sources conflict;
- whether retrieved instructions are being ignored;
- whether every source ID and final answer are supported.

Save a reviewed JSONL by setting `review_status` to `accepted` or `rejected`. Accepted rows require `approved_reconstruction` and `approved_answer`.

## B1 — Build the leakage-safe corpus and train SFT

```powershell
.\scripts\phase_9b_memory_reconstruction\run_phase_9b.ps1 `
  -Mode Train `
  -ReviewedReconstructions .\artifacts\phase_9b_memory_reconstruction\teacher_reconstruction_candidates.reviewed.jsonl `
  -TeacherModel .\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model `
  -UseLora `
  -Teacher4Bit `
  -MaxSteps 400
```

Each accepted canonical row creates three training tasks:

1. raw memory plus current state → structured reconstruction JSON;
2. reconstructed guidance → short final answer;
3. raw memory plus source state → short direct answer for Phase 10 compatibility.

The builder also writes DPO preference rows contrasting the approved reconstruction with a naive replay baseline. DPO is optional and should be run only after SFT/KD evidence is understood.

## B2 — Evaluate all strategies

```powershell
.\scripts\phase_9b_memory_reconstruction\run_phase_9b.ps1 `
  -Mode Evaluate `
  -ReviewedReconstructions .\artifacts\phase_9b_memory_reconstruction\teacher_reconstruction_candidates.reviewed.jsonl `
  -UseLora `
  -Teacher4Bit
```

The evaluator compares:

- `memory_off`;
- `raw_replay`;
- `prompted_reconstruction` using the unmodified Phase 9 student;
- `trained_reconstruction` using the Phase 9B adapter and two passes;
- `direct_candidate` using the Phase 9B adapter in one pass.

The main measurements are functional success, reconstruction decision accuracy, JSON parse validity, source-ID precision/recall, negative-transfer rate, raw-failure rescue, raw-success harm, latency, prompt tokens, generated tokens, and peak VRAM.

## B3 — Full run

After a reviewed packet exists:

```powershell
.\scripts\phase_9b_memory_reconstruction\run_phase_9b.ps1 `
  -Mode Full `
  -ReviewedReconstructions .\artifacts\phase_9b_memory_reconstruction\teacher_reconstruction_candidates.reviewed.jsonl `
  -UseLora `
  -Teacher4Bit `
  -MaxSteps 400
```

Without `-ReviewedReconstructions`, Full intentionally stops rather than training on unreviewed teacher output.

## B4 — Smoke test

```powershell
.\scripts\phase_9b_memory_reconstruction\run_phase_9b.ps1 `
  -Mode Train `
  -SmokeTest `
  -UseLora `
  -SkipKD `
  -MaxSteps 20
```

This uses `data/phase_9b_memory_reconstruction/reviewed.sample.jsonl`. It verifies code paths only and cannot support a research or release decision.

## B5 — Human handoff

After reviewing `reports/PHASE_9B_STOP_GATE.md`, mark an explicit PASS and materialize the selected checkpoint:

```powershell
python .\scripts\phase_9b_memory_reconstruction\finalize_phase_9b_selection.py `
  --candidate trained_reconstruction `
  --role phase10_candidate `
  --reason "Reduced negative transfer while the direct candidate remained worthwhile."
```

This writes `artifacts/references/phase_9b_student_reference.json`. Phase 10 prefers that manifest when present and otherwise falls back to the Phase 9 reference.

## Interpretation

- Prompted reconstruction wins but trained reconstruction does not: keep the runtime prompt and reject the adapter.
- Two-pass wins but direct candidate regresses: retain a runtime-only research result; do not silently pass it to Phase 10.
- Raw replay beats memory-off but reconstruction still helps: memory is useful, and applicability training adds value.
- Raw replay is worse than memory-off: the corpus exposes negative transfer; prioritize counterfactual and stale-state examples.
- JSON parse failures are common: strengthen format SFT before increasing KD or attempting RL.
