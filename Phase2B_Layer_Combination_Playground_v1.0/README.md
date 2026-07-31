# Phase 2B Layer-Combination Playground v1.0

This is an **independent experiment**, not another project phase. It never edits
Phase 2B reports, gates, catalogs, accepted frontier checkpoints, or downstream
profiles.

## What it tests

The quick run dropped:

```text
[26, 13, 14, 10, 6, 19]
```

The thorough V2 run dropped:

```text
[24, 23, 14, 19, 3, 13, 10, 21, 22, 5]
```

The default `core` suite creates raw checkpoints from the **same full SFT
reference** and evaluates:

| Variant | Dropped original layers | Question |
|---|---|---|
| `common_only` | `[10, 13, 14, 19]` | Is the overlap especially stable? |
| `quick_unique_only` | `[6, 26]` | What do only the quick-specific layers do? |
| `thorough_unique_only` | `[3, 5, 21, 22, 23, 24]` | What do only the thorough-specific layers do? |
| `nonoverlap_all` | `[3, 5, 6, 21, 22, 23, 24, 26]` | Can all disagreement layers be removed together? |
| `quick_set_raw` | quick set | What happens without the quick path's recovery history? |
| `thorough_set_raw` | thorough set | What happens without the thorough path's recovery history? |
| `union_all` | all 12 layers | Can every layer selected by either run be removed together? |

The `layer26` suite additionally isolates the marginal effect of removing layer
26 on top of the thorough structural set.

## Installation

Extract this folder directly inside the repository root:

```text
short-response-model-phase-0-1-2/
├─ .venv/
├─ scripts/
├─ artifacts/
└─ Phase2B_Layer_Combination_Playground_v1.0/
```

The experiment writes only to:

```text
playground_artifacts/phase2b_layer_combinations/
```

## Preview the plan without loading a model

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\Phase2B_Layer_Combination_Playground_v1.0\run_playground.ps1 `
  -PlanOnly
```

## Recommended first run

This evaluates the seven core structural controls on the 150-case development
probe:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\Phase2B_Layer_Combination_Playground_v1.0\run_playground.ps1 `
  -Suite core `
  -CaseLimit 150
```

For a cheaper smoke experiment, use `-CaseLimit 25` or `50` first.

## Focus specifically on layer 26

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\Phase2B_Layer_Combination_Playground_v1.0\run_playground.ps1 `
  -Suite layer26 `
  -CaseLimit 150
```

This compares the raw thorough set against:

```text
thorough + drop 6
thorough + drop 26
thorough + drop 6 and 26
```

## Preserve recovery and extend the actual thorough endpoint

The raw controls intentionally remove recovery as a confounder. After those
finish, this optional run starts from the saved recovered thorough endpoint and
then removes layer 6, layer 26, or both:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\Phase2B_Layer_Combination_Playground_v1.0\run_playground.ps1 `
  -Suite layer26 `
  -CaseLimit 150 `
  -IncludeExistingEndpoints `
  -IncludeRecoveredExtensions
```

This requires the local thorough V2 catalog and frontier checkpoints.

## Expensive follow-up only when the union regresses

`-LeaveOneInAblation` creates 12 extra variants. Each variant removes the full
union except one original layer. This can identify which retained layer most
repairs the union candidate, but it is deliberately not part of the default run.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\Phase2B_Layer_Combination_Playground_v1.0\run_playground.ps1 `
  -Suite core `
  -CaseLimit 150 `
  -LeaveOneInAblation
```

## Resume

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\Phase2B_Layer_Combination_Playground_v1.0\run_playground.ps1 `
  -Suite core `
  -CaseLimit 150 `
  -Resume
```

Resume reuses completed model manifests and evidence summaries.

## Outputs

```text
playground_artifacts/phase2b_layer_combinations/
├─ experiment_plan.json
├─ summary.json
├─ PLAYGROUND_REPORT.md
├─ reference/evidence/
├─ models/<variant>/
└─ evidence/<variant>/
   ├─ generation/results.jsonl
   ├─ generation/summary.json
   ├─ teacher_forcing/summary.json
   ├─ footprint.json
   ├─ output_deviation.json
   └─ OUTPUT_DEVIATION.md
```

## Interpretation boundary

A failed `union_all` candidate would **not** prove that layer 26 or either
original frontier was unreliable. It would show that individually or
path-conditionally removable layers are not necessarily jointly removable.
Likewise, a successful union is still exploratory until it passes independent
validation and human review.

Runtime is recorded but never used to rank or approve a candidate. The current
per-case timing can be noisy; quality, target NLL, case-level deviations, loaded
weight bytes, and repeated timing runs should be reviewed separately.
