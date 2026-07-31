# Phase 3 — Preference Alignment

Phase 3 trains DPO from exactly one human-approved upstream branch.

## Phase 2B-only branches

- `full_quality` — original all-layer SFT adapter;
- `balanced` — approved Phase 2B reduced checkpoint;
- `min_memory` — deepest approved Phase 2B checkpoint.

## Phase 2C branches

- `maximum_quality` — best measured Phase 2C recovery candidate;
- `balanced_training` — near-maximum quality with the smallest qualifying trainable state;
- `minimum_training_memory` — lowest-trainable-parameter candidate that passes every hard floor.

`-Phase2Profile` and `-Phase2CProfile` are mutually exclusive. Phase 2C profiles preserve the reduced checkpoint and continue the selected Phase 2C SFT adapter during DPO.

## Commands

Original branch:

```powershell
.\scripts\phase_3_preference_alignment\run_phase_3.ps1
```

Approved Phase 2B branch:

```powershell
.\scripts\phase_3_preference_alignment\run_phase_3.ps1 -Phase2Profile balanced
```

Approved Phase 2C branch:

```powershell
.\scripts\phase_3_preference_alignment\run_phase_3.ps1 -Phase2CProfile balanced_training
```

Profile branches write separate adapter, evaluation, and stop-gate paths. They do not overwrite the existing accepted Phase 3 result.

## Initialization behavior

- Legacy full quality continues the accepted Phase 1 SFT adapter.
- A Phase 2B reduced profile attaches a fresh all-remaining-layer DPO LoRA.
- A Phase 2C profile continues its selected targeted SFT adapter, preserving the selected layer/module structure.

Phase 4 ingredients remain compatible only when every seed uses the same source branch, architecture, adapter placement, rank, target modules, and parameter shapes.

## Gate

DPO must improve concise-correct preference without material factual, context, safety, completion, or category regression. Scripts never record a human PASS automatically.
