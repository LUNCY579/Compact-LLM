# Phase 3 — Preference Alignment

Phase 3 trains DPO from one explicitly selected Phase 2 profile:

- `full_quality` — original all-layer SFT adapter;
- `balanced` — deepest approved Phase 2B frontier model within the balanced score-loss ceiling;
- `min_memory` — deepest approved Phase 2B frontier model that still passes the progression rule.

The default legacy command still uses the original all-layer SFT adapter. Reduced profiles require a Phase 2B human PASS and explicit listing in `Approved profiles`.

## Commands

Original branch:

```powershell
.\scripts\phase_3_preference_alignment\run_phase_3.ps1
```

Approved profile branch:

```powershell
.\scripts\phase_3_preference_alignment\run_phase_3.ps1 -Phase2Profile balanced
```

Profile branches write separate adapter, evaluation, and stop-gate paths. They do not overwrite the existing accepted Phase 3 result.

## Initialization behavior

- `full_quality` continues training the accepted SFT LoRA adapter.
- A reduced profile is already a materialized recovered SFT checkpoint, so Phase 3 attaches a fresh all-remaining-layer LoRA adapter before DPO.

Phase 4 ingredients remain compatible only when all seeds use the same selected profile, model architecture, adapter rank, target modules, and parameter shapes.

## Gate

DPO must improve concise-correct preference without material factual, context, safety, completion, or category regression. Scripts never record a human PASS automatically.
