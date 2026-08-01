# Phase 8 — Sealed evaluation and release packaging

Phase 8 performs work rather than printing instructions: it evaluates the exact selected checkpoint on sealed data, collects A-I evidence, checks the complete gate chain, builds the release audit, creates a package manifest or copied-model package, and writes the final pending human gate.

```powershell
.\scripts\phase_8_release_candidate\run_phase_8.ps1 -PackageMode manifest
```

Use `-PackageMode copy` only when you intentionally want model bytes duplicated into the release package. Missing ablations are reported as missing, never fabricated. Commercial shipment remains blocked until `PHASE_8_STOP_GATE.md` is explicitly approved and rollback is tested. See `FILE_RESPONSIBILITIES.md`.

The sealed run is forced to `one_sentence`. The release audit blocks if that mode is missing or any user-visible reply exceeds the validated first-sentence boundary. The package includes `response_control.json` and `RESPONSE_CONTROL_CONTRACT.md` so the application runtime has an explicit launch contract.

## Quality evaluator v2.0

This phase consumes or produces evaluator schema `2.0`. The main score is not a length/completion average: hard intelligibility collapse receives zero, and phase gates separately enforce tail quality, category behavior, semantic and completion floors, reference alignment, and critical/major issue limits. Evidence created by the legacy evaluator must be regenerated before this phase can automatically pass. See the repository-root `QUALITY_EVALUATION_V2_AUDIT.md` and `QUALITY_EVALUATION_V2_MIGRATION.md`.

