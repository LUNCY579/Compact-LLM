# Phase 8 — Sealed evaluation and release packaging

Phase 8 performs work rather than printing instructions: it evaluates the exact selected checkpoint on sealed data, collects A-I evidence, checks the complete gate chain, builds the release audit, creates a package manifest or copied-model package, and writes the final pending human gate.

```powershell
.\scripts\phase_8_release_candidate\run_phase_8.ps1 -PackageMode manifest
```

Use `-PackageMode copy` only when you intentionally want model bytes duplicated into the release package. Missing ablations are reported as missing, never fabricated. Commercial shipment remains blocked until `PHASE_8_STOP_GATE.md` is explicitly approved and rollback is tested. See `FILE_RESPONSIBILITIES.md`.

The sealed run is forced to `one_sentence`. The release audit blocks if that mode is missing or any user-visible reply exceeds the validated first-sentence boundary. The package includes `response_control.json` and `RESPONSE_CONTROL_CONTRACT.md` so the application runtime has an explicit launch contract.

## Quality evaluator v3.0

This phase uses Quality Evaluator `3.0` and the shared relative-opportunity engine. Informal or mildly ungrammatical language is permitted when understandable; references are examples rather than unique answers; and behavioral restrictions are enforced only when explicitly present in the response contract. Promotion never depends on an absolute score, pass-rate, completion, category, NLL, or issue-count floor. The declared proportional resource benefit must be strictly greater than combined proportional quality opportunity loss. See `RELATIVE_OPPORTUNITY_EVALUATION_POLICY.md` and `EVALUATION_V3_MIGRATION.md`.

