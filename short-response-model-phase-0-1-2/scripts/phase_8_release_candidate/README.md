# Phase 8 — Sealed evaluation and release packaging

Phase 8 performs work rather than printing instructions: it evaluates the exact selected checkpoint on sealed data, collects A-I evidence, checks the complete gate chain, builds the release audit, creates a package manifest or copied-model package, and writes the final pending human gate.

```powershell
.\scripts\phase_8_release_candidate\run_phase_8.ps1 -PackageMode manifest
```

Use `-PackageMode copy` only when you intentionally want model bytes duplicated into the release package. Missing ablations are reported as missing, never fabricated. Commercial shipment remains blocked until `PHASE_8_STOP_GATE.md` is explicitly approved and rollback is tested. See `FILE_RESPONSIBILITIES.md`.
