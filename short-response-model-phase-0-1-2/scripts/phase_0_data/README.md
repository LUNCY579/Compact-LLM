# Phase 0 — Data Readiness

- `validate_dataset.py` checks schema, split metadata, IDs, and message boundaries.
- `check_leakage.py` checks cross-split template and near-prompt overlap.
- `build_human_audit_packet.py` creates a reproducible manual-review packet.
- `run_phase_0.ps1` runs all three checks.

Phase 0 is complete only after the generated audit packet has been reviewed and
weak records have been repaired or rejected.
