# Phase 0 — Data Readiness

- `validate_dataset.py` checks schema, split metadata, IDs, and message boundaries.
- `check_leakage.py` checks cross-split template and near-prompt overlap.
- `build_human_audit_packet.py` creates a reproducible manual-review packet.
- `run_phase_0.ps1` runs all three checks.

Phase 0 is complete only after the generated audit packet has been reviewed and
weak records have been repaired or rejected.


## Evaluation v3 policy

This phase uses Quality Evaluator `3.0` and the shared relative-opportunity engine. Informal or mildly ungrammatical language is permitted when understandable; references are examples rather than unique answers; and behavioral restrictions are enforced only when explicitly present in the response contract. Promotion never depends on an absolute score, pass-rate, completion, category, NLL, or issue-count floor. The declared proportional resource benefit must be strictly greater than combined proportional quality opportunity loss. See `RELATIVE_OPPORTUNITY_EVALUATION_POLICY.md` and `EVALUATION_V3_MIGRATION.md`.
