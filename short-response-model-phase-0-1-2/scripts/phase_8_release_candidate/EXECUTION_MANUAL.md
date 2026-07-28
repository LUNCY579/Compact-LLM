# Phase 8 execution manual

1. Approve Phase 7 with an explicit deployment mode.
2. Confirm the sealed split has not been used by Phases 0–7.
3. Run `.\scripts\phase_8_release_candidate\run_phase_8.ps1 -PackageMode manifest`.
4. Preserve the sealed summary and raw generations; do not tune the model after opening these results.
5. Review the A-I evidence matrix. Missing controls remain blockers or documented limitations.
6. Inspect `RELEASE_CANDIDATE_AUDIT.md`, package manifest, hashes, dependency/runtime configuration, and rollback path.
7. Run application-level smoke/performance/safety tests outside this model repository.
8. Approve `PHASE_8_STOP_GATE.md` only when all release blockers are resolved.
