# Phase 10 — student architecture reduction and challenger search

Phase 10 converts the human-approved Phase 9 student into a measured architecture frontier. It creates physically smaller dense candidates, recovery-distills a bounded subset under the original teacher, evaluates every candidate under the four Phase 9 RAG conditions, and requires a named human selection.

Main runner:

```powershell
.\scripts\phase_10_student_architecture\run_phase_10.ps1 -Mode Pilot -TeacherModel <accepted-teacher>
```

Use `EXECUTION_MANUAL.md` for the complete sequence and `FILE_RESPONSIBILITIES.md` for ownership boundaries.

## Phase 9 candidate compatibility

Before running the Phase 10 runner, call `finalize_phase_9_selection.py` with the exact candidate key from `reports/PHASE_9_ASSESSMENT.json`. Both legacy single-suite keys and profile-qualified hybrid keys are supported. Narrow random-initialized profiles remain research controls unless explicitly acknowledged after separate readiness review.
