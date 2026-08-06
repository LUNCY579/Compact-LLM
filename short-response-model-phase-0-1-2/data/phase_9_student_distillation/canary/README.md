# Phase 9 synthetic canary pack

This pack contains 16 fictional knowledge records and 44 prompts across independent document/topic groups.

It exists to test Phase 9 data splitting, review, training, retrieval, abstention, prompt-injection handling, social behavior, and procedure use with more than the six-row smoke packet. It is not product knowledge and must not be treated as release evidence.

Recommended use:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9.ps1 `
  -Mode Bootstrap `
  -TeacherModel .\artifacts\<accepted-teacher> `
  -Knowledge .\data\phase_9_student_distillation\canary\knowledge.canary.jsonl `
  -Procedures .\data\phase_9_student_distillation\canary\procedures.canary.jsonl `
  -Prompts .\data\phase_9_student_distillation\canary\prompts.canary.jsonl `
  -OutputRoot .\artifacts\phase_9_canary_v2
```

The teacher review packet will place each authored target into the candidate list when it is not duplicated by teacher generation. A human must still approve every accepted row. The canary pack can prove pipeline behavior; it cannot substitute for the 5,000+ domain-specific pilot proposed for a serious student-quality claim.
