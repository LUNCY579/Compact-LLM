# Verified execution manual — Phase 6

“Verified” here means the procedure contains explicit preflight, evidence, and review steps. It does **not** claim that your hardware run has already passed.

## Preflight

1. Activate the same virtual environment used by earlier phases.
2. Confirm the preceding human gate.
3. Copy the current artifact tree or record its hashes before changing it.
4. Confirm adequate free disk space and that no second GPU workload is running during latency/VRAM measurement.
5. Open this phase README and replace placeholder paths with the accepted prior-phase artifacts.

## Execution discipline

Run one command at a time. Keep stdout/stderr. Do not overwrite a successful experiment without archiving its manifest. Use fixed seeds for comparisons. When rerunning only evaluation, preserve training adapters and record the evaluator revision.

## Review discipline

Inspect lowest-scoring and safety-sensitive cases, not only averages. Compare against the correct control. Record failures and negative findings. A human gate is mandatory; scripts never self-approve.

## Recovery

Completed model/adapters are reusable if training code, data, seed, and configuration match. Evaluation can be rerun without retraining. Never mix evaluator versions in one ranking; regenerate all comparable summaries when evaluator logic changes.
