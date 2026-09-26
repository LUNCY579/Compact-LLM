# Active goal continuation

The full objective is in the user attachment `C:/Users/clclc/.codex/attachments/b29bca3e-7e82-4c96-b5cd-7972a0e1391b/pasted-text-1.txt`. The goal is ACTIVE and unfinished. Previous goal turn made progress: audit, report corrections, harness repairs, completed corrected smoke, and bounded training dispatch. Latest goal turn added and verified a dependent execution driver. Long-running training is a verified wait, not a blocker.

## Live native controllers — updated after verified interruption

The original unified sessions29535/34791 are gone. Native PID162016/194212 and all phase1 Python workers were also confirmed missing, so this was real process termination, not just an observation timeout. Cause is unknown. Interrupted B logged153 applied updates and validation75/150 but saved no resumable optimizer state or final predictions. Its evidence is preserved under `results/audit_20260905/interrupted_bounded_a11_arm_B_r1`; the controller_interrupt_receipt independently confirms the canonical checkpoint and frozen sources still match. It is not a scientific null.

Do NOT poll the obsolete unified session IDs or launch duplicate jobs. Current jobs were launched via hidden Start-Process with redirected logs and were revalidated alive after the launching tool returned:

- A1.1 recovery controller PID **57340**, UTC start **2026-09-05T23:39:56.5626195Z**. Dedicated `logs/resume_a11_remaining.ps1` retains completed A, restarts B from canonical initialization for the full fixed300-attempt budget, then runs C300 and combines. Status `logs/resume_a11_remaining_status.json`; launch identity/hash `logs/resume_a11_remaining.launch.json`. Model logs `logs/audit_a11_bounded_B_r1.log` and `..._C_r1.log`. Controller stdout/stderr `logs/resume_a11_remaining.stdout.log` / `.stderr.log`.
- A1 continuation controller PID **56296**, UTC start **2026-09-05T23:39:56.9023791Z**. `logs/continue_a1_after_a11.ps1` waits for exact recovery PID/start/handle and validates its launch path/SHA256, then audits A1.1 and executes A1 smoke plus bounded A/B/C, combine and final audits. Status `logs/continuation_a1_driver_status.json`; launch `logs/continuation_a1_driver_r1.launch.json`. Controller stdout/stderr `logs/continuation_a1_driver_r1.stdout.log` / `.stderr.log`.
- Its private audit directory is now `results/audit_20260905/continuation_0e0cfcfdedae4c338bf524a5f649ef34`. Old private output directory and receipts remain historical evidence.
- Verify actual native Get-Process plus expected UTC start ticks and Python process inventory. A status file alone is not liveness. PowerShell ConvertFrom-Json can parse ISO values as DateTime; compare UTC ticks, not strings against parsed objects.
- A1.1 cancellation marker: `logs/resume_a11_remaining.cancel`; A1 continuation cancellation marker: `logs/continuation_a1.cancel`. Markers stop future launches, letting a current child finish. Do not create them unless cancellation is intended.

No Python model/source/configuration changed during recovery. The A1 continuation guard now also accepts the exact dedicated recovery script path and matching launch receipt hash/PID/starttime; all earlier identity safeguards remain. Both PowerShell controllers are now frozen too. Do not edit them while live.

The continuation ends `COMPLETE_DOCS_PENDING`; it never runs Phase2/composition or completes the goal. A1 must NOT be started manually while it is live.

## Frozen execution

Do not edit `src/`, `scripts/*.py`, or `configs/*.json` while arms run: combine compares exact source/config hashes. Logs and audit-result helper files are outside that source hash set. All new model source was frozen before the corrected smoke.

Required interpreter: `../Chat-Bubble/.venv/Scripts/python.exe`, always repository-root cwd. Canonical read-only cores are only under `../Chat-Bubble/Rineya_Published_Cores/`. Every model/verification stage hashes before/after; retain `canonical_checkpoint_unchanged=true`.

Final recipe: layers17/18/19 attention only, protected0/1/2 frozen, rank4 alpha8 dropout0, LR1e-4, seed17, accumulation8, 300 optimizer attempts, mandatory gradient checkpointing. MAX_LENGTH3840 retains all original sampled packs. Full decoder input remains intact; answer-only vocabulary logits implement the same masked-answer objective, tested in15 CPU equivalence cases.47 mechanical tests passed. Attempts/applied/scaler skips/safety skips are separately persisted.

For every bounded arm and upcoming A1 smoke, both allocator environment variables equal:
`PYTORCH_ALLOC_CONF=backend:native,garbage_collection_threshold:0.7,per_process_memory_fraction:0.8`
`PYTORCH_CUDA_ALLOC_CONF=backend:native,garbage_collection_threshold:0.7,per_process_memory_fraction:0.8`
The original controller and continuation driver set these. VRAM preflight requires5500MiB free; driver waits if busy. This is a fixed allocator policy, not a model/KV/hyperparameter change.

## Audit and completed work

The audit was completed FIRST. Same-LCR prior-training concern was REFUTED: actual Stage A cache matches CC0 long-context plus short replay. Exact historical consumed-row trace remains unknown. Phase 0 arithmetic verified from raw predictions, but causal interference/perfect-retrieval/length-only conclusions were withdrawn. Full checklist statuses and evidence are in RESULT.md and `results/audit_20260905/*_notes.md`.

D1–D5, stale-arm refusal, exact rational gates, requested core/dtype, length retention, training safety, and resource accounting repaired. The separate auditor's info/resource binding defect was fixed before the corrected smoke; do not apply its old runtime workaround now.

Corrected A1.1 smoke is COMPLETE and independently validated. B/C each applied all6 updates, no skips. Its saved adapter values also pass (not prediction roundtrip). A1.1 bounded A reproduced170/256 target and74/200 matched preservation, from raw per-example scoring. The original B attempt was interrupted after153 logged updates. Read fresh restarted-B progress for current counts; do not add interrupted updates to the completed experimental budget.

Original stale smoke was moved out of active results paths after automatic recursive-deletion rejection. Preserved evidence: `audit_20260905/stale_phase1_smoke_original` and `prior_phase1_smoke`. First repaired smoke A/B completed but C was deliberately interrupted after excessive reserved-memory pressure; its independent post-interruption checkpoint hash matched. This is under `audit_20260905/pre_answer_logits_smoke`. It is not a scientific null. The full smoke was repeated after the verified allocation fix. No need to retry deletion.

## Remaining required work

1. Monitor/revalidate both live handles. Do not restart on an observation timeout. If a process actually fails, inspect its receipts/logs and preserve completed work before any repair.
2. Let both bounded core comparisons finish (A1 is still required even if A1.1 is negative). Driver will audit and run the remaining sequence.
3. Independently run root `scripts/audit_phase1_results.py --stages phase1_smoke phase1 phase1_smoke_a1 phase1_a1` after completion. Inspect every receipt/pair check and every discrepancy array; status labels alone are insufficient. All performance numbers must be recomputed from raw `per_example.jsonl`.
4. Run `results/audit_20260905/verify_saved_adapter_values.py --require-complete --stages phase1_smoke phase1 phase1_smoke_a1 phase1_a1`. It checks saved tensor names, finite/nonzero values, and norms only, not loaded prediction parity.
5. Apply the preregistered gate: target gain>=2 absolute points OR>=5% relative error reduction, preservation regression<=1point. Null/negative Phase1 stops programme; no compensating sweep. Phase2 only if actually licensed, and only winning method at17/18/19 versus17/18/19/25/27. Composition check NEVER licensed.
6. Finish RESULT.md with both core results, gradients/skips/costs, gate outcome, and which conclusions survived. Test the recorded prior that A1 closely reproduces A1.1; do not presume it. Teacher-forced loss is not a measured time-to-target-quality curve. Report conventional medians explicitly: runner dataset.json uses upper middle for even sample size.
7. Verify final requirements against raw evidence and mark goal complete ONLY when all requested work, including any licensed Phase2 and final report, is actually done. Current report is interim and explicit about pending bounded outcomes.

Reusable agents: audit_numbers, audit_design, audit_mechanics. No commits requested; lab remains untracked. Source/scripts/configs are frozen. Do not mark blocked just because training takes hours.


## Additional final-analysis helpers, prepared during live training

- `results/audit_20260905/analyse_phase1_frames.py` independently rescores raw target predictions and uses the exact Phase0 abstention rule. Run with the required interpreter after completion. Default stages are phase1 and phase1_a1; it writes phase1_frames.json/.md, with answerability, variant, task and split breakdowns. It never applies a gate. This is necessary to avoid calling an abstention-only composite gain an answerable-reading improvement. Current bounded output correctly analyzes A only and is PARTIAL.
- `completion_requirements.json` is a navigation/pending-work checklist, not proof of completion. `completion_snapshot.raw_numbers.json` is a current partial raw audit; final audit must be rerun.
- `external_process_exit_verification.json` confirms the continuation driver's exact fresh Process object / retained Handle / WaitForExit / Refresh / ExitCode pattern on hidden disposable processes exiting0and7. Driver/source/models were untouched. Real completed-smoke validation functions also passed before driver launch.
- Keep all outcome claims scoped to the preregistered LCR-to-demo transfer recipe. Lower teacher-forced validation loss is not automatically better autoregressive target accuracy or proof of faster target-quality attainment.
