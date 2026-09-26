# Active goal continuation

The full objective is in the user attachment `C:/Users/clclc/.codex/attachments/b29bca3e-7e82-4c96-b5cd-7972a0e1391b/pasted-text-1.txt`. The goal is ACTIVE and unfinished. Previous goal turn made progress: audit, report corrections, harness repairs, completed corrected smoke, and bounded training dispatch. Latest goal turn added and verified a dependent execution driver. Long-running training is a verified wait, not a blocker.

## Live native controllers — synchronized retry r2

DO NOT use old unified sessions29535/34791 or old PIDs162016/194212/57340/56296; all were confirmed gone. Original bounded B logged153 applied attempts before unexplained termination. Its archive is `audit_20260905/interrupted_bounded_a11_arm_B_r1`. Detached retry r1 applied4 attempts, then hit CUDA illegal memory access; both controllers exited1. Windows nvlddmkm events23:42:11–15 explicitly recorded TDR reset/restart. Earlier23:27 also had a driver error. Causation/hardware versus software origin remains uncertain. Canonical hashes stayed unchanged after both failures.

The failed r1 stage, previous completed A, and controller snapshots are now `results/audit_20260905/pre_sync_phase1`. Do not interpret interrupted/failed work as a null or add its updates to the accepted budget. The short diagnostic `cuda_first16_r1/diagnostic.json` completed16 full-input forward/backward microbatches with CUDA_LAUNCH_BLOCKING=1, zero optimizer updates, exact bounded data fingerprint, and unchanged canonical/source hashes. Its first-batch mean loss reproduces the original rounded first-batch loss. It does NOT prove full-run stability.

Current hidden Start-Process controllers were revalidated alive after launch:

- A1.1 r2 PID **121948**, UTC start **2026-09-06T00:04:38.4227648Z**. `logs/resume_a11_remaining.ps1` now regenerates A, then executes B300/C300 and combines, all with CUDA_LAUNCH_BLOCKING=1 and the original allocator policy. Status `logs/resume_a11_remaining_status.json`; current launch identity/hash `logs/resume_a11_remaining.launch.json`. Logs `logs/audit_a11_bounded_A_r2.log`, B_r2.log, C_r2.log. Controller stdout/stderr `logs/resume_a11_remaining_r2.stdout.log` / `.stderr.log`.
- A1 continuation r2 PID **121440**, UTC start **2026-09-06T00:04:38.7510242Z**. `logs/continue_a1_after_a11.ps1` waits for the exact predecessor handle/PID/start and source/launch SHA, audits it, runs A1 smoke then all bounded arms and final audits. It also sets CUDA_LAUNCH_BLOCKING=1 for every child. Status `logs/continuation_a1_driver_status.json`; launch `logs/continuation_a1_driver_r2.launch.json`; stdout/stderr `logs/continuation_a1_driver_r2.stdout.log` / `.stderr.log`.
- Private audit output directory: `results/audit_20260905/continuation_0c52c2e721184e3fac1ac3fd92942770`.
- No new unified session IDs exist for these detached jobs. Verify actual Get-Process and expected UTC ticks, plus Python process command lines. Do not trust status files alone. ConvertFrom-Json can convert ISO strings to DateTime; compare UTC ticks.
- Cancellation markers `logs/resume_a11_remaining.cancel` and `logs/continuation_a1.cancel` stop future launches; a current child finishes normally. Do not create unless cancellation is intended.

A1 must not be launched manually while its continuation is live. Both PowerShell controllers and Python sources are frozen now. The driver ends COMPLETE_DOCS_PENDING and never runs Phase2/composition or completes the goal.

## Frozen execution

Do not edit `src/`, `scripts/*.py`, or `configs/*.json` while arms run: combine compares exact source/config hashes. Logs and audit-result helper files are outside that source hash set. All new model source was frozen before the corrected smoke.

Required interpreter: `../Chat-Bubble/.venv/Scripts/python.exe`, always repository-root cwd. Canonical read-only cores are only under `../Chat-Bubble/Rineya_Published_Cores/`. Every model/verification stage hashes before/after; retain `canonical_checkpoint_unchanged=true`.

Final recipe: layers17/18/19 attention only, protected0/1/2 frozen, rank4 alpha8 dropout0, LR1e-4, seed17, accumulation8, 300 optimizer attempts, mandatory gradient checkpointing. MAX_LENGTH3840 retains all original sampled packs. Full decoder input remains intact; answer-only vocabulary logits implement the same masked-answer objective, tested in15 CPU equivalence cases.47 mechanical tests passed. Attempts/applied/scaler skips/safety skips are separately persisted.

For every bounded arm and upcoming A1 smoke, both allocator environment variables equal:
`PYTORCH_ALLOC_CONF=backend:native,garbage_collection_threshold:0.7,per_process_memory_fraction:0.8`
`PYTORCH_CUDA_ALLOC_CONF=backend:native,garbage_collection_threshold:0.7,per_process_memory_fraction:0.8`
The current r2 controllers set these and also set CUDA_LAUNCH_BLOCKING=1. This synchronized launch setting follows the successful bounded diagnostic; no model/data/optimizer code changed. A is regenerated under it for the final comparison. VRAM preflight requires5500MiB free; driver waits if busy. This is a fixed allocator policy, not a model/KV/hyperparameter change.

## Audit and completed work

The audit was completed FIRST. Same-LCR prior-training concern was REFUTED: actual Stage A cache matches CC0 long-context plus short replay. Exact historical consumed-row trace remains unknown. Phase 0 arithmetic verified from raw predictions, but causal interference/perfect-retrieval/length-only conclusions were withdrawn. Full checklist statuses and evidence are in RESULT.md and `results/audit_20260905/*_notes.md`.

D1–D5, stale-arm refusal, exact rational gates, requested core/dtype, length retention, training safety, and resource accounting repaired. The separate auditor's info/resource binding defect was fixed before the corrected smoke; do not apply its old runtime workaround now.

Corrected A1.1 smoke is COMPLETE and independently validated. B/C each applied all6 updates, no skips. Its saved adapter values also pass (not prediction roundtrip). The historical pre-sync A reproduced170/256 target and74/200 matched preservation. It is archived; use the regenerated r2 A for final comparison. The original B attempt was interrupted after153 logged updates. Read fresh restarted-B progress for current counts; do not add interrupted updates to the completed experimental budget.

Original stale smoke was moved out of active results paths after automatic recursive-deletion rejection. Preserved evidence: `audit_20260905/stale_phase1_smoke_original` and `prior_phase1_smoke`. First repaired smoke A/B completed but C was deliberately interrupted after excessive reserved-memory pressure; its independent post-interruption checkpoint hash matched. This is under `audit_20260905/pre_answer_logits_smoke`. It is not a scientific null. The full smoke was repeated after the verified allocation fix. No need to retry deletion.

## Remaining required work

1. Monitor/revalidate both live handles. Do not restart on an observation timeout. If a process actually fails, inspect its receipts/logs and preserve completed work before any repair.
2. Let both bounded core comparisons finish (A1 is still required even if A1.1 is negative). Driver will audit and run the remaining sequence.
3. Independently run root `scripts/audit_phase1_results.py --stages phase1_smoke phase1 phase1_smoke_a1 phase1_a1` after completion. Inspect every receipt/pair check and every discrepancy array; status labels alone are insufficient. All performance numbers must be recomputed from raw `per_example.jsonl`.
4. Run `results/audit_20260905/verify_saved_adapter_values.py --require-complete --stages phase1_smoke phase1 phase1_smoke_a1 phase1_a1`. It checks saved tensor names, finite/nonzero values, and norms only, not loaded prediction parity.
5. Apply the preregistered gate: target gain>=2 absolute points OR>=5% relative error reduction, preservation regression<=1point. Null/negative Phase1 stops programme; no compensating sweep. Phase2 only if actually licensed, and only winning method at17/18/19 versus17/18/19/25/27. Composition check NEVER licensed.
6. Finish RESULT.md with both core results, gradients/skips/costs, gate outcome, and which conclusions survived. Test the recorded prior that A1 closely reproduces A1.1; do not presume it. Teacher-forced loss is not a measured time-to-target-quality curve. Report conventional medians explicitly: runner dataset.json uses upper middle for even sample size.
7. Verify final requirements against raw evidence and mark goal complete ONLY when all requested work, including any licensed Phase2 and final report, is actually done. Current report is interim and explicit about pending bounded outcomes.

Earlier agent handles disappeared with the context reset. Use collaboration.list_agents before reusing an agent; recovery_driver handled the outer-driver guard update. No commits requested; lab remains untracked. Source/scripts/configs are frozen. Do not mark blocked just because training takes hours.


## Additional final-analysis helpers, prepared during live training

- `results/audit_20260905/analyse_phase1_frames.py` independently rescores raw target predictions and uses the exact Phase0 abstention rule. Run with the required interpreter after completion. Default stages are phase1 and phase1_a1; it writes phase1_frames.json/.md, with answerability, variant, task and split breakdowns. It never applies a gate. This is necessary to avoid calling an abstention-only composite gain an answerable-reading improvement. Current bounded output correctly analyzes A only and is PARTIAL.
- `completion_requirements.json` is a navigation/pending-work checklist, not proof of completion. `completion_snapshot.raw_numbers.json` is a current partial raw audit; final audit must be rerun.
- `external_process_exit_verification.json` confirms the continuation driver's exact fresh Process object / retained Handle / WaitForExit / Refresh / ExitCode pattern on hidden disposable processes exiting0and7. Driver/source/models were untouched. Real completed-smoke validation functions also passed before driver launch.
- Keep all outcome claims scoped to the preregistered LCR-to-demo transfer recipe. Lower teacher-forced validation loss is not automatically better autoregressive target accuracy or proof of faster target-quality attainment.
