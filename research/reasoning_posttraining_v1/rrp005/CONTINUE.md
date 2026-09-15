# Active goal continuation after RRP-005 preparation

**Update: use `A2_RUNBOOK.md` and `PLAN_A2.json` / `GRANT_A2.json`.** A real native run
completed qualification and started 548 inputs, then stopped at the 1.5 GiB runtime RAM
floor. Its predictions were not incrementally saved and are lost. The original lane is
retired in code. A2 adds durable per-case records, same-binding continuation and an explicit
budget amendment; old receipts and prior status below remain historical.

The last goal turn made substantive progress: it found a broken retention instrument,
blocked its use, built an independent visible-input procedural solver, quarantined unsafe
gold targets, preserved multi-turn histories, and froze a corrected source measurement.
The overall model-enhancement goal is **not complete**. No new pretrained-model calls or
optimizer steps have run in this task. Two consecutive goal turns have observed insufficient
host RAM, but both made meaningful CPU/code/data progress; do not mark blocked merely from
the counter when useful work is still available.

Current branch: `codex/dataset-retention-audit-v1`, descended from the fully incorporated
remote RRP head `d417e10`. The beginning-of-turn fetch found no new upstream RRP commits.
Implementation commit for this correction: `2cd46c1`.

Read `PLAN.md`, `CORRECTION.json`, `REPLAY_ADMISSION_AUDIT.json`, `PLAN.json`, `GRANT.json`
and `RUN_STATUS.json` here. RRP-004's continuation/report now point here explicitly.

## What changed the next action

- Every one of the 192 old procedural evaluation prompts omitted the rule and evidence.
  Its aggregate baseline is unsuitable as a retention gate. The controller rejects the old
  bank's hash before any model worker is reserved. Do not override this refusal.
- The procedural discovery pool contains 4,189 solver-verified rows, 129 same-result conflict
  targets and two unresolved ties. The 131 failures are listed individually and unmodified.
- The corrected main bank has 757 rows (11 selected ambiguous/inconsistent golds excluded),
  including 31 restored multi-turn conversations. Nonprocedural target correctness is still
  source-reported, not independently proved by a gold-self-scoring "positive control".
- 96 answerable procedural worlds have matched evidence-removal diagnostics. All 192 views
  are solver-verified. Use exactly 853 native inputs: 757 main plus 96 negatives, with positive
  aliases proven byte-identical. Never pool the main/probe metrics or count aliases twice.
- All rows fit the actual pinned tokenizer at 2048. Native execution is still unqualified.
- Final tests: **379 passed, 1 skipped**, `tests-final.xml`. `tests.xml` preserves an earlier
  failure caused by misplaced lines in a new test; it was fixed and rerun, not a model failure.

## Next native step

The source-only measurement is fully prepared, bounded and already authorized under the
user's explicit CPU/GPU instruction. The grant has zero training tokens and zero adaptive
model-selection rounds. Do not reuse historical grants or ask for the same permission again.

```powershell
& ./research/reasoning_posttraining_v1/rrp005/run_baseline.ps1 -RunId baseline-001
```

Prepared data: `C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/a1-prepared-003`.
Corrected banks: `C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/preservation-correction-002`.
Baseline journal: `C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/preservation-baseline-journal`.
Source: this checkout's complete `Supa/` directory. Runtime: sibling `.venv-mq001`.

One command was attempted at head `2cd46c1`; **the parent refused before reservation**, with
3.58 GiB free RAM versus the unchanged 4.6 GiB inference preload minimum. All journal counters
are verified zero; `baseline-001` remains unused; no model worker is live. A worker will also
check after native imports. Do not repeat the command until a fresh resource snapshot shows a
meaningful change. Never lower guards, kill foreign processes or delete old checkpoints.

The supervised worker qualifies inference at 2048 with six charged forward calls, no backward
pass or optimizer, then measures the new bank. Keep the checkout/code fixed while it runs.
There is no training permission hidden in the legacy `training_permission: true` engineering
field: this grant explicitly excludes training and gives it zero tokens.

After completion, score partitions with `preserve.summarize`, independent procedural
`grounded_replay.solve_prompt` checks and paired evidence sensitivity. Honor the exact aliases
in `CORRECTION.json`. Keep termination/caps and literal vs tolerant correctness distinct.
This result will be a corrected **source baseline**, not a learned model improvement.

## Further useful work

Finish the TRAIN-only replay sampler/admission pipeline using verified discovery rows,
complete parent-topic groups, actual token quotas, and explicit held-out/quarantine checks.
The 40/20/40 preservation/Forge-v1/Forge-v2 proposal remains a hypothesis, not a trained recipe.
Do not consume protected sealed material or broadly admit v3 prose merely because its own
validator says PASS: its semantic audit often tests nonempty targets rather than recomputing
their meaning.

Also investigate host-memory waste in the exact native loader, controller's duplicated
prepared payloads, and optimizer state. Any more memory-efficient loader/PEFT experiment
needs a separate prospective contract and measured native parity/resource qualification;
it must not relabel a lower guard as a pass for the old dense-tail recipe. Original 6 GiB
training minima remain intact. This is a potential route toward feasible local training,
not an established solution or a reason to modify a live experiment.
