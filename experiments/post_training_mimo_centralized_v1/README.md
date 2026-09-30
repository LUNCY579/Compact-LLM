# Centralized Titan 0.6B program

Primary: **Rineya-Titan-Experimental-1.0.0-devfreeze**, 596,049,920 parameters.
Weight SHA256: `537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7`.
See [RESEARCH_PROTOCOL](RESEARCH_PROTOCOL.md) and
[PRIMARY_STUDENT_RECEIPT](receipts/PRIMARY_STUDENT_RECEIPT.json).

Current state (30 September 2026): the streamed CE512 native mechanics trial
passed all eight checks, four disposable updates, bitwise save/resume and unchanged
base. Observed process-tree RAM peaked at 1.748646 GiB, with at least 2.498425 GiB
host headroom and 3.326172 GiB free VRAM in the recorded samples. The observed
RSS margin is only 1.387 MiB; sustained E1/data/evaluation overhead is unmeasured.
The old full E0 padding and 1024-token VRAM failures remain preserved.

The user-authorized laptop route uses GPU-resident BF16 weights and FP32 adapters,
a fixed 2 MiB read buffer, a 1.75 GiB worker RSS cap and a 1.25 GiB system reserve.
The [standalone trial](scripts/run_laptop_adapter_trial.cmd) waits for 3.0625 GiB
available and defaults to reproducing the bounded CE512 qualification. See the
[laptop runbook](LAPTOP_RUNBOOK.md) and [current plan](manifests/LAPTOP_EXECUTION_PLAN_V2.json).
No unchanged E0 repeat is needed after the complete scoped pass.

Native campaign totals are four exact loads and six disposable optimizer calls.
Generation/teacher calls and acquisition/retention measurements remain absent.
Full E0, capture, preservation, long context and production E1 remain unqualified.
Historical 7.4/6 GiB guards and E1/scheduler plans retain their original evidence.
CHECK/FINAL stay closed; current best remains unchanged Titan.

The next independently necessary work is the [source-joined TRAIN contract](TRAIN_SCOPE_CONTRACT.md).
It binds actual semantic TRAIN source records to all 5120 frozen arrays and the
CE512 consumer interface, preserving the entire data dose and final-assistant
supervision. Existing array validation alone only syntax-checks source-row hashes.
The committed validator passed all 5120 actual source/array joins in 3.329 seconds
with a 39.004 MiB reader peak. Eighteen new CPU controls passed. It never grants
native training; current capture and preservation/E1 prerequisites remain open. No
model or controller is currently running. Scoped native/evidence receipts are in
receipts/LAPTOP_NATIVE_STAGE_005_RESULTS.json and receipts/ce512_native_evidence_v2/.

The source branch was fetched and verified at
`03db218ceebc61a2bc04f2482cf7e094045e01c1`; the execution branch was pushed at that
same commit before implementation. Only this execution branch accumulates new work.

## Entry points

Run from repository root, using the measured CUDA environment
`C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe` for native work.

```text
C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe -B -m unittest discover -s experiments/post_training_mimo_centralized_v1/tests -v
python -B -m experiments.post_training_mimo_centralized_v1.scripts.bridge_data --registration experiments/post_training_mimo_centralized_v1/manifests/E1_DATA_REQUIREMENTS.json --output <new-directory> --source-revision <committed-code-sha>
C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe -B -m experiments.post_training_mimo_centralized_v1.scripts.supervise_native --config experiments/post_training_mimo_centralized_v1/configs/E0_TITAN_BF16.json --output artifacts/mimo_central/<new-e0-run-directory>
```

Every native run requires clean source and a new ignored artifact directory.
The supervisor uses the existing Opallo Windows Job Object/process-tree watchdog,
a cooperative GPU lock, RAM/RSS/time limits and a containment GO handshake.
Failures are durable; no retry until a meaningful resource-state change and no
tolerance relaxation. The accounting-v2 qualifier is itself natively UNQUALIFIED.
Reusable receipt JSON is copied into this package after the run; model binaries stay local.

Data preparation exports only TRAIN, TRAIN_SELECTION and DEV prompt/target rows;
CHECK exports identities only; FINAL remains unopened. Preparation alone does not
qualify replay, historical exclusion, tokenizer fit, or training admission.

The [method inventory](METHOD_INVENTORY.csv) has source/evidence boundaries for 68
components. Historical native measurements are not new Titan qualification. The
report's 344M resource estimates are not reused. BF16 stock Qwen3 does not inherit
historical DenseReference FP32 scores.

Cycle results: [report](CENTRALIZED_EXPERIMENT_REPORT.md), [PDF](CENTRALIZED_EXPERIMENT_REPORT.pdf),
[results matrix](RESULTS_MATRIX.csv), [resource ledger](RESOURCE_LEDGER.csv),
[next experiment](NEXT_EXPERIMENT.md). The research goal remains ongoing.

Continuation 002: 768 deterministic replay rows and 256 curated language candidates
are prepared and tokenized. Candidate dose: 5120 sequences / 680532 nonpadding /
63546 supervised tokens. Validation: 129 aggregate +5 focused CPU tests. See
[AMENDMENT_002](AMENDMENT_002.md), [AMENDMENT_003](AMENDMENT_003.md) and
[E1 candidate dose](manifests/E1_CANDIDATE_DOSE.json). No native E0/E1 grant follows.

Continuation 003: exact long-context DEV data are built (32 worlds, 256 rows;
2024/4072 full-template tokens). Independent artifact audit found no discrepancies;
172 CPU tests pass. Native resource fit and preservation remain unqualified.
Continuation004 completed the E1 input/schedule plan and diagnostic preflight:
5120 TRAIN rows,256 steps,16 acquisition+4 replay per step,seed 101 only. The
[frozen plan index](manifests/E1_EXECUTION_PLAN.json) binds the actual data and
source/config bytes. It grants no native execution. The enrollment importer
preserves parent groups and every attempt but cannot certify model capture or grading.

Reproduce the plan in a new directory with the pinned Python 3.11.15 interpreter:

```text
C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe -B -m experiments.post_training_mimo_centralized_v1.scripts.prepare_e1 --config experiments/post_training_mimo_centralized_v1/configs/E1_PLAN_V1.json --output <new-directory>
```

This command uses the standard library and cannot load or train a model. It requires
the local bound TRAIN files and clean committed planning code/config. Python version
is part of the shuffle contract: the preserved3.13 plan is superseded by the3.11 plan.
Cursor validity covers accounting only; no native optimizer/checkpoint/RNG restoration
is implied. Language DEV/CAL authoring and review-packet preparation were subsequently
completed; qualified grading remains external. Existing E1 plan receipts retain
their historical source bindings. AMENDMENT_008 changed the gate-checker source, so
rebind a new plan to qualified current source/runtime before native execution; preserve
the existing plans and receipts unchanged.

Continuation005 completed the [language candidate bank](manifests/LANGUAGE_DEV_CAL_CANDIDATE_INDEX.json):
64 DEV worlds,16 CAL worlds,96 candidate controls and two unassigned JSON review
packets. Specs were frozen before rendering; assistant cross-reviews and all repairs
are retained. Prompt max166 and reference/control max63 tokens including EOS fit
without truncation. No qualified reference labels, grader, native responses or source
membership exist. Script cost excludes unmetered assistant authorship/review.

Reproduce the candidate export after confirming all source/review files are committed:

```text
C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe -B -m experiments.post_training_mimo_centralized_v1.scripts.prepare_language_eval --output <new-directory>
```

Keep coordinator mappings separate from rating views. `validate_ratings` requires
fresh validated source rows and remains diagnostic; no packet can grant grading or
native qualification. The concrete DEV input/scorer adapter described below also
includes prompt slicing that prevents short-task reference-answer leakage.

Continuations 006-007 completed the [DEV prompt/scorer binding](manifests/DEV_INPUT_SCORER_BINDING.json):
832 rows/544 parent groups, 890874 prompt tokens, with evaluator targets in a separate
sidecar. Short-task prompt slices exclude 6883 reference-token positions. Scorers pin
all three verifier dependencies and keep language grading blocked. Production source
was committed before the artifact; 40 tests and the separate artifact audit passed.

The seven-turn [blocked audit](receipts/CONTINUATION_007_BLOCKED_AUDIT.json) is historical;
the fresh resumed audit now has three blocked turns; only the first launched v3/v4. Native collection, language
calibration, the 2048/4096 extension and native E1 execution remain incomplete.
The full goal is unachieved. No further heavy training is justified yet.

[AMENDMENT_008](AMENDMENT_008.md) adds durable attempted/returned/failed/pending E0
operation accounting and retains completed profiles after later failures. Original
math, inputs, numerical checks and resource floors are unchanged. Source review and
52 stdlib tests passed in 0.681 s (process 1.1190241 s); 12 existing tiny CPU tests
passed separately in 1.459 s (process 6.6929375 s). See
[accounting validation](receipts/E0_ACCOUNTING_V2_CPU_VALIDATION.json).
These are CPU/source checks, not native qualification or evidence of efficacy.
