# Centralized Titan 0.6B program

Primary: **Rineya-Titan-Experimental-1.0.0-devfreeze**, 596,049,920 parameters.
Weight SHA256: `537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7`.
See [RESEARCH_PROTOCOL](RESEARCH_PROTOCOL.md) and
[PRIMARY_STUDENT_RECEIPT](receipts/PRIMARY_STUDENT_RECEIPT.json).

Current state: exact static Titan identity measured; E0 blocked before model load
by host RAM (6.9113 GiB available versus 7.4 GiB admission). The prior 172+89 test runs and 43 new language controls plus two fingerprint
controls passed separately; 4608 open-role bridge parents are independently verified.
No trained candidate or capability improvement is claimed. The original 344M draft is
preserved under evidence/superseded_initial_draft and is out of primary scope.

The source branch was fetched and verified at
`03db218ceebc61a2bc04f2482cf7e094045e01c1`; the execution branch was pushed at that
same commit before implementation. Only this execution branch accumulates new work.

## Entry points

Run from repository root, using the measured CUDA environment
`C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe` for native work.

```text
C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe -B -m unittest discover -s experiments/post_training_mimo_centralized_v1/tests -v
python -B -m experiments.post_training_mimo_centralized_v1.scripts.bridge_data --registration experiments/post_training_mimo_centralized_v1/manifests/E1_DATA_REQUIREMENTS.json --output <new-directory> --source-revision <committed-code-sha>
C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe -B -m experiments.post_training_mimo_centralized_v1.scripts.supervise_native --config experiments/post_training_mimo_centralized_v1/configs/E0_TITAN_BF16.json --output artifacts/mimo_central/e0_titan_v1
```

Every native run requires clean source and a new ignored artifact directory.
The supervisor uses the existing Opallo Windows Job Object/process-tree watchdog,
a cooperative GPU lock, RAM/RSS/time limits and a containment GO handshake.
Failures are durable; no automatic retry or tolerance relaxation. Reusable receipt
JSON is copied into this package after the run; model binaries stay local.

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
5120 TRAIN rows,256 steps,16 acquisition+4 replay per step,seed101 only. The
[frozen plan index](manifests/E1_EXECUTION_PLAN.json) binds the actual data and
source/config bytes. It grants no native execution. The enrollment importer
preserves parent groups and every attempt but cannot certify model capture or grading.

Reproduce the plan in a new directory with the pinned Python3.11.15 interpreter:

```text
C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe -B -m experiments.post_training_mimo_centralized_v1.scripts.prepare_e1 --config experiments/post_training_mimo_centralized_v1/configs/E1_PLAN_V1.json --output <new-directory>
```

This command uses the standard library and cannot load or train a model. It requires
the local bound TRAIN files and clean committed planning code/config. Python version
is part of the shuffle contract: the preserved3.13 plan is superseded by the3.11 plan.
Cursor validity covers accounting only; no native optimizer/checkpoint/RNG restoration
is implied. Next independent work is fresh language DEV/CAL candidate authoring and
review packets under a new prospective amendment; qualified grading remains external.

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
native qualification. Next is the concrete DEV input/scorer adapter, including prompt
slicing that prevents target-bearing short-task arrays from leaking reference answers.
