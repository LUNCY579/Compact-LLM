# Centralized Titan 0.6B program

Primary: **Rineya-Titan-Experimental-1.0.0-devfreeze**, 596,049,920 parameters.
Weight SHA256: `537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7`.
See [RESEARCH_PROTOCOL](RESEARCH_PROTOCOL.md) and
[PRIMARY_STUDENT_RECEIPT](receipts/PRIMARY_STUDENT_RECEIPT.json).

Current state: exact static Titan identity measured; E0 blocked before model load
by host RAM (6.9113 GiB available versus 7.4 GiB admission). All 80 focused CPU
tests pass; 4608 original open-role bridge parents are independently verified.
No trained candidate or capability improvement is claimed. The original 344M draft is
preserved under evidence/superseded_initial_draft and is out of primary scope.

The source branch was fetched and verified at
`03db218ceebc61a2bc04f2482cf7e094045e01c1`; the execution branch was pushed at that
same commit before implementation. Only this execution branch accumulates new work.

## Entry points

Run from repository root, using the measured CUDA environment
`C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe` for native work.

```text
python -B -m unittest discover -s experiments/post_training_mimo_centralized_v1/tests -v
python -B -m experiments.post_training_mimo_centralized_v1.scripts.bridge_data --registration experiments/post_training_mimo_centralized_v1/manifests/E1_DATA_REQUIREMENTS.json --output <new-directory> --source-revision <committed-code-sha>
python -B -m experiments.post_training_mimo_centralized_v1.scripts.supervise_native --config experiments/post_training_mimo_centralized_v1/configs/E0_TITAN_BF16.json --output artifacts/mimo_central/e0_titan_v1
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
