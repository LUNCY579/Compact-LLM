# V2 terminal read-closure successor: frozen CPU candidate

The original frozen candidate passed25 CPU controls but failed independent immutable-read mutation controls. Its exact validator `1dcea32718c1599acaf0b6265a4d6e89285e55a49452be946cf35d91b4a0b2e7`, driver `a8448f856aa07e8263d314190e28e5c9c397557d767637b801224765999d3c8f`, test, HANDOFF, both passing receipts and independent failure controls are preserved byte-identically in `before_read_closure_repair_001/`. Preservation body is `a8e4683d19f18d43acf125cdbbd118f57596f3bfbe6904a4acb556d17195a886`. Existing earlier histories remain intact.

## Corrected frozen bytes

- Validator: `afca22e2a6f99dd5cb4485375392d8d3181951921c278ce7c5053866a718566e`.
- Terminal driver: `164ea401617ef459da9d5b346403b9ed09d1c2c378d3ba503a3eb9fff0d0f1db`.
- CPU test: `a0bd5b9d7f56ed635bce78270d35e77a2c8b0625bbaf16fa4158787309db2062`.
- Unchanged declaration fixture: `f2a768b4fc062e8b86d5126d179e5a6a250008189b3fba64f22f877bd6f43fa4`.

`READ_CLOSURE_CPU_VALIDATION_001.json` passes30 controls with zero failures/errors/skips in102.011 test seconds (102.469 command seconds). All four hashes match before/after. The fresh isolated test process loaded no torch/transformers/tensorflow/jax/flax/tokenizers/jinja2/markupsafe module. Receipt body is `1a95c02c6ae82086867303361237174496cf3df1f0ace9dac54d5df1ae305502`. The write-once capture script is `run_read_closure_cpu_validation.py`; this candidate still needs independent root/fresh-process review before integration.

## Independent failure controls and repair

The original source-snapshot and lease-terminal changes made from the injected asset callback still produced CPU reports. A registration change from the first injected commit callback occurred after semantic reconstruction but before the old fixed-file baseline, and was likewise accepted. An ignored `1e999` field in lease resume JSON was accepted. The four failures are retained in `FROZEN_CPU_AND_MUTATION_REVIEW_001.json` under the preservation directory, body `ab465cf0910f63864b88f681c564acc5693c1618d4198503cd6bf2a41d037a68`.

The independent complete trace2 fixture additionally reproduced acceptance of post-semantic changes to all four lease files and to each selected prior result/supervisor/audit file. Its receipt `LEASE_PRIOR_CLOSURE_MUTATION_REVIEW_001.json`, body `b98aea4a0b707102f23299f12856fc9e7258e1e6f550d0683a5f33a16ed04818`, explicitly identifies synthetic prior-audit metadata as CPU fixture material. Neither fixture path issued native audit authority.

The corrected driver uses a per-audit explicit `ReadSet`. It records each selected file descriptor before JSON semantics, refuses to rebase an existing descriptor, and rechecks every enrolled file and owned directory roster before emitting a report. Coverage includes current config/inventory/preservation/registration and its four evidence anchors, source and source snapshots, original/preserved raw/publication bytes, prior role files, all durable lease files, and exact inherited short-inference prerequisite evidence/helper bytes read by reconstruction. Native asset hashing also uses the explicit reader after the terminal/source gates; no native asset path was exercised here. Validator resource/journal/publication readers receive that same object without global monkeypatches.

Strict finite float parsing now covers every validator JSON read, including ignored lease/prior fields, and JSONL events/heartbeat. Duplicate/constant/bounded parsing remains closed. The original resource, numerical, dose, ownership, state, admission and cost predicates remain fixed.

Five added test methods cover snapshot/all-lease/extra-directory mutation denial, first-registration-read rebasing denial, ignored nonfinite lease JSON denial, a complete positive trace2 path plus all three prior-file mutation denials, and first/final selected-evidence byte anchoring. The original25 controls remain in the30-test run. Both complete trace1/trace2 positive reports retain the CPU kind, closed authority fields and `first_read_anchors_rechecked=true`.

A copied trace2 helper initially omitted its local `sha` function, failing before any audit. The exact rejected source variant and diagnostic are preserved under `test_trace2_fixture_helper_failure_001/`, failure body `9fa8fee79d66a47652e14f7c78dcb933e6247fc9a1ff0e605f0245c1e8c93256`; adding that test helper changed no production predicate.

## Root integration and unchanged gates

The root-compatible bundle needs both `_v2.py` files, this test under `tests/`, and the unchanged declaration fixture under `evidence/long_v2_audit_adaptation_reviewed_cpu_20261005/fixtures/`. The externally selected inventory must contain both exact corrected validator and terminal-driver hashes. Native dispatch at the staged path rejects before IO/probes even when its module flag is patched. The standalone validator audit entry remains disabled. CPU APIs always issue the CPU kind.

All edits are confined to this staging artifact directory and the reviewer's separate artifact directory. No root production source, final registration/config, actual PID/OS/resource probe, runtime binary/model asset, encoder/framework/GPU/model execution, model training/scoring, protected bank, merge or upload was performed. These tests qualify CPU orchestration only. Independent integration review, final external anchors/config/registration, current clean committed source and actual host/model qualification remain required. Complete M0, language authority, M1/M2 training and the real three-arm acquisition/transfer/retention comparison are still unfinished.
