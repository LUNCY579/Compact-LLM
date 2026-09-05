# Original native GPU v2.2 architecture failure audit

The original architecture matrix is incomplete because AM failed a registered generation guard. A0, A1, A3 and A16 each have complete 128-case artifact coverage. This note reports execution integrity and coverage only; it contains no architecture accuracy, effect estimate or candidate selection.

Audit snapshot: 2026-09-05T22:29:13.055853+00:00. No model or GPU work, checkpoint loading, source/config edits, Git commands, sealed-data access or confirmatory-bank inspection occurred. Parsed case objects were compared/hash-bound without extracting or summarizing score/correct fields. The only new files are in `00_audit/native_architecture_failure_20260905`.

## Valid completed evidence

| Arm | Complete populated shards | Cases in valid complete shards | Expected cases |
|---|---:|---:|---:|
| A0 | 30 | 128 | 128 |
| A1 | 30 | 128 | 128 |
| A3 | 30 | 128 | 128 |
| A16 | 30 | 128 | 128 |
| AM | 19 | 84 | 128 |

The deterministic 32-shard allocation leaves indices 15 and 23 empty, so 30 complete shards establish full coverage for an arm. There are 139 valid completed shards and 596 cases across them. Each complete run's result-file hash, exact ordered case IDs, per-case equality to the hash-bound results, expected logit inventory and every logit byte hash pass. All baseline references match the exact A0 receipt hash and matching shard/config/data/source identity. Complete receipts share the same canonical model identity and original runtime qualification. The 19 completed AM shards retain the qualified frozen control hash, zero trainable/control gradients and the registered parameter/FLOP inventory.

The audit inventoried 602 case/logit pairs in total: 596 from complete shards, five from the invalid A16 interruption, and one from the invalid AM failure. `receipt.json` lists exact case/logit paths and SHA-256 values, all 144 original terminal manifest hashes, and every completed run directory. Its 3,514 audit invariants have zero failures. The separate failure-integrity check adds seven passing checks.

## AM failure and partial count

Exact terminal receipt: `experiments/supa_adaptive_compute_v2/native_armr/evidence/01_architecture_dev/20260905T220307_858493Z_7cc95fc0/run_manifest.json`.

SHA-256: `38a7430a11950715e8a901539ec0db7f98442ccb03435420eefca66df7569fa9`.

This is AM shard 20, status FAILED, with **one saved completed case out of five scheduled cases**. Together with the 84 cases in successful AM shards, 85 AM case outputs were saved; only the 84 from COMPLETE shards are admissible completed evidence. No `results.json` or successful final manifest exists for shard 20. Its case/logit pair is preserved and freshly hashed in this audit; the failed terminal receipt does not contain a completed results/logit hash inventory, so that partial pair is an audit snapshot rather than a newly certified valid shard.

The exact error is `RuntimeError: malformed generation: repetition/script/Unicode guard`. The saved traceback points to `shared/generation.py:66`. The source combines repetition-loop, script-corruption, control-character and replacement-character checks into this one message. The receipt does **not** distinguish which subtype fired, and the rejected generation is not persisted as a completed case. This review makes no subtype attribution and does not reconstruct or tune against the rejected output.

The failure-time backbone integrity receipt records unchanged pre/post canonical state and checkpoint hashes, with zero backbone trainable parameters/gradients. These fields pass the additional check in `failure_integrity_checks.json`. The extra-control post-run state/freeze fields are absent because execution failed before that successful-completion section. This absence neither proves control mutation nor licenses a complete shard. AM still lacks 44 cases in valid completed shards, across 11 populated shard slots (including failed slot 20). The full five-arm architecture prerequisite remains unmet.

## A16 interruption and successful rerun

Interrupted attempt: `experiments/supa_adaptive_compute_v2/native_armr/evidence/01_architecture_dev/20260905T214247_466463Z_ced2c95f`.

Interruption receipt: `experiments/supa_adaptive_compute_v2/native_armr/evidence/01_architecture_dev/20260905T214247_466463Z_ced2c95f/interruption.json`; SHA-256 `e7ca1ba91aa2f1923e0e8d5cd4976c910f3f3c81d843ac5db3b3d845df4a2108`.

Successful new attempt: `experiments/supa_adaptive_compute_v2/native_armr/evidence/01_architecture_dev/20260905T214956_757135Z_03dc0a7a`.

The interrupted shard 16 retained all five scheduled case/logit pairs, but its process exited without a successful final integrity receipt. Its recorded cause is unknown, post-run state hash is UNAVAILABLE, and accepted_as_complete is false. All ten partial file hashes match the interruption receipt. Independently comparing the five retained cases with the successful rerun finds identical token-ID lists, exact stored gold-NLL equality and identical logit file bytes for all five cases. No scores were compared. This supports output reproducibility for the retained artifacts; the interrupted receipt remains invalid and only the successful new attempt contributes to A16 coverage.

Three earlier A0 attempts also remain FAILED: shards 2, 3 and 4 each failed RAM admission with zero saved cases (0.13, 0.41 and 0.08 GiB available versus the registered 0.50 GiB floor). Their fresh successful attempts provide coverage; the failed receipts remain separate.

## Registration, source and confirmation boundary

Original registration: `experiments/supa_adaptive_compute_v2/native_armr/evidence/00_preregistration/7c99dac8b3e72ad44ec2c1eeaaa67ac897fabe6d3209d362c1f78e01c3a7cca5/registration.json`.

Registration ID: `7c99dac8b3e72ad44ec2c1eeaaa67ac897fabe6d3209d362c1f78e01c3a7cca5`. Registration file SHA-256: `17f907463f5dd495c3f0ad889d2ef0f0a734f9edd99f68f5542c0b36f3b20e62`.

Original config: `experiments/supa_adaptive_compute_v2/native_armr/configs/protocol_gpu_v2_2.json`; file SHA-256 `4dbdb7c8b80c14cdedb1c9eb0d69a3c5ca6091f397a9e62ab018b08bab84d81d`. Its canonical config hash is `cd01ccbf01fe285d46f74891559589ee6bce62d945f5fc5fa351b48f5194f70f`. The registration contract digest and current original config bytes match.

Frozen architecture data: `experiments/supa_adaptive_compute_v2/data/generated/v2/native_armr/architecture_dev.jsonl`; SHA-256 `b8fe38a44d8d945405b4928f3ab61928bbfdc3d8642b723336ad8556d21ce3e6`. Only own-track architecture metadata were used to reconstruct case ownership, order and identity; answers/scorers were not used.

Qualified runtime receipt: `experiments/supa_adaptive_compute_v2/native_armr/evidence/00_runtime_qualification/20260905T191958_868458Z_82f1162e/run_manifest.json`; SHA-256 `52fe3ed8ccf9263c169ab06c85e2a30517b8eb13ad297581188d564f7fa1fc5c`.

Every audited started/final manifest retains the original registration and complete source map. Relevant source links are `shared/evaluation.py` (deterministic sharding, completion, baseline and confirmation prerequisites), `shared/generation.py:66` (generic guard), `shared/integrity.py` (freeze checks), `native_armr/src/operators.py` (original AM construction), and `native_armr/protocol/gpu_fp32_v2_2_amendment.md`. Exact registered/current hashes are recorded in the receipt. At this snapshot the first four relevant unchanged files other than operators match the original hashes; `native_armr/src/operators.py` differs during the separately active prospective amendment work. Historical manifests still bind the original source map. This audit does not qualify the modified current operator implementation or authorize combining it with original-registration science evidence.

There are **zero recorded native confirmatory runs and zero saved confirmatory cases**: no started receipt with a confirmatory role was found anywhere under native evidence, and no case artifact exists under the expected `02_confirmatory_run` stage. The original source checks the complete architecture matrix before creating any confirmatory run or loading a model. No original architecture attempt lacks a terminal receipt at the audit snapshot.

The supported interpretation is an incomplete original architecture campaign with an AM guard failure, preserved successful non-AM coverage, and confirmation still closed. No efficacy conclusion or control-scale choice follows from this audit.
