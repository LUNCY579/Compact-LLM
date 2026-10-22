# Prospective future trained-candidate contract — MRS-1

The user identified `Rineya_LCR_Full_v1_6000` as the intended dataset. The completed [P0 receipt](evidence/dataset-acceptance-001/DATASET_ACCEPTANCE.md) binds manifest `65995575f13187ef78464e4120d1768696dd9114c7dfd8a379ed3ca6e28b266d` and records **STOP_NOT_ACCEPTED**. Train/dev token fit and answer rederivation pass, but native band, structure, shortcut, balance and contamination requirements prevent training. A repaired immutable revision requires a new P0 receipt. No future trained checkpoint exists for this protocol. This document does not itself authorize training, weight changes, a new model tournament, confirmation, or release.

The complete prospective P0–P8 continuation is [PHASE_II_CONTINUATION.md](PHASE_II_CONTINUATION.md). It requires local dataset acceptance and a frozen training contract before optimization, followed by the same paired A/B/C/D evaluation. These documents are unfinished drafts until the experiment creates and verifies `FREEZE.json`; references below to frozen artifacts describe the contract that must hold at execution. The final freeze must bind both documents, and later dataset/checkpoint registrations must be separate create-only records.

When it exists, register its exact checkpoint bytes, config, tokenizer, storage dtype, native runtime, training repository/branch/SHA and dataset provenance in a new create-only `future-registration.json`. Do not edit MODEL_MANIFEST.json or FREEZE.json. The registration must name the existing FREEZE.json SHA-256 and include `model` plus `training_provenance`. A runnable model entry has the same fields as the selected D1 entry, a distinct ID/path/weight SHA, and exact hashes of every ASSETS member. The runtime parser rejects policy/config/tokenizer incompatibility before loading. Model-specific semantic incompatibility creates a new experiment, not an amendment.

`registration.training_provenance` must contain `dataset_acceptance`, `training_contract` and `training_execution`. Each is an object with `status: "PASS"`, a local `path`, and `sha256` of that exact receipt. The runner hashes and reads the referenced file and requires status PASS and respective stage `DATASET_ACCEPTANCE`, `TRAINING_CONTRACT` or `TRAINING_EXECUTION`. All three receipts must bind the same `dataset_manifest_sha256` as the provenance. P1/P2 must also bind its `source_checkpoint_sha256`, and P2 must name the future exported `checkpoint_sha256`. Provenance contains `source_repository`, `source_branch` and `source_sha` equal to the future model's training-source fields. Preserve receipts and referenced artifacts; integrity checks are not independent proof of truthful execution. The runner rejects stale/failed receipts and inconsistent source, dataset or checkpoint identities before loading.

The new model ID must not collide with either registered existing candidate. Its resolved checkpoint path and `model.safetensors` bytes/hash must each differ from the fixed old D1 comparator. All other ASSETS hashes, storage/runtime dtype, attention backend, policy and decoding fields must match that comparator exactly. Registering an unchanged old export under a new name cannot fill C/D; incompatible assets require a separately named experiment.

The primary old comparison is `d1-seed17-bf16-export`; frozen Supa 1.7 is a separately labelled secondary reference. Never choose the old comparator after seeing the future scores. This is a comparison between exact retained checkpoints, not a claim that their historical training regimes were randomized or otherwise causally identical.

For a future entry, `model.source_repository`, `model.source_branch`, and `model.source_sha` identify its actual isolated training source. They must be consistent with `training_provenance`. The evaluator's own source is separately recorded as `binding.source` and remains on the frozen evaluator lineage. D1's inherited audit-parent fields are clarified by its exact historical training checkout in MODEL_LINEAGE.json; do not copy those fields into the new trained model.

1. Use the SAME `MRS-bank-1` public/gold artifacts and exact SMOKE/CAL/CHECK assignments.
2. Do not regenerate, replace, paraphrase or trim bank items after seeing the new model.
3. Use the same R0/R1/R2/R3 and selector prompts and information projections.
4. S2 remains the primary deployable selector; S0/S1 remain diagnostics. Do not tune it on CHECK.
5. Preserve deterministic greedy decoding, native chat template with thinking disabled, FP32 eager execution, no TF32, zero custom bias/logit policy, EOS, batch one and runtime versions.
6. Preserve 64 tokens/role, A1=320, five-call A2/A3, 2048 prompt cap and 4096 experiment context cap. No hidden prompt truncation.
7. Preserve the whole-JSON parser, no repairs, and EOS/nontermination policy.
8. No role is rewritten because the new checkpoint struggles with it. If the format smoke fails, report the failure and stop the experiment for that candidate.
9. Any necessary model, tokenizer, policy, precision or runtime incompatibility means a separately named experiment; no silent amendment or pooled comparison.
10. Run ordinary AND multi-reasoning arms and the same repeated-direct/token/selector controls. Preserve all per-item outputs.

BF16 retained storage is part of the comparator identity. Loading those bytes into FP32 does not recover discarded training precision. A future FP32-only checkpoint is not silently rounded under this inference-only contract; a separately documented and authorized exact export would be a new registered checkpoint identity. No original checkpoint is altered.

| Cell | Checkpoint | Inference |
|---|---|---|
| A | fixed existing D1 export | A0 ordinary direct |
| B | same existing D1 export | A3/S2 M1 multi-reasoning |
| C | newly registered trained export | A0 ordinary direct |
| D | same newly registered export | A3/S2 M1 multi-reasoning |

Evaluate each checkpoint after fresh no-update native qualification under the same guarded serial workflow. The new candidate has its own 1-load, 1800-second, 115232-output-token, 128-MiB bound; the current task does not schedule that future execution. Use a new evidence directory. If old A/B could not run now, finish those frozen cells when resources permit before inspecting C/D performance. Do not invent old results or select a different old checkpoint.

Then run the existing `analysis factorial` command on complete scored result files. It requires exact same item IDs and freeze/bank/prompt/protocol digests and compatible native serving policies/runtime. It computes C-A, B-A, D-C and (D-C)-(B-A), all specified repair/harm masks, family breakdowns and paired world-cluster intervals. Missing cells remain missing. No significance or confirmatory claim is created by completing the table.
