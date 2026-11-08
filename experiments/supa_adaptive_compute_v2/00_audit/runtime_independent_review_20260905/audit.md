# Independent runtime evidence review — 2026-09-05

Native GPU qualification is supported by its frozen runtime gates. Cross-scale GPU qualification fails its frozen absolute-error gate and remains inadmissible. These are engineering outcomes; neither establishes an efficacy or transfer result.

Audit time: 2026-09-05T21:54:15.922880+00:00. Repository HEAD: `cdeb1e63297d51af6857753b92494e76c3f6f9ea`. Branch: `codex/supa-adaptive-compute-v2`.

Scope: read programme `AGENTS.md`, `GPU_STATUS.md`, the master goal, qualification implementations and validators, the two specified evidence directories, registrations, registered source files and own-track calibration manifests/files. A separate Python standard-library inspection reapplied the numerical gates, reconstructed expected coverage, and checked SHA-256 and Git content bindings. It loaded no model or checkpoint, launched no GPU work, accessed no sealed data, performed no architecture accuracy analysis, and ran no broad tests. Only this audit directory was written.

The receipt records **206 audit invariants checked, zero failed audit invariants**. This audit pass means the evidence is internally consistent, including the cross-scale failure; it does not turn that failed qualification into a pass.

## Native qualification

Evidence: `experiments/supa_adaptive_compute_v2/native_armr/evidence/00_runtime_qualification/20260905T191958_868458Z_82f1162e`.

Coverage is exactly eight first-sorted own-track calibration examples, one from each family, crossed with A0, A1, A3, A16 and AM: 40 unique comparisons. Prompt hashes match the frozen calibration. Each comparison covers hidden sites 8, 12, 13, 16 and 19. All 80 expected CPU/GPU first-token logit artifact names are present, and all 80 byte hashes match the report (48,745,680 bytes total).

The independently reapplied gates all pass, and every stored row gate/pass flag agrees. CPU/GPU greedy token sequences match in all 40 rows. Same-GPU repeat identity and same-device disabled/restored identity are recorded as true.

| Metric | Frozen gate | Observed worst value |
|---|---:|---:|
| First-token logit maximum absolute error | <= 0.002 | 4.1961669921875e-05 |
| First-token logit relative L2 | <= 0.0002 | 2.5817917451696963e-06 |
| First-token logit cosine | >= 0.999999 | 0.9999999999966989 |
| Hidden maximum per-token relative L2 | <= 0.0002 | 7.856709228698431e-05 |
| Hidden minimum per-token cosine | >= 0.999999 | 0.9999999969140225 |
| Absolute answer-NLL difference | <= 0.002 | 4.553794860839844e-05 |

Native hidden states have no frozen absolute-component-error cap; applying the cross-scale 0.002 cap to them would change the registered native gate. The qualifier evaluates FP32 tensors with FP64 diagnostic arithmetic.

The recorded CPU/GPU canonical state hash is `b23dec7edc2413189c9a35cc7a2596cdc03cb67b08628a0c443173c305d61551` and AM hash is `eedbc0ca788ee8a1193b455f7cf2c94cd1b5eeab288f1f2df448fa9ae208584c`. Their before/after and CPU/GPU identity records match; trainable backbone parameters and backbone gradients are zero. The Supa checkpoint identity matches the config.

## Cross-scale qualification

Evidence: `experiments/supa_adaptive_compute_v2/cross_scale/evidence/00_runtime_qualification/20260905T193023_051203Z_29be300a`.

All 656 expected labels/strata exist exactly once: 384 state comparisons (four Supa depths plus eight donor pre/post depths, eight calibration fixtures, four token strata), 256 projected B0 donor-delta comparisons (16 depth pairs, four held-out calibration fixtures, four strata), and 16 bridge-gradient comparisons (Supa L8, four donor layers, objectives B2/B3, up/bias). All/ordinary/special token counts partition consistently; final-token and gradient dimensions are correct. There are no empty special-token strata.

The eight fixtures exactly follow the frozen first-sorted-family alternating fit/holdout rule, with four fit and four held-out calibration scenarios. Their stripped prompt manifest and canonical digest match. Generation count is zero.

| Gate | Frozen criterion | Worst value | Failing rows |
|---|---:|---:|---:|
| Maximum absolute component error | <= 0.002 | 0.021484375 | 161 |
| Relative L2 | <= 0.0002 | 0.00018050146893744775 | 0 |
| Mean token cosine | >= 0.999999 | 0.9999999763539311 | 0 |

The 161 failures comprise 116 state and 45 projected-delta comparisons. All 16 gradient comparisons pass. Every saved pass flag agrees with the independently reapplied conjunction of all three gates. Cross-scale relative L2 is the global error norm divided by reference norm within each stratum, and cosine is the mean over nonzero token rows; these differ from native's maximum/minimum per-token hidden diagnostics. Gradient parameters are flattened to one row before comparison. The absolute gate is binding despite strong relative/cosine agreement and must not be waived retrospectively.

Same-GPU streamed captures, donor deltas, and bridge gradients are recorded bit-identical. All four donor layers have frozen/unchanged parameter records, equal CPU/GPU parameter bytes and zero backbone gradients/trainable parameters. The donor revision, model shard/config hashes and architecture-record hash match the registered donor record. The terminal manifest is FAILED, qualification_pass is false, and `failure.json` preserves the numerical failure. The ordinary admission code excludes this receipt.

## Registration and integrity

Native registration: `7c99dac8b3e72ad44ec2c1eeaaa67ac897fabe6d3209d362c1f78e01c3a7cca5`; execution commit: `e905e419c34bc3e1d0efb4f0082d0033b4f0b115`.

Cross-scale registration: `beec1869aad324d29191cd21e5f300bd6da3bdc0a52d8c5f11839ac4b98f5b7c`; execution commit: `c7cc6200164bb7d401b3413b489c32cb0da94d08`.

Both registration contract digests recompute correctly. Config byte hashes match registration, while canonical config hashes separately match execution receipts; these deliberately different hash representations are consistent. Started/final receipts bind the same registration, source map and dataset manifest. All 136 native and 139 cross-scale registered source hashes match current bytes. The programme source/config/calibration/manifest/registration files checked against their respective execution commits also match (37 native files, 38 cross-scale files, with Git newline normalization only). Both terminal report/manifest sets, including the cross-scale failure receipt, are committed at the audit HEAD.

The saved runtime identities both specify CUDA:0 FP32, eager attention, no cache, four threads, deterministic algorithms, highest matmul precision, disabled TF32 and the same RTX 4070 Laptop GPU/driver 610.62. This audit did not query the current GPU or claim a fresh runtime admission.

Native report SHA-256: `31f1759311ff4a70432def3bb55f52386b2ba7d801b3d9298e3c06bf7ced73c2`.

Cross-scale report SHA-256: `b037205c9ff8eca37017fa0575bb34ed87103f2eaec0cb97d7cc27df5bc456fb`.

## Material limits and interpretation

The audit independently verifies coverage, gate application, hashes and registration from saved evidence. It does not independently reproduce model arithmetic. Native stores CPU/GPU logit tensors, whose bytes were checked without deserialization; its hidden states, NLL calculations, repeated tensors and pre/post in-memory state hashes are represented by execution summaries. Cross-scale retains no raw comparison tensors. Therefore the numerical metrics and repeat claims cannot all be recalculated offline from these directories. Future prospective qualification evidence could preserve enough raw tensors or tensor-level digests to strengthen that audit, without changing the existing receipts or gates.

Canonical checkpoint bytes were not reread, and backbone state hashes were not regenerated. Their identities were cross-checked among config, reports and the registered donor record. The source verifier intentionally binds the user's preserved untracked historical graft by exact hash; the audit confirms that binding but does not imply every legacy dependency was committed at execution. No material mismatch was found within the inspected evidence.

The interpretation in `GPU_STATUS.md` is supported: native's passed profile licenses the unchanged native GPU architecture campaign subject to normal admission; cross-scale's failed GPU profile does not license GPU Stage 1 and supports continuing its unchanged CPU reference. This review does not separately requalify the later CPU registration or assess the active architecture campaign. Track-specific scientific decisions, evidence and runtimes remain separate. Qualification agreement is not a capability result, and a qualification failure is not evidence against useful donor transfer.

Machine-readable audit receipt: `receipt.json` (SHA-256 `f591f011eebe716261714d4b823e9d077bbcc55505b7b17c9fb81e11eee15b89`).
