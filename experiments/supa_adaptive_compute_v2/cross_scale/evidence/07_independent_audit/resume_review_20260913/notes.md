# Track B resumption review — 2026-09-13 UTC

Reviewed HEAD `aaeb4d48db4476762588fba7418f06202fda4e21`. The complete CPU Stage 1 evidence remains **MECHANISM_GO**, with the original unique winner B3 at Supa post-L19 / donor L24. This review performs no candidate selection change, export, calibration, generation, model loading, checkpoint reading, or scientific source edit.

## Verification

Reran the committed standard-library MatrixAuditor in memory: 4,931 checks passed, all 221 streamed input hashes and the complete attempt inventory exactly matched the saved independent audit. Coverage is 16 depth pairs × 6 objectives = 96 cells, with 24 eligible candidates among 48 B1/B2/B3 gate records. The original gate arithmetic and ranking reproduce the committed selection exactly. Dense activation and bridge artifacts were streamed for SHA-256 only. No tensors were deserialized.

Called the real `generation_stage/v1/src/upstream.py::verify_upstream` on `alignment-cpu-v2-002`, aggregate attempt 0 and the saved audit directory. It returned `VERIFIED_MECHANISM_GO`, including commitment checks on 109 required files. The existing metadata-only capture reader returned no activation bundle. There is no upstream admission blocker at the reviewed source version. Future stage source changes require fresh admission and committed source before any export or generation.

`receipt.json` preserves an initial review-harness rejection: an in-memory dictionary used integer donor-depth keys whereas saved JSON used strings. JSON normalization resolved the comparison; all evidence values and hashes were identical. `verification_receipt.json` is the successful corrected result. `generation_inventory_receipt.json` classifies the three downstream test files accurately; the broad presence boolean in the successful receipt counted test artifacts and must not be read as evidence of model generation.

## Selected pair and same-pair null controls

All numbers below are held-out donor-transformation metrics copied from verified saved scalar evidence, not a fresh numerical recomputation. The 32 held-out scenarios contain 2,555 tokens: 2,395 ordinary and 160 special; the final-token stratum contains 32 overlapping tokens. Full up/down/delta metrics for all six objectives are retained in `verification_receipt.json`.

| Objective | Stratum | Tokens | Delta cosine | Mean token-relative MSE | Predicted/native RMS |
|---|---|---:|---:|---:|---:|
| B0 | all_tokens | 2555 | 0.724253072241 | 0.485771673661 | 0.771689714905 |
| B0 | ordinary_tokens | 2395 | 0.710979338733 | 0.507393897031 | 0.765185722765 |
| B0 | special_tokens | 160 | 0.922944270703 | 0.162114017597 | 0.877464047512 |
| B0 | final_prompt_token | 32 | 0.725118339242 | 0.486669292214 | 0.626298492453 |
| B3 | all_tokens | 2555 | 0.845053183569 | 0.349057410553 | 0.928921129844 |
| B3 | ordinary_tokens | 2395 | 0.849177584211 | 0.282078462603 | 0.922426676382 |
| B3 | special_tokens | 160 | 0.783316061456 | 1.351648537683 | 1.035720816569 |
| B3 | final_prompt_token | 32 | 0.916452090961 | 0.161857332249 | 0.924660740155 |
| BS | all_tokens | 2555 | 0.221570693731 | 0.983266601441 | 0.425571426702 |
| BS | ordinary_tokens | 2395 | 0.213387085493 | 0.994954611280 | 0.427813938714 |
| BS | special_tokens | 160 | 0.344069079546 | 0.808311704155 | 0.384336314068 |
| BS | final_prompt_token | 32 | 0.212776031994 | 0.956498985711 | 0.333462267351 |
| BR | all_tokens | 2555 | 0.049888618541 | 9.051916740933 | 2.574845072351 |
| BR | ordinary_tokens | 2395 | 0.051462961452 | 8.000688911279 | 2.577832866638 |
| BR | special_tokens | 160 | 0.026322673104 | 24.787483316056 | 2.522135011150 |
| BR | final_prompt_token | 32 | 0.075279059474 | 5.147043695388 | 2.090336716986 |

B3 improves ordinary tokens and final tokens. Special-token delta cosine and token-relative MSE are worse than same-pair B0; this was not silently excluded and does not violate the original ordinary-token selection and all-token floor. The future generation stage still needs to test whether the aligned residual improves behavior.

## Cost metadata

The saved donor layer schema implies 100,930,816 donor parameters, 100,925,440 donor matrix parameters, 5,246,464 stored bridge coefficients, 5,245,440 active bridge coefficients, 106,177,280 total added stored parameters/coefficient values, and 106,176,256 active values. Downward bias cancels from the residual. The preserved matrix estimate is 212,336,640 added FLOPs per token at nonzero gate, excluding attention-score work, normalization and instrumentation. These are metadata-derived counts; no live tensor inventory or generation latency was measured.

## Admission paths

- Registration: `C:\Users\clclc\BusinessProject\ChatBubble-cross-scale\experiments\supa_adaptive_compute_v2\cross_scale\evidence\00_preregistration\f8e4a26742effa443dfb843a00b61f34df09394db217cd9bb72120b1de7255ea\registration.json`
- Capture: `C:\Users\clclc\BusinessProject\ChatBubble-cross-scale\experiments\supa_adaptive_compute_v2\cross_scale\evidence\01_alignment_matrix\alignment-cpu-v2-002`
- Aggregate: `C:\Users\clclc\BusinessProject\ChatBubble-cross-scale\experiments\supa_adaptive_compute_v2\cross_scale\evidence\01_alignment_matrix\alignment-cpu-v2-002\aggregate\attempt_000`
- Independent audit: `C:\Users\clclc\BusinessProject\ChatBubble-cross-scale\experiments\supa_adaptive_compute_v2\cross_scale\evidence\07_independent_audit\matrix_audit_20260907T004813_5299633Z`
- Selected original bridge artifact: `C:\Users\clclc\BusinessProject\ChatBubble-cross-scale\experiments\supa_adaptive_compute_v2\cross_scale\local\alignment-cpu-v2-002\pairs\S19_D24\attempt_000\bridges.pt`
- Bridge SHA-256: `8bbf84ee6d3bffaf84cf28bdb5de37e6b11132a46438680b31dd02afb609d71a`
- Selected pair metrics: `C:\Users\clclc\BusinessProject\ChatBubble-cross-scale\experiments\supa_adaptive_compute_v2\cross_scale\evidence\02_bridge_objectives\alignment-cpu-v2-002\S19_D24\attempt_000\metrics.json`
- Full verified context: `C:\Users\clclc\BusinessProject\ChatBubble-cross-scale\experiments\supa_adaptive_compute_v2\cross_scale\evidence\07_independent_audit\resume_review_20260913\verification_receipt.json` under `upstream_verification.verified.context`.

Use B3 → XG, BS → XS, and BR → XR from that exact original pair artifact. No refit, seed change, re-ranking or dose decision was performed. Supa/donor hidden widths are 1024/2560. Stage 1 registration is `f8e4a26742effa443dfb843a00b61f34df09394db217cd9bb72120b1de7255ea` and the reviewed generation-stage source-map hash is `5c5d422aee7c715854dee629424a4ecd86fd32e71da27a3ae050e5b10d8acef1`.

## Remaining gates and limits

No real Track B architecture or confirmation evidence was found in registered downstream directories or run-manifest role fields. The only downstream artifacts are the existing 56-test/6-subtest tiny-fixture preparation receipt, log and JUnit report. The confirmation bank was not opened during this review. No real bridge export exists. The stage README still says the matrix is incomplete; that statement is stale, while its warnings that generation protocol, dose selection, matched control and runtime qualification remain unfrozen are still relevant.

The current upstream verification succeeds, but it licenses no generation. The separate small generation protocol, mandatory XM design and calibration, XN availability decision, dose-selection rules, complete stage source binding and runtime qualification still need their own freeze. Mechanism fidelity is not capability evidence. Saved frozen-state and zero-gradient assertions were checked for complete consistency, not by opening live model weights.

