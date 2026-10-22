# MRS-1 implementation report

The frozen inference-only evaluation machinery passes software validation. Both real guarded attempts were refused for insufficient free host RAM before torch import or model loading. No model result exists.

Source: `LUNCY579/ChatBubble`, branch `codex/multi-reasoning-frozen-scout-v1-completion`, commit `a4a250338e85442b3762e9c5f0a620ee803c9969`. Freeze SHA-256: `278ce0962bd10864fba7d8ce2e7248579f18b22495d176c4f0f40db903a68837`. This report is additive evidence and does not amend the frozen source, bank, prompts, budgets, or protocol.

| Evidence category | Status |
|---|---|
| SOFTWARE PASS | TRUE: 96 scout tests and 19 supervisor tests |
| NATIVE EXECUTION PASS | FALSE |
| OPEN DEV RESULT | FALSE |
| MECHANISM SCOUT RESULT | FALSE |
| CONFIRMATORY CLAIM | FALSE |
| RELEASE CLAIM | FALSE |

Implemented and frozen: A0 ordinary direct; A1 single-call token allowance control; A2 four ordinary attempts plus selector; A3 independent R0/R1/R2/R3 roles plus checker-informed S2; analysis-only A4 oracle; S0/S1 selector diagnostics; and a software-tested M2 helper disabled in this experiment. The primary contrast is A3-A2; S1-A2 and A3-S1 separate candidate diversity from checker information. No implementation fixture is a pretrained-model efficacy result.

The immutable open bank contains 120 worlds: 8 SMOKE, 16 CAL and 96 CHECK, across register transitions, counterfactual deletion, prerequisite scheduling and evidence chains. Deterministic regeneration, independent solvers, structural uniqueness, separate gold/public files, parser behavior, label-blind role projections, paired scoring, all 16 future A/B/C/D outcomes, source/model/protocol bindings, create-only evidence, and resource/supervisor failure paths are tested. Native qualification and format smoke remain unexecuted.

The runner retains exact prompts and raw role outputs when generation occurs, uses one model and serial calls, and requires the existing live supervisor and GPU lease. Qualification, raw-file inventories, receipt bindings and scoring fail closed. The frozen future registration requires distinct checkpoint identity, compatible tokenizer/config/policy, and verified P0/P1/P2 receipt linkage. Paired analysis preserves repair/damage, family/difficulty strata, complementarity, selector diagnostics, costs, and C-A, B-A, D-C and interaction. `SOFTWARE_FIXTURE` and `OPEN_DEV_NATIVE_PAIRED` reports carry distinct evidence labels.

| Validation | Result | Receipt |
|---|---|---|
| Scout software suite | 96 passed in 3.86 s | [receipt](evidence/software-validation-001/receipt.json), [transcript](evidence/software-validation-001/pytest.txt) |
| Supervisor subprocess suite | 19 passed in 5.54 s | [receipt](evidence/supervisor-validation-001/receipt.json), [transcript](evidence/supervisor-validation-001/pytest.txt) |
| Supa guarded attempt | Refused: 2.099 GiB free host RAM < 6 GiB | [failure](evidence/native-supa-001/failure.json), [supervisor](evidence/native-supa-001/supervisor.json) |
| D1 guarded attempt | Refused: 2.265 GiB free host RAM < 6 GiB | [failure](evidence/native-d1-001/failure.json), [supervisor](evidence/native-d1-001/supervisor.json) |

Exact model assets and lineage are retained in [MODEL_MANIFEST.json](MODEL_MANIFEST.json) and [MODEL_LINEAGE.json](MODEL_LINEAGE.json). Repository, bank, prompt and protocol manifests are bound by [FREEZE.json](FREEZE.json). Both candidates retain their BF16 stored weights and planned FP32/eager/greedy/no-bias serving policy; no runtime conversion or model execution occurred in these attempts.

[CURRENT_CANDIDATE_RESULTS.md](CURRENT_CANDIDATE_RESULTS.md) gives the decision table. Machine-readable outcomes, resource measurements, complementarity placeholders and the explicitly empty raw-generation inventory are in the companion JSON reports. No model efficacy score or cost ratio is inferred from resource refusal.

The delivered `Rineya_LCR_Full_v1_6000` has a retained [P0 STOP receipt](evidence/dataset-acceptance-001/DATASET_ACCEPTANCE.json). P1/P2 remain closed and no training ran. The requested CC0 comparison is underway in a separate successor task; this Phase I report does not anticipate that comparison's conclusions.

Next, execute the unchanged old-model scout only when fresh host/GPU/disk measurements satisfy its frozen guards, using new evidence directories. A refused preload did not consume a loaded-model launch. Use the frozen [post-training contract](MULTI_REASONING_POSTTRAIN_CONTRACT.md), [continuation stages](PHASE_II_CONTINUATION.md), and [factorial plan](FUTURE_2X2_ANALYSIS_PLAN.md) when a fully accepted and trained future checkpoint exists. Do not produce a completed-study report while A/B/C/D remain missing.
