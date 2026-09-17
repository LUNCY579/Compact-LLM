# MRS-1 current-candidate results

**Software passes; model efficacy is unmeasured.** Both actual supervised native attempts stopped at the host-memory preload guard. Supa had 2.099 GiB and D1 had 2.265 GiB available against the frozen 6 GiB minimum. Both failure receipts report `model_imported=false`. There were zero model loads, generations, forward calls or scored cases.

| Evidence category | Status |
|---|---|
| SOFTWARE PASS | TRUE: 96 scout tests + 19 supervisor tests |
| NATIVE EXECUTION PASS | FALSE |
| OPEN DEV RESULT | FALSE |
| MECHANISM SCOUT RESULT | FALSE |
| CONFIRMATORY CLAIM | FALSE |
| RELEASE CLAIM | FALSE |

| Candidate | Exact weight SHA-256 | Attempt | Outcome |
|---|---|---|---|
| Frozen Supa 1.7 | `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b` | [native-supa-001](evidence/native-supa-001/failure.json) | BLOCKED ON RESOURCE LIMIT |
| Primary old D1 seed 17 BF16 export | `2d817d0c06ac11d655ff921780b4b132a6d7c966b31dd9a59e20e26eac24f4bd` | [native-d1-001](evidence/native-d1-001/failure.json) | BLOCKED ON RESOURCE LIMIT |

The planned serving policy is BF16 stored bytes loaded into FP32, eager attention, deterministic greedy decoding, no TF32 and no custom abstention bias. It has not received a fresh native execution pass. Scores, all paired deltas, complementarity, repair/damage counts and cost per recovered case are **null**, not zero. A/B and future C/D are all missing. Supervisor overhead is measured separately from nonexistent inference latency.

The exact prospective instrument is MRS-1 / MRS-bank-1 / MRS-prompts-1, with 96 CHECK worlds and fixed 8 SMOKE/16 CAL worlds. A0 uses one 64-token direct call; A1 allows 320 tokens in one direct call; A2 and A3 each use five logical calls with 320 allowed generated tokens. R0 is shared; 11 unique calls and at most 960 generated tokens per item cover all registered controls. The fixed primary future comparator is D1. The freeze SHA-256 is `278ce0962bd10864fba7d8ce2e7248579f18b22495d176c4f0f40db903a68837` at source commit `a4a250338e85442b3762e9c5f0a620ee803c9969`. [FREEZE.json](FREEZE.json) binds exact source/model/bank/prompt/protocol artifacts; future comparisons cannot tune those choices after results.

The delivered `Rineya_LCR_Full_v1_6000` separately received [P0_STOP_NOT_ACCEPTED](evidence/dataset-acceptance-001/DATASET_ACCEPTANCE.json): 41 native short-band violations were found in permitted train/dev, with unresolved structure/abstention/exclusion requirements. No holdout or protected payload was opened; full-package acceptance was not claimed. P1/P2 are closed. A CC0 comparison is currently underway in a separate successor task and has no conclusion in this report.

| Question | Evidence | Answer | Next step |
|---|---|---|---|
| 1. Does multi-reasoning beat ordinary direct inference? | Zero generated/scored cases | Unknown; A3-A0 is null | Run frozen A0/A3 once preload guards pass |
| 2. Does it beat repeated direct attempts? | No A2/A3 outcomes | Unknown; primary A3-A2 is null | Run registered A2/A3 and retain S1-A2/A3-S1 controls |
| 3. Is there complementary correctness between roles? | No role outputs | Unknown; all six pairwise matrices are null | Score frozen R0/R1/R2/R3 outputs by case and family |
| 4. Can the real selector exploit complementarity? | No candidate or selector outcomes | Unknown; S0/S1/S2 utilization is null | Measure real selection separately from A4 oracle headroom |
| 5. Where do treatment regressions occur? | No paired correctness outcomes | Unknown; repairs, damage and strata are null | Preserve every regression and report family/difficulty/answerability breakdowns |
| 6. What is the cost per recovered case? | No generated tokens or recovered cases | Undefined/null; CPU guard overhead is recorded separately | Use actual inference costs and positive recovery denominators only |
| 7. Which exact protocol is frozen for the future candidate? | Immutable MRS-1 freeze `278ce0962bd10864fba7d8ce2e7248579f18b22495d176c4f0f40db903a68837` | Same bank, roles, selectors, parser, greedy policy, budgets and scorer; old comparator D1 | Register a distinct compatible checkpoint without editing the freeze |
| 8. Is an internal/native mechanism justified? | Software validation and resource refusals only | No efficacy evidence supports internalization | First obtain real frozen complementarity and deployable-selector evidence |
| 9. What remains blocked? | Both host-memory preflights failed; dataset6000 P0 STOP; future checkpoint absent | Native qualification/A-B and P1/P2/C-D are incomplete | Preserve guards, finish separate CC0 comparison, repair/qualify dataset under its own gate |
| 10. What happens when the new trained checkpoint is available? | Frozen posttrain contract, P0-P8 plan and tested factorial code | Verify accepted data/training receipts and exact candidate compatibility; then run both C and D | Complete missing frozen old A/B before inspecting C/D; compute C-A, B-A, D-C and interaction |

Use fresh attempt paths such as `research/multi_reasoning_scout_v1/evidence/native-d1-002` and `native-supa-002` only after new preflight measurements pass; keep the refused `001` receipts intact. Invoke the existing runner from the isolated scout worktree as documented in [README](../../experiments/multi_reasoning_scout_v1/README.md), with one model loaded at a time. Do not reduce resource guards or silently alter the experiment.

See [machine-readable results](CURRENT_CANDIDATE_RESULTS.json), [complementarity status](COMPLEMENTARITY_MATRIX.json), [resource report](RESOURCE_REPORT.json), [raw-generation inventory](RAW_GENERATION_INVENTORY.json), and [implementation report](IMPLEMENTATION_REPORT.md). These reports make no confirmation or release claim.
