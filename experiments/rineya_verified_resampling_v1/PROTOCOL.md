# RVS-001 prospective protocol

**Draft for future registration; planning-only, not an executable authorization.**

The research direction is saved. Exact run identity, selected parents, decoding values, scorer rules, resource ceilings, and approval receipts remain unbound. No confirmatory or promotion claim is possible from this document alone.

## 1. Question and design

Primary question: under one frozen Titan checkpoint and sampling policy, how much do candidate coverage and allowed selected-answer accuracy change between 1 and 16 attempts, and what is the measured cost?

Independent variable: number of attempts available to the selector. Keep checkpoint, template, task, precision, per-attempt token cap, parser, and sampler fixed. Keep one separate greedy baseline; greedy accuracy is not sampled pass@1.

Proposed pilot: 128 DEV semantic parents, 32 each from arithmetic, text, sets, and registers. Use 16 independent stochastic attempts plus one greedy attempt per parent. Analyze k in {1,2,4,8,16}. Sampling count is 2,048; greedy count is 128; pilot total is 2,176 completions. This count excludes qualification, calibration, controls, retention, long-context work, and failures requiring later separately authorized repairs. It is not an all-in project budget.

No new chain-of-thought envelope, witness format, retrieval augmentation, adapter, training objective, or temperature change inside the primary bank. Any such change requires a separate arm or protocol amendment before outcomes.

## 2. Population and split governance

Use open DEV inputs only after their public/private export contracts are audited. Select parents by a fixed, recorded rule before candidate generation. Preserve the original split and semantic-parent identities. Do not select only previously wrong or source-correct questions based on this pilot's outcomes.

A disjoint subset of the existing open selection population may calibrate temperature and any selector threshold. Freeze its IDs, number of candidates, search grid, tuning objective, and all cost limits in an amendment. A two-temperature screen is a proposal, not a hidden default. Neither it nor a new selector may be tuned on the 128 pilot parents.

No CHECK/FINAL payload rendering, collection, oracle scoring, or training export is permitted here. Existing CHECK/FINAL remain closed. Exposed DEV does not become sealed by renaming it. For future family portfolios, all models use the same frozen parent population. For paired answerability and long-context variants, cluster by their common world rather than counting rows as independent tasks.

Preserve licensing and provenance, audit normalized-prompt and semantic-parent overlap, and disclose limits of decontamination. A hash-match check cannot certify unknown pretraining exclusion.

## 3. Required pre-collection freeze

Record immutable source commit; checkpoint/config/tokenizer/template hashes; actual loaded geometry and tied-weight checks; runtime/device versions; source and input digests; selected parent IDs; prompt token IDs; temperature/top-p/top-k and all penalties; RNG derivation; stopping rules; per-input output/total-context limits; strict parser revision; tie handling; candidate ordering; verifier feature allowlist; cost ceilings; and native qualification receipt.

Retain the parent's non-thinking template and task-specific answer contracts. Do not silently strip arbitrary reasoning or repair malformed outputs with an answer-aware parser. Legitimate empty text/set answers must follow the registered task semantics rather than a universal nonempty-answer rule. Explicitly test EOS and truncation handling.

Independent attempts use reset contexts and independently derived seeds. Prefixes are the first k attempts in frozen order, never reordered by correctness, completion length, or verifier success. Do not stop collection early when an oracle sees a success.

## 4. Arms and comparison roles

| Arm | Role | Gold available to selector? |
| --- | --- | --- |
| G | One greedy completion | No |
| S1 | First stochastic completion | No |
| RANDOM-k | Seeded uniform candidate choice | No |
| VOTE-k | Registered answer-normalized plurality/majority policy | No |
| VERIFY-k | Optional independently qualified public-information verifier | No |
| ORACLE-k | Coverage audit of the same candidate bank | Yes, evaluator only |

ORACLE-k is never a deployable arm. VERIFY-k is absent until qualified; do not relabel the deterministic answer grader as a deployment verifier. Freeze how ties, invalid candidates, tool failures, and no-valid-candidate pools are handled. A possible plurality tie rule is earliest valid candidate in frozen order; it must be selected prospectively, not after evaluation.

A checker that independently solves the task requires a direct-solver comparator. A selector that generates an answer outside the pool is reported separately from candidate-only selection. Any fallback abstention is an explicit output action with its own scoring and risk/coverage reporting.

## 5. Metrics and estimation

Primary descriptive contrasts: VOTE-16 minus S1 selected-answer accuracy and ORACLE-16 minus S1 correctness, paired on the same parents. Also report the separate contrast against G. VERIFY-16 is a later arm unless its entire definition is frozen before collection.

For the exact prefix pool at each k, report realized coverage C_prefix(k), selected accuracy A_prefix(k), and their matched gap. Conditional recovery is selected-correct parents divided by parents whose corresponding prefix contains a correct candidate; report null when that denominator is zero.

For a complete exchangeable n=16 stochastic bank with c_i correct attempts for problem i, additionally report the standard coverage estimator:

    pass_hat_i(k) = 1 - choose(n - c_i, k) / choose(n, k)
    pass_hat(k) = mean_i pass_hat_i(k)

Use stable product arithmetic and explicit c=0, c=n, k=1, and k=n cases. This is a random-subset coverage estimate. Do not subtract it from a differently distributed prefix selector result and call the difference matched selection loss. Report both prefix and estimator curves with their meanings. Any subset-averaged selector analysis needs its own label-blind, seeded subset plan.

Additional endpoints: malformed/unfinished-output rate; correct and false abstention; confident wrong answers; retained/gained/lost cases relative to each baseline; candidate and answer diversity; verifier acceptance and false-accept counts; accepted-answer precision; sample/prefill/decode/verifier/tool costs; and resource peaks. Diversity is descriptive, not evidence of reasoning.

Record every scheduled attempt slot and attempted/returned/failed/pending status. Timeouts and invalid outputs cannot disappear from denominators. Incomplete banks remain incomplete: report observed lower/upper coverage bounds and failure counts rather than claiming complete pass@16. If execution failures violate exchangeability, restrict the combinatorial estimator's interpretation and retain end-to-end prefix results. Never silently redraw failed samples under new seeds.

## 6. Statistical and interpretation safeguards

Use parent-level paired comparisons and parent-cluster uncertainty intervals; preserve all related rows within a bootstrap cluster. A thousand completions do not provide a thousand independent problem observations. Report within-domain results as well as the balanced aggregate. When the number of mechanisms is small, state that limitation.

The 128-parent screen is exploratory. Do not interpret nonsignificance as equivalence. Confidence intervals, meaningful gain/retention margins, multiplicity rules, and confirmatory sample size must be frozen before a later protected evaluation. Do not tune thresholds or sample size after inspecting protected outcomes. No promotion thresholds are silently invented here.

## 7. Leakage and verifier separation

Maintain separate public task views, public candidate views, private oracle records, and training-audit records. The selector receives only explicitly allowed fields; no answer, correctness, expected abstention, hidden source-family label, grader payload, or oracle scores. Remove privileged fields before serialization, not only from the model's prompt string. Use process/data-access isolation when feasible; a Python class name alone is not a security boundary.

All candidate correctness labels are produced after candidate collection by an independently bound oracle. Private grading artifacts and generated outputs are not automatically pushed to GitHub. A future training exporter accepts TRAIN only and rejects selection/DEV/CHECK/FINAL and replay-evaluation roles.

Challenge verifiers with correct minorities, repeated wrong majorities, all-wrong pools, malformed high-confidence outputs, absent evidence, and misleading answer-like distractors. Measure false acceptance across budgets. A test suite is only as strong as its available tests.

## 8. Controls and retention

Include task-appropriate random/constant baselines using public answer-space information or TRAIN frequencies. Include no-evidence and counterfactual controls where the task supports them. Never construct a guessing vocabulary from DEV golds. Extra control completions need explicit counts and cost ceilings before launch.

Selected-answer quality is not enough for promotion. A later qualified preservation panel must track source-correct losses, answerability, false abstention, and language behavior. The centralized language grader is unqualified at the parent snapshot; leave that endpoint unqualified rather than treating an uncalibrated judge as ground truth.

Long context is a separate phase: use the prepared 2024/4072-token prompts only after qualification at total sequence envelopes 2048/4096. Preserve the 24-token reserve, paired worlds, evidence positions, and distractor interventions. All eight related rows stay with their world. No long-context token count or run cost is included in the 2,176-completion pilot.

## 9. Resource plan and unchanged guards

The parent records a 7.4 GiB post-import host admission floor, 4 GiB free-GPU admission floor, 6 GiB worker-host limit, and external process-tree containment. Preserve those guards. Record fresh resource readings on the intended host; the source report's readings are historical, not measurements of this delivery's environment. Do not retry the same blocked state, lower thresholds, kill unrelated processes, or provision paid hardware automatically.

Qualification must bind the actual inference collector, exact weights, source revision, stochastic-token path, context envelopes, RNG behavior, and failure accounting. A separately registered inference-only envelope may be proposed, but does not inherit validity from CPU fixtures or an old training receipt. It must not relax the centralized campaign's rules.

Analytic ordinary BF16 KV estimate, not a measured fit certificate:

    KV_bytes = 2 * layers * kv_heads * head_dim * sequence_tokens * batch * bytes_per_element
    = 2 * 28 * 8 * 128 * 4096 * 1 * 2
    = 469,762,048 bytes = 448 MiB per cached sequence

Sixteen separate such caches use 7 GiB. Recorded weights alone are approximately 1.11 GiB at two bytes per parameter. Workspace, allocator overhead, and additional state are excluded, so sixteen ordinary full-length concurrent generations are not an 8 GiB fit plan. Begin with sequential sampling or a separately measured small microbatch.

Prefix reuse is a later qualified optimization; it may reduce repeated prompt work, not distinct continuation generation. Verify cloned cache isolation, no stale state, and unchanged output distributions. Record actual cache type, dtype, length, and sharing before making memory claims.

Account for prompts, decode tokens, all retries/failures, verifier calls, tool work, adapter switching, loading, and end-to-end wall time. Report steady-state and cold-start costs separately. Latency p50/p95 and cost per correct selected answer require actual measurements. Maximum pilot wall time, generation tokens, verifier calls, and output storage are null launch blockers until frozen.

## 10. Decision gates

G0: saved design only; current status.
G1: CPU contracts and adversarial tests implemented and reviewed; no native claim.
G2: exact-host inference qualification and explicit bounded collection authorization.
G3: complete fixed-policy exploratory pilot with all attempts and costs retained.
G4: qualified preservation and frozen selected system; separately approved protected confirmation.
G5: separately approved TRAIN-only resampling/distillation or other successor.

Stop on identity drift, private-field leakage, split collision, resource-floor violation, broken numerical/seed parity, unbounded work, or invalid grader contracts. Preserve partial evidence. Do not repair and silently continue the same scientific run.

If coverage is flat, do not automatically expand to hundreds of attempts. If coverage rises without selected accuracy, prioritize selection. If both rise but cost is high, optimize the operating point. If safety/preservation regresses, do not promote. Report the bounded conclusion even when negative.
