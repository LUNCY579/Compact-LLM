# Data card — Rineya Reasoning Forge RL 2.0.0

**Release date:** September 14, 2026, America/Chicago. **Language:** English. **Generation:** deterministic procedural Python, original fictional tasks, no external teacher calls. **Status:** machine-validated research candidate; human review and model efficacy unmeasured.

## Intended task

Acquire and evaluate a separation between detailed task solving and a compact public result. The package provides finite, explicitly defined problems whose answers, legal actions, optimization objectives, intermediate state transitions, and missing-evidence conditions can be checked without a learned judge.

This release is narrower than a general assistant corpus. It emphasizes algorithmic planning and sustained state rather than literature, open-world factual knowledge, nuanced conversation, creative writing, unrestricted programming, or multilingual tasks. Structured record syntax is deliberate: it improves mechanical verifiability but is not natural-language diversity equivalent to human documents.

## Canonical composition and derived views

There are 20,000 unique canonical formal cases: 16,000 TRAIN, 1,600 validation, 1,600 IID test, 640 transfer test, and 160 length stress. Every family has 2,000/200/200/80/20 rows respectively. Each canonical world has one final-only RL prompt and one reasoning-mode RL prompt. TRAIN and validation additionally have final-only and reasoning-SFT views: 17,600 examples in each SFT view, 35,200 total view records.

There are also 32,000 TRAIN preference pairs, 36,752 TRAIN local transition labels, and 10,000 teacher-forced episode decision views across all splits. Episode views come from 2,500 continuous-replanning parent worlds, with four stages per parent. Of those views, 8,000 belong to TRAIN, 800 to validation, 800 to IID test, 320 to transfer test, and 80 to length stress. The two pilot files reference 256 TRAIN and 128 validation worlds already counted above.

The 384 diagnostic probes use 64 existing IID-test parents and six views per parent: core-preserving short, early/middle/late/distributed long layouts, and an answer-changing task-evidence edit. The evidence edit is a dependent diagnostic descendant, not an independently sampled world. Some edits change a query or multiple parameters; these are not all minimal one-record interventions.

## Families

| Family | Required capability | Independent check |
|---|---|---|
| `ledger_replay` | Replay signed additions, transfers, scaling, voids, and a temporal cutoff; maintain four balances | Separately coded signed state updates |
| `register_machine` | Maintain three coupled registers through arithmetic, modular reduction, strict conditional branches, and swaps | Separate state-transition implementation |
| `temporal_route` | Reconcile approved edge versions, closures, and future drafts; optimize a directed route through ordered waypoints | Augmented-state Bellman–Ford versus constructive Dijkstra |
| `prerequisite_plan` | Search legal Boolean-flag actions with positive/negative preconditions, add/delete effects, and costs | Reverse value iteration with set-based transitions versus forward Dijkstra |
| `deadline_schedule` | Respect precedence, releases, deadlines, and one non-preemptive machine; minimize finish time | Exhaustive feasible-prefix DFS versus subset-dominance DP |
| `budget_selection` | Resolve approved limits; optimize item value under cost, weight, dependency, and exclusion constraints | Recursive subset traversal versus product enumeration |
| `evidence_chain` | Reconcile approved field versions and follow multi-hop relations; abstain on a missing approved field | Independent per-hop version selection |
| `continuous_replan` | Recompute a current dispatch route after accumulating approved updates and current-location changes | Snapshot route verification and causal-history checks |

Optimal-plan scoring checks the submitted witness, not exact equality to a single reference path. Equally optimal routes, action sequences, job orders, and item subsets receive credit. Budget selection uses maximum value then minimum cost; other planning families use their explicit minimum-cost or minimum-finish objective.

`UNKNOWN` means required approved evidence is missing in the defined task. `IMPOSSIBLE` means the complete specified action/constraint system has no feasible solution. Those labels are not interchangeable. They do not make claims about real-world uncertainty or impossibility outside the fictional rules.

## Long-context construction

TRAIN word tiers target 512, 1,024, 2,048, and 4,096 words, with 6,400, 4,480, 3,200, and 1,920 examples. Actual input lengths are 512–4,112 words. Layouts are balanced at 4,000 TRAIN examples each. The target scope's records are shuffled; temporal order must be recovered from tick fields. Same-grammar records from other scopes are distractors. Record IDs and field order are randomized.

Ledger and register problems increase relevant operations with context tier. TRAIN contains up to 128 ledger events or 96 register instructions before cutoff. Length stress includes up to 512 ledger events or 384 register instructions. Many other families intentionally separate long-context evidence selection from a much smaller planning space. Long input therefore does not imply that every word is needed for the solution.

The length-stress split has 80 examples targeting 8,192 words and 80 targeting 16,384 words; actual lengths are 8,192–16,400 words. Character spans of all target-scope records are retained. These spans indicate the target world, not a formally minimal proof or a verified minimal relevant-evidence set.

No tokenizer was run. Words, characters, and record counts are not token counts. Native token length, chat-template behavior, terminator identity, masks, runtime fit, and architecture-specific attention behavior all require separate qualification. The 160 stress worlds are a small diagnostic, not a robust broad benchmark.

## Transfer and shortcut limitations

Transfer increases the replay horizon at a given word tier, route graph size to nine nodes, workflow state width to eight flags with goal-off constraints, job count to eight, item count to twelve, or relation depth to six links plus a final field. These are parameter/horizon/goal shifts inside shared procedural grammars, not universally topology-heldout or template-heldout generalization tests.

Class proportions differ. For example, only 13 of 80 prerequisite-plan transfer worlds are solvable, whereas 67 are impossible. Report accuracy conditioned on ANSWER/UNKNOWN/IMPOSSIBLE, not only an aggregate that can hide majority-label behavior. The exact per-family class counts are in `reports/build_report.json`.

Other shortcut risks include fixed task instructions, finite entity vocabularies, decimal bit masks, repeated record-field schemas, easily filtered nonmatching scopes, solver-style traces, and limited corruption families. The process negatives perturb one state value or flag; the preference negatives are either a wrong concise result or prohibited extra prose. Those controls are not comprehensive adversarial training or general human preferences.

## Continuous reasoning and process supervision

Continuous here means state carried over many events and repeated decisions after causal updates. It does not mean continual learning, weight updates during inference, persistent cross-session memory, unbounded context, or training against unknown future observations.

The four dispatch snapshots expose only records known at that stage. Prior records persist and cutoffs increase. When a prior optimal path has a next edge, the next snapshot starts after that teacher move. When a previous task was unreachable or already complete, the generator may issue a new externally assigned starting location. This is an explicit limitation: the dataset is a sequence of teacher-forced dispatch decisions, not a physics-constrained on-policy route environment or a demonstrated long-horizon credit-assignment task.

A separate actual `PlanningEnvironment` supplies `reset`/`step`, persistent flag state, costs, terminal STOP/IMPOSSIBLE decisions, invalid-action termination, and step-budget truncation for prerequisite-plan worlds. Its observations do not include optimal costs or answers. No agent has been trained in it here.

## Reasoning traces

All 20,000 canonical worlds include a solver-generated method description and structured witness/replay checkpoints. The complete structured traces are independently reconstructed from visible records. For stateful replay and action planning, 198,770 transitions were independently replayed in the release validator. Trace wording is templated from these checked structures, not a human explanation-quality review.

These are reference solution artifacts, not private mental-state transcripts and not a guarantee of model reasoning faithfulness. An optimal witness plus exact oracle check is not a serialized proof of every rejected branch. Some impossible cases have a brief exhaustive-search conclusion rather than a detailed impossibility proof tree. The general outcome scorer never claims to validate arbitrary generated CoT prose; explicit local transition classification has its own evaluator.

## Validation and contamination boundaries

The independent release validator re-read all written files, parsed every model-visible canonical prompt, verified case/prompt identity, recomputed all 20,000 references using the separate audit algorithms, checked complete structured traces, validated both SFT views, and checked all process labels, preference directions, and causal snapshots. The two solvers share a grammar parser and authorship; agreement reduces certain bugs but is not independent formal certification.

Canonical IDs and exact normalized formal-case hashes are disjoint across splits. All derived views retain parent group IDs and source splits. Exact old/new prompt overlap is separately checked in the release audit; no semantic overlap audit against the user's private benchmarks is claimed. Published test labels remain developer-visible. The holdouts are not cryptographically sealed, independently administered, or immune to tuning contamination.

## Provenance and licensing

The inspected source archive was `Rineya_Reasoning_Forge_v1_FULL(1).zip`, SHA-256 `3d490c386bf5d0d832a860403253aeff29266ade84f93c384a0a3645cff0eaac`. Its role was design/context review; its rows were not copied into this release. No public benchmark rows, commercial documents, private account data, or model weights were included.

Data: CC0-1.0 intent. Original code: MIT. Primary research references are cited as design context, not imported corpus content. No legal warranty, model-quality guarantee, native integration claim, memory-fit claim, or architectural superiority claim is made.
