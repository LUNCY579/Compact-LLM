# Rineya Verified Resampling — RVS-001

**PLANNING_ONLY / NOT_RUNNABLE / NATIVE_UNQUALIFIED**

Recorded 2026-09-29. This package preserves the user's approved research direction following analysis of *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling*. Approval here is to save the work on a dedicated GitHub branch, not to execute a model experiment.

## Research question

Does Rineya fail because correct solutions are rarely generated, because an allowed selector cannot recognize generated correct solutions, or because recovering them costs too much?

The proposed order is: frozen-model diagnosis, deployable selection, resource optimization, then separately authorized TRAIN-only verified data generation and distillation. A correct answer appearing somewhere in a pool is not a deployed system result, and a deployment improvement is not automatically a checkpoint improvement.

## Lineage and exact candidate

| Field | Binding |
| --- | --- |
| Repository | `LUNCY579/ChatBubble` |
| New branch | `research/rineya-verified-resampling-v1` |
| Parent branch | `research/post-training-mimo-centralized-v1` |
| Parent commit | `ba683ac0fcadbf04e4e179200d035b6a980e388f` |
| Primary candidate | `Rineya-Titan-Experimental-1.0.0-devfreeze` |
| Recorded parameters | 596,049,920 |
| Recorded source precision | BF16 |
| Source weight SHA256 | `537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7` |

These identity facts are carried from the pinned centralized report [R1 in SOURCES.md]. This delivery does not independently load or hash model binaries. The older 344M plan is not the primary RVS candidate. No claim is made that Titan dominates other family members.

At the parent snapshot, centralized E0 is blocked by host RAM. Actual loaded weight tying, native parity, memory fit, generation throughput, and current-source acquisition/retention are not qualified. Historical measurements on other serving paths are not current BF16 baselines.

## Proposed smallest useful screen

One greedy completion and sixteen independently seeded stochastic completions for each of 128 preselected DEV parents, balanced across arithmetic, text, sets, and registers. Report budgets 1, 2, 4, 8, and 16 from the fixed stochastic bank. Total: 2,048 stochastic plus 128 greedy completions, or 2,176, excluding calibration, qualification, controls, retention, and later long-context work.

The pilot is exploratory. Dataset-parent IDs, decoding policy, strict parser, selector tie handling, seed derivation, cost limits, and qualification receipts must be frozen before any native collection. Missing values in `plan.json` are intentional blockers, not defaults. CHECK and FINAL remain closed.

## Package map

| File | Purpose |
| --- | --- |
| [RESEARCH_PLAN.md](RESEARCH_PLAN.md) | Mathematical rationale, interpretation rules, hypotheses, failure modes, and future research links |
| [PROTOCOL.md](PROTOCOL.md) | Proposed arms, metrics, splitting, leakage controls, resource accounting, and go/no-go decisions |
| [INTEGRATION_ROADMAP.md](INTEGRATION_ROADMAP.md) | Existing integration points, future interfaces, required tests, and bounded agent handoff |
| [SOURCES.md](SOURCES.md) | Primary paper/code references and immutable repository evidence |
| [plan.json](plan.json) | Explicit planning state, source identities, proposed counts, and closed permissions |

## Evidence and authorization ledger

| Item | Status in this delivery |
| --- | --- |
| Research analysis and implementation specification | Written |
| Machine-readable planning record | Written; not a runnable config |
| New RVS implementation or production changes | None |
| New RVS unit/integration tests | Not implemented; acceptance tests are specified |
| Native model loads / generated model tokens / optimizer updates | 0 / 0 / 0 |
| Teacher/API model calls / paid compute jobs | 0 / 0 |
| Protected evaluation payloads opened | 0 |
| RVS scientific outcome | UNMEASURED |
| Automatic experiment launch | Prohibited |

Do not reinterpret inherited PRLC/RHD fixtures as RVS implementation or native efficacy. Do not invoke inherited training commands merely because this branch contains those files. Do not overwrite historical protocols, reports, source weights, or existing run directories.

## Next permitted planning step

Implement and test a new CPU-only sample-bank/selection/evaluation boundary after explicit implementation approval. Native qualification and the 128-parent pilot require separate approval and completed entry gates. Record future amendments and actual receipts rather than changing this saved proposal into an apparent completed experiment.
