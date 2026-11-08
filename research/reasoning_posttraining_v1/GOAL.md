# GOAL — Rineya verifiable long-context reasoning post-training

## Mission and success definition

Continue the Rineya quality-first programme through implementation, qualified training, evaluation, and evidence-driven iteration. Improve semantic reasoning, multi-step planning, sustained state tracking, and long-context evidence use in an approximately 0.6B Rineya model while preserving useful ordinary responses, appropriate abstention, and concise public answers.

Prioritize better data, supervision, optimization, and bounded inference computation before architectural redesign. Seek improvement at the same native parameter scale; count actual unique and trainable parameters. Do not increase model size silently to satisfy the goal. A larger teacher or positive control is permitted only within separately authorized resources and must not replace the target on the scoreboard.

The September 8 research plan rejects a universal capacity conclusion from the tested failures. It does not guarantee that this checkpoint, this dataset, or this budget will improve. Preserve negative results and investigate testable bottlenecks. Neither prove a ceiling from another failed recipe nor claim to have broken a ceiling that was never established.

The deliverable is a measured, reproducible improvement or an honestly bounded, informative result with a resumable next experiment. A scaffold, passing unit tests, declining training loss, one favorable seed, or a published branch is not completion of the model-quality goal. Continue permitted work past those milestones.

## 1. Authority, isolation, and operational limits

Use the implementation, remote-write, model-execution, and compute permissions actually granted in this agent session. GOAL mode is not a spending authorization. Reuse an existing permit only if its exact scope applies and its budget remains unspent. Do not borrow consumed historical permits or expand a 25k-token registration into a larger run.

Before model execution, record numeric limits for total wall/GPU time, processed training tokens, rollout tokens, evaluation calls, disk use, and maximum adaptive rounds. Include qualification, failures, retries, and evaluation costs. If a required budget is absent, complete permitted audit/code/data work and request one concrete bounded approval. After approval, advance automatically within that envelope rather than asking permission for every routine step.

For modifications, create a new dedicated branch/worktree from the freshly inspected, appropriate live quality branch. Suggested successor: `research/rineya-reasoning-posttraining-v1`. Check branch ownership and local uncommitted work. Preserve all existing branches, immutable checkpoints, historical predictions, and other agents' work. Never force-push, delete branches, rewrite history, or overwrite published Opallo/Supa artifacts. Push and verify only the new branch when remote writes are authorized.

Allow only one model job at a time on the shared host. CPU model jobs also consume shared RAM. Keep process ownership, cooperative GPU locking, a real watchdog, RAM/VRAM/disk guards, and bounded cleanup. Never terminate someone else's applications, remove unexplained locks, lower guards to force a run, launch hidden jobs, or retry indefinitely. Do not start paid services, cloud GPUs, teacher APIs, or workflow dispatches without authorization. Inspect untrusted code before execution; do not enable remote model code automatically.

## 2. Inspect current evidence before choosing the next run

Read `AGENTS.md`, current registrations, source locks, qualification receipts, relevant code, and run outputs. Refresh live remote SHAs and local status. Historical anchors below are navigation leads, not guaranteed current state:

- `LUNCY579/ChatBubble`, `research/model-quality-targeted-gpu-v1`, observed at `8c3ed88a821b83e531ab682b3201879ea4fc928b`.
- `experiments/model_quality_gpu_v1/{README,PLAN,DESIGN,AUDIT,RUNBOOK}.md`, `MQ001_FIRST_PAIR.md`, `MQ001_CELL_PREREG.md`, and `MQ001_THREE_SEED.md`.
- Opallo's published `CORE.json`, source recipe, `OPALLO_PREREG.md`, `LONG_CONTEXT_RETENTION.md`, curated split builders/manifests, and current behavior/repair lanes.
- The attached `Rineya_Research_Plan_2026-09-08 (1).pdf`, especially measurement, conventional trainability, preservation, and agent-handoff sections.

At the observed head, the three-order report records response-only 13/51 versus whole-sequence 7/51, 7/51, and 8/51 at LR 1e-5. These use the same 238 training rows and the same DEV, with order seeds 17/23/29. The report's family-level p=0.0703 is not confirmatory success. Distinguish order replication from new training draws and independent evaluation worlds. Read the report's polarity-bias, objective-CE, and missing-supervisor qualifications. Four objective/LR combinations at 3e-6 and 3e-5 remain unrun in that record; do not confuse six executions at one LR with the completed six-cell LR matrix.

The inherited README describes an earlier implementation boundary. Reconcile it with later run receipts; do not repeat stale “no native training” statements. Conversely, a report does not repair missing raw evidence. Classify claims as source-reported, freshly reproduced, exploratory, confirmatory, blocked, or superseded. Preserve contradictions in a claim ledger instead of silently adopting whichever prose sounds most confident.

## 3. Recover and qualify reusable data

Locate the exact training selection, row order, source corpus, tokenizer, objectives, and checkpoint lineage that produced published Opallo. Distinguish the selected historical rows from the broader pool. Reusing that data, replaying its recipe from Supa, and continuing Opallo's weights are three different interventions.

Inspect relevant available pools, including `rag_reasoning_v3_large`/`splits_large`, general reasoning and natural communication corpora, short-response data, and `Rineya-Puzzle-Reasoning-51k`. These are candidates, not blanket-approved inputs. Keep previously contaminated or quarantined banks, including `splits_onpanel` if its quarantine still applies, out of clean training/selection claims. Missing historical rows must be reported, not reconstructed from a summary and called an exact replay.

Use the supplied original archives without modifying them:

`Rineya_Reasoning_Forge_v1_FULL(1).zip`
SHA-256: `3d490c386bf5d0d832a860403253aeff29266ade84f93c384a0a3645cff0eaac`

`Rineya_Reasoning_Forge_RL_v2_FULL.zip`
SHA-256: `d3bcd3d84c2d123c9876d334e867b0387f78cb7e93c7b625eafddbb0c685b663`

Verify actual bytes, manifests, and validators. Read v2's data card, training playbook, format/reward documentation, review checklist, and qualification summary. Its 20,000 canonical cases have 16,000 TRAIN worlds; related SFT, preference, process, and episode views are not independent tasks. Human review and native efficacy were not established by the release.

Create one lineage-preserving input contract recording source release/hash, original split, canonical world/parent IDs, scenario/structure identity where meaningful, response contract, evaluator version, review status, and actual token lengths. Keep gold answers, solver states, class labels, and selection metadata outside model-visible prompts unless explicitly designated training targets.

Only TRAIN sources enter optimization. Validation SFT demonstrations are not extra training data. Keep all paraphrases, layouts, reasoning/final views, counterfactual descendants, and episode stages of a parent together. Audit exact, normalized, near-duplicate, question, scenario, and structural overlap across old/new corpora and known historical evaluations. Do not demand impossible skeleton disjointness for trivial tasks; describe each split's actual generalization axis.

Separate deterministic validation from human review. Provide stratified review packets covering every selected family, label class, boundary condition, and length tier. Record who reviewed what; never manufacture human sign-off or self-certify an independent review. Apply one prospective inclusion policy across experimental arms. Quarantine unresolved label, rights, ambiguity, and leakage defects.

## 4. Freeze development and confirmation before optimizing

Separate TRAIN, repeatedly usable DEV, preservation/abstention DEV, and fresh confirmation. Historical panels and already-inspected Forge outputs are development/history. The distributed v2 test labels are developer-visible, not independently sealed. A checksum supplies identity, not secrecy or independence.

Reserve confirmation before adaptive selection. Keep its answers and per-item errors out of the optimizing agent's feedback, using an independent evaluation process where feasible. Include independently authored/rendered tasks beyond the Forge grammar, not merely different seeds or renamed entities. State any inability to establish independent confirmation.

Build same-world short/long and early/middle/late evidence variants. Vary operator depth separately from distractor length; separately test genuinely longer required state histories. Measure actual tokenizer lengths. V2's existing transfer split is primarily within-grammar parameter/horizon shift, and its target-scope spans are not certified minimal causal-evidence sets.

Test constant-label, cue-copy, last-mentioned, and other plausible shortcut baselines. Separate ANSWER, UNKNOWN, and IMPOSSIBLE strata and preserve their meanings. Include a positive-control exact solver and, where authorized, a qualified stronger model. A broken instrument is not a model failure.

Predeclare primary contrasts, task-family weights, cluster units, practical improvement/retention margins, uncertainty method, and the allowed number of model-selection reads. Suggested +5-point semantic improvement and at-most-2-point preservation loss are discussion defaults from the report, not universal requirements; finalize them using DEV variance and available budgets before confirmation. Never extend the test opportunistically until significance appears.

## 5. Make the native pipeline support the intended objective

Reuse qualified MQ-001 components where valid; do not rebuild working machinery for novelty. Its original source is dense Supa 1.7, approximately 596,049,920 unique parameters, with 314,618,880 trainable parameters in 220 last-20-block tensors. Recount and hash locally. The trainable count is not the total model size. Opallo continuation needs its own explicit source contract and baseline.

The old lane is a 25k-processed-token, single-turn, exact/numeric-output screen with an empty thinking stub and a 128-token evaluation limit. It is not already a long-CoT/RL trainer. Register a successor for reasoning modes, extended lengths, repeated exposure, planning executors, or RL; do not silently bypass old invariants.

Audit and fix the reported worker-output-directory/error-handler defect, supervisor memory overhead, and measured memory requirement. Restore supervised execution; do not copy the no-watchdog workaround. Keep mutable reports outside any source-hashed package where their addition would invalidate qualification. Preserve original receipts rather than retroactively manufacturing missing launch/supervisor records.

Pin model, tokenizer/template, runtime, precision, and evaluator. Prove generation-prefix equality, causal masks, padding/packing behavior, exactly-once label shifting, EOS inclusion, finite gradients, intended trainable participation, frozen-weight integrity, cached/uncached parity, and save/resume equivalence. Reject unsupported rows explicitly; never truncate away evidence, final answers, or EOS to make a batch fit. New bindings require new native qualification.

Use a fixed probe set for loss-progress comparisons. Losses on different minibatches are not a before/after learning test. Lower response CE is diagnostic, not independent proof of better generated answers.

## 6. Teach useful decomposition without mandatory verbosity

Maintain two explicit contracts: direct concise answers, and bounded task-solving traces followed by concise final answers. Treat traces as verifiable solution artifacts, not access to private model cognition. Tags alone neither create latent reasoning nor hide output.

Qualify the exact native thinking/non-thinking serialization. Do not append an active reasoning trace after a template that already closes an empty thinking stub. Train only intended assistant targets, including the selected reasoning fields/final answer/EOS, while excluding prompt, padding, and unintended headers. Preserve causal training: future target states are never input evidence.

Prefer compact evidence selection, operator choice, intermediate state, plan, and check targets. Do not indiscriminately copy enormous solver traces. Any compressed trace is a new checked target; arbitrary prefix truncation is not compression. Track reasoning and final-output tokens separately. Final-answer-only supervision does not automatically teach useful hidden multi-step computation.

Measure direct and reasoning-enabled inference separately, including all internal tokens and latency. A serving layer may suppress the reasoning channel, but must not suppress wrong answers or discard failed cases from scoring. The corpus's short structured answers do not justify forcing all ordinary conversation into JSON or an 80-character cap. Concision must follow each task's requested response contract.

## 7. Use a staged, informative training ladder

First run a bounded microfit on 32–64 verified TRAIN-only short problems. Prospectively define success, such as 95% canonical training correctness within the approved cap, with finite gradients and intact frozen weights. Failure triggers a focused optimization/mask/path diagnosis, not a capacity verdict. Microfit success is engineering evidence only.

Then resolve the most valuable remaining objective/LR uncertainty or expand the weak development instrument before multiplying experiments. Treat response-only as a promising default candidate, not a proven optimum. Preserve unrun registered cells as unrun unless an explicit amendment retires them. Do not launch a Cartesian product of models, LRs, adapters, curricula, and schedules.

For the initial data curriculum, a proposed processed-token allocation is 40% reviewed preservation/grounded-answerability data, 20% Forge v1 primitives, and 40% Forge v2 reasoning/planning. Reuse suitable Opallo-era data within the appropriate component. These are starting hypotheses, not established optimal ratios. Freeze the chosen quotas before training, report realized full-row allocations and supervised-token counts, and avoid blindly replaying a tiny historical selection to fill a quota.

Use a small sequential ablation ladder:

A. A reviewed pre-v2 SFT control using the selected qualified optimizer/objective.
B. The mixed corpus with concise final-answer supervision.
C. The same mixed task worlds with verified bounded reasoning/process supervision and concise finals.
D. The strongest qualified SFT parent plus verifiable RL, only after RL readiness passes.

For A versus B, hold the preservation component fixed and replace the v2 allocation with reviewed pre-v2 task-solving data in A. For B versus C, report both equal-world exposure and a separately budgeted token/compute-controlled view: longer traces are not free. Do not declare two arms simultaneously equal-world and equal-token when their targets differ. For D, compare against its frozen SFT parent and a continued-SFT control under a declared cost budget.

Begin with a small approved screen; enlarge exposure only after useful learning curves and retention survive. Separate unique-world expansion from extra epochs. Keep source, readout, data selection/order, WSD/beta2, update boundaries, and evaluator fixed where the contrast requires. Preserve the tail-20 anchor before comparing a qualified LoRA or broader-update challenger. Architecture changes, merging, pruning, and schedule/beta2 causal studies remain separate experiments.

## 8. Make verifiable RL earn its place

Before RL, test the verifier against correct alternative plans, invalid actions, nonoptimal witnesses, contradictory answers, malformed output, false UNKNOWN/IMPOSSIBLE, and reward-manipulation attempts. Keep semantic correctness and format compliance separately observable. Partial/process rewards require explicit checked fields; never credit arbitrary fluent reasoning as verified.

Run a small authorized on-policy pilot with the exact SFT parent. Measure successes, within-prompt reward variance, zero-variance groups, invalid outputs, entropy, truncation, and generation cost. If nearly every group fails identically, improve prerequisites or select a prospectively defined reachable TRAIN curriculum; do not purchase endless zero-signal rollouts. Keep excluded hard strata in evaluation and report all selection counts.

Use a pinned, actually tested GRPO or comparable implementation with explicit generation settings, group size, loss normalization, KL/reference policy, and update limits. Do not trust changing library defaults. Sample fresh completions; do not label static teacher demonstrations as on-policy RL. Start with correctness-dominated rewards. Add bounded efficiency incentives only as a separate ablation after correctness is stable; never reward merely being short or abstaining.

The static preference pairs support a separately labeled preference experiment, not a substitute for online RL. Local process negatives are limited transition corruptions, not a general process-reward model. Any learned reward proxy requires separate out-of-distribution and exploitation checks against the exact verifier.

Keep a preservation control during RL and record its objective/exposure explicitly. Preserve generation seeds, behavior-policy/checkpoint identity, raw completions, rewards, masks, and optimizer state. Keep gold references available to scoring, not to the policy prompt or an undeclared inference tool. Account for all sampled candidates, including rejected ones.

## 9. Advance long context and continuous decisions deliberately

Qualify short contexts first, then candidate 2K/4K native-token windows, followed by 8K/16K only where supported and affordable. Larger windows need their own qualification; no silent RoPE stretching. Check the combined prompt-plus-reasoning-plus-final budget. Word tiers in Forge are not native-token windows.

Cross length with operation depth, evidence position, distractor similarity, and required state-history size. Keep paired variants in one statistical cluster. Identify whether improvement is retrieval, binding, execution, or presentation. Oracle-evidence and extraction-then-solve arms are useful diagnostics; external execution is a tool-assisted system, not intrinsic model reasoning.

Use v2 dispatch snapshots as causal teacher-forced replanning data. They are not an on-policy physical routing environment or demonstrated long-horizon credit assignment. For actual action-taking experiments, qualify `PlanningEnvironment` or another explicitly specified simulator: model actions change state, future observations remain hidden, failures persist, and resets are recorded. Never replace a failed agent action with a teacher action and score it as agent success.

## 10. Run a bounded improvement loop, not an endless search

Maintain this cycle: inspect current errors → name one bottleneck and falsifying test → select or generate TRAIN-only tasks → verify and freeze a batch → train a candidate → compare on DEV and retention → keep, reject, or diagnose → record the next decision.

Prioritize mistaken operations, state drift, evidence/version confusion, invalid/nonoptimal plans, unsupported answers, and false abstention. Use exact solvers and checked task generation for new labels; the model's confidence or self-agreement is not a label oracle. Keep a uniform/replay control for failure-guided sampling to detect selection artifacts and forgetting.

Do not train on DEV/test error instances or near-duplicate “repairs.” Aggregate DEV findings may guide a predeclared curriculum family; document adaptation and keep a genuinely fresh confirmation bank. Count all attempted candidates and losses. Do not silently repair benchmark labels after viewing predictions.

Default to at most three adaptive research rounds unless the session grants a different finite limit. Stop a lane at its budget, a safety/validity failure, or a prospectively defined plateau. If authorized alternatives remain, move to the best-supported next experiment. A plateau retires the tested recipe, not all sub-billion research. Never stall in repeated status checks when no authorized next action can change the blocker.

## 11. Confirm and package only what the evidence supports

Select at most two finalists on DEV. Before confirmation, freeze checkpoint, inference policy, scorer, primary contrast, margins, and selection history. Use at least three independent training seeds/draws when claiming a reproducible gain and the budget supports them; report selection/order/initialization factors separately. Otherwise report the narrower exploratory result.

Compare world-paired semantic outcomes with appropriate cluster dependence and uncertainty. Repeated layouts and reused DEV worlds are not new independent evidence. Distinguish no statistically resolved effect, practically small effects, underpowered evidence, and equivalence under a predefined margin. Never pool repeated seed measurements into artificial sample size.

Report strict format, semantic correctness, feasibility/optimality, grounded evidence use, ANSWER/UNKNOWN/IMPOSSIBLE performance, false/unsupported answers, ordinary-language preservation, truncation/EOS, reasoning/final token counts, and measured latency/RAM/VRAM/disk. State exact denominators; nontermination and invalid outputs must not vanish. Primary accuracy is pass@1 under the declared policy; oracle-selected best-of-k is a separately costed result.

Do not generalize “reasoning improved” from a single templated domain. Seek transfer across at least two independently rendered composition domains and a non-registry task; otherwise release only a supported specialization claim. Preserve a frozen pre-update and pre-export reference. Requalify dense export, dtype conversion, adapter merge, and quantization independently. No automatic registry edit or production replacement.

After a candidate qualifies, test transfer of the recipe to another genuinely distinct Rineya weight lineage within a separately budgeted replication. Requalify each tokenizer, wrapper, abstention policy, and native parameter inventory. Shared-weight aliases do not establish family-wide replication; never copy UNKNOWN token IDs or biases between unqualified runtimes.

## 12. Persistent handoff and immediate next action

Maintain a compact scoped record set: `SOURCE_AUDIT.md`, `DATA_CONTRACT.json`, `EXPERIMENT_REGISTRY.json`, `RUN_LEDGER.jsonl`, `STATUS.json`, `RESULTS.md`, and `CONTINUE.md`, plus reproducible commands, tests, manifests, and actual run artifacts. Reuse existing equivalents instead of creating a parallel framework. Separate immutable source/qualification identity from mutable progress records.

Every continuation records current branch/SHA, model/data/runtime hashes, completed evidence, failed/unrun arms, remaining numeric budget, active job ownership, best checkpoint, exact next command, and genuine blockers. Report actual test counts/skips, optimizer steps, and completed evaluations—not intended counts. Check remote SHA/diff after authorized pushes; worktree cleanliness and ignored binary backup are separate checks.

Start now with a fresh targeted inspection, source/data inventory, current-run reconciliation, and the smallest missing prerequisite. Then implement and execute the next qualified, authorized experiment. Do not stop after another research proposal while executable approved work remains. Do not claim model-quality success until the declared semantic and preservation criteria are measured.
