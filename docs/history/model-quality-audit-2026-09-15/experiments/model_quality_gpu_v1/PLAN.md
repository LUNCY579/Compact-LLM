# Quality-first programme and decision plan

## Goal and current evidence

Improve useful, grounded responses while making explicit tradeoffs in existing capabilities, uncertainty, format compliance and serving cost. No technique has privileged status. We will not call an architectural change progress unless it outperforms suitable ordinary-model controls.

At the reviewed RAB parent, Opallo's recorded BF16 panel result is 73/150 with 41/47 reference successes retained; its FP32 candidate scored 74. The published core metadata records a strict long-context regression, 161/256 to 135/256. A post-hoc answerable-subset decomposition suggests that presentation and missing-evidence behavior are productive repair targets; it does not prove general retrieval equivalence. The historical UNKNOWN counter and substring-tolerant analysis are not semantic-abstention certificates. See `AUDIT.md` for exact sources.

Several historical interpretations remain hypotheses: WSD alone caused a combined schedule/beta2 effect; nonsignificance proves retention; all CED readers are harmful; short unsuccessful reasoning-data runs prove a 596M capacity ceiling. None is adopted as an assumption of this programme.

## Stage 0 — finish localized behavior diagnosis (existing lanes)

Complete the separately designed prompt-format/literal-UNKNOWN screen with its own qualified native collector, then the existing three-seed repair-versus-control training registration. Keep exact sentinel compliance, semantic abstention, unsupported answers, and false abstention on answerable questions separate.

The historical repair compares a 5,233-token/41-row control with a 5,302-token/42-row treatment. It is approximately, not exactly, token-matched. It reconstructs from Supa; continued correction of Opallo would be a separate study. Do not silently amend those conditions through MQ-001. Preserve failures, legacy scores and checkpoint identity.

A successful prompt policy improves the system; a successful weight update improves the model. Neither may borrow the other's measurements. Stage-0 criteria do not constitute broad release approval.

## Stage 1 — objective × LR screen (implemented here)

Freeze one reviewed quality-focused corpus, including explicit response contracts and missing-evidence cases, before native execution. Compile exactly one selected token stream for each paired screen. Train all six cells in README independently from the same source. A screen uses at most 150,000 processed tokens across six cells (actual full-row coverage is reported), excluding qualification/evaluation.

The first screen is exploratory. A single seed can detect a gross failure and rank development candidates; it cannot establish general superiority. Select at most two configurations using development criteria declared before their outputs are opened, then repeat them under at least two additional paired draws. Separate `selection_seed` (which rows) from `order_seed` (ordering). Holding selection fixed and changing order does not test robustness to different training examples.

Primary comparison: generated-answer quality on declared machine-checkable tasks, constrained by unsupported-answer, false-abstention and retention rates. Development response CE is comparable across objectives but is not the quality endpoint. Whole-sequence and response-only arms must see the SAME examples and optimizer boundaries. Never match supervised-token counts by giving response-only arms additional training examples.

This implementation freezes WSD and beta2 together. It cannot attribute improvements to WSD independently. The token-group accumulation policy is a new design choice and must not be confused with the historical one-update-per-row runs.

## Data preparation, not indiscriminate scale

Proposed initial pool: 2,000 examples — 800 grounded answerability rows (400 matched scenarios), 600 task-solving examples, 400 independently verified preservation examples, and 200 response-contract examples. These counts specify a future reviewed pool; it has NOT been built or cleared by this branch. Prefer selecting suitable existing owned data over inventing an uncontrolled corpus.

An answerability pair should differ in whether the required evidence is present, not in an easy label-revealing style. Apply eligibility constraints symmetrically before selecting the queried entity, following the independently audited SupaExperiment generator principle. Do not use a frozen model's answer as a gold label without checking it. Include expected EOS behavior and vary answer-only versus explanation-requested contracts deliberately.

Separate rights/source review, label verification, ID/family overlap, normalized prompt overlap, semantic/scenario overlap, and generator-shortcut auditing. Passing one does not prove all the others. The compiler automates exact ID/family/prompt/digit-blind checks; the remaining review is mandatory and explicit.

The compiler consumes bounded, explicitly prepared single-turn TRAIN and DEV JSONL files plus an open historical exclusion bank. It does not inspect sealed confirmation outputs. It rejects long rows instead of silently changing the population. Large or multi-turn source corpora need a separately documented selection/conversion step.

## Stage 2 — measured exposure expansion (queued)

For a replicated promising recipe, register 25k, 100k and 400k processed-token comparisons. Distinguish additional unique examples from extra passes over fixed examples. Match family mixtures and report unique scenarios, answer tokens, processed tokens, updates, wall time, peak GPU memory and host/disk usage.

MQ-001's native CLI caps execution at the frozen 25k design. Larger budgets require an explicit successor registration; there is no hidden override. Do not infer that more data can never help from one larger, compositionally different failed run. Conversely, do not continue training solely because compute is available.

## Stage 3 — schedule and parameter-update refinement (queued)

Compare WSD to cosine with the SAME 20% warmup, peak/final LR, beta2, dataset and exposure. A separate beta2 factorial can resolve causal attribution if that question matters. Do not select between incompletely decayed and fully decayed endpoints as though their schedules were equal. Any WSD continuation/decay branch must restore complete optimizer, scheduler position, RNG and data state.

Keep last-20 tuning as the ordinary anchor. A proposed layerwise-LR challenger freezes blocks 0–7, uses 0.25× base rate on 8–15, 0.5× on 16–23 and 1× on 24–27. This is unimplemented and unmeasured here. Another challenger is LoRA rank 32 across attention and FFN projections, with comparable tuning opportunity rather than an artificially forced dense LR. DoRA follows only if the LoRA comparison identifies a useful limitation. QLoRA is a resource-enabling path needing its own quantized baseline and serving validation.

## Stage 4 — larger ordinary model and verified distillation (queued)

Since size is not a fixed product constraint, evaluate an ordinary 4B-class source before its CED hybrid. Qwen3-4B is a continuity baseline, not a claim to be the globally best current model. Use the corrected task protocol and independently qualified runtime; don't compare across different truncation, thinking, output limits or serving precision.

Use a stronger model as a teacher only where answers and, where relevant, intermediate steps can be checked. Train concise, task-relevant explanations rather than universal verbosity. Distinguish task acquisition, output-format learning and memorization of repeated templates. DPO or verified reinforcement learning may help once the policy already generates useful successes; they are not a default replacement for a broken supervised target.

## Stage 5 — architecture, compression and tools (conditional)

Preserve cross-scale E28/D36. Reopen it when complementary encoder information/computation is measured beyond a strong 4B-only adaptation and tokenwise-reader control. Sparse sites and compact readers are hypotheses, not established repairs. Measure intervention strength in activations, not output-weight norms alone. Do not mix whitening, bridge-rank changes, new memory selection and a new optimizer in one uninterpretable comparison.

Pruning is initially a serving-budget tool: establish a strong larger model, then distill/prune/recover and compare quality at the desired cost. Do not prune an already small checkpoint merely to claim an architectural advance. Retrieval, calculators and code execution may improve an end-to-end system; report assisted and unassisted quality separately.

## Decision and confirmation

A candidate is worth promoting only after a prospective decision policy specifies meaningful gains and acceptable losses. Preserve family and repeated-question dependence, training-draw variability and multiplicity. Do not turn repeated measurements of the same prompts into independent test examples. A non-significant comparison is not proof of equivalence. Report effect estimates, paired gains/losses and uncertainty; choose noninferiority margins BEFORE confirmation.

The old arithmetic panel, already-inspected Forge subsets and published long-context outputs are development/history, not fresh confirmation. Reserve a new family/scenario-disjoint set. Validate actual exported BF16/quantized weights under intended generation settings before release. No acceptance threshold, release, registry change or sealed evaluation is automated by this branch.

## Selected primary-method references

These motivate hypotheses, not claims of success on Rineya:

* Transformers chat templating: https://huggingface.co/docs/transformers/v4.57.1/chat_templating
* PyTorch checkpointing: https://docs.pytorch.org/docs/2.10/checkpoint.html
* WSD budget flexibility: https://arxiv.org/abs/2410.05192
* LoRA learning/forgetting tradeoffs: https://arxiv.org/abs/2405.09673
* QLoRA: https://arxiv.org/abs/2305.14314
* Distilling Step-by-Step: https://aclanthology.org/2023.findings-acl.507/
* CALM model composition: https://arxiv.org/abs/2401.02412

Implementation decisions should be verified against the pinned source/runtime, not whichever documentation happens to be latest.
