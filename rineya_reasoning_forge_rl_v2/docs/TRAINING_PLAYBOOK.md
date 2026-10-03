# Bounded training playbook

This is an experiment plan, **not an executed training recipe**. Do not promote the dataset to a proven quality corpus merely because the generators and validators pass.

## 1. Freeze the model and integration boundary

Record the exact checkpoint bytes and configuration, native tokenizer/template files, runtime versions, output mode, loss mask, assistant turn terminator, maximum context, generation budget, trainable parameters, optimizer, seed, and batch/token budget. Keep the current official dense Supa baseline separate from architecture comparisons. This artifact does not choose a new model checkpoint, alter a research branch, or authorize a GPU run.

Use the exact local tokenizer, not an arbitrary upstream Qwen tokenizer assumed to match a modified Rineya vocabulary. `prepare_for_model.py` records the local tokenizer asset inventory and supports an expected inventory hash. The checkpoint label passed to it is descriptive and is not a verification of model weight bytes.

For RL prompts, reserve generation space: `prompt_tokens + completion_reserve <= context_limit`. For SFT, measure the full templated sequence. Drop overlength candidates without truncating, and report exclusion rates by family, word tier, and answer class. Then review the changed mixture; filtering can remove the hardest or most valuable examples.

```bash
python scripts/prepare_for_model.py --input pilot/train_256.jsonl \
  --output prepared/pilot_train_ids.jsonl --kind rl \
  --tokenizer /LOCAL/EXACT/TOKENIZER --checkpoint-label EXACT_CHECKPOINT_ID \
  --max-length 4096 --completion-reserve 512
```

The 4096/512 values are **illustrative choices**, not a Rineya fit claim. Set budgets only after your exact model's context and output mode are qualified. A long reasoning completion often needs more than a concise final result.

SFT preparation additionally requires `--terminator-id` and exact prefix parity between the native generation prefix and the full teacher sequence. It labels only the assistant suffix, including the declared native turn terminator and any trailing template tokens. If prefix parity fails, the script stops rather than guessing the boundary. Run this in an existing qualified Transformers environment; core bundle tools need no Transformers.

## 2. Review before learning

Use `docs/REVIEW_CHECKLIST.md`. Review actual rendered prompts, not just structured metadata. Include solved, unknown, impossible, short, long, and changed-evidence cases. Inspect at least one decoded native batch from each intended output mode and prove that no world metadata, oracle certificate, future episode observation, or gold final answer has entered an RL prompt.

Review the loss-bearing tokens. Do not optimize raw metadata columns, system/user evidence, padding, or test labels. Token-presence attention masks from the preparer do not implement an architecture-specific CED attention mask.

Keep a reviewer decision file outside the immutable release. A suggested review completion criterion is zero unresolved label, rule, truncation, leakage, or termination defects in the reviewed sample. That threshold is a proposed local policy, not measured human review in this bundle.

## 3. Solve first; distinguish public brevity from reasoning cost

Use three explicitly different candidate arms from the same starting checkpoint:

**Final-only skill SFT:** `data/sft_final/train.jsonl.gz`. This tests whether compact answer demonstrations suffice for these tasks.

**Reasoning acquisition:** `data/sft_reason/train.jsonl.gz`. These targets contain visible textual reasoning delimiters. A suitable runtime must map or separate them correctly. Do not assume that a model's native reasoning template is identical to this interchange format.

**Reasoning acquisition followed by concise-response adaptation:** train the reasoning phase, then use the corresponding final-only examples or a controlled mixture. This tests a possible compression/non-thinking tradeoff; it does not assume that hiding/deleting the trace preserves the reasoning benefit.

Match both context tokens and supervised tokens, not just row counts or optimizer-step counts. Reusing the same world in two modes increases exposure; balance that exposure in the control. Keep a broad-language recovery/compact-instruction component from your existing research program rather than replacing it with this narrow synthetic corpus.

The optimum mixture, schedule, learning rate, LoRA/full-tune choice, and number of epochs are deliberately unspecified. They depend on your exact model and native gates. No optimizer defaults in a document should be mistaken for a proven safe setting.

## 4. Add verifiable RL only after the base can sometimes solve the tasks

RL prompts plus a verifier are not on-policy rollouts. Sample fresh completions from the current trainable policy. Preserve every requested prompt ID and sample index, generation stopping reason, token count, and reward outcome. Use TRAIN for optimization; use validation only for development decisions.

The default reward is 1 for a strictly formatted correct/optimal task result and 0 otherwise. Plan feasibility and formatting are separately logged; plausible reasoning prose earns no positive reward by itself. Alternative optimal plans count as correct. Missing evidence and true task impossibility use different labels.

The custom reward adapter validates prompt/reference hashes and batch alignment. It follows the documented TRL custom-reward callable boundary, but its native trainer behavior has not been executed. Establish a small native integration receipt before optimization: repeated deterministic scoring, grouped-sample alignment, padding/EOS handling, truncation behavior, zero hidden gold in prompts, and no unknown ID fallback.

Track the fraction of prompt groups with nonzero reward variance, as well as all-wrong and all-correct groups. A tiny model that gets every sample wrong on a task supplies little relative learning signal under common grouped objectives. Adjust TRAIN curriculum using recorded diagnostics, not by repeatedly tuning on test feedback. Do not silently remove hard or impossible cases from reported evaluation.

## 5. Conciseness is a gated secondary experiment

The default callback applies **no length penalty**. There is an optional bounded efficiency term in `score_response`, capped at 0.05, so any accepted correct result retains at least 0.95 while incorrect results receive 0. This pointwise ordering is not a guarantee of stable policy optimization.

The final-format component penalizes unnecessary serialization, not a different equally optimal plan's content. Trace-length shaping requires an actual tokenizer-measured trace count and an explicit budget. Whitespace words are not an acceptable stand-in for that token budget. Do not penalize a reasoning channel as if it were user-visible prose without measuring both separately.

Compare correctness-only RL against the same RL setup plus optional efficiency shaping only after competence and native rollout gates pass. Freeze development stop rules beforehand. Watch for reduced coverage, increased UNKNOWN/IMPOSSIBLE guessing, invalid empty plans, premature STOP, and accuracy loss. Also report end-to-end tokens and latency: a concise public answer can still require expensive hidden generation.

## 6. Process labels and preference pairs are optional ablations

The process file provides local transition classification, not complete free-form CoT truth labels. Export it using `--kind process`. Each parent contributes a few sampled valid transitions and mechanically corrupted counterparts. Keep all same-parent derivatives in the same split. Do not train a process model on hidden test-parent transitions.

Preference pairs are synthetic: correct concise output versus a wrong concise output, and correct concise output versus prohibited extra prose. They can test exact-task output discipline but are not broad human preference data. Avoid treating these simple pairs as a substitute for verifier-based RL or as proof that a model learned better reasoning.

```bash
python scripts/export_views.py --root . --kind preferences --split train --output prepared/preferences.jsonl
python scripts/export_views.py --root . --kind process --split train --output prepared/process.jsonl
python scripts/export_views.py --root . --kind episodes --split train --output prepared/episode_prompts.jsonl
```

Episode export strips verifier fields. The original episode audit files contain references and must not be passed wholesale into a model context.

## 7. Evaluate without moving the goalposts

Freeze train/dev inputs after review. Preserve all test groups and your existing independently sealed/private evaluations. Compare paired predictions from matched checkpoints on the same IDs. Report macro-family results; solved/unknown/impossible strata; coverage and invalid format; feasible versus optimal plans; context-layout robustness; transfer; measured generation length and cost; and non-regression on a broader external suite.

Use `forge_rl.probes` to score the 384 paired diagnostic views. The independent analysis unit is the 64 parent worlds, not 384 unrelated examples. Evidence-edit responsiveness is different from layout invariance: one should change the answer and the other should not.

For architecture comparisons, keep checkpoint provenance, data, update operator, optimizer budget, output mode, and runtime constraints matched. This corpus alone does not establish a CED benefit, looped-flow benefit, or shipping readiness.
