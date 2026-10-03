# Bounded Rineya pilot and interpretation plan

## 1. Protect the experiment already in progress

Keep the current recovery run and its data/optimizer schedule unchanged until its planned evaluation point. Record the exact checkpoint, architecture/conversion revision, tokenizer, trainable parameters, optimizer state, learning-rate schedule, input tokens, and loss-bearing tokens. A reported advantage at 1e-5 over 1e-4 is a useful lead, not proof of universal non-regression. This bundle does not inspect or alter the user's repository permissions or experiment gates.

## 2. Qualify this candidate corpus locally

Run the stdlib tests and full release validator. Check the release file hashes. Audit overlap against the user's private/previous training and held-out files. Nothing was imported from those files during generation, but that alone is not an overlap guarantee.

Measure the exact model's rendered token lengths with `prepare_for_model.py`, separately for training and validation. Start with 2,048 as a **candidate filtering limit**, not as a claim of VRAM fit. Reject overlength rows without truncation. Verify model context support, padding, batching, loss masks, and a tiny forward/backward batch in the actual runtime before optimization.

Review at least two short and two long examples per family for readability, clear rules, supported labels, and nontriviality. For summary and communication, add human judgments of unsupported claims and naturalness. The current release has no independent human row review.

## 3. Separate architecture recovery from skill SFT

Broad next-token recovery and assistant-only instruction training have different losses. This corpus is primarily the latter. Do not turn its templated record padding into the model's entire next-token language diet.

Start from the chosen, qualified recovered checkpoint. Run a small skill-SFT comparison with identical optimizer settings: an approved general-data control versus the same mixture with approximately 20% of sampled SFT examples replaced by this custom corpus. This percentage is an initial hypothesis, not a proven optimum. Match total input and supervised-answer token budgets as closely as practical; record any mismatch rather than calling it a pure data comparison.

Use the local agent's already-qualified gentle learning rate as the initial candidate, without assuming 1e-5 is optimal for every parameter set or SFT mixture. A full-weight update, LoRA update, adapter-only update, and new-interface-only update need not share an optimal learning rate.

Use the 512/128 pilot first. Inspect several saved checkpoints rather than training many repeated epochs on the same 512 answers. Stop or roll back on a predefined general-language or task-regression limit. Avoid choosing checkpoints by training loss alone.

## 4. Expand only after a useful pilot

Move to the full training split, retain short and ordinary examples, and introduce longer rows only as the actual implementation's memory and context limits permit. Keep substantial non-synthetic general/instruction data. Do not respond to a disappointing pilot merely by generating ten times as many instances of the same templates: identify which error types need a new schema, better language variation, or missing knowledge.

## 5. Evaluate distinct questions separately

**Did the model learn these procedural tasks?** Evaluate per-family IID accuracy, not just one pooled average. The 14 exact families and two free-form families need different scoring.

**Did it generalize to harder instances?** Use transfer tasks, but describe their shared grammars and small per-family sample size.

**Did longer context or evidence position matter?** Use the paired probe views. Compare each parent world's short and long outputs; compare identical long record sets in different layouts. Use the parent world, not each view, as the independent unit.

**Did general capability regress?** Reuse your untouched general-language, dialogue, code/math, and real-document regression suites. Neither synthetic validation loss nor correct puzzle answers proves retention.

**Was any improvement architectural?** For a serious CED claim, distinguish these four cells:

| Architecture | No matched recovery | Matched recovery |
|---|---|---|
| Original Rineya backbone | Original checkpoint | Original architecture + same data/optimization budget |
| CED conversion | Converted, unrecovered checkpoint | Converted architecture + matched recovery |

Data improvements and extra training can help either architecture. Only comparing recovered CED against an untrained original conflates those effects. Efficiency claims additionally need measured prefill/decode latency, peak memory, and actual implementation behavior—not the architecture name alone.

## 6. Holdout discipline

Use validation for learning-rate, mixture, and checkpoint choices. Do not use the test or probe files in training. Reserve your independent final benchmark for the final planned comparison. A published labeled synthetic test is not the user's sealed benchmark, and repeated consultation can turn it into development data.

The release intentionally contains no training launcher. Data generation is not authorization to bypass your local model/conversion qualification gates.
