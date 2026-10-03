# Formats and reward boundaries

## Dataset surfaces

An RL row has exactly `id`, `group_id`, `family`, `split`, and `prompt`. The prompt is a system/user message pair. Only `prompt` is model input. IDs and family labels are bookkeeping columns; the verifier uses the ID to access a separate reference record.

An SFT row has `id`, `group_id`, `family`, `split`, and `messages`, with one final assistant target. There are no generated test SFT files. Export tools refuse test-as-SFT even when held-out export is explicitly enabled for evaluation.

Metadata contains the parsed formal world, reference answer, objective certificate, trace checkpoints, teacher trace, source parameters, prompt hashes, word counts, and target-character spans. These are privileged supervision/audit fields, not extra context for a student model.

Process and preference raw files are parent-ID-linked, avoiding another copy of every long prompt. Export utilities materialize trainer-friendly prompt/completion or prompt/chosen/rejected records. Both derivatives are TRAIN-only.

The shared `group_id` is the split boundary. The pilot, SFT modes, four decision snapshots, transition examples, preference pairs, and diagnostic views must not be randomly repartitioned after export.

## Strict final-only mode

A completion is exactly one JSON object, optionally surrounded by whitespace. JSON fences, `<think>` blocks, extra prose, repeated JSON objects, duplicate keys, NaN, infinities, extra fields, wrong types, floating-point stand-ins for integers, and guessed status labels are not accepted. The evaluator does not award credit because a correct answer appears somewhere inside an otherwise noncompliant response.

Required objects differ by family: `{"value":integer}`, `{"answer":string}`, `{"path":[...],"cost":integer}`, `{"actions":[...],"cost":integer}`, `{"order":[...],"finish":integer}`, `{"items":[...],"value":integer,"cost":integer}`, or the task-appropriate status object. This is a task-specific public-output contract, not a proposal that all general assistant answers should be JSON.

## Reasoning-then-final mode

Exactly one nonempty `<think>...</think>` block is followed by exactly one `<final>...</final>` block, with no extra text. The final content must satisfy the same task schema. The reasoning block is an interchange representation for supervised task solutions; native model templates may use different representations.

The outcome reward does **not** read arbitrary prose to certify it as correct reasoning. `trace_verified=false` is intentional for generated responses, even when the final object is correct. The separate structured-trace validator certifies released reference checkpoints; the local transition evaluator certifies predictions about explicitly supplied one-step transitions.

No substring extraction fallback silently turns a visible reasoning leak into a valid final-only response. A deployment layer that intends to hide reasoning must explicitly separate it before returning public content and evaluate that exact end-to-end contract.

## Reward equation

Default: `reward = 1.0` when strict format and semantic witness/answer checks pass, otherwise `0.0`.

Optional efficiency experiment: for a correct result, subtract at most 0.05 based on redundant final serialization or tokenizer-measured trace-budget excess. Incorrect output always receives zero. The canonical length comparison uses the submitted accepted object, not the generator's chosen optimal witness. This avoids a built-in preference for one arbitrary optimal plan.

A correct final answer can coexist with an unfaithful trace; final accuracy is not process accuracy. A feasible but suboptimal plan is logged as feasible but gets zero default reward. A valid status response need not contain a constructive plan; its correctness is checked against the exact finite oracle.

## Native callback and truncation

`VerifiableReward` accepts `completions`, `id`, `prompts`, and additional keyword arguments. It returns one float per completion and rejects unknown IDs, changed prompts, and misaligned columns. Its reference store is TRAIN or validation only, verifies the frozen metadata checksum before loading, and omits teacher traces from the in-memory reward index. The callback is tested with synthetic batches, not an actual TRL run.

The offline evaluator supports explicit stopping reasons and treats `length`/`max_tokens` as truncated, not completed-correct. A native trainer must independently implement and validate its truncation mask/reward treatment; the callback cannot infer reliable termination from absent metadata. The package does not silently claim a DAPO-style truncation policy or a particular trainer loss normalization.

The evaluator accepts repeated samples only with explicit `sample_id` and `--samples-per-prompt`. pass@k is estimated only for complete groups and includes the number of eligible groups. Missing-output accuracy uses the full expected denominator. Wilson intervals are descriptive for single-sample canonical evaluation; shared templates limit idealized IID interpretation.

## Resource and safety boundaries

Oracles reject unsupported state-space sizes and invalid costs/durations. JSON parsers reject duplicate keys and non-finite numbers. The response parser has explicit size limits. No arbitrary code evaluation, shell execution, network access, or physical actions are performed by task execution.

The optional token preparer loads only local tokenizer files with remote code disabled. It does not load model weights or start training. Its generated attention mask is a token-presence mask, not proof of compatibility with a custom architecture.
