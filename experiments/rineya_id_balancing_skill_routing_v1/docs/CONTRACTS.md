# Data, execution, and checkpoint contracts

## Whole-call pool

A pool contains `Expert(name, operation, artifact_sha256, qualified, prior, strength)`.
Every name is unique, and every expert performs the same operation. `qualified`
is a curator's explicit admission decision tied to real evidence; a boolean alone
is not sufficient evidence. `artifact_sha256` is the registered tree-manifest digest.
Native `core` means the registered base with adapters off.

`PoolRouter.choose(operation, quality, predicted_cost, applicable, record=False)`
accepts finite success estimates in [0,1], nonnegative forecast costs in a common
unit, and an optional Boolean applicability mask. Gold outcomes are absent from
this API. `quality_slack` masks alternatives before costs and control bias are
considered. Stable ties favor earlier registry order, so preserve that order.

`CallExecutor` passes only the selected expert, fixed strength and request to the
backend. Its namespace includes bank, expert and coefficient identity. There is
no built-in memoization or reused cross-expert KV cache. The supplied native backend
accepts only complete prompts and token limits and serializes adapter changes.
Registered native PEFT strengths must be 1.0; non-unit coefficients require an
explicitly qualified external backend, not silent PEFT rescaling.

## Row format

Rows are JSON lists. Every row has `id`, `parent_id`, `family`, `split`, and `payload`.
Split is one of `train`, `calibration`, `dev`, `confirm`. For whole-call native
execution, `family` equals the pool operation and `payload.prompt` is the complete
preformatted input (no implicit chat template or truncation). Gold `expected_state`
and SFT `target` live outside `payload`; the backend never receives them.

Routing adds `quality`, `predicted_cost`, optional `applicable`, and quality-checkpoint
provenance. Native evaluation requires accounted `quality_feature_seconds`.
Counterfactual capture adds `outcomes`, `observed`, `observed_cost`, expert order,
responses, all-expert calls, and aggregate acquisition cost. The missing-label mask
is honored when fitting the quality head. No expert without observed training
labels gets a fitted quality output by pretending missing outcomes are failures.

`stratum` groups OOD challenges. Related variants share `parent_id` for clustered
analysis. Replay reads outcomes only after choosing an expert. Mutation tests
verify that changing the gold outcome matrix cannot change the selected routes.

## Controller lifecycle

`begin()` freezes bias across a window. Call `observe(counts, target_mass)` only
for completed assignments. Both vectors must be nonnegative and have equal total
mass; counts are integral. `commit()` is called once after a successful optimizer
step or a completed calibration window. `abort()` discards that window on failure.
Do not update in backward recomputation or during evaluation/generation.

State includes schema, exact expert order, artifact/config identity, current bias,
previous error, deferred counts/targets, update count, and a digest. Open windows
cannot be checkpointed. Mismatched order, identity, shapes, or bounds stop restore.
The same training/calibration controller state is immutable throughout evaluation.

## LoRA wrapper

`SparseLoRALinear` accepts only an ordinary FP32 `nn.Linear`. Base weights are
frozen and kept as the original object. A and B factors and router weights are
trainable; B starts at zero for exact initial parity. Composition coefficients are
fixed buffers. The first task step can have zero router gradient while B is zero;
subsequent steps provide a gradient through original normalized Top-2 scores.
Top-1 is intentionally rejected rather than silently removing its gradient.

Only selected token-expert pairs are evaluated. The bias is detached and does not
enter combination weights. Padding requires a correctly shaped Boolean mask.
Only valid positions are counted. Auxiliary loss uses normalized assignment
fractions and an E scaling factor; tune its weight separately because it absorbs
normalization constants relative to the paper's generic loss expression.

`inject` validates every explicit path before replacing modules; no broad regex
surgery. `adapter_state` excludes base weights and includes routing state and
coefficients. Native saved banks use safe tensor-state loading with a separately
verified SHA256; source artifacts never contain checkpoint bytes. Restoration
checks identity and tensors before parameter copying. Full optimizer-state resume
is not implemented; `load_bank` supports evaluation restoration only.

## Native support boundary

A single CPU/CUDA process, local safetensors, dense small Qwen3 config, ordinary
unmerged LoRA, FP32, and disabled TF32 are supported by the opt-in path. Custom
sparse/tangent/quantized modules must use a previously qualified loader and the
in-memory injection seam where appropriate. Never replace an unsupported module
with a superficially similar dense layer to bypass admission.

Preflight checks available free VRAM and input hashes, but is not a GPU lease or
peak-memory predictor. Use the existing host resource lock before native runs.
No host execution, Qwen3 forward pass, actual adapter switching, Windows/CUDA
qualification, or Rineya capability result has been obtained by this implementation
session. Mocked backends establish API mechanics only.
