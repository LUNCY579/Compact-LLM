# Cross-Scale Capability Transfer Preregistration

Protocol version: `1.0.0`

Frozen before: architecture audit, donor projection, candidate construction,
or evaluation

Repository base: `27f807f039fae5c5cd20934100690606b5f9c9e7`

Research branch: `research/rineya-cross-scale-capability-transfer`

## Primary question

Can useful parameter-space information from official Qwen3-4B or Qwen3-8B be
projected into the coordinate system of a Supa / Qwen3-0.6B checkpoint and
improve the resulting approximately 0.6B model, without requiring a donor at
inference?

The intended result is a scientific determination, including a well-supported
negative result. A lucky development score is not success.

## Frozen scientific boundary

- Donors are official Qwen3-4B and Qwen3-8B checkpoints whose exact revisions
  and hashes will be recorded before use.
- The reference small model is official Qwen3-0.6B.
- The Rineya target is the latest published, single-checkpoint Supa A-series
  core that passes its published file manifest. The architecture audit must
  distinguish weight-space behavior from the non-weight abstention logit
  offset in Supa-A1.1. Both gated and ungated behavioral controls are reported;
  no donor merge may silently inherit credit for an external logit offset.
- The output state dict must retain the target architecture, exact target tensor
  shapes, and approximately the target parameter count. Donors are construction
  inputs only.
- All weight arithmetic is FP32. Storage precision is an explicit independent
  variable for finalists, never an implicit merge-method difference.
- Tensor semantics must be established programmatically. A method aborts on an
  unresolvable axis, head, vocabulary, tying, or key-layout ambiguity.

## Leakage firewall

No development, calibration, alignment, Fisher, layer selection, coefficient
selection, debugging, or recovery-training process may read:

- `Rineya_Invisible_Limit_Lab_v1/data/test2_answer_key.jsonl`;
- any file whose resolved path is under an Invisible-Limit sealed/TEST2 output
  directory;
- predictions, per-row errors, or decoded answers from Invisible-Limit Stage 6;
- the new study's sealed holdout after it is sealed, except through the single
  frozen final-evaluation entry point.

The implementation will enforce resolved-path denylisting and write an access
ledger. Existing Invisible-Limit Stage 6 files and protocols are read-only and
out of scope. Previously inspected Supa Showcase/Stability rows may be used only
as clearly labeled historical/preservation evidence, never as the new sealed
holdout and never as the sole selection substrate.

Calibration, MERGEDEV, DEV-FULL, and the fresh sealed holdout are disjoint by
normalized prompt hash, source document hash, scenario ID, and transformation
parent ID. Near-duplicate checks are performed before freezing each split.

## Hypotheses

### H1: heterogeneous interpolation

For donor `D` and projection `T`, a small positive coefficient can improve
Supa:

`M(alpha,D) = (1 - alpha) Supa + alpha T(Q_D)`.

### H2: cross-scale task vector

The relative direction

`Delta_D = T(Q_D) - Qwen3-0.6B`

contains more transferable information than the absolute donor, so

`R(alpha,D) = Supa + alpha Delta_D`

can outperform both Supa and matched absolute interpolation.

### H3: multi-scale consensus

The shared direction between projected 4B and 8B donors is more reliable than
donor-specific directions. The initial consensus is

`Delta_mean = (Delta_4 + Delta_8) / 2`.

Sign-consensus, cosine-filtered, TIES, DARE-TIES, Fisher-weighted, and normalized
shared-component variants are secondary arms.

### H4: scale-consistent coordinates

Transfer concentrates where `Delta_4` and `Delta_8` agree in sign, direction,
or relative magnitude. Agreement is measured globally and by layer, module,
tensor, and complete attention head where semantically valid.

### H5: depth localization

One-layer and coarse early/middle/late interventions will reveal beneficial and
harmful regions. Module localization follows only after a global transfer arm
passes the Merge GO gate.

### H6: interference-aware merging

TIES, DARE, DARE-TIES, Fisher weighting, magnitude masks, sign masks, or a
three-group coefficient schedule can exceed arithmetic transfer by reducing
cross-scale interference.

### H7: donor size is not monotonic

Qwen3-8B is not presumed superior. Raw capability may be offset by greater
projection distortion. All primary 4B/8B comparisons are paired under the same
projection family, coefficient policy, evaluator, and storage dtype.

## Architecture gate

Before transforming weights, record for all four checkpoints:

- resolved source revision and file hashes;
- config/model/tokenizer classes;
- layer count, hidden/intermediate sizes, Q and KV heads, head dimension;
- vocabulary size, embedding and output shapes, tied-embedding contract;
- norm and attention implementations, RoPE configuration, projection layouts,
  bias presence, dtype, special-token IDs, tokenizer hashes, and state-dict keys.

GO requires an explicit semantic mapping for every transformed tensor. State
dict validation requires zero missing/unexpected keys and exact shapes.

## Projection arms

The first confirmatory baseline is deterministic intersection with complete-head
attention selection and uniform full-depth mapping (`P1-D2-W2`). Additional
predeclared deterministic arms are prefix depth (`D1`), center-preserving depth
(`D3`), prefix width (`W1`), and deterministic norm-ranked structured width
(`W3`). D3's exact mapping rule is frozen in a manifest before evaluation.

Representation-based monotonic depth alignment (`D4`), PCA/SVD (`W4`), and any
learned linear mapper (`W5`) are exploratory. They use an independent calibration
split and cannot be presented as training-free confirmatory evidence.

Projection-only controls `T(Q4B)` and `T(Q8B)` must pass structural and functional
sanity tests before any merge using that projection is evaluated.

## Frozen coefficient grids

Absolute interpolation coarse grid:

`0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0`.

Task-vector and consensus coarse grid:

`-0.05, -0.02, 0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20`.

Negative coefficients stop after the first donor/projection pair if both
negative values cause either functional collapse or a preservation-family drop
of at least 10 percentage points from Supa.

Dense local search is allowed only between adjacent coarse points surrounding a
MERGEDEV winner. Its step and bounds are frozen before its predictions run.

## Advanced merge settings

- Cosine thresholds: `0, 0.1, 0.2, 0.3, 0.5, 0.7`.
- DARE drop probabilities: `0.05, 0.10, 0.20, 0.30, 0.50`.
- DARE and DARE-TIES use at least three predeclared seeds and report mean,
  standard deviation, best, and worst.
- TIES records trim fraction, elected signs, scaling, and conflict statistics.
- Fisher uses an independent balanced calibration corpus, clipped finite
  diagonal estimates, protected denominators, and a uniform-weight control.
- Initial single-layer coefficients are `0.02` and `0.05`; the exact choices
  may be reduced to the best safe low-alpha signal before localization, but not
  expanded based on holdout behavior.

## Evaluation levels

### MERGEDEV

A fast multi-family development suite for projection debugging, coefficient
screening, and algorithm screening. It is not used for final claims.

### DEV-FULL

A larger disjoint suite spanning general reasoning, paraphrase robustness,
counterfactuals, abstention, long-context retrieval, lost-in-the-middle,
multi-hop binding, common sense, mathematics, science, grounded QA/RAG, normal
conversation, instruction following, and short-response behavior.

### Fresh sealed holdout

A newly generated and audited bank unrelated to Invisible-Limit TEST2. Before
one final run, freeze dataset hash, evaluator hash/version, candidate hash,
model file hashes, tokenizer hash, inference settings, and statistical plan.
No result may trigger a method, donor, coefficient, mask, or checkpoint change.

## Inference protocol

Unless a family requires a documented alternative, comparisons use the same
tokenizer revision and chat template, greedy decoding, temperature 0, fixed
maximum new tokens, thinking disabled, fixed context/truncation behavior, seed,
batch size, device, and dtype. The resolved values are written to each run
manifest before inference.

0.6B evaluation is CPU-first with explicit thread count, CPU model, wall time,
TTFT, throughput, dtype, and peak memory. GPU utilization is checked before
every job. Active unrelated GPU work is never interrupted. Large-donor
activation capture may use a free GPU only when CPU is impractical.

## Primary selection rule

For each evaluation family `k`, normalize candidate score relative to frozen
Supa and Qwen3-0.6B baselines. Selection is lexicographic:

1. Reject functional failures and candidates with any family at least 10
   percentage points below Supa, more than 2% invalid generations, or a material
   refusal/formatting safety regression.
2. Maximize `min_k NormalizedScore_k`.
3. Maximize paired overall score.
4. Prefer the lower-compute, simpler, deterministic method.

A candidate passes Merge GO only if it exceeds Supa on MERGEDEV, passes all
preservation gates, and the improvement is not solely a non-weight logit-offset
effect. Localization GO requires the same positive signal on a larger DEV
subset. Confirmatory GO requires a completely frozen finalist packet.

## Candidate funnel

1. Projection viability on the minimum controls.
2. Coarse alpha screening on MERGEDEV.
3. Local alpha refinement around promoted regions only.
4. One-layer, layer-group, and module localization only for passing arms.
5. TIES/DARE/DARE-TIES/Fisher only for viable donor/projection pairs.
6. Deterministic successive halving, targeting roughly `100 -> 20 -> 8 -> 3`.
7. DEV-FULL promotion and finalist freeze.
8. One sealed confirmation.

Every generated or contemplated candidate receives an immutable registry ID and
status. Eliminated, failed, corrupt, and collapsed candidates remain recorded.

## Statistics

Per-row outputs retain scenario and transformation-parent IDs. Comparisons are
paired. Confidence intervals use scenario-cluster bootstrap; cluster-aware
permutation tests are used where suitable. Adaptive development intervals are
descriptive, not confirmatory. Error-complementarity sets include cases solved
by the merge despite both Supa and the projected donor being wrong.

## Low-cost recovery training

Only the strongest frozen training-free candidate may receive a tiny recovery
LoRA. It uses independent data and a preservation replay mix. Compare merge,
merge plus LoRA, and Supa plus the identical LoRA recipe. Recovery results are a
separate low-cost-training claim and cannot replace the training-free result.

## Success and interpretation

Scientific success requires a target-shaped candidate that passes all sanity
checks, improves robust DEV-FULL performance without severe preservation loss,
and reproduces the direction of improvement on the fresh sealed holdout. It
must require no donor at inference.

Small-alpha success with large-alpha collapse supports partial coordinate
compatibility. A 4B advantage supports projection compatibility over raw donor
scale; an 8B advantage supports transfer despite stronger compression.
Consensus beating both donors supports a shared scale direction. Task vectors
beating absolute interpolation supports relative rather than raw transfer.
Localized success supports depth/module concentration. TIES/DARE or Fisher
success supports interference or importance weighting. A development-only gain
is model-selection bias, not confirmation. A reasoning gain that destroys
general behavior is failure. No improvement is a valid negative result under
the tested projection assumptions.

## Amendment policy

This file is immutable after the first architecture-audit command. Any change
must be a new, timestamped amendment file that states what evidence was already
visible and labels affected analyses exploratory. Corrections do not overwrite
the frozen original.
