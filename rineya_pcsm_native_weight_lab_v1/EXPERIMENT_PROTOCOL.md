# PCSM pre-registered experiment protocol

## 1. Registration identity and status

This document freezes the decision rules for the Peer-Conditioned Stabilizer
Matrix experiment before a real held-out Phase-C outcome is available.

- Branch: `codex/exp-peer-stabilizer-matrix`
- Base commit: `b97cd8c636d1b1eebb9a450f369d789532f0dd4e`
- Authoritative frozen real budget:
  `results/phase_c/budget_manifest_v4.json`
- Budget canonical manifest identity:
  `88acd647d30ba1ea23b74b9a7f3079414c7cc6cb3991bd2b073681ab9ff937a9`
- Budget file SHA-256:
  `bc31ae08fcf8a1235d83887afb25aaaead409d56329b92d690a4ee8e8d2b6dc7`
- Real Phase C at documentation time: gradient caches and the resource pilot are
  complete; full qualification plans are **RUNNING**, while the confirmatory
  one-step outcome and advancement gate remain **PENDING**
- Phase D: **STOP unless a complete, valid gate artifact passes**

The checked-in synthetic results predate real-model outcome inspection and are
mathematical feasibility evidence only. The `SEALED` split remains unread during
selection and Phase C.

## 2. Research questions and hypotheses

### 2.1 Synthetic qualification hypothesis

When a controlled hidden sparse optimum is constructed from aligned peer
structural maps, real-peer PCSM should:

1. have lower final loss than random support;
2. have lower final loss than gradient TopK support;
3. have lower final loss than coordinate-shuffled peers;
4. recover more oracle support than random support; and
5. show decreasing independent meta-training loss.

This establishes only that the pathway can exploit peer-dependent information
when that information is made causally relevant by construction.

### 2.2 Primary real-model hypothesis

For the pre-specified Nimbus `mlp.down_proj` target, real compatible peers
predict a better one-step native update than peers whose coordinate structure
has been destroyed.

The primary paired difference is

\[
d_s = mean_b\{NLL_{coord\_shuffle,s,b}}
      -mean_b\{NLL_{real,s,b}\},
\]

where `s` is seed and `b` ranges over all five frozen support budgets. Positive
`d_s` favors real peers.

Null interpretation: if real peers are approximately equal to coordinate-
shuffled peers, coworker-specific coordinate information is unsupported.

### 2.3 Required robustness and secondary questions

- Does real-peer superiority survive a layer-identity/depth permutation?
- Does real peer information outperform target-only selection?
- Does support quality, rather than only the `A` scale head, explain any gain?
- Are results stable across support budgets and calibration subsets?
- Do exact-support and no-leakage invariants hold in every cell?

Target-only, gradient TopK, random, bootstrap, and causal-ablation comparisons
are important controls. They do not replace the pre-specified real-versus-
coordinate-shuffle primary test.

## 3. Model and target scope

The real experiment is intentionally narrow before any full SFT:

- base model: Rineya Nimbus 1.6.0 devfreeze;
- architecture: `Qwen3ForCausalLM`, 28 repeated layers;
- hidden size: 1,024;
- MLP intermediate size: 3,072;
- native dtype: BF16;
- declared maximum context: 40,960 tokens;
- target role: `mlp.down_proj`;
- target parameter: `model.layers.14.mlp.down_proj.weight`;
- target shape: `1024 x 3072`;
- target coordinate count: 3,145,728;
- eligible peers: every other same-role `mlp.down_proj` layer;
- self advice: forbidden exactly.

Base model weight SHA-256 is
`7b66b8d0ef06cf7f71bc9001b0c9a330d5f8a0eec1d11eb173ece70ca66ef49c`;
config SHA-256 is
`660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`.

No result from this single role/target may be generalized to all transformer
layers without replication.

## 4. Leakage-resistant data partitions

### 4.1 Group construction

Training lineage rows are connected whenever they share either an origin group
or source family. Whole connected components, not individual examples, are
assigned with a stable salted hash. The frozen allocation contains:

- 20 `META_TRAIN` semantic components;
- 5 `MASK_CAL` semantic components;
- all remaining training components unused;
- 1 outcome-blind validation component reserved for `ONE_STEP_DEV` candidates;
- test data reserved as `SEALED` and not opened by split construction.

The split manifest canonical identity is
`f15666248ce47dccad749199b47bd0f4fcf65bec7bc1f44f949929cc6dc78a2a`;
the file SHA-256 is
`980c5013cadb8770a02d80406d584fe5e1d40c3b01003ef57bb230b28e217afe`.

### 4.2 Split purposes

`META_TRAIN`

- calibrates generator/attention parameters;
- selects learning rate through first-order predicted loss change;
- never supplies final held-out evidence.

`MASK_CAL`

- supplies the task gradient used to build the fixed support and to take the
  one native update;
- never supplies the held-out outcome.

`ONE_STEP_DEV`

- measures actual assistant-token NLL after the frozen one-step update;
- may not influence descriptors, gradients, masks, attention, feature weights,
  stabilizer parameters, LR search, target choice, budget choice, or
  hyperparameter tuning.

`SEALED`

- is sourced from test data;
- remains unavailable until all model, arm, hyperparameter, and advancement
  decisions are frozen;
- is not needed to decide the Phase-C gate.

### 4.3 Outcome-blind resource freeze

Five examples per visible section were selected by stable hashes and had exact
content/token hashes frozen. To fit available hardware, the v4 budget then used
a quality-blind shortest-token rule, fixed before model execution:

- one 2,743-token `META_TRAIN` row;
- one 2,686-token `MASK_CAL` row;
- four `ONE_STEP_DEV` rows with 2,755, 2,785, 2,812, and 6,084 tokens.

The dev set therefore contains three approximately 2.8k examples and one 6.1k
long-context stress example. This is a real long-context probe but not a broad
long-context benchmark. The sample manifest file SHA-256 is
`83a22898dcb0efd5eed15d0811ac1956e689ff16f55e6f58f53d644e4fcbe42e`.

Every data read must reproduce frozen content hashes, tokenizer hash, full
token counts, and assistant-token counts. Silent truncation is forbidden.

## 5. Frozen budgets and exact K

The five real Phase-C fractions and resolved target support sizes are:

| Fraction | Percent | Exact K |
|---:|---:|---:|
| `0.0001` | 0.01% | 315 |
| `0.0005` | 0.05% | 1,573 |
| `0.001` | 0.10% | 3,146 |
| `0.005` | 0.50% | 15,729 |
| `0.01` | 1.00% | 31,457 |

K is resolved once by `floor(N * fraction + 0.5)`, clamped to `[1,N]`.
Every sparse arm in a seed/budget cell must have identical K. Support is
generated once, persisted, and never regenerated during the one-step run.
Therefore each static-plan support union must equal K.

## 6. Seeds and confirmatory arm matrix

The independent seed set is fixed to `11, 17, 23, 29, 31`.

Each of the five seeds and five budgets must run all eight arms from the same
checkpoint:

| Arm | Fixed mechanism |
|---|---|
| `no_update_frozen` | Reloaded base, no native update |
| `random_sparse_native` | Seeded exact-K support sampled uniformly without replacement; `A=1` |
| `gradient_topk_sparse_native` | Exact TopK of `abs(MASK_CAL gradient)`; `A=1` |
| `target_only_sparse_native` | Same generator family with peer inputs removed explicitly |
| `coordinate_shuffled_peer_pcsm` | Each peer's marginal coordinate distribution retained; flat coordinates independently permuted |
| `layer_shuffled_peer_pcsm` | Intact peers assigned incorrect depth identities |
| `real_peer_pcsm_bootstrap` | Real same-role peers with zero generator calibration steps |
| `real_peer_pcsm_calibrated` | Real same-role peers with the frozen 20-step calibration |

This is 200 one-step cells before per-example records. The pilot is a separate
resource/correctness screen: seed 17, budget 0.001, one dev row, and five arms.
It cannot pass the scientific advancement gate.

## 7. Required causal ablations

The following ten ablations are pre-registered. Overlaps with the core eight-arm
matrix are noted, but none may be silently substituted for another:

1. **Coordinate shuffle**: destroy peer coordinate structure while preserving
   each peer's value distribution (core arm).
2. **Layer shuffle**: assign intact peer descriptors/maps to incorrect depths
   (core arm and gate robustness test).
3. **Module-role shuffle**: deliberately admit incompatible module roles; label
   this as a causal stress test, never a production peer policy.
4. **Uniform peer attention**: use equal alpha over all eligible same-role peers.
5. **Nearest-layer only**: admit only adjacent compatible layers.
6. **Target-only**: remove peer inputs (core arm).
7. **Gradient-feature only**: recover ordinary gradient TopK inside the same
   scoring code path.
8. **No step-scale head**: fix `A=1` while retaining PCSM support.
9. **Random generator**: identical generator architecture with untrained random
   weights.
10. **Structural regularizer off/on**: compare coefficient 0.0 with the frozen
    0.05 row/column energy regularizer; never bundle it into the core method.

Coordinate and layer shuffle are mandatory for the advancement decision. The
complete suite is mandatory before a broad causal interpretation or final
Phase-D paper-style claim. Missing ablation artifacts must be reported as
missing; they may not be inferred from unit tests.

## 8. Frozen generator and optimizer choices

Generator:

- descriptor dimension: 24;
- hidden dimension: 64;
- calibrated steps: 20;
- bootstrap steps: 0;
- calibration learning rate: 0.01;
- exact-forward TopK STE temperature: 1.0;
- stabilizer bounds: `[0.25,1.0]`;
- primary structural regularizer coefficient: 0.0;
- regularizer ablation coefficient: 0.05;
- structural proxy learning rate: 0.0001.

Native one-step optimizer:

- `ExactSupportAdamW`;
- learning-rate grid: `5e-5, 1e-4, 2e-4`;
- LR selection uses `META_TRAIN` first-order prediction only;
- betas: `(0.9,0.999)`;
- epsilon: `1e-8`;
- selected-coordinate weight decay: `0.01`;
- `A` multiplies the total selected direction after Adam normalization,
  including selected decay;
- moments remain FP32 and exist only on support.

The decisive real and coordinate-shuffled arms use identical architecture,
data, generator steps, generator learning rate, K, optimizer, LR search space,
and evaluation budget. Classical random/gradient arms match K and native update
mechanism but intentionally do not receive extra generator compute. Bootstrap
versus calibrated is an explicit compute/mechanism ablation, not the decisive
capacity-matched peer test.

## 9. One-step execution procedure

For each arm/seed/budget cell:

1. Verify base model, config, tokenizer, split, sample, and gradient-cache
   identities.
2. Load same-role native weights and cached FP32 `META_TRAIN` / `MASK_CAL`
   gradients.
3. Calibrate the declared generator mode without any native optimizer step.
4. Freeze exact K once; persist mask, full `A`, attention, feature weights,
   metadata, and content hashes.
5. Discard selection state.
6. Load a fresh byte-identified base model.
7. Verify that the plan cell matches seed, budget, arm, target, mode, K, and
   calibration compute.
8. Select LR using `META_TRAIN` only.
9. Bind the exact plan, apply one `MASK_CAL` native update, and clear gradients.
10. Bitwise-audit the target and hash every unplanned parameter.
11. Evaluate actual `ONE_STEP_DEV` assistant-token NLL.
12. Persist the cell immediately with plan/support hashes, delta audit, compute,
    memory, and environment data.

No arm may reuse the mutated model from another arm.

## 10. Outcomes and metrics

### 10.1 Primary Phase-C outcome

Token-weighted mean assistant-target negative log-likelihood on the four frozen
`ONE_STEP_DEV` rows after exactly one native update. Score correlation alone is
not an outcome.

### 10.2 Required per-cell evidence

- per-example and aggregate NLL;
- seed, budget fraction, exact K, and chosen LR;
- plan hash and support hash;
- planned, union, changed-inside, and changed-outside coordinate counts;
- L2/max-absolute deltas inside and outside support;
- attention alphas and feature weights;
- stabilizer distribution and effective LR distribution;
- gradient norms;
- unplanned before/after fingerprints;
- wall-clock and peak allocated/reserved VRAM;
- optimizer state bytes and estimated checkpoint delta bytes;
- git branch/commit, base/model/config/tokenizer/dataset/split/sample hashes;
- Python, PyTorch, Transformers, CUDA, driver, and GPU identity.

Support overlap across seeds and calibration subsets, nearest-versus-distant
attention contribution, and context-length-stratified effects are reported as
exploratory diagnostics, not replacements for the aggregate gate.

### 10.3 Phase-D quality metrics, if authorized

Task/capability scores, long-context reasoning, ordinary-chat retention,
task-specific exact match, abstention behavior, train/dev loss curves, tokens,
steps, throughput, gradient norms, support stability, delta audit, peak VRAM,
wall-clock, optimizer/checkpoint bytes, and all reproducibility identities.

## 11. Statistical analysis

The real-versus-coordinate-shuffle analysis is **PRE-SPECIFIED CONFIRMATORY**.

1. Compute token-weighted dev NLL for each cell.
2. Average the five budget-cell NLLs within each seed with equal budget weight.
3. Compute five paired differences
   `coordinate_shuffled_loss - real_peer_loss`.
4. Report all five differences, wins/losses/ties, mean, median, sample standard
   deviation, standard error, and standardized paired effect `d_z` when the
   paired standard deviation is nonzero.
5. Compute a deterministic 10,000-resample paired percentile bootstrap 95%
   interval on the mean difference.
6. Report the exact two-sided paired sign-flip test as a secondary finite-sample
   diagnostic. With only five pairs its minimum attainable two-sided p-value is
   0.0625, so the gate is not defined by `p < 0.05`.

The layer-identity permutation comparison receives the same paired treatment.
Per-budget pairwise results and all other ablations are explicitly
**EXPLORATORY** unless a later protocol is frozen before access. They are
reported completely; no best-seed, best-budget, or uncorrected cherry-picking
is permitted.

## 12. Advancement gate

Advance to Phase D only if every condition holds:

1. the complete 5-seed x 5-budget x 8-arm qualification run and plan set exist;
2. real peers beat coordinate-shuffled peers in at least 4/5 seed aggregates;
3. the paired 95% interval for
   `coordinate_shuffled_loss - real_peer_loss` is strictly above zero;
4. real-peer superiority also survives layer-identity permutation (at least 4/5
   wins and interval strictly above zero);
5. real/shuffled arms have identical K, architecture, data, optimizer, LR search,
   generator capacity, and meta-training compute;
6. every exact native-delta audit passes and every unplanned parameter hash is
   unchanged;
7. `sealed_test_read` is exactly false in all input artifacts;
8. all required seed/budget/arm cells and plan identities are present exactly
   once;
9. no material retention or ordinary-chat regression is concealed;
10. the result is not invalidated by any rule below.

If any condition fails or evidence is missing, the verdict is
`STOP_NO_PHASE_D`. A pilot result, even if positive, cannot advance.

## 13. Invalidation rules

A run or cell is invalid if any of the following occurs:

- test or sealed data influences selection, tuning, or target/budget choice;
- a frozen content, token, model, config, tokenizer, cache, plan, support, or
  tensor hash mismatches;
- silent truncation or zero assistant-target tokens occurs;
- self advice or incompatible peers enter a production arm;
- K differs from the frozen budget or sparse arms have unequal K;
- a support changes after plan generation or union exceeds declared static K;
- any bit changes outside support;
- an unplanned parameter fingerprint changes;
- momentum, weight decay, or resumed state acts outside support;
- non-finite scores/scales/gradients or non-positive scale values occur;
- an arm is missing, duplicated, mislabeled, or bound to another seed/budget;
- real and shuffled generator capacity/meta compute differ;
- an arm reuses a mutated checkpoint;
- evaluation procedures or benchmark logic change after outcome inspection;
- only successful seeds/budgets are retained;
- previous negative evidence is overwritten.

Invalid runs remain preserved and labeled invalid; they are not silently
deleted or repaired in place.

## 14. Failure interpretations

- **Real approximately shuffled:** coworker-specific coordinate information is
  unsupported.
- **Real above shuffled but approximately target-only:** extra generator
  capacity or a generic structural prior may explain the gain.
- **One-step gain disappears in SFT:** local predictive signal does not survive
  optimizer dynamics.
- **Adaptation gain with retention collapse:** the support is destructive, not a
  successful general adaptation.
- **Only `A` helps:** evidence favors adaptive optimizer conditioning, not sparse
  support generation.
- **Only nearest layers matter:** replace the global peer graph with a simpler
  local graph in a new pre-registration.
- **Masks unstable across calibration subsets:** support estimation is noisy;
  report overlap and union.
- **No speed gain:** expected when dense kernels dominate; do not claim
  acceleration.

Negative outcomes are primary scientific results and must remain in raw
artifacts and documentation.

## 15. Phase D, conditionally pre-registered

Phase D is prohibited before a passed gate and a separate reviewed implementation
milestone. If authorized, every seed and arm reloads the same checkpoint,
regenerates a fixed plan from the frozen calibration procedure, reloads the
base, trains with identical data order/steps/batch/LR schedule/warmup/precision/
clipping/loss masking, evaluates frozen benchmarks, and completes a final
bitwise sparse-delta audit.

Required arms are:

A. frozen base;
B. random sparse native;
C. gradient TopK sparse native;
D. target-only sparse native;
E. coordinate-shuffled PCSM;
F. real-peer bootstrap PCSM;
G. real-peer calibrated PCSM;
H. globally parameter-count-matched LoRA;
I. appropriately capacity-matched dense native control;
J. optional full dense upper bound, explicitly unmatched.

Sparse trainable count is `sum(K)`. LoRA count is
`sum(r * (d_in + d_out))` over adapted matrices. Global counts must be matched
as closely as possible, and mismatches disclosed. Optimizer state, gradient
storage, checkpoint delta, peak VRAM, wall-clock, and throughput must be
measured. Sparse native tuning is not presumed faster because backward GEMMs
remain dense.

## 16. Evidence preservation

Preserve, without outcome-driven rewriting:

- source branch, base and final commit identities;
- `configs/*.yaml` and this protocol;
- split, sample, resource, and budget manifests;
- gradient cache manifests/tensors;
- every fixed plan JSON/safetensors pair;
- raw per-cell Phase-C run JSON and decision artifact;
- gate JSON with reasons and paired statistics;
- synthetic raw JSON, CSV, and summary;
- qualification command/output;
- environment/package/CUDA/driver/GPU details;
- failed/invalid artifacts and tracebacks;
- final native delta audits;
- Phase-D STOP or PASS decision.

The authoritative result report is [results/RESULTS.md](results/RESULTS.md).
