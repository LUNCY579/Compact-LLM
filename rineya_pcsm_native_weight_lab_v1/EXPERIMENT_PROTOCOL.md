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
  complete; a 54-plan schema-v1 partial qualification artifact is preserved but
  non-gating, schema-v2 hardening is **RUNNING**, and confirmatory outcomes plus
  the advancement gate remain **PENDING**
- Phase D: **STOP unless a complete, valid gate artifact passes**

The checked-in synthetic results predate real-model outcome inspection and are
mathematical feasibility evidence only. The `SEALED` split remains unread during
selection and Phase C.

### 1.1 Schema-v2 root of trust

Before any new qualification plan or outcome, generate, review, and commit
`results/phase_c/PREREGISTRATION.json`. It is the immutable root of trust—not a
later run summary. Its canonical self-hash excludes only the
`preregistration_sha256` field, while its file SHA-256 binds exact serialized
bytes. Planner and runner must verify both.

The frozen input chain is:

| Input | File SHA-256 | Internal self-seal |
|---|---|---|
| Nimbus model weights | `7b66b8d0ef06cf7f71bc9001b0c9a330d5f8a0eec1d11eb173ece70ca66ef49c` | n/a |
| Nimbus config | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` | n/a |
| Nimbus tokenizer | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` | n/a |
| Nimbus tokenizer config | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` | n/a |
| Nimbus vocabulary | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` | n/a |
| Nimbus merges | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` | n/a |
| aggregate tokenizer-artifact signature | `90a2903e36317f2a873560235218be86367fa0ed4209f53096b6d59325fed333` | n/a |
| `budget_manifest_v4.json` | `bc31ae08fcf8a1235d83887afb25aaaead409d56329b92d690a4ee8e8d2b6dc7` | `88acd647d30ba1ea23b74b9a7f3079414c7cc6cb3991bd2b073681ab9ff937a9` |
| `sample_manifest.json` | `83a22898dcb0efd5eed15d0811ac1956e689ff16f55e6f58f53d644e4fcbe42e` | `53242173b46e4fe950e354ec1751b7bd8f6ace222630d2c4b512e1065b6e23cc` |
| `split_manifest.json` | `980c5013cadb8770a02d80406d584fe5e1d40c3b01003ef57bb230b28e217afe` | `f15666248ce47dccad749199b47bd0f4fcf65bec7bc1f44f949929cc6dc78a2a` |
| `gradient_cache_summary.json` | `6db534112a84c5bc9310bfda3cd7f9020494ed555707256d025a3d0eab3298e5` | `38ed0abbb045fffd06f52c09976db044989f5dcf02cd2c289300788f32584d28` |
| `meta_train.json` | `938697a818dc702419930c64356af455e1a4561bcdccb69963b720f84b5a9382` | `46813781f5e85262ab7eb8037952768e807c4dd04d8bfcb2c8f93564769b4cfd` |
| `meta_train.safetensors` | `9758a7f131713b2cbb5cd080eb46f6c2c7c4a962780aff4fdda12eb8befa0e06` | n/a |
| `mask_cal.json` | `0df192568c6852407975ce9317db82d3afd33f95a53ecedbdaaf1f4305584b75` | `ed543398ad67f05547bd7fab94e61c26765738d16db5447e696bee50d92f1439` |
| `mask_cal.safetensors` | `7f6e502f5d20fa641c98976e5e479ad0bd60813a76942c8483eedb6eaf4dd686` | n/a |

The root additionally freezes the source-file table and aggregate source-tree
hash, exact calibration row IDs, target identity, exact-K table, generator,
optimizer, gate, access policy, the 40,960-token ceiling, exact encoded
`input_ids`/`labels` hashes for the one-row pilot and four-row qualification,
200 ordered run cells, 175 ordered plan cells, and deterministic layer-depth
permutations. Those component hashes are recomputed during generation. The
official preregistration file/self identities are authoritative only after the
committed artifact is generated; no smoke artifact hash is promoted in their
place.

Final preregistration identities:

- file SHA-256: **`PENDING_AFTER_CODE_AND_DOCS_COMMIT`**;
- canonical `preregistration_sha256` self-seal:
  **`PENDING_AFTER_CODE_AND_DOCS_COMMIT`**;
- frozen 33-file source-tree SHA-256:
  `e899999d2f33dfc327b493f25bb24bcd594757bfc01c3eb6f46a5c4db5c94b9c`;
- preregistration access-log SHA-256:
  `4578ca7b1e3792a6d446a655d1c220c6e2fd267c879fde1a88ff9f433bf0ced5`;
- encoded pilot signature:
  `aebb2d04037926dedf21f318dbb9ae9746f09758754f71858eb9583c8692449c`;
- encoded qualification signature:
  `edaf37f53e648c2abbda6448ed5b679371a249ded51ce88eb9acdbc28d5d448a`.

Only the two explicitly marked preregistration hashes depend on generating the
official timestamped artifact after this code/docs commit. A smoke artifact's
hash is deliberately not substituted. The source, access, tokenizer, encoded-
data, input-manifest, cache, grid, and mapping hashes above are already frozen
by their underlying bytes.

The source tree covers every package module and every qualification authority:

| Frozen source | SHA-256 |
|---|---|
| `scripts/audit_phase_c_gate.py` | `f3cb3bddc4637a2fda2e5d872289632c726d1de8cd27eb45a164b3fab1164aae` |
| `scripts/build_phase_c_plans.py` | `47d0f375deec83a0f16a5bce9a9a393da164c37bb99583bff4f41afbe31a2f20` |
| `scripts/freeze_phase_c_preregistration.py` | `cfe6e0b0ff092f7c43790add31f273983621bb8cb026e07c4aa1491e4c843a81` |
| `scripts/run_phase_c.py` | `59fd06c108798a51ec1fe80ff8c2abb36b2470b69feb828a6fbadee5d5e0b2e3` |
| `scripts/train_phase_d.py` | `240687e0420ad482a9a0493ebcc7bec8182d400e0672be39da7deae77dfe0915` |
| `src/rineya_pcsm/__init__.py` | `7eb70648f516bbd795806a8cd65ce71ceb94befe977c8120d2a6dd763bec903a` |
| `src/rineya_pcsm/application.py` | `a7d3187a8b96897eeabb3cf6fa808b2a774807e099307d292055f0aa6c750505` |
| `src/rineya_pcsm/architecture.py` | `cabc2d0c99e077447bfb50161db5f446b07eb48ec5a71f62b7703576526f2345` |
| `src/rineya_pcsm/attention.py` | `a29597a81a4b24bf0eaa537c581aea289eebe5273b50c4b9672601ddf9104c84` |
| `src/rineya_pcsm/calibration.py` | `ae622fac2e77101ae638028c123fac200f962bbdb7dc221ba2801dc237a1fe84` |
| `src/rineya_pcsm/config.py` | `0ff11091460f899e9cc67ab2a27abed86336f1093c0dfc94fd319d98ee9a2cf1` |
| `src/rineya_pcsm/controls.py` | `0629e68fb87a4d5e8630bb54a9fb96bc6d499e0d4e65431a9a179afbc6fe8967` |
| `src/rineya_pcsm/data.py` | `6e1c17175ba85f91c93e604c648f1f82fbbd8305036ce83b4dea83c9d5e807e5` |
| `src/rineya_pcsm/delta_audit.py` | `a9bfad4c357667b6a77283ea6e86e0b9fb88a077516d1d0d8e65d0e829875666` |
| `src/rineya_pcsm/descriptors.py` | `d6293b4d7edea58fe4087ced2d6a7233e71235fdd0b1c90cf940ed5666a63fe1` |
| `src/rineya_pcsm/gate.py` | `5578f7f1ac6afcdde793a1eadd99553015053f9127be7beb9bc69927d805af88` |
| `src/rineya_pcsm/generator.py` | `cbacb7d6d216a36769030520505f101a21a2ed277585ea75ef4a0f850d1b8fd6` |
| `src/rineya_pcsm/hashing.py` | `1487f0060ca4cc5e12d50f91549e5592363ed5c05c840191d4433783b37de280` |
| `src/rineya_pcsm/integration.py` | `bccd97afd07c42591e7c50bb5b632ea19ac0139cdd1794e8290cda95adea9d53` |
| `src/rineya_pcsm/losses.py` | `3a92fd8719350f38cdc87e9bb790120d49699a5061ef9ecdfe63f657a8becf31` |
| `src/rineya_pcsm/manifest.py` | `0633f6244a1d5e6c3f5ba7d331e30a5509b989e6e1917f3a0444c2e248bac4dd` |
| `src/rineya_pcsm/optimizer.py` | `4f0e2f0b46af1d30cab64fd534fca54ece56fa758467034489624ede5152328b` |
| `src/rineya_pcsm/persistence.py` | `1f80df5c84227d4583d5791715fcbdf4b70a4645053c9cecac681ff6b4c64e4c` |
| `src/rineya_pcsm/phase_c.py` | `79f043401aa2d0a8795e7dab7453ed34d770ac4d8093310017661f23665072cd` |
| `src/rineya_pcsm/phase_c_runtime.py` | `1ebb6a9a3656994086a540d4f9853420f021f57d6af803574db7b5fae9c494e5` |
| `src/rineya_pcsm/plan.py` | `be5ba565ef5933dfdd1674731d934a11d78ff82ece3d93cd500552dff401062e` |
| `src/rineya_pcsm/property_maps.py` | `ff1b560026121ae4d8c8f6aef1608de94e1c16329b00a98e8a1a23620ba472b3` |
| `src/rineya_pcsm/randomness.py` | `c16ebe72daab8fb7d0682e4de772085efc5eb6110de409502c18fcc70c884db9` |
| `src/rineya_pcsm/real_plans.py` | `0623fb7f30ba452e97ab1f7d3f638990ad229e4a4d7758a44d40c006af5025a3` |
| `src/rineya_pcsm/statistics.py` | `df0b2c6f54e4a4323ec90e855c96ba8d2103dfecf5979052667c80a8502211f5` |
| `src/rineya_pcsm/synthetic.py` | `4ae36020a72893482ae15f7f34fef9536f576261a8bbf22e30c392a202826e9b` |
| `src/rineya_pcsm/topk.py` | `9b84e65501d7a2f8dd92eab67b57a194e7554adfc28187445dc20816c6275ef2` |
| `src/rineya_pcsm/version.py` | `543b01b30cdf53b80db019086b1c55eadb22d22c659acdd68cdec76206496768` |

Schema-v2 planner invocation requires `--preregistration`. Schema-v2 runner
invocation requires `--preregistration`, `--split-manifest`, and
`--gradient-summary`; omitting any is an operational failure. The 54-record
`qualification_plans_legacy_incomplete.json` is schema 1.0 and has no
preregistration, access log, or complete cell grid, so it cannot be resumed into
or submitted to the schema-v2 gate.

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

The 200-cell grid contains 25 logical frozen baselines and 175 sparse cells.
Because `no_update_frozen` has no support plan, the preregistered plan grid is
exactly `5 x 5 x 7 = 175`. The run grid remains exactly
`5 x 5 x 8 = 200`; no arm is removed for compute convenience.

### 6.1 Frozen nonidentity layer-depth mappings

For the layer-shuffle arm, the root stores a deterministic map from each of the
27 eligible peer-layer identities to a permuted depth. Layer 14 is the target
and is absent. Each map is a bijection over the same 27 depths and is not the
identity map. A fixed point inside an otherwise nonidentity permutation is
allowed and counted below.

| Seed | Mapping SHA-256 | Fixed points | Compact `peer -> assigned depth` mapping |
|---:|---|---:|---|
| 11 | `4f87110fdf94b3a2402d61cd8bd276150b15d0a07fd56fbfd44c93e5bc6c8940` | 0 | `0->8, 1->4, 2->24, 3->0, 4->11, 5->12, 6->19, 7->23, 8->21, 9->17, 10->3, 11->10, 12->9, 13->6, 15->26, 16->15, 17->1, 18->16, 19->22, 20->18, 21->25, 22->27, 23->13, 24->7, 25->20, 26->2, 27->5` |
| 17 | `904c676fe64fab99b99c16b642101e3c73cb01011a60c6d5693034216309df07` | 2 | `0->10, 1->25, 2->2, 3->6, 4->1, 5->26, 6->16, 7->5, 8->7, 9->18, 10->22, 11->11, 12->27, 13->12, 15->21, 16->20, 17->23, 18->15, 19->3, 20->8, 21->9, 22->24, 23->19, 24->17, 25->4, 26->0, 27->13` |
| 23 | `878b971401241b14f6f055d828128790c91b96ef44023af8e46b3305d377bd43` | 1 | `0->13, 1->3, 2->17, 3->0, 4->22, 5->1, 6->10, 7->21, 8->27, 9->24, 10->23, 11->20, 12->7, 13->15, 15->25, 16->16, 17->8, 18->19, 19->4, 20->2, 21->26, 22->5, 23->6, 24->18, 25->9, 26->12, 27->11` |
| 29 | `697275db3c5c15870de4e8b4e57e9df479fc04f3401246cdbd3437a2bf9ce2fc` | 2 | `0->21, 1->15, 2->24, 3->1, 4->26, 5->17, 6->25, 7->16, 8->5, 9->19, 10->8, 11->3, 12->2, 13->11, 15->9, 16->12, 17->23, 18->10, 19->0, 20->4, 21->6, 22->22, 23->13, 24->7, 25->20, 26->18, 27->27` |
| 31 | `e7268b09b50effaf825181e912b7b0c175a5d4c888850d85a4fea04736043548` | 0 | `0->5, 1->20, 2->1, 3->19, 4->15, 5->9, 6->10, 7->22, 8->21, 9->8, 10->11, 11->24, 12->25, 13->2, 15->27, 16->12, 17->0, 18->23, 19->6, 20->16, 21->4, 22->3, 23->18, 24->7, 25->26, 26->17, 27->13` |

The plan builder must reproduce the exact preregistered map, mapping hash,
`nonidentity: true`, and `bijection: true` for every layer-shuffled cell. A
different mapping invalidates the plan before outcome evaluation.

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

### 8.1 Required schema-v2 signatures and access evidence

Every generated plan record must persist:

- `generator_architecture_sha256` over the exact generator configuration;
- `attention_architecture_sha256` over the exact attention configuration;
- generator and attention parameter counts plus their summed capacity;
- calibration step, backward, and trace counts;
- exact `META_TRAIN` and `MASK_CAL` manifest/tensor hashes and row IDs;
- selected-scale distribution, plan/support hashes, and a record self-seal.

Every run persists an optimizer signature over `kind`, `betas`, `epsilon`,
`weight_decay`, the LR grid, LR selection split, and `step_count=1`. It persists
a data signature over the complete ordered frozen `ONE_STEP_DEV` record list,
plus independent LR-grid, preregistration, plan-result, model, config,
tokenizer, split, sample, gradient-summary, and gradient-cache hashes.

Preregistration access is exactly one policy-authorized, non-sealed validation
read whose purpose is `freeze ONE_STEP_DEV token IDs and labels before
qualification`; its log is self-hashed. Planner access is exactly two logged,
non-sealed training-cache events:
`META_TRAIN` and `MASK_CAL`. Runner access is a policy-authorized validation
read for held-out loss plus non-sealed training-artifact reads for plans and
gradient provenance. Both logs are canonically hashed. `sealed_test_read` is
derived from the log rather than asserted informally and must remain false.

Frozen/common signature values are:

| Signature | SHA-256 / count |
|---|---|
| ordered 200-cell run grid | `057b64e9669e0cb4810b99f375110cd4fa4fd033269a9561c45c1b53186e1969` |
| ordered 175-cell plan grid | `6cd224c2fb11e24f3419daf3ceb09e1294df9082a9fc70a052e8fd3929cfc233` |
| optimizer signature | `cc69b2a8896510faf0a8fa215fbf4dd2c11af2feb975f641e82f504b0434bcab` |
| LR-grid signature | `94beba62ccbd12a7d24501f4ae0bf930b762da6b5b990117da8cca5a7623c487` |
| ordered four-row dev-data signature | `382f582de0ee3b13ab262a5830ad41705884c59553160dc5c6985ac28c56886c` |
| frozen generator-budget signature | `edfde7db644fce75f032dc7689d563ff18f8e9fc9a2d235d0d319d4ad81ccb61` |
| PCSM generator architecture / parameters | `2f1e428bbb872b30c30fa64ab5fbdb30595b12110e2da98ed29619018f6ac99a` / 5,387 |
| peer-attention architecture / parameters | `33b3560f9ad8e7c54621fec5e2f23fd662cad564700922156b1a4ee46028bc07` / 1,185 |
| peer-conditioned total capacity | 6,572 parameters |
| planner access log | `ac1914811bda49cf67acaca2b7f16af574d810cc8d08f4f85e6097f202fc6b33` |

The dev-data signature preserves the exact preregistered/budget row sequence;
it does not sort after the fact. Target-only and classical controls have their
declared smaller/zero attention capacity; the 6,572-parameter signature is the
matched peer-conditioned capacity used by real, coordinate-shuffled,
layer-shuffled, and bootstrap peer arms.

## 9. One-step execution procedure

For each arm/seed/budget cell:

1. Verify the committed preregistration self/file hashes and its complete source
   and input hash chain.
2. Verify base model, config, tokenizer, split, sample, gradient-summary, and
   gradient-cache identities.
3. Load same-role native weights and cached FP32 `META_TRAIN` / `MASK_CAL`
   gradients.
4. Calibrate the declared generator mode without any native optimizer step.
5. Freeze exact K once; persist mask, full `A`, attention, feature weights,
   metadata, and content hashes.
6. Discard selection state.
7. Load a fresh byte-identified base model for every sparse cell.
8. Verify that the plan cell matches seed, budget, arm, target, mode, K, and
   calibration compute.
9. Select LR using `META_TRAIN` only.
10. Bind the exact plan, apply one `MASK_CAL` native update, and clear gradients.
11. Bitwise-audit the target and hash every unplanned parameter.
12. Evaluate actual `ONE_STEP_DEV` assistant-token NLL.
13. Persist the cell immediately with plan/support hashes, delta audit, compute,
    memory, and environment data.

No sparse arm may reuse the mutated model from another arm. The frozen baseline
is the sole exception: it is evaluated once because its inputs are independent
of seed/budget, then cloned into all 25 logical baseline cells with an identical
field set and recomputed record self-seal. Existing baseline evidence is reused
only when its semantic hashes agree; resource/timing variation cannot mask a
loss mismatch. This reduces redundant compute without changing the 200-cell
statistical grid.

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

1. the committed schema-v2 preregistration is valid, the complete 175-cell plan
   set exists, and the complete 200-cell run has exactly one record per frozen
   seed/budget/arm cell;
2. real peers beat coordinate-shuffled peers in at least 4/5 seed aggregates;
3. the paired 95% interval for
   `coordinate_shuffled_loss - real_peer_loss` is strictly above zero;
4. real-peer superiority also survives layer-identity permutation (at least 4/5
   wins and interval strictly above zero);
5. real/shuffled arms have identical K, architecture, data, optimizer, LR search,
   generator capacity, and meta-training compute;
6. every exact native-delta audit passes and every unplanned parameter hash is
   unchanged;
7. preregistration, planner, runner, and gate access logs prove that sealed test
   data was never read and `sealed_test_read` is exactly false;
8. all required run/plan cells, signatures, source hashes, layer-depth mappings,
   and plan identities match the committed root exactly once;
9. no material retention or ordinary-chat regression is concealed;
10. the result is not invalidated by any rule below.

If any condition fails or evidence is missing, the verdict is
`STOP_NO_PHASE_D`. A pilot result, even if positive, cannot advance.

The schema-v2 gate does not trust asserted booleans. It finds the last commit
that changed `PREREGISTRATION.json`, proves that commit is an ancestor of the
current `HEAD`, and proves the working file is byte-identical to that committed
blob. It then revalidates all self/file hashes and access logs, requires exactly
175 plan and 200 run records,
reloads every persisted native plan, recomputes token-weighted held-out NLL from
per-example evidence, checks LR selection, capacity, layer mappings, and every
delta invariant, and only then computes the pre-specified statistics. A STOP
artifact is still self-sealed evidence. Phase D re-runs the full audit and
compares authorization/evidence/root identities; possession of an old PASS JSON
alone is insufficient.

## 13. Invalidation rules

A run or cell is invalid if any of the following occurs:

- test or sealed data influences selection, tuning, or target/budget choice;
- a frozen content, token, model, config, tokenizer, cache, plan, support, or
  tensor hash mismatches;
- preregistration is absent, uncommitted, not schema 2.0, has an invalid
  self-seal, or differs from planner/runner provenance;
- source-tree, generator, attention, optimizer, LR-grid, dev-data, or access-log
  signatures differ from their frozen identities;
- a layer-depth shuffle is identity, non-bijective, or differs from the
  preregistered seed mapping;
- silent truncation or zero assistant-target tokens occurs;
- self advice or incompatible peers enter a production arm;
- K differs from the frozen budget or sparse arms have unequal K;
- a support changes after plan generation or union exceeds declared static K;
- any bit changes outside support;
- an unplanned parameter fingerprint changes;
- momentum, weight decay, or resumed state acts outside support;
- non-finite scores/scales/gradients or non-positive scale values occur;
- an arm is missing, duplicated, mislabeled, or bound to another seed/budget;
- run count is not exactly 200 or plan count is not exactly 175;
- a schema-v1/legacy partial artifact is supplied to the schema-v2 gate;
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
- the committed `PREREGISTRATION.json`, its file/self hashes, source-file table,
  and aggregate source-tree hash;
- split, sample, resource, and budget manifests;
- gradient cache manifests/tensors;
- planner/runner/gate access logs and generator/optimizer/data signatures;
- `qualification_plans_legacy_incomplete.json` preserved verbatim and labeled
  historical/non-gating;
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
