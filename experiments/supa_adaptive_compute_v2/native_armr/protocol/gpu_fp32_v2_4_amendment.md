# Track A v2.4: pretrained parallel FFN compute control

This prospective amendment follows mechanical rejection of two independently
initialized controls. Preserve the alpha=1 v2.2 failure at
`evidence/01_architecture_dev/20260905T220307_858493Z_7cc95fc0` and the calibrated
random v2.3 failure at
`evidence/01_architecture_dev/20260906T005825_662929Z_4a8ebe46`, including all prior
registrations, completed runs and partial output. The latter receipt records
`RuntimeError: malformed generation: repetition/script/Unicode guard`; it does
not identify which component of that combined guard triggered. Neither failure
establishes a recurrence capability gain. No threshold is relaxed and no failed
item, negative control, seed, alpha or historical receipt is edited.

The design rationale and alternatives are recorded in
`planning/control_v2_4_review.md`. The single scientific change is replacement of
AM with a frozen pretrained parallel FFN transformation at approximately A3's
dominant linear budget. A0, A1, A3 and A16, all frozen data, scorer, inference
units, primary and secondary hypotheses, statistical thresholds, preservation
and validity gates, resource guards and downstream exclusions remain unchanged.
AM retains its arm name; the new config, protocol, qualification and registration
identities distinguish this operator from earlier random controls. Do not pool
their evidence.

## Fixed operation and copied-weight provenance

All layer indices are zero-based. Apply one residual injection after ordinary
Supa L13. Each of three branches uses one complete canonical MLP and its own
canonical post-attention RMSNorm from layers 12, 13 and 14:

```text
F_l(h) = mlp_l(post_attention_layernorm_l(h))
M(h)   = (F_12(h) + F_13(h)) + F_14(h)
h'     = h + alpha * M(h)
```

Every branch receives the identical post-L13 tensor `h`. Preserve the displayed
sum order, canonical learned norm weights and epsilon, SiLU and complete gate,
up and down matrices, using the installed Qwen3 module operations in FP32.
Coefficients are exactly one for all branches. There is no sequential residual
feedback between branches, attention operation, channel slicing, branch-weight
fit, shared norm replacement or added position. Ordinary L14 still executes
normally after this injection, as do every other ordinary decoder and the stem.

Layers are chosen by the fixed contiguous rule beginning at A3's first layer;
three whole MLPs give the nearest dominant-linear budget. No response, NLL,
architecture accuracy, alignment metric or holdout result selects their identity,
ordering or count. Pretrained features make this a semantic comparator to the
random transformation, but do not guarantee stability or usefulness under this
new input distribution and sum.

The implementation makes wrapper-local frozen copies of all three MLPs and
norms. Never attach these to the canonical backbone state dict. Record separate
source and copy module hashes for each layer, exact equality of those bytes,
composite control-state hash, parameter inventory and zero gradients/trainable
parameters. Backbone, checkpoint and copied state are verified before and after
execution. No optimizer or newly learned weights are introduced.

## Approximate compute budget and interpretation

Canonical geometry is hidden width 1024 and MLP width 3072, with 16 attention
heads, 8 KV heads and head dimension 128. Under the historical multiply-plus-add
estimator, one decoder adds 31,457,280 dominant-linear FLOPs per token; A3 adds
62,914,560. Each complete MLP adds 18,874,368, so three add 56,623,104:
**90% of A3's added dominant-linear work**, not exact parity. Three whole MLPs
are nearer than two (60%) or four (120%); no dummy work fills the difference.

| Inventory | AM v2.4 |
|---|---:|
| Copied matrix coefficients | 28,311,552 |
| Copied learned normalization coefficients | 3,072 |
| Additional resident and active copied coefficients | 28,314,624 |
| New learned coefficients beyond canonical backbone | 0 |
| Added dominant-linear FLOPs per token | 56,623,104 |

The additional FP32 parameter storage is 113,258,496 bytes, before activations
and runtime overhead. Logical coefficient values already exist in the backbone;
copies consume storage but add no newly learned information. Report actual peak
RSS/VRAM and measured latency, full model-level generation work, and these
distinct parameter quantities. Native recurrence A1/A16 still adds 31,457,280
FLOPs per token and A3 still adds 62,914,560; their estimator cannot derive its
budget from AM. The existing shared estimator uses each wrapper's explicit
`extra_matmul_flops_per_token`, so AM's extra full-decoder count is zero.

Attention-score/value products, normalization, instrumentation and gold NLL are
excluded from the dominant-linear generation estimator. A3 also performs extra
attention. An admitted positive A3 comparison supports this particular full
native block over this particular pretrained token-wise alternative, with the
explicit 10% budget difference. It cannot isolate attention versus block order,
establish superiority over all equal-FLOP computations, or prove recurrence is
universally unique. A3 beating A0 without beating AM leaves generic extra
pretrained computation as a viable explanation.

## One calibration rule and artifact-derived final config

Use only the existing frozen Track A calibration role: 96 fit and 32 holdout
original scenarios. This is reused engineering calibration, not a new independent
validation sample. Expose prompts and identity fields only; answers/scorers,
architecture outputs, confirmatory data and Track B inputs never enter fitting.
The separately frozen calibration protocol specifies streamed canonical FP32
capture, full original causal mask, positions and rotary inputs, and exclusion
of the union of tokenizer special and added-vocabulary IDs. No prompt truncation
or independent-token substitution is permitted.

At full-prompt post-L13 state `h`, compute the A3 residual
`L13(L12(h)) - h` and the new unscaled `M(h)` exactly as defined above. For each
scenario average over ordinary tokens:
`mean_feature(delta^2) / max(mean_feature(h^2), 1e-8)`. Use FP64 diagnostic
arithmetic on fixed FP32 forward tensors. Then average scenarios equally and
freeze `alpha = sqrt(mean_fit_A3_energy / mean_fit_M_energy)`. Both energies and
alpha must be finite and positive. Write the immutable fit result before
evaluating holdout residuals. There is no clipping, sign choice, dose grid,
direction fitting, layer/seed search, holdout tuning or answer objective. The
v2.3 scalar cannot be reused for this different transformation. Holdout and
token-category energy summaries are report-only.

Before real calibration, commit the separate calibration registration with
source/config/data/scorer and canonical checkpoint contracts. Successful evidence
belongs exclusively in `evidence/00_learned_control_calibration/`. Its verifier
must reconstruct the fixed formula and scenario coverage, check every branch
source/copy and composite state digest, verify model/copy freeze inventories,
and require the complete report/provenance artifacts to be committed.

Only then run `scripts/freeze_learned_control.py --calibration-artifact <report.json>`.
It exclusively creates `configs/protocol_gpu_v2_4.json`, deriving alpha from the
committed report and binding its canonical repository-relative path and SHA256.
No alpha override exists. An existing config is never replaced. The protocol
version is `native_armr_v2.4_gpu_fp32_pretrained_parallel_ffn_control` and runtime
qualification ID is `native_gpu_fp32_v2_4`. Every final config load verifies the
exact permitted differences, committed artifact, alpha and checkpoint identity.
AM construction checks its actual fresh copied-state hash and branch provenance
against the artifact before any forward. The CPU qualification copy retains the
same control and artifact while changing only its runtime device settings.

## Qualification, architecture rejection and untouched confirmation

Freeze and commit the final native registration before qualification. Run a new
CPU/GPU qualification on all five arms and all eight calibration fixtures using
the existing caller with the v2.4 config. CUDA:0 FP32, eager, no cache, TF32 off,
deterministic settings, four threads, CPU reference and every original numeric,
greedy-identity, repeated-GPU and restoration tolerance remain unchanged. Neither
v2.2 nor v2.3 qualification admits the replacement control.

After the new qualification passes and is committed, execute a fresh full
128-example architecture matrix under v2.4 in operational order A0, AM, A1, A3,
A16. This permits early rejection of AM after its complete A0 baseline while
retaining the required five-arm matrix and original registered arm order in
configuration. Use the same 32 deterministic shards and complete matching A0
before any candidate shard. Do not relabel unchanged arms' old results as v2.4.
Keep every failed attempt, partial artifact and interrupted process receipt.

Only a complete passing five-arm matrix licenses the unchanged 1,024 independent
confirmatory originals and 128 separate diagnostic variants. The primary
hierarchy remains A3-vs-A0 then A3-vs-AM; both require exact McNemar p <= .05 and
positive 95% paired scenario-bootstrap lower bounds. A3-vs-A0 must additionally
show >=2 accuracy points or >=5% relative error reduction; the original pooled
preservation bound, zero extra invalid outputs and all safety conditions remain
mandatory. Family/robustness analyses and secondary Holm comparisons keep their
original roles. Calibration and qualification are engineering evidence only.

If this chosen control fails, retain the failure and stop this amendment. It
does not license an automatic search for another control until one passes or a
weaker guard. Track B decisions remain separate. Routing, 8B escalation,
MLA/MHA/MTP, backbone training, PEFT, distillation and sealed evaluation remain
closed under their original gates.
