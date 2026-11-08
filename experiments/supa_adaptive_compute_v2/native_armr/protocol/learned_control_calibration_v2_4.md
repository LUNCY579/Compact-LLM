# Prospective v2.4 pretrained-control energy calibration

The v2.3 independently initialized control failed the unchanged malformed-output
guard after its fit-only alpha calibration. Its receipts and those of the
original alpha=1 control remain unchanged. This amendment preregisters one new
control for mechanical admission. No architecture accuracy, NLL or failed token
prefix selects its construction. It changes no recurrence hypothesis, primary
comparison hierarchy, scientific threshold, validity guard or downstream gate.
If this control fails, preserve its evidence and stop this amendment; it does
not license another automatic control search.

The control copies complete canonical Supa MLPs and their own learned
post-attention RMSNorms from zero-based layers 12, 13 and 14 into a frozen local
module. Each branch receives the same ordinary post-L13 residual stream `h`:

```text
F_l(h) = mlp_l(post_attention_layernorm_l(h))
M(h) = (F_12(h) + F_13(h)) + F_14(h)
h' = h + alpha * M(h)
```

The displayed FP32 summation order is fixed. Preserve canonical SiLU, matrices,
learned normalization weights and epsilon, using the installed HF modules.
No branch consumes another branch's output. There is no attention operation,
position change, matrix slicing, filler compute, channel duplication, optimizer
or trainable parameter. The ordinary L14 remains in the later full backbone.
These layers follow a fixed contiguous rule beginning at A3's first layer;
three whole FFNs are nearest A3's dominant-linear budget, without layer search.

The three branches use 28,311,552 existing matrix coefficients plus 3,072 learned
norm coefficients. Wrapper-local frozen copies therefore store 28,314,624
coefficients (113,258,496 FP32 bytes), all active in the extra transform, with
zero newly learned coefficients. Added dominant linear work is 56,623,104
multiply-plus-add FLOPs/token, 90% of A3's 62,914,560. This is approximate parity;
A3's attention score/value work is also outside that estimator. Later evaluation
must report this difference, actual latency and memory rather than treating AM
as exactly two decoder layers.

Only the existing frozen Track A calibration role enters calibration: 96 fit and
32 held-out original scenarios. These are reused engineering prompts, not fresh
independent validation. Reuse the existing prompt-only projection unchanged;
labels sharing the integrity-hashed container are immediately discarded. No
answer, scorer, NLL, generation, architecture, confirmation, diagnostic variant,
Track B or sealed input enters the calculation.

CPU FP32, eager attention, deterministic seed 17, four threads and no cache are
the reference. Stream actual canonical decoder layers 0 through 13 over complete
prompts, capped at 256 tokens with no truncation. No full LM or output head is
loaded. Replay extra L12 then L13 on `h`, preserving original positions, full
causal mask and rotary inputs, to form `delta_A3 = L13(L12(h)) - h`.
Load L14 only to obtain its copied MLP and normalization. Its decoder is not run.

For each scenario, exclude the union of tokenizer special and added-vocabulary
IDs. Average ordinary-token values of
`mean_feature(delta^2) / max(mean_feature(h^2), 1e-8)`, computing diagnostics in
FP64 from fixed FP32 inference tensors. Average scenarios equally, then freeze
exactly `alpha = sqrt(mean_fit_A3_energy / mean_fit_M_energy)`. Both energies and
alpha must be finite positive. Write immutable `fit_scale.json` after fit and
before measuring holdout residuals. No clipping, grid, alternate seed, sign,
branch-weight fitting, directional objective or fallback is allowed. Report
held-out energy and all/ordinary/special/final-token strata without selecting
alpha, layers or any setting from them. Do not reuse v2.3's numeric scalar.

Require at least 1 GiB available host memory for this streamed calibration and
keep process RSS below 6 GiB; each prompt operation has a 900-second bound and
the stage a 172800-second bound. These bounds apply only to this engineering
calibration, not the unchanged full-model admission contract. Respect the shared
single-model lease. Nonfinite values, elapsed or memory violations, trainable
parameters, gradients or mutation fail with an immutable failure receipt.

Use separate `00_learned_control_calibration_registration` and
`00_learned_control_calibration` evidence roots. Registration binds source,
config, legacy streamed helper, test, frozen input hashes and all scenario IDs;
commit it and source before execution. Every run binds branch, HEAD, runtime,
checkpoint/tokenizer/config and installed HF implementation, full-prefix/replay
state and buffers, each source/copy MLP and norm, composite control state,
parameter accounting, all case identities and tensor hashes. Check state,
buffers and zero gradients before/after. Failed and partial evidence is retained.
The artifact verifier reconstructs scenario aggregates and the fitted scalar,
checks source-copy equality and full provenance, and requires complete committed
artifacts for downstream use. Receipt checking is not an independent numerical
model rerun; raw tensor hashes alone cannot establish such a rerun.

Only successful committed calibration can supply alpha and control identity to
a separately frozen v2.4 generation config. Requalify all five arms on CPU/GPU
with the existing thresholds, then execute the complete new 128-case five-arm
architecture matrix. Earlier runs are not relabeled or pooled. Only a complete
passing matrix opens the untouched 1,024 primary confirmation cases and 128
linked diagnostics. Qualification and calibration are engineering evidence,
not efficacy. Track B decisions and all routing/attention/MTP/8B/distillation
gates remain independent and unchanged.
