# Prospective control-design review after AM v2.3 mechanical failure

Review date: 2026-09-06 UTC. This is a design recommendation, not a
preregistration, execution receipt, or capability result. No model was run for
this review. No architecture accuracy, answer NLL, generated response, failed
token prefix, confirmation row, or Track B data was inspected. The only new file
written by this review is this planning document, outside `source_files`' frozen
Native source inventory.

## Verified reason for reconsidering the control

The v2.3 failure manifest at
`native_armr/evidence/01_architecture_dev/20260906T005825_662929Z_4a8ebe46/run_manifest.json`
records AM, architecture shard 20, status FAILED, one completed example in that
attempt, and `RuntimeError: malformed generation: repetition/script/Unicode
guard`. The receipt does not establish which member of that combined guard
triggered. The random control's alpha was fixed at 1.091369554461779 by the
prospective fit-only energy formula. A runtime qualification pass does not
override this architecture failure.

Keep both the alpha=1 and calibrated-random experiments, their failures, and all
partial evidence intact. The evidence supports rejection of these specific
control configurations for mechanical admission. It does not establish that
native recurrence improves capability, or that every random transformation is
destructive. Do not repair this result by reducing alpha, replacing the random
seed, suppressing the failed item, changing the generation guard, or deleting
the matched control comparison.

## One recommended replacement

Preregister a **pretrained parallel FFN control** using three complete canonical
Supa MLPs and their corresponding post-attention RMSNorms from zero-based layers
12, 13, and 14. Apply one residual injection after ordinary L13. Let `h` be that
ordinary post-L13 residual stream, and define:

```text
N_l(h) = canonical post_attention_layernorm_l(h)
F_l(h) = canonical mlp_l(N_l(h))
M(h)   = (F_12(h) + F_13(h)) + F_14(h)
h'     = h + alpha * M(h)
```

All three branches receive the same `h`; a branch does not consume another
branch's output. Preserve the displayed summation order as part of the FP32
contract. Every MLP retains its complete gate, up and down matrices, canonical
SiLU, and learned normalization weights and epsilon. Use the actual installed
Qwen3 module operations. Keep all weights frozen, with no optimizer or gradients.
Do not add an attention operation or change positions, masks, stem layers, or
the ordinary L14 application later in the backbone.

The layer rule is fixed contiguous canonical layers beginning at the native
block's first layer, with the number of whole FFNs determined only by the nearest
dominant-matmul budget. It is not a search for a favorable layer. The first two
branches reuse A3's existing FFNs, giving the control relevant pretrained
features; the immediately following canonical FFN fills most of the remaining
compute budget without choosing the separately hypothesized L16. No behavior,
activation similarity, alignment result, or architecture score chooses this
ordering. Three equal branch coefficients and one global scalar are fixed;
there is no branch-weight fitting.

This is a stronger semantic comparator than independent Gaussian matrices
because its constituent features were learned in the canonical backbone. That
is a design rationale, not evidence that this new combination will be stable or
useful. Post-L13 inputs are not exactly the training inputs of all three FFNs,
and the sum can have harmful directions even with matched energy. Qualification
and the unchanged full architecture gate must determine its admissibility.

## Why not manufacture an exact wide MLP

The inspected canonical configuration has `hidden_size=1024`,
`intermediate_size=3072`, `num_attention_heads=16`, `head_dim=128`, and
`num_key_value_heads=8`. The proposed premise of a 4096-wide canonical MLP is
incorrect. The verified `config.json` SHA256 is
`89c4ae7398449895d044750c858614584b07db8d67130074920ef524c6eb5705`.
The installed Qwen3 implementation SHA256 is
`fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab`.

Selecting deterministic partial channels introduces an additional unvalidated
change to a learned FFN. An arbitrary channel slice has no guarantee of retaining
its useful representation, regardless of deterministic selection. Duplicating
channels or computing canceled contributions to hit a FLOP total can create
redundant work rather than a stronger control. Neither is necessary because the
master goal explicitly permits approximate dominant-matmul matching.

Keeping each branch's normalization explicit also avoids needless tensor
surgery. Folding learned gamma into input matrix columns is algebraically valid
for an otherwise identical RMS-normalized branch, but changes FP32 operation
order and can create different rounded tensors. A single shared learned gamma
would not preserve three different canonical norms. Whole explicit branches
make parameter provenance and correctness easier to inspect.

Sequential FFN residual passes are a plausible different experiment, but add
their own bounded native feedforward recurrence. The parallel single-injection
control makes the present distinction clearer: complete L12-L13 block reuse
versus a generic pretrained token-wise transformation at a comparable budget.
Do not run both designs and select whichever behaves better in this amendment.

## Exact scope of compute matching

Use the existing multiply-plus-add convention. With `d=1024`, `q=2048`,
`kv=1024`, and `f=3072`, a decoder layer's dominant matrix work is
`2 * (2*d*q + 2*d*kv + 3*d*f) = 31,457,280` FLOPs per token. A3 adds two such
layers, totaling 62,914,560. One intact MLP costs `6*d*f = 18,874,368`.

| Quantity | Proposed parallel control | A3 added operations |
|---|---:|---:|
| FFN evaluations | 3 | 2 |
| Added full decoder evaluations | 0 | 2 |
| Dominant linear FLOPs per token | 56,623,104 | 62,914,560 |
| Ratio of control to A3 linear FLOPs | 0.90 | 1.00 |
| Unique control matrix coefficients used | 28,311,552 | Count from the native execution inventory |
| Learned control RMSNorm coefficients used | 3,072 | Count from the native execution inventory |

Three whole FFNs are nearer to A3's budget than two (0.60) or four (1.20).
Accept the 0.90 ratio prospectively; do not add filler work. It is not exact
FLOP parity. A3 also performs attention-score/value products excluded from this
legacy dominant-linear estimator, and actual latency may differ substantially.
Report measured latency, memory, full model-level estimated work, and the
explicit 10% dominant-linear difference. Do not use the existing AM shortcut
that assumes its work equals two native layers: the new operator's estimator
must use its actual 56,623,104 added linear FLOPs per token.

Parameter accounting must distinguish logical unique weights, active weights,
and resident storage. Reusing module references adds zero unique learned
coefficients beyond the backbone; 28,314,624 existing coefficients participate in
the extra transformation. An implementation that makes frozen wrapper-local
copies stores that many additional coefficients but adds no newly learned
information. Record the implementation's actual choice and byte inventory; do
not call copied pretrained weights independently initialized or trainable.
Never attach a new module into the canonical backbone or mutate its state dict.

## Single prospective calibration rule

Use only Track A's existing frozen calibration role, with 96 fit and 32 held-out
original scenarios. Reusing engineering calibration for a newly preregistered
operator is permitted, but identify it as reused calibration rather than a new
independent validation sample. Prompt-only projection must keep answer and
scoring fields outside the computation. No Track B or efficacy input enters.

Capture full-prompt ordinary post-L13 states through actual canonical decoder
layers. Define `delta_A3 = L13(L12(h)) - h`, preserving the original full causal
mask, positions, and rotary inputs. Define the new unscaled residual as exactly
`M(h)` above. Exclude the union of tokenizer special IDs and added-vocabulary
IDs. For each scenario and residual, average ordinary-token values of
`mean_feature(delta^2) / max(mean_feature(h^2), 1e-8)`, then average scenarios
equally. Calculate diagnostic energies in FP64 from fixed FP32 forward tensors.

Freeze exactly `alpha = sqrt(mean_fit_A3_energy / mean_fit_M_energy)`, requiring
finite positive energies and alpha. Write the immutable fitted scalar before
evaluating holdout residuals. There is no clipping, dose grid, sign selection,
seed selection, directional fit to A3, or answer objective. Equalizing energy
does not equalize useful information. Holdout energy and token-category
diagnostics are report-only and cannot select a layer, alpha, normalization or
replacement operator. Do not reuse v2.3's numeric alpha: it belongs to a different
transformation. Hash each branch's actual canonical matrices and normalization,
the composite state, all input identities, formula, and source/config bindings.

## Required v2.4 boundary before any model execution

This document itself grants no execution admission. Before running the proposed
control or its calibration, implement and independently test it, write a new
v2.4 amendment and estimator configuration, freeze the source/data/scorer
contracts, and commit a separate calibration preregistration. That amendment
must name the random-control failure as the reason for a new pretrained control,
record the single chosen construction above, and explicitly preserve the
original primary A3-vs-A0 then A3-vs-AM hierarchy, all thresholds, diagnostics,
validity rules, resource contracts, and closed downstream gates.

After successful committed calibration, derive the final v2.4 config exclusively
from its hash-bound artifact. Keep old configs and evidence unchanged. Requalify
all five arms against CPU using the same numeric and greedy-identity tolerances;
the old control's qualification cannot certify this one. Run the complete fresh
128-case five-arm architecture matrix under the new registration, with complete
A0 followed by AM permitted for early mechanical rejection. Previous A0 byte
identity or unchanged A1/A3/A16 mathematics is not permission to relabel their
old runs as v2.4 evidence. Do not pool failed or previous-version results into
the new paired matrix.

Only a complete passing architecture matrix licenses the untouched 1,024-case
confirmation and separate 128 diagnostics. If this control fails, preserve that
failure and stop this amendment. A failure does not license an automatic search
for another control until one passes, or a weaker gate. Calibration and
qualification remain engineering evidence; neither is efficacy evidence.

## Interpretation that the resulting experiment can support

A3 beating A0 and this admitted control with the original paired thresholds
would support the particular full native block operator over this particular
pretrained FFN alternative, subject to the stated 10% dominant-linear budget
difference and extra attention cost. It would not establish superiority over
all equal-FLOP algorithms, isolate attention from layer order or other block
components, or show that recurrence in every form is uniquely useful. A3
beating A0 but not this control would leave generic extra pretrained computation
as a viable explanation. Either result is useful; no architecture is required
to win.

The sole next Track A experiment recommended here is this preregistered v2.4
pretrained parallel FFN-control campaign, beginning with its fit-only calibration
and mandatory qualification. It changes no Track B decision or experiment.
