# F-WANDA / Fisher-aware Rineya delta retention

## Scope and question

This is a preregistered, mechanics-first experiment on fine-tuning/task deltas.
It does not prune the dense base model and it does not claim that F-WANDA
improves Rineya until a later, independently drawn efficacy run satisfies the
registered comparison rule.

The question is whether empirical Fisher information identifies output rows of
a Rineya delta that are retention-critical, so that a fixed global delta keep
budget preserves retention better than matched historical magnitude and
activation-weighted controls.

## Primary-method verification

The primary source is Himanshu Mishra, “F-WANDA: Fisher-Reweighted
Post-Training Pruning for Sustainable Deployment of Large Language Models,”
arXiv:2608.00481, https://arxiv.org/abs/2608.00481. The paper’s Algorithm 1
and Appendix D Algorithm 2 were checked before implementation.

For a linear layer with input activations `X`, output-preactivation gradients
`G`, and weight matrix `W`, the paper specifies:

* `||X_j||_2 = sqrt(sum_t X[t,j]^2)`;
* `omega_i = (1/N) sum_t G[t,i]^2`, the empirical Fisher scalar for row `i`;
* `S_ij = |W_ij| * ||X_j||_2`, the WANDA score;
* `v_i = sqrt(omega_i)` and a water-filled row budget `k_i` subject to
  `sum_i k_i = K`, `k_min <= k_i <= d_in`;
* within each row, retain the top `k_i` entries by `S_ij`;
* use largest-remainder rounding after water-filling. The paper uses
  `K = ceil((1-s) d_out d_in)`, `k_min=1`, and clamps Fisher at `1e-8` in
  its LLaMA-2 setup.

The implementation therefore never computes `Fisher_i * score_ij`. A positive
row scalar would leave that row’s ranking unchanged. Fisher only changes the
number of entries retained per row.

## Rineya adaptation and controls

Rineya’s historical Titan-Sparse arm uses the fine-tuning delta
`delta = trained_fp32 - base`, keeps `floor(k * numel(delta))` entries per
trained tensor by `|delta|`, and resolves equal magnitudes by ascending flat
index. The comparable prior condition is `k=0.25`; this experiment freezes it
as the single initial operating point.

The paper’s global `ceil` rule is adapted here to the historical Rineya
`floor` rule so all three arms have exactly the same per-layer and aggregate
delta keep count. The allocation mechanism, row bounds, Fisher floor, and
largest-remainder tie rule remain explicit and deterministic. `min_keep=1` is
the registered model setting when the layer budget makes it feasible; the
mechanics library also supports `min_keep=0` for small synthetic qualification
fixtures where a positive floor would be infeasible.

* **F0 — historical magnitude delta pruning.** Per-tensor flat top-k on
  `|delta|`, stable descending score and ascending flat-index ties.
* **F1 — activation-weighted uniform allocation.** Score
  `|delta_ij| * ||X_j||_2`; allocate the layer’s exact target budget uniformly
  across rows using deterministic largest remainder; select top scores within
  each row.
* **F2 — Fisher-aware allocation.** Use the exact F1 score; compute empirical
  row Fisher from `dL/dy`; allocate by `sqrt(omega_i)` with water-filling,
  row bounds, and largest-remainder rounding; select top scores within each
  row.

Each layer is processed independently and can be streamed. The aggregate
  budget is the sum of the frozen per-layer budgets, and each arm’s mask hash,
  layer density, row budget, row Fisher, and saliency summary are retained.

## Calibration and planned efficacy run

The calibration corpus must be identical across F0/F1/F2 within a draw and
must be identified by immutable IDs and hashes before model calls. It must
not include a sealed evaluation set. Planned independent seeds are `17, 29,
43`; no seed has been run by this branch. No sealed-set tuning, selection, or
readback is permitted.

The later efficacy run must preserve paired item IDs and report target-skill
capability, retention, repairs, damages, unchanged-correct, unchanged-
incorrect, layerwise density, row-Fisher distributions, Fisher/retention
relationships, mask overlap, retained delta norm, exact checkpoint and delta
hashes, and resource measurements.

The initial pass criterion is preregistered as:

1. F2 improves retention versus matched F0 and F1 in the same direction over
   the planned independent draws;
2. target capability is within 1 percentage point of the strongest matched-
   sparsity control;
3. global density is exactly matched; and
4. no conclusion changes sparsity after observing results.

If F2 does not improve retention but Fisher correlates with retention-critical
behavior, that is a mechanism result, not a pruning-success claim.

## Determinism and resource guards

All integer budgets use largest remainder with fractional-part descending and
row-index ascending ties. Within-row masks use stable descending score order
with ascending column/flat-index ties. Statistics accumulate in float64;
masked deltas retain their source dtype; reconstruction casts the masked delta
to the base dtype before addition. NaN/Inf inputs are rejected.

Qualification is synthetic and CPU-safe. A real run must enforce the existing
RAM/GPU guards, expects an RTX 4070 8GB host, and treats a resource failure as
`BLOCKED`. No efficacy run is launched by this branch.

