# Design and mathematical contract

## 1. Scope: preserve computation before changing it

The prior cross-scale lab explored transporting donor weights into a fixed small-model shape. This package instead connects intact computations. It is a separate exploratory branch and neither reruns nor amends the old P4/P5/P6 conclusions.

There are three different achievements to measure: a connection can reconstruct activations; a composed model can remain coherent; a composed model can improve task performance. None implies the next. The source lab's later per-depth correspondence measurements motivate trying a bridge, not assuming that it will carry useful new behavior.

The initial scientific target is the exact historical Supa-A1.1 subject, with its FP16 storage and custom logit policy explicitly recorded. Supa Core 1.7 is a possible future separately registered host, never an alias. Source and native runtime qualification must precede model claims.

## 2. Boundary definitions are part of the experiment

All block indices in code are zero-based; prefix lengths are counts.

For a pre-normalized decoder block define:

```
z = h + Attention(Norm_attention(h), causal_context)
h_next = z + FFN(Norm_ffn(z))
```

A `post_attention_raw` capture is `z`, before `Norm_ffn`. A `block_output_raw` capture is `h_next`, before any final model normalization. `block_input_raw` is `h`.

For a midpoint large→small stitch, retain large blocks `[0:18]`, fit the bridge from large block 17's raw output to small block 13's raw output, and continue small blocks `[14:28]`. Reverse the direction with small `[0:14]` and large `[18:36]`. A capture from the output of block 18 is NOT the same boundary as a prefix containing 18 blocks.

Graft bridges are separately fitted at raw post-attention sites. Do not reuse raw block-output stitch maps at a post-attention graft site merely because widths match. The demonstration deliberately performs separate fits for these two site kinds.

## 3. Affine ridge fitting

For paired row-vector activations X (n × d_source) and Y (n × d_target), fit:

```
min_W,b sum_i ||x_i W + b - y_i||² + lambda ||W||²_F
b = mean(Y) - mean(X) W
W = (X_centered^T X_centered + lambda I)^(-1) X_centered^T Y_centered
```

The implementation does not explicitly invert a matrix. It uses centered sufficient statistics, a positive regularization floor, and a Cholesky solve in FP64. A Chan-style correction merges batch means and centered moments, avoiding the particularly unstable subtraction of two large uncentered second moments. The objective is a SUM over fitting tokens; the relative regularization is multiplied by mean input covariance trace, so its interpretation is explicit.

Sufficient-statistic memory depends on feature widths, not n². The CLI additionally caps combined capture payloads at 256 MiB, bounds solver workspace at 512 MiB, and checks an estimated incremental peak plus margin against available host/cgroup memory. These checks do not guarantee allocation or protect against another process consuming memory after preflight.

Each direction is fit independently. A learned reverse map is not assumed to be an inverse or a transpose. With different residual widths there cannot be a two-sided full-space inverse. An affine offset is useful for differences in residual means and is not penalized.

`FIT` alone determines the weights. `MAP_CHECK` supplies reported reconstruction metrics only. Any ridge hyperparameter selection requires a separately declared inner-development procedure or a new fit; repeatedly selecting on MAP_CHECK turns it into selection data. Full rectangular maps are the starting point. Rank 4/8/16 is not assumed sufficient, and rank compression is not part of this implementation.

## 4. FFN graft: preserve the host residual path

At a selected host raw post-attention site z_S:

```
u_L = U z_S + b_U
Delta_L = donor_FFN(donor_Norm_ffn(u_L))
Delta_S = V Delta_L
h'_next = (z_S + host_FFN(host_Norm_ffn(z_S))) + alpha Delta_S
```

Here column-vector notation is used. U maps the host residual to the donor residual basis; V is the LINEAR part of an independently fitted donor→host raw-residual map. The donor's existing normalization and FFN remain intact and frozen. No donor attention module, donor embedding or new KV bank is retained.

Why no return bias? If A(x) = Vx + b, then A(x + Delta) - A(x) = V Delta. Adding b to a residual displacement would inject a constant that is not part of the transported change. The code refuses a return bridge with a bias.

This construction still makes a substantive unproved assumption: a map fitted to ordinary donor residuals is appropriate for the donor FFN's displacement on mapped host states. The donor FFN can be evaluated off its training distribution. That is exactly why the nonzero path requires a functional test and why an ordinary adapter control is needed.

The host addition order is preserved explicitly. An inference bypass returns the original baseline computation, without evaluating the donor. A fixed alpha of zero is another literal no-op control. A trainable alpha initialized to zero is refused because the bridge gradient is then initially blocked. We do not initialize both the scalar gate and a downstream matrix to zero and advertise complete gradient coverage.

Alpha 0.05 and 0.10 are proposed screen values, not a discovered optimum. Any later normalization of the donor delta, learned gain constraint or extra gate would be a new operator with its own control, not an invisible numerical repair.

## 5. Ordinary no-donor control

`OrdinaryAdapter` computes a residual contribution using the frozen host normalization, an affine host→inner projection, SiLU, and a bias-free inner→host projection. At inner width d_donor its trainable inventory is:

```
2 * d_host * d_donor + d_donor
```

This exactly matches the two graft bridges, when alpha is fixed. It has no frozen donor FFN. Its output projection starts at zero, with a nonzero fixed gain, so it initially preserves the host and has a live output-projection gradient. The input projection receives a task gradient after the output projection moves. The tests make that distinction explicit.

Matching trainable parameters is NOT matching FLOPs, stored parameters or preprocessing cost. Donor captures and bridge calibration consume additional resources; report them. The same task examples, teacher outputs, token masks, optimizer schedule and permitted selection budget must be used for the trained comparison. A successful no-donor control may be the preferable deployment outcome.

## 6. Prefix/suffix stitching

The prefix contributes its embedding and retained blocks. The suffix contributes its retained blocks, final normalization and output head. The composed module stores only these retained components, not both owning full-model objects. External callers must also release their original model references to obtain those memory savings.

In a cross-source composition, the prefix embedding and suffix output table are distinct. They must not be silently tied just because each original model tied its own head. The 8B source is also recorded as untied in the parent audit. The analytic budget handles source tables and same-source identity controls separately.

Each block keeps its own attention geometry. Residual width is never calculated as head_count × head_dim: the inspected small Qwen3 shape has width 1024 but 16 query heads of dimension 128. Equal K/V sizes do not establish equal learned coordinates.

The initial runner is intentionally uncached and batch-one. All retained blocks process the same growing causal stream; a large prefix is not a prompt-once encoder. A future cache implementation must introduce fresh cache ownership, consistent positions and cached/uncached parity tests. It cannot reuse old layer indices or a single model's cache object by assumption.

## 7. Why individual weight rebasing is not the first implementation

A matching tensor shape means multiplication is defined, not that neuron/head bases are aligned. Similarly, generic rotations do not commute with coordinate-wise SiLU, the FFN's elementwise product, or an arbitrary learned diagonal normalization gain. Rectangular mappings also change full-space geometry and can discard directions.

Consequently, this package does not assume that one residual-space Procrustes map can safely rebase all attention and FFN weights. It keeps internal modules intact and tests their boundary compatibility directly. This is a design choice, not a theorem that stitching must outperform aligned weight transport.

## 8. Accounting and outputs

For the parent shapes, `planning.py` reproduces the recorded complete-model inventories (596,049,920; 4,022,468,096; 8,190,735,360 parameters). It then derives:

| Candidate | Stored parameters | Trainable bridge parameters, fixed alpha |
|---|---:|---:|
| Small host + one 4B FFN graft | 676,008,960 | 5,245,440 |
| Small host + one 8B FFN graft | 755,441,664 | 8,392,704 |
| 4B first 18 → small last 14 | 2,584,150,016 | 2,622,464 |
| Small first 14 → 4B last 18 | 2,584,153,088 | 2,624,000 |

These are analytic, nonquantized inventories including distinct retained tables. They are not measured RSS, optimizer memory, GPU fit, token/s or latency. The standalone parent custom logit policy is not a trainable matrix in these counts.

`composition_descriptor()` hashes source identities, sites/cuts, gains and current interface states. Recompute it after updates. The `.identity` field on a composed model describes the output-policy source, not a new candidate checkpoint; never use it alone to label candidate predictions.
