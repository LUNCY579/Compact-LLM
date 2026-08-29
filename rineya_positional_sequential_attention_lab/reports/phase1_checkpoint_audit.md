# Phase 1 — Structural checkpoint audit

Artifacts: [`manifests/model_manifest.json`](../manifests/model_manifest.json),
[`manifests/tensor_hashes.csv`](../manifests/tensor_hashes.csv),
[`results/model_pairwise_distance.csv`](../results/model_pairwise_distance.csv),
[`results/weight_delta_by_layer.csv`](../results/weight_delta_by_layer.csv),
[`results/weight_delta_by_head.csv`](../results/weight_delta_by_head.csv).
Each carries a sibling `*.provenance.json`. Source checkpoints were opened read-only;
`tests/test_immutability.py` re-verifies this.

## 1. The panel is one architecture

All four checkpoints are 28-layer Qwen3-0.6B: hidden 1024, intermediate 3072,
16 query heads, 8 KV heads (GQA group size 2), head_dim 128, vocab 151936,
rope_theta 1e6, tied embeddings, no attention bias, RMSNorm plus Qwen3's
`q_norm`/`k_norm` over the head dimension. Parameter count is identical to the
digit: **596,049,920** total, **440,467,456** non-embedding.

Pair classification:

| pair | class | rel. Frobenius | cosine | entries changed |
| --- | --- | ---: | ---: | ---: |
| qwen3 ↔ halo | architecture_compatible | 0.00682 | 0.999977 | 50.1 % |
| qwen3 ↔ nimbus | **config_identical** | 0.00548 † | 0.999985 † | 50.1 % † |
| qwen3 ↔ supa_a11 | architecture_compatible | 0.00681 | 0.999977 | 72.6 % |
| halo ↔ nimbus | architecture_compatible | 0.00148 | 0.9999989 | 50.1 % |
| halo ↔ supa_a11 | architecture_compatible | 0.00417 | 0.999991 | 73.3 % |
| nimbus ↔ supa_a11 | architecture_compatible | 0.00393 | 0.999992 | 72.0 % |

† **Scope correction.** `results/model_pairwise_distance.csv` computes this row over
311 tensors / 751 632 384 parameters, because Nimbus ships `lm_head.weight` as a copy of
`embed_tokens` and the audit counted both; every other row uses 310 / 596 049 920.
Recomputed on the common scope the row is 0.00548 / 0.999985 / **50.1 %**, not
0.00532 / 0.999986 / 34.9 %. The published "34.9 %" made Nimbus look like it changed far
fewer entries than Halo — in fact **both soups change exactly the same entry set**, since
both freeze the same nine blocks and 0.5014 = 19/28 of the non-embedding parameters. The
CSV is left as generated, with this correction recorded here; the scope column is a
Phase-7 fix.

No pair is byte-identical. Differences from `qwen3` in config schema are cosmetic
(`store_dtype`, `rope_parameters` dict vs flat `rope_theta`, the tied-`lm_head`
convention) except Supa-A1.1's four real fields: `architectures[0]`, `auto_map`,
`abstain_token_id`, `abstain_logit_bias`.

**Tokenizers are behaviourally identical across all four** — same vocabulary, same
chat-template render, same token ids on probe strings. Cross-model positional
comparison therefore runs on byte-identical token sequences. That removes a
confound that would otherwise be indistinguishable from a positional effect.

## 2. Every change lives in the linear projections

Mean relative Frobenius against Qwen3, averaged over the 28 layers:

| family | halo | nimbus | supa_a11 |
| --- | ---: | ---: | ---: |
| `q_proj` | 0.01127 | 0.00908 | 0.01435 |
| `k_proj` | **0.01334** | **0.01073** | **0.01708** |
| `v_proj` | 0.01221 | 0.00984 | 0.01535 |
| `o_proj` | 0.01174 | 0.00946 | 0.01410 |
| `gate_proj` | 0.01652 | 0.01327 | 0.01833 |
| `up_proj` | 0.01479 | 0.01189 | 0.01753 |
| `down_proj` | 0.01066 | 0.00860 | 0.01220 |
| `q_norm`, `k_norm` | 0 | 0 | 0 |
| `input_layernorm`, `post_attention_layernorm` | 0 | 0 | 0 |
| `embed_tokens`, `norm`, `lm_head` | 0 | 0 | 0 |

Two things worth stating plainly:

1. **Embeddings, the final norm, every layernorm, and both `q_norm`/`k_norm` are
   numerically identical to stock Qwen3** — exactly so in Halo and Nimbus (all 114 such
   tensors, zero differing entries), and in Supa-A1.1 with one exception:
   `embed_tokens`, where 22 163 of 155 582 464 entries (0.014 %) differ by at most
   3.0e-8 = 2⁻²⁵. That is Supa's float16 store flushing tiny bfloat16 values, not an
   edit; the table below prints 9.7e-09 rather than 0 for that cell. Nothing in this
   family was achieved by moving a normaliser or the vocabulary geometry.
   ("Bit-identical" would be the wrong word regardless: Halo stores float32 and Supa
   float16 against Qwen3's bfloat16, so the *bytes* necessarily differ — it is the
   decoded values that are equal.)
2. **`k_proj` is the most-changed attention projection in all three models.** K and V
   lead everywhere; O and Q swap between the soups (Halo and Nimbus: K > V > O > Q) and
   Supa-A1.1 (K > V > Q > O, 0.01435 vs 0.01410 — visible in the table above). An
   earlier draft of this report asserted one fixed order across all three and was
   contradicted by its own table eight lines higher. The key pathway — the one that
   decides *where* a head looks — moved most, and that part holds for all three. This is the first piece of
   evidence bearing on brief question 9, and it makes the K-side interventions of
   §33 a priority rather than an afterthought. It is *not yet* evidence that the
   change improved positional behaviour; that is Phase 3 and Phase 7's job.

MLP `gate_proj`/`up_proj` changed more than any attention projection, so these
merges are not primarily attention edits.

## 3. Halo and Nimbus share nine frozen layers

Layers **{6, 7, 8, 9, 10, 14, 18, 23, 24}** are byte-identical to Qwen3-0.6B in both
Halo and Nimbus — `‖Nimbus − Qwen3‖² = 0.000000` over those layers exactly. Supa-A1.1
touches all 28, but its delta in those nine layers (~0.0097–0.0107) is roughly 60 % of
its delta elsewhere (~0.016–0.019).

`Supa − Nimbus` is the LoRA adapter with zero soup contamination — and, since Supa is
*defined* as Nimbus plus that adapter, this holds in **all 28 layers**, not only the nine
frozen ones (an earlier draft framed it as a property of the frozen block; it is a
property of the construction). What the nine frozen layers add is different and still
useful: there `Nimbus ≡ Qwen3` exactly, so the adapter delta sits on an *unmodified base*,
which lets Phase 7 ask whether the adapter behaves differently on souped versus stock
weights. Over those nine layers ‖Δ‖² = 10.24.

Among touched layers, delta magnitude rises mildly with depth (layer 0 ≈ 0.0179 →
layer 26 ≈ 0.0202 for Halo).

## 4. The family is a line plus one direction

Cosine between delta vectors (all non-embedding tensors, float64):

| pair of deltas | cosine | best scalar fit | residual |
| --- | ---: | ---: | ---: |
| `Δhalo` vs `Δhalo_uncensored` | **+1.00000000** | `Δhalo_unc = 2.000000 · Δhalo` | **0.0001 %** |
| `Δhalo` vs `Δnimbus` | +0.9946 | `Δnimbus = 0.799887 · Δhalo` | 10.3 % (see below) |
| `Δhalo` vs `Δsupa` | +0.8123 | `Δsupa ≈ 0.8107 · Δhalo` | 58.3 % |
| `Δnimbus` vs `Δsupa` | +0.8165 | `Δsupa ≈ 1.0134 · Δnimbus` | 57.7 % |

### The Halo relation is exact, and it is measured, not inferred

An earlier draft *inferred* `Δhalo_uncensored ≈ 2 · Δhalo` from the two recorded
recipes. That was unnecessary: `Rineya-Halo-Uncensored-1.5.0-devfreeze` is on disk in
the same vault the audit already reads. Measured directly in float64 over all 310
shared tensors, **cosine = 1.00000000 and the coefficient is 2.000000 with a residual
of 1e-6** — both stores are float32, so this is exact arithmetic, not an approximation.
Halo is the exact midpoint of the Qwen3 → Halo-Uncensored ray.

**The 10.3 % Halo↔Nimbus residual is not a deviation from collinearity. It is bfloat16
store rounding, in full.** The orthogonal residual of `Δnimbus` about `Δhalo` is
`‖e‖ = 0.816612`; the bfloat16 rounding error of the *exact* float32 target
`Qwen3 + 0.8·Δhalo` is `‖r‖ = 0.816611`. Identical to six significant figures. So

```
Δnimbus = 0.8 · Δhalo     exactly, in the merge arithmetic
```

and every hedge in the earlier draft ("approximately", "to within a 10 % residual")
mis-attributed a storage artefact to the recipe. Applying Nimbus's own
`merge_recipe.json` (global α = 0.40, no per-layer alphas) to the real donor reproduces
the shipped Nimbus store bit-exactly on 99.9997 % of elements.

Supa-A1.1 is the only checkpoint that leaves the line: cosine ≈ 0.81 against both, with
a 58 % residual contributed by its LoRA adapter.

### Consequence for Phases 8 and 15 — corrected

An earlier draft of this report concluded that "blockwise α over Halo and Nimbus is a
one-dimensional search that cannot produce a new model, only a new α". **That was
wrong, and it was the headline planning claim.** Collinearity of the two *global*
deltas does not make the *blockwise* family one-dimensional.

Write the blockwise soup as `θ|_b = θ_halo|_b + α_b (θ_nimbus|_b − θ_halo|_b)`.
Substituting the exact relation `Δnimbus = 0.8 · Δhalo` gives

```
θ|_b = Qwen3|_b + (1 − 0.2 α_b) · Δhalo|_b ,      c_b := 1 − 0.2 α_b ∈ [0.8, 1.0]
```

so each of the **19 decoder blocks with a nonzero Halo delta** (the other nine are
frozen, §3) carries an *independent* scaling of its own share of one shared direction.
The reachable set is a 19-dimensional box, not a segment. Recomputed: a blockwise model
that splits the blocks to half the total delta energy sits **0.9819 away from the
uniform-α line, which is 50.0 % of the segment's own length** (0.2·‖Δhalo‖ = 1.9652) and
1.20× the storage residual the report itself treats as meaningful.

The accurate statement, which is narrower and more useful than either extreme:

- **Uniform global α over Halo and Nimbus is genuinely 1-D and already contains Nimbus
  at α = 0.80.** Running that sweep would rediscover a checkpoint we have.
- **Blockwise α is a 19-parameter search**, but a *constrained* one: it can only rescale
  each block's Halo-delta by a factor in a narrow 20 % band `[0.8, 1.0]`, never change a
  block's direction, and never touch the nine frozen blocks. That is a real search space
  and an open question, not a foregone conclusion.
- The genuinely **new** direction in the panel is the Supa adapter, which is off the
  Halo ray by a 58 % residual.

Phase 8 is planned against that corrected geometry: skip the uniform-α sweep as
analytically settled, treat blockwise α as a live 19-dimensional (but tightly bounded)
search, and treat the Halo-ray × adapter plane as the two-dimensional part.

## 5. Recorded approximation

Halo's store dtype is float32 (2.24 GiB). Inference runs it in bfloat16 to leave
headroom on an 8.59 GB card. This is a downcast of the run, never of the
checkpoint, and it is flagged in the manifest and in every result row Halo appears
in. Supa-A1.1 is exempt from any such policy: it must load in float16, because the
float16 store *is* the checkpoint's contribution, and a bfloat16 load would discard
the precision it exists to preserve.
