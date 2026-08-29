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
| qwen3 ↔ nimbus | **config_identical** | 0.00532 | 0.999986 | 34.9 % |
| qwen3 ↔ supa_a11 | architecture_compatible | 0.00681 | 0.999977 | 72.6 % |
| halo ↔ nimbus | architecture_compatible | 0.00148 | 0.9999989 | 50.1 % |
| halo ↔ supa_a11 | architecture_compatible | 0.00417 | 0.999991 | 73.3 % |
| nimbus ↔ supa_a11 | architecture_compatible | 0.00393 | 0.999992 | 72.0 % |

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
   bit-identical to stock Qwen3 in all three Rineya cores.** Nothing in this family
   was achieved by moving a normaliser or the vocabulary geometry.
2. **`k_proj` is the most-changed attention projection in all three models**, ahead
   of `v_proj`, `o_proj`, and `q_proj` in that fixed order. The key pathway — the one
   that decides *where* a head looks — moved most. This is the first piece of
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

That asymmetry is a free natural experiment: **in the nine frozen layers,
`Supa − Nimbus` is the LoRA adapter with zero soup contamination**
(‖Δ‖² = 10.24). Phase 7 uses it to separate "what the soup did" from "what the
adapter did" without having to re-derive either.

Among touched layers, delta magnitude rises mildly with depth (layer 0 ≈ 0.0179 →
layer 26 ≈ 0.0202 for Halo).

## 4. The family is a line plus one direction

Cosine between delta vectors (all non-embedding tensors, float64):

| pair of deltas | cosine | best scalar fit | residual |
| --- | ---: | ---: | ---: |
| `Δhalo` vs `Δnimbus` | **+0.9946** | `Δnimbus ≈ 0.7999 · Δhalo` | 10.3 % |
| `Δhalo` vs `Δsupa` | +0.8123 | `Δsupa ≈ 0.8107 · Δhalo` | 58.3 % |
| `Δnimbus` vs `Δsupa` | +0.8165 | `Δsupa ≈ 1.0134 · Δnimbus` | 57.7 % |

The Halo↔Nimbus relationship is not approximately collinear, it is collinear with a
**constant coefficient of 0.7999–0.8000 in every one of the nineteen touched layers,
to four decimal places.** Nimbus is, to within a 10 % residual, the point at
α = 0.80 along the ray from Qwen3 through Halo.

Nimbus's recorded recipe is a uniform global α = 0.40 soup of Qwen3 with
*Halo-Uncensored*. Reading those together implies `Δhalo_uncensored ≈ 2 · Δhalo`,
i.e. Halo sits at the midpoint of that same ray — consistent with Halo's own recipe
(a 50/50 average of two donor adapters). **The entire Halo/Nimbus branch lies on one
line through Qwen3.**

Supa-A1.1 is the only checkpoint that leaves the line: cosine ≈ 0.81 against both,
with a 58 % residual contributed by its LoRA adapter.

### Consequence for Phases 8 and 15

The brief's §34 (blockwise α on `θ_base + α·Δ`) and §35 (position-aware soup) assume
a family rich enough for interpolation to discover something. For Halo and Nimbus it
is not: blockwise α over those two is a **one-dimensional search that already
contains Nimbus at α ≈ 0.80**, so any "new" soup of the two is a point on a segment
we can characterise analytically. The genuinely two-dimensional part of the panel is
the plane spanned by the Halo ray and the Supa adapter direction. Phase 8 will be
planned against that geometry rather than run as a blind sweep, and the negative
result — "interpolating Halo with Nimbus cannot produce a new model, only a new
α" — is reported as a finding rather than buried.

## 5. Recorded approximation

Halo's store dtype is float32 (2.24 GiB). Inference runs it in bfloat16 to leave
headroom on an 8.59 GB card. This is a downcast of the run, never of the
checkpoint, and it is flagged in the manifest and in every result row Halo appears
in. Supa-A1.1 is exempt from any such policy: it must load in float16, because the
float16 store *is* the checkpoint's contribution, and a bfloat16 load would discard
the precision it exists to preserve.
