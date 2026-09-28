# NSA-QUAL-001 — Native Sparse Attention for compact Rineya

**Status: CPU reference mechanism implemented; real-checkpoint, GPU-kernel,
quality, training and cached-decode qualification remain open.** No pretrained
weights are downloaded, loaded, modified or published by this package. Nothing
runs on import. This is not a replacement for a published Rineya core.

Branch: `research/native-sparse-attention-v1`.
Base: `LUNCY579/ChatBubble@7d9b4fd450bc56405f6cec41459a7e83df7958ae`.
All additions live in this directory. No existing research branch is inherited
or modified. See `BRANCH_AUDIT.json` for the scope of repository inspection.

## Paper and research judgment

Primary source: Yuan et al., *Native Sparse Attention: Hardware-Aligned and
Natively Trainable Sparse Attention*, arXiv:2502.11089v2, 27 February 2025:
https://arxiv.org/html/2502.11089v2

NSA is worth a bounded long-context experiment, not an immediate production
swap. The paper combines compressed global attention, exact selected blocks,
and a local window with independent sigmoid gates (Eq. 5). Compressed attention
probabilities rank blocks; scores are summed within each KV group (Eq. 10).
The hard selection indices are not differentiable. Compressors learn through
an actual compressed-attention output path; this package does not invent a
straight-through gradient for top-k. The paper also separates all three K/V
paths to reduce shortcut learning (Section 3.3.3).

The main paper evaluates a 27B-total/3B-active MoE model, not a sub-1B retrofit.
Its 64 query heads / 4 KV groups give sixteen query heads per group. The pinned
Supa config has 16 / 8 = two. Kernel results on the paper's A100 implementation
therefore do not establish an RTX 4070 or Rineya speedup. Pretraining sparse
attention is also different from replacing attention in a pretrained model.

The social-media claim that other sparse methods are merely "hacks" is not our
research premise. The relevant question is whether this particular combination
improves the quality/cost frontier under Rineya's actual constraints.

## Implemented mechanism and declared deviations

For query q_t, the active output is

    o_t = sigmoid(a_cmp) * Attn(q_t, K_cmp, V_cmp)
        + sigmoid(a_sel) * Attn(q_t, K_sel[I_t], V_sel[I_t])
        + sigmoid(a_win) * Attn(q_t, K_win[local], V_win[local]).

1. **Aligned blocks:** compression length = stride = selection length = 32.
   This is the paper's explicit aligned special case in Section 3.3.2. It does
   not implement the paper experiment's overlapping 32/16/64 configuration or
   its general Eq. 9 mapping. Each summary becomes visible only at its block
   end. No future token may leak through a block summary.
2. **Compact compressor:** a head-shared learned intra-block position table,
   bottleneck MLP of width 32, and mean residual, separately for K and V. This
   is our declared parameterization, not an exact paper-model reproduction.
3. **GQA-aware selection:** one selected block set per query/KV group, obtained
   from summed compressed probabilities. Stable ties prefer earlier blocks.
   Eight total selected blocks include one initial and one current block;
   forced blocks are not an extra budget. Partial current blocks are causally
   masked at token level. A content-blind hashed-selection control is included.
4. **Independent paths first:** selected K/V reuse the native projections;
   compression and window paths each clone native K/V and key normalization
   into independent trainable parameters. A separately labeled shared-KV
   ablation avoids those copies. All variants retain native Q/O and head shape.
5. **Qwen3 position handling:** raw K/V are compressed before RoPE. K receives
   the corresponding head RMS normalization; a compressed K uses RoPE at the
   block's final position. V is never rotated. This is an explicit Qwen3
   adaptation. Ordinary Qwen3 source reference:
   https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py
6. **Honest reference execution:** selected raw K/V are gathered, not hidden
   behind a dense T-by-T masked attention operation. Query chunks bound a
   working gather, not total autograd memory. Routing uses a reference stable
   sort, not a fast top-k kernel. There is no fused Triton/CUDA implementation.

The compression path still gives approximately O(T^2 / block_size) pair work
at fixed block size, plus O(T * (selected_tokens + window)) exact/local work.
This is not linear attention. Full raw selected history is retained; sparse
access is not persistent KV eviction or guaranteed cache compression.

## Preserve the correct family geometry

Baseline: `Rineya-Supa-1.7.0-devfreeze`, with the main-branch config in
`Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze/config.json`.
Pinned config blob: `f5c3703b78ae2a478ae15b247e9f855e0ce2107b`.

28 layers, hidden width 1024, FFN width 3072, 16 query heads, 8 KV heads,
**head dimension 128**, vocabulary 151936. Q width is 2048; deriving head
size as 1024/16 would be WRONG. The vocabulary, tokenizer, depth, hidden/FFN
width and published files are unchanged. Budget counts assume bias-free native
Qwen3 projections and learned per-head-dimension RMSNorm scales.

The initial scout converts only zero-based layers **[6, 13, 20, 27]**. These are
predeclared evenly spaced probes, not a measured optimal layer placement.

| Configuration | Added parameters | Fraction of 596,049,920 baseline |
|---|---:|---:|
| Independent K/V, four layers | 18,031,168 | 3.0251% |
| Shared K/V, four layers | 1,252,928 | 0.2102% |
| Independent K/V, all 28 layers (not initial scout) | 126,218,176 | 21.1758% |
| Shared K/V, all 28 layers (not initial scout) | 8,770,496 | 1.4714% |

Independent four-layer total: 614,081,088 parameters. Shared four-layer total:
597,302,848. Parameter bytes are not optimizer/activation/peak-memory estimates.
The independent variant's 18M parameters are principally extra K/V matrices,
not a new vocabulary table. Tests compare analytic counts to constructed,
full-attention-width fixtures without loading a complete pretrained model.

At T=4096, the late-query upper bound is 128 summary positions + 256 selected
positions + 128 local positions = 512 positions versus 4096 dense positions.
These branches can overlap. This arithmetic is NOT an 8x latency prediction;
projections, compression, sorting, gathers, gates and the other 24 dense layers
still cost time. GPU memory and end-to-end latency must be measured separately.

## Interfaces and fail-closed limits

`Qwen3NSAAttention(..., mode="baseline")` directly delegates to its original
native attention, including the original cache/backend kwargs. The NSA paths
are not evaluated. This establishes an exact *software bypass*, not successful
conversion to active NSA. Switching to `mode="nsa"` evaluates the new mechanism;
there is no hidden dense fallback in that mode.

Active reference mode accepts full-prefix calls only. Use `use_cache=False`.
Incremental cached decoding, left padding, packed sequences, noncontiguous
positions, custom attention biases, quantized projections and native sliding
window variants are rejected rather than silently misinterpreted. Supported
masks are binary 2D right-padding masks or canonical 4D causal/key-padding masks.
Callers must provide the native cosine/sine embeddings for a prefix beginning
at position zero; the adapter cannot infer their provenance from their values.

Default reference ceilings: 4096 tokens for no-grad prefill, **256 tokens when
an autograd graph is required**, and a 256 MiB approximate chunk-gather scratch
ceiling. The latter is not a peak-memory guarantee. Chunking does not prevent
backward from retaining many chunks' activations. No 4k training claim is made.
The existing model's context limit is not changed by these reference ceilings.

Conversion validates all requested layers and constructs all wrappers before
mutating the model tree. `restore_qwen3` unwraps those same native objects.
Native tensor values are not rewritten, but wrapper `state_dict` paths change.
Do not publish this through ordinary `save_pretrained` and expect automatic
loading. Experiment export/restore and a streaming cache require a subsequent,
separately qualified implementation. No native deployment artifact is produced.

## Reproduce the local qualification

From repository root, using an existing environment with PyTorch and pytest:

```bash
python -m pytest experiments/native_sparse_attention_v1/tests -q
python -m experiments.native_sparse_attention_v1.qualify --output /tmp/nsa-qual-new
```

The output directory must not already exist. The runner performs only synthetic
CPU tests, records versions, source hashes and parameter calculations, and does
not fetch models, install packages, train, or launch background jobs. An optional
native Transformers API test uses a tiny *random* Qwen3 model only if Transformers
is already installed. A skip stays a skip; neither that test nor our local
Qwen3-shaped fixture is evidence about a published Rineya checkpoint.

Minimal opt-in use after a separately authorized local model load:

```python
from experiments.native_sparse_attention_v1 import NSAConfig, convert_qwen3
receipt = convert_qwen3(model, NSAConfig(), layers=[6, 13, 20, 27], mode="baseline")
# Compare baseline logits first. Then explicitly activate for full-prefix calls:
for i in receipt["layers"]:
    model.model.layers[i].self_attn.set_mode("nsa")
# Run under torch.no_grad() for prefixes beyond the 256-token autograd ceiling.
# model(..., use_cache=False); no model loading/training is performed here.
```

## Proposed next experiments — NOT authorization to launch

**G0: exact-host qualification.** Bind actual local Supa weight hashes, tokenizer,
chat template, config, Torch/Transformers versions, dtype and device. Never
substitute stock Qwen3 or a different family checkpoint. Require two consecutive
host-memory checks at least as strict as the currently approved host contract;
do not lower that contract to make a run pass. First test baseline-bypass logit
parity, active finite forward/backward at small T, prefix causality, padding,
and projected parameter/storage costs. No quality or speed promotion from G0.

**G1: small adaptation and retention study, after a new explicit grant.** Freeze
layer placement and a training/calibration budget before accessing a disjoint
held-out evaluation. Begin with dense baseline, dense continued-training control,
shared-KV NSA, shared-KV compressed+local, and shared-KV hashed selection at the
same block budget. This isolates selection without the independent path's 18M
extra-parameter confound. Keep learning tokens, seeds and optimizer settings
matched and report active trainable parameters, not merely allocated weights.
The independent-KV arm additionally needs a conventional parameter-matched
adapter control and an equal-compute comparison before any architecture claim.
Those training/parameter-matched-control runners are not implemented here.

Evaluate held-out NLL, exact retrieval with distractors and evidence at multiple
positions, multi-hop retrieval, short-context capability, refusal/abstention
behavior and format retention. Report paired repair/damage counts and intervals,
not a single cherry-picked score. Log branch gate values and ablate each branch:
a path that never contributes is not evidence that its mechanism was learned.
Do not tune on sealed family panels; assign fresh disjoint development/eval data.

**G2: systems work only after useful quality survives.** Implement and validate
fused block selection and a causal streaming KV/summary cache, then compare
against the existing optimized dense backend on the actual target GPU. Separate
prefill, decode and backward timings; record peak allocated/reserved VRAM, host
RSS, end-to-end model latency and total parameters. Include 512/1k/2k/4k; expand
8k/16k only with explicit memory/runtime qualification. A pure PyTorch reference
must not be marketed as the paper's hardware-aligned fast implementation.

**Stop/advance rule:** a quality/cost frontier improvement is the goal. Do not
promote because tests pass, because attention-pair counts shrink, or because one
untrained conversion happens to produce text. Conversely, an untrained retrofit
loss does not refute native sparse pretraining; keep these hypotheses separate.
