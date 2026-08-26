# Cornerstone-1 literature and novelty audit

Audit date: 2026-08-24. This is a scoped architecture audit, not a systematic
review. It uses the pinned RineyaBeta paper recorded in `REPO_PROVENANCE.md` and
primary paper or proceedings pages. RineyaBeta remains read-only.

## Claim boundary

The lab must not claim novelty for attention-derived KV importance, heavy-hitter
retention, sink-plus-recent caching, accumulated-attention bias, normalized
attention saliency, query-aware selection, KV quantization, adaptive head/layer
budgets, compressed memory for evicted KV, or recovery of evicted information.
All have direct prior art.

The narrower research object tested here is:

> explicit realized scoring-exposure state `C`, carried with each surviving
> semantic cache entry through repeated inter-chunk gather and eviction, and the
> behavior of `S / max(C, 1)^e` when `C`, compression-round count, and information
> access are causally controlled.

That sentence defines an experimental distinction, not a novelty finding. A
publishable claim would still require matched implementations, independently
registered evidence, and a broader search.

## Closest prior work

| Work | Primary contribution relevant here | Consequence for this lab |
|---|---|---|
| [StreamingLLM](https://arxiv.org/abs/2309.17453) | Preserves attention sinks and a recent window for bounded streaming inference. | B4 is only a matched positional control, not a full reproduction, and sink/recent retention is not novel. |
| [H2O](https://arxiv.org/abs/2306.14048) | Dynamically retains recent tokens and historical attention heavy hitters. | Raw cumulative attention importance and heavy-hitter eviction are not novel. |
| [Scissorhands](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a452a7c6c463e4ae8fbdc614c6e983e6-Abstract-Conference.html) | Uses persistence of token importance under a fixed KV budget. | Persistent importance statistics are established prior art. |
| [SnapKV](https://arxiv.org/abs/2404.14469) (NeurIPS 2024) | **Applies 1D max-pooling directly over attention scores before top-k selection**, to "retain the features surrounding the selected attention features" so salient regions survive as contiguous groups rather than isolated tokens. Reported kernel size 5, observation window 16, prompt KV 1024, with an ablation showing pooling improves accuracy. Their stated hypothesis is that *the initial portions of critical token clusters are weighted higher by attention*. | **This is pooling-over-attention-scores, i.e. exactly the "C2 over C1" construction.** It is not novel and must never be presented as such. Our own span-fragmentation diagnosis (`' Luc'` survives, `'an'` evicted) is a rediscovery of their stated hypothesis. Our `pool_width=13` differs from their 5 only as a hyperparameter at a different budget. The single narrower object left is pooling over *iteratively accumulated, exposure-normalized* significance across repeated compression boundaries -- SnapKV compresses once at end-of-prompt, this runtime compresses at every chunk boundary carrying `S`/`C` state. `N-073` is testing whether that carried state contributes anything at all; if it nulls, what remains is SnapKV with a larger kernel. |
| [NACL](https://aclanthology.org/2024.acl-long.428/) | Identifies bias in accumulated local attention statistics and combines proxy-token with random eviction. | The lab cannot claim to discover accumulated-attention bias; B6/B7 and stronger proxy controls matter. |
| [PyramidKV](https://arxiv.org/abs/2406.02069) | Varies cache allocation by layer using pyramidal information-flow observations. | Layer-wise allocation is not novel. Experiment B must isolate whether carried `C` adds predictive value. |
| [Ada-KV](https://arxiv.org/abs/2407.11550) | Derives and evaluates head-wise adaptive cache-budget allocation under global constraints. | Head-wise adaptive allocation is not novel and is a direct Experiment-B comparator. |
| [Quest](https://arxiv.org/abs/2406.10774) | Uses current-query/key page bounds for query-aware sparse attention. | Query awareness is valuable prior information and must be equalized across policies. |
| [KIVI](https://arxiv.org/abs/2402.02750) | Applies tuning-free asymmetric 2-bit KV quantization. | Quantization is orthogonal; byte comparisons must account for it without calling it Cornerstone 1. |
| [ZipCache](https://arxiv.org/abs/2405.14256) | Combines KV quantization with a normalized-attention saliency measure. | “Normalized attention score” is not available as a broad novelty claim; the lab must state exactly what its denominator counts. |
| [AhaKV](https://arxiv.org/abs/2506.03762) | Analyzes position-dependent bias in accumulated attention and corrects it using entropy-conditioned scaling plus value information. | Bias correction in general is not novel. Age/position and AhaKV-style controls are essential before attributing an effect to realized exposure. |
| [KVzip](https://arxiv.org/abs/2505.23416) | Scores query-agnostic KV importance through context reconstruction. | Query-agnostic reusable cache compression is distinct from this lab's offline query-conditioned path and must not be conflated. |
| [LAVa](https://aclanthology.org/2025.findings-emnlp.737/) | Jointly derives dynamic head and layer budgets from attention-output loss. | Generic joint layer/head allocation is established prior art; Experiment B can only test a Cornerstone-1-specific allocation signal. |
| [DynamicKV](https://aclanthology.org/2025.findings-emnlp.426/) | Dynamically changes per-layer budgets based on task-dependent layer activation patterns. | Task-aware dynamic budgets are not novel and provide another allocation comparator. |
| [ZigZagKV](https://aclanthology.org/2025.coling-main.596/) | Allocates layer budgets using layer uncertainty. | Entropy or uncertainty allocation alone cannot be described as a new architecture class. |

## Direct overlap with deeper architecture proposals

Recent work makes the boundary for Experiments D and E especially strict:

- [Tensor Cache](https://arxiv.org/abs/2605.22884) feeds evicted KV pairs into a
  fixed-size second-level associative memory.
- [IndexMem](https://arxiv.org/abs/2605.25475) combines learned KV eviction with
  a lightweight latent memory that compensates for evicted attention.
- [MomentKV](https://arxiv.org/abs/2606.01563) retains compact moment statistics
  of evicted keys and values to approximate their attention contribution.
- [RestoreKV](https://arxiv.org/abs/2608.01247) creates a compact learned restore
  cache to recover behavior after aggressive query-agnostic eviction.

Therefore significance memory slots, latent summaries, moment summaries, and
restore/recompute paths are not novel as broad mechanisms. The only potentially
distinct question in this lab is whether **realized exposure-normalized
significance** is a useful write, allocation, or recovery signal under matched
bytes and matched information access.

## Required comparison logic

Before any positive architectural claim:

1. Compare B2 raw accumulation with B3 true realized-count normalization.
2. Compare true `C` with constant, age-derived, shuffled, circular/wrong-token,
   delayed, noisy, and head-averaged counts.
3. Test the method-by-compression-round interaction while holding the prompt,
   query, model, final history budget, decoding, and seed fixed.
4. Compare against position/sink/recent, random, shuffled-significance, and
   oracle-evidence controls where mathematically valid.
5. For adaptive budgets, include uniform and non-`C` allocation controls and
   discuss Ada-KV, PyramidKV, LAVa, DynamicKV, and ZigZagKV explicitly.
6. For memory or recovery, compare equal-byte summaries/restoration and cite the
   direct 2026 overlap above.
7. Keep query-conditioned and query-agnostic claims separate.

## Current verdict

`NO_NOVELTY_CLAIM`. The present Halo evidence supports implementation fidelity,
family compatibility, and an exploratory retention signal only. Whether carried
exposure state is reusable beyond eviction remains an open causal question.
