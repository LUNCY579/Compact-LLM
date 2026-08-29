---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3-0.6B
---

# Rineya Nimbus 1.6 / Nimbus-A1

Nimbus 1.6 is the current Nimbus lineage core. `Nimbus-A1` is its A-series
tracking alias; the underlying immutable version and core ID remain 1.6.0.

## Construction

Nimbus is a 28-layer Qwen3-0.6B-family weight soup at uniform alpha 0.40 from
the official Qwen3 anchor toward Rineya Halo Uncensored 1.5. The layerwise-soup
lab selected candidate `cand_d1b08a8faf299ce4` as `global_best`.

- Weight bytes: 1,503,300,328
- SHA-256: `7b66b8d0ef06cf7f71bc9001b0c9a330d5f8a0eec1d11eb173ece70ca66ef49c`
- Layers / hidden size: 28 / 1024
- Architecture: plain `Qwen3ForCausalLM`; no remote code required

This is not the former `Rineya-Nimbus-Dense-1.0-nobias` checkpoint. That file is
weight-identical to Supa Core 1.7 and is retained only as Supa-A1's ablation
control.

## Measured behavior

Standardized 2026-08-26 vault protocol: BF16, greedy seed 17, thinking disabled,
24 generated tokens, 4096 context cap.

| Split | Strict exact |
|---|---:|
| Showcase64 | 30/64 |
| Stability192 | 83/192 |
| All256 | 113/256 |

An older source-selection run measured 29/64. The difference is protocol/run
drift, not a Halo/Nimbus label swap. Showcase is selection data and sealed TEST
has not been run.

## Role and limits

Nimbus is the application base for the long-context step-200 LoRA experiments
that produced Supa/Baya and the routed systems. Those descendants' scores must
not be attributed to this base core. The current Demo prompts are approximately
1.1K–1.55K input tokens; no 8K/16K/32K capability is established. Grounded RAG
remains weaker than the 19-layer Aegis specialist.

See [`CORE.json`](CORE.json) and
[`../FAMILY_PERFORMANCE_AND_RESEARCH.md`](../FAMILY_PERFORMANCE_AND_RESEARCH.md)
for exact provenance, family comparisons, and research limits.
