---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3-0.6B
---

# Rineya Lyra 1.4

Lyra is a 28-layer 50/50 cross-depth stitch using an equal FP32 delta soup of
dual-SFT v1 adapters trained at `5e-4` and `1e-5`.

- Weight SHA-256: `87c57b8eef24238cbba8bc98a38298c9c200a0fa05102b368d1c6cdaf363c5aa`
- Standardized Demo: 28/64 Showcase, 86/192 Stability, 114/256
- Recorded chat quality: approximately 0.8667, highest in the older 28L cohort

High automated chat-Q does not imply long-context or grounded-RAG leadership.
Sealed TEST and general 8K/16K/32K evaluation have not run. See `CORE.json` and
`../FAMILY_PERFORMANCE_AND_RESEARCH.md`.
