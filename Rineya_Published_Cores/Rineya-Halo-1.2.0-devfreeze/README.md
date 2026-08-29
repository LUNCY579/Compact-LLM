---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3-0.6B
---

# Rineya Halo 1.2

Halo uses the official 28-layer Qwen3 scaffold. Its 19 retained blocks are
blended 50/50 with an equal FP32 delta soup of the dual-SFT v1 `5e-4` and v2
`1e-4` adapters. The 28-layer core was assembled, not directly fine-tuned.

- Weight SHA-256: `38fb96fe286776919032c4418eef0a2534e2c2c2865c1465cc9dd4cfe7e36544`
- Standardized Demo: 29/64 Showcase, 86/192 Stability, 115/256
- Historical source-selection Showcase: 30/64

The 29/30 difference is protocol/run drift, not a Nimbus label swap. Halo is the
only core with held-out pooled KV-compression confirmation; that runtime result
must not be inherited by this plain full-cache checkpoint. Sealed TEST and
general 8K/16K/32K evaluation have not run. See `CORE.json` and
`../FAMILY_PERFORMANCE_AND_RESEARCH.md`.
