---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3-0.6B
---

# Rineya Aegis 1.1

Aegis is the 19-layer archived Pulse 1.0 teacher plus a merged LoRA trained for
one epoch at learning rate `5e-4` on 12,000 short-response and 4,410
grounded-RAG examples.

- Weight SHA-256: `15fab4e7ad2b7289f9e8945db32e3e242105c72e5d7f8ffa21b93817164639c5`
- Standardized Demo: 2/64 Showcase, 20/192 Stability, 22/256
- Grounded-RAG dual composite: 0.8382; oracle supported: 1.000; real supported: 0.7358

Aegis is retained as the measured grounded-RAG specialist, not as a
long-context core. Sealed dual-skill TEST has not run. See `CORE.json` and
`../FAMILY_PERFORMANCE_AND_RESEARCH.md`.
