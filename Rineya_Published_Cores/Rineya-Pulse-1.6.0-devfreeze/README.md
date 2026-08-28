---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3-0.6B
---

# Rineya Pulse 1.6

Pulse 1.6 is a 28-layer cross-depth Qwen3-family core. The 19 retained decoder
blocks are blended toward the Halo donor with the depth-ramped `late_rineya`
alpha shape (mean 0.5); embeddings, dropped blocks, final norm, and tied head
remain from official Qwen3.

- Weight SHA-256: `2357574db5557c288f57a6b97b2761830ea265baf3c6a1c0270357a7d0eaa322`
- Standardized Demo: 31/64 Showcase, 89/192 Stability, 120/256
- Recorded chat quality: 0.8612; dual composite: 0.7561

Showcase is selection data. Demo prompts are roughly 1.1K–1.55K tokens; sealed
TEST and general 8K/16K/32K evaluation have not run. Grounded RAG is weaker than
the 19-layer Aegis specialist. See `CORE.json` and
`../FAMILY_PERFORMANCE_AND_RESEARCH.md`.
