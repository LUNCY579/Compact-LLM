---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3-0.6B
---

# Rineya Halo Uncensored 1.5

Halo Uncensored uses the same donor direction as Halo but fully replaces the 19
retained blocks (`alpha=1.0`). “Uncensored” names an answer-biased experimental
pair; it is not a safety certification.

- Weight SHA-256: `c5d4a17f69ca932a5d4ad29e4f9ba1a29d7a137c08951fd89f61b4cb95444503`
- Standardized Demo: 31/64 Showcase, 96/192 Stability, 127/256
- Oracle abstention: 0.000; nine recorded RAG hard failures

This core should not be described as dual-skill or as the family RAG model. It
is retained for censored/answer-bias comparisons and as a System48 arm. Sealed
TEST and general 8K/16K/32K evaluation have not run. See `CORE.json` and
`../FAMILY_PERFORMANCE_AND_RESEARCH.md`.
