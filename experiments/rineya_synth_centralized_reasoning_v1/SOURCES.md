# Sources and evidence boundaries

Checked on 2026-10-02. URLs below are provenance, not an automatic download list.

## Primary external inspiration

1. Pierre-Carl Langlais et al., *It's All Training: A Fully Synthetic Single-Stage
   Recipe for LLMs*, arXiv:2609.37891v1, submitted 2026-09-29.
   https://arxiv.org/abs/2609.37891v1
   https://arxiv.org/html/2609.37891v1
   The primary abstract describes grounded synthetic amplification and training
   models with a single-stage recipe. Our proposal instead adapts the data-design
   principle to an existing Rineya checkpoint. It does not reproduce that training
   setup or assert the paper demonstrates Rineya retention, skill composition,
   native long-context improvement, or centralized-runtime performance.

2. Author-released SYNTH dataset card.
   https://huggingface.co/datasets/PleIAs/SYNTH
   The current card exposes source/generator metadata and separate reasoning and
   answer fields. These motivate provenance-aware filtering and export boundaries.
   This package does not mirror/download the dataset or certify every row. Pin a
   dataset revision and inspect its complete licensing/provenance before reuse.

Benchmark numbers from the preceding conversation are not copied into this
registration as reproduced evidence. No paper result is a forecast for Rineya.
The independent-verifier design, R/S pilot, retention assays, and preservation
policy are Rineya proposals/requirements, not claims attributed to the paper.

## Inspected repository provenance

- CRS-001 branch tip: `7459b614bcb017ae4f0160b15e0ccd53473bb990`.
  [Root charter](../../RINEYA_CENTRALIZED_REASONING_SYSTEM.md),
  [original experiment](../rineya_centralized_reasoning_system_v1/EXPERIMENT_PLAN.md),
  and [root agent guidance](../../AGENTS.md).
- That tip's base tree: `9223d7b22e4bbbab867107a241c67ea7410a0ede`.
- Centralized MiMo/Titan source snapshot: `19551ddebc87c93161c303865ce6d0f130de0093`.
- `main` observed before this change: `b0ca54bdc1b69e2b19d7baeca2e3fdde3ff0a7b2`.

The source charter supplies CRS-001's durable complementary status; its experiment
plan supplies the matched-tool procedure-memory question and protected-bank
boundaries. Both remain unchanged. Broader Rineya track relationships are design
interfaces, not a fresh audit of every native run. Refresh relevant authoritative
receipts before implementation or launch; historical branch tips are not live
hardware or capability qualification.
