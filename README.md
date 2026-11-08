# Compact-LLM — Rineya small-model research

**Experimental research on useful capabilities, efficient adaptation, and reliable evaluation in compact language models.** The Rineya family primarily builds on Qwen3-0.6B (approximately 596 million parameters), with 19-layer and 28-layer derivatives, adapters, task-vector combinations, single-pass mechanisms, and routed systems.

> **Repository status (8 November 2026):** `main` is a preserved historical baseline. The broader, actively consolidated research history lives on [`research/post-training-mimo-centralized-v1`](https://github.com/LUNCY579/Compact-LLM/tree/research/post-training-mimo-centralized-v1). Research results below come from recorded experiments, not a claim of production readiness or generalized superiority. The project contains both positive and negative findings.

## Selected measured findings

| Question | Recorded result | What it does **not** establish |
| --- | --- | --- |
| Can a small dense core improve controlled long-context reading? | **Supa Core 1.7** scored **161/256**, compared with **113/256** for Nimbus 1.6, in the recorded same-protocol Long Context Demo (BF16, greedy, 4,096-token context, 24 output tokens). | This is a controlled evaluation bank, not broad unseen-task or production performance. |
| How much does routing add? | **Supa 2.0 System48** reached **176/256** versus the Supa dense core's 161/256. The frontier system uses **nine inference arms**, not one weight file. | Its extra computation, selection procedure, and latency must not be confused with a single-pass model improvement. |
| Can abstention behavior improve without multiple model passes? | The **S1 normalized gate** reached **130/192** on the recorded Stability split against **121/192** for the reference arm. The separate baked **Supa-A1** model scored **125/192** in one pass. | The S1 result is marginal, the splits have been inspected, and the baked checkpoint is not identical to the S1 processor. |
| Can a frozen small model learn a state-transition skill? | An adaptation arm scored **289/384 in three recorded seeds** on the process-transition probe, from a source below the registered shortcut floors. | Long-context and abstention retention regressions remained; the acquisition gate alone was inadequate. |
| Do arithmetic gains preserve previous behavior? | **Opallo** recorded **73/150** arithmetic and **41/47** retained source successes; its strict long-context result fell from **161/256 to 135/256**, and literal `UNKNOWN` responses from **13/64 to 0/64**. | There is **no established overall model-quality winner**; these are distinct measurements and must not be collapsed into one score. |

The results are intentionally presented with their limitations. In particular, the Showcase 64 was used for selection; Stability 192 was subsequently inspected; and the family-level sealed chat/RAG tests were not completed as a general release qualification.

**Primary evidence:**
- [Common-protocol family comparisons](comparison.md) and [model lineage](lineage.md).
- [Published long-context family search report](Rineya_Long_Context_Weight_Geometry_Lab_v1/PUBLISHED_FAMILY_SEARCH_REPORT.md).
- [Current model-quality decision memo](https://github.com/LUNCY579/Compact-LLM/blob/research/post-training-mimo-centralized-v1/MODEL_QUALITY_STATUS.md) — source-backed limitations, corrections, and remaining qualification gaps.
- [State-transition and retention continuation](https://github.com/LUNCY579/Compact-LLM/blob/research/post-training-mimo-centralized-v1/CONTINUATION.md) — acquisition, dose, rehearsal, and independent retention failures.
- [Multi-reasoning frozen-scout evidence](https://github.com/LUNCY579/Compact-LLM/blob/research/post-training-mimo-centralized-v1/research/multi_reasoning_scout_v1/CURRENT_CANDIDATE_RESULTS.md) — an instructive schema-smoke failure, not a positive benchmark claim.

## Research directions

1. **Efficient model composition:** kept-layer/weight interpolation, LoRA scaling, task-vector arithmetic, model soups, and separating dense single-pass cores from multi-arm routing.
2. **Capability acquisition with retention:** teaching new skills with small data budgets while measuring regressions in long-context retrieval, abstention, and prior task behavior.
3. **Measurement and evidence discipline:** independent split roles, preregistered floors, strict source hashes, reproducible inference protocols, negative-result preservation, and distinction between CPU fixtures and native checkpoint execution.
4. **Architecture and inference:** cross-scale transfer, recurrence, factorized or fast-weight memory, capture and serving qualification, and decision/reasoning experiments. These tracks have different readiness levels.

## Where to start

| Resource | Purpose |
| --- | --- |
| [`roadmap.md`](roadmap.md) | Historical published-core roster and release/validation gaps |
| [`comparison.md`](comparison.md) | Detailed same-protocol model/system scores |
| [`lineage.md`](lineage.md) | Weight ancestry, donors, adapter recipes, and names |
| [`announcement.md`](announcement.md) | Historical limits on public claims |
| [Integrated research branch](https://github.com/LUNCY579/Compact-LLM/tree/research/post-training-mimo-centralized-v1) | Consolidated later experiments and evidence |
| [Consolidation record](https://github.com/LUNCY579/Compact-LLM/tree/research/post-training-mimo-centralized-v1/docs/repository-consolidation/2026-10-22) | Preserved source histories, blockers, and manual branch-pruning safeguards |

This is a collection of **independent experimental packages**, not a single turnkey training command. Read an experiment's protocol, environment pins, data permissions, and test instructions before running it; a passing synthetic or CPU test does not validate native-GPU model quality.

## Artifacts and reproducibility

Published core names include **Pulse, Aegis, Halo, Nimbus, Lyra, Supa, Baya, and Supa-A1**. Model checkpoints and many run artifacts are not stored in the Git tree; `Rineya_Published_Cores/` weight directories are gitignored. A private Hugging Face family backup is recorded as `Luncy1978/rineya-published-cores` (including Pulse 1.6, Nimbus 1.6, routed systems, adapters, and Supa-A1); **private storage is not a public download link**. Supa-A1's recorded publication commit is `b14103c71c240998171694a0d0397bd4e5d64d71`.

For experiment reproduction, use the exact source revision, model/hash manifest, data split and tokenizer specified by that experiment. Do not treat one lab's generated statistics as independently confirmed results for another lab.