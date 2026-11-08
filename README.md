# Compact-LLM — Rineya small-model research

**Experimental research on useful capabilities, efficient adaptation, and reliable evaluation in compact language models.** The Rineya family primarily builds on Qwen3-0.6B (approximately 596 million parameters), with 19-layer and 28-layer derivatives, adapters, task-vector combinations, single-pass mechanisms, and routed systems.

> **Repository status (8 November 2026):** `main` now contains the **integrated research history** formerly concentrated on `research/post-training-mimo-centralized-v1`; a pre-consolidation snapshot is retained at `archive/main-preconsolidation-20261108`. Distinct unresolved experiment branches and blocked PRs remain separate. Research results below come from recorded experiments, not a claim of production readiness or generalized superiority. The project contains both positive and negative findings.

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
- [Current model-quality decision memo](MODEL_QUALITY_STATUS.md) — source-backed limitations, corrections, and remaining qualification gaps.
- [State-transition and retention continuation](CONTINUATION.md) — acquisition, dose, rehearsal, and independent retention failures.
- [Multi-reasoning frozen-scout evidence](research/multi_reasoning_scout_v1/CURRENT_CANDIDATE_RESULTS.md) — an instructive schema-smoke failure, not a positive benchmark claim.

## Rineya Titan-series: acquisition, sparsity and composition

These are **Rineya's Titan experimental checkpoints and task vectors**, not claims about an unrelated architecture named Titans. All numbers below are tied to the listed experiment, split, checkpoint precision and evaluation protocol.

| Study | Recorded evidence | Interpretation / limit |
| --- | --- | --- |
| **Titan dense / transition skill** | The FP32 trained endpoint acquired **235/384** on the registered skill probe, while the stored BF16 Titan scored **234/384**; the published checkpoint's long-context readout was **162**, versus **157** for the unrounded endpoint. The adaptation used **54,172 processed tokens** and **28 optimizer updates** on 220 tensors in blocks 8–27. | Precision changes the update and can change individual outputs. This is a bounded task-specific result, not a general benchmark win. |
| **Titan-Sparse** | Keeping the top **25%** of update magnitudes in the eligible 314.6M-parameter footprint preserved **232/384** acquisition and **41/47** source arithmetic successes, but lost **7** previously correct long-context rows. | Sparse updates preserve much of the measured skill, **not** all source behavior; acquisition and retention need separate gates. |
| **BF16 export effect** | The recorded BF16 census erased **60.61%** of nonzero dense update elements, including all updates in **24/220** trained tensors. **28.5%** of the sparse update's nonzeros lie where the BF16-stored dense update is zero. | Stored Titan is **not** a superset of Titan-Sparse; reconstructing an FP32 delta from BF16-stored weights loses information. |
| **Titan-Sparse / Opallo geometry** | Global cosine **≈0.0025**, yet same-layer input-subspace collision median **3.15×** a Haar-random baseline. Strength-matched composition records **235→223** from attenuation and **223→219** with the second donor (the latter paired p=0.45). | Nearly orthogonal flattened vectors are not disjoint functional machinery. Half-strength averaging can cost capability even when interference is small. |
| **Titan model-soup mini** | Five 50/50 FP32 interpolation artifacts were materialized and independently rederived bitwise; **2,980,249,600** output-element comparisons matched the implemented formula. | **No native model scoring or quality validation** was performed on these five soups. Mechanical correctness is not model-quality evidence. |
| **Titan TriSkill — C donor** | The independently trained C donor gained **40/192 net semantic-correct rows** against Supa (world-level sign-test p≈0.00017). Every tested TT+C coefficient gained C content but failed T retention: TT **444/864**, versus **428**, **425** and **360** at β=0.25, 0.50 and 1.00. | **Registered result: STOP_NO_ELIGIBLE_BETA.** The 8-row T-loss ceiling was missed by every candidate; Stage B banks were not opened. This is a valuable negative result, not a three-skill success. |
| **Post-stop direction control** | A single per-tensor norm-matched random addition to TT at β=0.25 left T **444/864**, C **37/192** and H **86/112** unchanged, whereas learned δ_C at the same update norm reduced T to **428/864**. | A directional effect is plausible, but one exploratory post-outcome control is **not** causal or held-out proof. |

**Additional DMRF and tangent-space findings:** A [matched-norm N16 control](experiments/delta_multi_reasoning_v1/DMRF001_RESULTS.md) achieved **236/384** skill acquisition versus **235/384** for the rank-16 approximation, with both retaining **39/47** previously correct arithmetic answers. The reported 7.26M-parameter, 27.7 MB factored update is a *storage representation*, not a validated 27.7 MB inference checkpoint: the deployed dense FP32 artifact is approximately 2.38 GB. A [sealed abstention test](experiments/delta_multi_reasoning_v1/DMRF001_DECISION.md) recorded a significant **40/128 → 25/128** decline for the sparse Titan endpoint (0 gains, 15 losses); acquisition is not safe retention. The [Titan tangent-space method port](experiments/titan_tangent_arithmetic_v1/README.md) implements a frozen-anchor Jacobian-vector-product comparison of ordinary, post-hoc linearized and tangent-trained deltas, but **has not run on pretrained weights**. Together these suggest separating update magnitude, storage compression, nonlinear functional response and preservation measurements.

**Research implications.** Preserve both FP32 and exported checkpoint identities; record per-task *gains and previously correct rows lost*; choose independent donor coefficients rather than assuming checkpoint averaging is lossless; and measure same-layer subspaces and actual generation behavior, not cosine alone. Separate semantic correctness from formatting conventions. The TriSkill failure is a reason to examine causal interference and retention before proposing another mixture.

**Additional historical Titan lanes:** [geometric pilot](experiments/triskill_geometric_pilot_v1/REPORT_GAPFIX.md) recorded **67 CPU tests passing** with **zero native model forwards**, and [PARENTQ](experiments/triskill_geo001_parentq_v1/README.md) remained unfrozen and unqualified. The [Titan quality-repair preparation](experiments/titan_quality_repair_v1/README.md), [opposite-direction diagnostics](experiments/titan_quality_repair_diagnostics_v1/README.md) and [byte-bound offline evidence archive](experiments/research_evidence_20261005/README.md) are now preserved in `main` as historical research, **not** evidence of new model gains or native qualification.

**Source records:** [Titan soup evidence](experiments/titan_model_soup_mini_v1/RESULTS_PHASE_A.md), [sparse/Opallo geometry](experiments/delta_subspace_geometry_v1/RESULTS.md), [Titan-Sparse + Opallo pilot](experiments/titan_opallo_skill_vector_v1/RESULTS_PRE_NATIVE.md), [TriSkill Phase 0](experiments/titan_third_skill_v1/PHASE0_RESULT.md), [TriSkill Stage A and post-stop control](experiments/titan_third_skill_v1/RESULTS_STAGE_A.md), and [Titan offline-only quality archive](research/titan_joint_quality_offline_v1/README.md).

## November 2026 additive research imports

The [bulk import source manifest](docs/repository-consolidation/2026-11-08/BULK_IMPORT.md) records 14 exact non-overlapping experiment/research subtrees that were missing from this branch. They include the Fast-Weight Attention lab, TGSA and retrofit programmes, adaptive-compute experiments, Halo/positional experiments, and the MRS-2 reasoning successor. This is a **structural preservation import**, not fresh model qualification, and does **not** resolve outstanding shared-path branch conflicts or blocked scientific gates.

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
| [Integrated main research tree](https://github.com/LUNCY579/Compact-LLM/tree/main/experiments) | Consolidated later experiments and evidence |
| [2026-11-08 branch-tip audit](docs/repository-consolidation/2026-11-08/README.md) | Live-branch ancestry, safe-merge boundaries, and remaining work |
| [Consolidation record](https://github.com/LUNCY579/Compact-LLM/tree/main/docs/repository-consolidation/2026-10-22) | Preserved source histories, blockers, and manual branch-pruning safeguards |

This is a collection of **independent experimental packages**, not a single turnkey training command. Read an experiment's protocol, environment pins, data permissions, and test instructions before running it; a passing synthetic or CPU test does not validate native-GPU model quality.

## Artifacts and reproducibility

Published core names include **Pulse, Aegis, Halo, Nimbus, Lyra, Supa, Baya, and Supa-A1**. Model checkpoints and many run artifacts are not stored in the Git tree; `Rineya_Published_Cores/` weight directories are gitignored. A private Hugging Face family backup is recorded as `Luncy1978/rineya-published-cores` (including Pulse 1.6, Nimbus 1.6, routed systems, adapters, and Supa-A1); **private storage is not a public download link**. Supa-A1's recorded publication commit is `b14103c71c240998171694a0d0397bd4e5d64d71`.

For experiment reproduction, use the exact source revision, model/hash manifest, data split and tokenizer specified by that experiment. Do not treat one lab's generated statistics as independently confirmed results for another lab.