# ChatBubble Historical Experiment Program Audit — 2026-08-30

**Repository:** `LUNCY579/ChatBubble`  
**Historical audit date:** 2026-08-30  
**Evidence snapshot:** `main` at `27f807f039fae5c5cd20934100690606b5f9c9e7`, plus then-current unmerged research branches  
**Purpose:** preserve the verified historical development record, including positive, negative, inconclusive, blocked, infrastructure-only, and shipped outcomes.

This dated directory is an immutable historical snapshot. Later experiments should update a newer audit rather than silently rewriting the evidentiary meaning of this record.

## Contents

1. [Executive audit and corrections](README.md)
2. [Foundational program and family reconstruction](01_FOUNDATIONAL_AND_FAMILY_HISTORY.md)
3. [Product science and validated mechanism work](02_PRODUCT_SCIENCE_AND_VALIDATED_MECHANISMS.md)
4. [Frontier research laboratories](03_FRONTIER_RESEARCH_LABS.md)
5. [Program synthesis and dependency map](04_PROGRAM_SYNTHESIS.md)
6. [Status register, documentation roadmap, and evidence map](05_STATUS_ROADMAP_AND_EVIDENCE_MAP.md)

---

## 1. Executive conclusion

The agent report is **substantially correct about the late-August research frontier**, especially its three most important judgments:

1. **TGSA is the most sophisticated not-yet-validated operator family in ChatBubble.**
2. **The Halo significance / KV-compression program is the most extensively instrumented mechanism study.**
3. **The things with actual product or family-vault standing are still much more conservative:** dense/cross-depth soups, LoRA/task-vector geometry, answer-blind routing/cascades, and abstention/readout interventions.

However, the supplied report should **not** be retained as the canonical “all experiments in ChatBubble” audit without revision. It has four kinds of problems:

- **Coverage:** it starts its narrative too late and compresses the foundational short-response program into one line. ChatBubble carries real, stop-gated experiment artifacts beginning with dense SFT, static selective-layer tuning, progressive layer deletion, targeted recovery, DPO, model soup, context compaction, sparse simulation/runtime, release candidacy, and student/RAG recovery.
- **Snapshot drift:** the live remote now has **9 branches**, not 13, and **4 TGSA branches**, not 6.
- **Status drift:** some branch conclusions were superseded by later commits. In particular, TGSA v4 rejected its competence-task family, and cross-scale v2 did run TIES/DARE-class merge variants even though v1 did not.
- **Evidence-label errors:** several “PASS” files mean “the experiment made a sound go/no-go decision,” not “the proposed mechanism worked.” Phase 7 sparse runtime is the clearest example: the stop gate passed because it correctly selected the dense fallback, while the sparse-runtime hypothesis itself was negative.

The corrected high-level program arc is:

> **short-response competence → 19-layer structural frontier → targeted recovery/alignment → model soup → context/sparsity deployment studies → named 28-layer family reconstruction → long-context weight geometry and routing → abstention/readout → 4B comparison → significance-native compression mechanisms → next-generation operator labs (PCSM, PSA, TGSA, cross-scale).**

---

## 2. Audit conventions

### Two clocks

ChatBubble itself was created on **2026-08-02**, but it contains imported experiment records with human decisions dated **2026-07-27** and **2026-07-28**. This audit therefore distinguishes:

- **Repository lifecycle date:** when ChatBubble existed as this Git repository.
- **Evidence date:** when an experiment record says the experiment or gate was decided.

### Status vocabulary

| Status | Meaning |
|---|---|
| **CONFIRMED POSITIVE** | A registered or reasonably controlled experiment supports the target claim within its stated scope. |
| **CONFIRMED NEGATIVE** | The tested mechanism/claim failed within the tested scope. |
| **INCONCLUSIVE** | The run completed but competence, power, manipulation, or causal interpretation was insufficient. |
| **NO-GO / BLOCKED** | A prerequisite gate failed; downstream confirmatory/sealed work correctly stayed closed. |
| **ENGINEERING / SYNTHETIC ONLY** | Implementation, gradients, schemas, resource behavior, or synthetic mathematics were validated, but not a real-model scientific claim. |
| **OPEN** | Experiment is designed or partially run, but the decisive treatment/result is not committed. |
| **SHIPPED / VAULTED** | The artifact or policy is part of the published family/product science, regardless of whether its origin was a mechanistic discovery. |
| **INFRASTRUCTURE** | Dataset, harness, preregistration, or deployment machinery rather than a completed scientific result. |

A **human PASS** or **stop-gate PASS** is not automatically a CONFIRMED POSITIVE mechanism result. It can mean “the correct decision was made and the pipeline may proceed.”

---

# 3. Material corrections to the supplied agent audit

| Topic | Agent report | Corrected audit |
|---|---|---|
| Current branch count | 13 branches | **9 total current branches**, including `main`. |
| Current TGSA branch count | Six `research/tgsa-*` branches | **Four TGSA branches** currently exist. |
| Program start | 2026-08-02 | Repo created Aug 2, but carried experiment decisions begin **Jul 27**. |
| Foundational experiments | Mostly collapsed into “short-response adapter consensus” | Dense SFT through Phase 9 are first-class experiment history and should be explicit. |
| Supa / 42-of-64 score | Implies the `.8125` Supa arm is 42/64 | **Supa Core 1.7 is 40/64, 121/192.** The separate **Dense Gate42** routed recipe is 42/64, 124/192. |
| Cross-scale v2 | “Advanced merge variants largely reproduced v1; Fisher and extra cosine runs incomplete” | v2 explicitly ran **selective consensus, TIES, DARE, and DARE-TIES**. Fisher and several localization/8B controls remained closed. |
| Cross-scale status | “sealed negative” | **Not sealed.** It is a strong development-bound NO-GO/negative; the sealed bank was never opened. |
| TGSA v4 | Implies MTP waits for a “v4 GO” | v4 is already **`V4_REGIME_FAMILY_REJECTED`**. Canonical promotion now requires a **new competence preregistration and an actual GO**, then Phase 2A and 2B. |
| TGSA MTP semantics | Treated broadly as one complementarity program | Current canonical contract explicitly separates **sequential trajectory-conditioned MTP (2A)** from **parallel independent-offset MTP (2B)**. They are not replications and cannot be pooled. |
| PSA | “Measure where heads look, then steer” | Measurement/calibration has advanced; **intervention efficacy remains open**. Baseline runs exist, but no decisive treatment-arm result is committed. |
| PCSM real pilot | “not coworker evidence” | Correct direction, but stronger wording is warranted: the committed decision says **resource-valid but scientifically non-evaluable; advancement gate not evaluated; cannot advance**. |
| Significance confirmatory | “Broad confirmatory not run” | True only for the broad *significance-specific* claim. A meaningful **structural pooling/sink confirmatory** was run; significance-specific specificity still failed. |
| Phase 9 | Omitted from frontier narrative | Student/RAG recovery remains **blocked**, notably by **0/150 procedure correctness** in the recorded real-RAG gate. |
| State-sufficiency work | Omitted | A corrected 48k-example state-sufficiency/sequence experiment package exists, but **real LoRA/QLoRA scientific comparison was not run** in the committed build. |

---
