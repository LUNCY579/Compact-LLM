# Navigation

[Audit index](README.md) · [Foundational/family](01_FOUNDATIONAL_AND_FAMILY_HISTORY.md) · [Product science](02_PRODUCT_SCIENCE_AND_VALIDATED_MECHANISMS.md) · [Frontier labs](03_FRONTIER_RESEARCH_LABS.md) · [Synthesis](04_PROGRAM_SYNTHESIS.md) · [Status/evidence](05_STATUS_ROADMAP_AND_EVIDENCE_MAP.md)

---

# 8. State sufficiency / sequence-independence program

**Corrected lab:** `Sequence_Experiment/Rineya_State_Sufficiency_Lab_v1.0.1_CORRECTED`

**Question.** Is a compact current reasoning state sufficient for the next correct decision, or does the model need the full historical trajectory?

**Experiment design.**

- FULL-history adapter versus SPLIT/current-state adapter.
- Source sizes roughly:
  - TRAIN 43,200
  - DEV 2,400
  - TEST 2,400
  - ~189,922 transitions
- Corrected build replaced an invalid prior archive.
- Test harness passed and a deterministic reference smoke showed equivalence.

**Critical limitation.** The real LoRA/QLoRA adapters were **not trained in the committed build environment**. The deterministic smoke is a pipeline/reference check, not the scientific answer.

**Status:** **INFRASTRUCTURE READY; HYPOTHESIS UNTESTED IN COMMITTED REAL-MODEL EVIDENCE.**

---

# 9. PCSM: Peer / Coworker Stabilizer Matrix

**Branch:** `codex/exp-peer-stabilizer-matrix`

**Question.** Can other layers provide sparse “coworker” signals that regulate a target layer’s native-weight update more effectively than ordinary sparse tuning?

## What is established

- **143/143 tests** passed in the inspected qualification.
- Synthetic mathematical experiments show peer structure can beat selected controls.
- Software, schema, masking, and update machinery are functional.

## What is not established

The synthetic stabilizer ablation confidence intervals do not independently prove the stabilizer component itself.

The committed real-model Phase-C decision is explicit:

- `VALID_RESOURCE_PILOT`
- `pilot_scientifically_evaluable = false`
- `scientific_advance_gate = not_evaluated`
- `pilot_can_advance = false`

The inspected real pilot also showed the multiplicative stabilizer scales remaining at 1.0 in the relevant layers, so it cannot serve as evidence that a learned coworker regulation effect occurred.

**Status:** **ENGINEERING + SYNTHETIC FEASIBILITY; NO REAL COWORKER-MECHANISM EVIDENCE; PHASE D BLOCKED.**

---

# 10. PSA: Positional Sequential Attention

**Branch:** `research/positional-sequential-attention`

This is a sibling line to TGSA, focused first on **measurement of positional/head behavior** and only later on intervention.

## Measurement findings

The current branch contains a substantial calibration program. Among its useful internal conclusions:

- coarse answer accuracy is often too insensitive for positional measurement; gold NLL is more diagnostic;
- the original multi-hop setup was not sufficiently measurable, while guided variants were;
- positional deficit appears family-shaped;
- additive absolute-position plus distance features explain substantially more variance than either alone in the reported regression;
- Supa shows a different edge/middle attention distribution than the baseline in content-light settings;
- low-rank structure explains much of the measured causal-significance / positional-bias variation;
- weight-delta measurements predict positional bias better than they predict causal significance.

## Retractions / corrections

The branch also records several corrections, including:

- an article/format artifact,
- an evidence-position issue,
- a pooling confound,
- a position-axis mistake,
- a sign inversion.

That correction history is a strength of the evidence ledger, but it means earlier intermediate claims should not be quoted without their retractions.

## Intervention status

The Phase-8 intervention screen has baseline work, but the audited branch does **not** yet contain a decisive treatment-arm result that establishes steering efficacy. Later sequential/factorial/preservation/sealed phases remain downstream.

**Status:** **MEASUREMENT PROGRAM ADVANCED; STEERING MECHANISM OPEN; SEALED UNREAD.**

---

# 11. TGSA: Trajectory-Guided Sequential Attention

TGSA is the **most sophisticated unused operator** in ChatBubble, but it is also the track most in need of careful evidence wording.

## 11.1 Operator idea

Sequential attention heads/steps update a carrier state; finite-difference “trajectory” information (velocity/change) modulates the next update. The broader program also studies complementarity with multi-token prediction (MTP).

The code and ablation surface are advanced. That does **not** mean the scientific hypothesis has been validated.

## 11.2 Phase 1 v3 confirmatory

- 3 arms × 20 seeds = 60 runs.
- Evidence chain/registered machinery completed.
- Registered H1 contrasts were near zero.
- All arms were at or below a trivial-constant regime.

The postmortem found roughly:

- seen/train competence ~0.236
- holdout ~0.045

This is a **memorization / failed-generalization task regime**.

**Verdict:** **INCONCLUSIVE**, not a TGSA negative. The manipulation/task did not make a fair mechanism comparison possible.

## 11.3 Phase 1 v4 competence-first rerun

v4 tested multiple task families before allowing a mechanism claim.

Approximate behavior recorded:

- `direct_lookup`: inert / low competence
- `pointer_chase`: inert / low competence
- `state_update`: learns strongly, rising from very low initial performance to ~0.98

Because only one family learned, the registered task-band verdict is:

**`V4_REGIME_FAMILY_REJECTED`**

This is **not a refutation of TGSA**. It is a rejection of the competence regime intended to test TGSA.

## 11.4 Phase 2 TGSA × MTP

The current canonical path distinguishes two different causal experiments:

### Phase 2A — sequential MTP

Teacher-forced / recursive / trajectory-conditioned multi-token prediction. Later predictions depend on an evolving trajectory.

### Phase 2B — parallel MTP

Independent offset predictions in parallel.

These are **not replications**. Their causal graphs differ, and current documentation explicitly prohibits pooling or averaging them as if they estimated one effect.

Existing old-band Phase-2 runs can support claims about:

- mechanics,
- gradients,
- schema,
- execution.

They **cannot** be promoted into claims about:

- competence,
- TGSA mechanism efficacy,
- quality,
- complementarity,
- speed.

## 11.5 Canonical next gate

The report supplied by the agent says MTP cannot go confirmatory “without a v4 GO.” That is now stale wording.

v4 has already been rejected. The current canonical sequence is:

> **new competence preregistration → actual competence GO → Phase 2A sequential MTP → Phase 2B parallel MTP → only then consider higher-level TGSA/MTP complementarity claims.**

The pretrained Rineya retrofit branch is explicitly useful for integration rehearsal, not causal confirmation.

**Status:** **MOST ADVANCED UNUSED OPERATOR; SCIENTIFIC HYPOTHESIS NEITHER CONFIRMED NOR REFUTED; CURRENT COMPETENCE REGIME REJECTED.**

---

# 12. Cross-scale 4B/8B → 0.6B capability transfer

**Branch:** `research/rineya-cross-scale-qwen4b-8b`

This is the most ambitious parameter-space transfer program and the one with the strongest current **NO-GO**.

## 12.1 v1

### Structural findings

Several projection families were explored:

- direct truncation,
- normalization,
- ridge/Procrustes-like maps,
- target-basis spectral projection.

A target-basis low-rank spectral map (e.g. rank 8, low strength) was structurally viable.

### Donor agreement

The 4B and 8B donor/task deltas aligned strongly in direction:

- delta cosine approximately **0.932**
- sign agreement approximately **0.926**

This is important: the donor signal itself is not random.

### Behavioral result

A mean-consensus candidate showed only a tiny MERGEDEV movement and tied on the broader DEV-FULL surface. It did not become a finalist. The sealed bank stayed closed.

**v1 status:** **DEVELOPMENT NO-GO.**

## 12.2 v2 exploratory follow-up

This is where the supplied audit is stale.

v2 did run selected advanced merge variants:

- selective consensus,
- **TIES**,
- **DARE**,
- **DARE-TIES**.

On the fast MERGEDEV slice, the best advanced variants largely reproduced the same consensus behavior; none beat both relevant controls robustly.

A coherent 4B control on P5 also failed to create the missing transfer gain.

Not completed / correctly closed included:

- Fisher,
- full localization maps,
- some 8B P5/CKA controls,
- v2 DEV-FULL escalation,
- sealed evaluation,
- LoRA follow-on.

## 12.3 Correct interpretation

The important v2 update is not merely “more merge methods failed.”

It narrows the likely failure boundary:

> Because more sophisticated merge interference controls did not rescue behavior, the missing ingredient is increasingly likely to be the **cross-scale projection/operator \(T\)** or the assumption that a donor delta can be transported into the 0.6B target basis in this way—not simply naïve averaging interference.

**Critical terminology correction:** This is **not a sealed negative**. The sealed evaluation was never opened.

**Status:** **STRONG DEVELOPMENT-BOUND NEGATIVE / NO-GO; SEALED CLOSED.**

---

# 13. Cloud demo / deployment evidence

**Branch:** `product/rineya-family-cloud-demo-secure-delivery`

This branch should not be counted as an accuracy experiment.

It implements:

- stateless streaming demo behavior,
- bounded waiting-room / overload handling,
- security controls,
- deployment benchmarking,
- Oracle A1 conservative resource profile.

The accuracy research contract is intentionally separated from deployment. At the audited point, security and benchmark tooling existed, while a committed first real-host benchmark result was not part of the core `main` science record.

**Status:** **PRODUCT / DEPLOYMENT INFRASTRUCTURE, NOT AN ACCURACY LAB.**

---
