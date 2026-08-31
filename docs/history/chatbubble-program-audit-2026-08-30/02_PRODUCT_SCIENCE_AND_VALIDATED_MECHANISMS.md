# Navigation

[Audit index](README.md) · [Foundational/family](01_FOUNDATIONAL_AND_FAMILY_HISTORY.md) · [Product science](02_PRODUCT_SCIENCE_AND_VALIDATED_MECHANISMS.md) · [Frontier labs](03_FRONTIER_RESEARCH_LABS.md) · [Synthesis](04_PROGRAM_SYNTHESIS.md) · [Status/evidence](05_STATUS_ROADMAP_AND_EVIDENCE_MAP.md)

---

# 5. Long-context capability and routing program

## 5.1 Long-Context Weight Geometry Lab

**Question.** Once multiple competent family cores and Stage-A LoRA checkpoints exist, can cheap parameter-space operations and answer-blind routing move the capability frontier?

**Main mechanisms tested.**

- LoRA scale sweeps.
- Negative task-vector extrapolation.
- Per-anchor geometry.
- Answer-blind cascades.
- Dense two-arm gates.
- PEFT/routing policies.

**Key outcomes.**

Published dense cores:

| System | Showcase /64 | Stability /192 | Combined /256 |
|---|---:|---:|---:|
| Supa Core 1.7 | **40** | **121** | **161** |
| Baya Core 1.7 | **40** | **112** | **152** |
| Supa-A1 | **43** | **125** | **168** |

Published/research routing systems:

| System | Showcase /64 | Stability /192 | Combined /256 |
|---|---:|---:|---:|
| Dense Gate42 | **42** | **124** | **166** |
| DualScale Gate45 | **45** | **124** | **169** |
| System46 | **46** | **125** | **171** |
| System48 | **48** | **128** | **176** |
| S1 research gate | **47** | **130** | **177** |

**Important score correction.** The `.8125` Supa dense core is **40/64, 121/192**, not 42/64. **42/64 belongs to the separate Supa–Baya Dense Gate42 recipe.**

**Negative-vector finding.** Pulse benefited from extrapolating roughly beyond one checkpoint delta and subtracting a later delta, e.g. `1.25Δ200 − 0.25Δ250`; stronger extrapolation was also used in final cascade construction.

**Evidence shape.**

- Large development search surface: roughly 147 complete 64-row reports.
- The family/routing system clearly improves the program scoreboard.
- These are **not all one dense model**.
- Showcase was used as a selection surface; Stability was frozen.
- Methods such as Fisher/TIES/DARE were not the core completed methods in this particular lab.

**Status:** **CONFIRMED PRODUCT-SCIENCE POSITIVE; SHIPPED ROUTING/GEOMETRY.**

---

## 5.2 Abstention Gate Lab → Supa-A1

**Question.** Is part of the remaining error due to a readout decision—knowing when to emit `UNKNOWN`—rather than missing internal information?

**S1.** Use normalized first-token probability of `UNKNOWN` as an answer-blind gate.

- Research S1: **47/64 Showcase, 130/192 Stability**.
- Nominal McNemar p≈0.049 against `nimbus_low` in the BF16 run.
- More conservative durable claim: approximately **+7 Stability points** is repeatable; do not make the p-value the headline.

**S4 / Supa-A1.** Bake a constant logit offset/custom-code policy into Supa behavior without changing the underlying Supa weight bytes.

- Supa-A1: **43/64, 125/192, 168/256**.
- A float16 follow-up recovered additional Showcase points in the relevant implementation history.

**Interpretation.** This is one of the cleanest causal stories in the repository: some errors are decision/readout failures and can be improved without changing the dense core weights.

**Status:** **CONFIRMED POSITIVE / SHIPPED.**

---

## 5.3 Invisible Limit Lab: Rineya vs Qwen3-4B

**Question.** Does the optimized 0.6B-family system genuinely beat a 4B model uniformly, or is the apparent tie a composition effect?

**Sealed TEST, 1,024 examples.**

- Rineya-frozen: **721**
- Qwen3-4B: **720**
- Difference: **+1**
- p = **1.0**
- Confidence interval spans a large negative-to-positive range.

By family:

- Unanswerable: **Rineya +50**
- Paraphrase: **Rineya −22**
- Counterfactual: **Rineya −21**
- Other/base differences do not create a uniform win.

**Primary verdict.** The registered “uniform win across all four families” hypothesis is **not established**.

**Probe result.** A hidden-state linear probe reached approximately **AUC 0.999** on DEV for answerability-like information and produced a large improvement in the relevant frozen evaluation.

**Interpretation.**

- The aggregate 721-vs-720 tie is **not evidence of uniform 0.6B reasoning superiority**.
- Numerically, a large unanswerable/readout advantage compensates for deficits on paraphrase and counterfactual behavior.
- The comparison is also not a clean general-capacity contest: Rineya had task-specific LCR SFT, while the 4B baseline was zero-shot.

**Status:** **CONFIRMED COMPOSITION RESULT; PRIMARY HYPOTHESIS NOT ESTABLISHED.**

---

# 6. Significance-native / KV-compression program on Halo

This is the repository’s **most empirically studied mechanism family**.

## 6.1 What was being tested

The lab asks whether a learned/derived **significance signal** can improve long-context retention under KV-like compression beyond simple structural controls such as:

- recency/age,
- deterministic pooling,
- sinks,
- budget allocation,
- shuffled/random significance.

The program is unusually well instrumented: claims ledger, parity checks, multiple cores, oracle analysis, pooling sweeps, structural confirmations, and negative retractions.

## 6.2 Robust results

### Compression deficit is real

Across four cores, the uncompressed baseline B0 is approximately **28–31/64**, while the compressed B3 condition is about **18–21/64**. The loss is roughly 10–13 questions.

**Status:** **CONFIRMED POSITIVE** for the existence of a compression deficit.

### Selection capacity exists

The corrected oracle-v2 analysis found a **13-answer selection gap**, 13 wins / 0 losses, p≈0.0156.

This says there are cases where choosing the right retained material matters. It does **not** by itself prove that the implemented significance score is that selector.

**Status:** **CONFIRMED POSITIVE** for oracle selection opportunity.

### Span pooling is directionally useful

Representative Showcase sweep:

- POOL1 / B3-like: ~19/64
- width 3: ~21/64
- width 5: ~25/64
- width 7: ~26/64

A broader structural confirmatory later showed a pooling+sinks configuration could match the uncompressed score in the reported 192-example surface while using a lower persistent-byte budget.

**Status:** **CONFIRMED STRUCTURAL POSITIVE.**

## 6.3 What failed

The significance-specific claim did **not** survive controls cleanly.

A later controlled result reported approximately:

- real S/C: **86/192**
- absolute-position control: **81/192**
- shuffled: **71/192**
- random: **62/192**

But the registered real-vs-shuffled cluster comparison was not statistically convincing (reported p≈0.210).

Additionally, parts of the apparent signal are recoverable from deterministic age/position under the fixed schedule.

**Durable conclusion.**

> The lab establishes that **compression structure, recency, pooling, sinks, and budget matter**. It does **not** establish that the specific significance score S is the causal source of a quality gain.

This directly supports the agent report’s central interpretation, with one qualification: **a structural pooling/sink confirmatory was run**; what remains unconfirmed is the broad significance-specific mechanism.

**Status:**  
- Compression deficit: **CONFIRMED**  
- Structural pooling/sinks: **CONFIRMED**  
- Significance-specific causal quality advantage: **NOT ESTABLISHED / NEGATIVE UNDER KEY CONTROL**

---

# 7. Short-response adapter consensus mechanism lab

This track deserves more than “Phase 2 complete.”

**Corestone evaluation:** 4,050 examples.

**Mechanism.** Combine several specialist adapter policies using hard consensus / probability aggregation / abstention specialists, with frozen-then-sealed discipline.

**Representative Phase-2 findings.**

- Hard K=7 TEST accuracy reported around **17.259%** on its cornerstone.
- Hard policy gained many abstention successes while losing answerable cases.
- Compared with probability aggregation, hard consensus showed both substantial rescues and harms.
- On a cross-corpus transfer evaluation, the hard policy performed very poorly relative to raw probability and the abstention specialist.

**Durable interpretation.**

This is **not a universal theorem that hard consensus is superior**. The mechanism report attributes the local success to:

- benchmark-specific policy composition,
- directional aggregation,
- abstention behavior.

It also shows adaptive consultation can approach hard-policy DEV behavior with materially fewer expert forwards.

**Status:** **CONFIRMED LOCAL POLICY-COMPOSITION EFFECT; GENERAL TRANSFER NEGATIVE.**

---
