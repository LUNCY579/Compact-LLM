# Navigation

[Audit index](README.md) · [Foundational/family](01_FOUNDATIONAL_AND_FAMILY_HISTORY.md) · [Product science](02_PRODUCT_SCIENCE_AND_VALIDATED_MECHANISMS.md) · [Frontier labs](03_FRONTIER_RESEARCH_LABS.md) · [Synthesis](04_PROGRAM_SYNTHESIS.md) · [Status/evidence](05_STATUS_ROADMAP_AND_EVIDENCE_MAP.md)

---

# 4. Master chronology

## 4.1 Foundational short-response and structural program

| Evidence date | Experiment | Core question | Result | Scientific status |
|---|---|---|---|---|
| **Jul 27** | Phase 1 Dense SFT | Can full trainable adaptation impose the short-response contract? | Heuristic score **0.804 → 0.995**; median output **16 → 9 tokens**; completion 1.0. Human PASS. | **CONFIRMED POSITIVE** for the task contract. |
| **Jul 28** | Phase 2A static selective-layer tuning | Can only 1/2/4 selected layers match dense adaptation? | All candidates failed the quality/resource gate; repeated-seed stability passed. | **CONFIRMED NEGATIVE** for this static selective-layer strategy. |
| **Aug 2** | Phase 2B progressive depth reduction | How far can the 28-layer model be shortened before value turns negative? | Best retained frontier: **19 decoder layers**, 9 dropped; reported cumulative weight-memory gain **23.75%** vs quality opportunity loss **11.24%**. | **CONFIRMED POSITIVE** for a 19-layer structural frontier under this protocol. |
| **Aug 4–5** | Phase 2C targeted recovery | Can importance-guided tuning recover the 19-layer student cheaply? | `all_remaining` highest quality; `top1` achieved ~94.7% trainable reduction; `top4_mlp` attractive balanced point. Human PASS. | **CONFIRMED POSITIVE** for targeted recovery. |
| **Aug 5** | Phase 3 DPO | Does preference alignment improve the recovered model? | Small but positive score movement; stop gate passed. | **CONFIRMED POSITIVE, SMALL EFFECT**. |
| **Aug 5** | Phase 4 two-seed greedy model soup | Can cheap weight averaging improve robustness/quality? | Uniform soup of accepted seeds passed; mean reported quality ~0.7013 and no hard/critical failures. | **CONFIRMED POSITIVE** in-program; foundational precedent for later family soups. |
| **Aug 5** | Phase 4B post-alignment structural reduction | Is more structural compression worthwhile after alignment? | No candidate had positive value; dense reference retained. | **CONFIRMED NEGATIVE** for additional compression. |
| **Aug 5** | Phase 5 context compaction | Can context/memory tokens be reduced without measured quality loss? | ~**29.1% token reduction** with complete measured recall and no reported quality opportunity loss. | **CONFIRMED POSITIVE** within the harness. |
| **Aug 5** | Phase 6 sparse-compute simulation | Does a simulated sparse policy look promising? | 50% active-ratio simulation showed positive modeled margin. | **ENGINEERING / SIMULATION POSITIVE ONLY**. |
| **Aug 5** | Phase 7 sparse runtime | Does the sparse idea survive a real runtime test? | Sparse margin negative (~−0.0778); no measured latency/VRAM benefit sufficient to offset quality loss; **dense runtime selected**. | **CONFIRMED NEGATIVE** for sparse runtime; pipeline decision PASS. |
| **Aug 5** | Phase 8 release candidate | Is the dense path ready as RC? | `1.0.0-rc1`, no blockers in gate; dynamic sparse excluded. | **SHIPPED / RELEASE DECISION**. |
| later Aug | Phase 9 student distillation + grounded recovery | Can a much smaller student preserve generic behavior and gain grounded RAG capability? | Some promising Oracle/faithfulness/abstention signals, but no production candidate; real-RAG **procedure correctness 0/150**. | **BLOCKED / INCOMPLETE**. |

### What this establishes

The early program already discovered the pattern that later defines ChatBubble:

- broad adaptation works;
- naïve local/selective structure can fail;
- **structural reduction needs recovery**;
- cheap parameter-space combination can be useful;
- simulated efficiency gains must survive real runtime measurement;
- deployment decisions are separated from scientific mechanism claims.

That is the historical foundation for the later Rineya family vault.

---

## 4.2 Family construction, reconstruction, and soup geometry

### Teacher Soup Mini

**Mechanism.** Reconstruct LoRA deltas in FP32 and average teacher/adaptation deltas rather than averaging opaque adapter files.

**Observed result.** As a direct 19-layer teacher model, the recipes were weak on the long-context Showcase: the strongest reported rows were only around **5/64**.

**Interpretation.** This is **not evidence that teacher soup alone is a strong model**. Its importance is methodological: the reconstructed deltas later became donors for cross-depth stitching, where the geometry became useful after restoration into a 28-layer Qwen scaffold.

**Status:** **MECHANISTIC/RECIPE CONTRIBUTION; DIRECT MODEL WEAK.**

---

### Qwen3 Cross-Depth Soup Mini

**Mechanism.**

- Start from the official 28-layer Qwen3 scaffold.
- Blend only the **19 Phase-2B retained blocks** with 19-layer Rineya-family donors.
- Keep the nine previously dropped Qwen layers official.
- Keep embeddings, final norm, and LM head on the official scaffold.

**Representative results.**

- Nimbus 50/50: **24/64**
- Lyra 50/50: **28/64**
- Halo 50/50: **30/64**
- Full Halo-donor analogue: **31/64**
- `late_rineya`: **31/64**

**Interpretation.** This is one of the key transitions from “compressed student” to “family vault”: capability is not simply in a standalone 19-layer donor. Restoring an official 28-layer scaffold while injecting selected donor geometry can substantially improve long-context behavior.

**Status:** **CONFIRMED POSITIVE / FAMILY-CONSTRUCTION MECHANISM.**

---

### R2 Cross-Depth Soup Mini

**Mechanism.** Port the same donor/stitching idea onto the R2 32K scaffold to test whether the geometry is scaffold-specific.

**Representative committed scoreboard.**

- R2 Base: **11/64**, Oracle 1.0, dual composite ~0.7463.
- `r2_x_aegis`: **21/64**, Oracle 1.0, strongest dual composite ~**0.7729**.
- Teacher-donor variants: roughly **20–21/64**.
- `late_rineya`: **25/64**, Oracle ~0.9.
- Halo-donor α=1.0 analogue: **27/64**, but Oracle abstention collapsed to **0.0**.

**Interpretation.** Donor geometry has some cross-scaffold utility, but extreme blends can trade away abstention. This is useful evidence against treating Showcase accuracy as the only family objective.

**Status:** **POSITIVE CROSS-SCAFFOLD SIGNAL, NOT VAULTED.**

---

### Layerwise Soup Lab

**Mechanism.** Compare global, uniform, and layer-adaptive soup coefficients; later ported to the 28-layer Qwen3/Halo-U pair. Coefficients are selected on DEV before descriptive Showcase evaluation.

**Important family outcome.** The Qwen3/Halo-U global coefficient **α = 0.40** became the basis for **Nimbus 1.6**.

**Evidence limitation.** The sealed TEST for the relevant Halo pair remained off. This is therefore strong product-selection evidence, not a universal sealed generalization claim.

**Status:** **SHIPPED RECIPE / DEVELOPMENT-SELECTED.**

---

### Named family vault

The family construction program ultimately produced the public named cores and variants, including:

- Pulse
- Aegis
- Halo
- Lyra
- Nimbus
- Halo Uncensored
- Supa 1.7
- Baya 1.7
- Supa-A1

The important conceptual point is that the family is **not** a single monotonic training trajectory. It is a collection of reconstructed/stiched/merged cores and inference policies with different long-context, abstention, and style tradeoffs.

**Status:** **SHIPPED / VAULTED.**

---
