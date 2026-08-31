# Navigation

[Audit index](README.md) · [Foundational/family](01_FOUNDATIONAL_AND_FAMILY_HISTORY.md) · [Product science](02_PRODUCT_SCIENCE_AND_VALIDATED_MECHANISMS.md) · [Frontier labs](03_FRONTIER_RESEARCH_LABS.md) · [Synthesis](04_PROGRAM_SYNTHESIS.md) · [Status/evidence](05_STATUS_ROADMAP_AND_EVIDENCE_MAP.md)

---

# 18. Recommended canonical status labels for the repository

To prevent future audit drift, each major experiment should expose one machine-readable or highly visible status with both a **pipeline decision** and a **scientific conclusion**.

Recommended format:

```yaml
experiment: phase_7_sparse_runtime
pipeline_decision: PASS
scientific_conclusion: NEGATIVE
claim: "Sparse runtime does not beat the dense fallback under the registered resource/quality gate."
sealed_status: not_applicable
supersedes: phase_6_sparse_simulation_for_runtime_claims
```

This avoids the common ambiguity where “PASS” is interpreted as “the mechanism worked.”

The same convention would be particularly valuable for:

- TGSA v3/v4/Phase2
- PSA Phase8+
- cross-scale v1/v2
- significance N-series results
- PCSM Phase C/D
- Invisible Limit Stage 8

---

# 19. Documentation repairs recommended on `main`

## Priority 1 — create one canonical program index

A top-level `EXPERIMENT_PROGRAM_AUDIT.md` should contain:

- experiment ID,
- date,
- branch,
- commit,
- hypothesis,
- primary metric,
- status,
- sealed status,
- superseded-by link,
- final-report path.

This document can be generated from a small YAML/JSON registry to avoid manual drift.

## Priority 2 — fix stale top-level README state

At audit time, the top-level narrative lags completed work, most visibly around Invisible Limit. The README should distinguish:

- **published family state**
- **completed labs on main**
- **active unmerged research**
- **retired/superseded experiment contracts**

## Priority 3 — explicitly label Showcase-selection leakage boundaries

Every scoreboard should mark whether it is:

- selection/DEV,
- frozen stability,
- DEV-FULL,
- sealed TEST.

This is especially important for the family-geometry and layerwise-soup programs.

## Priority 4 — preserve negative results in the same index

Do not hide:

- static selective-layer failure,
- Phase4B extra-compression failure,
- sparse-runtime failure,
- Phase9 procedure blocker,
- cross-scale NO-GO,
- significance-specific control failure,
- TGSA competence-regime rejection.

They are central to the scientific story.

## Priority 5 — distinguish “algorithm exists” from “algorithm has evidence”

For PCSM, PSA, TGSA, and cross-scale, put an explicit banner in each branch README:

- implementation maturity,
- real-model evidence level,
- confirmatory status,
- sealed status,
- what claims are forbidden.

The current TGSA and cross-scale contracts already move in this direction and should be the model for the rest.

---

# 20. Current branch inventory

Current remote branches observed during this audit:

| Branch | Role | Current evidentiary standing |
|---|---|---|
| `main` | Published family vault + completed labs | Canonical shipped history |
| `codex/exp-peer-stabilizer-matrix` | PCSM | Synthetic/resource pilot only |
| `product/rineya-family-cloud-demo-secure-delivery` | Secure public demo | Product/deployment |
| `research/positional-sequential-attention` | PSA | Measurement advanced; intervention open |
| `research/rineya-cross-scale-qwen4b-8b` | Cross-scale transfer | Development NO-GO; sealed closed |
| `research/tgsa-mtp-rineya-complementarity-v1` | TGSA/MTP complementarity implementation | Non-evidentiary until competence GO |
| `research/tgsa-phase1-v4-competence-first` | TGSA competence gate | Regime-family rejected |
| `research/tgsa-phase2-mtp-complementarity` | TGSA Phase2 MTP semantics/mechanics | Mechanics only until new competence GO |
| `research/tgsa-rineya-pretrained-retrofit` | Pretrained Rineya retrofit | Integration rehearsal; non-evidentiary |

This replaces the stale “13 branches / six TGSA branches” snapshot in the supplied audit.

---

# 21. Bottom-line interpretation

The most defensible scientific story of ChatBubble is **not** “we found one exotic architecture that makes 0.6B beat everything.”

It is more interesting and more rigorous:

1. A compressed 19-layer lineage became viable only after recovery and alignment.
2. Cheap parameter-space operations—especially model soup, scaffold restoration, LoRA scaling, and task-vector geometry—created a surprisingly capable family.
3. Routing and abstention/readout mechanisms moved the practical frontier further than any single dense core.
4. The sealed 4B comparison showed that the aggregate frontier can hide very different skill composition; much of the apparent tie comes from answerability/abstention rather than a uniform reasoning win.
5. The significance lab then demonstrated an important discipline: an attractive mechanism can have a real oracle opportunity and still fail to beat simple structural controls causally.
6. The current next-generation operators are therefore correctly held to harder standards:
   - **PCSM:** no real mechanism evidence yet.
   - **PSA:** measurement before intervention.
   - **TGSA:** no mechanism conclusion until a learnable competence regime exists.
   - **Cross-scale:** strong development NO-GO under the present projection family; no sealed escalation.

That makes the current program substantially stronger scientifically than a scoreboard-only model-development effort. Its most valuable evidence is increasingly the **boundary between what works operationally and what has actually been causally established**.

---

# 22. Canonical one-line status register

| Track | Canonical one-line status |
|---|---|
| Dense short-response SFT | **Positive** |
| Static selective-layer tuning | **Negative** |
| 19-layer progressive depth reduction | **Positive frontier** |
| Targeted recovery | **Positive** |
| DPO | **Positive, small effect** |
| Early model soup | **Positive** |
| Additional post-alignment compression | **Negative** |
| Context compaction | **Positive** |
| Sparse simulation | **Simulation positive only** |
| Sparse runtime | **Negative; dense retained** |
| Phase-8 RC | **Shipped decision** |
| Phase-9 student/RAG | **Blocked / incomplete** |
| Teacher soup mini | **Weak direct model; useful donor method** |
| Qwen3 cross-depth soup | **Positive / family-forming** |
| R2 cross-depth soup | **Positive signal; not vaulted** |
| Layerwise soup | **Shipped development-selected recipe** |
| Long-context weight geometry | **Positive product science** |
| Abstention gate | **Positive / shipped** |
| Invisible Limit vs 4B | **Aggregate tie; uniform-win hypothesis not established** |
| Significance-native Halo | **Structural compression positive; significance-specific gain not established** |
| Adapter consensus | **Local positive composition effect; transfer negative** |
| State sufficiency | **Infrastructure ready; real scientific comparison unrun** |
| PCSM | **Synthetic/resource evidence only** |
| PSA | **Measurement advanced; intervention open** |
| TGSA | **Operator advanced; mechanism hypothesis untested fairly** |
| Cross-scale transfer | **Strong development NO-GO; sealed closed** |
| Cloud demo | **Product/deployment, not accuracy science** |

---

## Evidence map

The audit was reconciled against the repository’s current `main`, current branch list, and the committed experiment reports/ledgers, including paths such as:

- `short-response-model-phase-0-1-2/reports/PHASE_1_STOP_GATE.md`
- `short-response-model-phase-0-1-2/reports/PHASE_2_STOP_GATE.md`
- `short-response-model-phase-0-1-2/reports/PHASE_2B_quality_v4_full_20260802_213049_REPORT.md`
- `short-response-model-phase-0-1-2/reports/PHASE_2C_TARGETED_RECOVERY_20260804_REPORT.md`
- `short-response-model-phase-0-1-2/reports/PHASE_3_phase2c_balanced_training_STOP_GATE.md`
- `short-response-model-phase-0-1-2/reports/MODEL_SOUP_REPORT_phase2c_balanced_training.md`
- `short-response-model-phase-0-1-2/reports/PHASE_4B_CANDIDATE_RECOMMENDATION_phase2c_balanced_training.md`
- `short-response-model-phase-0-1-2/reports/PHASE_5_STOP_GATE.md`
- `short-response-model-phase-0-1-2/reports/PHASE_6_STOP_GATE.md`
- `short-response-model-phase-0-1-2/reports/PHASE_7_STOP_GATE.md`
- `short-response-model-phase-0-1-2/reports/PHASE_8_STOP_GATE.md`
- `short-response-model-phase-0-1-2/reports/PHASE_9_STOP_GATE.md`
- `Rineya_Teacher_Soup_Mini_v1/`
- `Rineya_Qwen3_CrossDepth_Soup_Mini_v1/`
- `Rineya_R2_CrossDepth_Soup_Mini_v1/`
- `Rineya_Layerwise_Soup_Lab_v1.0/`
- `Rineya_Long_Context_Weight_Geometry_Lab_v1/PUBLISHED_FAMILY_SEARCH_REPORT.md`
- `Rineya_Abstention_Gate_Lab_v1/README.md`
- `Rineya_Invisible_Limit_Lab_v1/FINAL_REPORT.md`
- `rineya_significance_native_architecture_lab/CLAIMS_LEDGER.md`
- `short-response-adapter-consensus-lab/reports/PHASE2_MECHANISM_REPORT.md`
- `Sequence_Experiment/Rineya_State_Sufficiency_Lab_v1.0.1_CORRECTED/`
- `rineya_pcsm_native_weight_lab_v1/results/`
- `Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1/`
- `Rineya_Cross_Scale_Capability_Transfer_Lab_v1/cross_scale_evidence/`
- `Rineya_Cloud_Demo_v1/`

Where a branch contains implementation scaffolding without a decisive real-model result, this audit labels it as infrastructure/open rather than inferring a scientific conclusion from code volume.
