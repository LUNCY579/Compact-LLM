# Navigation

[Audit index](README.md) · [Foundational/family](01_FOUNDATIONAL_AND_FAMILY_HISTORY.md) · [Product science](02_PRODUCT_SCIENCE_AND_VALIDATED_MECHANISMS.md) · [Frontier labs](03_FRONTIER_RESEARCH_LABS.md) · [Synthesis](04_PROGRAM_SYNTHESIS.md) · [Status/evidence](05_STATUS_ROADMAP_AND_EVIDENCE_MAP.md)

---

# 14. Supporting corpora and evaluation infrastructure

These are essential to reproduce the research but should not be mistaken for experiments.

## LCR SFT v1

- 2,000 logical packs / ~8,000 rows.
- 1,800 train packs / ~7,200 rows.
- 200 DEV packs / ~800 rows.
- Variants include base, paraphrase, counterfactual, and unanswerable.
- Designed to remain disjoint from the long-context Demo surface.

This corpus is central to the later Invisible-Limit interpretation because the Rineya side is task-adapted while Qwen3-4B is zero-shot.

## Frozen Demo / Showcase / Stability surfaces

The long-context family program repeatedly uses a frozen protocol with deterministic generation settings. These surfaces enable comparison across family cores and routing recipes, but **Showcase selection creates a selection surface**, so claims about generalization should distinguish Showcase, Stability, DEV-FULL, and sealed TEST.

## Claims ledgers / preregistrations

The later mechanism labs—especially significance, TGSA, PSA, and cross-scale—show a substantial improvement in research hygiene:

- explicit control rules,
- preregistered gates,
- retractions,
- frozen/sealed escalation,
- “do not advance” contracts.

That discipline is as important as any individual score.

---

# 15. What actually ships versus what remains research

| Category | Mechanism / artifact | Standing |
|---|---|---|
| **Ships / vaulted** | Dense family cores (Pulse/Halo/Nimbus/Supa/Baya/etc.) | **Yes** |
| **Ships / vaulted** | Cross-depth / model-soup recipes | **Yes** |
| **Ships / product science** | LoRA-scale / task-vector geometry | **Yes** |
| **Ships / product science** | Answer-blind cascades / gates (System46/48 class) | **Yes as routing systems, not one dense model** |
| **Ships** | Supa-A1 abstention/readout behavior | **Yes** |
| **Research finding** | Hidden-state answerability/readout signal | Strong evidence; not a blanket reasoning claim |
| **Research finding** | Structural span pooling / sinks under compression | Positive |
| **Research-only / not established** | Significance score S as causal quality mechanism | Not established |
| **Research-only** | Adapter consensus mechanism | Local effect; poor cross-corpus transfer |
| **Research-only** | R2 cross-depth donor transfer | Positive signal, not vaulted |
| **Research-only** | State-sufficiency | Real scientific run still missing |
| **Research-only** | PCSM | Synthetic/resource evidence only |
| **Research-only** | PSA steering | Open |
| **Research-only** | TGSA | No fair confirmatory mechanism result yet |
| **Research-only** | Cross-scale 4B/8B transfer | Development NO-GO |
| **Product-only** | Cloud demo secure delivery | Deployment track, not model science |

---

# 16. Dependency map: how the experiments relate

A compact causal/program map is:

```text
Phase-1 dense SFT
    ↓
Phase-2A static selective tuning  ── negative
    ↓
Phase-2B 19-layer structural frontier
    ↓
Phase-2C targeted recovery
    ↓
Phase-3 DPO
    ↓
Phase-4 seed/model soup
    ├─────────────→ context compaction → sparse sim → sparse runtime negative → dense RC
    │
    └─────────────→ teacher / cross-depth soup geometry
                         ↓
                  named 28-layer family vault
                         ↓
             long-context LoRA/task-vector geometry
                  ↙                    ↘
          routed cascades          abstention/readout
                  ↓                    ↓
           System46/48              Supa-A1
                         \          /
                          \        /
                     Invisible Limit vs 4B
                              ↓
                   "tie" decomposes by skill
                              ↓
       ┌──────────────────────┼─────────────────────────┐
       ↓                      ↓                         ↓
significance/KV          PSA positional           next operators
compression lab          measurement              PCSM / TGSA
                                                       ↓
                                                cross-scale transfer
                                                (current NO-GO)
```

The key scientific progression is:

1. **Find useful parameter-space geometry.**
2. **Separate output/readout failures from representation failures.**
3. **Interrogate whether the same gains can be explained by a native mechanism.**
4. **Only then attempt structurally novel operators or cross-scale transfer.**

---

# 17. Current scientific state by research question

| Question | Best current answer |
|---|---|
| Can a small Qwen3-family model be made much more useful with cheap operations? | **Yes.** Soups, scale geometry, routing, and abstention interventions materially move the frozen program scoreboards. |
| Is the family-vault frontier one dense universal model? | **No.** Many best results are routed/cascaded policies or distinct cores. |
| Did the 0.6B family uniformly beat Qwen3-4B? | **No.** The 1,024-case sealed aggregate is essentially tied; family-wise skill composition differs sharply. |
| Is abstention a genuine capability axis? | **Yes, strongly.** Both simple gates and hidden-state probes support a readout/answerability story. |
| Does significance-weighted KV retention cause the Halo quality gain? | **Not established.** Structural pooling/recency/sinks are the robust result. |
| Does hard adapter consensus generalize universally? | **No.** Its cornerstone behavior is benchmark/policy-composition specific and cross-corpus transfer is poor. |
| Does PCSM coworker regulation work on a real core? | **Unknown.** Synthetic/software evidence only; real pilot is scientifically non-evaluable. |
| Can PSA positional measurement be turned into a quality-improving intervention? | **Unknown/open.** Measurement is much further along than treatment. |
| Does TGSA improve competence? | **Unknown.** v3 was inconclusive and v4 rejected the competence regime before a fair mechanism test. |
| Does 4B/8B spectral/task-vector information transfer into 0.6B through the current projection? | **Development evidence says no.** Strong donor alignment did not yield robust target behavior; sealed stayed closed. |
| Is sparse runtime currently preferable to dense? | **No.** The real sparse-runtime phase selected dense fallback. |
| Is the compact student/RAG track ready? | **No.** Procedure correctness remains a decisive blocker in the committed Phase-9 gate. |

---
