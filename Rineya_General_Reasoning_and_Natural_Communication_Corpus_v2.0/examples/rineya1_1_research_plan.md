# Rineya1.1 Corpus v2.0 — Research Rationale and First-Run Plan

## Why this corpus exists

The first 16,410-row dual-SFT run showed that a compact 454M teacher can absorb both conversational and evidence-grounded behavior in one QLoRA epoch. The next data iteration therefore targets the remaining bottlenecks: general reasoning breadth, natural multi-turn communication, state updates, underdetermination, and reasoning-plus-dialogue integration.

The design follows the v2.3.1 lesson that raw repetition is not the dominant variable. Scale is increased only together with structural diversity, counterfactual boundaries, balanced classes, explicit failure contrasts, and preservation replay.

## Controlled first run

1. Keep the current accepted Rineya1 SFT adapter immutable.
2. Mix the 40,000-row recommended corpus with the local 4,410-row converted RAG training set exactly once.
3. Shuffle deterministically and run one QLoRA epoch at the previously successful `1e-4` learning rate.
4. Re-run the same 510-case chat DEV and official RAG DEV used for the current baseline.
5. Compare aggregate and per-category tails: task adequacy, supported, abstain, procedure, context updates, reasoning, empathy, safety, and brevity.
6. Freeze one SFT winner before DPO or any sealed evaluation.
7. Train DPO as a child artifact, never by overwriting the SFT winner.

## Suggested ablations before a larger sweep

| Run | SFT data | Purpose |
|---|---:|---|
| A0 | Current 16,410-row adapter | Frozen baseline |
| A1 | 40,000 v2 + 4,410 RAG | Recommended dual-skill candidate |
| A2 | 32,000 new-only + 4,410 RAG | Measures preservation-slice value |
| A3 | 16,000 reasoning + 4,410 RAG | Reasoning acquisition control |
| A4 | 9,600 communication + 4,410 RAG | Naturalness/adequacy control |
| A5 | 6,400 integrated + 4,410 RAG | Reasoning-in-dialogue control |

Only after A1 is reproducibly positive should a small LR sweep be considered. A sensible local screen is `5e-5`, `7.5e-5`, and `1e-4`, all at one epoch with identical seeds and evaluation.

## Release boundary

DEV improvement makes a candidate a research keep, not a release proof. Human review and one deliberately frozen sealed/test pass remain separate release gates.
