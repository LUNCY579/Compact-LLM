# v2 Phase A Outcome Gate

## Localization GO

**NO-GO.**

No H3/H6 candidate was two-control eligible on MERGEDEV-fast. Sign-agree, cosine-0.3, TIES (keep 0.20/0.50/0.80 on Δ4+Δ8), and DARE-TIES (p=0.20, seed 17) all scored 13/16, tying v1 mean consensus and beating ungated Supa by one case. DARE on Δ8 (p=0.20, seed 17) scored 12/16, tying Supa.

Therefore one-layer, group, and module maps are not authorized.

## Advanced-method interpretation

Δ4 and Δ8 already agree at cosine 0.932. Sign masks, cosine filters, TIES, and DARE-TIES reproduced the v1 consensus behavior on the frozen 16-row screen. Random DARE sparsification of Δ8 alone removed the one-case gain.

Complementarity (`mergedev/complementarity.json`): every 13/16 method’s only merge-only fix versus Supa is `mergedev_counterfactual_reasoning_770aaf5dae0c1fb5`, the same case v1 `CONS_MEAN_A1500` already gets. No candidate was correct when both Supa and v1 consensus were wrong.

## Disk / compute notes

- GPU remained busy; all 0.6B work stayed on CPU.
- CONS_COS05 and CONS_COS07 failed with `os error 112` (disk full). Unused P1–P3 projection caches were then deleted. Remaining cosine thresholds and extra DARE seeds were not rerun because the completed arms already showed the interference-aware methods collapsing to the v1 consensus score.
- Fisher estimation was not run (`not_run_compute`): GPU busy and no Localization GO.

## Phase C re-entry

P5-4B direction-transfer was coherent. `P5_TASK_4B_A1500` (Supa + 0.15·Δ_P5) scored 12/16, tying Supa and losing to v1 consensus. Still no two-control eligible candidate.

## DEV-FULL / sealed

Not authorized. No candidate beat both Supa and v1 consensus.
