# Cross-Scale v2 Addendum — Gated Methods, Then Better T

Protocol: `amendment_v2_exploratory.md` version `2.0.0-exploratory`

Parent: v1 `preregistration.md` `1.0.0` and `11_report/final_report.md` (immutable)

Research branch: `research/rineya-cross-scale-qwen4b-8b`

This addendum does not rewrite the v1 report. All v2 outcome runs are exploratory. DEV-FULL is an intermediate development set for new methods because it was already peeked for v1 `CONS_MEAN_A1500`. The unused sealed bank remains the only confirmatory set and was not opened.

## Abstract

v2 first ran the H3/H6 operators that v1 implemented but did not score: selective consensus, TIES, DARE, and DARE-TIES on the hash-verified P4-s010 deltas. On the frozen MERGEDEV-fast screen they reproduced the v1 mean-consensus score (13/16) or fell back to ungated Supa (12/16). None beat both controls. Localization GO failed; one-layer, group, module, and Fisher maps were not run.

Because the interference-aware methods collapsed onto the already-known consensus, Phase C tested one exploratory projection family. Linear CKA on 0.6B calibration hidden states recovered an identity depth map. A rank-8 direction-transfer SVD (P5) produced a coherent 4B→0.6B donor. As a projected control it scored 13/16 with weaker format/completion than Supa. Injected into Supa at α=0.15 it scored 12/16 and was not two-control eligible.

v2 therefore localizes the failure to the projection operator T, not to untested merge interference. No DEV-FULL, sealed, or recovery-LoRA run was authorized.

## What stayed frozen

- v1 protocol, final report, and `09_finalists/NO_GO.json`
- Official revisions: Qwen3-0.6B `c1899de…`, 4B `1cfa9a7…`, 8B `b968826…`; target `Rineya-Supa-A1.1-1.1.0-devfreeze`
- MERGEDEV-fast 16-row subset
- P4-s010 projected donors and deltas (Δ4 digest `8edf2d4f…`, Δ8 digest `2951457e…`, v1 consensus weight SHA `5758b7af…`)
- Invisible-Limit TEST2 / Stage 6 denylist
- CPU-first 0.6B evaluation; GPU refused while `nvidia-smi` showed the card busy

## Phase A — H3/H6 on P4-s010

| Candidate | MERGEDEV-fast | vs Supa (12/16) | vs CONS_MEAN (13/16) |
|---|---:|---|---|
| CONS_SIGN_A1500 | 13 | +1 | tie |
| CONS_COS03_A1500 | 13 | +1 | tie |
| TIES keep 0.20 / 0.50 / 0.80 on Δ4+Δ8 | 13 | +1 | tie |
| DARETIES p=0.20 seed 17 | 13 | +1 | tie |
| DARE8 p=0.20 seed 17 | 12 | tie | −1 |
| CONS_COS05 / COS07 | not scored | disk full (`os error 112`) | — |

Zero two-control eligible rows. The only merge-only fix versus Supa on every 13/16 method is `mergedev_counterfactual_reasoning_770aaf5dae0c1fb5`, already solved by v1 consensus. No candidate was correct when both parents were wrong.

Skipped after the gate: α ∈ {0.10, 0.20} expansion, remaining DARE seeds, TIES sets that include Δ_Supa, Fisher (λ ∈ {0.25, 1.0}), and all H5 maps.

## Phase B — H5 localization

Not run. See `layer_analysis/NO_GO.md`.

## Phase C — Better T

- D4 CKA, Qwen3-0.6B vs Supa, 16 calibration prompts (SHA `d2f1f415…`): identity mapping, score 27.82.
- 4B/8B CKA: `not_run_compute` (GPU busy).
- P5-4B direction-transfer: functional GO; MERGEDEV-fast control 13/16, completion 0.75, format 0.625.
- 8B P5: `not_run_compute`. The 4B T already re-entered Phase A; further T families were not stacked.
- `P5_TASK_4B_A1500` = Supa + 0.15·(P5_4B − Qwen3-0.6B): functional GO; MERGEDEV-fast 12/16, completion 1.0, format 0.875. Ties Supa; loses to v1 consensus.

## Phase D — DEV-FULL, sealed, LoRA

Not authorized. No survivor beat both controls. Sealed JSONL was never read.

## Answers to the twenty research questions

Each item is **v1 result**, then **v2 update**.

1. **Can 4B improve Supa?** v1: not robustly; it contributed to a MERGEDEV-only consensus gain. v2: P5-4B is a coherent projected control at 13/16 on MERGEDEV-fast but does not beat v1 consensus; the P5 task vector ties Supa at 12/16.

2. **Can 8B improve Supa?** v1: not robustly. v2: no new 8B outcome. 8B P5 and 8B CKA were `not_run_compute`.

3. **Is 8B a better donor?** v1: inconclusive. v2: unchanged; 8B-specific T was not run.

4. **Best projection method?** v1: P4 rank-8 spectral transfer at strength 0.10, the only coherent family. v2: P5 is also coherent and matches consensus semantic score as a control, with worse format/completion. Neither T produced a two-control merge win.

5. **Best interpolation coefficient?** v1: development selected mean-consensus α=0.15; it failed DEV-FULL. v2: the frozen α=0.15 grid was reused; no new coefficient beat both controls.

6. **Does small alpha beat 50/50?** v1: small transfer was necessary for coherence. v2: 50/50 remained unauthorized.

7. **Task vector versus absolute interpolation?** v1: representative versions tied Supa on the fast screen. v2: P5 task vector again ties Supa (12/16) and is worse than mean consensus.

8. **Are delta4 and delta8 aligned?** v1: yes (cosine 0.932, sign agreement 0.926). v2: that alignment explains why sign/cosine/TIES/DARE-TIES reproduced mean consensus.

9. **Evidence for a common scale trajectory?** v1: yes for spectral statistics; no robust behavioral gain. v2: same. Interference-aware merges did not extract an extra direction.

10. **Which layers are most transferable?** v1: not estimated. v2: still not estimated; Localization GO failed.

11. **Which modules are most transferable?** v1: no behavioral localization. v2: unchanged.

12. **Attention or MLP deltas?** v1: no outcome comparison; descriptive cosines were similarly high. v2: unchanged.

13. **Middle/late concentration?** v1: not tested. v2: not tested.

14. **Do TIES/DARE reduce interference?** v1: tooling only. v2: on real P4-s010 checkpoints they do not beat mean consensus. TIES and DARE-TIES tied it; DARE on Δ8 alone removed the one-case gain.

15. **Does Fisher weighting help?** v1: tooling only. v2: `not_run_compute` after Localization GO failed and the GPU stayed busy.

16. **Can 4B+8B consensus beat either donor?** v1: MERGEDEV +1 vs Supa, DEV-FULL tie. v2: selective consensus and TIES/DARE-TIES did not improve on that mean.

17. **Can the final model beat prior strong Rineya baselines?** v1: no robust final model. v2: no v2 finalist.

18. **Are improvements preserved outside development selection?** v1: no; MERGEDEV gain vanished on DEV-FULL. v2: no new method earned a DEV-FULL or sealed run.

19. **Does inference remain approximately 0.6B?** v1: yes, 596,049,920 parameters. v2: yes for P4 merges, P5-4B, and `P5_TASK_4B_A1500`.

20. **Compute required?** v1: 0 GPU-hours for frozen-valid evaluation. v2: GPU remained occupied by other jobs and was not taken. All 0.6B construction and MERGEDEV-fast screens ran on CPU. 4B/8B activation CKA, 8B P5, and Fisher were recorded `not_run_compute`. Disk-full failures for cosine τ ∈ {0.5, 0.7} were preserved in the registry.

## Claim boundary

v2 is an exploratory follow-up. It does not claim sealed confirmation, 8B-equivalence, or that a MERGEDEV-only gain is a robust Supa improvement. The scientifically justified conclusion is that H5/H6 cannot rescue P4-s010, and one better-T attempt (P5) remains a near-null.

## Final v2 conclusion

The skipped merge operators were not the missing ingredient. Δ4 and Δ8 already agree so strongly that sign, cosine, TIES, and DARE-TIES are operationally the same as mean consensus on the frozen screen. A direction-transfer T can stay coherent at 0.6B and still fail to beat both ungated Supa and the v1 consensus once injected. The bounded negative v1 finding stands; v2 tightens the failure boundary from “untested interference methods” to “the projection T itself.”
