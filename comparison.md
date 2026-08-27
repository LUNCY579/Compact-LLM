# Family eval comparison

Numbers only. No ranking. Geometry, recipes, and dates live in [`lineage.md`](lineage.md). Public-copy rules live in [`announcement.md`](announcement.md). Index: [`roadmap.md`](roadmap.md).

**Bold** = the largest stored number in that column (ties all bold), including rates that count defects. `—` = that field was not stored or that eval was not run for this core.

The vault has eight dense cores plus four published routed systems. **Supa 2.0**
and **Baya 2.0** name the 48/46 frontier systems; Supa Core 1.7 and Baya Core
1.7 remain immutable 40-point dense dependencies.

**Rename note (2026-08-26).** Supa was formerly named Taria; Baya was formerly named Lumina. This identity-only rename did not change either weight file or any score below.

Last compiled: 2026-08-27.

**Roster note (2026-08-21).** Active **Pulse** = 1.6.0 (`late_rineya`, 28L). Active **Nimbus** = 1.6.0 (soup-lab `global_best` α=0.40). Prior Pulse 1.0 / Nimbus 1.3 are under `_superseded/`. Phase 8 sealed-chat row below is the **archived** Pulse 1.0 teacher only.

## Current common-protocol Long Context Demo baseline (2026-08-26)

Every row below was generated from the actual published-core directory with the same protocol: BF16 load, greedy seed 17, thinking disabled, 24 generated tokens, 4096 context, no truncation.

| Core | Showcase 64 | Stability 192 | All 256 | All-four packs / 64 |
|---|---:|---:|---:|---:|
| Pulse 1.6 | 31 | 89 | 120 | 0 |
| Aegis 1.1 | 2 | 20 | 22 | 0 |
| Halo 1.2 | 29 | 86 | 115 | 0 |
| Nimbus 1.6 | 30 | 83 | 113 | 0 |
| Lyra 1.4 | 28 | 86 | 114 | 0 |
| Halo Uncensored 1.5 | 31 | 96 | 127 | 0 |
| **Supa 1.7** | **40** | **121** | **161** | **13** |
| **Baya 1.7** | **40** | **112** | **152** | **14** |

Supa/Baya dense-vs-PEFT prediction parity is 251/256 and 249/256 respectively. Dense scores are reported above; adapter-form scores are not silently inherited.

### Fast-algorithm family frontier

| Recipe | Arms | Showcase | Stability | All 256 | All-four packs / 64 |
|---|---:|---:|---:|---:|---:|
| Supa-Baya Dense Gate42 | 2 | 42 | 124 | 166 | 18 |
| Nimbus DualScale PEFT Gate45 | 2 | 45 | 124 | 169 | 19 |
| **Baya 2.0 System46** | 3 | 46 | 125 | 171 | 21 |
| **Supa 2.0 System48** | 9 | **48** | **128** | **176** | **21** |

All four rows are now backed up as runnable system bundles under
`Rineya_Published_Cores/systems/`. They use prediction-only routing and never
read answers at inference time. They are not single dense checkpoints.

### Best tested recipe by compatible anchor

| Anchor | Best recipe | Showcase | Stability |
|---|---|---:|---:|
| Pulse | negative-vector UNKNOWN gate | 43 | 118 |
| Halo | checkpoint-200 scale gate | 41 | 112 |
| Lyra | checkpoint-200 scale gate | 41 | 118 |
| Nimbus | checkpoint-200 scale gate | 45 | 124 |
| Halo Uncensored | baseline retained | 31 | 96 |
| Aegis | native 19L Pulse/Aegis LERP 0.5 | 6 | not promoted |

---

---

## What was measured

| Eval | Protocol | Items | Source |
|---|---|---:|---|
| Frozen long-context showcase | greedy, 24 new tokens, context 4096, seed 17, thinking off | 64 | Pulse: `Rineya_Long_Context_Demo_v1/outputs/greedy_soup_v2_showcase_report.json` (Phase 8 accepted teacher = Pulse). Aegis: `…/aegis_showcase_report.json`. 28L: `Rineya_Qwen3_CrossDepth_Soup_Mini_v1/outputs/*_report.json`. Official Qwen3-4B/8B: `Rineya_Long_Context_Demo_v1/outputs/qwen3_4b_showcase_report.json` and `qwen3_8b_showcase_report.json` (NF4 4-bit; 8GB GPU cannot hold bf16 4B/8B) |
| Dual-skill chat DEV | Phase 1 `dev.jsonl`, 32 tokens, `eos_only`, quality v4.5 | 510 | Pulse / Aegis: `short-response-model-phase-0-1-2/artifacts/teacher_dual_sft_v1/lr_sweep/SELECTION_REPORT.md` (2026-08-17). Aegis there is PEFT on Pulse; vault Aegis is the merged dense file. 28L: `Rineya_Qwen3_CrossDepth_Soup_Mini_v1/outputs/phase_evals/*/chat_dev/summary.json` |
| Dual-skill RAG DEV | Phase 9 `dev.reviewed.jsonl`, oracle_rag + real_rag, `query_aware_v21` | 752×2 | Pulse / Aegis: same SELECTION_REPORT. 28L: `…/phase_evals/*/rag_dev/summary.json` and `…/phase_evals/scoreboard.json` |
| Dual composite | `0.30·chat_Q + 0.15·task + 0.20·oracle_supp + 0.15·real_supp + 0.10·real_proc + 0.10·oracle_abstain` | — | Recorded on Aegis (SELECTION_REPORT) and on 28L (`phase_evals/scoreboard.md`). Pulse has no recorded composite |
| Phase 8 sealed chat | 110 rows, `one_sentence`, 32 tokens, quality v4.5 | 110 | Pulse only: `short-response-model-phase-0-1-2/reports/RELEASE_CANDIDATE_AUDIT.json` |
| Dual-skill sealed chat TEST / RAG TEST | — | — | `not_run` for all six family cores |

## Eval coverage

| Eval | Pulse | Aegis | Halo | Nimbus | Lyra | Halo Uncensored |
|---|---|---|---|---|---|---|
| Showcase 64 | run | run | run | run | run | run |
| Chat DEV 510 | run (sweep baseline) | run (sweep selected, PEFT) | run | run | run | run |
| RAG DEV 752×2 | run (sweep baseline) | run (sweep selected, PEFT) | run | run | run | run |
| Dual composite | — | run | run | run | run | run |
| Phase 8 sealed chat 110 (`one_sentence`) | run | — | — | — | — | — |
| Dual-skill sealed chat TEST / RAG TEST | `not_run` | `not_run` | `not_run` | `not_run` | `not_run` | `not_run` |

---

## Showcase (64-row frozen long-context)

| Name | Hits / 64 | Strict exact | First-line exact | Unanswerable | CF pair correct+changed | Base+paraphrase both | Base/paraphrase agree | ≤12-word rate | Mean response words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pulse | **31/64** | **0.4844** | **0.4844** | **0.1250** | **0.4375** | **0.5000** | 0.6250 | 0.9062 | 3.2188 |
| Aegis | 4/64 | 0.0625 | 0.0625 | 0.0000 | 0.0000 | 0.0625 | 0.5625 | **1.0000** | 2.6094 |
| Halo | 30/64 | 0.4688 | 0.4688 | **0.1250** | 0.3750 | 0.4375 | 0.5625 | 0.9062 | 3.2500 |
| Nimbus | 29/64 | 0.4531 | 0.4531 | **0.1250** | 0.3125 | 0.3750 | 0.4375 | 0.9219 | 3.0781 |
| Lyra | 28/64 | 0.4375 | 0.4375 | **0.1250** | 0.3125 | 0.4375 | 0.6250 | 0.9062 | 3.2344 |
| Halo Uncensored | **31/64** | **0.4844** | **0.4844** | 0.0000 | 0.3750 | **0.5625** | **0.6875** | 0.9844 | 2.5781 |

Coverage is 64/64 on all six. All-four-variants-correct is 0.0000 on all six. Pulse 1.6 source: `…/alpha_recipes/late_rineya_report.json`. Nimbus 1.6 source: `…/lab_fillins_cpu/showcase/global_best_report.json`.

### By variant (strict exact, 16 rows each)

| Name | Base | Paraphrase | Counterfactual | Unanswerable |
|---|---:|---:|---:|---:|
| Pulse | **0.5625** | 0.6875 | **0.5625** | **0.1250** |
| Aegis | 0.1250 | 0.0625 | 0.0625 | 0.0000 |
| Halo | 0.5000 | 0.6875 | **0.5625** | **0.1250** |
| Nimbus | 0.4375 | 0.6875 | **0.5625** | **0.1250** |
| Lyra | 0.4375 | 0.6875 | 0.5000 | **0.1250** |
| Halo Uncensored | **0.5625** | **0.8750** | 0.5000 | 0.0000 |

### By difficulty (strict exact)

| Name | Hard (n=40) | Medium (n=24) |
|---|---:|---:|
| Pulse | 0.3750 | **0.6667** |
| Aegis | 0.1000 | 0.0000 |
| Halo | 0.3500 | **0.6667** |
| Nimbus | 0.3750 | 0.5833 |
| Lyra | 0.3250 | 0.6250 |
| Halo Uncensored | **0.4250** | 0.5833 |

### By needle profile (strict exact, 16 rows each)

| Name | Distributed | Early | Late | Middle |
|---|---:|---:|---:|---:|
| Pulse | **0.7500** | 0.1875 | **0.6250** | 0.3750 |
| Aegis | 0.0000 | 0.0625 | 0.1875 | 0.0000 |
| Halo | 0.6875 | 0.1875 | **0.6250** | 0.3750 |
| Nimbus | **0.7500** | 0.1875 | 0.5000 | 0.3750 |
| Lyra | 0.6875 | 0.1250 | **0.6250** | 0.3125 |
| Halo Uncensored | 0.6250 | **0.2500** | **0.6250** | **0.4375** |

### By length bucket (strict exact)

| Name | Compact long (n=32) | Extended long (n=16) | Standard long (n=16) |
|---|---:|---:|---:|
| Pulse | 0.4375 | **0.6250** | 0.4375 |
| Aegis | 0.0938 | 0.0000 | 0.0625 |
| Halo | 0.4375 | 0.5625 | 0.4375 |
| Nimbus | 0.4063 | 0.5625 | 0.4375 |
| Lyra | 0.4063 | 0.5000 | 0.4375 |
| Halo Uncensored | **0.4688** | 0.3750 | **0.6250** |

### Reference: official Qwen3 scale (not family cores)

There is no official Qwen3 **3B** or **9B**. Closest dense cards: `Qwen/Qwen3-4B` and `Qwen/Qwen3-8B`, saved at repo-root `Qwen3-4B/` and `Qwen3-8B/` (gitignored). Same greedy / 24 tokens / 4096 / seed 17 / thinking-off protocol. 0.6B was bf16. 4B and 8B were NF4 4-bit (`--load-in-4bit`) so they fit an 8GB GPU. Not included in the family bold maxima above. Chat DEV / RAG DEV / sealed: `not_run`.

| Name | Load | Hits / 64 | Strict exact | Unanswerable | CF pair correct+changed | Base+paraphrase both | Base/paraphrase agree | All-four | ≤12-word | Mean words |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | bf16 | 14/64 | 0.2188 | 0.1875 | 0.1250 | 0.1250 | 0.4375 | 0.0000 | 0.8906 | **4.4063** |
| Qwen3-4B | NF4 4-bit | 43/64 | 0.6719 | 0.3125 | 0.6875 | 0.6875 | 0.8125 | 0.2500 | 0.9219 | 3.5938 |
| Qwen3-8B | NF4 4-bit | **53/64** | **0.8281** | **0.4375** | **0.9375** | **0.9375** | **0.9375** | **0.4375** | **0.9844** | 2.5000 |

| Name | Base | Paraphrase | Counterfactual | Unanswerable | Hard (n=40) | Medium (n=24) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0.2500 | 0.1875 | 0.2500 | 0.1875 | 0.2250 | 0.2083 |
| Qwen3-4B | 0.6875 | 0.8750 | 0.8125 | 0.3125 | 0.5250 | 0.9167 |
| Qwen3-8B | **0.9375** | **1.0000** | **0.9375** | **0.4375** | **0.7500** | **0.9583** |

| Name | Distributed | Early | Late | Middle | Compact long | Extended long | Standard long |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0.3750 | 0.0625 | 0.3125 | 0.1250 | 0.2500 | 0.1875 | 0.1875 |
| Qwen3-4B | **0.8750** | 0.5000 | 0.6875 | 0.6250 | 0.7500 | 0.6250 | 0.5625 |
| Qwen3-8B | **0.8750** | **0.6250** | **0.9375** | **0.8750** | **0.8125** | **0.8125** | **0.8750** |

### Family + official Qwen3 showcase (same 64 rows)

Bold = largest in this table, including the official scale models.

| Name | Hits / 64 | Strict exact | Unanswerable | CF pair correct+changed | Base+paraphrase both | All-four | ≤12-word |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pulse | **31/64** | **0.4844** | **0.1250** | **0.4375** | **0.5000** | 0.0000 | 0.9062 |
| Aegis | 4/64 | 0.0625 | 0.0000 | 0.0000 | 0.0625 | 0.0000 | **1.0000** |
| Nimbus | 29/64 | 0.4531 | 0.1250 | 0.3125 | 0.3750 | 0.0000 | 0.9219 |
| Lyra | 28/64 | 0.4375 | 0.1250 | 0.3125 | 0.4375 | 0.0000 | 0.9062 |
| Halo | 30/64 | 0.4688 | 0.1250 | 0.3750 | 0.4375 | 0.0000 | 0.9062 |
| Halo Uncensored | **31/64** | **0.4844** | 0.0000 | 0.3750 | 0.5625 | 0.0000 | 0.9844 |
| Qwen3-0.6B | 14/64 | 0.2188 | 0.1875 | 0.1250 | 0.1250 | 0.0000 | 0.8906 |
| Qwen3-4B (NF4) | 43/64 | 0.6719 | 0.3125 | 0.6875 | 0.6875 | 0.2500 | 0.9219 |
| Qwen3-8B (NF4) | **53/64** | **0.8281** | **0.4375** | **0.9375** | **0.9375** | **0.4375** | 0.9844 |

---

## Lab history: Halo-donor α recipes (Pulse 1.6 source)

`late_rineya` is now vaulted as **Pulse 1.6.0**. Other recipes stay lab-only. Source: [`Rineya_Qwen3_CrossDepth_Soup_Mini_v1/outputs/alpha_recipes/scoreboard.md`](Rineya_Qwen3_CrossDepth_Soup_Mini_v1/outputs/alpha_recipes/scoreboard.md).

| Recipe | Family | Mean α | Hits / 64 | Notes |
|---|---|---:|---:|---|
| `late_rineya` | shape mean 0.5 | 0.5000 | 31/64 | **→ Pulse 1.6.0** |
| `uniform_0.5` (Halo) | uniform | 0.5000 | 30/64 | vault Halo |
| soup-lab `global_best` α=0.40 | uniform | 0.4000 | 29/64 | **→ Nimbus 1.6.0** |

---

## Dual composite (recorded formula)

`0.30·chat_Q + 0.15·task + 0.20·oracle_supp + 0.15·real_supp + 0.10·real_proc + 0.10·oracle_abstain`.

| Name | Dual composite |
|---|---:|
| Pulse | 0.7561 |
| Aegis | **0.8382** |
| Halo | 0.7381 |
| Nimbus | 0.7444 |
| Lyra | 0.7445 |
| Halo Uncensored | 0.6823 |

Pulse / Nimbus dual numbers: [`…/lab_fillins_cpu/scoreboard.md`](Rineya_Qwen3_CrossDepth_Soup_Mini_v1/outputs/lab_fillins_cpu/scoreboard.md).

---

## Dual-skill chat DEV (510)

| Name | Chat Q | Task adequacy | Median tok | P90 tok | Duplicate rate | 1-sentence viol. | Legacy heur. | Intel pass | Hard fails | Safety | Min Q | Quality pass (diag.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pulse | 0.8612 | 0.5067 | 9.0 | 15.0 | 0.3353 | 0.4529 | — | **1.0000** | **0** | — | — | — |
| Aegis | 0.8594 | 0.4978 | — | — | — | — | — | — | — | — | — | — |
| Halo | 0.8619 | 0.5117 | **11.0** | 17.0 | 0.3627 | **0.5294** | 0.9850 | **1.0000** | **0** | **1.0000** | **0.39** | 0.9961 |
| Nimbus | 0.8649 | 0.5204 | **11.0** | **18.0** | 0.3392 | 0.5275 | 0.9784 | **1.0000** | **0** | — | — | **0.9980** |
| Lyra | **0.8667** | **0.5289** | **11.0** | 17.0 | 0.3412 | **0.5294** | 0.9869 | **1.0000** | **0** | **1.0000** | **0.39** | 0.9961 |
| Halo Uncensored | 0.8603 | 0.5035 | 9.0 | 12.0 | **0.4353** | 0.4451 | **0.9987** | **1.0000** | **0** | **1.0000** | **0.39** | **0.9980** |

Pulse / Nimbus 1.6 from `lab_fillins_cpu/*/chat_dev/summary.json`. Aegis beyond SELECTION_REPORT was not stored in that report.

### Quality dimensions (v4.5 means)

Pulse / Aegis: only `task_adequacy` is in SELECTION_REPORT. Other dimension cells for those two are —.

| Name | Brevity | Comprehens. | Context cons. | Contract | Naturalness | Factual | Logic/fact | Mechanical | Prompt-fn align | Completeness | Mode fulfill | Safety | Task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pulse | — | — | — | — | — | — | — | — | — | — | — | — | 0.2952 |
| Aegis | — | — | — | — | — | — | — | — | — | — | — | — | 0.4978 |
| Halo | 0.9908 | 0.9986 | **1.0000** | **1.0000** | **0.9992** | **1.0000** | **1.0000** | 0.9986 | 0.5117 | **1.0000** | 0.5117 | **1.0000** | 0.5117 |
| Nimbus | 0.9913 | **0.9993** | **1.0000** | **1.0000** | **0.9992** | **1.0000** | **1.0000** | **0.9993** | 0.5032 | **1.0000** | 0.5032 | **1.0000** | 0.5032 |
| Lyra | 0.9920 | 0.9986 | **1.0000** | **1.0000** | **0.9992** | **1.0000** | **1.0000** | 0.9986 | **0.5289** | **1.0000** | **0.5289** | **1.0000** | **0.5289** |
| Halo Uncensored | **0.9990** | **0.9993** | **1.0000** | **1.0000** | **0.9992** | **1.0000** | **1.0000** | **0.9993** | 0.5035 | **1.0000** | 0.5035 | **1.0000** | 0.5035 |

---

## Dual-skill RAG DEV

### Supported / abstain / procedure

| Name | Oracle supp | Real supp | Oracle abstain | Real abstain | Oracle proc | Real proc | Oracle unsup. confident | Real unsup. confident | RAG hard fails | RAG intel pass | RAG mean Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pulse | 0.8509 | 0.6208 | 0.8705 | **0.7266** | 0.7267 | 0.7133 | — | — | — | — | — |
| Aegis | **1.0000** | **0.7358** | **1.0000** | — | — | **0.9533** | — | — | — | — | — |
| Halo | 0.8472 | 0.6283 | 0.6906 | **0.7266** | 0.7133 | 0.7000 | 0.3094 | **0.2734** | 0 | **1.0000** | 0.6560 |
| Nimbus | 0.8434 | 0.6075 | 0.7770 | **0.7266** | 0.7000 | 0.6933 | — | — | — | — | — |
| Lyra | 0.8415 | 0.6283 | 0.7266 | **0.7266** | 0.6933 | 0.7000 | 0.2734 | **0.2734** | 0 | **1.0000** | 0.6620 |
| Halo Uncensored | 0.8698 | 0.6491 | 0.0000 | **0.7266** | 0.7933 | 0.7733 | **1.0000** | **0.2734** | **9** | 0.9940 | 0.6295 |

### Oracle / real diagnostics

Pulse / Aegis: these fields were not stored in SELECTION_REPORT.

| Name | Oracle mean Q | Real mean Q | Oracle calib. | Real calib. | Oracle conflict safe | Real conflict ack | Oracle inject | Stale prec. | Social usable | Faithfulness | Oracle median tok | Real median tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pulse | — | — | — | — | — | — | — | — | — | — | — | — |
| Aegis | — | — | — | — | — | — | — | — | — | — | — | — |
| Halo | 0.6642 | 0.6479 | 0.3947 | **0.5000** | **1.0000** | **0.6032** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **7.0** | 8.0 |
| Nimbus | **0.6982** | 0.6560 | **0.4868** | **0.5000** | **1.0000** | **0.6032** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **7.0** | **10.0** |
| Lyra | 0.6725 | 0.6516 | 0.4342 | **0.5000** | **1.0000** | **0.6032** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **7.0** | 8.0 |
| Halo Uncensored | 0.6003 | **0.6588** | 0.0000 | **0.5000** | **1.0000** | **0.6032** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **7.0** | **10.0** |

Oracle conflict explicit acknowledgment is 1.0000 for the four 28L cores. Real_rag P90 completion tokens is 14 for those four. Oracle_rag P90 completion tokens is 14 for those four. Prompt-injection and stale/current precedence are 1.0000 on both oracle_rag and real_rag for those four.

---

## Phase 8 sealed chat (110; archived Pulse 1.0.0-rc1 only)

Different control than dual-skill DEV (`one_sentence` vs `eos_only`). Applies to **`_superseded/Rineya-Pulse-1.0.0-rc1`**, not active Pulse 1.6.0.

| Name | Cases | Chat Q | Task adequacy | Intel pass | Hard fails | Safety | Median tok | P90 tok | Duplicate rate | Quality pass (diag.) | Min Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pulse 1.0 (archived) | **110** | **0.8524** | **0.4743** | **1.0000** | **0** | **1.0000** | **7.0** | **12.0** | **0.3909** | **1.0000** | **0.71** |
| Aegis | — | — | — | — | — | — | — | — | — | — | — |
| Halo | — | — | — | — | — | — | — | — | — | — | — |
| Nimbus | — | — | — | — | — | — | — | — | — | — | — |
| Lyra | — | — | — | — | — | — | — | — | — | — | — |
| Halo Uncensored | — | — | — | — | — | — | — | — | — | — | — |

Pulse Phase 8 human decision (not a numeric score): `PASS`, reviewer Cheng Li, 2026-08-05, version `1.0.0-rc1` (`reports/PHASE_8_STOP_GATE.md`).

---

## Recorded human-review fields

| Name | Recorded field | When |
|---|---|---|
| Pulse | Phase 8 human commercial `PASS` | 2026-08-05 |
| Aegis | `CORE.json` `human_review.decision` = `pass_public_announcement` | 2026-08-19 |
| Halo | `CORE.json` `human_review.decision` = `pass_public_announcement` | 2026-08-19 |
| Nimbus | `CORE.json` `human_review.decision` = `pass_public_announcement` | 2026-08-19 |
| Lyra | `CORE.json` `human_review.decision` = `pass_public_announcement` | 2026-08-19 |
| Halo Uncensored | no `human_review` block; `status` = `recorded_uncensored_pair` | 2026-08-19 |
