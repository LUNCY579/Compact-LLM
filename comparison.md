# Family eval comparison

Numbers only. No ranking. Geometry, recipes, and dates live in [`lineage.md`](lineage.md). Public-copy rules live in [`announcement.md`](announcement.md). Index: [`roadmap.md`](roadmap.md).

**Bold** = the largest stored number in that column (ties all bold), including rates that count defects. `—` = that field was not stored or that eval was not run for this core.

Every named family core appears in every table: Pulse, Aegis, Halo, Nimbus, Lyra, Halo Uncensored.

Last compiled: 2026-08-19.

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
| Pulse | 0/64 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3125 | 0.9062 | **5.0625** |
| Aegis | 4/64 | 0.0625 | 0.0625 | 0.0000 | 0.0000 | 0.0625 | 0.5625 | **1.0000** | 2.6094 |
| Halo | 30/64 | 0.4688 | 0.4688 | **0.1250** | **0.3750** | 0.4375 | 0.5625 | 0.9062 | 3.2500 |
| Nimbus | 24/64 | 0.3750 | 0.3750 | **0.1250** | 0.1250 | 0.2500 | 0.4375 | 0.9062 | 3.1719 |
| Lyra | 28/64 | 0.4375 | 0.4375 | **0.1250** | 0.3125 | 0.4375 | 0.6250 | 0.9062 | 3.2344 |
| Halo Uncensored | **31/64** | **0.4844** | **0.4844** | 0.0000 | **0.3750** | **0.5625** | **0.6875** | 0.9844 | 2.5781 |

Coverage is 64/64 on all six. All-four-variants-correct is 0.0000 on all six. ≤12-word source values: Pulse / Halo / Nimbus / Lyra `0.90625`; Halo Uncensored `0.984375`.

### By variant (strict exact, 16 rows each)

| Name | Base | Paraphrase | Counterfactual | Unanswerable |
|---|---:|---:|---:|---:|
| Pulse | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Aegis | 0.1250 | 0.0625 | 0.0625 | 0.0000 |
| Halo | 0.5000 | 0.6875 | **0.5625** | **0.1250** |
| Nimbus | 0.2500 | 0.6875 | 0.4375 | **0.1250** |
| Lyra | 0.4375 | 0.6875 | 0.5000 | **0.1250** |
| Halo Uncensored | **0.5625** | **0.8750** | 0.5000 | 0.0000 |

### By difficulty (strict exact)

| Name | Hard (n=40) | Medium (n=24) |
|---|---:|---:|
| Pulse | 0.0000 | 0.0000 |
| Aegis | 0.1000 | 0.0000 |
| Halo | 0.3500 | **0.6667** |
| Nimbus | 0.3000 | 0.5000 |
| Lyra | 0.3250 | 0.6250 |
| Halo Uncensored | **0.4250** | 0.5833 |

### By needle profile (strict exact, 16 rows each)

| Name | Distributed | Early | Late | Middle |
|---|---:|---:|---:|---:|
| Pulse | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Aegis | 0.0000 | 0.0625 | 0.1875 | 0.0000 |
| Halo | **0.6875** | 0.1875 | **0.6250** | 0.3750 |
| Nimbus | 0.6250 | 0.1250 | 0.5000 | 0.2500 |
| Lyra | **0.6875** | 0.1250 | **0.6250** | 0.3125 |
| Halo Uncensored | 0.6250 | **0.2500** | **0.6250** | **0.4375** |

### By length bucket (strict exact)

| Name | Compact long (n=32) | Extended long (n=16) | Standard long (n=16) |
|---|---:|---:|---:|
| Pulse | 0.0000 | 0.0000 | 0.0000 |
| Aegis | 0.0938 | 0.0000 | 0.0625 |
| Halo | 0.4375 | **0.5625** | 0.4375 |
| Nimbus | 0.3438 | 0.3750 | 0.4375 |
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
| Pulse | 0/64 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9062 |
| Aegis | 4/64 | 0.0625 | 0.0000 | 0.0000 | 0.0625 | 0.0000 | **1.0000** |
| Nimbus | 24/64 | 0.3750 | 0.1250 | 0.1250 | 0.2500 | 0.0000 | 0.9062 |
| Lyra | 28/64 | 0.4375 | 0.1250 | 0.3125 | 0.4375 | 0.0000 | 0.9062 |
| Halo | 30/64 | 0.4688 | 0.1250 | 0.3750 | 0.4375 | 0.0000 | 0.9062 |
| Halo Uncensored | 31/64 | 0.4844 | 0.0000 | 0.3750 | 0.5625 | 0.0000 | 0.9844 |
| Qwen3-0.6B | 14/64 | 0.2188 | 0.1875 | 0.1250 | 0.1250 | 0.0000 | 0.8906 |
| Qwen3-4B (NF4) | 43/64 | 0.6719 | 0.3125 | 0.6875 | 0.6875 | 0.2500 | 0.9219 |
| Qwen3-8B (NF4) | **53/64** | **0.8281** | **0.4375** | **0.9375** | **0.9375** | **0.4375** | 0.9844 |

---

## Dual composite (recorded formula)

`0.30·chat_Q + 0.15·task + 0.20·oracle_supp + 0.15·real_supp + 0.10·real_proc + 0.10·oracle_abstain`. Pulse has no recorded composite in SELECTION_REPORT.

| Name | Dual composite |
|---|---:|
| Pulse | — |
| Aegis | **0.8382** |
| Halo | 0.7381 |
| Nimbus | 0.7814 |
| Lyra | 0.7445 |
| Halo Uncensored | 0.6823 |

---

## Dual-skill chat DEV (510)

| Name | Chat Q | Task adequacy | Median tok | P90 tok | Duplicate rate | 1-sentence viol. | Legacy heur. | Intel pass | Hard fails | Safety | Min Q | Quality pass (diag.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pulse | 0.7984 | 0.2952 | 7.0 | — | — | — | — | — | — | — | — | — |
| Aegis | 0.8594 | 0.4978 | — | — | — | — | — | — | — | — | — | — |
| Halo | 0.8619 | 0.5117 | **11.0** | 17.0 | 0.3627 | **0.5294** | 0.9850 | **1.0000** | **0** | **1.0000** | **0.39** | 0.9961 |
| Nimbus | 0.8602 | 0.5032 | **11.0** | **18.0** | 0.3627 | 0.5196 | 0.9837 | **1.0000** | **0** | **1.0000** | **0.39** | **0.9980** |
| Lyra | **0.8667** | **0.5289** | **11.0** | 17.0 | 0.3412 | **0.5294** | 0.9869 | **1.0000** | **0** | **1.0000** | **0.39** | 0.9961 |
| Halo Uncensored | 0.8603 | 0.5035 | 9.0 | 12.0 | **0.4353** | 0.4451 | **0.9987** | **1.0000** | **0** | **1.0000** | **0.39** | **0.9980** |

Pulse / Aegis columns beyond SELECTION_REPORT were not stored in that report. 28L `contract_fulfillment` on this chat DEV is 1.0000 for Halo, Nimbus, Lyra, and Halo Uncensored.

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
| Pulse | 0.3189 | 0.1774 | 0.0000 | — | — | 0.0667 | — | — | — | — | — |
| Aegis | **1.0000** | **0.7358** | **1.0000** | — | — | **0.9533** | — | — | — | — | — |
| Halo | 0.8472 | 0.6283 | 0.6906 | **0.7266** | 0.7133 | 0.7000 | 0.3094 | **0.2734** | 0 | **1.0000** | 0.6560 |
| Nimbus | 0.8906 | 0.6736 | 0.8273 | **0.7266** | **0.8667** | 0.8600 | 0.1727 | **0.2734** | 2 | 0.9987 | **0.6771** |
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

## Phase 8 sealed chat (110; Pulse protocol)

Different control than dual-skill DEV (`one_sentence` vs `eos_only`). Not mixed into the DEV tables.

| Name | Cases | Chat Q | Task adequacy | Intel pass | Hard fails | Safety | Median tok | P90 tok | Duplicate rate | Quality pass (diag.) | Min Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pulse | **110** | **0.8524** | **0.4743** | **1.0000** | **0** | **1.0000** | **7.0** | **12.0** | **0.3909** | **1.0000** | **0.71** |
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
