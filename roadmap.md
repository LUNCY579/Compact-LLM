# Rineya local core roadmap

Developer ledger of **officially recorded local models** and the **base plus every dataset** that produced them.

Weights live under `Rineya_Published_Cores/` (gitignored). Provenance is `CORE.json` plus this file. Phase artifacts under `short-response-model-phase-0-1-2/artifacts/` are **sources**, not the recorded vault.

**Rule.** A model is “officially recorded” only after it has a vault directory, a `core_id`, and a `CORE.json`. Adapters, unfinished runs, and eval-only outputs stay in the working notes below.

Last updated: 2026-08-19. Family eval evidence tables compiled for manual review. Times below are UTC unless marked otherwise.

---

## Public announcement board

Human DEV chat review on 2026-08-19: auto Phase 1 scores are **reference only**. Replies from Nimbus / Halo / Lyra are coherent, short, and usable. Gate **passed**. These four cores (plus Pulse, already 1.0) may be named in public copy. **Rineya 1.0 commercial claim stays on Pulse.** Sealed chat/RAG TEST is still `not_run` and does **not** block this announcement.

| Name | Role to announce | Status | Next dated item |
|---|---|---|---|
| **Pulse** | Commercial 1.0 short-chat teacher (19L) | Phase 8 human PASS (2026-08-05) | Website 1.0 copy as today |
| **Aegis** | 19L dual-skill (chat + grounded RAG) | `public_announcement_ready` | Optional sealed TEST; DPO not started |
| **Halo** | 28L family; strongest frozen long-context among 50/50 (30/64) | `public_announcement_ready` | Optional sealed TEST |
| **Halo Uncensored** | Halo donor α=1.0 replace; uncensored pair (31/64; oracle abstain 0.00) | `recorded_uncensored_pair` | Censored vs uncensored experiments vs Halo |
| **Nimbus** | 28L family; Aegis donor; strongest 28L RAG of the three | `public_announcement_ready` | Optional sealed TEST |
| **Lyra** | 28L family; chat auto-Q highest of the three (0.867) | `public_announcement_ready` | Optional sealed TEST |

Do **not** announce Halo Uncensored as dual-skill or as Rineya 1.0. Pair it with **Halo** for censored vs uncensored work. Do not announce unfinished student runs or raw v2 adapters.

**How to talk about the 28L family.** Chat quality matches 19L Aegis (~0.86 auto; human pass). Long-context showcase beats official Qwen3-0.6B (14/64) and 19L Aegis (4/64). Grounded RAG is still weaker than 19L Aegis (Nimbus best of the three: oracle 0.891 / real 0.674 vs Aegis 1.0 / 0.736). Keep Aegis as the 19L RAG specialist in public copy.

Read-only audit of the 50/50 chat DEV runs (510 rows each): hard failures **0**, intelligibility **1.0**, safety_boundary **1.0**. Low auto `task_adequacy` (~0.50) is scorer prompt-mode mismatch (generic acknowledgments vs a canned reference), not incoherence. A few fluent replies are world-knowledge-wrong (wrong DPO expansions) while still scoring ~0.72 — the scorer does not check that.

### Timeline

| When (UTC) | Event | Track |
|---|---|---|
| 2026-08-05 | Pulse Phase 8 human commercial PASS (Cheng Li) | done |
| 2026-08-17 | Aegis dual-SFT v1 `5e-4` LR freeze accepted | done |
| 2026-08-19 00:23–00:25 | Vault Pulse + Aegis | done |
| 2026-08-19 18:55–19:56 | Vault Halo, Nimbus, Lyra (50/50 28L) | done |
| 2026-08-19 18:57–20:41 | 28L Phase 1 chat DEV + Phase 9 RAG DEV | done |
| 2026-08-19 ~20:47 | Human DEV chat pass; public announcement ready | **done** |
| 2026-08-19 ~21:00 | Vault Halo Uncensored (Halo donor α=1.0 replace) | **done** |
| next | Public / website copy for the named family (Pulse 1.0 + four others) | ready to write |
| next | Optional sealed chat TEST (110) + RAG TEST | `not_run`; not blocking announcement |
| next | DPO on Pulse/Aegis lineage | `not_run` |
| later | Student cores | not a publish candidate yet |

---

## Recorded cores (vault)

| Recorded (UTC) | Name | Version | Status | Local directory | Weight SHA-256 (prefix) |
|---|---|---|---|---|---|
| 2026-08-19 00:23:36 | **Pulse** | 1.0.0-rc1 | Phase 8 human commercial PASS | `Rineya_Published_Cores/Rineya-Pulse-1.0.0-rc1` | `58ce704c…` |
| 2026-08-19 00:25:16 | **Aegis** | 1.1.0-devfreeze | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Aegis-1.1.0-devfreeze` | `15fab4e7…` |
| 2026-08-19 18:55:44 | **Halo** | 1.2.0-devfreeze | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze` | `38fb96fe…` |
| 2026-08-19 19:54:38 | **Nimbus** | 1.3.0-devfreeze | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Nimbus-1.3.0-devfreeze` | `7ae45098…` |
| 2026-08-19 19:54:59 | **Lyra** | 1.4.0-devfreeze | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Lyra-1.4.0-devfreeze` | `87c57b8e…` |
| 2026-08-19 21:00:00 | **Halo Uncensored** | 1.5.0-devfreeze | recorded uncensored pair for Halo; oracle abstain 0.00 | `Rineya_Published_Cores/Rineya-Halo-Uncensored-1.5.0-devfreeze` | `c5d4a17f…` |

Pulse is a dense copy of the Phase 8 teacher. Aegis is that same base with teacher dual-SFT v1 `5e-4` LoRA merged in. Halo, Nimbus, and Lyra are 28-layer Qwen3 with kept blocks 50/50-averaged against different 19-layer donors. Halo Uncensored uses the same Halo donor with kept blocks **fully replaced** (α=1.0). Sources were not overwritten.

---

## Family eval evidence

Numbers only. **Bold** = the largest stored number in that column (ties all bold), including rates that count defects. `—` = that field was not stored or that eval was not run for this core.

Every named family core appears in every table: Pulse, Aegis, Halo, Nimbus, Lyra, Halo Uncensored.

### What was measured

| Eval | Protocol | Items | Source |
|---|---|---:|---|
| Frozen long-context showcase | greedy, 24 new tokens, context 4096, seed 17, thinking off | 64 | Pulse: `Rineya_Long_Context_Demo_v1/outputs/greedy_soup_v2_showcase_report.json` (Phase 8 accepted teacher = Pulse). Aegis: `…/aegis_showcase_report.json`. 28L: `Rineya_Qwen3_CrossDepth_Soup_Mini_v1/outputs/*_report.json` |
| Dual-skill chat DEV | Phase 1 `dev.jsonl`, 32 tokens, `eos_only`, quality v4.5 | 510 | Pulse / Aegis: `short-response-model-phase-0-1-2/artifacts/teacher_dual_sft_v1/lr_sweep/SELECTION_REPORT.md` (2026-08-17). Aegis there is PEFT on Pulse; vault Aegis is the merged dense file. 28L: `Rineya_Qwen3_CrossDepth_Soup_Mini_v1/outputs/phase_evals/*/chat_dev/summary.json` |
| Dual-skill RAG DEV | Phase 9 `dev.reviewed.jsonl`, oracle_rag + real_rag, `query_aware_v21` | 752×2 | Pulse / Aegis: same SELECTION_REPORT. 28L: `…/phase_evals/*/rag_dev/summary.json` and `…/phase_evals/scoreboard.json` |
| Dual composite | `0.30·chat_Q + 0.15·task + 0.20·oracle_supp + 0.15·real_supp + 0.10·real_proc + 0.10·oracle_abstain` | — | Recorded on Aegis (SELECTION_REPORT) and on 28L (`phase_evals/scoreboard.md`). Pulse has no recorded composite |
| Phase 8 sealed chat | 110 rows, `one_sentence`, 32 tokens, quality v4.5 | 110 | Pulse only: `short-response-model-phase-0-1-2/reports/RELEASE_CANDIDATE_AUDIT.json` |
| Dual-skill sealed chat TEST / RAG TEST | — | — | `not_run` for all six family cores |

### Identity

| Name | Version | Layers | Stitch α | Donor | Weight SHA prefix |
|---|---|---:|---:|---|---|
| Pulse | 1.0.0-rc1 | 19 | — | Phase 8 accepted teacher (copy) | `58ce704c` |
| Aegis | 1.1.0-devfreeze | 19 | — | Pulse + v1@5e-4 LoRA | `15fab4e7` |
| Halo | 1.2.0-devfreeze | 28 | 0.5 | v1@5e-4 + v2@1e-4 soup | `38fb96fe` |
| Nimbus | 1.3.0-devfreeze | 28 | 0.5 | vault Aegis | `7ae45098` |
| Lyra | 1.4.0-devfreeze | 28 | 0.5 | v1@5e-4 + v1@1e-5 soup | `87c57b8e` |
| Halo Uncensored | 1.5.0-devfreeze | 28 | 1.0 | same donor as Halo | `c5d4a17f` |

### Eval coverage

| Eval | Pulse | Aegis | Halo | Nimbus | Lyra | Halo Uncensored |
|---|---|---|---|---|---|---|
| Showcase 64 | run | run | run | run | run | run |
| Chat DEV 510 | run (sweep baseline) | run (sweep selected, PEFT) | run | run | run | run |
| RAG DEV 752×2 | run (sweep baseline) | run (sweep selected, PEFT) | run | run | run | run |
| Dual composite | — | run | run | run | run | run |
| Phase 8 sealed chat 110 (`one_sentence`) | run | — | — | — | — | — |
| Dual-skill sealed chat TEST / RAG TEST | `not_run` | `not_run` | `not_run` | `not_run` | `not_run` | `not_run` |

### Showcase (64-row frozen long-context)

| Name | Hits / 64 | Strict exact | First-line exact | Unanswerable | CF pair correct+changed | Base+paraphrase both | Base/paraphrase agree | ≤12-word rate | Mean response words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pulse | 0/64 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3125 | 0.9062 | **5.0625** |
| Aegis | 4/64 | 0.0625 | 0.0625 | 0.0000 | 0.0000 | 0.0625 | 0.5625 | **1.0000** | 2.6094 |
| Halo | 30/64 | 0.4688 | 0.4688 | **0.1250** | **0.3750** | 0.4375 | 0.5625 | 0.9062 | 3.2500 |
| Nimbus | 24/64 | 0.3750 | 0.3750 | **0.1250** | 0.1250 | 0.2500 | 0.4375 | 0.9062 | 3.1719 |
| Lyra | 28/64 | 0.4375 | 0.4375 | **0.1250** | 0.3125 | 0.4375 | 0.6250 | 0.9062 | 3.2344 |
| Halo Uncensored | **31/64** | **0.4844** | **0.4844** | 0.0000 | **0.3750** | **0.5625** | **0.6875** | 0.9844 | 2.5781 |

Coverage is 64/64 on all six. All-four-variants-correct is 0.0000 on all six. ≤12-word source values: Pulse / Halo / Nimbus / Lyra `0.90625`; Halo Uncensored `0.984375`.

### Showcase by variant (strict exact, 16 rows each)

| Name | Base | Paraphrase | Counterfactual | Unanswerable |
|---|---:|---:|---:|---:|
| Pulse | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Aegis | 0.1250 | 0.0625 | 0.0625 | 0.0000 |
| Halo | 0.5000 | 0.6875 | **0.5625** | **0.1250** |
| Nimbus | 0.2500 | 0.6875 | 0.4375 | **0.1250** |
| Lyra | 0.4375 | 0.6875 | 0.5000 | **0.1250** |
| Halo Uncensored | **0.5625** | **0.8750** | 0.5000 | 0.0000 |

### Showcase by difficulty (strict exact)

| Name | Hard (n=40) | Medium (n=24) |
|---|---:|---:|
| Pulse | 0.0000 | 0.0000 |
| Aegis | 0.1000 | 0.0000 |
| Halo | 0.3500 | **0.6667** |
| Nimbus | 0.3000 | 0.5000 |
| Lyra | 0.3250 | 0.6250 |
| Halo Uncensored | **0.4250** | 0.5833 |

### Showcase by needle profile (strict exact, 16 rows each)

| Name | Distributed | Early | Late | Middle |
|---|---:|---:|---:|---:|
| Pulse | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Aegis | 0.0000 | 0.0625 | 0.1875 | 0.0000 |
| Halo | **0.6875** | 0.1875 | **0.6250** | 0.3750 |
| Nimbus | 0.6250 | 0.1250 | 0.5000 | 0.2500 |
| Lyra | **0.6875** | 0.1250 | **0.6250** | 0.3125 |
| Halo Uncensored | 0.6250 | **0.2500** | **0.6250** | **0.4375** |

### Showcase by length bucket (strict exact)

| Name | Compact long (n=32) | Extended long (n=16) | Standard long (n=16) |
|---|---:|---:|---:|
| Pulse | 0.0000 | 0.0000 | 0.0000 |
| Aegis | 0.0938 | 0.0000 | 0.0625 |
| Halo | 0.4375 | **0.5625** | 0.4375 |
| Nimbus | 0.3438 | 0.3750 | 0.4375 |
| Lyra | 0.4063 | 0.5000 | 0.4375 |
| Halo Uncensored | **0.4688** | 0.3750 | **0.6250** |

### Showcase reference: official Qwen3-0.6B (not a family core)

Same protocol as the family showcase. Not included in the bold maxima above.

| Name | Hits / 64 | Strict exact | Unanswerable | CF pair correct+changed | Base+paraphrase both | Base/paraphrase agree | ≤12-word | Mean words | Base | Para | CF | Unans | Hard | Medium |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 14/64 | 0.2188 | 0.1875 | 0.1250 | 0.1250 | 0.4375 | 0.8906 | 4.4063 | 0.2500 | 0.1875 | 0.2500 | 0.1875 | 0.2250 | 0.2083 |

### Dual composite (recorded formula)

`0.30·chat_Q + 0.15·task + 0.20·oracle_supp + 0.15·real_supp + 0.10·real_proc + 0.10·oracle_abstain`. Pulse has no recorded composite in SELECTION_REPORT.

| Name | Dual composite |
|---|---:|
| Pulse | — |
| Aegis | **0.8382** |
| Halo | 0.7381 |
| Nimbus | 0.7814 |
| Lyra | 0.7445 |
| Halo Uncensored | 0.6823 |

### Dual-skill chat DEV (510)

| Name | Chat Q | Task adequacy | Median tok | P90 tok | Duplicate rate | 1-sentence viol. | Legacy heur. | Intel pass | Hard fails | Safety | Min Q | Quality pass (diag.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pulse | 0.7984 | 0.2952 | 7.0 | — | — | — | — | — | — | — | — | — |
| Aegis | 0.8594 | 0.4978 | — | — | — | — | — | — | — | — | — | — |
| Halo | 0.8619 | 0.5117 | **11.0** | 17.0 | 0.3627 | **0.5294** | 0.9850 | **1.0000** | **0** | **1.0000** | **0.39** | 0.9961 |
| Nimbus | 0.8602 | 0.5032 | **11.0** | **18.0** | 0.3627 | 0.5196 | 0.9837 | **1.0000** | **0** | **1.0000** | **0.39** | **0.9980** |
| Lyra | **0.8667** | **0.5289** | **11.0** | 17.0 | 0.3412 | **0.5294** | 0.9869 | **1.0000** | **0** | **1.0000** | **0.39** | 0.9961 |
| Halo Uncensored | 0.8603 | 0.5035 | 9.0 | 12.0 | **0.4353** | 0.4451 | **0.9987** | **1.0000** | **0** | **1.0000** | **0.39** | **0.9980** |

Pulse / Aegis columns beyond SELECTION_REPORT were not stored in that report. 28L `contract_fulfillment` on this chat DEV is 1.0000 for Halo, Nimbus, Lyra, and Halo Uncensored.

### Dual-skill chat DEV quality dimensions (v4.5 means)

Pulse / Aegis: only `task_adequacy` is in SELECTION_REPORT. Other dimension cells for those two are —.

| Name | Brevity | Comprehens. | Context cons. | Contract | Naturalness | Factual | Logic/fact | Mechanical | Prompt-fn align | Completeness | Mode fulfill | Safety | Task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pulse | — | — | — | — | — | — | — | — | — | — | — | — | 0.2952 |
| Aegis | — | — | — | — | — | — | — | — | — | — | — | — | 0.4978 |
| Halo | 0.9908 | 0.9986 | **1.0000** | **1.0000** | **0.9992** | **1.0000** | **1.0000** | 0.9986 | 0.5117 | **1.0000** | 0.5117 | **1.0000** | 0.5117 |
| Nimbus | 0.9913 | **0.9993** | **1.0000** | **1.0000** | **0.9992** | **1.0000** | **1.0000** | **0.9993** | 0.5032 | **1.0000** | 0.5032 | **1.0000** | 0.5032 |
| Lyra | 0.9920 | 0.9986 | **1.0000** | **1.0000** | **0.9992** | **1.0000** | **1.0000** | 0.9986 | **0.5289** | **1.0000** | **0.5289** | **1.0000** | **0.5289** |
| Halo Uncensored | **0.9990** | **0.9993** | **1.0000** | **1.0000** | **0.9992** | **1.0000** | **1.0000** | **0.9993** | 0.5035 | **1.0000** | 0.5035 | **1.0000** | 0.5035 |

### Dual-skill RAG DEV (supported / abstain / procedure)

| Name | Oracle supp | Real supp | Oracle abstain | Real abstain | Oracle proc | Real proc | Oracle unsup. confident | Real unsup. confident | RAG hard fails | RAG intel pass | RAG mean Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pulse | 0.3189 | 0.1774 | 0.0000 | — | — | 0.0667 | — | — | — | — | — |
| Aegis | **1.0000** | **0.7358** | **1.0000** | — | — | **0.9533** | — | — | — | — | — |
| Halo | 0.8472 | 0.6283 | 0.6906 | **0.7266** | 0.7133 | 0.7000 | 0.3094 | **0.2734** | 0 | **1.0000** | 0.6560 |
| Nimbus | 0.8906 | 0.6736 | 0.8273 | **0.7266** | **0.8667** | 0.8600 | 0.1727 | **0.2734** | 2 | 0.9987 | **0.6771** |
| Lyra | 0.8415 | 0.6283 | 0.7266 | **0.7266** | 0.6933 | 0.7000 | 0.2734 | **0.2734** | 0 | **1.0000** | 0.6620 |
| Halo Uncensored | 0.8698 | 0.6491 | 0.0000 | **0.7266** | 0.7933 | 0.7733 | **1.0000** | **0.2734** | **9** | 0.9940 | 0.6295 |

### Dual-skill RAG DEV (oracle_rag / real_rag diagnostics)

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

### Phase 8 sealed chat (110; Pulse protocol)

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

### Recorded human-review fields

| Name | Recorded field | When |
|---|---|---|
| Pulse | Phase 8 human commercial `PASS` | 2026-08-05 |
| Aegis | `CORE.json` `human_review.decision` = `pass_public_announcement` | 2026-08-19 |
| Halo | `CORE.json` `human_review.decision` = `pass_public_announcement` | 2026-08-19 |
| Nimbus | `CORE.json` `human_review.decision` = `pass_public_announcement` | 2026-08-19 |
| Lyra | `CORE.json` `human_review.decision` = `pass_public_announcement` | 2026-08-19 |
| Halo Uncensored | no `human_review` block; `status` = `recorded_uncensored_pair` | 2026-08-19 |

---

## Pulse (`rineya.core.pulse.v1.0.0-rc1.19L.454M.dense`)

**What it is.** 19-layer Qwen3 dense checkpoint, hidden size 1024, ~454M, `model.safetensors` 1,817,909,640 bytes. Local one-line context-faithful chat. Dynamic sparse routing is not in this release.

**Official dates**

| When | Event |
|---|---|
| 2026-08-02 | Phase 2B frontier used by the soup (`phase_2b_quality_v4_full_20260802_213049`, stage 09 drop to 19 layers) |
| 2026-08-05 | Phase 8 human PASS, reviewer Cheng Li, version `1.0.0-rc1` (`reports/PHASE_8_STOP_GATE.md`) |
| 2026-08-19 00:23:36–00:23:42 | Copied into the published-cores vault (CPU, sources unmodified) |

**Base model (ancestry, in order)**

1. Hugging Face **`Qwen/Qwen3-0.6B`** (~28 decoder layers, ~600M). Alibaba pretraining; not a local dataset.
2. Local dense teacher after Phase 2B reduction **28 → 19** layers:  
   `short-response-model-phase-0-1-2/artifacts/phase_2b_quality_v4_full_20260802_213049/frontier/stage_09_drop_14_24_6_23_8_10_18_9_7/model`
3. Phase 4 greedy soup (equal 0.5 / 0.5) of LoRA ingredients **`seed_37` + `seed_49`**, merged to a full HF checkpoint:  
   `short-response-model-phase-0-1-2/artifacts/phase_4_model_soups_phase2c_balanced_training/accepted_model`  
   (soup manifest also records that `seed_23` and `phase3_primary` were tried and rejected)
4. Vault copy **Pulse** (same weight SHA as that accepted model: `58ce704c729485fc8a83094c432a0e591959187587ba2b5eab9809c23d402b68`)

**All local datasets in this lineage**

| Stage | Dataset | Path | Rows / notes |
|---|---|---|---|
| Phase 0–1 / 2C base SFT | Original short-response train | `short-response-model-phase-0-1-2/data/train/train.jsonl` | 2,950 |
| Phase 2C recovery add-on | Targeted concise-reasoning templates | mixed into `short-response-model-phase-0-1-2/data/phase_2c_targeted_recovery/train.jsonl` | +124 → **3,074** train |
| Phase 2C eval mix | Recovery DEV | `…/data/phase_2c_targeted_recovery/dev.jsonl` | 570 (510 base DEV + 60) |
| Phase 2B probe (reduction, not SFT mix) | Structural probe | `short-response-model-phase-0-1-2/data/phase_2_structural_reduction/probe_150_thorough_v2.jsonl` | probe-only |
| Held-out eval (not training) | Chat DEV / sealed | `data/dev/dev.jsonl` (510), `data/sealed/sealed.jsonl` (110) | never mix into train |

Pulse itself is a **merge of adapters**, not a new SFT run. The soup ingredients were trained on the Phase 2C train mix above, on the reduced 19-layer teacher.

**Developer notes.** Website / commercial **1.0 claim stays on Pulse**. Aegis / Halo / Nimbus / Lyra may be named publicly as additional cores (2026-08-19 human DEV chat pass); they are not a replacement 1.0. Pulse showcase long-context exact-match on 2026-08-19 was **0/64** (verbose replies, wrong entities); that is an eval of Pulse, not a new core.

---

## Aegis (`rineya.core.aegis.v1.1.0-devfreeze.19L.454M.dual-sft-v1-lr5e4`)

**What it is.** Same 19-layer / 1024 / ~454M dense file after merging teacher dual-SFT v1 LoRA at **`5e-4`**. Dual-skill 19L (short chat + grounded RAG). Public announcement ready. Sealed TEST not run.

**Official dates**

| When | Event |
|---|---|
| 2026-08-16 | Dual-SFT v1 mix seed `20260816`; QLoRA train label `teacher_dual_sft_v1_lr5e4` (1 epoch, 16,410 examples, ~2.1 h GPU) |
| 2026-08-17 | LR sweep **accepted**: `lr5e4`. Sealed chat / RAG TEST / DPO still `not_run`. Rollback = Phase 8 teacher |
| 2026-08-19 00:23:45–00:25:16 | LoRA merged on CPU into Pulse’s base; vault write Aegis (weight SHA `15fab4e7…`) |
| 2026-08-19 ~20:47 | Human pass for public announcement (DEV chat; auto scores reference-only). Sealed TEST still `not_run`. |

**Base model**

- Dense base: **Pulse / Phase 8 accepted teacher** (SHA `58ce704c…`)
- Adapter: `short-response-model-phase-0-1-2/artifacts/teacher_dual_sft_v1/lr_sweep/lr5e4/sft_run/adapter` (SHA `6f765184…`)

**All datasets used to train the Aegis adapter** (`teacher_dual_sft_v1/mix.jsonl`, 16,410 rows)

| Bucket | Dataset | Path | Rows |
|---|---|---|---:|
| Chat / behavior | Rineya Short-Response Behavior Corpus v1.0 combined SFT | `Rineya_Short_Response_Behavior_Corpus_v1.0/data/sft/rineya_short_response_combined_train.jsonl` | 12,000 |
| Grounded RAG | RAG reasoning v3 large, converted from reviewed train | source `short-response-model-phase-0-1-2/data/phase_9_student_distillation/rag_reasoning_v3_large/reviewed_splits/train.reviewed.jsonl` → `artifacts/teacher_dual_sft_v1/rag_v3_train.sft.jsonl` | 4,410 |

Chat fraction ~73%, evidence fraction ~27%. Official `train.reviewed.jsonl` was not modified; assistant targets were copied from `approved_answer` into the converted SFT view.

DEV used during that SFT run: `data/dev/dev.jsonl` (510). Not a training mix member.

Aegis also **inherits Pulse’s entire Phase 0–4 lineage** (Qwen3-0.6B → 19-layer soup). Dual-SFT v1 did not retrain from the 28-layer Hugging Face card.

**Developer notes.** Do not advertise Aegis as Rineya 1.0 (that stays Pulse). Announce it as the 19L dual-skill / RAG specialist. Do not retune the frozen `5e-4` winner after seeing sealed TEST.

---

## Halo (`rineya.core.halo.v1.2.0-devfreeze.28L.qwen3-crossdepth-v1-5e4-v2-1e4`)

**What it is.** Official **28-layer Qwen3-0.6B** with the 19 Phase-2B-kept decoder blocks 50/50-averaged against a rematerialized 19-layer donor: equal FP32 **delta soup** of teacher dual-SFT **v1 @ 5e-4** and **v2 @ 1e-4**, both on Pulse. Dropped Qwen3 layers plus embed / final norm / tied lm_head stay official Qwen3. Public announcement ready (human DEV chat pass 2026-08-19). Sealed TEST not run.

Frozen 64-row showcase (thinking off): **30/64** strict exact vs official Qwen3-0.6B **14/64** and Aegis **4/64**.

**Official dates**

| When | Event |
|---|---|
| 2026-08-19 | Cross-depth lab scored `qwen3_x_v1_5e4_v2` at 30/64 |
| 2026-08-19 18:55:44–18:57:31 | Vault write Halo (CPU stitch; Pulse and Aegis not overwritten; weight SHA `38fb96fe…`) |
| 2026-08-19 ~20:47 | Human pass for public announcement (DEV chat). Sealed TEST still `not_run`. |

**Base / ingredients (not a new 28L SFT run)**

| Piece | Path / identity |
|---|---|
| 28L scaffold | Local `Qwen3-0.6B/` (Hugging Face `Qwen/Qwen3-0.6B`) |
| 19L donor base | **Pulse** / Phase 8 accepted teacher (`58ce704c…`) |
| Donor adapter A | `teacher_dual_sft_v1/lr_sweep/lr5e4/sft_run/adapter` (mix **16,410**) |
| Donor adapter B | `teacher_dual_sft_v2/sft_run/adapter` at **1e-4** (mix **44,410**) |
| Merge | `merge_delta_soup.py` equal weights, then `stitch_cross_depth.py` α=0.5 on kept layers |

**Datasets in the donor adapters**

| Adapter | Mix | Rows | Contents |
|---|---|---:|---|
| v1 5e-4 | `teacher_dual_sft_v1/mix.jsonl` | 16,410 | Short-Response Behavior Corpus v1.0 combined SFT 12,000 + RAG v3 train 4,410 |
| v2 1e-4 | `teacher_dual_sft_v2/mix.jsonl` | 44,410 | General Reasoning Corpus v2.0 recommended 40,000 + RAG v3 train 4,410 |

Halo **inherits Pulse’s Phase 0–4 lineage** plus those two dual-SFT mixes **through weight-space averaging**. It was not trained as a 28-layer model.

**Developer notes.** Do not advertise Halo as Rineya 1.0. Announce it as the 28L long-context family lead among the 50/50 cores (30/64). Its uncensored sibling is **Halo Uncensored** (same donor, α=1.0, 31/64, oracle abstain 0.00). Unanswerable on the frozen showcase stayed 2/16. Phase 1 chat DEV + Phase 9 RAG DEV: `Rineya_Qwen3_CrossDepth_Soup_Mini_v1/outputs/phase_evals/`. Chat quality stayed with Aegis (~0.86); RAG supportedness is below 19L Aegis (Halo oracle 0.847 / real 0.628 vs Aegis 1.0 / 0.736).

---

## Nimbus (`rineya.core.nimbus.v1.3.0-devfreeze.28L.qwen3-crossdepth-aegis`)

**What it is.** Official **28-layer Qwen3-0.6B** with the 19 Phase-2B-kept decoder blocks 50/50-averaged against vault **Aegis**. Dropped Qwen3 layers plus embed / final norm / tied lm_head stay official Qwen3. Public announcement ready (human DEV chat pass 2026-08-19). Sealed TEST not run.

Frozen 64-row showcase (thinking off): **24/64**.

**Official dates**

| When | Event |
|---|---|
| 2026-08-19 | Cross-depth lab scored `qwen3_x_aegis` at 24/64 |
| 2026-08-19 19:54:38–19:54:56 | Vault write Nimbus (CPU stitch; Aegis donor SHA `15fab4e7…` unchanged; weight SHA `7ae45098…`) |
| 2026-08-19 ~20:47 | Human pass for public announcement (DEV chat, including the Nimbus 510-row trace). Sealed TEST still `not_run`. |

**Base / ingredients**

| Piece | Path / identity |
|---|---|
| 28L scaffold | Local `Qwen3-0.6B/` |
| 19L donor | **Aegis** (`15fab4e7…`) |
| Merge | `stitch_cross_depth.py` α=0.5 on kept layers (no new adapter soup) |

Inherits Aegis’s dual-SFT v1 mix (16,410) through the dense Aegis donor. Best 28L RAG of the vaulted three (oracle 0.891 / real 0.674 / oracle abstain 0.827); still below 19L Aegis.

---

## Lyra (`rineya.core.lyra.v1.4.0-devfreeze.28L.qwen3-crossdepth-v1-5e4-1e5`)

**What it is.** Official **28-layer Qwen3-0.6B** with the 19 Phase-2B-kept decoder blocks 50/50-averaged against an equal FP32 delta soup of teacher dual-SFT **v1 @ 5e-4** and **v1 @ 1e-5**, both on Pulse. Public announcement ready (human DEV chat pass 2026-08-19). Sealed TEST not run.

Frozen 64-row showcase (thinking off): **28/64**.

**Official dates**

| When | Event |
|---|---|
| 2026-08-19 | Cross-depth lab scored `qwen3_x_v1_5e4_1e5` at 28/64 |
| 2026-08-19 19:54:59–19:56:26 | Vault write Lyra (CPU stitch; existing cores not overwritten; weight SHA `87c57b8e…`) |
| 2026-08-19 ~20:47 | Human pass for public announcement (DEV chat). Sealed TEST still `not_run`. |

**Base / ingredients**

| Piece | Path / identity |
|---|---|
| 28L scaffold | Local `Qwen3-0.6B/` |
| 19L donor base | **Pulse** (`58ce704c…`) |
| Donor adapter A | `teacher_dual_sft_v1/lr_sweep/lr5e4/sft_run/adapter` (mix **16,410**) |
| Donor adapter B | `teacher_dual_sft_v1/lr_sweep/lr1e5/sft_run/adapter` (same v1 mix, cooler LR) |
| Merge | `merge_delta_soup.py` equal weights, then `stitch_cross_depth.py` α=0.5 |

---

## Halo Uncensored (`rineya.core.halo-uncensored.v1.5.0-devfreeze.28L.qwen3-replace-v1-5e4-v2-1e4`)

**What it is.** Same 28-layer Qwen3 scaffold and the **same 19-layer donor as Halo** (v1@5e-4 + v2@1e-4 soup), but the 19 Phase-2B-kept decoder blocks are **fully replaced** (stitch α=1.0). Dropped Qwen3 layers plus embed / final norm / tied lm_head stay official Qwen3. This is the **uncensored pair** for Halo: compare the two when a later experiment needs a model that answers instead of refusing.

Frozen 64-row showcase (thinking off): **31/64** (best exact-match so far; shorter replies than Halo). Phase 9 RAG DEV oracle abstain **0.00** (Halo 50/50 is 0.691). Chat auto-Q 0.860. Dual-skill score is worse than every 50/50 sibling. **Not** a public dual-skill announcement. Sealed TEST not run.

**Official dates**

| When | Event |
|---|---|
| 2026-08-19 | Lab scored `qwen3_replace_v1_5e4_v2` at 31/64; oracle abstain 0.00 |
| 2026-08-19 20:58:42–21:00:00 | Vault write Halo Uncensored (CPU stitch; Pulse, Aegis, Halo, Nimbus, Lyra not overwritten; weight SHA `c5d4a17f…`) |

**Base / ingredients**

Same table as Halo, except Merge is `stitch_cross_depth.py` **α=1.0** on kept layers.

**How to use the pair.** Halo = 50/50 blend on kept blocks (refuses more on grounded RAG). Halo Uncensored = full replace on kept blocks (answers when Halo would abstain). Lab replace artifacts are kept; do not delete them to save disk.

---

## Local artifacts that are **not** recorded cores

These exist on disk. They are not vault entries.

| Artifact | When / status | Base | Datasets |
|---|---|---|---|
| Teacher dual-SFT **v2 `1e-4`** adapter | Finished GPU SFT (`teacher_dual_sft_v2`, 44,410 rows, ~7.4 h) | Pulse teacher | **v2 mix**: Rineya General Reasoning Corpus v2.0 recommended 40,000 (32,000 new + 8,000 v1 preservation) **+** RAG v3 train 4,410. See `artifacts/teacher_dual_sft_v2/mix_manifest.json` |
| Teacher v2 hot screen **`5e-4` then `3e-4`** | `5e-4` finished; `3e-4` not used for Halo | Pulse teacher | Same v2 mix (44,410) |
| Student **greedy** dual-SFT v2 | Finished CPU LoRA 2026-08-18 window (`student_dual_sft_v2_greedy`, 44,410 rows) | 12-layer `phase_9_capability_composition_v22_lr1e4_s1200/track_d_greedy_soup/final_model` | Same v2 mix. Real RAG DEV dropped vs baseline; not a publish candidate |
| Student **frozen-anchor** dual-SFT v2 | CPU train in progress | `…/frozen_anchor` | Same v2 mix |
| Qwen3 **replace-kept** Aegis / Lyra-donor (α=1.0) | Lab stitch 2026-08-19; weights kept, **not vaulted** | 28L Qwen3 + Nimbus/Lyra donors | Showcase 23 / 30. Oracle abstain 0.30 / 0.29. Halo-donor replace is vaulted as **Halo Uncensored**. |

---

## Datasets on disk that have **not** produced a recorded core

| Dataset | Role | Path |
|---|---|---|
| Rineya General Reasoning and Natural Communication Corpus v2.0 | Teacher/student dual-SFT v2 mix (not in Pulse/Aegis) | `Rineya_General_Reasoning_and_Natural_Communication_Corpus_v2.0/` |
| Rineya Puzzle Reasoning 51K v1.0.0 | Puzzle SFT/eval; no trained core yet | `Rineya_Puzzle_Reasoning_51K_v1.0.0/` |
| Long-Context Demo v1.0 | **Frozen eval only** (do not train on these rows) | `Rineya_Long_Context_Demo_v1/` |
| Sequence-independence dataset v1.0 | Experiment corpus; `split_transitions/train.jsonl` stays local (~368 MB) | `Sequence_Experiment/` |
| Context harness | Runtime, not a weight | `Harness/` |
| Preference / DPO packs (v1 5k, v2 12k) | Not applied to Pulse or Aegis | corpus `data/dpo/` folders |

---

## How to add the next recorded core

1. Materialize with `Rineya_Published_Cores/_materialize_core.py` (refuses overwrite).
2. Confirm `CORE.json` has `core_id`, timestamps, base path, adapter path, and weight SHA.
3. Add a row to the vault table above and a section with **base + every train file**.
4. Leave unfinished adapters in “not recorded cores” until merge + `CORE.json`.
