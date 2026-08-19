# Family lineage

How each named core was made: dates, geometry, donors, and every train mix. Scores live in [`comparison.md`](comparison.md). Public-copy rules live in [`announcement.md`](announcement.md). Index: [`roadmap.md`](roadmap.md).

Weights live under `Rineya_Published_Cores/` (gitignored). Provenance is each core’s `CORE.json` plus this file. Phase artifacts under `short-response-model-phase-0-1-2/artifacts/` are **sources**, not the recorded vault.

**Rule.** A model is “officially recorded” only after it has a vault directory, a `core_id`, and a `CORE.json`.

Times are UTC unless marked otherwise. Last updated: 2026-08-19.

---

## Timeline

| When (UTC) | Event |
|---|---|
| 2026-08-02 | Phase 2B frontier used by Pulse’s soup (`phase_2b_quality_v4_full_20260802_213049`, stage 09 drop to 19 layers) |
| 2026-08-05 | Pulse Phase 8 human commercial PASS (Cheng Li), version `1.0.0-rc1` |
| 2026-08-16 | Dual-SFT v1 mix seed `20260816`; Aegis QLoRA train label `teacher_dual_sft_v1_lr5e4` |
| 2026-08-17 | Aegis dual-SFT v1 `5e-4` LR freeze accepted |
| 2026-08-19 00:23:36–00:23:42 | Vault **Pulse** |
| 2026-08-19 00:23:45–00:25:16 | Vault **Aegis** |
| 2026-08-19 18:55:44–18:57:31 | Vault **Halo** |
| 2026-08-19 18:57–20:41 | 28L Phase 1 chat DEV + Phase 9 RAG DEV |
| 2026-08-19 19:54:38–19:54:56 | Vault **Nimbus** |
| 2026-08-19 19:54:59–19:56:26 | Vault **Lyra** |
| 2026-08-19 ~20:47 | Human DEV chat pass for Aegis / Halo / Nimbus / Lyra |
| 2026-08-19 20:58:42–21:00:00 | Vault **Halo Uncensored** |

Open next items (copy, sealed TEST, DPO) are on [`roadmap.md`](roadmap.md).

---

## Spectrum (what each core is)

| Name | Version | Layers | Stitch α | Donor | Role |
|---|---|---:|---:|---|---|
| Pulse | 1.0.0-rc1 | 19 | — | Phase 8 accepted teacher (copy) | Commercial 1.0 short-chat teacher |
| Aegis | 1.1.0-devfreeze | 19 | — | Pulse + v1@5e-4 LoRA | 19L dual-skill (chat + grounded RAG) |
| Halo | 1.2.0-devfreeze | 28 | 0.5 | v1@5e-4 + v2@1e-4 soup | 28L family; Halo donor blend |
| Nimbus | 1.3.0-devfreeze | 28 | 0.5 | vault Aegis | 28L family; Aegis donor blend |
| Lyra | 1.4.0-devfreeze | 28 | 0.5 | v1@5e-4 + v1@1e-5 soup | 28L family; cooler-LR v1 soup blend |
| Halo Uncensored | 1.5.0-devfreeze | 28 | 1.0 | same donor as Halo | Uncensored pair for Halo (kept blocks replaced) |

Pulse is a dense copy of the Phase 8 teacher. Aegis is that same base with teacher dual-SFT v1 `5e-4` LoRA merged in. Halo, Nimbus, and Lyra are 28-layer Qwen3 with kept blocks 50/50-averaged against different 19-layer donors. Halo Uncensored uses the same Halo donor with kept blocks **fully replaced** (α=1.0). Sources were not overwritten.

| Recorded (UTC) | Name | Status | Local directory | Weight SHA-256 (prefix) |
|---|---|---|---|---|
| 2026-08-19 00:23:36 | **Pulse** | Phase 8 human commercial PASS | `Rineya_Published_Cores/Rineya-Pulse-1.0.0-rc1` | `58ce704c…` |
| 2026-08-19 00:25:16 | **Aegis** | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Aegis-1.1.0-devfreeze` | `15fab4e7…` |
| 2026-08-19 18:55:44 | **Halo** | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze` | `38fb96fe…` |
| 2026-08-19 19:54:38 | **Nimbus** | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Nimbus-1.3.0-devfreeze` | `7ae45098…` |
| 2026-08-19 19:54:59 | **Lyra** | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Lyra-1.4.0-devfreeze` | `87c57b8e…` |
| 2026-08-19 21:00:00 | **Halo Uncensored** | recorded uncensored pair for Halo | `Rineya_Published_Cores/Rineya-Halo-Uncensored-1.5.0-devfreeze` | `c5d4a17f…` |

---

## Pulse (`rineya.core.pulse.v1.0.0-rc1.19L.454M.dense`)

**Spectrum.** 19-layer Qwen3 dense checkpoint, hidden size 1024, ~454M, `model.safetensors` 1,817,909,640 bytes. Local one-line context-faithful chat. Dynamic sparse routing is not in this release. Commercial **1.0** stays on Pulse.

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

---

## Aegis (`rineya.core.aegis.v1.1.0-devfreeze.19L.454M.dual-sft-v1-lr5e4`)

**Spectrum.** Same 19-layer / 1024 / ~454M dense file after merging teacher dual-SFT v1 LoRA at **`5e-4`**. Dual-skill 19L (short chat + grounded RAG). Public announcement ready. Sealed TEST not run.

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

Aegis also **inherits Pulse’s entire Phase 0–4 lineage** (Qwen3-0.6B → 19-layer soup). Dual-SFT v1 did not retrain from the 28-layer Hugging Face card. Do not retune the frozen `5e-4` winner after seeing sealed TEST.

---

## Halo (`rineya.core.halo.v1.2.0-devfreeze.28L.qwen3-crossdepth-v1-5e4-v2-1e4`)

**Spectrum.** Official **28-layer Qwen3-0.6B** with the 19 Phase-2B-kept decoder blocks 50/50-averaged against a rematerialized 19-layer donor: equal FP32 **delta soup** of teacher dual-SFT **v1 @ 5e-4** and **v2 @ 1e-4**, both on Pulse. Dropped Qwen3 layers plus embed / final norm / tied lm_head stay official Qwen3. Uncensored sibling: **Halo Uncensored**.

**Official dates**

| When | Event |
|---|---|
| 2026-08-19 | Cross-depth lab scored `qwen3_x_v1_5e4_v2` |
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

---

## Nimbus (`rineya.core.nimbus.v1.3.0-devfreeze.28L.qwen3-crossdepth-aegis`)

**Spectrum.** Official **28-layer Qwen3-0.6B** with the 19 Phase-2B-kept decoder blocks 50/50-averaged against vault **Aegis**. Dropped Qwen3 layers plus embed / final norm / tied lm_head stay official Qwen3.

**Official dates**

| When | Event |
|---|---|
| 2026-08-19 | Cross-depth lab scored `qwen3_x_aegis` |
| 2026-08-19 19:54:38–19:54:56 | Vault write Nimbus (CPU stitch; Aegis donor SHA `15fab4e7…` unchanged; weight SHA `7ae45098…`) |
| 2026-08-19 ~20:47 | Human pass for public announcement (DEV chat, including the Nimbus 510-row trace). Sealed TEST still `not_run`. |

**Base / ingredients**

| Piece | Path / identity |
|---|---|
| 28L scaffold | Local `Qwen3-0.6B/` |
| 19L donor | **Aegis** (`15fab4e7…`) |
| Merge | `stitch_cross_depth.py` α=0.5 on kept layers (no new adapter soup) |

Inherits Aegis’s dual-SFT v1 mix (16,410) through the dense Aegis donor.

---

## Lyra (`rineya.core.lyra.v1.4.0-devfreeze.28L.qwen3-crossdepth-v1-5e4-1e5`)

**Spectrum.** Official **28-layer Qwen3-0.6B** with the 19 Phase-2B-kept decoder blocks 50/50-averaged against an equal FP32 delta soup of teacher dual-SFT **v1 @ 5e-4** and **v1 @ 1e-5**, both on Pulse.

**Official dates**

| When | Event |
|---|---|
| 2026-08-19 | Cross-depth lab scored `qwen3_x_v1_5e4_1e5` |
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

**Spectrum.** Same 28-layer Qwen3 scaffold and the **same 19-layer donor as Halo** (v1@5e-4 + v2@1e-4 soup), but the 19 Phase-2B-kept decoder blocks are **fully replaced** (stitch α=1.0). Dropped Qwen3 layers plus embed / final norm / tied lm_head stay official Qwen3. Compare with Halo when an experiment needs a model that answers instead of refusing. Not a dual-skill public announcement. Sealed TEST not run.

**Official dates**

| When | Event |
|---|---|
| 2026-08-19 | Lab scored `qwen3_replace_v1_5e4_v2` |
| 2026-08-19 20:58:42–21:00:00 | Vault write Halo Uncensored (CPU stitch; Pulse, Aegis, Halo, Nimbus, Lyra not overwritten; weight SHA `c5d4a17f…`) |

**Base / ingredients**

Same table as Halo, except Merge is `stitch_cross_depth.py` **α=1.0** on kept layers.

**How to use the pair.** Halo = 50/50 blend on kept blocks. Halo Uncensored = full replace on kept blocks. Lab replace artifacts are kept; do not delete them to save disk.

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
