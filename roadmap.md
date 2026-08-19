# Rineya local core roadmap

Developer ledger of **officially recorded local models** and the **base plus every dataset** that produced them.

Weights live under `Rineya_Published_Cores/` (gitignored). Provenance is `CORE.json` plus this file. Phase artifacts under `short-response-model-phase-0-1-2/artifacts/` are **sources**, not the recorded vault.

**Rule.** A model is “officially recorded” only after it has a vault directory, a `core_id`, and a `CORE.json`. Adapters, unfinished runs, and eval-only outputs stay in the working notes below.

Last updated: 2026-08-19 (UTC). Times below are UTC unless marked otherwise.

---

## Recorded cores (vault)

| Recorded (UTC) | Name | Version | Status | Local directory | Weight SHA-256 (prefix) |
|---|---|---|---|---|---|
| 2026-08-19 00:23:36 | **Pulse** | 1.0.0-rc1 | Phase 8 human commercial PASS | `Rineya_Published_Cores/Rineya-Pulse-1.0.0-rc1` | `58ce704c…` |
| 2026-08-19 00:25:16 | **Aegis** | 1.1.0-devfreeze | DEV freeze; sealed TEST not run | `Rineya_Published_Cores/Rineya-Aegis-1.1.0-devfreeze` | `15fab4e7…` |

Pulse is a dense copy of the Phase 8 teacher. Aegis is that same base with teacher dual-SFT v1 `5e-4` LoRA merged in. Sources were not overwritten.

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

**Developer notes.** Website / commercial 1.0 claim stays on Pulse until Aegis has sealed TEST. Pulse showcase long-context exact-match on 2026-08-19 was **0/64** (verbose replies, wrong entities); that is an eval of Pulse, not a new core.

---

## Aegis (`rineya.core.aegis.v1.1.0-devfreeze.19L.454M.dual-sft-v1-lr5e4`)

**What it is.** Same 19-layer / 1024 / ~454M dense file after merging teacher dual-SFT v1 LoRA at **`5e-4`**. Intended dual-skill DEV freeze (short chat + grounded RAG). Not sealed.

**Official dates**

| When | Event |
|---|---|
| 2026-08-16 | Dual-SFT v1 mix seed `20260816`; QLoRA train label `teacher_dual_sft_v1_lr5e4` (1 epoch, 16,410 examples, ~2.1 h GPU) |
| 2026-08-17 | LR sweep **accepted**: `lr5e4`. Sealed chat / RAG TEST / DPO still `not_run`. Rollback = Phase 8 teacher |
| 2026-08-19 00:23:45–00:25:16 | LoRA merged on CPU into Pulse’s base; vault write Aegis (weight SHA `15fab4e7…`) |

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

**Developer notes.** Do not advertise Aegis as Rineya 1.0. Do not retune the frozen `5e-4` winner after seeing sealed TEST.

---

## Local artifacts that are **not** recorded cores

These exist on disk. They are not vault entries.

| Artifact | When / status | Base | Datasets |
|---|---|---|---|
| Teacher dual-SFT **v2 `1e-4`** adapter | Finished GPU SFT (`teacher_dual_sft_v2`, 44,410 rows, ~7.4 h) | Pulse teacher | **v2 mix**: Rineya General Reasoning Corpus v2.0 recommended 40,000 (32,000 new + 8,000 v1 preservation) **+** RAG v3 train 4,410. See `artifacts/teacher_dual_sft_v2/mix_manifest.json` |
| Teacher v2 hot screen **`5e-4` then `3e-4`** | In progress / previously OOM; not a complete adapter | Pulse teacher | Same v2 mix (44,410) |
| Student **greedy** dual-SFT v2 | Finished CPU LoRA 2026-08-18 window (`student_dual_sft_v2_greedy`, 44,410 rows) | 12-layer `phase_9_capability_composition_v22_lr1e4_s1200/track_d_greedy_soup/final_model` | Same v2 mix. Real RAG DEV dropped vs baseline; not a publish candidate |
| Student **frozen-anchor** dual-SFT v2 | CPU train in progress | `…/frozen_anchor` | Same v2 mix |

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
