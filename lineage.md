# Family lineage

How each named core was made: dates, geometry, donors, and every train mix. Scores live in [`comparison.md`](comparison.md). Public-copy rules live in [`announcement.md`](announcement.md). Index: [`roadmap.md`](roadmap.md).

Weights live under `Rineya_Published_Cores/` (gitignored). Provenance is each core’s `CORE.json` plus this file. Phase artifacts under `short-response-model-phase-0-1-2/artifacts/` are **sources**, not the recorded vault.

**Rule.** Dense cores use `CORE.json`. Routed systems use `SYSTEM.json` and a
separate system catalog; their scores must never be inherited by one weight file.

Times are UTC unless marked otherwise. Last updated: 2026-08-27.

---

## Timeline

| When (UTC) | Event |
|---|---|
| 2026-08-02 | Phase 2B frontier used by Pulse 1.0 soup (`phase_2b_quality_v4_full_20260802_213049`, stage 09 drop to 19 layers) |
| 2026-08-05 | Pulse 1.0 Phase 8 human commercial PASS (Cheng Li), version `1.0.0-rc1` |
| 2026-08-16 | Dual-SFT v1 mix seed `20260816`; Aegis QLoRA train label `teacher_dual_sft_v1_lr5e4` |
| 2026-08-17 | Aegis dual-SFT v1 `5e-4` LR freeze accepted |
| 2026-08-19 00:23:36–00:23:42 | Vault **Pulse 1.0.0-rc1** (19L) |
| 2026-08-19 00:23:45–00:25:16 | Vault **Aegis** |
| 2026-08-19 18:55:44–18:57:31 | Vault **Halo** |
| 2026-08-19 18:57–20:41 | 28L Phase 1 chat DEV + Phase 9 RAG DEV |
| 2026-08-19 19:54:38–19:54:56 | Vault **Nimbus 1.3.0** (Aegis donor) |
| 2026-08-19 19:54:59–19:56:26 | Vault **Lyra** |
| 2026-08-19 ~20:47 | Human DEV chat pass for Aegis / Halo / Nimbus / Lyra |
| 2026-08-19 20:58:42–21:00:00 | Vault **Halo Uncensored** |
| 2026-08-20 | Halo-donor kept-layer α recipes; best stitch `late_rineya` 31/64 |
| 2026-08-20 | Soup-lab Qwen3 × Halo Uncensored; `global_best` = uniform α=0.40 |
| 2026-08-21 01:14 | Roster replace: archive Pulse 1.0 + Nimbus 1.3; vault **Pulse 1.6** (`late_rineya`) and **Nimbus 1.6** (`global_best`) |
| 2026-08-26 23:47 | Transactional paired publication of **Supa 1.7** and **Baya 1.7** after BF16 dense parity and Demo-256 validation |
| 2026-08-27 19:59 | Transactional publication of **Supa System48**, **Baya System46**, PEFT Gate45, and Dense Gate42 with exact adapters, routing, hashes, and replay evidence |
| 2026-08-27 | Transactional promotion of the former `Rineya-Nimbus-Abstain-1.0` checkpoint as **Supa-A1**; `Nimbus-A1` assigned to the actual Nimbus 1.6 lineage |

Open next items (new calibration/final splits, sealed TEST, DPO) are on [`roadmap.md`](roadmap.md). The Supa-A1 private Hub backup is commit `b14103c71c240998171694a0d0397bd4e5d64d71`.

**Rename/alias note (2026-08-26).** Supa was formerly Taria (`rineya.core.taria.v1.7.0-devfreeze.28L.qwen3-nimbus-lcr-stage-a-step200-scale-0p8125`); Baya was formerly Lumina (`rineya.core.lumina.v1.7.0-devfreeze.28L.qwen3-nimbus-lcr-stage-a-step200-scale-1p25`). This identity-only rename preserved both weight hashes and all frozen evidence. Historical artifact and publisher-script paths keep their original names for auditability.

## Frontier systems (2026-08-27)

| Name | Exact form | Showcase | Stability | All 256 |
|---|---|---:|---:|---:|
| **Supa 2.0** | 9-arm Nimbus/Pulse/Halo/Halo-U ordered cascade | **48** | **128** | **176** |
| **Baya 2.0** | 3-arm Nimbus/Pulse UNKNOWN cascade | **46** | **125** | **171** |
| Nimbus DualScale PEFT Gate45 | Nimbus + step-200 at 0.8125/1.25 | **45** | **124** | **169** |
| Supa-Baya Dense Gate42 | Supa Core 1.7 + Baya Core 1.7 | **42** | **124** | **166** |
| **Supa-A1** | one dense weight + baked UNKNOWN-logit custom class | **43** | **125** | **168** |

The routing is nonlinear and prompt-dependent. System48/46/45 cannot be
represented by a single dense merge without a new distillation experiment and
fresh evaluation. The prior dense Supa/Baya names remain version-qualified as
**Supa Core 1.7** and **Baya Core 1.7**.

---

## Dense-core spectrum (what each checkpoint is)

| Name | Version | Layers | Stitch α | Donor | Role |
|---|---|---:|---:|---|---|
| Pulse | 1.6.0-devfreeze | 28 | shape mean 0.5 | Halo donor (`late_rineya`) | 28L family; long-context stitch |
| Aegis | 1.1.0-devfreeze | 19 | — | archived Pulse 1.0 + v1@5e-4 LoRA | 19L dual-skill (chat + grounded RAG) |
| Halo | 1.2.0-devfreeze | 28 | 0.5 | v1@5e-4 + v2@1e-4 soup | 28L family; Halo donor blend |
| Nimbus | 1.6.0-devfreeze | 28 | 0.40 uniform | Halo Uncensored endpoint | 28L family; soup-lab global_best |
| Lyra | 1.4.0-devfreeze | 28 | 0.5 | v1@5e-4 + v1@1e-5 soup | 28L family; cooler-LR v1 soup blend |
| Halo Uncensored | 1.5.0-devfreeze | 28 | 1.0 | same donor as Halo | Uncensored pair for Halo (kept blocks replaced) |
| Supa Core | 1.7.0-devfreeze | 28 | Nimbus + 0.8125 long-context delta | Stage-A checkpoint 200, trained from Halo then transferred to Nimbus | Dense low-scale dependency arm; 40/64 |
| Baya Core | 1.7.0-devfreeze | 28 | Nimbus + 1.25 long-context delta | Same checkpoint/direction as Supa | Dense high-scale dependency arm; 40/64 |
| Supa-A1 | 1.0.0-devfreeze / tracking A1 | 28 | Supa tensor + baked bias 5.15625 | Same tensor as Supa 1.7; custom model class | One-pass abstention core; 43/64, 125/192 |

Active **Pulse** and **Nimbus** keep the roster names but are new 28L stitches (2026-08-21). The Phase 8 19L teacher and the prior Aegis-donor Nimbus live under `_superseded/`. Aegis ancestry still uses the archived Pulse 1.0 teacher.

| Recorded (UTC) | Name | Status | Local directory | Weight SHA-256 (prefix) |
|---|---|---|---|---|
| 2026-08-21 01:14:07 | **Pulse** | 1.6.0 roster replace; sealed TEST not run | `Rineya_Published_Cores/Rineya-Pulse-1.6.0-devfreeze` | `2357574d…` |
| 2026-08-19 00:25:16 | **Aegis** | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Aegis-1.1.0-devfreeze` | `15fab4e7…` |
| 2026-08-19 18:55:44 | **Halo** | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze` | `38fb96fe…` |
| 2026-08-21 01:14:12 | **Nimbus** | 1.6.0 roster replace; sealed TEST not run | `Rineya_Published_Cores/Rineya-Nimbus-1.6.0-devfreeze` | `7b66b8d0…` |
| 2026-08-19 19:54:59 | **Lyra** | public announcement ready; sealed TEST not run | `Rineya_Published_Cores/Rineya-Lyra-1.4.0-devfreeze` | `87c57b8e…` |
| 2026-08-19 21:00:00 | **Halo Uncensored** | recorded uncensored pair for Halo | `Rineya_Published_Cores/Rineya-Halo-Uncensored-1.5.0-devfreeze` | `c5d4a17f…` |
| 2026-08-26 23:47:23 | **Supa** | Showcase selected; Stability validated; sealed TEST not run | `Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze` | `962abc5b…` |
| 2026-08-26 23:47:26 | **Baya** | Showcase selected; Stability validated; sealed TEST not run | `Rineya_Published_Cores/Rineya-Baya-1.7.0-devfreeze` | `1737cd1e…` |
| 2026-08-27 | **Supa-A1** | S4 added post-confirmation; Showcase threshold; Stability previously inspected; sealed TEST not run | `Rineya_Published_Cores/Rineya-Supa-A1-1.0.0-devfreeze` | `962abc5b…` |

---

## Pulse (`rineya.core.pulse.v1.6.0-devfreeze.28L.qwen3-late-rineya-alpha-shape`)

**Spectrum.** Official **28-layer Qwen3-0.6B** with Phase-2B-kept decoder blocks blended against the Halo donor using the non-uniform **`late_rineya`** α shape (mean 0.5; α rises with depth). Dropped Qwen3 layers plus embed / final norm / tied lm_head stay official Qwen3. Showcase **31/64**. Dual composite DEV **0.7561**.

**Official dates**

| When | Event |
|---|---|
| 2026-08-20 | CrossDepth α-recipe sweep; `late_rineya` best new stitch (31/64) |
| 2026-08-21 01:14:07 | Vault write as **Pulse 1.6.0**; archived prior Pulse 1.0.0-rc1 under `_superseded/` |

**Base / ingredients**

1. Official **`Qwen3-0.6B`** (28L)
2. Halo donor = Pulse-teacher soup of dual-SFT **v1@5e-4 + v2@1e-4** (same donor lineage as Halo)
3. Lab weights: `Rineya_Qwen3_CrossDepth_Soup_Mini_v1/artifacts/alpha_recipes/late_rineya`
4. Vault SHA `2357574db5557c288f57a6b97b2761830ea265baf3c6a1c0270357a7d0eaa322`

**Superseded Pulse 1.0.0-rc1** (19L Phase 8 teacher, SHA `58ce704c…`) remains the dense base for **Aegis** and the commercial PASS artifact. Do not delete `_superseded/Rineya-Pulse-1.0.0-rc1`.

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

## Nimbus (`rineya.core.nimbus.v1.6.0-devfreeze.28L.qwen3-x-halo-uncensored-alpha-0.40`)

**Spectrum.** Official **28-layer Qwen3-0.6B** with **uniform global α=0.40** toward vault **Halo Uncensored** (soup-lab alias `global_best`, candidate `cand_d1b08a8faf299ce4`). Showcase **29/64**. Dual composite DEV **0.7444**. Chat DEV quality **0.8649**.

**Official dates**

| When | Event |
|---|---|
| 2026-08-20 | Soup-lab global stage accepted α=0.40 on chat DEV v4.5 |
| 2026-08-21 | CPU showcase 29/64 + dual chat/RAG fill-in |
| 2026-08-21 01:14:12 | Vault write as **Nimbus 1.6.0**; archived prior Nimbus 1.3.0 under `_superseded/` |

**Base / ingredients**

1. Official **`Qwen3-0.6B`** (α=0 endpoint)
2. Vault **Halo Uncensored 1.5.0** (α=1 endpoint)
3. Lab weights: `Rineya_Layerwise_Soup_Lab_v1.0/runs/qwen3_halo_uncensored/candidate_cache/cand_d1b08a8faf299ce4`
4. Vault SHA `7b66b8d0ef06cf7f71bc9001b0c9a330d5f8a0eec1d11eb173ece70ca66ef49c` (bf16 preserve-anchor serialization)

**Superseded Nimbus 1.3.0** was the Aegis-donor 50/50 stitch (24/64). Kept under `_superseded/Rineya-Nimbus-1.3.0-devfreeze`.

### Prior Nimbus 1.3.0 (archived)

**Spectrum (historical).** Official **28-layer Qwen3-0.6B** with the 19 Phase-2B-kept decoder blocks 50/50-averaged against vault **Aegis**. Dropped Qwen3 layers plus embed / final norm / tied lm_head stay official Qwen3.

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

## Supa Core 1.7 (`rineya.core.supa.v1.7.0-devfreeze.28L.qwen3-nimbus-lcr-stage-a-step200-scale-0p8125`)

**Spectrum.** BF16 dense Nimbus 1.6 plus the early Stage-A long-context LoRA checkpoint 200 at scale **0.8125**. The adapter was trained from **Halo 1.2**, not Nimbus, on the disjoint CC0 long-context corpus; transfer to Nimbus was a later weight-space experiment. Stage A itself is incomplete (latest checkpoint 250, NaN eval losses, no CC0 validation/test). Supa is the low-scale paired-answer arm, not evidence of a finished SFT run.

**Publication gates**

- source Nimbus weight SHA `7b66b8d0…`;
- adapter weight SHA `517966d7…`, rank 16, encoded alpha 26;
- FP32 safe merge followed by one BF16 save;
- exact 310-tensor BF16 inventory and tied-head reload checks;
- dense/PEFT parity measured, not assumed: 251/256 raw predictions matched;
- dense Demo result: 40/64 Showcase, 121/192 Stability, 161/256 overall;
- final dense weight SHA `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`.

Recorded transactionally with `Rineya_Published_Cores/_publish_taria_lumina.py`. Showcase was selection data; Stability was frozen validation. Sealed TEST not run.

---

## Baya Core 1.7 (`rineya.core.baya.v1.7.0-devfreeze.28L.qwen3-nimbus-lcr-stage-a-step200-scale-1p25`)

**Spectrum.** The same Nimbus/Stage-A checkpoint-200 direction as Supa, but at scale **1.25** (encoded alpha 40). Baya is the high-scale abstention and counterfactual arm used by paired UNKNOWN-gate recipes.

**Publication gates**

- identical pinned Nimbus and adapter sources to Supa;
- FP32 safe merge followed by BF16 save;
- exact 310-tensor BF16 inventory and tied-head reload checks;
- dense/PEFT parity measured: 249/256 raw predictions matched;
- dense Demo result: 40/64 Showcase, 112/192 Stability, 152/256 overall;
- final dense weight SHA `1737cd1ee2bdde714c259adf2012ae44c89ad82a06797653343bad7e3cf88bba`.

Supa Core 1.7 and Baya Core 1.7 are the preserved dense pair. Their 42-point
gate is now published as Dense Gate42; the Supa/Baya 2.0 names belong to the
48/46 routed systems. Sealed TEST not run.

---

## Supa-A1 (`rineya.core.supa-a1.v1.0.0-devfreeze.28L.qwen3-nimbus-lcr-step200-scale-0p8125-abstain-b5p15625`)

**Spectrum.** Supa-A1 promotes the local checkpoint formerly called
`Rineya-Nimbus-Abstain-1.0`. Its 1,192,135,096-byte tensor is byte-identical to
Supa Core 1.7 (`962abc5b…`). The behavioral change is an `auto_map` Qwen3 model
class that adds **5.156249985098839** to token **51378** (`UNKNOWN`) before
greedy argmax.

**A-series identity and sibling control**

- public/tracking name: **Supa-A1**;
- semantic vault version: `1.0.0-devfreeze`; tracking label: `A1`;
- actual Nimbus 1.6 receives the **Nimbus-A1** tracking alias;
- the former `Rineya-Nimbus-Dense-1.0-nobias` is a Supa-A1 ablation control,
  not a separate Nimbus core, because its tensor is the same Supa weight.

**Publication gates**

- source Nimbus weight `7b66b8d0…` and step200 adapter `517966d7…` at 0.8125;
- FP32 LoRA merge followed by BF16 save;
- gate spec `5f6cc42e…`, source checkpoint manifest `c613d0fd…`;
- source machine paths sanitized into `SOURCE_CHECKPOINT.json`;
- actual checkpoint results: 43/64 Showcase, 125/192 Stability, 168/256;
- offline S4 reference: 46/64, 125/192 (not inherited by the checkpoint);
- transactional promotion, full hash verification, Python compile check, and
  `trust_remote_code=True` smoke load;
- sealed TEST not run.

The two-logit S4 gate is bias-bakeable because the softmax normalizer cancels.
Greedy decoding and a one-token `UNKNOWN` are required. The class applies its
bias at every decode step although the tuned gate was first-token-only; the
256-row audit found no mid-answer `UNKNOWN`.

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
| Halo-donor kept-layer **α recipes** | Lab 2026-08-20 UTC; **not vaulted** | Same Halo donor (v1@5e-4 + v2@1e-4) on the 19 kept Qwen3 layers; dropped layers untouched | Frozen showcase 64, bf16. Uniform 0.5 = Halo 30/64 (reused). Uniform 1.0 = Halo Uncensored 31/64 (reused). Best new stitch `late_rineya` 31/64 (+1 vs Halo; CF pair 0.4375). Below +2 hit floor. See `Rineya_Qwen3_CrossDepth_Soup_Mini_v1/outputs/alpha_recipes/scoreboard.md`. |

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
