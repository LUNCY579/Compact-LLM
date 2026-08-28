# Rineya family model registry

Complete inventory of every weight artifact the Rineya family depends on, listed
under its **original working label** rather than its roster/family name, so each
item stays distinguishable before public naming is settled.

Companion documents: [`README.md`](README.md) (policy), [`CATALOG.json`](CATALOG.json)
(machine record), [`systems/README.md`](systems/README.md) (routed systems),
[`../roadmap.md`](../roadmap.md) (index), [`../lineage.md`](../lineage.md) (recipes),
[`../comparison.md`](../comparison.md) (evals).

Last updated: 2026-08-27.

**Storage convention.** `Rineya_Published_Cores/**/*.safetensors` is gitignored
(`.gitignore:57`) and `artifacts/` is gitignored (`.gitignore:97`). Weights stay
local; `CORE.json`, `SYSTEM.json`, configs, adapter configs and manifests are
committed. The private Hugging Face vault carries the binary backup; Supa-A1
and its self-contained evidence were added at Hub commit `b14103c71c240998171694a0d0397bd4e5d64d71`.

---

## 0. Name map — working label → roster name

| Working / original label | Current roster name | Where the original label still appears |
|---|---|---|
| **Taria** 1.7 (scale 0.8125) | Supa Core 1.7 | frozen parity evidence, `artifacts/.../published_dense_parity_v2/taria/`, `_publish_taria_lumina.py` |
| **Lumina** 1.7 (scale 1.25) | Baya Core 1.7 | frozen parity evidence, `artifacts/.../published_dense_parity_v2/lumina/`, `_publish_taria_lumina.py` |
| `Rineya-Nimbus-Abstain-1.0` | **Supa-A1** 1.0.0-devfreeze | sanitized former identity in `Rineya-Supa-A1-1.0.0-devfreeze/SOURCE_CHECKPOINT.json` |
| Nimbus 1.6 | **Nimbus-A1** tracking alias | `CATALOG.json` tracking-label map; no duplicate checkpoint |
| `Rineya-Nimbus-Dense-1.0-nobias` | Supa-A1 ablation control | `Rineya-Supa-A1-1.0.0-devfreeze/evidence/`; not a distinct named core |
| `nimbus_pulse_consensus_cascade_showcase48` | Supa 2.0 System48 | artifact filenames, `configs/final/` |
| `pulse_strong_over_nimbus_gate46` | Baya 2.0 System46 | artifact filenames, `configs/final/` |
| `nimbus_scale0p8125_1p25_unknown_gate` | Nimbus DualScale PEFT Gate45 | artifact filenames |
| `taria_lumina_unknown_gate` | Supa-Baya Dense Gate42 | artifact filenames |
| Pulse, Aegis, Halo, Nimbus, Lyra, Halo Uncensored | unchanged | — |

Old `core_id`s remain valid and resolvable via each entry's `former_identity`
block in `CATALOG.json`.

---

## 1. Tier 1 — dense cores already in the vault (8) · **keep all**

Each is one `model.safetensors` plus `CORE.json`. Showcase figures are strict-exact
over the frozen 64-row demo suite.

| # | Working label | Directory | Params/arch | Weight bytes | SHA-256 prefix | Showcase | Role |
|---|---|---|---|---:|---|---:|---|
| 1 | **Pulse 1.6** | `Rineya-Pulse-1.6.0-devfreeze` | 28L Qwen3 | 2,384,234,968 | `2357574d…` | 31/64 | `late_rineya` α-shape; Halo donor. Strong frozen long-context core; Lyra has the higher recorded chat-Q |
| 2 | **Aegis 1.1** | `Rineya-Aegis-1.1.0-devfreeze` | 19L / 454M | 1,817,909,640 | `15fab4e7…` | 2/64 | RAG specialist. Weak on long-context; keep for grounded work |
| 3 | **Halo 1.2** | `Rineya-Halo-1.2.0-devfreeze` | 28L Qwen3 | 2,384,234,968 | `38fb96fe…` | **29/64** | Cross-depth soup v1@5e-4 + v2@1e-4. Donor for Pulse/Halo-U |
| 4 | **Nimbus 1.6** | `Rineya-Nimbus-1.6.0-devfreeze` | 28L Qwen3 | 1,503,300,328 | `7b66b8d0…` | **30/64** | Soup-lab `global_best` α=0.40. **Base for every Nimbus/Supa/Baya long-context arm** |
| 5 | **Lyra 1.4** | `Rineya-Lyra-1.4.0-devfreeze` | 28L Qwen3 | 2,384,234,968 | `87c57b8e…` | 28/64 | Cross-depth soup v1@5e-4 + v1@1e-5 |
| 6 | **Halo Uncensored 1.5** | `Rineya-Halo-Uncensored-1.5.0-devfreeze` | 28L Qwen3 | 2,384,234,968 | `c5d4a17f…` | 31/64 | α=1.0 replace. Uncensored pair for Halo; **arm in System48** |
| 7 | **Taria 1.7** → Supa | `Rineya-Supa-1.7.0-devfreeze` | 28L Qwen3 | 1,192,135,096 | `962abc5b…` | 40/64 | Nimbus + step200 @ **0.8125**, dense. Best single dense core |
| 8 | **Lumina 1.7** → Baya | `Rineya-Baya-1.7.0-devfreeze` | 28L Qwen3 | 1,192,135,096 | `1737cd1e…` | 40/64 | Nimbus + step200 @ **1.25**, dense. High-scale abstention arm |

Held-out stability192 / all256 for the two 1.7 cores: **Taria 121/192, 161/256** ·
**Lumina 112/192, 152/256**.

> Halo 1.2 and Nimbus 1.6 values above use the standardized 2026-08-26 vault
> rerun: Halo `strict_exact` 0.453125 = 29/64; Nimbus 0.46875 = 30/64.
> Earlier source-selection runs genuinely measured Halo 30 / Nimbus 29. The
> difference is cross-run protocol drift, not a label transposition; §7 records
> both evidence epochs.

---

## 2. Tier 2 — LoRA deltas that are load-bearing · **preserve both copies**

These are not cores, but **no routed system or gated model can be reproduced
without them**. Their working copies remain under gitignored `artifacts/`; the
private Hugging Face family vault also contains the adapter weights inside the
published system bundles. Git keeps configs and manifests, not binaries.

| Working label | Canonical path | Adapter bytes | SHA-256 prefix | Used by |
|---|---|---:|---|---|
| **step200** (`halo_stage_a_step200_adapter`) | `artifacts/rineya_long_context_weight_geometry_v1/selected/halo_stage_a_step200_adapter` | 40,422,168 | `517966d7…` | **Everything long-context**: Taria/Lumina 1.7, all `nimbus_*` and `halo_*` arms, System48/46/45, Abstain-1.0 |
| **step250** (`stage_a/checkpoint-250`) | `artifacts/halo_long_context_sft_v1/stage_a/checkpoint-250` | 40,422,168 | `a508de22…` | `pulse_step250` arm (System48 only) |
| **pulse_negative_strong** (`neg_200_250_1p5_m0p5`) | `artifacts/rineya_published_core_search_v1/negative_vectors/neg_200_250_1p5_m0p5` | 80,792,504 | `79bff1c6…` | `pulse_strong` arm (System48, System46) |

**step200 is the single highest-value non-core artifact in the family.** The local
copy and private Hub copy must both be retained; losing both would make Taria 1.7,
Lumina 1.7 and all four systems unreproducible from source.

---

## 3. Tier 3 — arm configurations (recipes, not separate weights)

Nine named arms. Each is `base core + adapter @ scale`, materialised at inference.
They are *not* separate checkpoints and should not be stored as such.

| Arm label | Base core | Adapter | Scale | Appears in |
|---|---|---|---:|---|
| `nimbus_low` | Nimbus 1.6 | step200 | **0.8125** | System48, System46, Gate45 · **dense form = Taria 1.7** |
| `nimbus_high` | Nimbus 1.6 | step200 | **1.25** | System48, System46, Gate45 · **dense form = Lumina 1.7** |
| `nimbus_mid` | Nimbus 1.6 | step200 | 1.0625 | System48 |
| `pulse_strong` | Pulse 1.6 | pulse_negative_strong | 1.0 | System48, System46 |
| `pulse_base` | Pulse 1.6 | — | — | System48 |
| `pulse_step250` | Pulse 1.6 | step250 | 0.75 | System48 |
| `halo_low` | Halo 1.2 | step200 | 0.25 | System48 |
| `halo_high` | Halo 1.2 | step200 | 0.75 | System48 |
| `halo_uncensored` | Halo Uncensored 1.5 | — | — | System48 |

System48 therefore touches **4 distinct base cores** (Nimbus, Pulse, Halo, Halo-U)
and **3 adapters** across 9 forward passes.

---

## 4. Tier 4 — abstention checkpoints · **Supa-A1 promoted**

Source results remain under `artifacts/rineya_abstention_gate_v1/`; frozen
checkpoint evidence is mirrored into the Supa-A1 core. S4 was added after the
S1 Stability run; its threshold itself was selected on showcase64 only.

| Working label | Form | Passes | Cores | showcase64 | **stability192** | p vs `nimbus_low` |
|---|---|---:|---:|---:|---:|---:|
| `nimbus_low` (reference) | base + adapter | 1 | 1 | 42 | 121 | — |
| `Rineya-Nimbus-Dense-1.0-nobias` | dense merge, bias 0 (Supa-A1 ablation; not a named core) | 1 | 1 | 40 | 121 | 1.0000 |
| **Supa-A1** (formerly `Rineya-Nimbus-Abstain-1.0`) | dense merge + **baked** logit bias | **1** | **1** | 43 | **125** | 0.2188 |
| S4 gate applied offline | processor, bias form | 1 | 1 | 46 | 125 | 0.2188 |
| **S1 gate** (`unknown_variant_logprob`) | processor, normalised form | 1 | 1 | **47** | **130** | **0.0490** |

**Supa-A1** — 1,192,135,096 bytes, SHA-256 `962abc5b…`, byte-identical
to Taria/Supa Core 1.7. The promoted directory adds `CORE.json`, sanitized source
provenance, frozen evidence, `config.json`, and `modeling_abstain.py`. Recipe:
Nimbus 1.6 + step200 @ 0.8125, merged in float32, saved bfloat16, plus
`abstain_logit_bias = 5.156249985098839` on token `51378` (`UNKNOWN`).

**S1 is the strongest preregistered primary single-pass result** (130/192).
Exploratory S2 has the higher raw count at 131/192 but was never primary. S1's
preregistered row-level comparison clears p < 0.05 by one row; multiplicity and post-outcome
pack-level sensitivity are discussed in `FAMILY_PERFORMANCE_AND_RESEARCH.md`.
It has no published checkpoint yet — see §6.

---

## 5. Superseded · archived, not deleted

| Working label | Directory | Notes |
|---|---|---|
| Pulse 1.0.0-rc1 | `_superseded/Rineya-Pulse-1.0.0-rc1` | Phase 8 commercial 19L teacher; Aegis base |
| Nimbus 1.3.0-devfreeze | `_superseded/Rineya-Nimbus-1.3.0-devfreeze` | Aegis-donor 50/50, 24/64 |

---

## 6. Shippability — what is a model and what is not

| Item | Single runnable directory? | Why |
|---|---|---|
| Tier 1 dense cores (8) | **Yes** | Plain Qwen3 checkpoints, any runtime |
| Supa-A1 no-bias evidence | **Yes**, but not separately published | Plain Qwen3 ablation; tensors duplicate Supa Core 1.7 |
| **Supa-A1** (baked S4, 125) | **Yes**, needs `trust_remote_code=True` | Rule is `z_UNKNOWN − z_top1 ≥ τ`; the softmax normaliser **cancels**, so it reduces to a constant added to one logit. Verified: offset exactly +5.15625 on token 51378, max absolute change on the other 151,935 logits = 0.0 |
| **S1 gate (130)** | **Yes — but only as a model class, never as weights.** See below | Rule is `logsumexp(z_U) − logsumexp(z) ≥ τ`; the normaliser does **not** cancel |
| System48 / System46 / Gate45 / DenseGate42 | **No, and never** | Routing compares *decoded text across separate models* |

### 6a. Can the S1 flagship be saved as a runnable single dense model?

**Not as weights. Yes as a single self-contained model directory.** These are
different claims and the distinction is the whole answer.

*Why weights cannot express it.* A weight edit can only change the logit vector
affinely (`z → z + Δ`, or an `lm_head` reparameterisation). S4 collapses because
both of its terms carry the same partition function, which cancels, leaving a
two-logit comparison — exactly what a bias vector encodes. S1 compares a
**normalised** quantity, `logsumexp(z_U) − logsumexp(z)`, against a constant. The
region of hidden-state space where that fires is not a halfspace in logit space,
so no fixed bias, and no `lm_head` rewrite, reproduces it for all inputs.

*Why a single directory can still ship it.* The mechanism already used for
Abstain-1.0 — a `Qwen3ForCausalLM` subclass registered through `auto_map` and
loaded with `trust_remote_code=True` — is fully general. `modeling_abstain.py`
already overrides `forward()` and mutates `outputs.logits`; S1's decision is
computable from precisely that tensor. A `Qwen3S1AbstainForCausalLM` would, on
the prefill call only, compute `log_softmax` over the final-position logits, take
`logsumexp` over the abstention-token set, and if it meets the threshold raise
`z_UNKNOWN` to `max(z) + 1` — forcing the argmax without reordering anything else,
which is exactly what `NormalisedAbstentionGate` does today.

One engineering wrinkle: the gate is defined on the **first generated token only**.
The processor enforces this with `input_ids.shape[-1] == prompt_length`, which a
model class is not given. Inside `forward()` the correct hook is the prefill call
(`past_key_values` empty), whose final-position logits *are* the first generated
token's distribution. This is both implementable and **stricter** than the shipped
S4 class, which applies its bias at every decode step (a documented deviation,
measured harmless at b≈5.16 but not guaranteed at larger biases).

*The cost of that route.* `trust_remote_code=True` is required — no change from
Abstain-1.0, but it means neither gated model is a plain checkpoint. Serving
stacks that refuse remote code, and export targets like GGUF/llama.cpp, would
need the rule reimplemented in that runtime. If the requirement is strictly *"a
checkpoint any stock loader runs with zero custom code"*, then **only the ungated
dense cores qualify**, and both 125 and 130 are out.

### 6b. The recommendation this leads to

Do **not** invest in shipping S1. The bakeable family is not the weaker one — it
is merely the badly-tuned one:

| bias `b` | showcase64 | stability192 | p |
|---:|---:|---:|---:|
| 6.562 | 46 | **129** | 0.0386 |
| 6.156 | 46 | 127 | 0.0703 |
| 5.750 | 46 | 126 | 0.1250 |
| **5.156** *(shipped)* | 46 | **125** | 0.2188 |

Four biases **tie** at the showcase optimum and diverge by 4 points on the now-inspected Stability split. The
pre-registered tie-break (fewest added abstentions) picked the most conservative.
The best achievable pure-bias gate on stability is **133/192 at b = 8.28** —
higher than S1's 130 — and showcase gives no way to find it.

So the pure-bias form, which ships as one directory and needs ~12 lines, has a
**higher ceiling** than the normalised form that needs a processor. The blocker is
threshold selection, not expressiveness. Create a new pack-disjoint calibration
split **and** a second untouched final test, then select the bias on calibration
only. That is worth more than packaging another post-hoc gate.

---

## 7. Gaps and risks

1. **Two valid Halo/Nimbus result epochs must not be conflated.** Historical
   source-selection artifacts measured Halo **30** / Nimbus **29**. Standardized
   2026-08-26 reruns of the actual vault directories measured Halo **29** /
   Nimbus **30**. Live `CORE.json`, `CATALOG.json`, and announcement copy now use
   the standardized values; historical tables should remain explicitly labeled.
2. **The abstention gate needs custom-code serving.** The baked gate is one model
   execution, but `trust_remote_code=True` is required and the exact equivalence
   assumes greedy decoding and a one-token `UNKNOWN` surface.
3. **Sealed TEST has never been run** for any core or system. Every number here is
   showcase (selection) or stability192 (validation).
4. **No routed system's headline is statistically significant.** Against the 42
   baseline on showcase64: System48 +6 (p=0.109), S1 +5 (p=0.125), System46 +4
   (p=0.289). Head-to-head, 48 vs 46 p=0.500, 47 vs 46 p=1.000, 47 vs 48 p=1.000.
5. **Dense materialisation is lossy.** Folding step200 @ 0.8125 into BF16 preserves
   magnitude (`‖realised‖/‖intended‖ = 1.0134`) but corrupts direction by 12–28%
   per tensor. Effect: showcase 42→40, stability 121→121. Independently corroborated
   by Gate45 (PEFT, 45) → DenseGate42 (dense, 42).
