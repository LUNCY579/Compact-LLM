# RESULT — gate-off ablation of the Phase 1 adapters

Ablation id: `gate_off_nora_v1`. Run 2026-09-06.
Cores: **Supa-A1.1** (float16) and **Supa-A1** (bfloat16) — both, complete.

**Single manipulated variable:** the constant abstention logit bias the A-series
cores add to token **51378 (`UNKNOWN`)** inside `Qwen3AbstainForCausalLM.forward`.
`gate on` = the canonical **+5.156249985098839**; `gate off` = **0.0**, applied in
memory only, at both training and evaluation (see §2 — the training side was
already gate-free, provably).

Everything else is byte-identical to the original experiment: same frozen
backbones, same saved adapter weights, same curriculum, splits, seeds, optimizer,
schedule, step count, accumulation, decoding protocol, target suite, preservation
set and scorer — reused by importing the lab's frozen modules, not reimplementing
them. No adapter was tuned. No threshold was tuned. No new data was introduced.

---

## 1. Primary question and answer

> Does NoRA still underperform when the pretrained fixed abstention gate is
> removed, or was the previous failure primarily caused by NoRA–gate
> miscalibration?

**On the long-context target metric, the previous failure was primarily
NoRA–gate miscalibration, and this replicates across both cores.** With the gate
removed, NoRA moves from well below the frozen baseline to above it, and clears
both preregistered target thresholds on both checkpoints.

| core | NoRA gate ON | frozen gate ON | NoRA gate OFF | frozen gate OFF | gate-OFF gain |
|---|---:|---:|---:|---:|---:|
| A1.1 | 133/256 | 170/256 | **171/256** | 162/256 | **+3.52 pts** (RER +0.096) |
| A1 | 128/256 | 168/256 | **175/256** | 161/256 | **+5.47 pts** (RER +0.147) |

**On preservation it was not.** The gate changes **0 of 200** preservation
predictions on either core. NoRA's ~11-point preservation regression is real
adapter damage, and it still fails the preregistered gate.

So the answer is *split by metric*, and neither half should be quoted alone.
Under the preregistered gate NoRA **fails in every setting on both cores**, so
**Phase 2 remains unlicensed**.

---

## 2. Retraining was not required, and that is a measured result

`Qwen3AbstainForCausalLM.forward` calls `super().forward(...)` — where the
causal-LM loss is computed — and only afterwards mutates the returned `logits`
row for token 51378. The lab consumes `out.loss` in both the training step
(`training.py:371`) and the validation pass (`training.py:112`), never the
mutated logits. Independently, `training.py:351` already wraps the whole
optimisation in `gate_disabled(model)`.

Measured, not merely argued (`prove_gate_loss_independence.py`, full 160-row dev
set, A1.1 NoRA adapter):

| | value |
|---|---|
| weighted dev loss, gate ON | 0.04797873464580445 |
| weighted dev loss, gate OFF | 0.04797873464580445 |
| bit-identical | **true** |
| per-example bit-identical | **160 / 160** |
| max absolute difference | **0.0** |

That figure also reproduces arm C's recorded final validation loss of
**0.047979**, confirming the dev set was reconstructed exactly.

**Therefore the saved adapters already ARE the gate-off-trained adapters, and the
recorded validation-loss trajectories already ARE the gate-off trajectories.** A
retrain from the original NoRA initialization would reproduce the same weights;
it was correctly skipped, not omitted for convenience.

---

## 3. Harness validity

The gate-OFF numbers are only trustworthy because the same harness reproduces
the gate-ON history exactly. `verify_ablation_runs.py` reports **PASS** on all
twelve runs:

| run | core | arm | gate | target | reproduces stored rows | adapter attached |
|---|---|---|---|---:|---|---|
| `a11_A_frozen_gate_on_control` | a11 | A | on | 170/256 | **yes** | n/a |
| `a11_B_lora_gate_on_control` | a11 | B | on | 110/256 | **yes** | yes |
| `a11_C_nora_gate_on_control` | a11 | C | on | 133/256 | **yes** | yes |
| `a11_A_frozen_gate_off` | a11 | A | off | 162/256 | n/a | n/a |
| `a11_B_lora_gate_off` | a11 | B | off | 165/256 | n/a | yes |
| `a11_C_nora_gate_off` | a11 | C | off | 171/256 | n/a | yes |
| `a1_A_frozen_gate_on_control` | a1 | A | on | 168/256 | **yes** | n/a |
| `a1_B_lora_gate_on_control` | a1 | B | on | 108/256 | **yes** | yes |
| `a1_C_nora_gate_on_control` | a1 | C | on | 128/256 | **yes** | yes |
| `a1_A_frozen_gate_off` | a1 | A | off | 161/256 | n/a | n/a |
| `a1_B_lora_gate_off` | a1 | B | off | 164/256 | n/a | yes |
| `a1_C_nora_gate_off` | a1 | C | off | 175/256 | n/a | yes |

Reproduction compares every field except the per-run `inference_seconds` stamp,
joined on `(evaluation, example_id)`: **0 mismatches across all 456 rows** for
every control. Attachment is a positive control — an adapter arm must not score
identically to the frozen arm under the same gate.

**Independent cross-check of the gate-OFF configuration itself**, against results
the lab already had from a different script (`phase0_run_eval.py`, condition
`A_full`):

| | this ablation | pre-existing ungated Phase 0 |
|---|---:|---:|
| a11 frozen gate-off | 162/256 (41 showcase / 121 stability) | `phase0_eval_a11_ungated` 162/256 (41 / 121) |
| a1 frozen gate-off | 161/256 | `phase0_eval_a1_ungated` 161/256 |
| identical `raw_prediction` (a11) | **256 / 256** | — |

Two independent code paths produce byte-identical predictions on all 256 rows.
This validates the gate-off configuration, not merely the gate-ON replay.

Canonical checkpoints and frozen base parameters hashed unchanged on every run.
One run (`a1_A_frozen_gate_off`) aborted first attempt on a transient
`CUBLAS_STATUS_EXECUTION_FAILED`, left an empty directory, and was re-run
cleanly; the harness refuses to write into a non-empty label directory.

---

## 4. Requested measurements

All on the 256-row demo suite at FULL context, 192 answerable / 64 unanswerable,
plus the 200-item preservation tripwire.

### Total, answerable, unanswerable, answerable `UNKNOWN`, preservation

| core | arm | gate | target /256 | answerable /192 | unanswerable /64 | answerable `UNKNOWN` | preservation /200 |
|---|---|---|---:|---:|---:|---:|---:|
| A1.1 | A frozen | on | 170 | 148 | 22 | 2 | 74 |
| A1.1 | A frozen | **off** | **162** | **149** | **13** | **0** | **74** |
| A1.1 | B LoRA | on | 110 | 59 | 51 | 116 | 69 |
| A1.1 | B LoRA | **off** | **165** | **122** | **43** | **35** | **69** |
| A1.1 | C NoRA | on | 133 | 71 | 62 | 116 | 52 |
| A1.1 | C NoRA | **off** | **171** | **115** | **56** | **66** | **52** |
| A1 | A frozen | on | 168 | 147 | 21 | 2 | 74 |
| A1 | A frozen | **off** | **161** | **148** | **13** | **0** | **74** |
| A1 | B LoRA | on | 108 | 57 | 51 | 117 | 68 |
| A1 | B LoRA | **off** | **164** | **121** | **43** | **37** | **68** |
| A1 | C NoRA | on | 128 | 65 | 63 | 124 | 53 |
| A1 | C NoRA | **off** | **175** | **120** | **55** | **65** | **53** |

Preservation is identical under both gate settings on every arm of both cores.
On A1.1, removing the gate changes **56 of 256** NoRA target predictions (44
fixed, 6 broken, net **+38**) and **0 of 200** preservation predictions.

### Preregistered gate, applied within each gate setting

Baseline is always the frozen arm of the *same core* under the *same* gate
setting. Threshold: gain ≥ 2 absolute points **or** RER ≥ 5%, **and**
preservation regression ≤ 1 point.

| core | arm | gate | target | gain | RER | pres. regression | gain gate | pres. gate | overall |
|---|---|---|---:|---:|---:|---:|---|---|---|
| A1.1 | B LoRA | on | 110 | −23.44 | −0.698 | +2.50 | fail | fail | **fail** |
| A1.1 | B LoRA | off | 165 | +1.17 | +0.032 | +2.50 | fail | fail | **fail** |
| A1.1 | C NoRA | on | 133 | −14.45 | −0.430 | +11.00 | fail | fail | **fail** |
| A1.1 | C NoRA | off | 171 | **+3.52** | **+0.096** | +11.00 | **PASS** | fail | **fail** |
| A1 | B LoRA | on | 108 | −23.44 | −0.682 | +3.00 | fail | fail | **fail** |
| A1 | B LoRA | off | 164 | +1.17 | +0.032 | +3.00 | fail | fail | **fail** |
| A1 | C NoRA | on | 128 | −15.62 | −0.455 | +10.50 | fail | fail | **fail** |
| A1 | C NoRA | off | 175 | **+5.47** | **+0.147** | +10.50 | **PASS** | fail | **fail** |

LoRA never clears the gain gate; NoRA clears it on both cores once the gate is
off. Neither ever clears preservation.

### Per-task-type breakdown — what the gate was hiding (A1.1)

Under the gate, NoRA looked as though it had destroyed the model's strongest
capability: `causal_disambiguation` fell from a perfect 32/32 to 9/32. With the
gate off it returns to 30/32. The destruction was abstention, not capability
loss.

| task_type | frozen OFF | NoRA ON | NoRA OFF | LoRA OFF | NoRA OFF − frozen |
|---|---:|---:|---:|---:|---:|
| alias_resolution | 4/32 | 7/32 | 8/32 | 18/32 | +4 |
| causal_disambiguation | 32/32 | **9/32** | **30/32** | 22/32 | −2 |
| count_with_exclusions | 16/32 | 15/32 | 15/32 | 16/32 | −1 |
| latest_verified_record | 24/32 | 31/32 | **31/32** | 19/32 | **+7** |
| multi_condition_identity | 18/32 | 10/32 | 20/32 | 21/32 | +2 |
| object_state_tracking | 24/32 | 19/32 | 24/32 | 28/32 | 0 |
| rule_exception | 15/32 | 16/32 | 16/32 | 16/32 | +1 |
| temporal_correction | 29/32 | 26/32 | 27/32 | 25/32 | −2 |

### The preservation damage is real, specific, and not about abstention

The gate changes 0 of 200 preservation predictions, so this is entirely the
adapter. It is also entirely one class, on both cores:

| core | arm (gate off) | counting_sets | conditional_rules | `UNKNOWN` predictions |
|---|---|---:|---:|---:|
| A1.1 | A frozen | 37/100 | 37/100 | 0 |
| A1.1 | B LoRA | 34/100 | 35/100 | 0 |
| A1.1 | C NoRA | **15/100** | **37/100** | 0 |
| A1 | A frozen | 37/100 | 37/100 | 0 |
| A1 | B LoRA | 33/100 | 35/100 | 0 |
| A1 | C NoRA | **16/100** | **37/100** | 0 |

NoRA leaves `conditional_rules` untouched (37 vs 37 on both cores) and loses ~21
points on `counting_sets` alone. Of the 25 A1.1 counting rows the frozen model
answered correctly and NoRA lost, the failure mode is not abstention or
malformed output but **wrong arithmetic in a well-formed short answer**:

| reference | frozen | NoRA |
|---|---|---|
| 16 | `16` | `12` |
| 15 | `15 route cards` | `8 routes` |
| 30 | `30 sensor tags` | `10.` |
| 32 | `32` | `64` |
| 15 | `15` | `12` |

So the adapter degrades counting/arithmetic specifically while preserving
conditional-rule reasoning. This is genuine capability damage and is the reason
NoRA fails the gate even with the abstention bias removed.

### Validation-loss trajectory

Teacher-forced, at attempts 75/150/225/300, and gate-independent by §2:

| core | arm | 75 | 150 | 225 | 300 |
|---|---|---:|---:|---:|---:|
| A1.1 | B LoRA | 0.4269 | 0.3024 | 0.2037 | 0.2199 |
| A1.1 | C NoRA | 0.1150 | 0.1321 | 0.0443 | 0.0480 |
| A1 | B LoRA | 0.4352 | 0.3093 | 0.2046 | 0.2238 |

NoRA's loss is ~4× lower than LoRA's throughout while its answerable accuracy is
worse. A quarter of the dev set has the single token `UNKNOWN` as its entire
target, so this loss rewards confident abstention; it is not a quality measure.

### First-token `UNKNOWN` logit / margin diagnostics

Raw logits at the first generated position, gate off. Margin =
`z_UNKNOWN − max z_other`; the gate's own documented rule is
`abstain iff z_UNKNOWN + b ≥ max z_other` under greedy decoding, so its effect is
recovered analytically and needs no second generation pass.

Answerable rows (n = 192):

| core | arm | mean `z_UNKNOWN` | mean margin | self-abstain (b=0) | abstain (b=5.156) | flipped by the bias alone | margin shift vs frozen |
|---|---|---:|---:|---:|---:|---:|---:|
| A1.1 | A frozen | 9.46 | **−12.88** | 0 | 2 | 2 | — |
| A1.1 | B LoRA | 17.81 | −4.04 | 35 | 116 | **81** | **+8.84** |
| A1.1 | C NoRA | 22.11 | −4.26 | 66 | 116 | **50** | **+8.62** |
| A1 | A frozen | 9.45 | **−12.92** | 0 | 2 | 2 | — |
| A1 | B LoRA | 17.92 | −3.90 | 39 | 119 | **80** | **+9.02** |
| A1 | C NoRA | 22.56 | −3.69 | 66 | 124 | **58** | **+9.23** |

Unanswerable rows, A1.1: frozen mean margin −6.72 (13 self-abstain, 22 gated),
LoRA +0.27 (43 / 51), NoRA +4.31 (56 / 62).

**Mechanism.** The frozen model holds a ~13-nat margin against `UNKNOWN` on
answerable rows, so a +5.156 bias is harmless (2/192 abstentions). Training moves
that margin ~8.6–9.2 nats toward `UNKNOWN` — 25% of the curriculum has `UNKNOWN`
as its entire target, and `UNKNOWN` is exactly token 51378. The remaining ~4-nat
margin is smaller than the bias, so the *unchanged* constant becomes destructive.
The gate was calibrated for the published model and is never re-calibrated after
adaptation. Every quantity in this mechanism replicates across both cores.

Limitation: the offset is applied at every decode step, not only the first, so
this diagnostic explains first-token abstention, not every downstream token.

---

## 5. What this does and does not license

**Diagnostic conclusions (supported):**
1. NoRA's catastrophic target-metric failure under the canonical gate was
   primarily NoRA–gate miscalibration. Removing the gate recovers 38 net rows on
   A1.1 and takes NoRA above the frozen baseline on both cores.
2. The miscalibration is quantified: adapters consume ~9 nats of a ~13-nat safety
   margin, leaving less than the +5.156 bias. Replicated on both cores.
3. NoRA's preservation regression is *not* gate-related at all (0/200 predictions
   changed) and is confined to `counting_sets` arithmetic.
4. No retraining was required; the gate is provably absent from the objective.

**Not licensed by this ablation:**
- **This is not an efficacy claim for NoRA.** Its gate-off gain decomposes as
  −34 answerable / +43 unanswerable on A1.1 and −28 / +42 on A1: the entire net
  gain is better abstention on genuinely unanswerable items, not better
  long-context reasoning. NoRA still self-abstains on 66/192 answerable rows with
  no gate at all, on both cores.
- NoRA still **fails the preregistered Phase 1 gate** in every setting on both
  cores, so **Phase 2 remains unlicensed** and the composition check remains
  forbidden.
- Two cores, one seed, one recipe, one 256-row suite, no multiplicity correction.
  No statistical significance is claimed.
- Removing a published model's abstention gate changes its deployed contract;
  this is a diagnostic configuration, not a recommendation.

**Recommended next single variable (not run):** re-calibrate the bias for the
adapted model, or hold the gate active during training so the adapter learns
against the decoder it will be judged by. Either is a *new* experiment and needs
its own preregistration.

---

## 6. Evidence

Everything under `ablations/gate_off_nora_v1/results/`:

- `gate_loss_independence.json` — the bit-identical loss proof
- `a11_*_gate_off/`, `a1_*_gate_off/` — per_example.jsonl, first_token_diagnostics.jsonl, manifest.json
- `a11_*_gate_on_control/`, `a1_*_gate_on_control/` — matched controls
- `verification.json` — reproduction and attachment verification, 12 runs, status PASS

Harness: `run_gate_off_eval.py`, `prove_gate_loss_independence.py`,
`verify_ablation_runs.py`. These live outside `src/`, `scripts/` and `configs/`,
which are SHA-256 frozen into every prior run receipt; those 23 files were
verified unchanged, so all previous evidence remains valid and reproducible.

Known harness gaps, recorded rather than silently fixed: the row-level
reproduction check and the adapter-identity checks live in
`verify_ablation_runs.py` rather than inside `run_gate_off_eval.py`, so a
gate-OFF run does not itself refuse to proceed without a passing control — the
operator must run the verifier, as was done here.
