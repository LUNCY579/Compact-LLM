# Rineya Opallo 1.0.0-devfreeze

A 596M-parameter dense Qwen3 core: **Supa 1.7.0-devfreeze with its last 20 transformer
blocks finetuned** on 5,233 supervised tokens of general-reasoning SFT under a
Warmup-Stable-Decay schedule. Embeddings, the final norm and blocks 0-7 are frozen and
byte-identical to Supa.

Opallo is the shipped form of the best arm found by RAB-001. It is an **arithmetic
word-problem improvement that did not cost measurable in-distribution reasoning** — and
that sentence is the whole claim. What it is not is set out under *Limits* below, which is
the part worth reading before using it.

## Results

| Instrument | Rows | Supa (base) | **Opallo** | Measured on |
|---|---:|---:|---:|---|
| Arithmetic panel, permissive | 150 | 47 | **73** | this BF16 core |
| Arithmetic panel, strict | 150 | 41 | **59** | this BF16 core |
| Reference answers retained | 47 | 47 | **41** | this BF16 core |
| Reasoning Forge `test_iid`, exact | 200 | 32 | 30 | float32 checkpoint |
| Reasoning Forge `test_transfer`, exact | 200 | 44 | 36 | float32 checkpoint |

The panel is 150 arithmetic word problems in three classes — `percentage_reasoning`,
`ratio_proportion`, `time_duration` — scored greedy on CPU against a frozen reference
receipt. Full per-row record and verdict in [`PANEL.json`](PANEL.json); provenance and
integrity checks in [`CORE.json`](CORE.json).

**Read every panel count against the instrument's noise floor.** Same-config
different-seed pairs on these 150 rows run **15-24 discordant** (mean 18.8), and one such
pair reached p=0.0352 on seed alone. 73-vs-47 is far outside that band. Smaller gaps on
this instrument are not.

Across four independent seeds this recipe averages **69.25** panel rows; seed 17, published
here, is the best of the four. The single-seed figure is therefore an optimistic point on
a real distribution, not a stable expectation.

## Precision

The checkpoint trained in **float32**; the vault is **BF16**, and casting 314M parameters
moves essentially all of them (worst relative delta 3.9e-3, which is exactly 2⁻⁸). That
cast was measured rather than assumed, against a threshold fixed beforehand in
[`../../OPALLO_PREREG.md`](../../OPALLO_PREREG.md):

| | float32 checkpoint | **BF16 published core** |
|---|---:|---:|
| Panel, permissive | 74 | **73** |
| Reference answers retained | 41 | **41** |
| Rows with byte-identical output | — | **140/150** |
| Rows with identical termination | — | **150/150** |

One row flipped, in the losing direction, with no compensating gain. **73 is the honest
number for this file**; 74 belongs to a float32 artifact that is not what ships here.

## Limits

* **General reasoning is unchanged, not improved.** Forge `test_iid` goes 32 → 30 out of
  200, which is inside that instrument's 7-12 discordant noise floor (p=0.804). RAB-001
  tried schedule, depth, dose, architecture, weight averaging and an 11.5× dose of
  purpose-built reasoning data, and moved this number by zero. On a 596M model that reads
  as capacity, not as a method failure.
* **Out-of-distribution output discipline is slightly worse.** Forge `test_transfer` goes
  44 → 36 and termination 184 → 178. Pooled over four seeds this regression is real
  (p=0.020), though it does not survive Bonferroni correction across the four tests run.
* **Outputs get longer off-distribution.** About 160 of 200 shifted-distribution outputs
  are longer than the base model's (p≈1e-22). This is cosmetic: across 170 wrong outputs
  the correct answer was present in **zero** of them, so the extra tokens are not a
  hidden-answer scoring artifact — but they do mean the model is less terse than Supa
  outside its training distribution.
* **Six reference answers were lost.** Opallo keeps 41 of the 47 the base answered
  correctly and gains 32 it did not.
* **No gate is opened and no sealed test was run.** This is an exploratory research core
  promoted to a named slot. `licenses_efficacy_claim` is false in `CORE.json` and stays
  false.

## Recipe

| | |
|---|---|
| Base | `Rineya-Supa-1.7.0-devfreeze`, weight sha256 `962abc5b…` |
| Trainable | blocks 8-27, 220 tensors, **314,618,880 params (52.8%)** |
| Frozen | embeddings, final norm, blocks 0-7 — byte-identical to base |
| Optimizer | AdamW, lr 1e-5, β₂ **0.95**, wd 0, clip 1.0, micro-batch 1 |
| Schedule | **WSD** — linear warmup 20%, stable, 1-√ decay over the final 20% to 10% |
| Data | `outputs/splits_large` TRAIN, 41 rows / 5,233 supervised tokens |
| Seed | 17 |
| TF32 | forced off everywhere |

Two of those choices are load-bearing and were measured, not inherited. **WSD is worth
+7.5 panel rows** over the inherited cosine schedule on tail-trained arms — 4 of 4 seeds,
pooled p=0.00029 — but it is arm-specific and does nothing measurable on a full-parameter
run (p=0.70). **Depth matters**: last20 beats last8 (p=0.004), and full 28-block training
beats last8 (p=1e-4) and last12 (p=0.004).

Against **full-parameter** finetuning at the same dose, seed and schedule — which scores
76/150 and keeps 37 of 47 — Opallo's arm is **indistinguishable on accuracy**: paired
McNemar over the 30 discordant rows splits 16-14, p=0.86. It reaches that while training
52.8% of the weights and keeping **four more** reference answers. Under the inherited
cosine schedule full-parameter training scores 73/150 (kept 41). The case for Opallo is
cost and retention, not a demonstrated accuracy win over training everything.

## Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "Rineya-Opallo-1.0.0-devfreeze"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype="bfloat16")
```

Weights are tied (`lm_head.weight` is not serialized), 310 tensors, 596,049,920 parameters,
1,192,135,096 bytes. Tokenizer and config assets are copied byte-exact from Supa, so any
Supa-compatible pipeline runs Opallo unchanged.

## Reproduce

```bash
PYG=C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe
$PYG experiments/rab001_exploration/arm_dose.py \
     B28last20 1e-5 5200 17 1.0 wsd:0.2:0.2 0.95 final

PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
$PY experiments/rab001_exploration/materialize_opallo.py \
    --tag B28last20_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95 \
    --codename Opallo --version 1.0.0-devfreeze
$PY experiments/rab001_exploration/validate_published_core.py \
    --core Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze \
    --against B28last20_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95 \
    --min-correct 70 --min-retained 38
```

Materialization refuses to run if the base weights or any tokenizer asset no longer hash
to what the checkpoint recorded, if any trained tensor lands inside the declared frozen
set, or if any frozen tensor comes out non-identical to the base. Validation refuses to
run if the frozen source predictions no longer reproduce the reference set.
