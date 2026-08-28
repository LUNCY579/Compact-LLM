# Numerical robustness of the S1 abstention gate

**The S1 headline of 130/192 and p = 0.0490 is specific to one arithmetic path.
Three other numerical realisations of the same model and the same frozen gate
give 128/192 and p ≈ 0.12–0.14. The significance does not survive a change of
precision and should not be reported as an established result.**

The ungated model scores 121/192 in every realisation. The *model* is
numerically stable; the *gate* is not.

---

## 1. Measurements

Same weights, same `checkpoint-200` adapter at scale 0.8125, same frozen gate
specs, same frozen protocol otherwise (greedy, seed 17, 24 new tokens, 4096
context, thinking disabled). Only the arithmetic differs.

### stability192

| realisation | ungated | +S1 | +S4 | pred drift | S1 gate flips | S1 row McNemar p | S1 pack sign p |
|---|---:|---:|---:|---:|---:|---:|---:|
| base+adapter, **BF16** (frozen protocol) | 121 | **130** | 125 | — | — | **0.0490** | **0.0129** |
| base+adapter, **FP32** | 121 | 128 | 125 | 2 | 2 | 0.1435 | 0.0574 |
| dense merge, **BF16** store | 121 | 128 | 125 | 3 | 4 | 0.1185 | 0.0386 |
| dense merge, **FP16** store | 121 | 128 | 125 | 2 | 2 | 0.1435 | 0.0574 |

### showcase64 (tuning split)

| realisation | ungated | +S1 | +S4 |
|---|---:|---:|---:|
| base+adapter, BF16 (frozen protocol) | 42 | 47 | 46 |
| base+adapter, FP32 | 41 | 44 | 45 |
| dense merge, BF16 store | 40 | 43 | 43 |
| dense merge, FP16 store | 41 | 44 | 45 |

"drift" and "flips" are counted against the BF16 base+adapter reference.

## 2. Why the gate is fragile where the model is not

Greedy decoding emits `argmax(z)`. An argmax decision has **margin**: a small
perturbation of the logits rarely changes which token is largest, which is why
the ungated score is 121 in all four realisations.

A threshold decision has **no margin**. S1 fires when the abstention mass
crosses −5.824; every row sitting near that value is a coin flip under any
perturbation. BF16 carries 8 mantissa bits, so folding a scale-0.8125 LoRA into
BF16 storage leaves 17.25 % of the delta as rounding noise (worst tensor 28 %);
even the unmerged path differs from FP32 by enough to move 2 rows.

## 3. Storage precision is not the limiting factor

| storage dtype | mantissa bits | delta lost to rounding | worst tensor | file size (0.6B) | stability +S1 |
|---|---:|---:|---:|---:|---:|
| bfloat16 | 8 | 17.25 % | 28.02 % | 1.19 GB | 128 |
| float16 | 11 | **2.19 %** | 3.60 % | 1.19 GB | 128 |
| float32 | 24 | 0 % | 0 % | 2.38 GB | — |

FP16 stores the merged weights 8× more faithfully than BF16 **at identical file
size**, with no overflow risk (max \|W\| = 192 against the 65504 limit; 0.19 % of
weights fall below the FP16 normal range and become subnormal, not zero). It
halves the mechanical drift (3 → 2 predictions, 4 → 2 gate flips).

It does not change the score. The residual difference is arithmetic *order* —
the reference computes `W·x + s·B(A·x)` as two matmuls, a merged core computes
`(W + sBA)·x` as one — and no storage format fixes that. FP16 is still the right
default for a dense core, but for fidelity, not for accuracy.

## 4. What is and is not supported

**Supported.** The gate is worth roughly **+7 on stability** (121 → 128) across
every realisation tested. That is the same size as System48's +7, obtained with
one model execution instead of nine.

**Not supported.** That the gain is statistically significant. Only the BF16
base+adapter path clears p < 0.05, by one row, and every more-accurate
realisation lands at p ≈ 0.12–0.14. Under the 48-pack sign test the picture is
the same shape: 0.0129 on the BF16 path, 0.0386–0.0574 elsewhere.

**Not supported.** That any configuration beats any other. S1 vs System46 pack
p = 0.109; S1 vs System48 p = 0.549; S1 vs Supa-A1 p = 0.109.

## 5. Method

- FP32 probe: the canonical runner with `--torch-dtype float32`, unmerged
  base+adapter, `--adapter-scale 0.8125`.
- FP16 dense: merged in float32, cast to float16, run with
  `--torch-dtype float16`.
- Gate specs are the frozen showcase-selected ones; **nothing was re-tuned.**
  Applying an already-frozen gate to differently-rounded weights introduces no
  selection freedom, so these runs add no selection bias — but they are
  additional reads of stability192, which has no sealed split behind it.
- Pack sign test: 48 scenario packs, exact two-sided sign test over packs whose
  correct-row count changed. Rows within a scenario share one context and are
  not independent, so this is the more defensible test of the two.

## 6. Files

```
precision_probe/fp32_adapter_{showcase,stability}.jsonl
dense_capture/dense_{showcase,stability}_confidence.jsonl     BF16 dense
dense_capture/fp16_{showcase,stability}_confidence.jsonl      FP16 dense
checkpoint/Rineya-Nimbus-Dense-FP16/                          FP16 dense core
```
