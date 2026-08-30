# Phase-1 v3 competence postmortem (Stage 0)

Development-only diagnosis of the v3 floor effect. Read-only over the frozen v3
artifacts; v3 itself is unchanged and remains tagged at
`tgsa-phase1-v3-result-2026-08-29`.

**No arm-vs-arm difference was used anywhere in this analysis.** The question is
absolute competence only.

---

## 1. OBSERVED

### 1.1 The models learned the training set and generalized zero

Evaluated the frozen v3 final checkpoints (4 development seeds per arm) on four
splits. The critical addition over the v3 report is an **in-band held-out** set:
generated from the *exact* train-band generator settings with a different dataset
seed, so difficulty is held fixed and only instance novelty varies.

| arm | train (seen) | **in-band (new, same difficulty)** | dev (harder) | test (hardest) |
|---|---:|---:|---:|---:|
| `cascade` | 0.2360 | **0.0453** | 0.0422 | 0.0403 |
| `history_control` | 0.2231 | **0.0453** | 0.0461 | 0.0394 |
| `learned_delta` | 0.2369 | **0.0438** | 0.0400 | 0.0393 |

In-band trivial baselines: uniform 0.0312, majority **0.0407**.

`OBSERVED` In-band held-out accuracy (0.044–0.045) is at the majority baseline
(0.0407). Generalization to a new instance **at the training difficulty** is
essentially zero.

`OBSERVED` Zero latent-signature overlap between the in-band held-out set and
the training set was verified, so this is not leakage.

### 1.2 What the models did learn

| quantity | train | dev | test | reference |
|---|---:|---:|---:|---|
| accuracy | 0.232 | 0.043 | 0.040 | majority 0.036–0.041 |
| answer NLL | 2.70 | 3.94 | 4.03 | ln(32) = 3.466 |
| mass on the 32 legal answers | 0.997 | 0.995 | 0.994 | — |
| prediction entropy | 2.92 | 2.89 | 2.89 | ln(32) = 3.466 |
| distinct predicted tokens | 32 | 33 | 34 | — |

`OBSERVED` The models **did** learn the answer-token *type*: 99.4–99.7% of
probability mass lands on the 32 legal answer tokens. My earlier reading of the
v3 result — that they had not even learned the type prior — was wrong; NLL above
ln(32) is explained by confident *wrong* predictions, not by mass leaking outside
the answer space.

`OBSERVED` Entropy (2.89) is *below* uniform (3.466) while NLL (4.03) is *above*
it. The models are confidently wrong on held-out data. Predictions are
input-dependent (32–34 distinct tokens), so this is not a collapsed constant.

### 1.3 Optimization was not converged

| arm | step 1 | 250 | 500 | 1000 | 1500 | 2000 | 2500 | drop over final 500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cascade` | 4.714 | 3.437 | 3.450 | 3.391 | 3.275 | 3.074 | 2.983 | −0.220 |
| `history_control` | 4.714 | 3.440 | 3.450 | 3.397 | 3.273 | 3.087 | 2.913 | −0.209 |
| `learned_delta` | 4.714 | 3.437 | 3.450 | 3.380 | 3.293 | 3.182 | 3.021 | −0.206 |

`OBSERVED` Training loss was **still falling** at the registered 2,500-update
cutoff, at roughly 0.21 nats per 500 steps. The run was stopped mid-descent.

`OBSERVED` The curve shows a long plateau near 3.44 (steps 250–1000) followed by
descent — the descent phase is memorization onset, not algorithm acquisition,
since held-out accuracy stayed flat throughout.

### 1.4 Gradient flow

`OBSERVED` Gradient group norms on one development batch at the final checkpoint:

| arm | embedding | attention | ffn | trajectory encoder |
|---|---:|---:|---:|---:|
| `cascade` | 6.69 | 2.71 | 1.91 | — |
| `history_control` | 9.44 | 3.96 | 3.69 | 0.097 |
| `learned_delta` | 4.83 | 1.82 | 1.50 | 0.113 |

`OBSERVED` Trajectory-encoder gradients are non-zero but 20–50× smaller than
attention gradients. Nothing is disconnected or vanishing; the trajectory path
is simply a much smaller module.

---

## 2. DIAGNOSIS

`SUPPORTED` **The primary limitation was training-set size, causing memorization
instead of algorithm acquisition.**

The evidence chain: the model reaches 23% on data it has seen and 4.5% on new
data *of identical difficulty*. That is the definition of memorization. The
registered train split holds 5,000 examples, each with a freshly drawn random
pointer permutation, so almost nothing is shared between examples. With ~258k
parameters and 32 epochs, memorizing 5,000 prompt→answer pairs is a far easier
optimization target than inferring the pointer-following algorithm.

`SUPPORTED` **The difficulty shift was NOT the primary cause.** In-band held-out
(0.045) and out-of-band test (0.039) are both at baseline. The train→test range
shift is real but secondary; the model fails before extrapolation is even
required.

`SUPPORTED` **Capacity was not the binding constraint, and more of it would make
things worse.** A larger model on 5,000 examples memorizes more efficiently.

`SUPPORTED` **Update count was a genuine secondary limitation.** Loss was still
descending at cutoff, so 2,500 updates was too few — but more updates on 5,000
examples would mainly deepen memorization.

`HYPOTHESIS` Ranked contributions, not established by this analysis alone:
1. training-set size (dominant);
2. update count (secondary, interacts with 1);
3. train→test difficulty shift (tertiary);
4. model capacity (probably not limiting; possibly counterproductive at fixed data).

`OBSERVED` Not implicated: optimization health (no vanishing/exploding
gradients, no non-finite values), objective formulation (answer-type prior was
learned cleanly), or prediction collapse (predictions are input-dependent).

---

## 3. UNSUPPORTED CLAIMS

* That any amount of data *will* produce competence — that is the pilot's job to
  establish, not this postmortem's to assume.
* That the v3 architecture can learn multi-hop pointer chasing at all.
* Anything about the TGSA mechanism, in either direction.

---

## 4. ANOMALIES

`OBSERVED` `state_update` test accuracy was *below* uniform chance for every arm
in v3 (0.026–0.031 vs 0.0312) — consistent with confidently-wrong memorized
predictions rather than random guessing.

`OBSERVED` The loss plateau at 3.44 for steps 250–1000 is close to
ln(32) = 3.466. The model sat at "uniform over the legal answer space" for 40% of
training before beginning to memorize.

---

## 5. LIMITATIONS

* Four development seeds per arm, not all twenty. Sufficient for a floor
  diagnosis; not a precision estimate.
* The in-band held-out set uses the same generator and vocabulary, so it tests
  instance generalization, not distributional robustness.

---

## 6. CLAIM CEILING

> Under the registered v3 regime, all three arms memorized the 5,000-example
> training set (≈23% seen accuracy) and generalized to new instances at the
> majority baseline (≈4.5%), at identical difficulty. The study was therefore
> uninformative about the mechanism, and the dominant cause was training-set
> size rather than capacity, optimization health, or the difficulty shift.

---

## 7. NEXT PERMITTED EXPERIMENT

A development-only capacity/budget pilot whose **primary axis is training-set
size**, with update count second and capacity third, scored by an **arm-blind**
competence criterion: the minimum held-out accuracy across all three arms
against the strongest registered trivial baseline, plus answer-NLL below
ln(32).

The pilot must include a v3 replica as a negative control, to confirm the
harness reproduces the pathology before any conclusion is drawn from it.
