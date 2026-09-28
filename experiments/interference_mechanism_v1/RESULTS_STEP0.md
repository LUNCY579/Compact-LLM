# INTERFERENCE-MECH-001, step 0: interference falls on knife-edge rows (exploratory)

**Zero forwards.** The data is FA-001's exact teacher-forced record on the **exposed** V2 bank
(192 rows), plus the recorded generations. Script: `margin_fragility.py`. Output:
`MARGIN_FRAGILITY.json`. Nothing here was registered.

## Why

CONFIRM-001 confirmed real interference: ADD-100 loses 56 of SPARSE's rows, gains 22, and every
loss is a content error. 39 of those 56 losses are also lost by AVG. That pattern points to
fragile rows rather than targeted damage.

## Measure

A row's margin under an arm is the weakest gold position's log-odds under teacher forcing, made
tolerant of the one trailing period that the primary rule accepts:
- on JSON rows, `}` and `}.` are pooled at the last position;
- on integer rows, EOS and `.` are pooled at the end position.

**Correction during this analysis:** the first version used gold-only log-odds. That charged
ADD-100 for the period the primary rule accepts. Its margin sign agreed with generation on 121/192
rows; the tolerant margin agrees on 168/192. SPARSE agrees on 179/192.

## Results

| question | result |
|---|---|
| Does a **low** SPARSE margin predict that ADD-100 **loses** a SPARSE-correct row? | **AUC 0.77**, permutation p = 0.001. Median margin 0.05 on lost rows vs 0.94 on kept rows. |
| Does a low SPARSE margin predict that **AVG** loses a row? | AUC 0.72, p < 2e-4. Median 0.29 vs 1.28. |
| Among SPARSE-wrong rows, does a margin **near zero** predict that ADD-100 **gains** the row? | **AUC 0.87**, p = 2e-4. Median −0.82 on gained rows vs −2.33 on the other wrong rows. |

**Interference is concentrated at the decision boundary, in both directions.** The rows dOpallo
breaks are the ones SPARSE was barely getting right. The rows it fixes are the ones SPARSE was
barely getting wrong. Losses outnumber gains because SPARSE has more knife-edge correct rows than
knife-edge wrong ones, and because the mean shift is negative (−0.28 logits on the weakest token).

**Unstructured noise is ruled out.** A zero-mean-plus-shift Gaussian perturbation with the observed
SD (0.80 logits) predicts **30 lost / 2.8 gained**. The observed counts are **11 / 6**. The shift
is strongly anti-correlated with SPARSE's margin (r = −0.72): confident rows lose more log-odds in
absolute terms, and low-margin rows are disturbed less than iid noise would disturb them. So
dOpallo's effect on content is **structured**, not iid noise. Part of that correlation may be
regression toward the mean of a noisy per-row minimum. This analysis cannot separate the two.

## What it suggests (hypotheses, not findings)

1. **Interference will be predictable from the host skill's margins.** Before adding a donor,
   teacher-force the host skill and count its knife-edge rows. That count bounds what any
   perturbation can flip. This is the same "predict from endpoints" idea as
   `convention_localization_v1/RESULTS_STEP0B.md`.
2. **A margin-raising remedy.** If the cost falls on low-margin rows, raising the skill donor's
   coefficient slightly above 1 should protect them, provided the skill's dose curve is still rising
   at 1.0. On CONFIRM-001's bank the curve is still rising: 0.40 → 279, 0.634 → 373, 1.0 → 447.
   Registered for a GPU step in `STEP1_PREREG.md`.

## Limits

One exposed bank, 11 lost and 6 gained rows. Teacher-forced margins are an imperfect proxy for
greedy correctness (sign agreement 168–179/192). There are no teacher-forced margins on CONFIRM-001's
bank, so the V3X losses cannot be tested this way without new forward passes.
