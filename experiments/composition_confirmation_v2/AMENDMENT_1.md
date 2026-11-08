# CONFIRM-002 — Amendment 1 (pre-data: no arm has been scored on any CONFIRM-002 bank or in Stage A)

Written 2026-09-25, together with the builder, scorer, analyses and drivers, and committed before any of them
runs. It fixes what PREREG.md left implicit and records one deviation from its ordering.

## 1. Deviation: the fresh banks were generated BEFORE Stage A

PREREG §4.2 says banks are enlarged "before [they are] generated" if Stage A projects power < 0.80. The two
banks were generated at their base sizes (commit 51345bb5) before Stage A ran. **No model has read either
bank.**

Enlargement, if Stage A calls for it, is therefore done by **extension**, as CONFIRM-001's Amendment 1 did.
The base rows stay byte-identical and in order, and new rows are appended.
- **T:** `T_BANK_V4X` = T_BANK_V4 plus 864 rows, seed 20260927, content-disjoint from everything including
  T_BANK_V4.
- **Boundary:** `HSKILL_SOUP_V3X` = HSKILL_SOUP_V3 plus the next 48 rows per class under the same salt order,
  giving 96 per class. The answerable half cannot grow (109 is the whole remaining pool).

Thresholds that are percentages keep their meaning. Fixed row counts scale with the boundary size:
C2 and G_H become 5% of the boundary rows. C4's −5 rows stays, because the answerable half does not grow.
The extension must be committed before Stage B.

## 2. Gates and the limbs they guard

- T gates G1–G3 guard **C1** and **C5**.
- **G_H** guards **C3**, because its ratio needs H to exist solo.
- **C2** and **C4** have no calibration-arm gate.

A rejected limb leaves the Holm family, and Holm runs over the evaluable members only. If the T gates and
G_H both fail, Stage B stops after the gates.

## 3. Holm p-values

- **C2:** exact two-sided McNemar.
- **C1, C3, C4:** the one-sided bootstrap tail at the margin: P*(stat ≤ margin), over 10,000 paired
  resamples with seed 20260925.

A limb passes only if its bound or gain condition AND its Holm significance both hold.

## 4. C3 bootstrap

Rows are resampled jointly across BASE, SPARSE, H25 and TH25. A draw whose denominator (H25 − BASE) is ≤ 0
counts as failing, i.e. P_H ≤ 0.75.

## 5. Stage A specification

- **Canary decisions:** the per-row credited decision of `boundary_rule.score_row` on all 224 VIEWS rows,
  plus the three counts.
- **Canary failure:** Stage B does not run until the native-probe path is implemented and itself registered.
- **Power:** 1,000 outer simulations × 1,000 inner bootstrap draws. Development per-row outcomes are
  resampled jointly across arms: T from T_BANK_V3X (BASE and SPARSE are CONFIRM-001's recorded predictions),
  boundary and answerable from VIEWS. Stage B's own decision functions are applied.

## 6. Question core, for the descriptive cluster bootstrap

The user text, lower-cased, with every digit run replaced by `#`.

## 7. Disk

The three new arms (about 7.2 GB) are kept from Stage A through Stage B. The drivers refuse to start below
10 GiB free.
