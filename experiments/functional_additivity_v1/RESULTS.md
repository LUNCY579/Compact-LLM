# FUNCTIONAL-ADDITIVITY-001 — results

Date: 2026-09-23. This is the preregistered teacher-forced function-space
analysis, not SOUP-MINI-001 CAL/CHECK and not a quality, retention, or release
claim.

## Execution receipt

- Bank: `fresh_dual_skill_instrument_v1/T_BANK_V2.json`, 192 rows,
  SHA-256 `fc094136a5b581cb9cc03393b1e5959ed8f69247638c81e37cf7f68fbf132e01`.
- Arms: BASE, SPARSE, OPALLO, ADD-O75, ADD-100; existing FP32 artifacts only.
- Runtime: Python 3.11.15, torch 2.6.0+cu124, transformers 5.14.1,
  NVIDIA GeForce RTX 4070 Laptop GPU, FP32, CUDA, TF32 disabled, one
  teacher-forced forward per row.
- Raw receipt: `TF_RAW.json`, SHA-256
  `ebc36b26871ddf0e1a3b99d66a075c339e5945ceae4177208f3b56fe9307615a`.
- Analysis receipt: `ANALYSIS.json`, SHA-256
  `57999b2d3007990d48d727866780169ad7899c07db174496bc062fcf7ce8341d`.
- Final scorer SHA-256: `faf8d8f8d444db7bae8261989656a918913b914b0ed694d49a55f083a17457b2`;
  final analyzer SHA-256: `ab00cb0490246552d2e772bc306e60eb424a82a92eaf5d4367bcffe61e0e5389`.

## Gates

All preregistered gates passed:

| Gate | Result |
|---|---|
| G-A template | 384/384 recorded prompts reproduced; negative control rejected |
| G-B strict canary | Exact set match for every arm; symmetric difference 0 |
| G-C determinism | SPARSE reload max Δ content = 0; max Δ closing = 0 |
| G-D denominator | `mean(g_S)=49.0537`, bootstrap CI `[44.5292, 53.3910]` |

Strict teacher-forced canary counts were BASE `0`, SPARSE `102`, OPALLO `1`,
ADD-O75 `63`, and ADD-100 `6`, exactly matching the recorded generation sets.

## Primary result: content clipped log-odds

The registered interaction is:

```text
I = ADD-100 - SPARSE - OPALLO + BASE
κ = mean(I) / mean(SPARSE - BASE)
```

| quantity | result |
|---|---:|
| mean interaction | `-3.1371` |
| mean Sparse content effect | `49.0537` |
| κ | `-0.06395` |
| bootstrap 95% CI | `[-0.07395, -0.05453]` |
| registered outcome | **ADDITIVE** |

The entire CI lies inside the preregistered ±0.10 equivalence band. On the
content readout, adding Opallo to Titan-Sparse behaves as approximately additive
within the registered tolerance; it is not classified as sub-additive
interference.

### Family breakdown

| family | κ | 95% CI |
|---|---:|---:|
| ledger_replay | `-0.0357` | `[-0.0477, -0.0235]` |
| prerequisite_plan | `-0.4691` | `[-0.5201, -0.4305]` |
| register_machine | `-0.0523` | `[-0.0624, -0.0420]` |

The prerequisite-plan subset is substantially more attenuated, with a small
Sparse denominator; the aggregate result remains additive because the other two
families dominate the bank-level signal.

## Secondary readouts

- Decision-position clipped log-odds: κ `-0.06395`, CI `[-0.07401,
  -0.05457]` — also **ADDITIVE**.
- Unclipped log-probability: κ `-0.06578`, CI `[-0.11216, -0.02297]` —
  **UNRESOLVED** under the registered rule. This is a readout sensitivity,
  not a primary-gate failure.
- Closing/EOS clipped log-odds: κ `-1.0752`, CI `[-1.2031, -0.9585]` —
  **SUB-ADDITIVE**. The strong negative is confined to the final answer/EOS
  convention channel, not the content skill channel.
- Content `P_S`: `0.8276`, CI `[0.8205, 0.8348]`; additive prediction
  `0.8916`. On unclipped log-probability, `P_S=0.9954`, CI `[0.9863,
  1.0053]`, showing that probability-scale saturation materially changes the
  apparent preservation ratio.
- β-curvature at α=1: content mean `0.5456`, CI `[0.4887, 0.6013]`;
  closing mean `0.8758`, CI `[0.8076, 0.9448]`.
- The registered 11-row SPARSE-correct / ADD-100-period-tolerant-wrong set
  had mean content interaction `-3.4421`, versus `-3.1186` on the other rows.

## Interpretation

This completes the missing function-space experiment. The result supports the
new theory in its narrow, registered form: on Titan-Sparse's transition skill,
full-strength dOpallo addition is functionally additive in the content logits,
despite the much larger apparent strict-generation collapse caused by the
closing convention. The geometry result and this content-level result point in
the same direction.

It does not establish that Opallo's own skill was acquired: the instrument has
no O-side bank. It also does not establish broad quality, retention, release
fitness, or local linearity beyond the tested α=1 / β∈{0.75,1} points.

---

## Erratum 1 (2026-09-23): audit corrections and a post-hoc exact re-score

Full numbers are in `amendment_1/AMENDMENT_1.md`. The re-score's canary matches `TF_RAW` bitwise
on every arm. **The registered primary stands exactly as scored:** κ −0.06395, **ADDITIVE**.
Four statements above need correcting:

1. **"Decision-position clipped log-odds … also ADDITIVE" is wrong.** `analyse.py` applied the
   registered position rule per row, which reproduces the primary. Per position, as PREREG
   line 71 registers it, the result is κ −0.1185, CI [−0.1295, −0.1089]: **SUB-ADDITIVE**.
2. **The primary CI excludes 0, which this file never said.** I < 0 on 167/192 rows. There is a
   small, resolved interference, so "functionally additive" should read "additive within the
   registered ±10% margin".
3. **The ADDITIVE label depends on the clip.** The ±10 clip caps 21.2% of SPARSE's content
   positions but only 7.0% of ADD-100's. With exact log-odds (post-hoc), κ is −0.0986 at clip 12
   (UNRESOLVED) and −0.1038 unclipped (the registered rule would call that SUB-ADDITIVE).
4. **"Closing: SUB-ADDITIVE" labels a readout that had no registered outcome rule.** It is also
   scale-dependent: unclipped I_closing is +0.029, CI [−0.22, +0.30]. The claim that the
   negativity is "confined to the closing channel" does not hold, because content carries a real
   6–10% interference, and −47% in prerequisite_plan.

Reported here for the first time: the closing consistency expectations (192/192 rows; −8.77),
secondary 8 (exact except OPALLO 9 vs 8), and secondary 7. In ADD-100 the period variant beats
the gold continuation on 182/192 rows, against 0 for SPARSE. The run's provenance is in the
PREREG erratum.
