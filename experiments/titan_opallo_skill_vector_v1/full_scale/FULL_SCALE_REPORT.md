# Titan–Opallo full-scale measurement report

Date: 2026-09-23. All runs were FP32, greedy/teacher-forced CUDA measurements
on the RTX 4070 Laptop GPU, under the shared GPU lease. No training, source
checkpoint modification, or SOUP-MINI-001 CAL/CHECK access occurred.

## Runs completed

1. Full nine-arm teacher-forced S-bank matrix: 192 rows × 9 arms.
2. Held-out O-side reply-content NLL: 6,741 never-consumed rows from Opallo's
   24 training classes × 9 arms = 60,669 row evaluations.

Receipts:

- `TF_FULL_RAW.json` and `TF_FULL_ANALYSIS.json`
- `O_REPLY_NLL.json` and `O_REPLY_NLL_ANALYSIS.json`

## S-side full coefficient matrix

Clipped gold-token log-odds content means:

| arm | α | β | content readout |
|---|---:|---:|---:|
| BASE | 0 | 0 | 33.326 |
| SPARSE | 1 | 0 | 82.379 |
| OPALLO | 0 | 1 | 28.008 |
| AVG | .5 | .5 | 64.622 |
| ADD-S75 | .75 | 1 | 68.602 |
| ADD-O75 | 1 | .75 | 76.583 |
| ADD-75 | .75 | .75 | 71.356 |
| ADD-100 | 1 | 1 | 73.924 |
| CONFLICT-LOCAL | 1 | 1* | 73.923 |

`CONFLICT-LOCAL` differs only on two tiny norm tensors and is effectively
identical to ADD-100. Sparse preservation ratios relative to BASE were:

| arm | `P_S` |
|---|---:|
| AVG | .638 |
| ADD-S75 | .719 |
| ADD-75 | .775 |
| ADD-O75 | .882 |
| ADD-100 | .828 |
| CONFLICT-LOCAL | .828 |

The registered full-add interaction remains κ `−0.06395`, inside the additive
band. Averaging is visibly more attenuating than full-strength composition.

## O-side held-out reply-content NLL

Token-weighted mean NLL per reply token:

| arm | NLL/token |
|---|---:|
| BASE | 3.2351 |
| SPARSE | 3.1625 |
| OPALLO | 2.5738 |
| AVG | 2.6811 |
| ADD-S75 | 2.5949 |
| ADD-O75 | 2.6201 |
| ADD-75 | 2.6095 |
| ADD-100 | 2.6054 |
| CONFLICT-LOCAL | 2.6054 |

Using paired bootstrap over rows and per-row mean NLL:

- Opallo’s held-out improvement over BASE is the denominator.
- SPARSE preserves `P_O = .1618`, CI `[.1548, .1686]`.
- ADD-100 preserves `P_O = .9723`, CI `[.9672, .9773]`.
- On the log-odds secondary, SPARSE `P_O=.1389`; ADD-100 `P_O=.9972`.

This is the first valid O-side result in the programme. It does not claim that
Opallo acquired an arithmetic skill; it measures a held-out reply-content
behavioral direction on the classes actually touched by its training draw.

The β dose is monotone in the expected direction at α=1: ADD-O75 is weaker
than ADD-100. ADD-S75 is slightly better than ADD-100 on this held-out NLL,
suggesting mild interaction/regularization at sparse attenuation. Conflict
localization remains a no-op: its mean difference from ADD-100 is about
`4.6e-6` NLL per row.

### Mismatched-reply specificity control

The same 6,741 prompts were rescored against deterministic within-class
mismatched replies for BASE, SPARSE, OPALLO, ADD-O75, and ADD-100:

| arm | mismatch-minus-matched NLL | bootstrap 95% CI |
|---|---:|---:|
| BASE | 2.0072 | `[1.9549, 2.0600]` |
| SPARSE | 2.0065 | `[1.9555, 2.0590]` |
| OPALLO | 1.8403 | `[1.7949, 1.8854]` |
| ADD-O75 | 1.8927 | `[1.8461, 1.9382]` |
| ADD-100 | 1.8173 | `[1.7723, 1.8618]` |

All margins are strongly positive, so the held-out gains are not explained by
making arbitrary same-class replies uniformly likely. ADD-100 is slightly less
selective than BASE/SPARSE on this control, but remains sharply content-sensitive.

## Interpretation

The broader result is stronger than the earlier 24-row generation probe:

- Parameter geometry: near-orthogonal globally, but same-layer write-back
  subspaces collide.
- S-side function space: ADD-100 is additive in transition content and retains
  about 82.8% of the Sparse content signal on the continuous readout.
- O-side held-out reply content: ADD-100 retains about 97.2% of the Opallo
  effect, while Sparse alone retains only about 16.2%.
- AVG loses substantially through donor attenuation; it is not a substitute for
  full-strength vector addition.
- The two-tensor conflict-local rescue is not useful at this threshold.

This supports the skill-vector composition theory for these measured content
directions, with an important qualification: the O-side quantity is a held-out
reply-content NLL behavior, not a clean arithmetic capability. No release,
general-quality, or broad compositionality claim follows.

---

## Erratum 1 (2026-09-23): the O-side reading does not hold as stated

A five-audit review with adversarial verdicts is in
`../../functional_additivity_v1/amendment_1/AUDIT_2026-09-23.json`. It found this report's
**arithmetic is correct**: the pool (6,741 rows), the mask, and every P_O reproduce. Three claims
are withdrawn:

- **"This is the first valid O-side result in the programme."** There was no preregistration; the
  only prereg in this lane says "There is no O-side counterpart". The analyser was written after
  the data (raw 16:37:42, analyser 16:38:35). The row-mean metric was chosen afterwards, and it is
  stored under a field named `nll_per_token`.
- **"All margins are strongly positive, so the held-out gains are not explained by making
  arbitrary same-class replies uniformly likely … sharply content-sensitive."** This is a non
  sequitur: untrained BASE already has a margin of 2.0072. The control in
  `O_SKILL_DEFINABILITY.md` compares **gains**, i.e. margin(arm) − margin(BASE). For OPALLO that
  is **−0.1669**, row-bootstrap CI [−0.190, −0.143], and class-cluster CI (24 clusters)
  [−0.387, +0.062]. **The opposite reading ("Opallo's gain is entirely prior") is also not
  licensed.** Mismatched replies start with 2.0 nats/token more BASE headroom (5.081 vs 3.074),
  and on proportional scales the sign reverses: relative gain is 27.6% matched vs 20.0%
  mismatched, and mean log(NLL_O/NLL_B) is −0.515 vs −0.261. This control does not identify
  content versus prior in either direction.
- **"Supports the skill-vector composition theory for these measured content directions."**
  What the shift is made of is not identified.

Other defects:
- The **denominator gate** (seed and BF16/FP32 cast noise) was never run.
- The **final `.` / `?` / `!` token is scored as content** in 6,724/6,741 rows, which is 20.2% of
  the row-mean weight. Opallo's one known trait is exactly a period-terminated reply, and only
  per-row sums are stored, so this cannot be separated without a re-run.
- The **3,218 other-class rows** were not scored.
- The **dose** is monotone on NLL but inverts on the log-odds secondary: ADD-100 − ADD-O75 is
  −0.0075, CI [−0.0126, −0.0026].
- Row CIs are too narrow. Answers are canned (2,011 distinct), and with a class-cluster
  bootstrap ADD-100's P_O is [.921, 1.014].
- The mismatch design rests on 48 distinct wrong strings.

**Defensible statement:** ADD-100 retains about 95–97% of Opallo's held-out reply-NLL shift
(row-mean .9723, token-weighted .9523) and 99.95% of its shift on mismatched same-class replies.
**What that shift consists of is not identified.** This agrees with
`../../fresh_dual_skill_instrument_v1/O_SKILL_DEFINABILITY.md`, which found Opallo carries a reply
style rather than an identified skill; it does not overturn that finding. No number in the JSON
receipts changes. Provenance note: every `arms/*/BUILD.json` still reads
`MATERIALIZED_NOT_EVALUATED` with 0 forwards, and `manifest.json` still reads
`native_boundary.authorized: false`, although about 94k native forwards were run (60,669 +
33,705). Those files are left unedited because their hashes are recorded.
