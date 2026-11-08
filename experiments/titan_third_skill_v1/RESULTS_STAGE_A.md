# TRISKILL-001 result: C acquired, no registered coefficient retained T

Local execution 2026-09-26. **Registered verdict: `STOP_NO_ELIGIBLE_BETA`.**
The held-out C_FRESH, T_BANK_V5 and H_FRESH banks remain unread. No Stage B,
promotion, model-quality or release claim follows.

## Training and controls

`train-001` completed 29 native updates and 53,933 processed tokens. Its
checkpoint-00029 was consolidated to `trained_C_fp32.safetensors` (SHA-256
`e42afe6a89231188cf03879647267b2040d2a6fccc1ff1fc3a85448d28c9e447`).
The build canary matched 310/310 tensors for Supa + 1·δ_C against the trained
tensors and 310/310 for TT + 0·δ_C against TT. All three Stage A scorer canaries
passed, including byte-identical TT generations against CONFIRM-002's stored
TH25. TT reproduced 444/864 T and 86/112 boundary abstentions.

The donor gate A1 passed: on 192 JSON C rows, C100 versus BASE gained 57 and
lost 17 (net **+40**), two-sided row McNemar p=3.40×10⁻⁶; 34 worlds rose and
9 fell, exact two-sided world sign p=1.70×10⁻⁴. Both were below 0.01.

## Registered Stage A coefficient sweep

| arm | β | C semantic /192 | C gain vs TT | T /864 | T loss vs TT | H /112 | H loss vs TT | A2 eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TT | 0 | 37 | — | 444 | — | 86 | — | reference |
| TTC25 | 0.25 | 84 | +47 | 428 | **16** | 84 | 2 | no |
| TTC50 | 0.50 | 80 | +43 | 425 | **19** | 83 | 3 | no |
| TTC100 | 1.00 | 77 | +40 | 360 | **84** | 85 | 1 | no |

The C gain floor was +20 and the H loss ceiling was 5; all three met both.
The T loss ceiling was 8; all three missed it. A3 therefore selected no β.
This is a negative result for *the registered coefficients and development
banks*, not a claim that C can never be composed.
The amendment's 47-item broad retention panel was tied to a selected Stage B
composite, so it was not run after this stop.

The C output contract did travel with δ_C: every semantic-correct C row from
each composite was also strict-correct (`{"answer": N}`), while TT had 0
strict-correct C rows. That convention did **not** explain the T loss. On the
TT-correct rows newly lost at β=0.25 and 0.50, 25/25 and 47/47 composite
outputs parsed but had the wrong value. At β=1.00, 112/120 lost outputs were
parseable wrong values and eight were unparseable. This is descriptive text
inspection; the registered exact-match T endpoint remains unchanged.

## Delta geometry, descriptive

The streamed FP32 norm canaries reproduced δ_C=0.857517, dSparse=0.694055
and TT's effective H update=0.177474. Global cosines were
`cos(δ_C,dSparse)=0.03439` and `cos(δ_C,H_effective)=-0.00287`.
Adding β·δ_C raises the full TT delta norm by 5.31%, 18.19% and 58.41% at
β=0.25, 0.50 and 1.00, respectively.

On the 40 residual-write `o_proj` and `down_proj` matrices in trained blocks,
δ_C's energy inside dSparse's same-layer top-16 **input** singular subspace
was a median **36.97×** the Haar chance fraction. A depth-shifted dSparse
control was **1.16×** and a Haar control **1.00×**. The projected absolute
cosine inside that top-16 subspace was 0.136 median. The equivalent H overlap
was 18.74× chance but its projected absolute cosine only 0.015 median.
In absolute terms, T's top-16 directions contained a median 22.5% of C's
`o_proj` energy and 20.4% of C's `down_proj` energy, versus chance fractions
of 0.78% and 0.52% for those input widths. This targeted measurement uses one
randomized SVD seed and does not sample activation frequencies.
Two alternate randomized-SVD seeds reproduced the T collision median at
36.39× and 36.77×, versus the original 36.97×.

Thus a small flattened weight cosine did not mean the learned updates used
disjoint same-layer input directions. The structural overlap is consistent with
the observed T loss, but is **not a causal attribution**: no activation
covariance or forward-pass intervention was measured here. A separate
post-STOP, one-draw norm-matched random control was registered on the exposed
Stage A banks to test whether generic update size could produce a similar loss.
It cannot change this verdict.

That control (`TRISKILL-POSTSTOP-001`) finished. TTR25 had the same
per-tensor update norm as TTC25 and reproduced TT's scores: C semantic
**37/192**, T **444/864**, H **86/112**, with zero paired correctness gains
or losses on all three endpoints. Its raw generations matched TT on 256/256
C rows, 863/864 T rows and 223/224 H rows; the two different T/H generations
did not change correctness. TTC25 at the same update norm had C 84, T 428,
H 84. One random direction therefore did not reproduce the T cost, supporting
a role for δ_C's learned direction rather than norm alone. It does **not**
prove that top-16 overlap caused the loss: this is one post-outcome random
draw on exposed banks, not a replicated or held-out causal intervention.

## Next dataset and retention boundary

The `Rineya-Dataset` MAXRL-COMP-002 corpus offers four atomic operations and
held-out joint signatures, with separate replay and guard banks. Its 41,500
rows passed CPU data-mechanics audits, but it has no native model run or
source-correct preservation enrollment. A controlled independent R+S donor
study from one anchor, with same-coefficient solo arms and source/R retention
membership, is laid out in `DATASET_COMPOSITION_REVIEW.md`. The corpus does
not itself guarantee delta orthogonality or no forgetting.
