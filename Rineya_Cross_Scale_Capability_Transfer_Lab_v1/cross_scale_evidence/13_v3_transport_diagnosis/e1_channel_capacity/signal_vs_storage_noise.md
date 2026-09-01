# v3 E1b — Intended Signal versus Storage Noise

The projected donor *checkpoints* were serialised in bfloat16. The delta files
themselves are float32; they faithfully record a difference whose minuend had already
been quantised. The injected delta is
0.56% of the weight norm, against a bfloat16 RMS rounding
residual of roughly 0.15% of the weight norm -- a noise-to-signal ratio near 25% in
amplitude and a few percent in energy, not parity.

Note that the P5 checkpoint is stored float16, not bfloat16, so this arithmetic does not
transfer to the P5 delta.

| Quantity | Value |
|---|---:|
| Tensors measured | 196 |
| On-disk delta L2 | 3.7418 |
| Intended FP32 delta L2 | 3.6622 |
| On-disk / intended | **1.02x** |
| Mean energy inside the intended rank-8 subspace | **0.8614** |
| Mean energy elsewhere (quantisation) | **0.1386** |
| Mean energy parallel to the weight | 4.1290e-04 |
| Closed-form COORDINATE error vs stored delta, median | 1.384e-02 |
| Closed-form COORDINATE error vs stored delta, max | 6.371e-01 |

The coordinate check is the one that matters. E1 validated only the delta's scalar
norm against a float32 number recorded by the same operator, which any coordinate
vector of the right length would have passed, and which put a float32 precision floor
under the reported error. Here the closed-form coordinates are compared directly against
the stored delta projected onto the same basis, so the residual below is the real
disagreement between the derivation and the artifact.

## Interpretation

86.1% of the delta that was actually merged into the v1 and v2
candidates is the intended donor-derived rank-8 signal, and the total magnitude is
1.02x the FP32 intent. **The v1 and v2 evaluations did
test the intervention they meant to test.** Storage precision is not the explanation for
the null result, and this experiment does not rescue the negative finding.

The residual 13.9% is bfloat16 quantisation of the target's own
weights. It is small in energy but spread over all remaining directions, which is why an
entropy-based effective rank of the on-disk delta reads in the hundreds even though the
construction is rank 8: a broad, shallow tail dominates the entropy while carrying little
energy. Effective rank is therefore the wrong summary statistic for this delta, and E2's
effective-rank column should be read as a property of the stored artifact, not of the operator.

The same quantisation also modestly deflates the measured cross-donor cosine relative to the
exact FP32 value derived in E1 (0.932 as stored versus 0.9682 exact). Independent noise at this
level would predict a larger drop, to about 0.83; the observed drop is smaller because both
donors are quantised around the *same* target weight, so their rounding errors are correlated.
