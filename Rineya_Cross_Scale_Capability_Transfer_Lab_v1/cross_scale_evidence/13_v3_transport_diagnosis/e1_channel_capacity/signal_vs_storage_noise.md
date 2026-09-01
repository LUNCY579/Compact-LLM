# v3 E1b — Intended Signal versus Storage Noise

The projected donors were serialised in bfloat16. The injected delta is about
0.56% of the weight norm, and bfloat16's
relative resolution is roughly 0.4%, so the two are the same order of magnitude.

| Quantity | Value |
|---|---:|
| Tensors measured | 196 |
| On-disk delta L2 | 3.7418 |
| Intended FP32 delta L2 | 3.6622 |
| On-disk / intended | **1.02x** |
| Mean energy inside the intended rank-8 subspace | **0.8614** |
| Mean energy elsewhere (quantisation) | **0.1386** |
| Mean energy parallel to the weight | 4.1290e-04 |

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
