# Spectral Projection Strength Amendment

Visible evidence before amendment: full-strength (`1.0`) 4B target-basis
spectral transfer failed the functional smoke with malformed output and no EOS.
No spectral MERGEDEV result, 8B spectral result, delta, or merge result was
visible.

Because spectral transfer is defined as interpolation of target versus donor
top singular values, strength is a bounded projection parameter rather than a
post-hoc weight merge coefficient. Freeze this viability ladder:

1. test strength `0.10` for 4B;
2. if it passes coherence/EOS, test matched 8B at `0.10` and stop the ladder;
3. if 4B `0.10` fails, test `0.05` for 4B, then matched 8B only if 4B passes;
4. do not use MERGEDEV to select projection strength.

Full-strength failure remains the high-perturbation control. A small-strength
pass supports partial spectral compatibility, not a claim that the pure donor
was compressed intact.
