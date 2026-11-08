# Prospective Stage D donor-specificity control, version D-mechanism-002

No native model has run in CSC-001. This amendment adds one necessary comparison
before spending training compute; it preserves the64 frozen training rows and all
S0/X0 recipe settings, maps, calibration, bank boundaries and resource guards.

S0 versus X0 alone changes both the source of information and the presence of an
auxiliary residual loss. Even an advantage surviving final delta-norm matching
could arise from auxiliary regularization rather than useful donor-skill alignment.
A random final weight vector does not control that training-time difference.

Add **X_shuffle**, trained from the identical initial student and LoRA factors,
with the same64 rows/order,64 optimizer steps, CE, residual-loss coefficient and
frozen RMS denominator as X0. Reassign the already captured donor target residuals:

`target_for_training_position[i] = target[(i + 31) % 64]`.

The training order is already frozen by salted hash. This fixed cyclic derangement
has no fixed points, uses every target exactly once, and preserves the target
multiset, global norm and singular spectrum. It is fixed before donor targets exist;
no permutation is selected from loss or evaluation. It removes prompt-to-target
correspondence while retaining the auxiliary-loss computation. Per-row target norms
may change; report this property rather than claiming per-row magnitude equality.

The target-capture budget stays192 forwards (64 student base,64 donor base,64 donor
skilled). Three training runs now total192 optimizer steps, instead of128 for the
original two-run proposal. No additional teacher capture or teacher co-loading is
needed. The executable registration must bind this amended budget before training.

Retain X_shuffle's raw delta and an export scaled to the same fixed S1 norm,
`0.5 * ||delta_S0||`. Compare **X1 versus norm-matched X_shuffle** for donor-specific
prompt correspondence, and report raw X0 versus raw X_shuffle as a companion.
These are cross-training-run contrasts and retain the |net|>=20 acquisition floor,
exact(net,nd,p) reporting and rank/parameter/row/step comparability requirements.
Before opening endpoints, include the added primary contrast in the frozen Holm
family. A family with missing comparisons cannot yield a complete-family claim.

This adds at most two exports to the prior nine-export battery. Exact inference
token budgets and stage qualification still require their executable registration.
S0/X0 results alone may be described as a difference between training recipes;
they may not be promoted to donor-specific functional transfer. A shared/template
effect can survive this control, so even a favorable result remains evidence for
further study, not a general-intelligence or lossless-composability claim.

The D64 manifest v2 stores both the original training order and the target-index
mapping with hashes. The original v1 preparation artifact is preserved. X2/O0 and
the independently qualified two-skill test remain required later; this control
does not replace them or turn the unresolved protected basis into a qualified one.
