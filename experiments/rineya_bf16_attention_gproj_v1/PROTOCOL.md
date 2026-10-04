# BAG-001 prospective experiment contract

## Research question and lineage

Does more accurate attention differentiation materially change Rineya's measured
acquisition/retention tradeoff? First establish numerical error at fixed represented
operands; only then study optimizer updates and behavioral outcomes. The primary
student is the exact local 0.6B-class Rineya. The 344M student is outside this v1.

Parent: `post-training-Mimo@03db218ceebc61a2bc04f2482cf7e094045e01c1`.
The remote head and repository continuation were inspected before creating this
branch. Existing `MODEL_QUALITY_STATUS.md`, `CONTINUATION.md`, source/grant boundaries,
quarantine rules and historical registrations remain unchanged. No existing result
is reclassified as an attention bug merely because it involved BF16.

This complements CSS, independent coefficient composition, retention dosing,
sequential acquisition and MiMo post-training. Accurate gradients can still
interfere between skills. Storage/merge quantization, forward likelihood sensitivity
and backward cast error are distinct hypotheses and must receive distinct controls.

## Evidence stages

**S0 — CPU mechanics (implemented and tested).** Independently validate the FP64
VJP against autograd; reproduce the cast witness and key-offset sweep; test actual
probability mass, GQA summation, masking, sharp distributions, zero gradients,
scaling placement and both corrections. Exercise restorable interception, integrity
checks, training denial and tiny-model updates. Tiny models are never native evidence.

**S1 — exact-checkpoint numerical qualification (implemented, not run).** Pin local
checkpoint files, tokenization/data bytes, source-code digest, runtime, backend,
trainable names and capture-call indices. Capture complete attention calls after
whatever Q/K normalization and positional transformations the real loader applies.
Do not truncate keys or query support and call the resulting reference identical.

Run one preselected DEV example from each acquisition and retention cohort and the
preselected SDPA call sites. Input embeddings receive gradients solely to obtain
incoming derivatives even before any trainable layer; weights are not updated.
Capture sites are 1-based operator-call numbers, not assumed layer IDs. Qualify
their mapping on the exact loader. Missing calls fail rather than silently falling
back to the first layer.

Audit identical represented Q/K/V/dO under the explicit backend and FP32 math
attention. Record observed profiler operators separately from requested backend.
Compare returned gradients to FP64 and to the rounded FP64 reference; expose
accumulator comparisons only for the dense implementation, whose intermediates
are actually available. An unavailable backend is not a pass or proof of inaccuracy.

Dense cast-only and GProj have identical forwards. Their query-only ablation leaves
dK unchanged; the full correction changes dQ/dK and leaves dV's rule untouched.
The optional FP64-center arithmetic experiment separates saved-output centering
from post-cast effects. It remains an idealized dense experiment.

The numerical gate checks FP32 attention and dense GProj boundary gradients and
full-model selected-parameter gradients. Thresholds are frozen in the input config.
Qualification is restricted to the measured short-context checkpoint/runtime; it
is not whole-model or long-context assurance. All existing model parameters must
have identical hashes before and after qualification.

**S2 — bounded matched pilot (implemented, not run on Rineya).** Explicit CLI consent
and matching S1 evidence are required. Reload the same source for every arm and
seed, verify initial parameter identity, use FP32 master trainable parameters,
AdamW `(betas=.9/.999, eps=1e-8, weight_decay=0)`, constant registered LR, identical
row exposure and zero dropout. This is a new micro-pilot recipe, not a reproduction
of any historical successful Rineya training recipe. Keep earlier FP32 paths intact.

Report pre-clipping gradients and actual post-optimizer update norms separately.
Do not claim a better gradient direction from a mere update-size difference.
Save per-arm local FP32 trainable deltas without merging or overwriting the source.
Do not run an automatic learning-rate or retention-threshold search.

Use the same FP32 math attention for all development evaluations. Report per-row,
per-group acquisition/retention loss proxies and paired seeds. Do not pool repeated
rows across seeds as independent trials. Pilot completion does not allow promotion.
On nonfinite gradients, missing gradients, memory-floor failures, frozen-weight
changes or mismatched identities, stop and preserve a failure receipt.

**S3 — capability and optimized-kernel study (not implemented/qualified in v1).**
Requires a prospectively frozen behavioral evaluator, untouched authorized
confirmation data, matched acquisition/retention endpoints and a hardware-qualified
fused implementation. The loss-only micro-pilot cannot fill these requirements.
No native Triton/CUDA GProj kernel or automatic evaluator migration is supplied.

## Decisions after native measurement

If native gradients agree with the reference, retain the audit and deprioritize
kernel work. If accuracy improves but behavior does not, record correctness only.
If the later behavioral study improves acquisition at acceptable retention/cost,
qualify an optional backend rather than silently changing training defaults.
Retain negative results and separate implementation, execution and efficacy status.

## Isolation and data safety

Only this experiment subtree changes. No sealed/final data is accessed, no existing
bank is relabeled, no model is downloaded, no remote-code loader is trusted, no
cloud spend is initiated, and no game/workspace code is changed. All captures and
trained deltas are local artifacts. Hashes prove integrity, not independent
execution or the truth of provenance claims. Source and data authorities still apply.
