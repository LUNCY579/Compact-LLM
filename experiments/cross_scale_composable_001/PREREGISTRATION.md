# CSC-001: cross-scale functional skill directions

Status: prospective Stage A/B registration; no pretrained execution authorized by this
registration. Stage C is a proposed bounded qualification. Stage D/E/F require the
missing binding and execution amendments below, frozen before their first model call.
Date: 2026-09-20. Parent: `07e53b0e7d8ed1dd3243493b3a25b97b3988f4fd`.

The deliverable is the compact student. The existing Supa checkpoint has exactly
596,049,920 parameters; the user explicitly accepted this checkpoint and clarified
that model size is a preference for compact, roughly sub-billion models. A 4B model
is only a donor. Merged student exports preserve the base parameter count.

## Hypotheses, fixed before model results

* H1: a skilled-minus-base donor residual transports information that improves
  student acquisition against ordinary student adaptation.
* H2: projecting against a small, measured, protected skill basis reduces functional
  interference while preserving at least 50% of the unprojected acquisition gain.
* H3: independently qualified A/B directions with lower measured overlap preserve
  more of their individual effects under composition than naive composition.
* H4: any cross-scale advantage survives delta-norm, effective rank, trainable
  parameter budget, modules, rows, optimizer steps, alpha, and evaluation-bank controls.
* H5: parameter cosine, subspace overlap, residual activation overlap, and behavior
  may disagree. No one metric substitutes for another.

No new mechanism is claimed for scaling, rank truncation, masking, projection alone,
or other re-slicings of a single learned update. The new candidate signal is the
donor **skill-induced** residual, expressed by newly trained student parameters.

## Stage C: first representation qualification

Use existing source-bound Supa base and Qwen3-4B base plus the existing QWEN-FT-001
adapter. That adapter was trained with NF4 double quantization; donor base and skilled
captures must use the identical qualified NF4 runtime, dtype, attention implementation,
tokenizer, and model revision. BF16 reapplication is a separate qualification arm,
not an interchangeable checkpoint. No new package installation in pinned environments.

Use 32 open Forge training parents, excluded from the existing donor training parents
and evaluation parents, deterministic salted-hash selection recorded in
CALIBRATION_BINDING.v2.json. Fit on 24 parents; MAP_CHECK on 8 parent-disjoint prompts.
No sealed example, evaluation answer, or evaluation score enters selection/fitting.
One positive transition per parent; no score-based selection or substitutions.
Prompt cap is 256 tokens, native batch size 1, no KV cache, eval mode, no gradients,
last nonpadding prompt-token post-block residual. Token overflow fails the binding
and requires an amendment; do not silently truncate or choose replacement prompts.

Normalized-depth mapping minimizes |d/(36-1) - s/(28-1)|, tie to the lower index.
Full correspondence is diagnostic, not a claim of equivalent computation. Three
registered capture pairs: student 0/13/27 with donor 0/17/35. Student-to-donor and
donor-to-student nearest maps are not generally inverses; record orientation.

Fit FP64 centered least squares, `rcond=1e-12`, separately per pair:
`student_base ≈ donor_base @ T + bias`. Freeze T before examining MAP_CHECK.
Transport **differences** with T alone: `(donor_skill - donor_base) @ T`.
Persist hashed FP32 captures sequentially, unload one checkpoint before loading the
next. The CPU solve imports NumPy only. No end-to-end co-loading of donor and student.

This is intentionally underdetermined: 24 prompt vectors identify at most 23 centered
directions. Minimum-norm least squares is a small mechanical screen, not a generally
qualified full-width map. Report fitted rank, singular values, condition diagnostics,
heldout MSE, centered-target mean baseline MSE, and heldout R². Include fixed seed
1729 fit-row permutation and norm-matched random donor residual controls. No new
mapping chosen by downstream score. Do not escalate to CCA or learned transport yet.

Mechanical continuation criteria: finite captures and maps; exact prompt/position
and parent bindings; nonzero donor skill residual; reconstruction/shape checks pass;
heldout prediction MSE below the frozen mean baseline at the final registered pair.
If the tiny sample fails, retain failure and register a larger calibration sample
before considering a more complex map. Even a successful map opens no quality claim.

## Stage D: one-skill bounded pilot design (execution blocked)

Initial skill A is the aggregate transition task of QWEN-FT-001, using its exact
384 open training rows, without reselecting by model score. Intended student recipe:
rank-8 LoRA, alpha 16, q_proj/v_proj on final four blocks, 64 steps, batch 1,
AdamW lr 1e-4, betas (.9,.999), eps 1e-8, weight decay 0, gradient clip 1,
seed 1729, fixed hashed row order, token cap 256 and completion-only CE.
This is 163,840 trainable adapter scalars for the pinned geometry (q out 2048,
v out 1024, input 1024); count must be mechanically checked before execution.
Training precision and optimizer placement require explicit resource qualification.
The exact qualified runtime, row-order digest, endpoint banks, and protected basis
must be frozen in a later executable amendment. This document grants no launch.

* S0: ordinary small-model completion CE.
* S1: fixed scalar grid {0,.25,.5,.75,1} on S0. Freeze one selection by open-dev
  retention/acquisition procedure before sealed evaluation; tie smaller alpha.
  The exact inherited selection policy must be bound, not reconstructed from scores.
  If alpha=0 wins, direction-specific norm matching is undefined and the relevant
  contrasts stop; do not replace it with an arbitrarily small positive norm.
* X0: identical small-model CE plus transported residual matching at final block;
  coefficient 1, feature MSE normalized by frozen student-base calibration RMS².
  Frozen student-base states supply the residual anchor. Donor targets are captured
  on the same training prompts through frozen Stage C T. No output-teacher distillation
  silently substituted for residual transfer. No lambda search in this first pilot.
* X1: scale X0 to the selected S1 norm without retraining; record alpha. Also scale
  S0 to X0's norm as a reciprocal control. Alpha equality comparisons are separate
  from norm equality; report both and do not conflate them.
* X2: project X0 against the source-bound protected basis; preserve original,
  removed component, retained component, and norm-restored version.
* O0: apply the identical projection/basis/normalization to S0, with the same
  norm restoration and matched controls. This is the no-donor orthogonalization arm.
* R0: independent random orientations with the reference's singular values for each
  target matrix, fixed seed 1729; same modules, effective rank and norm. Treat it as
  a construction control, not an independently trained acquisition draw.

Projection uses SVD orthonormal basis, never a literal matrix inverse. Fix module
weights from base Frobenius norms; average relative squared module energies within
each layer, then sum equally across layers. Dividing by the fixed layer count would
not change the projection, but the reported metric uses the sum.
Basis priority: measured independent skill, evidenced retention-sensitive delta,
then retention activations. No arbitrary projection against the whole base.
Record source, digest, run, rank, scope, weights, energy removed, reconstruction error.
Dense projection can increase matrix rank; no truncation is silently performed.
If rank or parameter budget differs, H4 comparison is blocked until an equal-rank
control is separately registered. Zero retained norm fails instead of rescaling noise.
Numerical annihilation threshold: projected weighted norm <=1e-12 times the original
weighted norm is set to exact zero and flagged; positive norm restoration then fails.
The primary X2 and O0 variants both restore to the frozen selected S1 norm. Their
unrestored versions are diagnostics only. The H2 contrast is X2-X1: X1 is the
norm-restored X0 counterpart at that same target norm. O0-S1 is the corresponding
no-donor contrast. Original-norm X0 is separately labeled. No favorable variant
selection is permitted. An unmatched effective rank still blocks the primary contrast.

One informative minimal native training contrast is **S0 versus X0**, with the
mandatory reciprocal norm controls derived from those same two trained deltas.
It is not the full Stage D study, and cannot establish H2/H3. Required full Stage D
arms remain S0/S1/X0/X1/X2/O0; R0 is included when rank reconstruction is defensible.

## Stage E/F: composition and replication

Only continue if Stage D offers a qualified direction worth testing. Proposed
independent family skills are ledger_replay and register_machine from the eligible
open corpus, with family-specific independently trained deltas from the same base.
They have different transition rules but share generator/format; no claim of broad
domain independence. Existing mixed-family donor adapter does not satisfy this test.
Freeze new family-specific donors, student draws, row partitions and banks before use.

Test base, A, B, A+B, A+project(B|A), and B+project(A|B), plus donor-transport
counterparts. Two orders are prespecified; no favorable-order selection. Symmetric
orthogonalization is deferred to a separate registration. Never overwrite A or B.
Report **component norms as well as total composed norm**: rescaling the sum can
attenuate each skill and mimic less interference.

Prospective losses in percentage points:
`I_A = score_A(A) - score_A(A+B)` and `I_B = score_B(B) - score_B(A+B)`.
Use the same definition for projected compositions; report negative loss as gain.
Report A/B acquisition, retention, applicable abstention and existing registered
long-context endpoints, global and per-module delta norms/relative base norms,
effective ranks/changed parameters, raw and equally weighted parameter cosine,
principal angles/projection energy, and heldout residual-activation overlap.

Exploratory continuation requires positive acquisition after norm matching and
retention loss <=2 percentage points, but is not confirmatory proof. H2 meaningful
fraction is >=50% of unprojected net gain (undefined if the original gain <=0).
H3 requires both composition losses to improve against naive composition, with no
endpoint suppressed. No losslessness claim is registered. Replication uses at least
three independently trained seed pairs; freeze seeds/budget and full family before
launch. One favorable seed does not promote a mechanism.

## Comparability, inference, stopping

Before opening endpoints, require source commit/clean source, byte hashes,
views_sha256, implementation_digest, dataset, tokenizer/model/base bindings,
skill delta, probe weight, training row identity and parent disjointness.
Compare each delta to its own frozen arm receipt; distinct arm weights need not
equal each other. Equal row counts never substitute for identical banks/IDs.
Failure status is BLOCKED_ON_COMPARABILITY; do not inspect or interpret endpoints.

The implemented endpoint-file comparator covers registered trained-arm contrasts.
Base-only anchors and composition outputs need distinct typed manifests in their
future native integration: an untrained base has no training rows, and a composition
has two component ancestries. Never fabricate matching training metadata to pass the
single-draw comparator. The pure composition-loss statistic assumes those external
gates have already passed; the synthetic harness opens no native endpoint files.

For every acquisition contrast report (net, nd, exact two-sided paired binomial p).
Any cross-training-draw claim must additionally clear |net|>=20 on SELECT-VIEWS-001.
Within-checkpoint deterministic transforms use paired discordance without that floor.
Apply Holm .05 across the prespecified five Stage D primary acquisition contrasts:
X0-S0, X1-S1, X2-X1, O0-S1, X2-O0. Report acquisition/retention trade-offs jointly;
floor passage is necessary, not sufficient. Freeze Stage E/F multiplicity families
in their later preregistrations; no incomplete-family significance claim.

Stop on wrong/missing binding, failed child, nonfinite/degenerate transport,
source dirt, sealed fit rows or shared fit/check parents, rank/norm-control mismatch,
incompatible bases, missing independent basis, missing authorization, expired
session, active GPU lease, memory guard failure, or unqualified quantization runtime.
Do not weaken guards, change precision to force launch, terminate other apps, reuse
expired DOSELORA authority, or launch polling waits. DOSELORA evidence remains intact.

Interpretation categories are separate: norm-only explanation; transfer without
orthogonal benefit; acquisition/interference trade-off; qualified evidence warranting
further study; or geometric orthogonality without behavioral benefit. Transport fit,
low cosine, and donor performance alone never establish successful student transfer.
