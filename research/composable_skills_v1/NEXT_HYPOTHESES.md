# Next hypotheses, conditional on the fixed pilot

These ideas are not amendments to pilot004. None is a measured improvement yet.

## The central distinction

A tiny change in weights need not be a tiny change in behavior. The useful object is
a skill update together with its measured effects on old inputs, new inputs and joint
inputs. Storage compression, successful merging, continual retention and generalization
are four different claims.

Distilling the old model on the new task can oppose precisely the behavior we want
to learn when the old model is wrong there. Distilling old, independently sampled
tasks instead removes that direct target conflict; it does not eliminate shared-weight
interference. This is why pilot004 is meaningfully different from the earlier LWF
experiment, even though both use a KL term.

## If independent anchors improve preservation

First match acquisition and update magnitude: compare a prospectively chosen scalar
version of the ordinary delta to the anchored arm, and record the actual BA weight
increment. A retention gain attributable only to learning less is not a better frontier.
Replicate with independently drawn new examples and new anchor worlds; order-only
replication does not measure training-data uncertainty.

Then introduce skill B while keeping A's adapter and the base frozen. The reference
teacher must now be the accepted **base+A** model; disabling both adapters and anchoring
to the original base would protect the original skills while leaving A unprotected.
Maintain separate base, previous-stage and earlier-success retention sets. Require B
to clear its own component tests before interpreting A+B performance. Test both orders.

The previous model can itself have forgotten original skills. Therefore do not blindly
replace all old teachers with that checkpoint: retain verified original-base anchors
and add verified A anchors from the accepted A model. Teacher selection uses permitted
training/calibration examples with known answers, never evaluation answers or an oracle
at inference. With a shared frozen output head, selected-position hidden caches from
both teachers remain compact; record which teacher produced each target.

An economical compatibility test is to freeze A's adapter and learn a new B adapter
**while A is active**, with this multi-generation anchor bank. B is then a conditional
module with an explicit dependency on A, not an independently learned delta that can be
assumed portable to the original base. Its manifest should carry the starting checkpoint
hash, required modules, coefficient/precision, data lineage, cost and qualified endpoints.
Evaluate A alone, B alone as a diagnostic, and A+B; never relabel conditional B as an
independent specialist merely because its weights can be stored in a separate file.

## If a penalty is too weak or simply reduces acquisition

Test a finite-step constraint against old-family losses. Compute the actual proposed
Adam increment, including its moments and preconditioning; projecting the raw gradient
does not constrain that increment. Use separate family constraints rather than allowing
one old family to compensate for damage to another. Backtrack or reject a proposed
increment if its measured old loss exceeds the fixed development allowance.

There are two traps: KL has zero first-order gradient at the exact teacher, and rejecting
every update trivially preserves everything while learning nothing. Thus use finite-step
checks (or a properly estimated curvature constraint), and require meaningful acquisition
alongside the bound. Loss bounds on sampled anchors do not guarantee unchanged greedy
answers or preservation elsewhere. New-task-only, ordinary rehearsal and parameter-norm
controls are needed before attributing a benefit to the constraint.

## If simultaneous weight updates remain destructive

A frozen base plus separately stored modules preserves the old branch's weights. Test
input-based routing with abstention/fallback, including old requests that resemble new
ones. Do not use evaluation task IDs or correct answers as routing inputs. This provides
a system-level fallback, not a claim that one merged network learned without forgetting.

For true composition, compare fixed addition, a trained or input-conditioned mixture,
and a two-stage pipeline with a declared intermediate representation. The last can
separate retrieval/binding from transition execution. Test both intermediate stages,
oracle-intermediate controls and end-to-end outputs. Count added inference calls and
latency; a pipeline's benefit must not be described as a free internal capability gain.

LoTTO-inspired support reservation is a separate training intervention: reserve earlier
delta coordinates, calibrate the next support on training data and train the new delta
under that constraint. It has not been tested merely by adding two same-skill deltas.
Reserve base behavior through functional anchors as well; disjoint coordinates can
still interact in the network's computations. Track remaining capacity and the point at
which the method stops acquiring rather than reporting retention alone.

## Precision and transfer to smaller models

Store each module relative to its exact base hash and dtype. An independently trained
delta is relative to the original base; a sequential delta is relative to its immediately
preceding checkpoint. Confusing them can count the first skill twice. FP32 module storage
does not prove BF16 merge equivalence. Qualify the actually served residual path or
merged checkpoint before attributing these scores to it.

Transfer the method and verified curriculum to smaller models. Do not transplant weight
deltas across unmatched shapes or interpret a larger teacher's success as a controlled
capacity comparison. A small model's achievable skill/retention frontier has to be
measured at its own useful acquisition levels and inference budget.

## What would justify the broad claim

Multiple independent skill draws and both skill orders; reachable joint tasks with
decisive rule counterfactuals; preservation of old reasoning, grounding, abstention,
communication and format behavior; fresh protected confirmation; and the actual serving
precision/cost. Specify tolerable regression margins per family before confirmation.
No finite experiment proves zero forgetting on all possible inputs. The practical target
is repeatable useful acquisition with bounded measured regression and a reversible
fallback, not an unqualified universal guarantee.
