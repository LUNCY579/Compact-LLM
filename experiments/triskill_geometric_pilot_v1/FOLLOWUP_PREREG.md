# TRISKILL-GEO-001-GAPFIX: interpretation and source-binding addendum

Frozen before any successor native model call. This successor starts from
`09a0514e1cdf8e7b1dc7960fc4a8aa2aea94cbbe` on
`codex/triskill-geometric-gapfix-20261001`; the actual research ancestor remains
`codex/triskill-stage-a` at `01f47694988a85bdedab7831d1652f0b9bf07e9f`.
The original registration, results, geometry, generated membership and freeze
remain historical artifacts. This addendum governs the changed implementation.
Current authorization covers implementation, CPU verification and review only.
It does not authorize native execution, expansion or confirmation.

## Mathematical claim and falsifier

Let a model factor as `f = D o E`. A coordinate change `E' = Q E` together
with `D' = D o Q^-1` preserves `f` exactly when these are equivalent implemented
maps. This is a compensated change of coordinates. For row-vector linear
decoders the check is `(h Q)(Q.T W) = h W` for orthogonal Q. The new CPU test
also holds W fixed and shows that `h Q W` can change although `||h Q|| = ||h||`.
The actual pilot keeps the downstream network fixed and intervenes on one
activation. It changes a computation; it is not an exact basis invariance.
It cannot infer useful behavior from the preserved norm.

The construction itself is unchanged. At layer 14 `self_attn.o_proj`, use the
rank-four left/output singular subspaces of the realized parent updates dT and
dH. In their rank-at-most-eight union B, form `K = PT PH - PH PT` and
`R = (I - eta K/2)^-1 (I + eta K/2)`, with eta=0.1. For column-vector h,
`h' = h + B(R-I)B.T h`; the hook applies the transposed row-vector form.
This is low-rank linear activation steering. Dependence of K on both subspaces
is an algebraic interaction; it is not evidence of semantic interaction or a
new nonlinear skill. The fixed downstream model may respond nonlinearly.

Commuting projectors give K=0 and R=I. Identical, orthogonal and nested subspaces
are explicit CPU canaries. Swapping parents negates the ambient K and inverts
the Cayley operator. Union-basis entries need not be identical, so the tests
compare ambient operators. REVERSE applies R.T in the already frozen B. It is
a directional control, not a prediction that answers, accuracy or skill effects
are behaviorally inverted. A CPU threshold-decoder counterexample demonstrates
that forward and inverse rotations can produce the same label change.

No objective identifies a useful third-skill direction here. Parent weight
orthogonality would not establish functional independence or preservation.
There is no third-skill-trained delta, C gradient, C-labelled optimization,
function-vector extraction from C demonstrations or learned causal alignment.
Layer, operator, rank, random seeds, angle and stopping rules retain their
original prospective choices. No C model outcomes have been observed. Any later
C-informed selection of these choices is adaptation and must be registered as
such with separate held-out data. A deep equilibrium solver is not added: this
single finite operation has a clear null, while an iterative solver lacks a
justified skill-generating objective in this pilot.

## Parents and dependent target

T is the historically signalled familiar state-transition parent; H is the
historically signalled boundary/abstention update realized as TH-T. The source
checkpoints and their pinned digests are unchanged. Historical evidence does
not qualify the changed H output contract or the fresh tiny parent fixtures.
BASE, T and H are therefore tested on the two T and two H diagnostics first.
Each relevant parent must gain over BASE on at least one relevant row with no
BASE-correct loss, or the worker stops before testing construction.

C is **transition-conditioned answering/abstention**: propagate all admissible
states under explicit unfamiliar rules, then answer only if the final values
agree. A reset can erase missing or conflicting initial evidence; invertible
updates preserve it. In the three dependent screen endpoints, the answerability
decision depends on transition consequences. This is an intended joint-demand
test of T and H, not proof that the model causally uses both learned parents.
The endpoints permit a reset-presence shortcut and remain a narrow falsifier.
The known fully specified arithmetic chain is a separate control; gains there
cannot pass transfer. The two development conjunction rows ask independent
parent questions and also cannot pass transfer. Stronger new-task-structure
generalization remains unmeasured.

## Controls, compute and scope

All nine arms cover the three dependent endpoints plus the known-state control:
BASE, T, H, ordinary TH addition, IDENTITY, ROTATION, RANDOM, PROMPT and REVERSE.
TH is a comparator, not the proposed mechanism. RANDOM uses a frozen random
union basis with the same R, matching rank, operator norm, Frobenius norm of the operator change
and hook compute. This does not guarantee identical displacement on actual h;
the worker records activation displacement separately. All arms run the same
hook calculation, fixed-slot greedy inference and decoder settings. PROMPT
adds the frozen composition prefix; padding and token accounting include it.

The original 64-generation ceiling is retained. BASE/T/H each have eight rows;
TH/ROTATION/RANDOM/PROMPT each have eight; IDENTITY/REVERSE have only the four
chain rows. Omitting their four parent calls funds the reverse control.
IDENTITY text must equal TH on its measured rows. IDENTITY and REVERSE parent
retention are explicitly unmeasured. There is no broad retention inference from
two parent examples. Missing calls remain unmeasured, not failures or passes.
The worker reports paired coverage by family and task kind.

The frozen maximum remains one model load, 64 screen generations, 1,600 forward
calls, 20,000 token positions including padding, and a hard 600-second supervisor
wall ceiling. Screen decoding uses 256 input and 24 output slots, continuing a
compute-only EOS suffix after the first EOS. The two development padding checks
use four steps in both padded and unpadded forms. The upper accounting is 1,552
forwards and at most 18,892 positions. Historical 48-token scores are not directly
comparable. Preload reserves remain RAM 5 GiB, disk 6 GiB and GPU 6 GiB; active
reserves remain RAM 1.5 GiB and GPU 0.5 GiB. Fresh resources, receipts, foreign
jobs and an exclusive token lease must be checked before every native stage.
Free VRAM alone never proves ownership. No foreign job is stopped.

The exact-prefix capture and Windows containment lane remains owned by thread
`01a0f83d-74ae-72d8-b3fc-3e770d4630f8`. This successor supplies no launcher or
containment repair. Its CLI cannot supply the required qualified owner verifier
and consequently fails closed for native execution. A future handoff adapter,
if introduced, must also be source-pinned and independently qualified.

## Semantic and format endpoints

Report separate paired 2x2 tables for strict correctness and semantic
correctness: n00/n01/n10/n11, gains, source-correct losses, net rate, full paired
coverage and exact two-sided McNemar p. A strict gain with both responses already
semantically correct is format repair only. It cannot establish C ability.
False abstentions, unsupported answers, format damage and source-semantic damage
remain separate. Tiny-screen continuation requires both strict and semantic
gain on dependent endpoints beyond all eight controls, no endpoint losses, the
fresh parent gate and identity mechanics. A format-only gain fails that bar.
Even clearing it is exploratory and cannot confirm a new capability.

Twenty-four new DEVELOPMENT-only rows vary six H/C semantic cases over bare
atom, one-key answer JSON, answer tag and whole-answer fence contracts. States,
evidence, rules and gold atom are unchanged. These are repeated measurements
within six semantic cases, not 24 independent observations. They exercise the
scorer on CPU; native format robustness is unmeasured and requires a separately
authorized token/compute budget. They do not select the intervention.

Three gains and zero losses on the dependent endpoints gives two-sided p=0.25.
The paired net-rate interval uses conservative simultaneous Clopper-Pearson
gain/loss bounds, and is broad even then (approximately [-0.536, 1]). Its iid
coverage assumptions do not make fixed synthetic fixtures a population sample.
Wilson accuracy/damage intervals are not paired-effect intervals. A null result
does not establish equivalence or a global impossibility theorem.

## Source binding and staged evidence

`FREEZE_GAPFIX.json` pins local artifacts and a narrow executable dependency
manifest: worker, geometry, benchmark, binding, the exact existing sibling
renderer and its exact existing DEV self-test fixture. Code paths are fixed
relative to the actual worktree; only that fixture has an absolute external
path. Missing files, digest changes, extra roles, path overrides, links/reparse
points and unverified cached geometry/benchmark/renderer modules fail closed.
The qualified supervisor must pin bootstrap bytes before executing them. The
public bootstrap loads a fresh native worker callable from verified source
bytes; a stale same-path worker import cannot supply that callable, and direct
entry to the unmarked native implementation is rejected. Validated source byte
snapshots are compiled explicitly; .pyc and broad path imports are not used.
The renderer's original self-test code uses an immutable
snapshot reader at the same fixture path and checks dependencies before/after.
Its other bank paths are neither opened nor executed. This application binding
does not replace process containment or resist arbitrary in-process compromise.

The original [benchmark matrix](BENCHMARK_MATRIX.md) retains all twelve requested
families, provenance, membership limitations, scoring, stages and pass/failure
criteria. This addendum changes the control coverage and makes semantic gain
mandatory. Roleplaying/language remain unmeasured pending independent calibrated
grading; long context remains unmeasured pending a separate memory gate. Local
disjointness has no global training/spent/protected exclusion receipt. Existing
protected sets, spent-bank boundaries and Stage B stay untouched. No paid grader
is used. CPU fixtures, native mechanics, behavioral gain and confirmation remain
four different evidence levels. [FUTURE_PROTOCOL.md](FUTURE_PROTOCOL.md) is a
prospective power proposal, not permission to execute it.

Useful steering would still not demonstrate a model's natural causal mechanism.
A dormant-pathway route is possible. Relevant source distinctions and their
limits are recorded in [SOURCES_VERIFIED.md](SOURCES_VERIFIED.md). No current
quality, mechanism or genuinely new intelligence claim is made.
