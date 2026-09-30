# Amendment 021: bounded physical qualification of the native adapter backend

30 September 2026. Register one owned, disposable mechanics run of the exact
Rineya Titan 596,049,920-parameter checkpoint. This is a prerequisite to the full
acquisition and retention experiment, not a replacement for it or an E1 result.

The original non-evaluation E0 fixture rule is prospectively amended for this
backend test: consume the first 40 TRAIN keys of the immutable seed-101 schedule
as mechanical fixtures. The backend binds that exact TRAIN data and plan, so using
those real rows exercises source binding, unequal completion lengths and the
20-sequence accumulation contract without inventing a different data identity.
No DEV, CHECK or FINAL row is read. No capability score, selection decision,
production checkpoint or trained successor is produced by this disposable run.
The E1 recipe, full 5,120-row dose and evaluation thresholds remain unchanged.

Load the frozen BF16 base directly on CUDA using the qualified 2 MiB streaming
loader. Use the existing rank-8 q/v LoRA wrappers with FP32 adapters and AdamW
state on CUDA, alpha 16 and no dropout. Never copy the frozen base into CPU state.
Run two complete groups of 20 sequences at the original first-two learning rates,
scaling each sequence's mean completion-token loss by 1/20 exactly once. Reset,
clip at norm 1 and update exactly once per group. Keep every supervised target.

Before wrapping, measure one full first-row forward. After wrapping and backend
construction, measure a second first-row forward. Require bitwise raw-logit
parity and completion-loss agreement within the already fixed 1e-4 tolerance.
The first actual backend loss must also agree with that independent loss.

Save and publish completed mechanical boundaries 0 and 1 using the lower-level
backend API. Boundary 1 is not an E1 production checkpoint; engine V2's production
checkpoint schedule stays 0/64/128/192/256. Capture independent GPU copies of
state-1 adapter parameters and Adam state plus Python/Torch CPU/CUDA RNG BEFORE
save/restore. Restore boundary 1 BEFORE accumulating any sequence of step 2.
Compare restored parameters, moments, scalar steps and RNG bitwise against that
independent snapshot. Restoration must not roll back later work or erase costs.

At state 1, independently compute step-2 gradients with 20 autograd.grad calls,
each on loss/20, accumulating in the same parameter and sequence order. Require
live parameters, optimizer state, .grad and RNG to remain unchanged. Compare these
gradients to the native 20-backward accumulation at prospective atol 1e-6 and
rtol 1e-5. These tolerances are frozen before any physical result, not adapted to
failures. Verify completion CE against selected-log-softmax NLL at 1e-4.

For an independent check of clipping and Adam restoration, clone only state-1
adapter parameters and optimizer state, assign copies of the ACTUAL native
pre-clip gradients, and apply the same clipping and step-2 AdamW choices. Then
clip/update the live backend directly and compare resulting parameters, moments,
steps and RNG bitwise. This isolates optimizer continuation; it does not claim
bitwise equality of independently accumulated gradients or cross-process resume.

Journal separately: native backend methods, actual backend Torch primitives,
loader primitives, parity forwards and reference forward/gradient/clip/optimizer
operations. Preserve attempted, returned, failed and pending counts on failure,
including an original constructor exception's detached primitive snapshot.
The 40 native rows consume 5,248 input positions and 463 supervised positions;
20 reference forwards add 2,876 and 251; two parity forwards add 202 and 22.
Maximum total: 62 model forwards, 8,326 input and 736 supervised forward positions.
There are two native optimizer steps and one separately accounted reference step.
Two outer training reset calls plus the reset performed inside restoration mean
three actual backend gradient-reset primitives. Count both layers explicitly.

Keep the existing limits: worker RSS <=2 GiB, available-system reserve >=1.25 GiB,
GPU free >=4 GiB at admission and >=0.5 GiB during work, disk free >=4 GiB,
600 seconds total and the existing 100 ms external Windows process watchdog.
Pre-worker admission requires two consecutive observations of at least
3.3125 GiB available RAM (reserve, full worker budget and 64 MiB supervisor).
After imports, reuse the registered 60-second resource-only wait with two
consecutive dynamic-budget observations and a fresh pre-load check. A timeout
or floor violation performs no model retry. Do not kill user applications.

Require clean committed source/runtime/config bindings, exact model assets,
unchanged frozen-base and asset hashes after the run, finite confined gradients,
durable owned adapter-only artifacts and complete dose/cost evidence. CPU control
tests validate orchestration only. Physical qualification remains unproven until
the contained GPU run passes. Capture, language, long-context and sustained-E1
prerequisites remain open; no numerical tolerance is loosened and no current-best
recipe is promoted by this amendment.
