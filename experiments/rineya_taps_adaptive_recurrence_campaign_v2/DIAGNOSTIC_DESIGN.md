# H3 canonical first-token sensitivity — prospective apparatus

This package contains no loader/model worker or native command. Its CPU fixtures
do not qualify Rineya. A separate coordinator-owned runner, source/runtime/data
registration, resource forecast and independent reviewer are required before any
native call. Original inference wrappers and their no-gradient guards are intact.

Prospective population: all original allowlisted DEV input prefixes<=256 tokens
under the already-bound native tokenizer. The two2237-token retrieval prompts are
excluded by this shape rule alone. The remaining14 short rows are the population,
not an outcome-selected subset. Apply the inherited canonical-answer convention
and use only its first native token; missing canonical target is NOT_APPLICABLE
with zero calls, not a guessed label. Record every exclusion/absence before native
measurement. This is a canonical next-token loss surrogate, not semantic correctness.

The ordinary L12-L13 block gives raw d1=h14-h12. One extra raw proposal gives
d2=F(h14)-h14. A fixed externally supplied alpha controls h14+alpha*d2. Alpha1
takes the proposal value directly. The controlled value is then detached and
cloned into a single gradient leaf. Only the actual remaining native layers,
final normalization and custom abstention-aware head propagate gradients from CE
to that leaf. Parameters require_gradFalse and gradNone throughout; no optimizer
or Rineya training exists. Target token is used only after forward logits, never
appended to input or supplied as conditioning.

At alpha1, compute G with autograd.grad(loss,state), and sensitivity< G,d2 >.
Central differences at1±.01 each recompute the ENTIRE model/operator/suffix on the
same prefix in a fresh wrapper. No saved-state counterfactual is used. One
diagnostic costs3 full forward calls plus1 suffix state-gradient evaluation;
the two finite-difference values do not redundantly evaluate gradients. All work,
failures, target skips, resources and parent wall time must be charged. No-extra
qualification records no invented second update/moment observation.

Device is explicit: CPU only for apparatus, cuda:0 only under future GPU
registration. Diagnostic-only guards require frozen eval, enabled grad mode
outside inference mode, FP32 parameters/states, batch1, no padding, no caches,
<=256 tokens, exact full-prefix positions, same replay mask/position objects,
L0-L2 one-shot, finite logits/loss/state gradients and exact method restoration.
Compiled/checkpointed hosts and overlapping recurrence wrappers are refused.
The wrapper changes neither inference guards nor source assets. Native source
and loaded-state/custom-head hashes, every buffer's device/dtype, cache/exception
restoration, unit/noextra value parity, finite values and maximum256 gradient
resource shape must qualify afresh. A forward-only2248 receipt does not qualify
this gradient tape.

CUDA calls additionally require an explicit resource/deadline callback and CUDA
synchronization callback. Frozen parameter and buffer device/dtype checks run
before each forward; the resource callback runs again after suffix/gradient work.
The fixed-alpha state uses a cloned, non-inference leaf. The moment summaries use
the closed two-observation formula in FP32; exact bitwise agreement with v1's
incremental bias-corrected feature arithmetic is not claimed and must be checked
or explicitly distinguished in a future feature-comparison registration.

Variance/progress features are offline two-observation FP32 proxies. Stable
movement toward a wrong fixed point can have positive balance and positive loss
sensitivity: local stability/persistence is not task progress. The analytic tests
include a nonlinear suffix and constant abstention offset; a linear internal
probe would not satisfy the contract.

Native outputs should bind prefix/target/scorer/source hashes, target convention,
surrogate loss, analytic/finite-difference sensitivities and error, d1/d2 norms,
two-observation features, preserved state/asset checks,3 call charges,1 backward
charge and resource samples. Hidden tensors need not be exported. These
label-dependent journals must remain separate from generation trajectories and
the DEV A5 marginal pool. Full-gold NLL remains disabled.

This small same-source DEV diagnostic is exploratory. Sign agreement and
feature association do not establish semantic answer gains, conditional theory
assumptions, out-of-family prediction or a deployable controller. Any subsequent
controller fit consumes development labels offline, requires prospective fitting
and selection rules and must never read diagnostic labels/gradients online.

## CPU apparatus qualification

Final exclusive run: runs/sensitivity-apparatus-002. **15 passed,0 failed,0 skipped**
in3.10 seconds using the existing read-only Python3.11.15 environment, -B and
pytest cache disabled. The earlier13-test run is preserved separately. Tests use
only small frozen CPU fixture weights, never canonical/native model assets.

The tests independently check the chain-rule result against a hand-derived
nonlinear suffix Jacobian with a constant abstention offset; whole-model central
differences against autograd; variance against centered double-precision weights;
unit/noextra values against ordinary/legacy fixtures; original method/hook/owner
restoration after a suffix exception or resource refusal; unchanged parameter
values and gradNone; missing-target zero work; cache, prefix length, padding,
positions, training and inference-mode refusals; and stable-but-wrong updates.
CUDA guard/synchronization refusal is tested without touching CUDA.

| Artifact | SHA256 |
|---|---|
| sensitivity.py |5d0af45a8e91f0a00d91f83244105d06f1beba6f422cc3cd6d20c2d43d762f0d|
| tests/test_sensitivity.py |e11159218a0d54fced011492bd6888c53091f06a8c96a91461c4aa345dc2e724|
| final junit.xml |42058ec16686063f8218b77873cb26887383ee259a9347e856badfc64a9c2834|
| final pytest.log |4ef6ecf29ccf08787ae7f37ea6aae0c29b381881a26a16c4db94eb902128ce39|

The author issues no independent-review or native PASS. The third reviewer must
inspect this apparatus/design before the coordinator registers a future native
qualification and diagnostic. The final CPU run checks apparatus behavior only.
