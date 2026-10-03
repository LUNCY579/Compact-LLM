# SYNTH-NQ-001: bounded current-runtime mechanics

This scope prepares the owner's requested M0/M1/M2 comparison. It is a separate
current-runtime qualifier, not an authorization for the central E1 program,
protected banks, production adaptation or a model-quality claim.

The original numerical failure and all historical scopes remain intact. The
candidate keeps exact BF16 Titan parameter objects and bytes resident on GPU,
while computing embedding outputs, all 197 linear sites, hidden activations,
norms, rotary values, attention and MLP operations in FP32. Rank-8 q/v adapters
and Adam states remain FP32. This tests a specific upstream-rounding hypothesis;
neither CPU toy tests nor an FP32 output head establish its native success.

## Frozen qualification contract

`native_contract.write_scope` produces a new `configs/*.json` after source review.
It binds current Python/Torch/Transformers/safetensors/psutil versions, relevant
Python and native-library files, all production source, exact published assets,
the immutable source-policy 2 bundle and the first 40 M1 TRAIN occurrences.
Execution requires a clean committed checkout. Offline mode, disabled Hub
kernels, eager attention, no autocast/TF32 and deterministic CUDA settings are
explicit. Interpreted modern RoPE/layer configuration and nonpersistent buffers
must match the source contract, rather than being inferred from file hashes.

The frozen sequence dose is 8,928 native input positions and 413 supervised
positions over two full groups of 20. Reference and capture work is separately
counted, with a total cap of 200 forward calls / 102,400 input positions. A root
model hook counts actual attempted forwards, including parity/capture/reference
calls. Event accounting covers named API boundaries; internal ATen operators
are not claimed to be individually metered.

Checks include streamed exact loading, 310 frozen base tensors and the tied
embedding/head, 112 confined FP32 adapter factors, zero-adapter output equality,
all-FP32 activation observations, positional completion+EOS loss, independent
selected-log-softmax and accumulated gradients, two native updates, one separately
counted copied-gradient reference Adam update, and state-1 restoration against
an earlier detached parameter/moment/Python/CPU/CUDA0 RNG snapshot. Restoration
precedes group 2. Checkpoints contain only bounded adapter/optimizer/RNG state
and are loaded from the same hash-verified owned bytes.

Greedy capture uses a TRAIN prompt with only model-generated continuation tokens.
Two repeats and a bulk comparison must agree in argmax and satisfy unchanged
maximum/mean selected-logprob limits **0.03125 / 0.01**. Completion loss retains
2e-6 error, gradients retain atol1e-6/rtol1e-5, and restoration is bitwise. Any
failed check stops the scope; no tolerance widening or unchanged automatic retry.

## Containment and evidence

Admission requires two distinct current observations before model work:
3.3125 GiB host available, 4 GiB GPU free and 4 GiB disk free. Worker process-tree
RSS stays at most 2 GiB, host reserve at least 1.25 GiB and GPU free at least
0.5 GiB. The whole scope is bounded by 600 seconds.

The worker starts suspended and hidden, is assigned to a Windows Job with
kill-on-close and no breakaway, and receives ownership/heartbeat files before
resuming. External RSS/system monitoring samples every 100 ms independently
of GPU CLI sampling; CPU sample freshness is at most 200 ms. GPU probes run
every 500 ms with a 2-second timeout/freshness bound. The worker additionally
checks current CUDA memory at operation boundaries. Dead PID/create-time and
actual process handles govern lease disposition; a lock file alone is not a
live process. No unrelated application is terminated.

Outputs live in a new direct `artifacts` child. Existing receipts and failed
work are retained. A successful result proves only this named short mechanics
scope. Fresh unchanged-base enrollment, study scoring, language/long-context
preservation, sustained training admission and the actual three-arm run remain
separate gates. Protected CHECK/FINAL remain closed throughout.
