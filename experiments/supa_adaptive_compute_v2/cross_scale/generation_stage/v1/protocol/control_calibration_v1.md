# Track B XM control and label-free scalar calibration v1

This prospective stage owns its control, seed, data, scalar and evidence. It
requires the committed complete Stage 1 matrix and independent audit admitting
the B3 post-Supa-L19 / Qwen3-4B-L24 winner. The unchanged exported XG/XS/XR
tensors and complete source/config/data bindings must be committed and verified
before calibration. This document does not itself license real execution.

XM is a standalone Supa-width residual transform, with independent CPU normal
draws, seed 2702, standard deviation .02. It applies unit RMS normalization with
epsilon 1e-6 followed by `Wdown(silu(Wgate h) * Wup h)`. Width 1024 and intermediate
width 34560 yield 106,168,320 frozen matrix coefficients and 212,336,640 added
multiply-plus-add FLOPs per token. This exactly matches the dominant matrix work
of the selected donor plus its two affine bridge matrices. Norms, attention,
instrumentation and answer NLL are excluded; report actual latency, active
parameters and peak memory separately. This is an untrained generic control;
beating it does not establish superiority over a learned same-scale operator.

At post-L19, the treatment is `h + g * alpha_B * M(h)`. The prospective gate grid
is .03, .10, .20, .40 for all residual arms. Mechanical zero/disabled conditions
bypass M and preserve exact output identity. XM does not repeat a decoder, alter
positions/masks or modify any backbone tensor. Its module stays wrapper-local.
The shared implementation of FrozenSwiGLU is engineering reuse only; no Native
calibration, fitted alpha, seed, operator factory, results or data are imported.

Use only Track B's already frozen 128 original calibration scenarios, comprising
96 fit and 32 held-out scenarios. Parse only identity and prompt fields from the
immutable calibration JSONL; discard answer/scorer/oracle/fact fields immediately.
No architecture-development or confirmation inputs/outputs are opened. Hash
verification covers the immutable JSONL container without consulting labels.

For each calibration prompt, reuse its qualified raw post-L19 Supa state from the
upstream SHA-256-verified activation cache. Deserialize with `torch.load` using
`weights_only=True`, `map_location='cpu'`, and `mmap=True`. Validate all and only
the used S19 slice, prompt IDs, original token sequence, lengths and fit/holdout
partition. Do not page in other Supa slices or donor-native activation slices;
never load the full Supa model. The registered tokenizer reconstructs the full
prompt, and its special plus added-vocabulary token union defines excluded IDs.
The complete prompt must be at most 256 tokens; truncation is forbidden.

The teacher is the **unscaled mapped XG delta**, not a native-donor state target:

`z = exported_XG.project(h19)`

`delta_T = (actual_frozen_L24(z) - z) @ exported_XG.down`

Use the unchanged historical bridge RMS normalization, upward bias and downward
matrix. Downward bias cancels from the residual. Replay the actual CPU FP32 eager
decoder on the full prompt with the original positions, causal mask and donor
RoPE. No tokenwise independent decoder replay, prefix truncation, gate scaling,
directional fit, weight training, donor-native labels or efficacy data is allowed.

For each fit scenario s, calculate in FP64 reductions:

`E_T[s] = mean_ordinary_token(mean_feature(delta_T**2) / max(mean_feature(h19**2), 1e-8))`

`E_M[s] = mean_ordinary_token(mean_feature(M(h19)**2) / max(mean_feature(h19**2), 1e-8))`

Then freeze `alpha_B = sqrt(mean_scenario(E_T) / mean_scenario(E_M))`. This is a
single fixed formula with equal scenario weights. No scalar search, clipping,
fallback, selection by holdout or fitting directions is permitted. Both fit
energies and alpha must be finite and positive; otherwise fail with a receipt.
Persist the exclusive `fit_scale.json` after 96 fit transformations and before
the first held-out teacher/control transformation. Only then calculate the 32
held-out scenarios. Holdout diagnostics describe energy transfer; they cannot
change alpha. Preserve every outcome, including a destructive generic control.

Report 128 immutable per-scenario receipts, each with all/ordinary/special/final
token energy and RMS metrics, token IDs, full and retained lengths, original
position/mask/RoPE hashes, immutable S19 input hash and teacher/control delta
hashes. Report donor output/input RMS, constrained to [.05,10], plus fit/holdout
scenario aggregates and scaled control energy. Hash donor, rotary nonpersistent
buffers, copied bridges and control tensors before/after; record zero trainable
parameters and gradients, bridge source/copy identity, canonical/donor checkpoint
and implementation hashes, export/cache/source/data/config/registration hashes,
runtime, branch/commit, start/end, wall time, peak RSS and any failure.

CPU FP32/eager/no-cache and four threads are the reference. Require 2 GiB available
physical memory before loading the mmap cache, one donor layer and the ~405 MiB
control. Enforce 6 GiB process RSS, 900 seconds per prompt and 172800 seconds for
the entire stage at operation boundaries. There is one programme model lease.
GPU generation needs a separately frozen CPU/GPU qualification; this CPU scalar
calibration alone is not a GPU qualification or a capability result.

Use new exclusive directories under stage-v1 `evidence/00_control_calibration`
and registrations under `evidence/00_control_calibration_registration`. Failed
or interrupted receipts are retained. The final generation config may derive
alpha and state identity only from a verified committed successful artifact.
Any changed control, dose grid, formula or input role requires a new prospective
stage version. Confirmation remains closed until the later Stage 2 mechanical
and dose-selection gates pass.
