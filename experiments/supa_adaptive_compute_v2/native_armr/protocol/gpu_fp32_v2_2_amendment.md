# Track A v2.2: prospective GPU FP32 runtime amendment

The user explicitly authorized GPU acceleration on 2026-09-05 after the two
CPU loading attempts stopped before evaluating any examples. This amendment
implements that authorization. It is effective only after source, configuration,
dataset manifest and this amendment are frozen and committed. The original
`preregistration.md`, `configs/protocol.json`, CPU registration and failure
receipts remain unchanged. No candidate or scientific decision is selected from
GPU outputs. The superseding execution configuration for GPU attempts is
`configs/protocol_gpu_v2_2.json`.

## Preserved science

All five candidates A0/A1/A3/A16/AM, recurrence counts and locations, random
control seed and coefficients, 1,024 primary scenarios, 128 diagnostics,
128 architecture-development scenarios, token caps, scorer, data boundaries,
statistical hierarchy, preservation/validity gates and downstream closed gates
are exactly those of v2.1. The 32 deterministic shards and complete architecture
matrix prerequisite remain in force. CPU and GPU evidence cannot be mixed into
one paired matrix: the full configuration hash, registration and source identity
must agree across A0 and every candidate. GPU science begins only after its own
CPU/reference runtime qualification has passed.

## GPU runtime and independent resource admission

Use the same frozen canonical checkpoint and float32 tensors on `cuda:0`, eager
attention, seed 17, four CPU threads, greedy generation and full-prefix
recomputation with no cache. Disable TF32 for CUDA matmul and cuDNN, keep float32
matmul precision at `highest`, and configure deterministic CUDA algorithms and
cuBLAS before model execution. Quantization, reduced precision, backbone
training and an attention-kernel substitution are not authorized by this
amendment. Model and extra-operator state hashes and all frozen/gradient
inventories are checked before and after execution.

AM is constructed from the same local **CPU float32** generator with seed 7301,
in the same gate/up/down draw order. Transfer its resulting tensors to CUDA
without redrawing or converting precision. The GPU control must have exactly
the CPU reference state hash and retain 31,457,280 frozen parameters. CUDA
initialization using a seed with the same integer would not prove tensor identity.

The GPU worker requires at least 0.5 GiB available host memory and, independently,
3.0 GiB available device memory before loading; allocated/reserved CUDA memory
must remain within the 6.5 GiB device limit. The existing 8 GiB process RSS bound
and 900-second per-example elapsed budget remain. These are resource controls,
not eligibility thresholds selected from accuracy. Use host-streamed loading and
bounded hashing so a CPU replica is not required to remain resident alongside
the GPU model. Record GPU model name, device capability, total/free VRAM, CUDA
runtime, TF32/precision/determinism settings and peak allocated/reserved VRAM in
addition to host peak RSS. Synchronize CUDA at timed boundaries so measured
latency includes completed device work. Report RSS and VRAM separately; summing
them does not create an interchangeable memory budget.

All heavy model workers continue to execute sequentially under the shared
machine lease. Resource failure produces a new immutable failed attempt; it
does not license changing candidates, questions, dtype or guards. A stopped or
failed attempt is never reused as the output directory of a retry.

## Prospective CPU/reference qualification

Qualification uses the Track A **calibration** role only. Select the first
example in immutable-ID order from each of the eight frozen families, then
compare every A0/A1/A3/A16/AM arm on these eight prompts (40 paired cells).
These calibration outputs neither enter confirmation nor select an arm or gate.
The runtime qualification ID is `native_gpu_fp32_v1` and its immutable receipt
must match the GPU registration, sources, configuration, frozen dataset manifest
and checkpoint identities. GPU science callers fail before model loading if no
unique compatible completed qualification exists. A failed qualification stays
failed and cannot be overridden through an execution command-line switch.

For **every** comparison, require first-token maximum absolute logit error
at most 0.002, relative logit L2 error at most 0.0002, logit cosine at least
0.999999, absolute answer-NLL error at most 0.002, raw-hidden relative L2 error
at most 0.0002 and hidden cosine at least 0.999999. Also require identical full
greedy output token IDs between CPU and GPU, and exact within-GPU repeated
output identity. Means across cells cannot conceal one failing cell. Both
runtimes must preserve ordinary positions/masks, disabled-operator identity,
restored-model identity, frozen weights and zero backbone/control gradients.
The shared qualifier binds the detailed input/output evidence and records each
comparison before issuing its gate. These tolerances are frozen before this
qualification runs; they cannot be loosened after seeing discrepant outputs.

The CPU reference keeps the same operators, float32/eager/no-cache computation,
seed, threads and full item caps. It is run sequentially and can release host
memory before CUDA execution. Its reduced host admission threshold is solely
part of this bounded engineering qualification: it does not amend the old
CPU campaign or authorize a broad CPU science run under GPU resource settings.

Only successful qualification licenses the unchanged five-arm architecture
matrix, then the unchanged primary/diagnostic campaign under this GPU
registration. Qualification and tiny CUDA unit tests establish implementation
agreement; neither establishes capability or replaces paired scientific results.
# Diagnostic details

Native hidden agreement uses the maximum relative L2 error and minimum cosine
across individual prompt-token vectors at sites 8,12,13,16,19. Logit agreement
uses the full first-token vocabulary vector. Comparisons are computed in CPU
FP64 after FP32 inference; no diagnostic cast changes model computation. Every
one of the 40 case/arm comparisons must pass, not only their average. The
admission validator independently reconstructs coverage and thresholds from the
frozen calibration manifest and report. State hashing streams fixed-size chunks
while preserving the historical byte digest. GPU loading uses an explicit single
device map, and CPU/meta/offloaded registered parameters or buffers are rejected.
