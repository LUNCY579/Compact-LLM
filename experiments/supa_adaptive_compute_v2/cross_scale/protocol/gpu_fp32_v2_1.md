# Track B GPU FP32 execution amendment v2.1

The user authorized local GPU acceleration on 2026-09-05. The original
`transformation_alignment_v2.md`, CPU configuration and all historical evidence
remain immutable. This prospective amendment applies only to
`configs/alignment_gpu_v2_1.json`. It changes the execution device and resource
admission, preserving the original scientific question, data/scorer, 96-cell
matrix, objectives, seeds, 96 optimizer steps, full prompts, controls and gates.
No confirmation results have been inspected to choose these runtime criteria.

Execution uses the existing torch 2.12.1+cu126 and transformers 5.14.1 runtime,
CUDA device 0, FP32, eager attention, four CPU threads, deterministic algorithms,
TF32 disabled, highest FP32 matmul precision, CUBLAS_WORKSPACE_CONFIG=:4096:8,
and no KV cache. FP16/BF16, autocast, quantization and backbone updates are absent.
GPU registration is separate from the CPU registration; evidence cannot mix.

CUDA admission requires 0.5 GiB free physical RAM and 3 GiB free VRAM. Process
RSS remains bounded by 6 GiB and reserved CUDA memory by 6.5 GiB. These limits
reflect CPU staging and GPU computation, not a reinterpretation of CPU admission.
Supa may reside wholly in FP32 on GPU for its canonical capture fixture. The
4B model never resides wholly on GPU: the existing audited checkpoint reader
materializes only one actual Hugging Face decoder at a time. Residual states
and saved activation chunks remain detached CPU FP32; one complete prompt at
a time is transferred to the decoder's GPU device. CPU FP64 ridge solves retain
their original normal equations, weighting and random/permutation construction.
Actual donor forwards and B2/B3 differentiation and Adam updates run in GPU
FP32; the downward map is fixed, moved to GPU for forward metrics, and saved
with all bridge artifacts back on CPU. Diagnostics accumulate in CPU FP64.

Before a real GPU alignment capture or pair fit, a committed successful
`cross_gpu_fp32_v1` runtime receipt is mandatory. It binds this exact registered
configuration, source map, checkpoint bytes, Track B dataset manifest, library
versions, GPU hardware and math settings. The qualifier consumes only Track B
prompt-only calibration metadata: the first sorted example from each of the
first eight sorted families, alternating fit and holdout partitions. There are
four fit and four held-out distinct scenarios. Tokens are never truncated and
must match across the two tokenizers. No answer, efficacy score or generation
enters qualification.

The qualifier compares CPU versus GPU streamed Supa raw post-layer states at
8/12/16/19 and donor raw post-layer states at 11/12/15/16/19/20/23/24. It also
fits CPU B0 ridge bridges on the four fit prompts for all 16 registered pairs
and compares actual projected donor deltas on all four held-out prompts. Every
comparison reports all, ordinary, special/control and final-prompt tokens;
empty special strata are explicitly empty. The same GPU donor forward must
repeat bit-identically. Complete GPU streamed captures of both models are also
repeated on all eight fixtures and must match bit-identically at every site.
At Supa L8 and every donor depth, one full fit prompt
also tests CPU/GPU B2 and B3 gradients through the actual frozen decoder into
the upward weight and bias; these are gradient checks, not training results.
Parameter gradients are compared as whole vectors. Backbone and rotary state
hashes must remain unchanged and backbone gradients must remain absent.
Every GPU bridge-gradient check repeats the backward and must match both
parameter gradients and scalar loss exactly. A separate report validator
reconstructs the expected 656 comparison rows from the frozen fixture rule,
checks their complete token strata and dimensions, recomputes every numerical
threshold, and verifies checkpoint and backbone-integrity evidence before
the caller accepts a runtime receipt.

Every nonempty state, delta and bridge-gradient comparison must have relative
L2 error <=0.0002, mean row cosine >=0.999999, and maximum absolute component
error <=0.002. Thresholds are frozen before running the qualifier; a failure
is preserved and does not license widening the thresholds in this version.
The separately executed full capture retains the original exact equality of
same-GPU canonical Supa hooks and streamed Supa states at all capture sites.
Pair fitting retains exact native replay on the same GPU execution device.

A passing runtime receipt licenses **Stage 1 only**. It establishes numerical
compatibility of the GPU implementation, not transformation quality, model
capability, selected depths, future gates or cross-track runtime equivalence.
The original mechanism gate still requires all 16 pairs and all six objectives.
Stages 2/3 generation, 8B, routing, distillation and all other closed mechanisms
remain governed by the original scientific gates.

Run qualification with the historical Python environment after freezing and
committing the GPU registration:

```powershell
& C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe experiments/supa_adaptive_compute_v2/cross_scale/scripts/qualify_gpu.py --config experiments/supa_adaptive_compute_v2/cross_scale/configs/alignment_gpu_v2_1.json --dataset-manifest <registered-cross-scale-manifest>
```

Commit the resulting immutable receipt/report before invoking `run_alignment.py`
with the GPU configuration and a new run ID. Failed CPU or GPU evidence is
preserved; no failed attempt is retried in its existing directory.
