# Runtime execution status

Snapshot: **2026-09-13 17:42 UTC**. GPU use is authorized and active for Native v2.4. Heavy model jobs run sequentially under the shared lease; evidence remains specific to its track and registered runtime.

Reference environment: `C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe`, torch 2.12.1+cu126, transformers 5.14.1. CPU and qualified CUDA:0 use FP32/eager, four threads, greedy full-prefix generation and no cache. RTX 4070 Laptop GPU execution disables TF32 and uses deterministic CUDA, highest matmul precision and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

| Runtime/stage | Evidence | Admission |
|---|---|---|
| Native v2.4 CUDA | 40/40 CPU/GPU comparisons PASS, exact greedy/repeat/disabled/restored checks | Frozen five-arm architecture campaign active |
| B Stage 1 GPU alignment | FAILED: 161/656 absolute errors exceeded .002; thresholds unchanged | Not admitted; preserved separately from CPU results |
| B Stage 1 CPU | Complete 16-pair/96-cell matrix and independent audit PASS | Mechanism selection complete |
| B Stage 2 CPU/CUDA | B-only calibration, final CUDA config and registration committed/verified; no qualification phase yet | X0 CPU phase queued, then per-cell CPU/GPU comparisons required |

Native qualification maximum absolute logit error was 7.247925e-5, hidden relative L2 7.856709e-5 and absolute NLL error 4.553795e-5. Peak CUDA reservation: 2,663,383,040 bytes; wall time: 313.01 s. Evidence: `native_armr/evidence/00_runtime_qualification/20260906T203929_861509Z_7b0b0fc8/`. These are compatibility measurements, not a controlled accelerator speed benchmark.

A0 and pretrained AM have each completed 128 architecture cases. The full A0/AM control review passed 3,743 checks, including recorded frozen state, zero gradients, raw validity and registered RMS/cost bounds; it did not replay tensors. The new baseline also exactly matches the prior baseline in all 128 token/NLL/logit-file records. The controller is now executing A1, with A3/A16 still scheduled. Confirmation remains closed until the complete five-arm architecture audit passes.

The Native queue is `native_armr/evidence/00_execution_queue/20260913T171349_1363153Z_2732bedc/`, PID 3512. Three zero-example AM RAM-admission failures remain preserved, including .43 GiB versus the unchanged .50 GiB staging floor on shard22; all were followed by fresh attempts. The frozen GPU profile requires 3 GiB free VRAM and permits at most 6.5 GiB CUDA reservation and 8 GiB RSS. Available VRAM does not remove host-memory requirements.

B calibration completed 96 fit/32 holdout scenarios in 63.72 seconds, peak RSS 2,019,373,056 bytes. Alpha 1.6186783995925944 is committed and independently audited. The final CUDA screen registration is `dcfc67e0828a3b47da41980a88ce47813b4f8b63c184502f6898bbeaf7222fd3`; each cell needs its own CPU reference and CPU/GPU numerical/identity checks. X0 must qualify and finish its baseline before candidates generate architecture outputs.

The one-phase B queue, PID 110104, waits for the Native controller to exit, the shared lease to clear, and 4.25 GiB host headroom. Evidence: `cross_scale/generation_stage/v1/evidence/00_execution_queue/20260913T1720_X0_cpu/`. It runs only X0 CPU qualification and then returns for receipt review. It does not automatically qualify GPU, generate efficacy examples, modify settings or retry failures. Do not launch a duplicate model worker.

[Full report and resume commands](C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/CURRENT_REPORT.md). Sealed evaluation and downstream mechanisms remain closed; earlier failed runtime/control attempts are preserved in their original directories.
