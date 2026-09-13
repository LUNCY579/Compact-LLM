# Runtime execution status

Snapshot: **2026-09-13 17:21 UTC**. GPU use is authorized. Scientific evidence stays specific to its track, protocol and qualified runtime; heavy model jobs run serially. Capacity failures are preserved and are not efficacy results.

Reference environment: `C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe`; torch 2.12.1+cu126, transformers 5.14.1. CPU and qualified CUDA:0 use FP32/eager, four threads, greedy full-prefix generation and no cache. RTX 4070 Laptop GPU execution disables TF32 and uses deterministic CUDA, highest matmul precision and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

| Runtime/stage | Evidence | Current permission |
|---|---|---|
| Native v2.4 CUDA | 40/40 CPU/GPU comparisons PASS, exact greedy/repeat/disabled/restored identities; qualification at `native_armr/evidence/00_runtime_qualification/20260906T203929_861509Z_7b0b0fc8/` | Frozen v2.4 architecture screen may resume |
| B Stage 1 GPU alignment | FAILED: 161/656 absolute comparisons exceeded .002; thresholds unchanged | Not admitted; no GPU/CPU alignment evidence mixing |
| B Stage 1 CPU reference | Complete 16-pair/96-cell matrix and independent 4,931-check audit | Mechanism selection complete |
| B Stage 2 CPU/CUDA generation | Calibration COMPLETE, verified/committed and independently audited; CUDA configuration derived; no real per-cell qualification yet | Per-cell CPU/GPU qualification required first |

Native v2.4 qualification maximum absolute first-logit error was **7.247925e-5**, hidden relative L2 **7.856709e-5**, and absolute NLL error **4.553795e-5**. Peak CUDA reservation was **2,663,383,040 bytes**, wall time **313.01 s**. These are compatibility measurements, not a controlled GPU speed benchmark.

The independent 978-check audit at 03:21:25 UTC found v2.4 architecture receipts containing **A0 128 complete / AM 16 complete / A1 0 / A3 0 / A16 0**; the same 37 attempt receipts were present at 17:13 UTC. AM has two zero-example RAM-admission failures; the latest (03:05 UTC) measured 0.31 GiB available versus the frozen 0.50 GiB staging floor. The old queue's session and PID 230268 are absent; an additional `interruption_observed_20260913T1713.json` records that without inventing its stop time or model integrity. A new controller started at 17:13:49 UTC. A fresh 03:22:45 UTC byte audit verified 198 historical files and the canonical checkpoint. The v2.4 model profile requires 3.0 GiB free VRAM, with maximum 6.5 GiB CUDA reservation and 8 GiB RSS; operational launchers may wait for more headroom.

Current queue evidence: `C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/native_armr/evidence/00_execution_queue/20260913T171349_1363153Z_2732bedc/`. A queue/lock file alone does not prove a worker is live. Interrupted runs need verified PID absence and explicit abandonment; complete shards are never overwritten. Do not start another programme model while its lease is held, and do not stop unrelated workloads.

B Stage 2 remains separate from failed Stage 1 GPU qualification. Its prospective CPU mechanics and CPU/GPU per-cell comparisons must pass before that cell generates architecture outputs. X0 must pass and complete first. A failed candidate cell is retained and its dose becomes ineligible, while other fixed cells can continue. The artifact-derived CUDA configuration is `cross_scale/generation_stage/v1/configs/generation_screen_gpu_v1.json`; its registration `dcfc67e0828a3b47da41980a88ce47813b4f8b63c184502f6898bbeaf7222fd3` is committed at `513d194`.

[Full current report and resume commands](C:/Users/clclc/BusinessProject/ChatBubble-cross-scale/experiments/supa_adaptive_compute_v2/CURRENT_REPORT.md). Confirmation, sealed evaluation and downstream mechanisms remain closed.

B-only XM calibration registration `bf99e713104573b7aafb8f5511278fd473bcd406ff1d6fece3d0ebe5787534a0` is committed and preflight-verified. Its earlier single-attempt queue at `cross_scale/generation_stage/v1/evidence/00_execution_queue/20260913T032108_4919029Z_calibration/` ended **QUEUE_BUDGET_EXHAUSTED at 03:51:35 UTC** without a model launch. The actual calibration at `cross_scale/generation_stage/v1/evidence/00_control_calibration/20260913T171252_566611Z_669e5d42/` is **COMPLETE**: 128 scenarios, alpha **1.6186783995925944**, **63.72 s** and peak RSS **2,019,373,056 bytes**. This CPU reference step computes the B-only control scale from cached activations and frozen operators; the report records unchanged checkpoints/states and zero gradients. Official verification passed, evidence was committed at `2ae253f`, and an independent audit passed **3,373 checks with 310 input hashes**. No Stage 2 efficacy generation has begun.

At 17:21:16 UTC, terminal Native manifests contain A0 128/128 and AM 51/128, with A1/A3/A16 pending and the same two pre-load RAM failures. The controller remains active. The B X0 CPU qualification queue in `cross_scale/generation_stage/v1/evidence/00_execution_queue/20260913T1720_X0_cpu/` waits for that controller to exit and for 4.25 GiB host headroom; the real caller retains its unchanged admission guards. No B qualification phase has launched.
