# GPU execution continuation

The user authorized GPU acceleration on 2026-09-05. This continuation uses the
local NVIDIA GeForce RTX 4070 Laptop GPU (8 GiB VRAM), driver 610.62, with the
qualified Python environment. FP32/eager/no-cache, deterministic algorithms,
highest float32 matmul precision and disabled TF32 are fixed prospectively.

CPU protocols and their failed admission receipts remain unchanged. The new
GPU configurations retain the same data, candidates, bridge objectives and
scientific gates. CPU and GPU science shards cannot be aggregated together.
The shared runtime now places models/tokens explicitly on CUDA and tracks VRAM
separately from host RAM. State hashes are byte-compatible streamed hashes.

| Track | GPU registration | Runtime qualification | Scientific execution |
|---|---|---|---|
| Native | `7c99dac8b3e72ad44ec2c1eeaaa67ac897fabe6d3209d362c1f78e01c3a7cca5` | PASS: 40/40 identical greedy comparisons; fixed numerical gates passed | Architecture campaign licensed |
| Cross-scale | `beec1869aad324d29191cd21e5f300bd6da3bdc0a52d8c5f11839ac4b98f5b7c` | FAILED frozen absolute-error criterion; relative/cosine/gradient gates passed | Use unchanged CPU reference for Stage 1 |

Tests: 321 passed and 148 subtests passed; after recording driver identity,
54 focused tests passed again and the actual CUDA runtime identity query passed.
Compilation passed. The 198 historical files, original CPU configs/protocols,
frozen bank JSONL, semantic generator and scorer remain unchanged.
Receipt: `00_audit/gpu_validation_v1/validation_receipt.json`.

Run from the repository root with the existing Python executable:

```powershell
$researchPython = 'C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe'
$programme = 'experiments/supa_adaptive_compute_v2'
& $researchPython "$programme/scripts/qualify_native_gpu.py" --config "$programme/native_armr/configs/protocol_gpu_v2_2.json" --manifest "$programme/data/generated/v2/native_armr/manifest.json"
& $researchPython "$programme/cross_scale/scripts/qualify_gpu.py" --config "$programme/cross_scale/configs/alignment_gpu_v2_1.json" --dataset-manifest "$programme/data/generated/v2/cross_scale/manifest.json"
```

Each qualifier has its own immutable evidence directory and must pass before
that track's science runs. Commit passing receipts/reports before admission.
Native compares all five arms on eight own-track calibration examples, plus
same-GPU repeat and disabled/restored identity. Track B validates 656 numerical
comparisons covering streamed captures, actual donor deltas and bridge gradients,
with exact same-GPU repeats. These are engineering measurements, not efficacy.

Once qualified, native campaign commands use
`--config native_armr/configs/protocol_gpu_v2_2.json` (under the programme root)
and the unchanged 32-shard schedule. Cross-scale commands use
`--config cross_scale/configs/alignment_gpu_v2_1.json` and a fresh run ID.
All full-model/GPU jobs remain sequential under the shared machine lease.

## First real qualification attempt

Native qualification directory:
`native_armr/evidence/00_runtime_qualification/20260905T185820_846212Z_273b3b24/`.
It completed the 40 CPU case/arm reference logit artifacts and 23 GPU artifacts.
A separate long-context research process then occupied 6.527 GiB dedicated VRAM;
Windows counters showed our model displaced into shared memory and available
host RAM falling to 0.009 GiB. The owned qualification process was stopped, its
partial tensors and immutable interruption receipt preserved, and its lease
released after verifying process exit. No other process was stopped.

Canonical checkpoint files still match the qualified reference. In-memory
post-stop state could not be measured; qualification is **not passed**. This
resource interruption does not change any numerical threshold or scientific
candidate. The competing process subsequently exited, restoring about 7.1 GiB
free VRAM and host RAM. Resume in a fresh qualification directory using the
unchanged committed registration; never reuse this failed directory.

## Native qualification passed on retry

Receipt directory:
`native_armr/evidence/00_runtime_qualification/20260905T191958_868458Z_82f1162e/`.
The independent report validator checked all 40 case/arm comparisons, their
frozen calibration identities, model/control hashes and every saved logit hash.
All greedy sequences matched CPU; same-GPU repeated outputs, NLL and captured
hidden tensors were identical, and disabled/restored behavior was exact.

- Maximum absolute first-token logit error: `4.1961669921875e-05`.
- Maximum per-token hidden relative L2 error: `7.856709228698431e-05`.
- Maximum absolute answer-NLL error: `4.553794860839844e-05`.
- Peak CUDA reservation: `2.50390625` GiB; total qualifier wall time 398.30 s.
- CPU/GPU canonical state and seeded AM state hashes matched exactly.

The qualification establishes native GPU runtime compatibility, not efficacy.
It licenses the unchanged native architecture-development matrix. Observed
qualification latency was affected by a separate GPU workload; it is not a
controlled accelerator speed benchmark. Track B still needs its own qualification.

## Track B GPU qualification outcome

The completed run is
`cross_scale/evidence/00_runtime_qualification/20260905T193023_051203Z_29be300a/`.
All 656 comparisons were recorded. Of these, 161 failed the frozen absolute
component error cap of 0.002: 116 state comparisons and 45 projected-delta
comparisons. No comparison failed relative L2 or cosine; all 16 CPU/GPU bridge
gradient comparisons passed. All required same-GPU repeats were bit-identical.
Maximum absolute difference was 0.021484375; maximum relative L2 difference was
0.0001805015 and minimum mean-token cosine was 0.9999999764. The failed report
and thresholds are preserved. This is a runtime-qualification failure, not a
result about useful donor transfer.

Track B will therefore execute its original, already qualified CPU FP32
reference. The original CPU configuration, bridge objectives, 16 depth pairs,
data, optimization budget and mechanism gates are unchanged. A fresh CPU
registration binds the current implementation, whose CPU path passed the
combined tests; it does not admit the failed GPU profile or mix runtimes in
the alignment matrix. Track A retains its passed native GPU qualification.
This resource/backend decision precedes any new mechanism selection or efficacy
outcomes. Scientific thresholds are not widened after observing the GPU report.
