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
| Native | `7c99dac8b3e72ad44ec2c1eeaaa67ac897fabe6d3209d362c1f78e01c3a7cca5` | Pending actual CPU/GPU calibration comparison | Pending qualification |
| Cross-scale | `beec1869aad324d29191cd21e5f300bd6da3bdc0a52d8c5f11839ac4b98f5b7c` | Pending streamed state/delta/gradient comparison | Pending qualification |

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
