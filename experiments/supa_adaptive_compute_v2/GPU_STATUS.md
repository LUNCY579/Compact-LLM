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

## Native architecture execution and resource-aware resumption

GPU A0 architecture shards 0 and 1 completed, preserving 9 examples. Admission
for shard 2 then failed at 0.13 GiB available host memory versus the GPU
profile's 0.50 GiB staging floor. A separate Python workload was concurrently
using substantial host/GPU memory. The failed shard evaluated zero examples;
the two completed shards remain valid and are not rerun. No candidate comparison
or efficacy conclusion is available from this partial baseline.

Track B CPU capture `alignment-cpu-v2-001` also failed admission, at 1.52 GiB
available host memory versus the unchanged CPU profile's 3.00 GiB requirement.
Its source registration is
`f8e4a26742effa443dfb843a00b61f34df09394db217cd9bb72120b1de7255ea`.

`scripts/run_native_when_available.ps1` is an operational launcher around the
unchanged registered Python campaign. It waits for 2.5 GiB host RAM, 3.5 GiB
free VRAM and a released programme lease, resumes only missing shards, and
retries only zero-example RAM/VRAM admission failures. Other failures stop for
review. Each queue has its own hashed-script manifest and log directory; no
scientific source, setting, candidate, input or scorer changes. PowerShell
parsing and a read-only resource inspection passed. The scientific registration
was verified unchanged after adding the launcher.

```powershell
& experiments/supa_adaptive_compute_v2/scripts/run_native_when_available.ps1 -Role architecture_dev -MaxQueueSeconds 7200
```

## Original native control rejected by the architecture guard

A0, A1, A3 and A16 each completed all 128 architecture-development cases on
the qualified GPU runtime. Original AM reached shard 20 and failed the registered
malformed-generation guard. The operational queue stopped for review; it did
not retry this as resource admission. Exact receipt:
`native_armr/evidence/01_architecture_dev/20260905T220307_858493Z_7cc95fc0/run_manifest.json`.
The evaluator's failure receipt does not retain the failed output prefix, so
the particular repetition/script/Unicode subtype is not inferred.

No confirmation examples have run. The original five-arm matrix is incomplete,
and no efficacy comparison or GO claim is licensed. Failed and interrupted
attempts, including the interrupted A16 shard 16 and its separate successful
rerun, remain immutable.

A prospective Native v2.3 engineering amendment is being implemented. It will
set the frozen seeded AM scalar by one fixed ratio of A3/control residual
energies on 96 calibration fit scenarios, with 32 held-out calibration scenarios
used only for reporting. The fitting input is prompt activations without answer
labels. No scalar search, accuracy tuning, guard relaxation or control removal
is permitted. Calibration source and formula must be frozen and committed before
execution, and successful calibration evidence must be committed before a new
GPU config is derived. The new profile requires fresh CPU/GPU qualification
and a fresh complete architecture matrix before untouched confirmation opens.

Track B CPU capture `alignment-cpu-v2-002` completed under its separate unchanged
CPU registration in 491.56 seconds, peak RSS 4,450,623,488 bytes. The canonical
fixture identity passed exactly, with 96 fit and 32 held-out calibration prompts.
The metadata verifier confirmed the completed receipts and activation SHA256
`8d132b3992a8b7599926d692e1ea776772b6fa5186e5a8a57ac3f3382dc5f872`.
Its sources and scientific settings are independent of the Native-only amendment.
The first of 16 pairs, S8/D12, subsequently completed all six bridge objectives
in 128.65 seconds. B3 held-out ordinary delta cosine was 0.79367 versus B0=0.32144,
with token-relative MSE 0.37049 versus 1.39826. These are partial-matrix descriptive
results, not a candidate selection or mechanism GO. Actual model jobs remain sequential.

The resource launcher now accepts `-ConfigName protocol_gpu_v2_3.json` explicitly;
its default retains the original v2.2 path. It logs the selected path and hashes
and still retries only zero-example memory admission failures. PowerShell parsing
and the read-only resource inspection passed. This operational script is excluded
from the Python scientific source binding and does not change either track's
registered scientific settings.

## Calibrated native v2.3 profile frozen

The committed label-free calibration completed in 62.75 seconds with exact
alpha `1.091369554461779`, without search or clipping. It slightly increases the
original control strength. Held-out relative energy is 0.0960649343 for A3 and
0.0957779170 for the scaled control. All state/buffer/checkpoint identities were
preserved; no weights were trained, answers scored or tokens generated.
Report: `native_armr/evidence/00_control_calibration/20260905T223621_219545Z_a6c46c00/report.json`.
The independent standard-library audit passed 3,012 consistency checks and
reproduced the scalar exactly from recorded scenario measurements; raw tensors
are hash-only, so this was not independent model execution.

Native GPU v2.3 registration is
`86e768e08bf3576dd8ce80ec939b7e876fb5fb7449d41a3aa51c497ce6607b44`.
It binds the committed calibration artifact and unchanged qualification/efficacy
gates. Fresh CPU/GPU qualification is pending resource admission. Operationally,
the v2.3 architecture order is A0, AM, A1, A3, A16; confirmation retains its original
arm order. Use the registered control-first order explicitly:

```powershell
& experiments/supa_adaptive_compute_v2/scripts/run_native_when_available.ps1 -ConfigName protocol_gpu_v2_3.json -ArmOrder control-first -Role architecture_dev
```

The above command is eligible only after the new qualification passes and its
evidence is committed. The one-stage resource launcher prefers that pending
qualification when host RAM and VRAM permit, or advances the unchanged Track B
CPU matrix when only host capacity is available. It never retries failed science
or overlaps programme model workers. Its first queue completed S8/D12. The second
was closed at 23:12 UTC after about 19 minutes without sufficient capacity and
without launching a model. The actual CPU caller admission probe rejected
2.39 GiB available host RAM versus 3.00 GiB required; free VRAM was about 1.84 GiB
versus the native profile's 3.00 GiB floor. No programme model/controller remains
active. Pending stages resume with the same frozen scientific inputs:

```powershell
& experiments/supa_adaptive_compute_v2/scripts/run_next_stage_when_available.ps1 -MaxQueueSeconds 3600
```

The full goal remains open. Capacity limits do not establish a negative efficacy
or donor-mechanism result.

## Native v2.3 GPU qualification passed

Successful receipt:
`native_armr/evidence/00_runtime_qualification/20260905T235200_982255Z_36998242/`.
All 40 comparisons passed, with exact greedy identities, GPU repeats and
disabled/restored behavior. The independent admission validator recomputed the
gates and checked all 80 logit hashes. Maximum absolute logit error was
4.1961669921875e-5, hidden relative L2 7.856709228698431e-5, and absolute NLL error
4.553794860839844e-5. Peak CUDA reservation was 2,688,548,864 bytes; wall time
313.05 seconds. Qualification/report commitment:
`39122c92c9fa930ebab99a90135ae92aa06c4f7b`.

Failed predecessors stay intact. `20260905T233146_952002Z_9d317bbe` failed RAM
admission before loading (2.94 versus 3.00 GiB). `20260905T233921_233855Z_41832612`
completed 40 CPU fixtures then lost its CUDA context before GPU loading. Windows
logged an NVIDIA TDR reset during its CPU phase; a fresh-process CUDA probe
subsequently passed. Those observations support retrying the unchanged profile,
but do not establish which process initiated the reset. No threshold was relaxed.

The fresh architecture campaign is running in the registered control-first
order. Queue: `native_armr/evidence/00_execution_queue/20260906T000050_6403813Z_434bb308/`.
The complete five-arm architecture gate still precedes confirmation.
Track B has two completed CPU pairs (S8/D12 and S8/D16) and 14 pending, with no
selected bridge. Its prepared future export/graft interfaces passed 56 tests and
6 subtests, but no generation protocol or real export is licensed.
