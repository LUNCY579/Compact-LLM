# CLARE-RINEYA-001 — native smoke procedure and exact-host handoff

A failed preflight is a **BLOCKED** status, never a research result. Nothing below loosens a
threshold: the floors are the vendored `hostbudget` gate (3.6 GiB free host RAM to start a GPU
training job, 3.0 GiB for probes, guard 1.5 GiB between updates), 3.5 GiB free VRAM, a 4 GiB disk
floor after the registered artifact estimate, and the shared GPU lease.

## 1. Preconditions (check, do not assume)

| resource | how it is measured | need |
|---|---|---|
| host RAM | `GlobalMemoryStatusEx` via `vendor/hostbudget.py` | ≥ 3.6 GiB free *after* the job's own Python/torch import |
| GPU present | `Win32_VideoController` lists "NVIDIA GeForce RTX 4070 Laptop GPU" | present (the dGPU was disabled once during this session) |
| VRAM | `nvidia-smi --query-gpu=memory.free` | ≥ 5.5 GiB for qualification/training, ≥ 3.5 GiB for evaluation |
| GPU lease | `C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock` absent | acquired by the job itself |
| disk (C:) | `shutil.disk_usage` | artifacts + 4 GiB (see §4) |
| interpreter | hermes venv | `C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe` (3.11.15, torch 2.6.0+cu124, transformers 5.14.1) |

Do not diagnose low free RAM from idle cores, and do not attribute it to a GPU job without
checking `\GPU Adapter Memory(*)\Shared Usage`. Never bypass `hostbudget`.

## 2. Offline checks (Phase 1–2; CPU only, no model)

```powershell
cd C:\Users\clclc\BusinessProject\ChatBubble-titan-clare-sparse-v1\experiments\clare_sparse_composition_v1
& C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests
& C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe -B cli.py plan
```

## 3. Native smoke = Phase 3 qualification (one command, ~10–15 min, one lease)

```powershell
& C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe -B -u cli.py native-qualify --out qualification\native-<YYYYMMDD-HHMM>
```

What it does, in order (all recorded in `NATIVE_QUALIFICATION.json`):
1. preflight (host, GPU presence, VRAM, disk, lease); BLOCKED ⇒ exit 3 with the blockers;
2. loads Supa **straight to the GPU** (`device_map`, so host RAM never holds the model), hash-pinned;
3. installs the BF16-storage / FP32-compute patches and `DeltaHost`; verifies 596,049,920 parameters,
   60 footprint modules = 188,743,680 coordinates, and that no base parameter requires grad;
4. **parity gate**: last-position logits of the patched model vs a plain FP32 load on two TRAIN R
   prompts (≤ 1e-4, equal argmax), with both the chunked and the cached LM head;
5. two dense AdamW updates on TRAIN R rows (the arm-A / discovery memory peak), timed;
6. a discovery probe on a pretend-free set (90 % of the footprint): mask respected, exact reset,
   λ-gate reading, exact-budget selection time;
7. two sparse updates on the selected mask (the D/E Stage-2 memory peak);
8. base weights unchanged (checksum);
9. base competence smoke on 6 CALIBRATION rows per train signature (never DEV/TEST);
10. projected minutes per skill stage and per 960 evaluation rows.

Status `QUALIFIED` requires every gate except the λ reading, which is informational here (the
registered λ gate runs per skill inside arm E at the registered probe length).

## 4. Full run = Phase 4 (the chain; detached through WMI)

Only WMI-launched processes survive a session/app restart on this machine. `wmi_runner.py` passes
no arguments to its target, which is why the chain takes none.

```powershell
$d  = 'C:\Users\clclc\BusinessProject\ChatBubble-titan-clare-sparse-v1\experiments\clare_sparse_composition_v1'
$py = 'C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe'
$r  = 'C:\Users\clclc\BusinessProject\ChatBubble-titan-clare-sparse-v1\experiments\interference_mechanism_v1\wmi_runner.py'
$cl = "`"$py`" -u `"$r`" `"$d\run_phase4.py`" `"$d`" `"$d\runs\phase4.log`" `"$d\runs\PHASE4_RC.json`""
New-Item -ItemType Directory -Force "$d\runs" | Out-Null
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cl; CurrentDirectory=$d; ProcessStartupInformation=$si}
```

Wait on `runs\PHASE4_RC.json` (and `runs\clare-rineya-001\CHAIN_STATUS.json`), **not** on the
process, and do not arm a `tail -f` monitor on the log (monitors have killed jobs here). To resume
after a BLOCKED stop, a failure or a driver reset, relaunch the same command: `wmi_runner.py`
refuses if `phase4.log`/`PHASE4_RC.json` exist, so rename those two files first (never delete a
run directory). Completed skills, stages and scored rows are skipped.

Tiers (the chain stops at the first BLOCKED step, so the most important results land first):

GPU time below is DERIVED from the Phase 3 measurements (1.69 s per ≤ 2048-token update ≈ 14 min
per 1,000-row stage on the longest (R) prompts; 37–50 s selection) and from the native cache check
(1.355 s per generated row with the evaluation cache, ≈ 46 min per arm's 2,048 scheduled rows;
1.79 s uncached).

| tier | content | new artifacts on C: | GPU time (DERIVED, cached evaluation) |
|---|---|---|---|
| 0 | qualification + LR build gate (≤ 3 dense R stages + 150 CALIBRATION rows each) | ~0 | 0.3–1 h |
| 1 | seed 0, RUSP: A, D, E (train, evaluate, interact) + base | ~3.5 GiB (A 4 × 0.703 GiB dense; D, E ~0.33 GiB each) | ~6 h |
| 2 | seed 1, RUSP: A, D, E | ~3.5 GiB | ~5.5 h |
| 3 | seed 0, PSUR: D, E | ~0.7 GiB | ~3.5 h |
| 4 | seed 0, RUSP: B (3 new dense deltas), C | ~2.4 GiB | ~2.5 h |

**Disk is the binding constraint for the full run:** tiers 0–4 need ≈ 10.1 GiB of artifacts plus the
4 GiB floor ≈ **14 GiB free on C:**. At 8.5 GiB free (2026-09-27), tier 1 fits and tier 2 will
refuse at preflight. Freeing space is an operator decision (the soup lane's scored arms are the
documented reclaim pool); trained deltas are irreplaceable GPU output and must be backed up by
hash, never deleted to make room.

## 5. Secondary endpoint needing an operator decision

The n = 47 general-retention panel is scored by
`ChatBubble-ced-native-loader-qualification/experiments/rab001_exploration/retention_probe.py` in
the rab001 venv, and it writes `RETENTION_PROBE_<label>.json` **into that other session's
worktree**. It needs an FP32 HF arm directory with Supa's six sidecars byte-identical and a
`MATERIALIZATION.json` receipt (2.2 GiB per arm, built → scored → weights deleted with a tombstone).
Not run and not automated here until the operator approves writing there (or provides a clean
checkout with the identical `architecture_benchmark_001` tree, digest `a329eb7b`).

## 6. Status at hand-off time (2026-09-27)

- Phase 1 (implementation) and Phase 2 (synthetic qualification) complete: see
  `qualification/SYNTHETIC_QUALIFICATION.json`.
- Phase 3 native qualification: attempt 1 **BLOCKED at preflight on host RAM** (3.55 GiB free vs
  the 3.6 GiB floor; `qualification/native-20260927/`). Attempt 2 **QUALIFIED**, all 11 gates
  (`qualification/native-20260927-attempt2/`). **Re-qualified on the committed code** `6759d95b`
  (clean tree, code digest `a0306194`): QUALIFIED, 11/11, numerics bit-identical to attempt 2
  (`qualification/native-20260927-committed-6759d95b/`, the current reference). Earlier in the session the NVIDIA adapter was absent
  (services disabled 16:00:44–16:01:08Z) and host RAM was at 0.48 GiB.
- Evaluation cache native check **PASS** (`qualification/EVAL_CACHE_CHECK_2.json`): 14/14 recorded
  responses reproduced byte-for-byte with the caches on, cached == uncached outputs, 1.32× faster
  (1.786 → 1.355 s/row). The first attempt was BLOCKED on disk (`qualification/EVAL_CACHE_CHECK.json`).
- **Disk:** 5.42 GiB free on C: (it fell to 0.84 GiB during the session from writes outside this
  study). Arm A's training preflight needs ≈ 6.8 GiB; all of tier 1 ≈ 7.5 GiB.
- Phase 4: not started. No real training has occurred.
