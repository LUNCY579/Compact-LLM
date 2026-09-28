# Runbook: the two exploratory GPU batteries, then CONFIRM-002 Stage A and Stage B

> **UPDATED 2026-09-25: the operator authorised the full chain ("chain Stage A and Stage B after the
> batteries").** Launch `experiments/composition_confirmation_v2/run_chain.py`, NOT `run_gpu_batteries.py`
> alone. Use the same WMI command below, with the target replaced by `…\composition_confirmation_v2\run_chain.py`,
> cwd `…\composition_confirmation_v2`, log `run_chain.log` and rc file `CHAIN_RC.json`. The chain runs the
> batteries, then Stage A. It runs Stage B only if Stage A exited 0, and Stage B's own preflight also requires
> Stage A's canary PASS and no pending extension. Progress is in `CHAIN_STATUS.json`. The whole window is
> about 3.5 h.

**Authorised by the operator on 2026-09-24:** "run the batteries when DOSELORA sends ended."
**Trigger:** the DOSELORA-001 / SMALLVD-001 session sends "ended". Do not use the lock file's absence
as a stand-in for that message: its queue releases and retakes the lock between steps.

## Before launch
1. Confirm "ended" was received, and that `%TEMP%\rineya-gpu-0.lock` is absent or belongs to no live
   process under its own rule.
2. Send any live peer a one-line notice: about 80–90 min, one lease, about 3.5 GB host.
3. Confirm that none of these exist: `convention_localization_v1/{arms,scores,STEP1_SCORES.json,STEP1_RESULT.json}`
   or `interference_mechanism_v1/{arms,scores,IM_SCORES.json,STEP1_RESULT.json}`.

## Launch (through WMI; since the 2026-09-24 app restart, only WMI-launched jobs survive)
- Interpreter: `C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe`. This is the
  development environment. Do NOT use `.venv-rab001-cu126`: it has no `accelerate`.
- Target: `experiments/interference_mechanism_v1/run_gpu_batteries.py`, with cwd in the same directory.
- Runner: `wmi_runner.py` in this directory. It opens its own log and writes an rc file on exit.
  Launch it from PowerShell:
  ```powershell
  $d = 'C:\Users\clclc\BusinessProject\ChatBubble-titan-model-soup-mini-v1\experiments\interference_mechanism_v1'
  $py = 'C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe'
  $cl = "`"$py`" -u `"$d\wmi_runner.py`" `"$d\run_gpu_batteries.py`" `"$d`" `"$d\run_gpu_batteries.log`" `"$d\RUN_RC.json`""
  $si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
  Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cl; CurrentDirectory=$d; ProcessStartupInformation=$si}
  ```
  Wait on `RUN_RC.json`, not on the process.

## What it does
- One GPU lease for the whole window. The positive controls of both builders run once, first.
- Then each arm is built, verified, scored and its weights deleted, because disk is tight: C: had
  9 GB free. At most one new arm is on disk at a time, and BUILD.json keeps its sha256.
- Battery A: ADD-O50/O60/O90, H1, H1c on T_BANK_V2.
- Battery B: SPARSE-125, ADD-A125, DARE-O50, DARE-O10, DISJOINT-DARE on T_BANK_V3X.
- After the lease is released, it runs `convention_localization_v1/analyse_step1.py` and
  `interference_mechanism_v1/analyse_step1.py`.

## After
- Send DOSELORA and any live peer "ended".
- Write results in registration order in each directory's RESULTS_STEP1.md, as exploratory.
- Commit through a private index to `research/titan-model-soup-mini-v1`, and push if the operator
  still wants pushes.
