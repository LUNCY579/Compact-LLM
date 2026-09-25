# Runbook: the two exploratory GPU batteries

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
- Use a runner that opens its own log (`run_gpu_batteries.log`) and writes `RUN_RC.json` on exit.
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
