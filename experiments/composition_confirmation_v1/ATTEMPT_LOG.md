# CONFIRM-001 — attempt log

## Attempt 1 (2026-09-24 16:54:55 to 16:55:32): environment failure before any row was generated

**What happened.** The driver was launched with `.venv-rab001-cu126` (torch 2.10.0+cu126,
transformers 4.51.3). RAM (6.79 / 6.78 GiB), lease acquisition and preflight all passed: the six
existing arms were verified against their BUILD.json hashes. `score_arms.py` then failed inside the
**first** `from_pretrained` call (BASE), with
`ValueError: Using a device_map or tp_plan requires accelerate`. That venv has no `accelerate`.
The driver released the lease and stopped with rc 1 after 36 s.

**Why the read budget is intact.**
- The traceback sits in model loading for the first arm, before any prompt was tokenised or
  generated (`ATTEMPT1_run_confirm.log.txt`, lines 13–40).
- `score_arms.py` writes its output only after every arm in the step has finished. No
  `CONF_GATES.json`, `GATES_RESULT.json`, `arms/` or per-arm summary line exists. The log contains
  0 `"arm"` summary lines.
- No generation from T_BANK_V3X exists anywhere. The bank's single read has not been spent.

**Cause, and whose error it was.** Mine. I checked that the interpreter version matched
development (3.11.15) but never ran a model load with it. Every development score in this programme
(CALIBRATION_V2, V2_ARMS, MD_SCORES) records torch **2.6.0+cu124** and transformers **5.14.1**. That
is the hermes-agent venv, and FA-001 names it as `sys_executable`. This was a gate that had never
been reached (memory `gates-must-be-able-to-fail-and-pass`), caught only by colliding with it.

**Fix, verified before relaunch.**
- The driver is relaunched with the development interpreter
  `C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe`: torch 2.6.0+cu124,
  transformers 5.14.1, accelerate 1.14.0, psutil 7.2.2.
- Every subprocess inherits it through `sys.executable`.
- **Canary**, under the lease: SPARSE was loaded through `score_arms.py`'s exact code path, and
  6/6 recorded V2 generations (two per family, `CALIBRATION_V2.json`) were reproduced **byte for
  byte**. Only the exposed V2 bank was touched.

No registered file changed. The environment matches the development scoring environment exactly,
which is the stronger comparability choice. PREREG.md did not name an environment.

Artifacts kept: `ATTEMPT1_PREFLIGHT.json` (all six hashes match), `ATTEMPT1_run_confirm.log.txt`
and `ATTEMPT1_RUN_RC.json`.
