# TRISKILL-001 Stage A launch handoff

The native `train-001` run ended successfully after 29 updates and 53,933 processed
tokens. `trained_C_fp32.safetensors` was consolidated from checkpoint-00029;
`DELTA_C_CONSOLIDATION.json` binds its SHA-256. `BUILD_CANARY.json` records 310/310
matching tensors for both positive controls and a delta norm of 0.8575171663.

The Codex continuation runs from the registered `9da67940` commit in an isolated
worktree. The trained tensor was copied byte for byte from the prior checkout;
its SHA-256 was verified after the copy. The two JSON receipts were copied too.

`run_stage_a_wait.py` is a resource-only wrapper. It waits for two readings with
at least 5.0 GiB free host RAM and 6.0 GiB free disk, plus an unheld GPU lease,
then calls the registered `run_stage_a.py` without changing its arms, banks,
coefficients, scoring, canaries, or stop rules. It times out after four hours.
The wrapper does not read model outputs or Stage B banks.

The first WMI launch was interrupted by Windows with `0xC000013A` during the
wrapper's initial 10-second wait; it never called Stage A. A second launch used
relative target paths and exited 2 before calling Stage A. Their logs are
`STAGE_A_LAUNCH_ATTEMPT1.log` and `STAGE_A_LAUNCH_ATTEMPT2.log` (with its RC JSON).
The actual Stage A run began at 16:42:08 local in a persistent terminal session,
with its output captured in `STAGE_A_RUN.log` and final `STAGE_A_RC.json`.

It scored BASE, TT, C100, C25, C50 and TTC25. A1 passed. At the next arm build,
the 5.0 GiB RAM check refused TTC50 at 4.34 GiB; the runner released its GPU
lease and exited 1. All completed units remain resumable. Before resuming, the
BASE/TT byte canaries, A1 and TTC25's completed row counts and weight-deletion
receipt were independently rechecked and passed. The wait wrapper now permits
this verified partial state, while still refusing an existing final result.
