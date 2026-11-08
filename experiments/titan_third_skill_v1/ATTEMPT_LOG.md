# TRISKILL-001 Phase 0 — attempt log and Amendment 1 (execution layer only)

## Attempt 1 — 2026-09-26 14:59:59–15:06:58 local, rc 1

- Launched through WMI at `a96fd42f` + `ac6f6cd0`. Every canary passed: the template, the scorer
  (640 rows) and all three weight hashes. The lease `triskill001-phase0` was taken and then
  released cleanly by its context manager.
- Completed units: BASE × {A, C, P, R, S}, SPARSE × {A, C}. SPARSE/P died mid-bank with
  `RuntimeError: CUDA error: unknown error`.
- **Cause:** a GPU driver reset. The System log shows `nvlddmkm` event 14 at 15:06:48, then event
  153 ×4 at 15:06:48–53, after event-13 graphics exceptions from 15:01. The card is shared with
  the Unity editor's graphics. This is an execution failure, unrelated to any model output.
- Outputs are archived **unchanged** in `attempt1/`: seven prediction files, the log and the rc.
  No `PHASE0_SCORES.json` was written. The per-bank counts printed to the log were read in session.

## Amendment 1 — written before attempt 2 launches

These changes are to the execution layer only. Banks, arms, decoding, scorer, floors, gates and
candidate order are all **unchanged**.

1. `screen_source.py` checkpoints per (arm, bank). Each unit's file is written atomically (tmp,
   then rename). On resume a unit is reused only if its ids and prompt hashes equal the bank's.
   `PHASE0_SCORES.json` is aggregated from the unit files alone.
2. `run_phase0.py` runs each attempt in a **fresh process**, because a driver reset kills that
   process's CUDA context. There are at most 3 attempts, and each re-checks the RAM and lease gates.
3. **Determinism canary (new, registered here).** Attempt 2 starts from an empty
   `phase0_predictions/`, so it regenerates all 15 units. `analyse_phase0.py` records whether each
   of attempt 1's seven units is **byte-identical** to its attempt-2 regeneration. Greedy FP32
   decoding should reproduce exactly, as the soup lane's canaries have (224/224). Any mismatch is
   reported first in the result, and makes every Phase 0 number provisional.
