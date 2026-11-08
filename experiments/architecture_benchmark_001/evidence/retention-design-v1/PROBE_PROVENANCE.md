# The 60-row "probe" is panel rows 1-60, and no long-run weights survive

Two claims from the 2026-09-14 design review, verified here against the artifacts.

## 1. `rigorous/` and `longbudget/` are NOT an independent readout

`data/monitor_probe.jsonl` (the 150-row panel) and `outputs/splits_large/task_probe.jsonl` have
an **identical id sequence, 150 of 150**. Both harnesses then slice the first N rows:

    rigorous_pilot.py:70   task_rows = jl(SPL / 'task_probe.jsonl')[:N_TASK]   # run at 60
    long_budget.py:91      task_rows = jl(SPL / 'task_probe.jsonl')[:N_TASK]   # default 60

So every `rigorous/` and `longbudget/` "task hit" is a score on **panel rows 1-60**, under a
different scorer (normalised-string / number-sequence rather than the typed grammar).

Consequences, in both directions:

* They cannot corroborate a panel result — it is the same rows. Any argument of the form
  "it also shows up on the rigorous set" is circular.
* But the three-seed `rigorous` family is therefore a **direct panel-subset variance estimate**,
  which is more useful than it was being credited for. On rows 1-60 under the numeric scorer,
  `source_control` spans 21/31/21 and `ced_last4` spans 18/29/22 across seeds 17/29/43.

## 2. No usable weights survive from either family

    rigorous_pilot.py   0 occurrences of torch.save / save_file / state_dict()
    long_budget.py      1 -- long_budget.py:360, a periodic resume state

The review asserted neither script saves anything. That is wrong for `long_budget.py`, which
writes `STATE_{TAG}.pt` (model *and* optimizer state_dict) every `SAVE_EVERY_S = 1800` seconds
and atomically replaces it. Its own comment at line 80 says what it is:

    SAVE_EVERY_S = 1800   # the resume state is ~7.15 GB; this is crash recovery, not history

It is a single rolling file, not per-milestone history, and **no `.pt` matching it exists on
disk** — it was removed after the run completed. `outputs/arm-checkpoints/` holds exactly 9
directories, all written by `arm_dose.py`.

The review's conclusion therefore stands even though its reasoning was wrong on this point:
there are no recoverable weights from either family. Proposals to "re-probe an existing
`ced_last4` checkpoint" or "re-probe the long run's nine checkpoints" are not free experiments —
they require retraining, roughly 10 minutes and roughly 6.5 hours respectively.

## What this changes

* Nothing in the panel results is invalidated; this is about what can CORROBORATE them.
* The 500k-token run's scores are panel scores, so the withdrawn "decline past 50k" claim was
  always a claim about the panel — see the withdrawal note in `DISTINGUISHABILITY_PREREG.md`.
* A cheap CED re-probe is unavailable. `arm_dose.py` also cannot construct a `ced_lastN` arm:
  its `_ARM_RE` accepts only `B28[lastN]`, `H[lastN]` and `P[lastN]`. `retention_probe.py` can
  score such an arm, so the missing piece is a save path, not a scorer.
