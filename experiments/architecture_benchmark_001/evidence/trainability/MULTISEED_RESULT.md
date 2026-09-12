# Multi-seed matched comparison: CED loses, replicated, with the gap 6× the seed variance

The first result in this programme with genuine replication, a real task probe, and
family-disjoint data. It supersedes the single-seed 2×2 in `MATCHED_COMPARISON.md`, which
stands but rested on one run per cell and a 6-prompt probe.

Exploratory, pinned CPU runtime, official Supa 1.7.0-devfreeze. Outside the hashed root — no
receipt, no gate. Open splits only; no sealed or DEV access; no efficacy claim.

## Design

| | |
|---|---|
| arms | frozen source vs C28 (CED, 14 encoder + 14 shared-memory decoder) |
| seeds | 17, 29, 43 (the protocol's registered training seeds) |
| recipe | AdamW lr 1e-5, wd 0, clip 1.0, gradient checkpointing, **all** parameters trainable |
| budget | 40 steps, 40 **distinct** training rows per seed (no cycling) |
| TRAIN | 800 family-disjoint rows across 36 classes |
| DEV | 40 rows, family-disjoint, never a gradient step |
| TASK | 60 held-out prompts from `percentage_reasoning` / `ratio_proportion` / `time_duration` — classes the frozen source demonstrably can do, **excluded from TRAIN/DEV** so this measures **retention** |

The seed shuffles TRAIN, so each seed trains on a near-disjoint 40-row subset (measured
overlap 1/40, 3/40, 1/40). That is the only stochastic element, so between-seed spread is a
clean measure of run-to-run variance at a fixed budget.

## Built-in control — PASSES

Step-0 metrics must be **identical** across seeds for a given arm, because the seed cannot
touch an untrained model. They are:

    source_control   dev CE 5.578166   task 20/60   identical across 3 seeds
    C28              dev CE 12.887974  task  0/60   identical across 3 seeds

This is the control that caught the earlier vacuous-seed bug. It now confirms the seed
affects training order and nothing else.

## Result

| arm | step | dev CE (mean) | per-seed range | task (pooled) | 95% CI |
|---|---:|---:|---|---:|---|
| source | 0 | 5.5782 | — | 60/180 (33.3%) | [26.9%, 40.5%] |
| source | 20 | 2.5285 | [2.478, 2.599] | 68/180 (37.8%) | [31.0%, 45.0%] |
| **source** | **40** | **1.7857** | [1.686, 1.851] | **73/180 (40.6%)** | **[33.7%, 47.9%]** |
| C28 | 0 | 12.8880 | — | 0/180 (0.0%) | [0.0%, 2.1%] |
| C28 | 20 | 4.5585 | [4.456, 4.644] | 0/180 (0.0%) | [0.0%, 2.1%] |
| **C28** | **40** | **3.1104** | [2.998, 3.214] | **0/180 (0.0%)** | **[0.0%, 2.1%]** |

**CED penalty at step 40: +1.3247 nats.** The largest within-arm seed spread is **0.2155**,
so the penalty is **6× the run-to-run variance**, and the per-seed CE ranges of the two arms
are **completely disjoint**:

    source  {1.8506, 1.6862, 1.8203}
    C28     {3.1198, 2.9979, 3.2135}

Task rates are disjoint at 95% by a wide margin. **C28 scored 0/60 on every one of three
seeds — 0/180 prompts, not one correct answer.**

## C28 is fluent and empty

Its failure is no longer token salad. At step 40 it produces grammatical, on-topic text with
no arithmetic in it:

    expected "64."          got "No, the current price is 10 minutes."
    expected "40."          got "In the current state, the current value is 10 minu…"
    expected "45 minutes."  got "In the current state, the current version is not a…"
    expected "90 minutes."  got "In the current state, the current time is 10:10 p.…"

Termination is 58–60/60 and distinct outputs 28–38/60, so the nondegeneracy and termination
checks alone would *not* flag it. Only the task probe does.

## KL is decisively falsified as a capability proxy

The sharpest single observation here:

| arm | dev KL to original source | task |
|---|---:|---:|
| source (trained) | 3.047 – 3.297 | **73/180** |
| **C28 (trained)** | **2.761 – 3.045** | **0/180** |

**C28 ends up *closer* to the original source distribution than the trained source itself —
and is completely incapable.** Any recovery plan using KL-to-teacher as its endpoint would
rank C28 above the working model. Combined with the earlier finding that CE improves while
capability is destroyed at lr 1e-4, **both** candidate likelihood endpoints in the registered
G3 shape are now measured to be invalid on their own.

## What this establishes, and what it does not

**Established.** At a matched recipe on family-disjoint data with three seeds, CED loses by
1.32 nats — 6× the seed variance, with disjoint ranges — and retains **none** of a capability
the source holds at 40.6%. Training does not rescue a displaced initialization: C28's dev CE
improves 9.8 nats (12.89 → 3.11) while its task score never leaves zero. Fluency recovery and
capability recovery are different things, and C28 achieves only the first.

**Not established.** Nothing beyond 40 steps or ~5k supervised tokens — a longer run could
still move C28, and this does not exclude it. Nothing about replay, frozen-teacher
regularisation, or a warmup/decay schedule. The probe is three numeric classes; retention of
other capabilities is unmeasured. Three seeds bound run-to-run variance at a fixed budget,
not variance across data subsets or hyperparameters. No sealed or registered DEV evaluation
was touched.

**Note on the source's own trend.** Its task rate *rises* with training (33.3% → 37.8% →
40.6%) even though the probe classes are excluded from TRAIN — positive transfer from the
held-in classes. Its per-seed spread at step 40 is 21/31/21 of 60, so the trend is weaker than
the endpoint suggests and should not be read as a result on its own.

## Reproduction

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
$PY experiments/rab001_exploration/build_splits.py
for s in 17 29 43; do
  for a in source_control C28; do
    $PY experiments/rab001_exploration/rigorous_pilot.py $a $s 40 1e-5 40 60 20
  done
done
$PY experiments/rab001_exploration/aggregate_rigorous.py
```

~14 min per source run, ~28 min per C28 run (C28 terminates later, so its generation eval
costs more). CPU only. Artifacts: `outputs/rigorous/RIGOROUS_*.json`,
`outputs/rigorous/AGGREGATE_MULTISEED.json`.
