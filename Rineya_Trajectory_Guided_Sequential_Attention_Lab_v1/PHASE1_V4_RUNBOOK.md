# TGSA Phase 1 v4 — operator and agent runbook

Every command below was executed end to end by
`scripts/phase1_v4_pipeline_rehearsal.py` at toy scale before this file was
written, so the sequence is validated rather than reconstructed. What the rehearsal cannot
validate is scale, and §0 says what that costs.

v4 differs from v3 in three operational ways:

1. a **competence gate** must return `GO` before the matrix may be dispatched;
2. the registered config is **generated**, not edited;
3. the regime is ~10× the data and ~8× the updates, so the matrix is a
   multi-hour job rather than a multi-minute one.

Everything else — the barrier discipline, the single frozen evaluate, the
evidence chain, the forbidden actions — is unchanged from
`PHASE1_V3_RUNBOOK.md` and that file remains the reference for those.

---

## 0. Budget before you start

Measured on this machine (RTX-class, 8 GB): **~8.8 min per 10,000 updates per
run**, one run per (arm, seed) cell.

| seeds | cells | at 20k updates | at 30k updates |
|---:|---:|---:|---:|
| 10 | 30 | ~8.8 h | ~13 h |
| 20 | 60 | ~17.6 h | ~26 h |
| 30 | 90 | ~26 h | ~40 h |

Two concurrent runner processes fit in 8 GB (each holds ~1.2 GB) and push the
GPU from ~40% to ~85%, so the wall clock is roughly 60% of the table. Do not run
more than two: the third contends without adding throughput.

`OBSERVED` Host RAM, not GPU memory, is the binding constraint when anything
else is running. A CPU power audit at 22 workers exhausted system RAM and killed
a GPU training process with a CUDA OOM that looked like a GPU problem and was
not. Keep ≥4 GB host RAM free, and do not run the power audit and the matrix at
the same time on this machine.

---

## Setup

Same as v3. The isolated environment and the "never reuse an evidence root" rule
carry over verbatim — see `PHASE1_V3_RUNBOOK.md` §Setup.

---

## 1. Produce the competence report

Development-only, arm-blind. This is the artifact that gates everything else.

```bash
python scripts/phase1_v4_competence_run.py --output <evidence>/competence_easy50k_20k.json --band easy_nontrivial --train-examples 50000 --holdout 3000 --steps 20000 --seeds 1,2,3
```

Exit code 0 means `GO`, 1 means `NO_GO`. The report records per-seed held-out
accuracy, per-family accuracy, distinct prediction counts, and the loss curve of
every run — all of which the gate consumes and none of which the pilot recorded.

If the gate itself has changed since a report was written, `require_go` rejects
it on schema version rather than honouring a stale rule. Recompute the verdict
from the stored raw runs instead of re-measuring:

```bash
python scripts/regate_competence_report.py --input <stored.json> --output <regated.json>
```

It cannot manufacture a `GO` — a stored `GO` whose raw runs fail the current
thresholds re-gates to `NO_GO` — and it never modifies its input.

**Read the reasons on a `NO_GO`, then ask the ladder what is permitted:**

```bash
python scripts/phase1_v4_escalation.py <report1.json> <report2.json>
```

It answers with the one move the registered ladder allows, moving a single axis
of the regime that failed, and refuses past four distinct regimes. It is not
licence to keep tuning until the gate passes — see
`PHASE1_V4_PREREGISTRATION.md` §Escalation.

`OBSERVED` Diagnose *before* escalating. A dispersion failure has two possible
causes with opposite responses, and only one is on the ladder:

```bash
python scripts/phase1_v4_convergence_diagnostic.py <report1.json> <report2.json>
```

Stalled runs still descending at the final step mean the budget truncated a
phase transition, and more updates is the registered response. Stalled runs that
have plateaued mean the transition is stochastic, which more updates will not
fix.

## 2. Re-run the power audit at the realised seed variance

`OBSERVED` The v3 seed count of 20 was powered for σ_a = 0.02. Carrying it
forward without re-checking is the failure the v4 pilot nearly walked into: it
produced a regime whose realised σ_a was ≈ 0.20, where no realistic seed count
detects the target effect.

Read the realised per-arm between-seed SD out of the competence report
(`detail.per_seed_holdout_accuracy.<arm>.sd`), then audit around it:

```bash
python scripts/phase1_power_simulation.py --output power/v4_power.json --trials 10000 --seed-counts 10,15,20,25,30,40 --effects 0.000,0.010,0.015,0.020,0.030 --seed-sds <realised>,<2x realised> --central-seed-sd <realised> --elevated-seed-sd <2x realised> --stratum-count 3 --examples-per-stratum 1333 --workers 8
```

`OBSERVED` The simulator is cheap at v4's structure — measured 1.2 ms/trial at
10 seeds rising to 6.9 ms at 60, so a 10,000-trial condition costs 0.2–1.1 min
and a 200-condition sweep finishes in well under an hour on 4 workers. An
earlier revision of this runbook claimed a full sweep was ~10⁹ operations per
condition and would not finish; that was wrong, and it was wrong because it was
estimated rather than measured. Sweep the whole σ_a range.

`OBSERVED` **`--stratum-count 3`, not v3's 17.** The certified band fixes pointer
chains at 2 hops and every sequence is ≤ 32 tokens, so `context_bucket` is
constant and the primary estimand has three strata:
`pointer_chase:h2:short`, `state_update:s2:short`, `state_update:s3:short`.
Auditing at v3's structure would report the power of a study nobody is going to
run. See `PHASE1_V4_PREREGISTRATION.md` §Stratum structure.

If `recommended_seed_count` is null, the design is not powered and **must not be
registered**. `build_phase1_v4_config.py` enforces this.

## 3. Generate the registered config

Never hand-edit it. The builder copies the regime from the competence report,
the seed count from the power report, and the entire `analysis` block from v3
verbatim, then SHA-256 binds both inputs into `registration`.

```bash
python scripts/build_phase1_v4_config.py --competence <competence.json> --power <power.json> --output configs/phase1_identifiability_v4.json
```

Copy the competence report **next to the config**: the runner resolves
`competence_report` relative to the config's own directory.

`OBSERVED` The generated config sets `dataset.split_generation_order` to
`["test", "dev", "train"]`. Do not "simplify" this away. Splits are generated in
sequence and each excludes the signatures already taken, so whichever runs first
is the only unbiased draw. Generating train first on this band consumed the
entire 7,680-item non-trivial two-step `state_update` space and left the test
split with three-step items only — an unintended difficulty shift, and one that
cut the primary estimand from three strata to two. Check
`dataset_manifest.json`'s `unbiased_split` field after preparing data.

## 4. Rehearse before dispatching

Cheap, and it has already caught one defect that would have aborted the first
cell of the matrix.

```bash
python scripts/phase1_v4_pipeline_rehearsal.py
```

Runs the full sequence at toy scale including a negative control that flips the
gate to `NO_GO` and asserts the runner refuses **before creating an output
root**. A clean git working tree is required — the runner refuses confirmatory
dispatch on a dirty checkout, which the rehearsal will surface for you.

## 5–9. Prepare, train, barrier, evaluate, verify, analyze

Identical to v3 except that the config now carries `competence_report`, so
`run_phase1_model_study.py` consults the gate before it creates anything.

```bash
python scripts/prepare_phase1_data.py --config configs/phase1_identifiability_v4.json --output-dir <evidence>/dataset
```

```bash
python scripts/run_phase1_model_study.py --config configs/phase1_identifiability_v4.json --dataset-dir <evidence>/dataset --output-dir <evidence>/study --phase train
```

```bash
python scripts/run_phase1_model_study.py --config configs/phase1_identifiability_v4.json --dataset-dir <evidence>/dataset --output-dir <evidence>/study --phase barrier
```

```bash
python scripts/run_phase1_model_study.py --config configs/phase1_identifiability_v4.json --dataset-dir <evidence>/dataset --output-dir <evidence>/study --phase evaluate
```

```bash
python scripts/verify_phase1_evidence.py --config configs/phase1_identifiability_v4.json --output-dir <evidence>/study --dataset-dir <evidence>/dataset --expect-confirmatory true
```

```bash
python scripts/analyze_phase1_model_study.py --config configs/phase1_identifiability_v4.json --output-dir <evidence>/study --dataset-dir <evidence>/dataset
```

The barrier must pass before `evaluate` is run even once. `evaluate` is the
single frozen exposure and is not repeated.

---

## Sharding the train phase

At 20 seeds the train phase is long enough that it will be interrupted. It is
resumable per cell, and `--arms` / `--seeds` shard it. Two concurrent shards is
the throughput ceiling on this machine.

```bash
python scripts/run_phase1_model_study.py --config configs/phase1_identifiability_v4.json --dataset-dir <evidence>/dataset --output-dir <evidence>/study --phase train --arms cascade,history_control --seeds 101,202,303
```

`OBSERVED` Concurrent shards writing `preflight.json` can collide on Windows
with a `PermissionError`. Harmless: the unrestricted barrier pass regenerates
the complete preflight. Do not "fix" it by serialising the whole matrix.

---

## Forbidden during confirmatory execution

Everything forbidden in v3, plus:

* **Editing the competence report after the matrix is dispatched.** Its SHA-256
  is bound into the config's `registration` block and the verifier will catch it.
* **Re-running the competence step until it returns `GO`.** The escalation ladder
  is one step, and it is registered in advance.
* **Selecting a regime because `learned_delta` did well on it.** Selection is
  arm-blind on the minimum across all three arms. The pilot contains a live case
  where the temptation was real and the rule held.

---

## Interpreting a `NO_GO`

A `NO_GO` is a result about the *experimental setup*, never about the mechanism.
It says the regime cannot support an interpretable contrast — the same thing v3
discovered after spending the full matrix. Discovering it for the price of a
development run instead is the entire purpose of the gate.
