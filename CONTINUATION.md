# Model-quality continuation

## 2026-09-17: completed clean pilot and corrected historical interpretation

Read [RESULTS_AND_PROGRESS.md](docs/model-quality/audit-2026-09-17/RESULTS_AND_PROGRESS.md)
before resuming. Six historical D1 seeds exist; three surviving D1 model exports
are hash-verified and marked for preservation. New RRP-015 control and rehearsal
training/evaluation are complete, with full checkpoints retained. On the new
selection instrument, Supa/Opallo/control/rehearsal score14/8/234/225. Control
passes experimental Gate v2; rehearsal fails source retention. Neither establishes
broad preservation or release readiness. Historical and new384-row instruments
must not be compared as identical rows.

New corrections: M4's13 reported UNKNOWN matches include four `UNKNOWN.` outputs
(only9 literal); short arithmetic contacts reached old rehearsal training;
nonzero superiority margins were tested against zero; new training now excludes
both scored-parent banks and requalifies natively before optimization. Original
registrations and raw historical scores are preserved rather than overwritten.

The older continuation below is retained for chronology; its universal replay
claims and current-status statements must be read with the newer audit.

## Where this stands now (2026-09-16)

A one-step **state-transition** skill was taught to the frozen Supa 1.7 source and
replicated. Read these in order:

1. **[PROCESS_TRANSITION_FLOOR.md](PROCESS_TRANSITION_FLOOR.md)** — the gap.
   Neither Supa nor Opallo can apply one state-update rule; both score **below**
   the shortcut floor on all three families.
2. **[PROCESS_TRANSITION_ARM.md](PROCESS_TRANSITION_ARM.md)** — the fix, replicated.
   **289/384 in all three seeds**, every family clearing its floor. Retention passed
   2 of 3 seeds.
3. **[PROCESS_DOSE_FRONTIER.md](PROCESS_DOSE_FRONTIER.md)** — the trade. New skill
   rises with dose (239 → 304) while retention falls; D1 at a quarter the data
   clears everything with margin.
4. **[PROCESS_REHEARSAL.md](PROCESS_REHEARSAL.md)** — rehearsal makes **panel**
   retention reliable (3/3 vs 2/3, disjoint distributions) and beats dose reduction
   at equal panel retention.
5. **[BROAD_RETENTION.md](BROAD_RETENTION.md)** — **read this before believing 4.**
   On long context both arms lose against Supa's 161/256 and the rehearsal arm loses
   *more* (139 vs 150). Abstention collapses in both (13 → 3 → 1 of 64).
6. **[REHEARSAL_COMPOSITION.md](REHEARSAL_COMPOSITION.md)** — why. Rehearsal protects
   **only what it resembles**. Abstention-class rehearsal restores abstention to
   **exactly Supa's 13/64**; reasoning-class rehearsal gives the best long context and
   fails panel retention; communication-class rehearsal costs 26 long-context rows.
   Composition is the intervention, not volume.
7. **[REHEARSAL_BLEND.md](REHEARSAL_BLEND.md)** — and why "span everything" fails.
   A 50/50 abstention+reasoning blend protected **neither**: abstention 13 → **0**,
   long context below both components. Protection is a **threshold in budget share**,
   not a gradient. Three of four predictions falsified.
8. **[DATASET_REQUIREMENTS.md](DATASET_REQUIREMENTS.md)** — the analytical breakdown:
   every rule for teaching a new skill without losing an old one, each with the
   measurement that produced it and the failure it prevents. §2.1 is **corrected** by
   document 7.
9. **[DATASET_SURVEY.md](DATASET_SURVEY.md)** — what is actually on this host. Full-file
   census of 148,000 rows. **Nothing here can rehearse long context** (longest row: 940
   characters). The retention panel is a **literal subset** of the v3 reasoning corpus.
   29,764 clean rehearsal rows are untouched. 53,000 preference pairs exist with no code
   to use them. Corrects document 7's scope.
10. **[PROCESS_DOSE_RELIABILITY.md](PROCESS_DOSE_RELIABILITY.md)** — **read this before
    trusting the gate.** D0 passes the gate **3/3** while acquiring **0/3**; D1 acquires
    **6/6** while passing **4/6**. The gate has no acquisition term, so training on less
    passes it more reliably *because* it learns less.
11. **[PROCESS_ASYMMETRIC_DOSE.md](PROCESS_ASYMMETRIC_DOSE.md)** — the best-behaved arm
    measured. Dosing one family separately (`ledger_replay` 128, others 64) clears all
    three floors **3/3** and is the **only** configuration where long context is n.s.
    against Supa in **every** seed (156–158, spread 2). But it does not restore that
    family's rate — identical parents score ~0.15 lower than in D1, so acquisition tracks
    total volume too — and the gate failure moved from retention to the panel.

### RRP-013 / RRP-014 result (2026-09-16, latest)

**Nine runs: D1 (384 rows) at six seeds, D0 (192 rows) at three.**

| arm | gate | exact 95% CI | all three floors cleared | retained | long ctx | `UNKNOWN` |
|---|:--:|---|:--:|---:|---:|---:|
| **D1** (384) | **4/6** | [0.223, 0.957] | **6/6** | 36–40 | 147–160 | 5–12 |
| **D0** (192) | **3/3** | [0.292, 1.000] | **0/3** | 38–44 | 153–155 | 10–13 |
| *Supa* | — | — | *0/3* | *47* | *161* | *13* |

D0's deficit is one family in every seed: `ledger_replay` scores 0.2812–0.3203 against
its 0.2422 `copy_before` floor, where D1 scores 0.4844–0.6328. It carries the strongest
shortcut of the three families, and at 64 parents/family the model is no longer
distinguishable from copying.

**Neither rung is a clean win.** D1 learns all three skills and fails retention 2/6; D0
passes every gate and never demonstrates the third skill. Six seeds still cannot pin a
pass rate — the interval only narrowed from [0.094, 0.992] to [0.223, 0.957].

Two of five registered predictions were falsified: D0 did **not** clear all three floors,
and abstention is **not** uniformly broken (5–13 across arms, one seed at Supa's exact 13).

**RRP-014 then split the dose by family** — `ledger_replay=128, prerequisite_plan=64,
register_machine=64`, 256 rows, three seeds, on rows already drawn for D1 and D0 (verified
per seed: A1's `ledger_replay` parents are exactly D1's, the other two exactly D0's).

| | A1 (256) | D1 (384) | D0 (192) |
|---|---|---|---|
| all three floors cleared | **3/3** | **6/6** | **0/3** |
| gate | 2/3 | 4/6 | 3/3 *(by not learning)* |
| **long context n.s. vs Supa** | **3/3** | 4/6 | 2/3 |
| long-context range | **156–158 (spread 2)** | 147–160 (spread 13) | 153–155 |

A1 is better than D1 on **retention and abstention in all three matched seeds** and is the
only arm whose long context is never significantly worse than the untrained source. Three
of its six predictions were falsified — most importantly, `ledger_replay` scored
0.4375–0.4766 on *identical parents* where D1 scored 0.4844–0.6328, so **per-family dosing
is not per-family independent**: cutting the other families cost this one ~0.15.

**No configuration passes the gate reliably** — D1 4/6, A1 2/3, D0 3/3 only by failing to
learn. Every rung trades somewhere and the gate sees only part of the trade.

### The one-sentence state

The new skill is real, large and replicates exactly; **panel** retention is
manageable and rehearsal manages it; **long-context and abstention are damaged by
every arm measured, the registered gate is blind to both — and RRP-013 shows it is
blind to acquisition too, which a lower dose silently exploits.**

### What must happen before anything here is promoted

The gate is `panel ≥ 62 AND references retained ≥ 37 of 47`
(`ADJUDICATION_001.md` D3). It has no long-context term, no abstention term and no
acquisition term, and RRP-010/013/014 show damage in all three places.

**That registration was written, and its result is that the gate cannot be extended
with anything in this repository.** See
`rineya-dataset-retention-records/forge-v2-prereg-001/GATE_V2_PREREG.md`. All three
candidate terms are barred, by two independent mechanisms:

| candidate term | status | the bar |
|---|---|---|
| long context | **BARRED** | `QUARANTINE.json`: 256 rows, `bars: [TRAIN, SELECTION, CONFIRMATION]`, reason `SPENT`. A gate *is* a selection use. |
| abstention | **BARRED** | computed over the 64 unanswerable rows **inside that same bank**; no separate permission, and not independent of the long-context score. |
| acquisition | **BARRED** | absent from `QUARANTINE.json`, but **26 of 26** `PROCESS_*.json` files declare `"never for selection"` and **384 of 384** views carry `view_role: CANDIDATE_NOT_ADMITTED_OR_TRAINED`. |

A bar stated in the instrument binds as firmly as one in the registry. Promoting the
acquisition probe to a gate term would retroactively contradict 26 recorded evidence
files.

**GATE v2 is therefore defined but inoperative**: GATE v1 conjoined with "every family
clears its published binding floor", where that conjunct activates only once
**SELECT-VIEWS-001** exists — a fresh 384-view partition declared selection-admissible
*at birth*, with floors published before any model call. It is buildable: 4,771 parents
sit unused beyond the current probe's 384. Every threshold in GATE v2 pre-exists
(`62` and `37` from `DATASET_PROGRAMME_PREREG.md:140-141`; the acquisition conjunct
introduces no new threshold, reusing the exact-binomial lower-bound rule already
published in `PROCESS_PROBE_FLOORS.md`). **`62` was deliberately not lowered** despite
RRP-014 seed 29 failing at 61.

**SELECT-VIEWS-001 is now built** (`select-views-001/`, sha256 `7dd53f2a…`): 384 views,
128/family, `view_role: ADMITTED_FOR_SELECTION`, zero parent overlap with the existing
probe, floors published before any model call (0.2344 / 0.0781 / 0.0703). Scored on the
untrained source it reproduces the old probe's baseline exactly — 14/384 overall, 0/3
floors cleared — which is the evidence it measures the same capability. **GATE v2's third
conjunct is operative for future arms.**

`process_arm` gained a repeatable `--exclude-views`, and **every future arm must pass
`--exclude-views <records>/select-views-001/SELECT_VIEWS.jsonl`**. Measured, not assumed:
without it a D1-dose draw at seed 17 trains on **46 of the gate instrument's own parents**
and is then judged by them.

### What the adversarial review of GATE v2 found

Three independent gate designs, each attacked by three lenses. **All eight returned
verdicts refuted the design under review — seven FATAL, one SERIOUS.** Four findings hit
the registration itself and are recorded as Amendment 6:

1. **The panel term is an improvement requirement, not a preservation one.** The frozen
   source scores `panel = 47` against a threshold of 62, and five *function-preserving*
   controls score 47 with 47/47 retained — perfect retention — and all five **fail**.
2. **`62` is the worst of a different deliverable's four seeds** (74, 62, 73, 68). Recorded,
   deliberately **not changed**: lowering it now, knowing RRP-014 seed 29 failed at 61, is
   the post-hoc move the registration refuses.
3. **Bank-level quarantine cannot catch text-level leakage, and this repo has a measured
   instance.** `splits_onpanel/train.jsonl` shares **47/150 verbatim panel stems** (102/150
   digit-blinded) while passing all eight id and template-family disjointness checks,
   because different template families generate identical stems. That leak never reached
   anything — every completed arm used `splits_large` (0/150) — but **GATE v2 has no
   text-level guard**, so any future arm not training on `splits_large` must publish a
   digit-blinded stem-overlap count against the panel before it is gated.
4. **The permissive bare policy is load-bearing and still marked `pending`.** Under strict,
   `source_correct_total` is 41 and no arm has ever reached 37 retained.

One review claim was **checked and rejected**: the "of 47" denominator is *not*
candidate-dependent — permissive `source_correct_total` is 47 in 40 of 40 artifacts and the
probe aborts before writing if the source fails to reproduce. The field that varies is
`reference_size` (47–85), which the gate does not read.

Until a fresh confirmation instrument exists, arms are judged by GATE v2 with long context
and abstention reported alongside, so the remaining gaps stay visible rather than silently
tolerated.

Do **not** switch the recommendation to D3 because D3 wins on long context.
`QUARANTINE.json` bars that suite for `[TRAIN, SELECTION, CONFIRMATION]`; RRP-010
was registered report-only. Selecting on it after seeing it is the post-hoc gate the
bar exists to prevent.

### That hypothesis was tested, and confirmed

RRP-011 varied rehearsal composition and nothing else. Composition swings long
context by 26 rows (118–144) and abstention across its entire range (0–13) on
identical transition data. **Rehearsal protects what it resembles.** The practical
rule is in `DATASET_REQUIREMENTS.md` §2.1: a rehearsal mixture must span every
capability you intend to keep, because anything absent is unprotected and anything
over-represented can do harm.

**And the follow-up rule was wrong.** "Span every capability you keep" was tested in
RRP-012 and falsified: at a fixed budget, splitting rehearsal across two capabilities
protected neither. Halving the abstention component took it from 13/64 to 0 — a
threshold, not a gradient. At a fixed budget you choose **one** capability to protect.
Whether a *larger* rehearsal budget could protect two is untested.

**Still open: long context, and now settled as unaddressable here.** No composition
tested repairs it — every arm sits 1–43 rows below Supa's 161. If the resemblance
principle holds the fix is long-context rehearsal, and `DATASET_SURVEY.md` §2 closes
that door with a full-file census: across all 148,000 rows of all seven corpora the
**longest single row is 940 characters (~261 tokens)**, and none exceeds 4,000. There is
nothing on this host to rehearse long context with. Commissioning the LCR corpus is not
one lever among several — it is the only one.

### Operational traps that cost launches this session

- **[docs/LANE_INTERLEAVE_HAZARD.md](docs/LANE_INTERLEAVE_HAZARD.md)** — RAB scorers
  write into the worktree; the RRP controller refuses a dirty worktree. A
  train→score→train loop refuses itself. **Train every arm, then score every arm.**
  Cost this session: three launches.
- Checkpoints are ~3.5 GiB each, two per run, against a 14 GiB `DISK_FLOOR`. Reclaim
  after scoring; see `DISK_RECLAIM_001.md`.
- A grant's `new_disk_bytes` and wall-second budget are usually what refuse a
  *second* run in the same journal, not tokens.

### Registrations, all written before their measurements

`rineya-dataset-retention-records/forge-v2-prereg-001/` — floors and amendments,
`ADJUDICATION_001.md` (four operator decisions + two amendments), `RRP008_DOSE_PREREG.md`,
`RRP009_MIXTURE_PREREG.md`, `RRP010_BROAD_RETENTION_PREREG.md`,
`RRP011_COMPOSITION_PREREG.md`, `RRP012_BLEND_PREREG.md` (+ amendment 1),
`RRP013_RELIABILITY_PREREG.md`, `RRP014_ASYMMETRIC_DOSE_PREREG.md`,
`GATE_V2_PREREG.md`.

Work on `codex/dataset-retention-audit-v1`. No release or quality claim is open:
every run records `MICROFIT_EXECUTED_NOT_QUALITY_CONFIRMATION` and
`quality_improvement_established: false`.

---

## Inherited context

The prior audit remains authoritative for everything it covers:
[MODEL_QUALITY_STATUS.md](MODEL_QUALITY_STATUS.md), which audits 50 live heads
across five repositories and answers the twelve next-iteration decisions. Its
reasoning-ceiling reading is **qualified** by the above: that reading was
established against final-answer SFT on long-prompt instruments, and it does not
survive contact with a short, atomic, solver-verified target.

The inherited RAB continuation is preserved
[verbatim in history](docs/history/model-quality-audit-2026-09-15/CONTINUATION.md).
