# Model-quality continuation

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
7. **[DATASET_REQUIREMENTS.md](DATASET_REQUIREMENTS.md)** — the analytical breakdown:
   every rule for teaching a new skill without losing an old one, each with the
   measurement that produced it and the failure it prevents.

### The one-sentence state

The new skill is real, large and replicates exactly; **panel** retention is
manageable and rehearsal manages it; **long-context and abstention are damaged by
every arm measured, and the registered gate is blind to both.**

### What must happen before anything here is promoted

The gate is `panel ≥ 62 AND references retained ≥ 37 of 47`
(`ADJUDICATION_001.md` D3). It has no long-context term and no abstention term, and
RRP-010 shows damage in exactly those two places. **A configuration chosen on a gate
blind to two-thirds of the damage should not be shipped.** Extending the gate needs
its own registration, written before the next arm runs.

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

**Still open: long context.** No composition tested repairs it — every arm sits
17–43 rows below Supa's 161. If the resemblance principle holds, the fix is
long-context rehearsal, and **that dataset does not exist in this worktree**
(`Rineya_LCR_SFT_v1` ships a 12-pack sample; the v3 bulk files average ~500
characters against ~1,000-word contexts). Building or obtaining it is the highest-value
remaining data work.

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
`RRP009_MIXTURE_PREREG.md`, `RRP010_BROAD_RETENTION_PREREG.md`.

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
