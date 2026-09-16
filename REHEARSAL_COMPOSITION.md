# Composition is the mechanism: rehearsal protects what it resembles, and nothing else

**RRP-011. The hypothesis is confirmed. Two of five registered predictions are
wrong, and one of them is wrong in the direction that matters most.**

RRP-010 showed rehearsal protecting the arithmetic panel while *damaging* long
context and abstention, and offered an untested mechanism: the pool skews toward
short conversational classes. RRP-011 varies **only which rehearsal rows are
eligible** — same 768 transition rows at seed 17, same ~108k rehearsal-token
budget, same recipe — and measures all three instruments.

## The result

| arm | rehearsal drawn from | new /384 | panel | ret /47 | gate | **long ctx /256** | **`UNKNOWN` /64** |
|---|---|---:|---:|---:|:--:|---:|---:|
| **Supa** (source) | — | 14 | 47 | 47 | — | **161** | **13** |
| D3 | none (1,536 transitions) | 289 | 71 | 37 | PASS | 150 | 3 |
| M1 | all 53 classes | 268 | 74 | 40 | PASS | 139 | 1 |
| **M2** | `communication` | 265 | 70 | 40 | PASS | **118** | 5 |
| **M3** | `reasoning` | 273 | 69 | **34** | **FAIL** | **144** | **0** |
| **M4** | 7 abstention classes | 267 | **77** | 39 | PASS | 137 | **13** |

Every arm clears all three transition floors. New skill is flat across arms
(265–273) — composition does not move acquisition.

**Composition moves everything else.** Long context spans **118–144** across arms,
a 26-row swing on identical transition data. Abstention spans **0–13**, the entire
available range.

## What was predicted, and what actually happened

1. *"M2 (communication) is worst on abstention and long context."* — **Half right.**
   M2 is decisively worst on long context (118, worse than every other arm and 32
   below Supa). But on abstention M2 scores **5**, *better* than M1's 1 and M3's 0.
2. *"M3 (reasoning) is better than M2 on both."* — **Split.** Better on long context
   (144 vs 118) and **worse** on abstention (0 vs 5).
3. *"M4 recovers abstention materially, above M1's 1 and D3's 3."* — **Confirmed,
   and far beyond the prediction.** M4 scores **13/64 — exactly Supa's number.**
   Abstention is not partially recovered; it is fully restored.
4. *"No arm reaches Supa's 161."* — **Confirmed.** Best is M3 at 144.
5. *"Panel retention stays ≥ 37 in all arms."* — **Falsified.** M3 retains **34** and
   fails the gate, despite drawing rehearsal from the *reasoning* group.

## The finding, stated plainly

**Rehearsal protects the capability it resembles, and only that one.**

- Rehearse on abstention classes → abstention returns to source level (13/64), and
  the panel does best of any arm (77).
- Rehearse on reasoning classes → long context does best of any rehearsal arm
  (144), and panel retention *fails* (34).
- Rehearse on communication classes → long context collapses to 118, the worst
  measured anywhere.
- Rehearse on everything (M1) → middling on all three, protecting none of them
  particularly.

This is why RRP-010 looked like "rehearsal damages the model". It does not. M1's
mixed pool simply contained too little of what long context and abstention needed,
and enough conversational material to actively hurt long context.

**The corollary is the practical one: a rehearsal mixture must span every capability
you intend to keep.** Anything absent from the mixture is unprotected, and anything
over-represented can do harm. "Add some old data" is not a strategy; *which* old
data is the whole intervention.

## What this does not establish

**Long context is still damaged in every arm** — 118 to 144 against Supa's 161. No
composition tested here repairs it, and the reason is a data limit stated in the
pre-registration before any arm ran: **there is no long-context training data in
this worktree.** `Rineya_LCR_SFT_v1` ships a 12-pack sample and no corpus; the v3
bulk files average 489 and 523 characters against a Demo-suite context of ~1,000
words. Every arm here varies content at nearly constant length, so **"it was the
shortness" remains untested.** If the resemblance principle holds, long-context
rehearsal would be the thing that repairs long context — and that dataset does not
exist here.

**No arm is selected on these numbers.** `QUARANTINE.json` bars the Long-Context
Demo for `[TRAIN, SELECTION, CONFIRMATION]` and RRP-011 was registered report-only.
M4 is the most interesting arm; it is **not** thereby a recommended configuration.
Promoting the winner of a four-arm sweep on a barred suite is the post-hoc gate the
bar exists to prevent. A confirmation needs a fresh instrument and its own
registration.

Single seed per arm. Four arms on one 256-row suite is four comparisons; the
reading is the *ordering*, not any per-arm significance. M3's gate failure at
retained 34 sits just below the D3 seed spread (35–37) and should not be
over-read from one seed.

## Reproduce

```bash
bash rineya-dataset-retention-records/run-composition.sh
```

Pre-registration with all five predictions, the `integrated`-arm feasibility
addendum, and the report-only binding is in
`forge-v2-prereg-001/RRP011_COMPOSITION_PREREG.md`, written before any arm was
built. Artifacts `PROCESS_rrp011_M{2,3,4}.json`,
`RETENTION_PROBE_rrp011_M{2,3,4}.json`, `LONGCTX_rrp011_M{2,3,4}.json`.
