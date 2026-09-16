# Rehearsal makes retention reliable, and beats spending less

**RRP-009, three seeds. The retention distributions do not overlap.**

RRP-007 left a specific problem: training on one-step transitions gains the new
skill identically across seeds (289/384 every time) while retention passes or
fails depending on the draw — 2 of 3 seeds cleared the gate, never retaining more
than the 37 minimum. RRP-008 showed dose moves both ends of that trade but cannot
escape it.

This lane adds rehearsal — v3 preservation rows interleaved with the transitions —
and replicates it at the same three seeds.

## The comparison

Both arms use the same recipe, window, optimizer and scored instruments. **M1 uses
exactly D2's transition corpus** plus ~108k tokens of rehearsal, so its total
compute matches D3's.

| | new skill /384 | panel /150 | **retained /47** | gate | strict retained /41 |
|---|---|---|---|---|---|
| **D3** transitions only, 1,536 rows | 289 / 289 / 289 | 58–71 | **35–37** | **2/3 PASS** | 26–27 |
| **M1** 768 transitions + rehearsal | 255–270 | 69–74 | **38–40** | **3/3 PASS** | **29–32** |

Source baseline: 14/384 new skill, 47/150 panel.

**The retention distributions are disjoint.** M1's worst seed retains 38; D3's best
retains 37. Under the stricter policy the separation is wider still — M1 29–32
against D3 26–27, with no overlap. The effect is not an artifact of the permissive
scorer, which was registered as prediction 4 and is the thing most likely to have
explained a panel gain.

**Every M1 seed clears every transition floor.** `ledger_replay` 100–112/128,
`prerequisite_plan` 83–88/128, `register_machine` 68–73/128, against floors of
0.2422 / 0.0703 / 0.0547.

## The three registered comparisons

**M1 vs D2 — isolates rehearsal at fixed transition dose.** Identical transition
rows; rehearsal added. New skill −8, panel +6, retained +5, and a gate **failure
becomes a pass**. Eight transition answers buy five retained answers and the gate.

**M1 vs D3 — isolates how to spend a fixed budget.** Same total compute, half of it
rehearsal. New skill −21 (289 → 268 at seed 17), retained +3, and 2/3 → 3/3 on the
gate. Spending half the budget on old data buys reliability.

**M1 vs D1 — the result that matters.** D1 cut dose to reach retention 40 and paid
for it with new skill 239. M1 reaches the same retention 40 at **268**. At equal
retention, rehearsal recovers **29 transition answers that dose reduction throws
away.** The frontier moves outward rather than sliding along itself, which is what
distinguishes a working intervention from a cheaper operating point.

## What made the test honest

The rehearsal source is the v3 preservation slice: 14,000 rows, ids
`rineya_grnc_v2_*`, **zero id overlap** with the 150-row panel. But 750 of its rows
carry one of the panel's own three `reply_class` values. **Those 750 are excluded**,
leaving 13,250 rows over 59 classes, further restricted to the 10,353 single-turn
rows so everything compiles through the already-qualified path.

Without that exclusion, a panel gain would be indistinguishable from practising the
tested task. With it, the gain is transfer from other classes. `process_mix.py`
refuses a rehearsal source in which the exclusion drops nothing, so the guard
cannot silently become a no-op.

The rehearsal rows are never scored and never enter any instrument. The training
probe is drawn from the transition half only.

## Limits

Three seeds vary **the transition draw only** — the rehearsal draw is identical
across them, so this establishes that the rehearsal benefit is robust to which
transitions you pick, not that it is robust to which rehearsal rows you pick. That
is a narrower claim than "rehearsal replicates" and is stated as such.

The mixture ratio was fixed at ~50/50 before any arm ran and **was not swept**. A
better ratio may exist; finding it by trying several and reporting the best is the
failure mode `RRP009_MIXTURE_PREREG.md` forbids, and no such sweep was run.

Retention here means one 150-row panel covering 3 of v3's 39 reply classes, plus
the 47 source reference answers. It is not general capability. The strict
`G1_RETENTION` zero-loss-per-class rule still **fails** on every arm in this
programme, M1 included; D3 of `ADJUDICATION_001.md` chose the softer gate before
any number existed, and clearing one is not clearing the other.

No release or quality claim attaches. Every run records
`status: MICROFIT_EXECUTED_NOT_QUALITY_CONFIRMATION` and
`quality_improvement_established: false`.

## The recommendation

**Rehearsal at ~50/50, at the D2 transition dose.** It is the only configuration
measured here that clears the gate on all three seeds, and it does so while scoring
higher on the new skill than the dose-reduction alternative that reaches the same
retention.

Stated plainly: *if you want the model to learn this skill and keep what it had,
spend half your budget showing it what it already knew.* That is cheaper than it
sounds — M1 costs the same GPU as D3 and less than D4.

## Reproduce

```bash
PY=C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe
$PY -m experiments.reasoning_posttraining_v1.process_mix \
    --binding <records>/a1-prepared-003/SOURCE_BINDING.json \
    --views <records>/process-transition-probe-001/CANDIDATE_VIEWS.jsonl \
    --rehearsal Rineya_General_Reasoning_..._v3.0/data/sft/rineya_v1_v2_preservation_slice_14000.jsonl \
    --source Supa --per-family 256 --selection-seed 17 \
    --out-prepared <dir> --out-plan <plan>
```

Pre-registration in `forge-v2-prereg-001/RRP009_MIXTURE_PREREG.md`, written before
anything was built. Artifacts `PROCESS_rrp009_M1{,_s23,_s29}.json` and
`RETENTION_PROBE_rrp009_M1{,_s23,_s29}.json`. Checkpoints deleted after scoring per
`DISK_RECLAIM_001.md`; receipts, plans and corpora retained. Train every arm before
scoring any arm — see `docs/LANE_INTERLEAVE_HAZARD.md`.
