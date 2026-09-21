# The training selection hash EQUALS the evaluation bank hash — and that is benign

**2026-09-20. Zero model calls, zero training.** Premise check, prompted by noticing that δ_H's
`selection_sha256` begins `7dd53f2a…` — the acquisition bank's prefix.

## What the collision is

The sha256 `7dd53f2a751b54c349c2ef571e8f9c5153737f4a1ed3df5eafe25f95980e78a3` is recorded **69
times** across the records, under two different field names:

| field | where | what it means |
|---|---|---|
| `selection_sha256` | `compose001/H`, `rep001/s23`, `rep001/s29`, `rrp015` — **training preparations** | the bank the training selection was drawn **against** |
| `views_sha256` | `binding-integration001`, `binding-mixture001`, … — **evaluation results** | the bank rows were **scored on** |

Located by content: it is `rineya-dataset-retention-records/select-views-001/SELECT_VIEWS.jsonl`,
found by hashing 17,860 candidate files rather than by trusting a filename.

**So every δ_T arm was trained against the same artifact its acquisition is scored on.** On its
face that is the contamination question for the programme's central number.

## It is not contamination. Verified four ways.

Both sides reduce to one namespace, `RRFR2-<hex24>`:

| comparison | views | train | shared |
|---|---:|---:|---:|
| `id`, prefix-stripped | 384 | 384 | **0** |
| stem (drop `-stepN-valid`) | 384 | 384 | **0** |
| views `parent_id` vs train stem | 384 | 384 | **0** |
| views `source_id` vs train stem | 384 | 384 | **0** |

> **384 evaluation views and 384 training rows, drawn from one pool, sharing no row and no
> parent.** `selection_sha256` records *which pool the draw was audited against*, not what was
> trained on. The acquisition numbers — 235, 219, 223, 304 — are generalisation.

This independently reproduces `two-disjoint-384-row-sets` on the **596M native lane**, where that
memory established it for the 4B LoRA lane. Same structure, separately checked.

## The part worth keeping: my first version of this check was VACUOUS

I first compared a stem of the view `id` against a stem of the train `id` and got 0, and nearly
stopped there. But view ids carry an `RRP-select:` prefix and train ids do not:

    view   RRP-select:RRFR2-00792c2005de482d5e6f4d27-step47-valid
    train              RRFR2-e1b510003ec367143ea26839-step2-valid

**The two sides could not have matched even if every row were identical.** The check was
guaranteed to return 0 — it measured the prefix, not the overlap, and it returned exactly the
answer I was hoping for.

> **A check whose two sides live in different namespaces is not a weak check. It is not a check.
> And it is indistinguishable from a passing one.**

This is the same failure mode as the `norm_combined` bug caught an hour earlier — a statistic
that looks right, is computed without error, and answers a different question than the one asked.
The canary rule says wrong extractors are caught by collision with a known number; **a
disjointness check has no known number to collide with. Its expected answer is 0, which is also
its failure output.**

**Mitigation, and it is cheap:** any disjointness check must first assert the two sides overlap
in *form* — same namespace, same id grammar, plausible cardinalities — and ideally include a
**positive control** (intersect each side with itself, confirm it returns n). Here: both sides
384, both `RRFR2-<hex24>`, self-intersection 384. Only then is 0 informative.

## Scope

Checked for **δ_T (s23)** and **δ_H** against `select-views-001`. `s29`, `rrp015` and the
HSKILL-VIEWS-002 instrument (`ba1f2982…`, δ_H's actual scoring bank, a different hash entirely)
are **not** checked here.
