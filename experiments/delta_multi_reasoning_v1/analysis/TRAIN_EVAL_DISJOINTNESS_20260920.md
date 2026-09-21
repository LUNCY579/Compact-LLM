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

---

# Amendment 1 — the namespace sentence is TRUE FOR δ_T AND FALSE FOR δ_H

**Raised by the paired session, and correct.** The document above says:

> *"both sides reduce to one namespace, `RRFR2-<hex24>`"*

**That describes δ_T's 384 training rows only.** δ_H's 648 rows carry **no `RRFR2` stem at all** —
they are `rehearsal:rineya_srb_v1_safety_boundary_00356` and siblings, from
`v3_preservation_slice`. Stem extraction returns **384 stems from the views and ZERO from δ_H's
train file.**

> **So the `views ∩ H_train = 0` line in the table above is VACUOUS — the exact defect this
> document was written to warn about, committed in the same document, one table below the
> warning.**

The T result stands unchanged: T and the views share the `RRFR2-<hex24>` namespace, a collision
was possible, and none occurred. **Only the H row was unearned.**

**The H case now has a real check**, run by the paired session with the positive control this
document asked for:

| check | result |
|---|---|
| *positive control:* inject a view's text into the H set → detected | **True** — a match IS detectable |
| exact normalized-text overlap | **0** |
| H prompts sharing a 20-gram with any view | **0 of 648** |

**Independently, on the HSKILL banks, this session ran the same check and it was live there:**
`HSKILL-CONFIRM-002` and δ_H's training set share **four id prefixes**
(`rineya_srb_v1_{safety_boundary, clarification, knowledge_boundary, uncertainty}`) from the same
corpus — so a collision was possible — and the overlap is 0 by id **and** by normalized user text,
with `CONFIRM ∩ CONFIRM = 224` as the positive control.

## What this costs the general lesson, and what it does not

The finding is unchanged: **train and eval are disjoint, and `selection_sha256` on a training
preparation records the pool the draw was audited against, not what was trained on.**

What changes is the confidence it was established with. **A namespace assertion written as prose
is not a namespace assertion.** I asserted the shared namespace for the pair I had inspected and
then extended the sentence over a second pair I had not — which is how the vacuous check survived
being written up in the document that defines it.

> **The positive control has to be CODE THAT RUNS PER PAIR, not a sentence about the pair you
> looked at.** Assert it inside the check, for each comparison, and let it refuse.
