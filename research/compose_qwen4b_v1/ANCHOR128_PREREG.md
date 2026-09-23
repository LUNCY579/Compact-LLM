# ANCHOR-128 — the 128-token island has EIGHT arms and NO baseline

> **STATUS: REGISTERED, NOT RUN.** Verified at this line: zero source/baseline artifacts exist on
> bank `7dd53f2a` at `max_new_tokens 128`. The only baselines on that bank are at **96**.

**Class: INSTRUMENT ANCHOR. One leg, ~270 s, no arm built, no model trained.**
**Deliberately NOT part of DPACQ-001.** DPACQ-001 registered four rungs and it will run exactly
four; folding an unregistered fifth leg into a live battery is how a registered scope quietly
becomes whatever was convenient. This is its own measurement with its own file.

---

## 1. The gap

Bank `7dd53f2a` carries two decode budgets and they are not comparable:

| `max_new_tokens` | arms | baseline present? |
|---:|---:|---|
| 96 | 70 | **yes** — `rrp015_supa` **14/384** and `sv001_supa` **14/384**, two independent artifacts agreeing exactly |
| **128** | **8** | **NO** |

**Every acquisition number in the 128 family is currently quoted without a source anchor at its
own decode budget** — the four dense rungs, the LoRA anchor, the extrapolated arm, both binding
arms, and now the DPACQ rungs. That is a gap for interpreting all of them, not a detail of one.

## 2. Why it cannot be filled by borrowing

The obvious move is to quote the 96-token source (14) against a 128-token arm. **That is exactly
the borrowed-scale error both this lane and a peer's amended their documents for within the last
hours**, and doing it here would be the third instance.

**And the direction of the bias is genuinely unknown.** More decode budget gives more chances to
emit a canonical answer *and* more room to break canonical form. Neither of us has a measurement
of which dominates, so **not even the sign of the correction can be assumed.**

## 3. What the anchor adjudicates — a live disagreement, stated before it is resolved

Two readings of DPACQ rung 1 (**52/384**, clearing 1 of 3 family floors) are on the table, and
**they differ only in where the source sits**:

| reading | requires | whose |
|---|---|---|
| *"β = 0.155 is a near-no-op; the arm is at source level"* | source at 128 ≈ 50 | the peer's |
| *"a no-op would score like the source; this is worse than a trivial predictor, so it is degraded"* | source at 128 ≈ 14 | **mine** |

> **A third possibility neither of us named, and it is the one the peer surfaced:** if the source
> at 128 is near 14, then rung 1 at 52 has **ACQUIRED ~38 views while failing two family
> floors** — a stranger object than either reading, and not describable as no-op *or* as
> degraded.

**Registered now: I will not write "no-op" or "actively degraded" about rung 1 until this leg
exists.** What the floor comparison supports is exactly *"fails 2 of 3 family floors where the
dense arm at identical norm clears 3 of 3"*, and that is the whole of what it supports.

## 4. The measurement

```
process_probe.py --model <Rineya-Supa-1.7.0-devfreeze>
                 --label supa_select_mnt128
                 --device cuda --instrument select-views-001 --max-new-tokens 128
```

Same probe, unmodified, same `implementation_digest a329eb7bb06f` as all eight arms.
**No prediction is registered.** An anchor is not a hypothesis test; predicting it would invite
reading a number as confirming something. The single registered commitment is the **use**:

> The resulting value becomes the source baseline for the 128-token family, and **any acquisition
> claim on that family made after it exists must quote it.**

## 5. Gates

1. Bank `views_sha256 7dd53f2a…`, `views 384`, `max_new_tokens 128`, digest `a329eb7bb06f` —
   asserted on the artifact after writing, as for every DPACQ rung.
2. **Create-only**: `process_probe.py:263` refuses rather than skipping.
3. Per-family floors recomputed from the frozen views before any model loads.
4. **Runs only after DPACQ-001's four registered rungs complete**, so it cannot be mistaken for
   part of that battery.

## 6. What it does NOT license

- **Not a retro-fix for 96-token comparisons.** Arms at 96 keep the 96 baseline.
- **Not a correction factor.** It does not license rescaling any 128 number by a 96/128 ratio;
  the two families stay separate.
- **Not a DPACQ result.** DPACQ-001 has four rungs and this is not one of them.
