# Requalification review: what survives the bug fixes, what does not

Written after finding three bugs in my own harness plus one eval-design error. The question
here is narrow and mechanical: **for every result produced in this programme, is it
contaminated, and does it need re-running?**

Nothing is assumed safe. Each row below was checked against the specific bug, not waved
through by category.

## The four defects

| # | defect | mechanism |
|---|---|---|
| **B1** | the seed did nothing | fixed data order, fixed weights, no dropout, greedy decode — nothing consumed the seed, so "3 seeds" would have been one run reported three times |
| **B2** | leading-integer scoring | compared only the first integer, so `10:55 AM` matched `10:25 AM.` and `62.9` matched `62.5` |
| **B3** | a print could kill a run | non-ASCII model output + cp1252 console killed three C28 runs *after* their compute |
| **B4** | eval chosen without checking the teacher | `arithmetic_chain` picked as the retention probe, but the frozen source scores **0/20** on it |

## Verdict per result

| result | affected by | status |
|---|---|---|
| **Parity / numerics** | | |
| Strict FP32 parity, all arms (401,863 / 726,228 violations) | none | **stands** — no scoring, no seed, no generation |
| FP64 lane (3.8e-13 / 4.1e-12) | none | **stands** |
| Per-layer divergence probe, encoder layer 0 onward | none | **stands** |
| HOTFIX-002 read-precision falsification | none | **stands** |
| Displaced-layer law (CED 13 → 12.5854, local 14 → 12.9136) | none | **stands** — CE only, no task scoring |
| CED depth sweep (N = 1…14) | none¹ | **stands** |
| **Arm quality at initialization** | | |
| CE / perplexity / KL / top-1 for B28, D40, C28, C40 | none | **stands** — teacher-forced CE, no scorer involved |
| Generation: B28 6/24, D40 0/24, C28 0/24, C40 0/24 | none¹ | **stands** |
| **Gates** | | |
| Ladder requalification at 5 digests, all inert | none | **stands** |
| `G1_QUALITY` B28 PASS / D40 FAIL (3 checks after HOTFIX-004) | none | **stands** |
| **Training** | | |
| 2×2 matched comparison (`recovery_pilot`, lr 1e-4 and 1e-5) | B2? | **stands — verified, not assumed**² |
| Source baseline by class (`baseline_scan`) | **B2** | **RE-RUN, done** — four classes were inflated by one row each |
| Multi-seed runs (`rigorous_pilot`) | **B1 + B2 + B3** | **DISCARDED**, retained as bug evidence; re-running |

¹ Generation in these used *strict full-string* comparison, not leading-integer, so B2 does
not apply. Strict matching can only under-count, and the arms scored here answered in bare
form (`3`) or produced garbage, so the counts are exact either way.

² **Checked rather than assumed.** B2 only bites when an answer contains more than one number
or a decimal. All 24 `pilot.jsonl` answers are single bare integers (`3`, `5`, `7`, `9`, `11`,
`8`, …) — measured: zero answers with a decimal or a second number. So leading-integer and
full-number scoring are *identical* on that set, and the 6/6 vs 0/6 finding is unaffected.

## What was re-run, and what changed

`baseline_scan` re-measured with the corrected scorer:

| class | lenient | corrected |
|---|---:|---:|
| percentage_reasoning | 70% | **65%** |
| ratio_proportion | 50% | **50%** |
| time_duration | 35% | **30%** |
| estimation_sanity | 35% | **30%** |
| sequence_pattern | 10% | **5%** |
| arithmetic_chain, counting_sets, unit_conversion, comparison_ranking, schedule_ordering | unchanged | unchanged |

Four classes were inflated by exactly one row each — real, and small at n=20.

### The probe selection was NOT re-fitted, deliberately

My stated rule was "classes above 30%". Under the corrected numbers `time_duration` sits at
**exactly 30.0%**, and `estimation_sanity` — which was not in the original selection — also
sits at 30%. Re-selecting now would be fitting the probe to the corrected measurement, which
is the same error as moving a gate threshold after seeing an arm's score.

So the probe stays as chosen: **percentage_reasoning, ratio_proportion, time_duration**. The
justification does not depend on the knife-edge: the corrected rates cluster cleanly as
{65, 50, 30, 30} against a next tier of {10, 10, 5, 5, 0, 0, 0}, so all three remain far above
the floor. `estimation_sanity` is excluded on its own merits as well — only 17/20 of its
answers are numeric and they are hedged (`About $60.`), which makes exact matching unreliable
rather than merely strict.

**Consequence: the splits do not need rebuilding.** The probe classes are unchanged, so
`train.jsonl`, `dev.jsonl` and `task_arithmetic.jsonl` remain valid.

## Structural fix, so this class of bug cannot recur

The scorer was duplicated across three scripts and the copies drifted — first too strict
(full-string, which inverted a conclusion), then too lenient (leading-integer, which inflated
task scores). There is now exactly one implementation in
`experiments/rab001_exploration/scoring.py`, imported by every consumer, with an 11-case
self-test that **runs at import** so a drifted or broken scorer fails loudly instead of
silently producing numbers.

All twelve exploration scripts now also reconfigure stdout/stderr to UTF-8, so output
formatting can never again destroy a completed measurement.

## Honest note on how these were found

None of the four defects was found by code review. B1 surfaced because two seeds produced
bit-identical output; B2 because a trajectory log showed correct answers while hits read
zero; B3 because runs died with compute already spent; B4 because a smoke test showed the
teacher at 0/20. **Exact controls and cheap smoke tests found every one of them**, and three
separate adversarial-review workflows produced nothing, all dying on usage limits.

That is the operational lesson worth keeping: run the identity control, run the two-step
smoke test, and look at the raw samples — before trusting any aggregate.
