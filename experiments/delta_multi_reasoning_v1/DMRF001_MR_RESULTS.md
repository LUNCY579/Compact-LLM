# DMRF-001 — multi-reasoning repairs nothing, and the candidates are why

**CHECK split, n = 128, systems B (frozen Supa) and D (the promoted k=0.25 core). Four
inference conditions each. The five contrasts were predeclared in code and committed
(`1cc9a08`) before any CHECK outcome was read. Nothing was rejected under Holm-Bonferroni.
Long context and abstention remain report-only. Nothing selected, nothing promoted.**

Matched conditions on every pass: transformers 4.51.3 rendering with `enable_thinking=False`,
greedy, TF32 off, FP32, `max_context` 4096, **0 rows skipped for overflow** on all five passes
of both systems.

---

## The scores

| condition | B total | D total | transition (32) | multi_step (32) | arith (32) | longctx (32) |
|---|---:|---:|---|---|---|---|
| **O** ordinary | 36 | **48** | B 3 / D 14 | B 1 / D 1 | B 0 / D 1 | B 32 / D 32 |
| **T** token-matched | 36 | **48** | B 3 / D 14 | B 1 / D 1 | B 0 / D 1 | B 32 / D 32 |
| **REP** 4 tries + selector | 36 | **49** | B 2 / D 15 | B 2 / D 1 | B 0 / D 1 | B 32 / D 32 |
| **MR** 4 roles + selector | **32** | **48** | B 0 / D 14 | B 0 / D 1 | B 0 / D 1 | B 32 / D 32 |
| MR with S0 majority | 35 | 49 | B 2 / D 15 | B 1 / D 1 | B 0 / D 1 | B 32 / D 32 |

## The five predeclared contrasts

| contrast | repair | damage | net | world-clustered CI on mean per-item diff | p | verdict |
|---|---:|---:|---:|---|---:|---|
| **D: MR vs O** | **0** | **0** | **+0** | [+0.0000, +0.0000] | 1.000 | WITHIN_NOISE |
| **D: MR vs REP** | 0 | 1 | −1 | [−0.0234, +0.0000] | 1.000 | WITHIN_NOISE |
| **B: MR vs O** | **0** | **4** | **−4** | [−0.0625, −0.0078] | 0.125 | WITHIN_NOISE |
| **Delta × MR** | — | — | **+4** | [+0.0078, +0.0625] | 1.000 | WITHIN_NOISE |
| **D: T vs O** | 0 | 0 | +0 | [+0.0000, +0.0000] | 1.000 | WITHIN_NOISE |

**Holm-Bonferroni over exactly those five: nothing rejected.** Smallest p is 0.125 against a
0.010 threshold.

---

## Finding 1 — on the delta candidate, MR is the identity

`D: MR vs O` is **0 repairs and 0 damage over 128 items**, and the world-clustered interval is
[0, 0] because the per-item difference is exactly zero everywhere. MR did not merely fail to
help on net; **it selected the same answer as the ordinary solve on every single item.**

`D: T vs O` is also exactly zero. So the token allowance is not the story either: giving the
ordinary solve MR's entire 392-token output budget in one call changes nothing.

The one condition that moved anything on D was **REP**, at +1 (one repair, no damage) — four
paraphrases of the *same* operation, which is the opposite of the differentiated-reasoning
hypothesis. And `D: MR vs REP` is −1, so MR is behind four tries at the same thing.

## Finding 2 — the roles, not the selector, are the binding constraint

This is the part that generalises beyond this backend.

| system | R0 | R1 | R2 | R3 | any role correct | best single | **role oracle headroom** |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 36 | 27 | 35 | 13 | 36 | 36 | **0** |
| D | 48 | 47 | 48 | 47 | 50 | 48 | **2** |

`uniquely_solved_by` is **{R0: 0, R1: 0, R2: 0, R3: 0}` on D** — not one item was solved by a
single role alone. On B the oracle headroom is **exactly zero**: every item any role solved,
R0 had already solved.

So there were **2 items out of 128** available to any selector on D, and **none** on B. A
perfect label-aware chooser over these candidates would score 50 and 36.

And the selector was not the problem. Measured utilisation on D is **0.960** — it captured 96%
of the cases where a correct candidate existed — with `selector_malformed_mr = 0`. On B it was
0.889. The machinery worked; there was nothing for it to find.

**The roles did diverge textually** — `ROLES_DIVERGE` on both systems, with all-roles-identical
text on 20% of B's items and 66% of D's. So this is not the name-only failure the check exists
to catch: the prompts genuinely induced different outputs. Those different outputs just landed
on the same answers, or on wrong ones. Differentiated *operations* produced correlated
*outcomes*.

## Finding 3 — the derived checker findings were worth nothing

The S1 diagnostic, on the prospectively fixed quarter-sample (n = 32): **13 correct with the
checker findings withheld, 12 with them shown.** Withholding was marginally better. 36 findings
were emitted on D, so the mechanism fired; showing them to the selector did not help. One row
on 32 items is noise, and the honest reading is that the findings carried no measurable value —
not that hiding them helps.

## Finding 4 — the interaction is positive for the wrong reason

`Delta × MR = +4` with a world-clustered interval of [+0.0078, +0.0625] that excludes zero.
Read naively that is "MR helps the delta candidate more than the frozen source", which would be
synergy.

It is not. The simple effects are **D: 0** and **B: −4**. MR did not help D at all; it *hurt B*
and left D alone. A positive difference-in-differences built from one zero effect and one
negative effect is **differential damage, not synergy**, and reporting it as an interaction
without its components would invert the meaning. This is exactly why `interaction()` returns the
simple effects alongside the contrast.

---

## What limits this result

Stated plainly, because it bounds the claim more than anything above.

**Two of four strata are saturated or at floor, so the effective measurement is narrow.**
`longctx_mixed` is **32/32 for both systems** — the ceiling I flagged when the gate passed, and
it cannot show movement in either direction. `arith_competence` (D 1/32) and `multi_step`
(D 1/32) are near floor. The real dynamic range is the **transition** stratum alone, where D
scores 14/32. So this is a 32-item measurement wearing a 128-item label, and the zero-repair
finding is strongest there and weakest everywhere else.

**One draw, one bank, one backend.** REP-001's 7-row acquisition spread means every net here is
WITHIN_NOISE by construction; the informative quantities are the *paired* counts (0 repairs) and
the *role oracle headroom* (0 and 2), neither of which is a noise statement.

**The backend was expected to be hostile to this.** `roles.py` recorded before the run that the
596M family is drilled to emit exactly one JSON field, and that asking it to audit premises was
asking for something its training never rewarded. The outcome is consistent with that, and it
is the reason this result should not be read as "multi-reasoning does not work" in general — it
is "multi-reasoning added nothing on a heavily format-constrained 596M backend whose four roles
produced correlated outcomes".

## Cost

| item | value |
|---|---|
| actual deduplicated calls per system | **1,312** |
| logical deployment calls per system | 1,536 |
| registered reuse | one `direct` generation serves O, REP[0] and MR R0 |
| S1 extra calls | 32 (accounted separately) |
| net additional correct cases | **0** on D, **−4** on B |
| cost per net additional correct case | **NOT REPORTABLE** — net gain is zero or negative, so a ratio would describe a benefit that was not observed |

The refusal in the last row is enforced by `analysis/paired.py::cost_per_net_correct` and
asserted in `tests/test_paired.py`.

## Answers to the driving questions

**"Does MR beat repeated direct attempts on the delta candidate?"** No. MR is −1 against REP,
and REP is the only condition that gained anything (+1 over O).

**"Are role successes complementary, and can the real selector exploit them?"** They are not
complementary: role oracle headroom is 0 on B and 2 of 128 on D, with no role uniquely solving
anything. The selector could exploit what little existed — utilisation 0.960 — so the selector
is exonerated and the candidate generator is not.

`quality_improvement_established` false. `release_permitted` false. Nothing promoted.
