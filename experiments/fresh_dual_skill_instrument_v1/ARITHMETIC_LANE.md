# The arithmetic lane — what it needs, and the trap in it

The transition lane (Titan-Sparse's donated skill) is instrumented: procedural rows, an oracle
validated 384/384, exact-match scoring. The arithmetic lane (Opallo's donated skill) is **not**,
and this records exactly why, so the next attempt does not rediscover it.

## Why it is harder

Opallo's skill lives in three free-text classes — `percentage_reasoning`, `time_duration`,
`ratio_proportion` — with answers like `40.`, `$4.`, `10:25 AM.`, `90 minutes.` There is no
exact-string contract to score against, so the scorer is itself a research artifact.

## The trap: a count collision is not a validation

The recorded Opallo panel gives 73/150 under the `permissive` policy **and lists the 73
`correct_ids`**. I tried eight plausible scoring rules against the 150 recorded predictions:

| rule | n correct | overlap with the recorded 73 |
|---|---|---|
| exact / normalized / period-stripped | 61 | 61 |
| gold-as-substring | 68 | 64 |
| **first number equal** | **73** | **69** |
| last number equal | 80 | 69 |

`first_num_eq` **reproduces the headline 73 exactly** — and is wrong on 4 rows in each direction.
Had I validated on the count alone, I would have shipped a scorer that disagrees with the
recorded policy on 8 of 150 rows while looking like a perfect canary match.

> Validate an extractor on the **identity of the scored set**, not on its cardinality.

The eight disagreements show what the real policy does:

- accepts `$4.00.` for gold `$4.` — exact-rational value equality, not string equality
- accepts `1 hour and 30 minutes.` for gold `90 minutes.` — **unit conversion**
- rejects `10:45 AM.` for gold `10:25 AM.` — first-number agreement on the hour is not a match
- rejects `It ends at 4:10 PM.` for gold `4:10 PM.` — **format compliance**; a prose prefix fails

## The real scorer exists — use it, do not reimplement it

`ChatBubble-ced-native-loader-qualification/experiments/architecture_benchmark_001/retention.py`
implements this as an exact-rational, unit-aware, prefix-aware grammar (`parse_rational`,
`_parse_duration_seconds`, `_parse_clock_minutes`, `score_prediction`, `target_from_audit`). It
constructs no floats and documents its own limits at lines 17–30, including that it rejects valid
free-form English that means the right thing.

It is in another worktree. **Import it read-only; do not copy, fork or edit it.**

## What a fresh arithmetic bank still needs

1. **Typed targets, not gold strings.** `score_prediction` takes a typed target
   (`{kind: clock|duration|quantity|count|number, value: <rational>, ...}`) built by
   `target_from_audit` from an audited label row. A generated bank can emit typed targets
   directly, since the generator knows each answer's kind and exact rational value — this is
   easier than for the recorded bank, which needed a human audit pass.
2. **A `bare_policy` decision, declared up front.** `permissive` vs `strict` is the one
   reviewable axis and it moves the recorded score 73 vs 59. Pick and register it before scoring.
3. **A validation run before use:** score the 150 recorded predictions with the imported grammar
   and require the output set to equal the recorded 73 `correct_ids` **by identity**. If it does
   not, the harness is wrong — stop.
4. **Build gates of its own**, mirroring the transition lane: Supa low, Opallo high, separation
   large. The recorded reference is Opallo 73/150 permissive with Supa's source-correct n = 47.

## Consequence for the composition question

Until this lane exists, **no joint-composition claim is available from this instrument**. The
transition lane alone can show whether a composed arm preserves *Titan-Sparse's* skill; it cannot
show whether Opallo's skill survived alongside it. Both halves are required before
"acquires a new skill with barely any loss" can be tested rather than argued.

---

## Superseded before building — read `O_SKILL_DEFINABILITY.md` first (2026-09-23)

The premise that arithmetic is "Opallo's donated skill" does not hold. Opallo trained on 41
conversational rows and **zero** arithmetic-class rows, so its panel score is transfer.
Titan-Sparse scores 72/150 on the same panel against Opallo's 74, and the rows each donor gains
overlap far above chance (21 of 31, p = 7.8e-7). So does a delta trained only on abstention rows.
The Opallo-specific residue is about 5 rows, inside the seed floor. A bank built to this plan
would have an "Opallo high" gate that Titan-Sparse also passes. **Do not build it as specified.**
The one O-endpoint still definable is described in `O_SKILL_DEFINABILITY.md` §Consequences 4.
