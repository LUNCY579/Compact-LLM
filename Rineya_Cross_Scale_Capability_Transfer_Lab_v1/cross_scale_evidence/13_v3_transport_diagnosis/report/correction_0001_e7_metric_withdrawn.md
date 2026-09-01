# Correction 0001 — E7's capability verdict is withdrawn

Raised: 2026-09-01, while assessing a proposal to retarget the study at Aegis.

Corrects: `e7_answer_likelihood/` and the E7 section of
`v3_1_donor_side_report.md`. Those files are left in place; this record states
what is wrong with them, per the amendment policy that corrections do not
overwrite.

## What E7 concluded

> Qwen3-8B beats Supa on 37 of 55 cases, sign test p = 0.015 — a real advantage.
> But the donor wins broadly by almost nothing and loses catastrophically where a
> terse format is required. That is a formatting and calibration gap running in
> the target's favour, not a knowledge gap.

Both halves of that are wrong.

## Why the significance was not real

E7's sign test counted every non-zero difference, and the metric is saturated:
17 of 55 cases have **both** models above −0.01 log-probability per answer token.
Excluding near-ties dissolves the result:

| tie threshold excluded | 8B beats Supa | sign test p |
|---|---:|---:|
| none (as published) | 37/55 | **0.015** |
| \|delta\| <= 0.001 | 32/50 | 0.065 |
| \|delta\| <= 0.01 | 20/38 | **0.871** |
| \|delta\| <= 0.1 | 10/28 | 1.000 |

The published p-value is an artifact of counting ties as wins.

## Why the "format gap" story is backwards

The family's own frozen Showcase64 bank scores the donors on the *same* rows
under the *same* greedy / 24-token / 4096-context / seed-17 / thinking-off
protocol (`comparison.md`, "Reference: official Qwen3 scale"):

| model | Showcase64 | mean words |
|---|---:|---:|
| Aegis 1.1 | 4/64 | — |
| Qwen3-0.6B | 14/64 | 4.41 |
| Supa 1.7 | 40/64 | — |
| Qwen3-4B (NF4) | 43/64 | 3.59 |
| **Supa-A1.1 (the actual study target)** | **45/64** | — |
| **Qwen3-8B (NF4)** | **53/64** | **2.50** |

So on a real task-level scorer there is a gap in the 8B's favour, monotone in
donor scale, and the 8B is the **tersest** model in the table — the opposite of
E7's premise that the donors are penalised for being verbose.

**Amended 2026-09-01.** This section first quoted Supa 1.7 at 40/64 and called
the gap 13 points. That is the wrong model. Every stage of this study — v1, v2,
v3 and v3.1 — used `Rineya-Supa-A1.1-1.1.0-devfreeze` as the target, and its own
`CORE.json` claims **45/64** Showcase. Against the real target the gap to
Qwen3-8B is **8 points**, not 13.

Two consequences follow. First, **Qwen3-4B at 43/64 is below the target** and
should be retired as a donor; that retrospectively explains v1's H7, the 4B's
poor showing in E7, and its failure to beat controls in v2. Second, the gap
estimate is soft in both directions: A1.1's own card labels its 45/64 a
"tuning-split checkpoint evaluation" and claims no statistical significance,
while the 8B's 53/64 was measured at NF4 4-bit and was never selected on this
bank. Those biases run opposite ways, which is precisely why the confirmatory
re-run below is needed before anything is built on the number.

Two features of that table make the gap conservative rather than generous. The
donors were run at NF4 4-bit while the family cores were bf16/float16, and
`FAMILY_PERFORMANCE_AND_RESEARCH.md` records that Showcase64 "was used
repeatedly for selection" for family cores but never for the donors. Both
handicaps run against the donors.

## Why E7's instrument failed

Teacher-forced log-probability of one reference string measures agreement with
that string, not task success. The frozen scorer in
`src/rineya_cross_scale/evaluation/scoring.py` normalises and casefolds, so
`"F"` passes where E7 forced the token `"f"`. The `short_response` generator
in `banks.py` prints the prompt with `word.upper()` and stores `word[0]`
lowercase, so E7 charged every model for a capitalisation the real scorer
ignores. Four families are almost entirely surface form, and they carry the whole
effect: removing `short_response`, `common_sense`, `science` and `mathematics`
moves the Supa-minus-8B margin from **+2.530** to **−0.044** over the remaining
39 cases.

`FAMILY_PERFORMANCE_AND_RESEARCH.md` section 4.1 had already recorded the
general form of this failure — "answer log probability ... did not improve
Showcase accuracy. The model is often confidently wrong." E7 should have cited it.

## A separate defect found in the same review

The Aegis dry run compared donors by their lift over a target. That statistic is
target-invariant by algebra:

    mean(A - T) - mean(B - T) = mean(A) - mean(B)

Verified numerically: the dry run's Supa-over-8B lift on Aegis is
`+2.530126870`, and E7's already-published `qwen8b__vs__supa` mean delta is
`-2.530126870`. Substituting any target changes one additive constant and no
ranking. The dry run could not have produced new information about donor
ranking, whatever target was chosen.

## What this changes

- **E7's verdict is withdrawn.** The claim "no meaningful capability gap to
  transfer" is not supported, and the evidence points the other way.
- **v3 and v3.1's transport findings are unaffected.** E1/E1c (the shape-matched
  null), E2/E3 (target-basis alignment), E5 (the CKA atlas) and E6/E6b (the
  translator and its redundancy result) do not depend on E7. The P4/P5 operators
  remain characterised, and the withdrawn 0.932 statistic stays withdrawn.
- **The one conclusion that must be re-opened** is the strategic one: v3.1 read
  outcome D partly from "there is little distinctive signal to send through the
  channel." An 8-point Showcase gap against the real target is a distinctive signal. The channel
  result (E6: correspondence exists at cosine 0.982 but is not low-rank, and
  donor representations are redundant *for predicting Supa's representations*)
  stands on its own, but it no longer licenses "there is nothing to transfer".

## Next experiment this justifies

Re-score Qwen3-8B and Supa-A1.1 on the frozen Showcase64 rows with the family's own
committed scorer, in one controlled run, at matched storage precision, both
`enable_thinking` settings. 64 prompts x 2 models x 2 flags, no new tooling. It
is the gate on everything downstream: if the 8-point gap reproduces under one
scorer and one precision, a real capability gap exists and the transfer question
is live again — with **Supa-A1.1** as the target, not Aegis. Qwen3-4B is retired
as a donor either way, since it scores below the target.

If it does not reproduce, the line closes on the transport evidence alone.

## On the Aegis proposal specifically

Not supported, for a reason independent of the above. `qwen3_replace_aegis` in
`Rineya_Qwen3_CrossDepth_Soup_Mini_v1/outputs/scoreboard.md` already puts
Aegis's 19 decoder blocks verbatim back into the official 28-layer scaffold and
scores **23/64 against Aegis's 4/64**, with no donor, no fitted map, and exact
shape match. Aegis's deficit is nine missing layers of serial computation, and
the repair is a free within-family weight copy. Retargeting there would
rediscover Halo at higher cost, and untuned Qwen3-0.6B already scores 14/64.
