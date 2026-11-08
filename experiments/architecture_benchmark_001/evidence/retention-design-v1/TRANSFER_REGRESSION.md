# Capability is preserved in-distribution and lost under distribution shift

`FORGE_NOISE_FLOOR.md` concluded that "no capability was lost" was established rather than
underpowered, because the accuracy comparisons cleared forge's 7-12 discordance floor and stayed
null. That conclusion holds for `test_iid` -- the split I happened to probe first -- and does not
survive the transfer split.

## The measurement

Four same-configuration WSD checkpoints (B28last20, 5,200 tokens, seeds 17/23/29/41) against the
frozen source, on both forge evaluation splits, 200 rows each, paired McNemar.

    split            source   seeds            pooled accuracy            pooled termination
    test_iid         32/200   30,27,32,30      37-28, disc 65, p=0.321    42-0,  disc 42, p~0
    test_transfer    44/200   36,38,41,40      48-27, disc 75, p=0.0203   26-9,  disc 35, p=0.006

The transfer split's own noise floor, from the six same-config seed pairs: **5-10 discordant,
mean 8.5** -- comparable to IID's 7-12, so the split is not simply noisier.

**All four seeds score below the source on transfer** (36, 38, 41, 40 against 44). On IID they
straddle it (30, 27, 32, 30 against 32).

## Why this was invisible until now

Every individual transfer comparison is null: p = 0.134, 0.286, 0.607, 0.454. A single run --
which is how every forge claim in this programme was made before today -- shows nothing. The
effect appears only when four independent draws are pooled, exactly as with WSD on the panel,
where 1 of 4 pairs was individually significant and the pooled test gave p=0.0003.

It was also invisible on `test_iid`, whose pooled accuracy test is null at p=0.321. Probing one
split and generalising was the error; `test_iid` is forge's in-distribution split and
`test_transfer` is its deliberate shift, and they answer different questions.

## Honest limits

* **Multiplicity.** Four pooled tests were run (2 splits x 2 metrics). Under Bonferroni at
  alpha = 0.0125, p = 0.0203 does **not** survive. What argues against it being a multiplicity
  artifact is consistency: 4/4 seeds below source, in the predicted direction, on a split with an
  independently measured floor. Reported as a real but modest effect that fails strict
  correction.
* The effect size is small: 44 -> 38.75 mean, about 5 rows of 200.
* Termination is lost on both splits, but far more severely in-distribution (42-0) than under
  shift (26-9), because the source itself terminates worse on transfer (184/200 against
  199/200).
* `test_transfer` is **easier** for the source than `test_iid` (44 against 32), so "transfer"
  here means a different distribution, not a harder one.

## What this does to the programme's conclusions

**Stands:** WSD is worth +7.5 panel rows, 4/4 seeds, pooled p = 0.00029. That is measured on the
panel and is unaffected.

**Corrected:** "no capability was lost" becomes "no capability was lost in-distribution; a
modest loss appears under distribution shift, pooled p = 0.020, not surviving strict
multiplicity correction". `TWO_INSTRUMENT_SUMMARY.md` and `FORGE_NOISE_FLOOR.md` both overstate
this and are superseded on that point.

**Sharpened:** the answer to "skill gains without catastrophic forgetting" is now three-part.
Gains on the narrow arithmetic panel: yes, replicated. In-distribution reasoning: preserved.
Out-of-distribution reasoning: a small but consistent loss. The third part only became visible
because a second evaluation split existed and four seeds were run on it.

---

## Addendum: the regression does not scale with trainable parameters, but output discipline does

Two further checkpoints on `test_transfer`, 200 rows, paired against the source:

    model                     trainable    exact    term    acc vs source
    source                            -   44/200  184/200               -
    SOUP of 4 WSD                314.6M   41/200  177/200   10-7, disc 17, p=0.629
    B28last20 wsd s29            314.6M   41/200  181/200    9-6, disc 15, p=0.607
    B28last20 wsd s41            314.6M   40/200  180/200   10-6, disc 16, p=0.455
    B28 full wsd                   596M   39/200  168/200   12-7, disc 19, p=0.359
    B28last20 wsd s23            314.6M   38/200  180/200   14-8, disc 22, p=0.286
    B28last20 wsd s17            314.6M   36/200  178/200   15-7, disc 22, p=0.134

**Accuracy does not scale with trainable parameters.** Full B28 updates 596M weights against the
tail arm's 314.6M and lands at 39, inside the tail arms' 36-41 range, with a paired result no
worse than theirs. The hypothesis that the shift regression would grow with the number of
updated weights is not supported.

**Output discipline does scale, sharply.** Full B28 against each tail arm on termination:

    vs wsd_s17   13-3, disc 16, p=0.0213
    vs wsd_s23   15-3, disc 18, p=0.0075
    vs wsd_s29   15-2, disc 17, p=0.0023
    vs wsd_s41   15-3, disc 18, p=0.0075
    POOLED       58-11, disc 69, p < 1e-6

Same-config termination floor on this split is 4-9 discordant (mean 6.5), so all four
comparisons clear it and all four are individually significant -- this one does not rest on
pooling. Training 596M weights costs roughly twice the output discipline of training 314.6M, on
identical data with an identical schedule.

**The soup remains inert.** 41/200 on transfer -- its best ingredient's score, not above it --
and 17 discordant against the source at p=0.629. Together with its IID null this is a clean
negative: uniform weight averaging does nothing measurable here on either axis. Two mechanisms
were predicted for it and neither held: that averaging disjoint data draws would cancel
draw-specific verbosity (it did not; the soup terminated worst of its group on IID), and that it
would restore robustness under shift, which is model soup's actual published claim (it did not;
it matched its best ingredient).

## Consequence for the deliverable

`B28last20 + WSD` and full `B28 + WSD` are indistinguishable on the panel (p = 0.8555) and on
transfer accuracy. They are NOT indistinguishable on behaviour: the full arm loses 16 rows of
termination against the source where the tail arm loses 4-6, at p < 1e-6 between them.

So the tail arm is preferred on evidence rather than on cost alone -- same measurable capability,
four more panel reference answers retained, markedly better output discipline under distribution
shift, and half the trainable parameters. That distinction was invisible this morning, when the
150-row panel was the only instrument and the two arms differed by two rows of noise.
