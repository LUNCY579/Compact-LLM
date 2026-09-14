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
