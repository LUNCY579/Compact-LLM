# The panel's noise floor is 15-24 discordant rows, and it invalidates several conclusions

R1, pre-registered before any score was seen: run one arm at four seeds and measure the spread.
`B28last20`, lr 1e-5, 5,200-token budget, cosine, identical in every respect except the seed.

    seed 17   67/150        spread   9 rows
    seed 23   58/150        mean  61.8
    seed 29   60/150        sd     3.9
    seed 41   62/150

Same-arm seed-pair discordance -- pure noise, since the configuration is identical:

    seed17 vs seed23   b=12 c= 3   15 discordant   p=0.0352
    seed17 vs seed29   b=14 c= 7   21              p=0.1892
    seed17 vs seed41   b=11 c= 6   17              p=0.3323
    seed23 vs seed29   b= 7 c= 9   16              p=0.8036
    seed23 vs seed41   b= 8 c=12   20              p=0.5034
    seed29 vs seed41   b=11 c=13   24              p=0.8388
                                   min 15, max 24, mean 18.8

**A same-configuration pair reached p = 0.0352.** The programme's alpha of 0.05 therefore
produces false positives from the seed alone. Any single-seed result with p between about 0.01
and 0.05 and fewer than ~24 discordant rows must be treated as unproven.

The seed is not a reshuffle. At a 5,200-token budget a run consumes about 41 of 10,000 TRAIN
rows -- 0.4% of one epoch -- and seeds 17 and 23 draw sets with **zero overlap**. The seed
selects which data the model sees.

## The pre-registered rule had a hole, and it did not matter

The rule read: spread <= 5 rows AND discordance < 10 keeps contrasts viable; spread >= 10
cancels low-power contrasts. Observed spread is 9, which falls in an undefined gap. The
conjunctive first branch settles it regardless: discordance is 15-24 against a required < 10, so
branch one fails outright and the gap never needed adjudicating. Recording this because writing
a rule with a hole and then choosing a branch after seeing the data is exactly the failure the
pre-registration exists to prevent.

## What survives

| comparison | discordant | p | status |
|---|---:|---:|---|
| B28last8 ~ B28 full | 36 | 0.0001 | survives |
| B28last12 ~ B28 full | 35 | 0.0041 | survives |
| B28last8 ~ B28last20 | 32 | 0.0041 | survives |
| P+WSD ~ H+WSD | 31 | 0.0107 | survives |

Two findings stand. **Depth matters** -- training more blocks genuinely helps, on three
independent pairs all well clear of the floor. **Cross-position shared memory harms** -- the CED
arm whose read may attend across positions is beaten by the same-token control at equal
parameters, 31 discordant rows.

## What does not survive

| comparison | discordant | p | why it fails |
|---|---:|---:|---|
| B28last20 ~ B28 full | 22 | 0.5726 | a null AT the floor; underpowered, not equivalent |
| B28last16 ~ B28last20 | 16 | 0.0251 | inside the band where same-config pairs hit p=0.0352 |
| B28last8 ~ B28last12 | 9 | 0.5391 | below the floor |
| Hlast4 ~ B28last8 | 4 | 0.6250 | far below the floor |

**The efficiency claim is withdrawn.** "B28last20 matches full B28 at 53% of trainable
parameters" was reported as a result. It is a null result at the noise floor. "Not
distinguished" is not "equivalent", and at 22 discordant rows against a mean noise of 18.8 the
comparison carries almost no information. The depth *ordering* survives; the specific claim that
20 blocks suffices does not.

## Consequences for planned work

Cancelled, because the effects are smaller than the floor:

* the two-ingredient weight soup (B28 seed 17 at 73 and seed 23 at 58) -- an average differing
  from either ingredient by 3-5 rows cannot be separated from re-running one of them;
* the `union` and `delta` depth merges, for the same reason;
* any further single-seed arm comparison expected to yield under ~24 discordant rows.

The instrument, not the ideas, is the binding constraint. 150 rows with a 15-24 row noise floor
cannot resolve the differences now being chased. The next investment is a better-powered
evaluation: `rineya_reasoning_forge_v1` carries 16 families, 800 IID test rows, plus dedicated
transfer and length-generalization splits, against this panel's 150 rows and no generalization
split at all. Its floor must be measured the same way before it is trusted.
