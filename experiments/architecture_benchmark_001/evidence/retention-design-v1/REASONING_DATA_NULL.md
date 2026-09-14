# Reasoning data does not improve this model's reasoning, at any dose tried

The one untested lever for general capability was training data that exercises reasoning rather
than conversation. Chunked cross-entropy lifted the ~250-token ceiling that made this impossible,
so it could finally be asked. The answer is no.

## The measurement

B28last20 + WSD, mixed splits of Rineya Reasoning Forge rows (<=1,024 tokens) interleaved with
`splits_large` conversational rows. All contamination guards clear on both splits.

                          forge tokens  rows   panel   forge IID        forge transfer
    source                           0     0      47   32/200 t199      44/200 t184
    splits_large only                0     0      74   30/200 t186      36/200 t178
    mixed 30%, 25k budget        7,856    10      48   34/200 t191      34/200 t181
    forge 75%, 119k budget      90,208   121      63   34/200 t168      30/200 t157

Paired McNemar against the source:

    forge75 IID       acc 17-19, disc 36, p=0.8679   term 31-0, disc 31, p~0
    forge75 transfer  acc 23- 9, disc 32, p=0.0201   term 27-0, disc 27, p~0

Floors: IID accuracy 7-12 discordant, termination 6-11; transfer accuracy 5-10, termination 4-9.

## What it says

**11.5x the reasoning data produced identical IID accuracy.** 34/200 at 7,856 forge tokens and
34/200 at 90,208. Against the source that is 17-19 discordant at p=0.868 -- a perfect null, with
discordance well above the floor, so this is demonstrated equivalence rather than an
underpowered result. The +2 observed at the smaller dose was noise, exactly as its own null test
said at the time.

**Transfer accuracy got significantly worse.** 30/200 against the source's 44, 23-9 discordant,
p=0.0201. More reasoning data made out-of-distribution reasoning worse, not better.

**Output discipline collapsed.** Termination 168/200 on IID and 157/200 on transfer, against the
source's 199 and 184, at 31-0 and 27-0. Far worse than any `splits_large`-trained model, and the
degradation scales with the forge share: 191 -> 168 on IID as forge goes 30% -> 75%.

So the forge-heavy model is worse on every axis except the panel, where it is worse than
`splits_large` alone (63 against 74).

## Why this is a capacity statement, not a data or method statement

The data is clean: procedurally generated, independently solver-verified, zero overlap with the
panel (0/150 verbatim and blinded), and its evaluation splits are disjoint from its training
split by construction and by checked id. The method is sound: the loss was verified numerically
equivalent to the original before use (9.5e-07 on value, 3.3e-06 on gradient), and the run
completed all 347 steps with the device-integrity control passing.

What did not happen is the model learning the skill. A 596M model at 16% on this benchmark did
not move after 90k tokens and 121 worked examples of it. That is consistent with the task being
out of reach at this scale rather than with the exposure being insufficient -- the dose went up
11.5x and the accuracy did not move a single row.

## A framing error worth recording

I predicted the panel would collapse further at 75% forge share "because only 25% of tokens
remain conversational". It did not: 63 against the 30% run's 48. The panel tracks **absolute**
conversational tokens, not their share -- the 75% run carries ~30k conversational tokens against
the 30% run's ~17k, more in absolute terms despite the smaller fraction. Reasoning in shares
where the quantity is what matters; I did this several times before noticing.

Note also that `splits_large` alone at 5,200 tokens still beats both mixtures on the panel (74),
so panel performance is not monotonic in conversational volume either. It tracks dose position,
and 5,200 sits near the optimum found in the earlier sweep while the larger runs are past it,
showing the same peak-then-decline (70 -> 63 here, 74 -> 64 there).

## What would still be worth trying

Not more of this data. The dose-response is flat at zero across an 11.5x range.

The remaining candidates are a larger model, or a task the model is closer to being able to do.
Neither is reachable on this host: the model is fixed at 596M, and finding a reasoning benchmark
this model is on the cusp of would be a data-construction project rather than an experiment.
