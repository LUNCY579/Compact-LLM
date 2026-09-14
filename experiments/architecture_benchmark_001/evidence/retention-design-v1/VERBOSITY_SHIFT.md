# It is not forgetting how to stop. It is a verbosity shift, and that changes the remedy.

`FORGE_TRANSFER.md` reported that three trained models each lost 13-16 rows of termination on
Rineya Reasoning Forge while recovering zero, and called it catastrophic forgetting. The
termination counts are right. The characterisation was wrong, and the correction matters because
it points at a different fix.

## What the generations actually show

Every forge probe stores its generations, so the failing rows can simply be read.

    answer : '160'
    source : 'UNKNOWN'                          terminated
    trained: '166 - 10 - 13 - 28 + 11 + 22 +'   did not terminate

    answer : 'venue=Observatory; format=email; day=Fri'
    source : 'Use Observatory for my venue.'                             terminated
    trained: 'Use Observatory for my venue. Use Friday for my day. Use ...'

    answer : 'moves=27; next=1:B->A'
    source : '0; NONE'                                     terminated
    trained: '1. Move 1: Move 1. Move 2: Move 2. Move 3 ...'

The trained model keeps going where the source stopped. Note the first row: it emits an
**arithmetic chain** on a task that is not arithmetic.

## The effect is global, not a tail

Output length over all 200 rows, against the source, per model:

    model                 longer   shorter   median chars   sign-test p
    B28last20 + WSD          160        32       7 -> 10      1.1e-21
    B28last20 cosine         163        29       7 -> 10      7.7e-24
    B28 full cosine          162        32       7 -> 10      4.1e-22

Roughly 160 of 200 rows got longer against about 30 shorter, in three independently trained
models. The 13-16 rows that stopped terminating are the tail of that shift, not a separate
failure of the stop-token mechanism.

## Why the original reading was wrong

Forge **accuracy is null** for all three models: p = 0.3018, 0.5078, 0.8036 on paired McNemar.
No reasoning capability was lost. What changed is response style: training on `splits_large`'s
36 conversational and reasoning classes teaches longer, work-showing answers. That helps on the
150-row arithmetic panel, where showing arithmetic is close to the target behaviour, and costs
exact match on a benchmark whose answers are terse categorical labels like `UNKNOWN`, `ON`,
`LIAR`, or a bare integer.

"Catastrophic forgetting" implies lost capability and points at architecture, dose, or replay.
None of those is the lever here.

## What this means for the remedy

* It is a **response-style** problem. Style-matched training data, a terse-answer instruction, or
  an inference-time length or stop-sequence control would all address it. Changing the
  architecture would not.
* It is still a genuine deployment regression. A model that rambles where it used to answer in
  one token produces visible noise, and 7 -> 10 median characters understates it at the tail.
* It does **not** undermine the panel gains. 47 -> 74 on the panel is real and survives the
  panel's noise floor against shallower arms. The model did not trade reasoning for verbosity;
  it gained panel capability and separately became more verbose.
* The panel cannot see this at all. Its own termination went 148/150 -> 150/150 for the same
  models, because verbosity is *rewarded* on an arithmetic benchmark that accepts worked answers.

## Standing caveats

n = 1 per checkpoint on forge and forge's own seed-noise floor is unmeasured, so the accuracy
nulls are consistent with noise rather than demonstrated equivalence. The verbosity effect needs
no such floor at p ~ 1e-22, reproduced three times.

The cheapest confirmation available, requiring no training: score the same checkpoints on forge
with a terse-answer system instruction or a hard length cap. If exact match recovers toward the
source's 32/200, style is confirmed as the whole story.

---

## Postscript: verbosity does not mask correct answers, so no inference-time fix exists

This document previously proposed the cheapest confirmation as re-scoring with a terse-answer
instruction or a hard length cap: if exact match recovered toward the source's 32/200, style
would be the whole story. That test can be answered from the stored generations without running
anything, and the answer is that it would recover **nothing**.

Across four models, over every forge row the model gets wrong under exact match:

    model                             wrong   answer is a PREFIX   equals first line   appears anywhere
    B28last20 + WSD                     170                    0                   0                  0
    B28last20 + WSD seed 29             168                    0                   0                  0
    SOUP of 4 WSD checkpoints           168                    0                   0                  0
    B28 full, cosine                    173                    0                   0                  0

Zero, on every measure. When these models are wrong the reference answer is not buried after a
preamble, not on a later line, and not present anywhere in the output. A stop-at-first-answer
rule, a length cap, and a terser instruction would each recover zero rows.

Three consequences:

* The verbosity shift is **cosmetic**. It is a deployment annoyance -- a model that rambles where
  it used to answer in one token -- and it costs no accuracy. That is consistent with forge
  accuracy being null in all six trained models, and with the earlier finding that on rows which
  stopped terminating the source was correct on zero of them.
* The models genuinely do not know these answers: 168-173 wrong of 200. No decoding change fixes
  ignorance.
* **Nothing has improved forge accuracy.** Every trained model scores 27-32 against the source's
  32. Training on `splits_large` improves panel arithmetic by up to +27 rows and leaves general
  reasoning exactly where it started.

The last point is the real ceiling of the current setup, and it is not a scoring artifact,
a schedule artifact, or a dose artifact. Improving general reasoning would require training data
that exercises it -- which on this host is blocked by the ~250-token sequence limit recorded in
`LONG_CONTEXT_LIMIT.md`, since forge rows run ~1,400 tokens median.
