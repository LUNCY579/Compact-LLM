# Confirmatory protocol — single-pass decode-time abstention gate

Written before any threshold was inspected. The sweep script refuses to select a
threshold on any split other than the one named here.

## Question

Can a threshold on a confidence signal extracted from **one** greedy forward
pass of `nimbus_low` recover most of the routed cascade's gain, removing the
multi-pass requirement?

`nimbus_low` = `Rineya-Nimbus-1.6.0-devfreeze` + halo Stage-A `checkpoint-200`
LoRA at scale 0.8125. Reference scores: 42/64 showcase, 121/192 stability.

## Generation contract (frozen, unchanged)

BF16, greedy, seed 17, Qwen thinking disabled, 24 max new tokens, 4096 max
context, truncation forbidden, runner
`Rineya_Long_Context_Demo_v1/scripts/run_local_hf.py`.

The confidence capture adds only `--emit-logprobs`, which stores the
already-computed per-step logits alongside each prediction. Decoding is
untouched. **Validated:** all 256 captured predictions, input-token counts and
generated-token counts are identical to the frozen baseline artifacts
(`nimbus_step200_scale0p8125_showcase64.jsonl`,
`nimbus_step200_scale0p8125_stability192.jsonl`).

## Signals (all from the single pass, no extra prefill)

Fifteen signals in four families:

- **first content token** — chosen-token logprob, top1−top2 logprob margin,
  top1−top2 probability margin, entropy
- **whole answer** — sum / mean / min token logprob, mean / min logprob margin,
  mean / max entropy (over content tokens, excluding EOS)
- **UNKNOWN mass at step 0** — logprob of the `UNKNOWN` token, log-sum over the
  abstention surface forms (`UNKNOWN`, `unknown`, `Unknown`, and the
  leading-space variants; each is a single token in this tokenizer, so the mass
  is exact, not a first-token approximation)
- **UNKNOWN mass relative to the greedy top-1**, for both of the above

## Gate

`prediction := "UNKNOWN"` when the gate fires, otherwise the model's own text.
The gate can only add abstentions; it never overwrites an existing abstention
with an answer.

## Threshold selection — tuning split is `showcase` (64 rows) ONLY

Candidate thresholds are the midpoints between consecutive observed values on
the tuning split, plus the two out-of-range endpoints. That enumerates every
partition the signal can express, and never pins the operating point to a single
tuning observation.

**Primary selection rule (single signal only), in order:**

1. maximise strict-exact correct answers on showcase64
2. tie-break: fewest added abstentions (least behavioural change)
3. tie-break: highest abstention precision
4. tie-break: signal name ascending, then threshold ascending

Two-signal AND/OR conjunctions are swept and reported as **exploratory only**.
They do not select the primary gate.

## Confirmation — `stability` (192 rows), evaluated ONCE

The selected gate is applied to the stability capture exactly once, with no
re-tuning. Reported for both splits:

- correct / total, from the published `evaluate_predictions.py --key`
- abstention precision and recall
- discordant pair counts `b` (gate-only correct) and `c` (baseline-only correct)
- **exact two-sided McNemar p-value** against `nimbus_low`

## Frozen gates (written after the showcase sweep, before stability was touched)

Three gates, and only these three, are carried to stability192. No re-tuning.
`S1` is the primary claim; `S2` and `S3` are reported alongside it and will not
be promoted to primary regardless of how they score.

| id | rule | showcase64 | prec | rec | showcase p |
| --- | --- | --- | --- | --- | --- |
| **S1 primary** | `unknown_variant_logprob >= -5.82437` | 47/64 (+5) | 0.750 | 0.562 | 0.125 |
| S3 conservative | `unknown_exact_logprob >= -5.71051` | 46/64 (+4) | 1.000 | 0.438 | 0.125 |
| S2 exploratory | `answer_mean_entropy >= 0.362745 AND unknown_variant_logprob >= -6.62711` | 48/64 (+6) | 0.818 | 0.562 | 0.031 |

S2 is the best of 210 signal pairs x 2 boolean modes x every threshold pair, so
its showcase p-value is uncorrected for a search of roughly 10^6 candidates and
carries no evidential weight on its own. Three gates are evaluated on stability,
so a Bonferroni-corrected alpha of 0.05/3 = 0.0167 is also reported.

## Decision rule

The number that matters is stability192. On that split the existing cascades are
not statistically distinguishable from the single model — System48 vs nimbus_low
is 13 vs 6, p = 0.167; System46 vs nimbus_low is 9 vs 5, p = 0.424. A gain is
**not** claimed as an improvement unless the exact McNemar p-value on stability
clears 0.05. If it does not, the result is reported as a point estimate that
fails its own significance test, in the same terms as the cascades it replaces.

## Reference points

| | showcase64 | stability192 |
| --- | --- | --- |
| nimbus_low (1 pass) | 42 | 121 |
| Baya 2.0 System46 (3 passes) | 46 | 125 |
| System48 | 48 | 128 |
| oracle perfect abstention on nimbus_low | 55 | 159 |

