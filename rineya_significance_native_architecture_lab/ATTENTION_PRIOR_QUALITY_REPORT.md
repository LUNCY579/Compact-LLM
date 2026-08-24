# Experiment C Quality Screen — Halo Attention Prior

## Decision

`NO_MEASURABLE_ANSWER_GAIN / SPECIFICITY_NOT_SUPPORTED`

The frozen `lambda=0.01` attention prior changed every evaluated logits trajectory
and final cache state, but it changed **no decoded answer** on the 12 selected Halo
clusters. Normalized `S/C` did not outperform raw `S` or shuffled significance on
the frozen evidence endpoint. This is exploratory, unregistered evidence and does
not support Finding B or a feedback-safety claim.

## Frozen design

The screen reused the exact 12 IDs in
[`halo_endpoint_validation_v2_registered.yaml`](configs/halo_endpoint_validation_v2_registered.yaml)
without changing their prompts, questions, tokenizer, model, or B0-eligibility
status. The new attention-prior conditions and contrasts were not part of that
registration, so this run is classified `EXPLORATORY_NOT_REGISTERED`.

Every cell used:

- published Rineya Halo Qwen3-0.6B, CPU bfloat16, SDPA, seed 41;
- B3 normalized cache selection with B128 retained history and C128 chunks;
- sinks 4, recent 32, pooling disabled, query probe at every non-final boundary;
- the same question tokens and exact rollback;
- manual greedy decoding for at most 32 tokens;
- no answer/evidence/oracle access during inference.

The five conditions were C0/B3 with no prior, normalized `S/C`, raw `S`,
deterministically shuffled normalized significance, and age-only. Every nonzero
condition used `lambda=0.01`, frozen from the compatibility smoke before this
quality run and never tuned on its outcomes. The resolved configuration is
[`halo_attention_prior_quality_screen_v1.yaml`](configs/halo_attention_prior_quality_screen_v1.yaml).

## Execution safeguards

The runner was deliberately interrupted after 1/60 cells, then resumed from the
exact durable prefix. It wrote resident-position and feedback artifacts before
checkpointing each cell. Only after all 60 predictions were durable did it hash
and open the evaluation view.

The same-process C0 gate compared no prior against a normalized prior with
`lambda=0` before answer access. It passed exact equality for:

- every logits trajectory;
- generated token IDs;
- complete K/V/S/C/position/entry-ID cache state;
- final resident positions;
- retained-set diagnostics;
- zero nonzero-bias applications.

The matching audit passed all nine checks for model, seed, B128/C128 contract,
cache policy, query access, resident prefill size, decoder, cell order, and
answer-access separation.

## Quality results

| Condition | Correct | Endpoint v1 mean | Attention entropy | Retained diversity | Churn |
|---|---:|---:|---:|---:|---:|
| C0 B3, no prior | 5/12 | 0.21131 | 0.348050 | 0.282761 | 0.583119 |
| Normalized `S/C` | 5/12 | 0.21577 | 0.347888 | 0.282832 | 0.583337 |
| Raw `S` | 5/12 | 0.21577 | 0.347886 | 0.282799 | 0.583310 |
| Shuffled `S/C` | 5/12 | 0.21577 | 0.348089 | 0.282944 | 0.583317 |
| Age-only | 5/12 | 0.21280 | 0.348015 | 0.282921 | 0.582746 |

Every pairwise answer comparison was 0 wins, 0 losses, and 12 ties. Predictions
were text-identical across all five conditions for every cluster. This was not an
inactive intervention: each comparison against normalized `S/C` had 0/12 equal
logits trajectories and 0/12 equal final cache states.

Family accuracy was identical across conditions: R1 3/4, R3 2/4, and R4 0/4.
The endpoint movement was confined to small retention differences, principally
R4, without any decoded-answer movement.

## Whole-cluster paired contrasts

Normalized `S/C` minus each control on endpoint v1:

| Control | Mean delta | 95% cluster bootstrap CI | Exact sign-flip p |
|---|---:|---:|---:|
| No prior | +0.00446 | [-0.00298, 0.01190] | 0.5000 |
| Raw `S` | 0.00000 | [0.00000, 0.00000] | 1.0000 |
| Shuffled `S/C` | approximately 0 | [-0.00446, 0.00446] | 1.0000 |
| Age-only | +0.00298 | [0.00000, 0.00893] | 1.0000 |

The frozen specificity rule therefore returns `NOT_SUPPORTED`: normalized `S/C`
did not exceed every raw, shuffled, and age control. In particular, raw and
normalized feedback had the same mean endpoint and decoded outcomes, while the
shuffled control matched them to numerical precision at the aggregate endpoint.

## Feedback diagnostics

The bounded prior remained numerically valid in all cells:

- maximum absolute nonzero bias: `0.01999999955`, within the `0.02` bound;
- maximum bias on current-forward keys: `0.0`;
- maximum bias on invalid/masked keys: `0.0`;
- strict all-evidence survival: 0/12 in every condition.

Normalized `S/C` slightly reduced normalized attention entropy relative to C0
by `-0.000162` (CI `[-0.000240, -0.0000865]`, exploratory p `0.00244`) and
relative to shuffled significance by `-0.000201` (CI `[-0.000248, -0.000149]`).
It was indistinguishable from raw `S` (`+0.0000020`, CI spanning zero). These
small mechanistic shifts did not translate into answer changes and are subject
to multiple exploratory comparisons.

Retained-set diversity and churn were essentially unchanged against no-prior,
raw, and shuffled controls. Normalized feedback had slightly more churn than the
age control (`+0.000591`, CI `[0.000381, 0.000823]`), again without quality gain.
Exact early/middle/late position capture was recorded at every compression
boundary. Exact false-positive persistence could not be scored because the
frozen evaluation view names distractor/stale segments but does not include
their exact tokenizer-position spans.

## Interpretation limits

This result addresses one compact model, one frozen lambda, one B128/C128
schedule, and 12 clusters selected for B0 capability. It does not show that all
soft significance priors fail, that a trained per-layer prior would fail, or
that feedback is safe at longer horizons. It does show that this untuned bounded
prior is not specifically benefiting from normalized `S/C` on this controlled
screen.

The canonical machine-readable run is
[`results/runs/halo_attention_prior_quality_screen_v1`](results/runs/halo_attention_prior_quality_screen_v1/).

