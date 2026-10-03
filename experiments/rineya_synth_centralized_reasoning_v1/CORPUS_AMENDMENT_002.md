# CPU source-policy amendment 002 — 2026-10-03

V1's fully crossed pool is retained under
`data/crossed_v1_blocked_20261003/`, with source snapshots, all 640 candidate
records/arrays and the frozen enrollment. It is **not admitted for training**.
[The independent constraint diagnostic](evidence/matching_v1_constraint_diagnostic.json)
proves its failure: for shared integer weights 1–6 and equal S cardinality/R
first-request-depth margins, M2-minus-M1 input tokens are at least 736. This
exceeds the frozen 512-token residual. The exact certificate uses public rows
and tokenizer measurements, not student predictions or held-out outcomes.

The cause includes an avoidable intervention confound. V1's `request_rebound`
changes a request to a one-hop distractor, irrespective of the original depth.
It therefore changes depth as well as endpoint. Restoring the registered depth
margin then costs extra input tokens. That rejects this specific matching
procedure; it does not reject condition-diverse supervision or the research track.

Before a new corpus/solver attempt, register source-policy 2:

- Construct the alternate unrequested source chain at the **same depth** as the
  first requested chain, with a different terminal item.
- Permit `request_rebound` to choose only a novel endpoint reached through a
  chain of that same depth. Internal shorter aliases are ineligible.
- Keep the two cyclic depth interventions, endpoint intervention, S interventions,
  full condition/render crossing, original-space witness policy and item remaps.
- Retain the 64-source candidate-generation budget, 20-parent shared enrollment
  rule and all existing tokenizer, budget, position, accumulation and marginal
  checks. Freeze the new entire catalog/component assignment/enrollment before
  tokenization. V2 source bytes, components and enrolled parents may differ;
  they are not mislabeled as the already frozen V1 pool.
- Preserve V1 and its proven block. Do not remove parents from it, replace one
  arm's rows, pad examples, truncate targets or widen a gate to rescue it.

This is an isolated, separately recorded CPU source correction. It uses no model
feedback and opens no native, training, protected-bank, paid-service or release
gate. Equal marginal cardinality/depth and bounded answer-position residuals do
not establish matched joint difficulty or supervised-token counts. Report those
remaining distributions and describe any later result as comparing the two
frozen curricula. Shared occurrence order also does not equalize every update's
condition distribution; report per-update exposure/counts.
