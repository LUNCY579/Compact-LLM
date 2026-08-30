# Evaluator Amendment 0002

Visible evidence before amendment: official Qwen3-0.6B on MERGEDEV only. No
full Supa, projected-donor, merge, coefficient, layer/module, DEV-FULL, or
sealed result was visible.

The audit found three deterministic scoring defects:

- numeric scoring extracted the first operand rather than the final reported
  number;
- binary-choice content such as `Answer: YES.` failed exact-string scoring;
- correct content such as `Dina is the youngest` and `the code is UNKNOWN`
  failed because formatting and capability were conflated.

Amendment:

- numeric semantics use the final number in the response;
- binary choices use the final recognized allowed choice token;
- categorical reasoning and abstention use content-keyword semantics;
- every case may carry a separate `format_scorer` for exact/concise adherence;
- selection reports semantic family score and format adherence separately.

This amendment completes baseline evaluator debugging. The next generated bank
is frozen before any full Supa or candidate run. Later scorer changes require a
new independent bank and make affected model comparisons exploratory.
