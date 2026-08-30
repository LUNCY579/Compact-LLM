# Evaluator Amendment 0001

Visible evidence before amendment: four official-Qwen3-0.6B MERGEDEV smoke
responses from the `common_sense` family. No Supa, donor, merge, coefficient,
layer, module, DEV-FULL, or sealed result was visible.

The v1 static-fact scorer required full normalized phrase equality even though
the prompt requested a concise natural answer. Two responses expressed the
expected fact inside a sentence and were scored wrong; two responses were
substantively wrong. This made the scorer measure surface form rather than the
predeclared capability.

Amendment:

- static science/common-sense facts use deterministic content-keyword matching;
- expected and response tokens receive the same documented light stemming;
- EOS completion rate remains a separate short-response/formatting metric, so
  accepting a correct sentence does not hide verbosity or truncation;
- the prior bank and smoke run are labeled invalid for baseline claims;
- all four banks are regenerated and rehashed before any Supa or candidate run.

This is an evaluator-debugging correction, not model selection. If another
scoring defect appears after Supa or merge results are visible, affected results
become exploratory and a new independent bank is required.
