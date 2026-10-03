# Independent LORA-596 audit — 2026-09-19

LORA-596 improves arithmetic totals but does not establish learning without forgetting.
The new transition skill reaches only one of three registered family floors. This audit
re-scores existing raw answers on CPU: **zero model calls, zero training, no source mutations**.
It is exploratory historical analysis, not an independent confirmation experiment.

The machine-readable audit is [LORA596_AUDIT.json](LORA596_AUDIT.json), including input
SHA-256 hashes, lost/gained row IDs, per-class and per-variant results. Reproduction uses
[audit_lora596.py](../../experiments/continual_skill_audit_v1/audit_lora596.py).

| Endpoint | Supa | LORA-596 | Old successes lost | Newly correct | Interpretation |
|---|---:|---:|---:|---:|---|
| Acquisition | 14/384 historical baseline | 70/384 | Not used as retention bank | — | 1/3 family floors cleared |
| Arithmetic, permissive | 47/150 | 68/150 | **0/47** | 21 | Observed preservation on this bank only |
| Arithmetic, strict | 41/150 | 58/150 | **3/41** | 20 | Net +17 hides old-item losses |
| Long context | 161/256 | 155/256 | **9/161** | 3 | Losses affect six distinct scenarios |
| Literal UNKNOWN, insufficient | 13/64 | 13/64 | **0/13** | 0 | Exactly the same 13 items retained |
| Literal UNKNOWN, answerable | 0/192 | 0/192 | — | — | No new literal over-abstention |

The 14/384 acquisition baseline is historical context from the recorded programme; this
audit independently re-scores the candidate acquisition, both arithmetic sides, and both
long-context sides. All 384 candidate JSON correctness flags reproduce. Per-family counts
are ledger 27/128, prerequisite 5/128, register 38/128. Only register's two-sided 95% exact
lower confidence limit exceeds its registered shortcut floor.

## What the total long-context score conceals

| Variant | Source → candidate | Lost / gained |
|---|---:|---:|
| Base | 53 → 50 | 4 / 1 |
| Counterfactual | 38 → 39 | 1 / 2 |
| Paraphrase | 57 → 53 | 4 / 0 |
| Unanswerable | 13 → 13 | 0 / 0 |

The apparent preservation of abstention is real for these exact items. It does not compensate
for lost answerable binding competence. The row-level exact McNemar p is 0.146; flipping
whole scenario contributions instead gives 0.246. Both are descriptive results on reused
data, neither is evidence of equivalence. Four variants of a scenario stay together in
the latter calculation. Its null additionally assumes sign symmetry/exchangeable arm labels.

Even **0/47** observed losses has a one-sided exact 95% upper loss-probability bound of
**6.18%**, under iid row sampling. Those 47 successes span only 16 of 19 arithmetic templates.
Grouping by template leaves 0/16 affected templates, with an illustrative iid upper bound
of 17.07%. Templates were not randomly sampled, so this is a sensitivity calculation,
not a population estimate. Neither analysis warrants a universal no-forgetting claim.

## Comparability corrections

1. The preregistration specifies FP32 training. The actual trainer requests `torch.bfloat16`,
   and TRAIN.json says BF16. The adapter is later merged into an FP32 source. FP32 export
   does not retroactively make training FP32.
2. LoRA targets all 28 blocks; the full-finetuning control targets blocks 8–27. This scope
   difference was disclosed but still prevents a one-variable causal comparison.
3. The LoRA trainer uses update-index linear warmup/decay; the control uses processed-token
   scheduling with square-root decay. The same phrase “20/60/20 WSD” does not establish parity.
4. TRAIN.json logs learning rate **after** `scheduler.step()`. It records the next update's
   rate, not the one just applied. The successor must log both if both are useful.
5. The published retention report's `reference_size` 68/61 is the **union** of source and
   candidate successes, not the immutable source denominators 47/41. Its `lost_total`
   matches rescoring, but the enlarged denominator must not be used for forgetting risk.
6. The same learning rate does not match functional update size across parameterizations.
   A smaller LoRA delta with weaker acquisition cannot by itself identify a better
   preservation mechanism. Report the acquisition/retention/cost frontier.

These are limitations of this run. They do not establish that LoRA succeeds or fails as a
method, or that 596M capacity is sufficient or insufficient.

## Verification and scope

The frozen arithmetic source-correct ID sets reproduce exactly under both typed policies.
All 256 source and 256 candidate long-context correctness flags reproduce from raw answers
using the historical acceptance rule. Acquisition and arithmetic bank hashes match recorded
hashes. Runtime digest, device, context, decode limit, Transformers version and numerical
settings match for the long-context comparison.

Historical long-context artifacts do not cryptographically bind the prompt-bank file;
the current bank reproduces all scores but cannot repair that missing provenance. Model
weights were not rehashed or executed in this audit. The label inventory records machine
review, with human review pending; this audit does not invent that review.

The historical long-context acceptance rule normalizes periods and scores only the first
line. Literal UNKNOWN was audited separately. No historical registrations, scores or
published model cores were overwritten. Quality improvement and release flags remain false.
