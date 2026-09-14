# Long-context retention pre-registration

Written **before** running either model. RAB-001 has measured Opallo on an arithmetic panel
and on the Reasoning Forge. It has never measured it on the instrument Supa was selected by.

## The blind spot

Supa 1.7.0-devfreeze is a **long-context core**. Its published identity is 40/64 Showcase,
121/192 Stability, 161/256 overall on the Rineya Long-Context Demonstration Suite, and that
is the capability the family roster keeps it for. Opallo retrained 20 of its 28 blocks on
5,233 tokens of short conversational SFT -- every training row about 128 tokens, against
demo contexts of roughly 860 words.

Nothing in this programme has checked what that did. "Without losing its existing strong
ability" cannot be asserted while the model's headline ability is unmeasured.

## Protocol

| | |
|---|---|
| Instrument | `Rineya_Long_Context_Demo_v1/data/all_prompts.jsonl`, 256 rows |
| Runner | `Rineya_Long_Context_Demo_v1/scripts/run_local_hf.py` |
| Scorer | `Rineya_Long_Context_Demo_v1/scripts/evaluate_predictions.py` |
| Decoding | greedy, `max_new_tokens` 24, seed 17, truncation forbidden |
| Models | `Supa/` and `Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze`, both BF16 |

**The published 161/256 is not the comparator.** It was produced under Transformers 5.14.1
and this host pins 4.51.3. Both models are therefore run here, in the same session, under
the same runtime, and Supa's score measured *here* is the baseline. If my Supa run does not
land near 161/256 that is a runtime difference and gets reported as one, not silently
absorbed into an Opallo claim.

## Pre-registered test

Paired McNemar, exact, two-sided, over discordant rows only, alpha = 0.05, on all 256 rows.

* **Significant and Opallo lower** -> Opallo has lost long-context ability. The dataset plan
  must then carry long-context preservation data, and that becomes its first constraint.
* **Significant and Opallo higher** -> report it and do not build on it without replication;
  a single unreplicated gain on a new instrument is exactly the kind of result this
  programme has withdrawn before.
* **Not significant** -> report the minimum detectable effect alongside the counts. A null
  on 256 rows is not evidence of no change, and will not be written up as one.

Reported but **not** tested, because they are strata of the same 256 rows and testing each
would be the multiple-comparisons error this programme has already made once: per-variant
scores (base / paraphrase / counterfactual / unanswerable), per-task-type, per-needle-profile,
and `unanswerable_accuracy` -- abstention being the known family weak spot at 0.203125.

## What is not being decided here

This measures retention. It does not license any dataset, mixture or training decision by
itself; those follow from the measurement and get their own pre-registration.
