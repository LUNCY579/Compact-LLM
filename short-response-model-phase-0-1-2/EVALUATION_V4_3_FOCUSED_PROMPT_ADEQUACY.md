# Evaluator v4.3 — focused prompt adequacy and robust partial credit

## Purpose

Evaluator v4.3 is a concentrated correction over v4.2. It does not replace the category-neutral, observable-defect architecture. It fixes the remaining two systematic errors found in the coherent 150-response Phase 2B run:

1. mechanically clean but functionally poor open-ended replies often fell into a generic score bucket; and
2. valid paraphrases of strict requirements were sometimes rejected by narrow phrase matching.

The evaluator still receives the complete benchmark record, but only the actual messages and explicit case-level response contract affect case scoring. `category` and reference wording remain reporting-only.

## Governing rules

1. **Category neutrality is invariant.** Renaming, deleting, or deliberately mislabeling `category` cannot change a score, defect, dimension, or promotion result.
2. **References are examples, not answer keys.** Reference overlap is review-only unless a fact is explicitly authored in the case contract.
3. **Open-ended adequacy is continuous.** A coherent reply may be weak, incomplete, or misaligned without receiving a vague defect label.
4. **Strict requirements allow graded evidence.** A direct paraphrase can receive full credit; partial evidence can receive partial credit; keyword presence alone is insufficient.
5. **Concrete failures remain concrete.** Corruption, factual errors, missing consent, context reversal, leakage, and other scored issues retain named evidence-based labels.
6. **No new absolute promotion floor exists.** Phase decisions still compare proportional resource gain with cumulative and incremental quality-opportunity loss.

## Focused architecture

### A. Universal integrity

Every reply is checked for decoding corruption, repetition, malformed structure, truncation, leakage, and other category-independent failures. The 150 real collapsed outputs remain a permanent regression suite and must all score zero with a critical concrete defect.

### B. Explicit response contracts

Objective requirements use scored evidence options rather than one brittle expression. A requirement may define:

- several accepted paraphrase patterns with different credit;
- a default score when evidence is absent;
- adversarial score caps when misleading keywords are present;
- a minimum fulfillment score;
- major or minor severity when the requirement remains unmet.

Examples corrected in v4.3 include:

- “That is not sufficient; it does not guarantee.” correctly rejects a FLOPs-only speedup claim;
- “The sealed set should stay untouched.” preserves isolation;
- “Not clearly.” communicates a visibility limitation with partial credit;
- “It reduces model size and memory.” is accepted as a valid ten-word compression overview;
- “I need the exact metrics and benchmarks.” receives partial evidence-seeking credit without being treated as a complete release boundary.

### C. Prompt-grounded adequacy

For selected open-ended benchmark cases, the individual record contains an explicit `prompt_adequacy` rule authored from its actual system and user messages. It is not generated from category or reference text.

The rule contributes continuous scores to task adequacy and, where appropriate, factual or context consistency. It emits no issue label and creates no hidden veto. A smooth multiplier prevents a clearly poor reply from retaining a high aggregate score merely because unrelated dimensions default to one:

```text
prompt_adequacy_factor = 0.65 + 0.35 × prompt_adequacy_score
```

This intentionally preserves nonzero partial credit. It is not a hard pass/fail gate.

### D. Broad fallback

Cases without an authored prompt-adequacy rule use the existing broad message-grounded heuristic. This keeps arbitrary user-authored benchmarks usable without inferring behavior from their reporting tag.

## Real coherent regression

The 150 coherent Phase 2B outputs were human-adjudicated before running v4.3 as `good`, `weak`, or `poor`, each with an expected score interval.

Results:

- all **150/150** scores fall within their human-authored intervals;
- 89 good replies average **0.972**;
- 32 weak replies average **0.690**;
- 29 poor replies average **0.440**;
- no coherent reply is incorrectly marked as critical corruption;
- the old dominant `0.720` bucket is substantially reduced and prompt-specific distinctions are restored.

See `reports/QUALITY_V4_3_REAL_COHERENT_REGRESSION_REPORT.md`.

## Comprehensive verification surface

The v4.3 packet contains 611 deterministic adjudications:

- 150 official reference-contract cases;
- 150 real collapsed outputs;
- 150 real coherent outputs with human score bands;
- all 31 reporting categories represented with good and poor controls;
- all 60 scored issue labels;
- 17 focused v4.3 false-high, false-low, paraphrase, adversarial-keyword, and category-invariance regressions;
- valid informal grammar, multilingual text, technical notation, code, paths, measurements, one-word replies, role-play, safety, memory, context, leakage, truncation, repetition, and partial factual answers.

No finite packet can contain every sentence a language model might generate. “Complete coverage” here means every formal evaluation mechanism, supported response surface, scored label, official category, and known real failure family in this repository.

## Non-goals

v4.3 does not:

- use an external LLM judge;
- infer obligations from category names;
- require reference wording;
- label every awkward reply as defective;
- convert a continuous prompt-fit score into an independent promotion veto;
- claim that deterministic rules replace final human review.
