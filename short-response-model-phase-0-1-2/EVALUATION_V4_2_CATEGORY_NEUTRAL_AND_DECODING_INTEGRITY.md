# Evaluator v4.2 — category-neutral scoring and decoding integrity

## Purpose

Evaluator v4.2 corrects two remaining design defects in v4.1:

1. `category` could indirectly create expectations, change weights, and influence model selection.
2. Mechanically corrupted outputs could receive high scores when no narrow corruption pattern fired.

The new evaluator keeps categories only as reporting tags and evaluates each reply from the actual system/user messages, explicit case-level assertions, and visible output surface.

## Category policy

`category` is retained because it is useful for benchmark balancing and regression reports. It may answer questions such as “did factual cases degrade more than role-play cases?” It is not an input to generation and is now forbidden from affecting case scoring.

Changing, removing, or deliberately mislabeling a category must leave these fields identical:

- quality score and dimensions;
- scoring weights;
- detected defects and severity;
- contract fulfillment;
- prompt-function alignment;
- opportunity loss and promotion eligibility.

`expected_modes` from older records is ignored with a zero-cost diagnostic. The v4.2 benchmark contains no category-derived mode list or question rule.

## Three independent evidence layers

### 1. Universal decoding and mechanical integrity

Every reply is checked without using category or reference wording. Concrete signals include:

- empty or replacement-character output;
- invalid control characters;
- repeated token, phrase, or clause loops;
- fragmented line layouts;
- abnormal tokenizer fragmentation corroborated by another defect;
- digit-letter fusion and unexplained numeric fragments;
- punctuation or delimiter corruption;
- internal token fragmentation;
- isolated non-utterance fragments;
- multiple independent decoding anomalies.

Mixed scripts are not an error by themselves. Versions, measurements, paths, code identifiers, arithmetic, emoji, technical acronyms, and ordinary one-word replies have explicit false-positive protections.

### 2. Explicit case-level contracts

A strict test may declare an objective fact, context value, safety boundary, consent requirement, forbidden claim, or output constraint. These requirements must be authored in the individual record. References remain examples rather than unique answer keys.

Contracts support partial credit. Missing one secondary detail is not treated the same as a wholly wrong answer.

### 3. Continuous prompt-function alignment

For open conversation, the evaluator derives broad possible functions from the actual messages, not from category. This is permissive and continuous. A reply can acknowledge, ask a follow-up, express empathy, disagree, reassure, offer a next step, or use another natural strategy. Weak alignment lowers a dimension; it does not create a vague defect label.

## Real-collapse regression

The 150 supplied Phase 2B responses are preserved as a mandatory regression corpus. All 150 are human-adjudicated unusable decoding corruption. Evaluator v4.2 assigns all 150 a zero score and at least one critical concrete corruption label. The decision is category-invariant.

See `reports/QUALITY_V4_2_REAL_COLLAPSE_REGRESSION_REPORT.md`.

## Calibration scope

The deterministic calibration packet contains 444 cases covering:

- all 150 official benchmark references;
- all 31 reporting categories, without using them for scoring;
- all 60 scored labels;
- the 150 real collapsed outputs;
- valid polished, informal, grammatically imperfect, multilingual, technical, mathematical, code, path, version, and emoji surfaces;
- open conversation, factual and context contracts, partial answers, wrong answers, safety boundaries, privacy, consent, role-play, leakage, repetition, truncation, and decoding corruption;
- adversarial keyword-present-but-wrong and valid-paraphrase cases;
- category-removal, category-renaming, and legacy-`expected_modes` invariance.

“Every kind” means every response surface and evaluator mechanism formally supported by this repository. No finite packet can enumerate every sentence a language model could produce.

## Promotion policy

Categories remain visible in reports but category-tail statistics are diagnostic-only. No category score contributes to the combined opportunity loss. Phase promotion continues to use the shared proportional rule:

```text
eligible = resource gain > combined quality opportunity loss
```

Technical invalidity remains binary; quality differences remain proportional.
