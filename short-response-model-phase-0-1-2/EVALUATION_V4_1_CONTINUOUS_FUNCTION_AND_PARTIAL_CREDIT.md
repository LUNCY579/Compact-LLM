# Quality Evaluator v4.1 — Continuous Function, Partial Credit, and Adversarial Calibration

## Status

Evaluator v4.1 is the authoritative quality evaluator for new repository evidence. It supersedes v4.0 for automated Phase 1–8 comparisons. Historical v4.0 artifacts remain valid as archived evidence but must be rescored before they are mixed with v4.1 results.

## Why v4.1 exists

Evaluator v4.0 correctly removed subjective catch-all labels, but two weaknesses remained:

1. mechanically clean open-ended replies could receive a perfect score even when they did not perform the requested conversational function; and
2. strict contracts were often matched as binary regular expressions, so a valid paraphrase could fail while a keyword-bearing wrong answer could pass.

v4.1 keeps the observable-defect architecture and adds continuous functional scoring and requirement-level partial credit.

## Governing rules

1. **Scored labels name observable defects or explicit contract violations.** There is no scored `unresponsive`, `weak relevance`, unusual-vocabulary, or polarity-word label.
2. **Open conversational cases declare acceptable response modes.** The evaluator assigns continuous confidence to the authored modes and uses the best compatible mode as quality evidence.
3. **Strict requirements are explicit.** Category names and reference wording never silently create mandatory content.
4. **Requirements receive partial credit.** Missing one part of an otherwise substantially correct answer is different from an entirely wrong answer.
5. **Keyword presence is not sufficient.** Contract rules can require multiple independent concepts, weighted components, or one of several approved paraphrase patterns.
6. **Review-only diagnostics have zero score cost.** Reference dissimilarity, unusual vocabulary, multilingual text, and minor grammar remain visible only when useful for review.
7. **Candidate promotion remains relative.** The shared opportunity engine compares proportional resource benefit against proportional quality opportunity loss. No absolute score floor was added.

## Evaluation layers

### Layer A — Mechanical and observable defects

Examples include empty output, encoding replacement characters, control characters, repetition loops, broken clause sequences, truncation, reasoning leakage, role leakage, and explicitly unsafe instructions.

### Layer B — Explicit contract fulfillment

Every requirement has:

- an ID and concrete label;
- an authored description;
- a severity and weight;
- transparent evidence rules;
- a continuous score in `[0, 1]`;
- an optional partial-credit severity rule.

Supported evidence structures include phrase/regex alternatives, concept groups, weighted components, response modes, and adversarial combinations.

### Layer C — Functional response-mode fulfillment

The official benchmark currently authors nine response modes:

- `acknowledgment`
- `celebration`
- `clarification_or_follow_up`
- `empathy`
- `boundary_or_disagreement`
- `commitment_or_reassurance`
- `playful_or_in_character`
- `brief_opinion_or_comment`
- `supportive_next_step`

Additional detected modes may support contract evidence, but only modes explicitly accepted by a case affect that case's functional score.

### Layer D — Continuous quality dimensions

Each reply reports:

- comprehensibility;
- mechanical integrity;
- contract fulfillment;
- response-mode fulfillment;
- task adequacy;
- factual correctness;
- logic/factual consistency;
- safety boundary behavior;
- context consistency;
- completeness;
- brevity appropriateness; and
- conversation naturalness.

## Calibration coverage

The authoritative packet is:

```text
data/evaluation/quality_v4_1_comprehensive_calibration.jsonl
```

It contains **276 deterministic cases** covering the complete response surface currently supported by the repository:

- all **31** official Phase 2B categories, each with good and poor behavior;
- all **150** official reference/contract pairs;
- all **54** scored taxonomy labels;
- all **9** authored functional modes;
- informal and grammatically imperfect but understandable replies;
- alternate valid conversational strategies;
- multilingual and unusual-vocabulary replies;
- partial factual answers;
- context and memory errors;
- safety and privacy boundary failures;
- role and internal-reasoning leakage;
- encoding, truncation, markup, token-loop, phrase-loop, and broken-clause corruption;
- real Phase 2B false-positive and false-negative regressions; and
- adversarial keyword traps where a relevant word appears but the required meaning is absent.

This is exhaustive for the evaluator's **declared supported categories, labels, and modes**. It is not a claim that a finite deterministic evaluator can enumerate every sentence humans may ever produce.

Run the strict verifier:

```powershell
python .\scripts\common\verify_quality_evaluation_v4_1.py
```

Expected result:

```text
Evaluator v4.1 verification passed: 276/276 cases, 31 categories, 54 scored labels, 9 authored modes.
```

The generated report is:

```text
reports/QUALITY_V4_1_COMPREHENSIVE_CALIBRATION_REPORT.md
```

## Important corrected examples

| Reply | v4.1 treatment |
|---|---|
| `I need the source and metadata.` | Valid clarification/evidence request. |
| `I cannot tell that without a location or device.` | Valid knowledge boundary. |
| `Not without proper consent.` | Valid core privacy/publication refusal; de-identification remains recommended. |
| `LoRA is a low-rank adapter that fine-tunes a pretrained model.` | Substantially correct but incomplete; receives partial credit rather than binary failure. |
| `A cache that stores model states.` | Related but incomplete KV-cache definition; receives partial credit. |
| `No—sealed agreements take precedence.` | Fails the uncertainty contract despite containing the word `sealed`. |
| `That is loss pruning.` | Fails the compression contract because a keyword alone is not an explanation or scope-narrowing response. |
| `That is still not enough proof.` in a celebration case | Mechanically valid, but low continuous celebration-mode fulfillment; no subjective defect label is invented. |
| `You gives. you gives. You can you is the duplicate.` | Critical broken-clause/repetition corruption with concrete labels. |

## Maintenance rule

A new scored label or authored mode may affect official selection only after:

1. its definition and evidence are documented;
2. positive, negative, partial, and keyword-trap examples are added;
3. the comprehensive packet still covers every supported category, label, and mode; and
4. the full test suite and relative-opportunity audit pass.
