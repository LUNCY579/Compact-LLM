# Relative Opportunity Evaluation Policy

**Version:** 1.0  
**Status:** Approved governing amendment  
**Applies to:** Phase 0–8, structural-reduction experiments, and the Phase 2B playground

## Governing decision

Official promotion decisions do not use absolute quality floors such as `0.95`, `0.98`, `0.99`, a maximum fixed quality loss, or a zero-issue veto.

For an efficiency candidate relative to its declared reference:

```text
resource_gain = proportional reduction in the phase's primary resource
quality_opportunity_loss = weighted proportional degradation across available quality evidence
opportunity_margin = resource_gain - quality_opportunity_loss
eligible = technical_validity AND opportunity_margin > 0
```

The comparison to zero is not a quality threshold. It asks only whether measured benefit is greater than measured opportunity cost.

For a quality-only phase at unchanged inference cost, quality gains and losses are compared proportionally and the candidate advances when net relative quality change is positive.

## Conversational quality contract

The model is intended to communicate like a capable person texting, not like an English examination answer key.

The evaluator therefore permits:

- casual and informal language;
- minor grammar, vocabulary, punctuation, and word-order mistakes;
- slang and abbreviations;
- natural follow-up questions;
- clarification instead of one reference wording;
- alternate reasonable conversational strategies;
- names, code, and words outside a small benchmark vocabulary.

Examples that must not be major failures solely for wording or questioning:

```text
Did it complete successfully?
How did the metrics look?
Did it succeed?
How the other adapter looks?
That layer you are targeting?
Which sector you are referring to?
Probably not tbh.
```

A reply loses meaningful quality only when the imperfection materially harms comprehension, task adequacy, logic, context, factuality, safety, or usability.

## References and response contracts

A reference answer is an example, not the unique correct response.

Open conversational cases use reference similarity only as a weak diagnostic. Reference/content-anchor influence is stronger for explicit factual compression, memory recall, safety boundaries, required entities, required decisions, and other contracts where content must be preserved.

Restrictions are created only by explicit benchmark fields. A category name or reference punctuation must not silently create `question_required`, `question_forbidden`, exact wording, or another behavioral rule.

Recommended schema:

```json
{
  "response_contract": {
    "open_ended": true,
    "acceptable_modes": ["acknowledgment", "natural_follow_up", "brief_opinion"],
    "required_facts": [],
    "forbidden_claims": [],
    "question_required": false,
    "question_forbidden": false
  }
}
```

## Continuous evidence

Case analysis reports continuous dimensions:

1. comprehensibility;
2. task adequacy;
3. logic and factual consistency;
4. safety and boundary behavior;
5. context consistency;
6. response completeness;
7. brevity appropriateness;
8. conversational naturalness;
9. content-anchor preservation.

Candidate comparison also considers mean quality, worst-decile quality, category losses, issue-cost change, severe-collapse rate, and target NLL when available.

No single quality dimension independently vetoes a valid candidate. Serious failures receive large proportional costs so their frequency and severity are visible in the opportunity calculation.

## Technical validity

Binary blocking remains appropriate only when the comparison is not technically valid, including:

- missing or corrupt checkpoints;
- incomplete serialization;
- missing result files;
- non-finite measurements;
- incompatible architecture or tokenizer;
- mismatched evaluator versions or case sets;
- unreproducible provenance;
- missing required human approval.

These are evidence-validity conditions, not absolute quality standards.

## Phase primary benefits

| Phase | Primary proportional benefit |
|---|---|
| 1 | Net relative quality improvement over the base |
| 2 | Training-memory or trainable-state reduction |
| 2B | Loaded weight-tensor memory reduction |
| 2C | Trainable-state/training-memory reduction |
| 3 | Net relative alignment-quality improvement |
| 4 | Net relative quality improvement at unchanged inference cost |
| 4B | Structural weight-memory or measured-compute reduction |
| 5 | Context-token or KV-memory reduction |
| 6 | Active-compute reduction |
| 7 | Measured latency or peak-VRAM reduction |
| 8 | Pareto comparison plus explicit human release approval |

Secondary metrics remain visible and may contribute to the quality-loss profile, but they do not silently replace the phase's declared primary benefit.

## Weight profiles and sensitivity

All decision proportions are versioned in
`configs/relative_opportunity_evaluation.json`. Every opportunity result records
balanced, logic-oriented, and conversational-permissive profile sensitivity.
The selected profile controls the official calculation, while disagreement
across profiles is visible for human review rather than hidden in code.

`data/evaluation/quality_v3_1_calibration.jsonl` contains approved policy examples
and leaves human rankings explicitly `PENDING`. Run
`scripts/common/calibrate_quality_profiles.py` to generate the deterministic
calibration report before changing weights.

## Reporting

Every official comparison should report:

- reference and candidate resource usage;
- resource-gain fraction;
- reference and candidate quality dimensions;
- relative loss by dimension;
- combined quality opportunity loss;
- opportunity margin;
- Pareto status;
- technical invalidity reasons, if any;
- human decision status.

Legacy absolute thresholds may be accepted as deprecated command-line aliases for compatibility, but they must be ignored and clearly identified as non-controlling.

## Observable-defect amendment — Evaluator v4.0

Evaluator v4.0 removes subjective catch-all labels from official scoring. The evaluator must not emit or score labels whose meaning is merely that a reply seems unresponsive, weakly relevant, lexically unusual, or polarity-mismatched.

A scored issue is valid only when it is one of the following:

1. a concrete visible mechanical/structural defect with reproducible evidence;
2. an explicit response-contract violation stored in the benchmark record; or
3. a directly observable leakage, contradiction, or safety defect whose triggering evidence is stated.

Examples of valid concrete labels include `encoding_replacement_character`, `repeated_token_loop`, `broken_clause_sequence`, `internal_reasoning_leakage`, `peak_vram_fact_missing_or_wrong`, and `consent_requirement_missing`.

The following are review-only and have zero score cost: reference wording dissimilarity, low lexical overlap, uncommon vocabulary, alternate conversational mode, minor grammar, and script/language differences that do not create objective corruption.

Category names and reference answers do not create requirements. Strict cases must declare `response_contract.requirements`, `response_contract.prohibitions`, question constraints, and domain mappings directly in the JSONL. A legacy benchmark without those contracts is not valid official Phase 2B selection evidence.

Evaluator v4 summary evidence includes `critical_defect_case_rate`, `observable_defect_case_rate`, `mechanical_failure_case_rate`, `contract_violation_case_rate`, `mean_mechanical_integrity_score`, and `mean_contract_fulfillment_score`. These enter the same proportional opportunity calculation and do not create hidden absolute promotion floors.

Any new scored label requires a versioned definition, positive and negative regression examples, human adjudication, and inclusion in `configs/quality_issue_taxonomy_v4.json` before it can affect official candidate selection.

## Continuous-function amendment — Evaluator v4.1

Evaluator v4.1 preserves the v4.0 ban on subjective catch-all defect labels and adds two official continuous dimensions: explicit response-mode fulfillment and requirement-level partial contract fulfillment. Mechanically valid text no longer receives perfect task adequacy merely because it avoids a concrete defect. Conversely, a substantially correct short answer is not treated the same as an unrelated or entirely wrong answer because one optional detail is absent.

The shared opportunity profile now includes `response_mode_fulfillment`. All new Phase 1–8 summaries must use quality evaluator `4.1` and opportunity policy `1.3`; mixed-version evidence must be rescored rather than compared automatically. The authoritative calibration packet must cover every declared benchmark category, every scored label, every authored response mode, partial answers, and adversarial keyword traps before evaluator changes are accepted.
