# Quality Evaluator v4.0 — Observable Defects and Explicit Contracts

## Status

This document is the authoritative implementation guide for Evaluator v4.0. It supersedes the v3.0 and v3.1 semantic-overlap evaluators for all new automated comparisons. Historical evidence remains archived but cannot be mixed with v4 evidence.

## Why v4 exists

Evaluator v3.0 allowed some ordinary-word generation collapse to score too highly. Evaluator v3.1 corrected that failure but relied too heavily on lexical prompt/reference overlap and broad semantic labels. It consequently marked many valid replies—natural follow-up questions, informal grammar, alternate social responses, and unusual vocabulary—as major failures.

Evaluator v4 resolves both problems by refusing to score a judgment unless the problem can be named and reproduced.

## Label admission rule

A scored label is allowed only when all of the following are true:

1. it identifies a concrete defect or an explicitly authored contract violation;
2. the output or contract contains reproducible evidence for the label;
3. the label explains what must be fixed in the generated response;
4. positive and negative regression examples exist; and
5. the label is documented in `configs/quality_issue_taxonomy_v4.json`.

Catch-all labels such as semantic unresponsiveness, weak relevance, unusual vocabulary, or surface polarity contradiction are prohibited.

## Evaluation layers

### A. Observable output integrity

`scripts/common/observable_quality.py` detects mechanical or structural defects such as:

- empty or corrupted output;
- Unicode replacement/control characters;
- repeated-token or repeated-phrase loops;
- symbol-dominated or micro-fragmented output;
- reproducible malformed clause sequences;
- internal reasoning, role-header, or hidden-instruction leakage;
- directly dangerous gas/medication reassurance;
- direct contradiction of a user-stated condition.

Multilingual text and mixed scripts are not defects by themselves.

### B. Explicit response-contract validation

Strict requirements live in each benchmark row under `response_contract`. Category names and references do not silently create requirements.

Example:

```json
{
  "response_contract": {
    "schema_version": "4.0",
    "open_ended": false,
    "min_tokens": 1,
    "max_tokens": 24,
    "requirements": [
      {
        "id": "peak_vram",
        "label": "peak_vram_fact_missing_or_wrong",
        "severity": "major",
        "description": "The reply must preserve the stated peak VRAM value of 6.4 GB.",
        "match": {
          "any_regex": ["\\b6\\.4\\s*gb\\b"]
        }
      }
    ],
    "domains": {
      "factual": ["peak_vram"]
    }
  }
}
```

A failing reply receives `peak_vram_fact_missing_or_wrong`, not a broad relevance label.

### C. Continuous relative-comparison evidence

Case scores aggregate mechanical integrity, contract fulfillment, factual/context/safety domain fulfillment, completeness, brevity, and low-impact naturalness. Candidate selection still uses the shared relative-opportunity rule:

```text
primary resource gain > combined proportional quality opportunity loss
```

No absolute score floor or zero-issue veto is reintroduced.

## Review-only diagnostics

These may assist a human reviewer but have zero score cost:

- low reference wording overlap;
- unexpected conversational mode;
- unusual or technical vocabulary;
- multilingual or mixed-script output;
- minor grammar irregularity;
- generic assistant phrasing;
- missing non-mandatory recommended content.

References are examples and diagnostics, not unique answer keys.

## Contracted Phase 2B probes

Official Phase 2B architecture selection requires an explicit v4 contract on every probe case. The repository provides:

```text
data/phase_2_structural_reduction/probe_50_v4.jsonl
data/phase_2_structural_reduction/probe_150_thorough_v4.jsonl
```

The Phase 2B runner and migration tool reject legacy uncontracted probes. New probes can be authored with:

```powershell
python .\scripts\common\build_v4_response_contract_benchmark.py `
  --input .\path\to\legacy_probe.jsonl `
  --output .\path\to\contracted_probe_v4.jsonl
```

The generated contracts must still receive human review before use in an important model-selection run.

## Human adjudication and supplied-output review

`data/evaluation/quality_v4_adjudication.jsonl` contains 24 approved positive and negative regression cases. It verifies that informal, alternate, multilingual, and low-reference-overlap replies remain defect-free while concrete collapse and contract failures receive exact labels.

The supplied 150-output terminal log was reviewed under v4. The comparison report is:

```text
reports/QUALITY_V4_PASTED_150_REVIEW.md
```

The prior evaluator marked 77 cases major and another 35 warning-only. Evaluator v4 produces 25 scored-defect cases, all tied to explicit missing requirements; it produces no critical defect for those 150 coherent outputs. This is a read-only calibration result, not an approval of any Phase 2B candidate.

## Verification

Run from the repository root:

```powershell
python .\scripts\common\calibrate_quality_profiles.py --strict
python .\scripts\common\verify_quality_evaluation.py
python .\scripts\common\audit_relative_opportunity_policy.py
pytest -q
```

Any mismatch blocks an official evaluator release.

## Known limitations

Evaluator v4 is deterministic and offline. It cannot discover every possible factual or social failure without an explicit contract. That limitation is intentional: the automated system must not invent subjective defects. Important new benchmark cases therefore require contract authoring and human adjudication. Raw replies, contract results, teacher-forced loss, category evidence, and human review remain necessary before a model is promoted.
