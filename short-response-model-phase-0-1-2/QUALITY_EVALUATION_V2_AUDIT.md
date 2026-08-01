# Quality Evaluation v2.0 — Repository-Wide Audit and Correction Report

## Executive conclusion

The previous evaluator was unsafe for automated model acceptance because its main score was usually the unweighted average of only three surface checks: non-empty output, apparent completion, and token range. A long unintelligible reply could therefore receive `0.667`, while shorter word salad could receive `1.000`. Several later phases added NLL, category, semantic, similarity, and human-review evidence, but many of those measurements inherited the same blind spot or were only diagnostic.

Quality evaluator v2.0 replaces that contract throughout Phases 1–8 and the independent Phase 2B playground. It does not pretend to solve semantic evaluation perfectly. Instead, it makes failure modes visible, assigns quality-oriented weights, applies hard vetoes to unmistakable collapse, measures low-tail and category behavior independently, and refuses to reuse stale legacy summaries.

## Corrected scoring model

Every generated case now exposes eight separate dimensions:

| Dimension | Default weight | Purpose |
|---|---:|---|
| Intelligibility | 30% | Human-readable, non-corrupted, non-repetitive language |
| Task compliance | 20% | Explicit include/exclude, clarification, question, and echo requirements |
| Semantic safety | 20% | Safety boundaries, uncertainty, empathy, role/prompt leakage, and category rules |
| Reference alignment | 10% | Fuzzy, token, and content-anchor agreement with the target when available |
| Completeness | 8% | Complete utterance rather than a fragment |
| Brevity | 6% | Target-length behavior without making length the primary objective |
| Non-echo | 3% | Avoid copying the user unnecessarily |
| Surface naturalness | 3% | Formatting, punctuation, line, and repetition quality |

Weights are normalized and category-aware. Factual, context, memory, and safety cases assign more weight to task, semantic, and reference evidence. Naturalness, empathy, and playful roleplay assign more weight to readable, socially natural output. The source of truth is `configs/quality_evaluation.json`.

A case passes only when its weighted score is at least `0.80`, it has no hard failure, and it has no major issue. A hard or critical failure forces the case score to `0.0`; it cannot be averaged away by correct length or punctuation.

## Hard intelligibility vetoes

The evaluator now detects and records distinct failure types, including:

- Unicode replacement and invalid control characters;
- consecutive token, single-token, repeated n-gram, and character-loop collapse;
- excessive symbol corruption;
- fragmented micro-line or line-collapse output;
- abnormal tokenizer-fragmentation ratios;
- extreme or corrupted mixed-script output while preserving legitimate multilingual replies;
- benchmark-local lexical word salad;
- malformed punctuation inside word fragments;
- combined low-alignment word-salad patterns;
- empty or nonlinguistic output.

The pasted failure examples that previously received `0.667` or `1.000` now produce `quality_score = 0.0` and explicit hard-failure labels.

## Diverse semantic and behavioral checks

The semantic audit now separates critical, major, and warning-level issues. It includes:

- explicit required-content and forbidden-content enforcement;
- required clarification and unwanted follow-up detection;
- excessive user echo;
- prompt, role, instruction, and reasoning leakage;
- unsupported certainty in knowledge-boundary cases;
- gas-leak and medication safety checks;
- empathy contradictions and inappropriate empathy refusal;
- interpersonal-conflict escalation;
- role-identity breaks;
- weak apology-repair warnings;
- generated placeholders and generic canned-assistant warnings.

These rules are deterministic safeguards, not a learned factual judge. Human review remains mandatory for factual correctness, subtle relevance, social nuance, and rare safety cases.

## Aggregate quality contract

Each summary now includes:

- mean quality and the historical legacy score side by side;
- quality-pass and intelligibility-pass rates;
- hard-, critical-, major-, and warning-case counts by type;
- minimum, P10, and worst-decile quality;
- mean score for every quality dimension;
- category quality, pass, intelligibility, semantic, completion, alignment, critical, major, and hard-failure metrics;
- reference token-F1, fuzzy similarity, anchor recall, and combined alignment when references exist.

The ranking-only quality index is deliberately tail-aware:

- 45% mean quality;
- 20% worst-decile quality;
- 15% quality-pass rate;
- 10% semantic-pass rate;
- 10% reference alignment.

Hard failures remain independent vetoes rather than part of this average.

## Independent category gates

Repository gates no longer examine only a category's weighted mean. They also compare category-specific:

- semantic-pass rates;
- complete-utterance rates;
- quality-pass rates;
- reference-alignment scores.

This prevents a good global average from hiding one damaged behavior class.

## Phase-by-phase corrections

| Area | Correction |
|---|---|
| Phase 1 | Base/SFT evaluation writes v2 evidence; gate checks mean gain, quality index, tail, pass, intelligibility, semantics, categories, completeness, brevity, critical and major issues. |
| Phase 2 static | Layer rows carry v2 fields; stale cached rows are rebuilt; ranking prioritizes hard/critical/major failures, tail quality, pass rate, mean quality, and resource reduction. |
| Phase 2B | Progressive deletion requires a valid v2 reference, zero hard/new critical/new major collapse, tighter quality/NLL/category floors, and stale-cache invalidation on resume. |
| Phase 2C | Selective recovery uses v2 mean/index/tail/category semantic and completion evidence, NLL, trainable-state savings, and zero-tolerance new issues. |
| Phase 3 | DPO must preserve v2 mean, index, tail, category, semantic, alignment, completion, and intelligibility quality. |
| Phase 4 | Greedy soups reject ineligible ingredients, preserve quality-v2 floors, and print the exact regressed category in rejection reasons. |
| Phase 4B | Structural candidates are evaluated with v2; recommendation requires valid dense evidence and independent tail/category/hard-failure gates. |
| Phase 5 | Compact and full-context replies are scored case by case with v2; retrieval success cannot hide reply-quality loss. |
| Phase 6 | Dense/masked sparsity simulations compare v2 quality, categories, tail, hard failures, NLL-like evidence, and no longer imply speed from oracle masking. |
| Phase 7 | Dense and sparse runtime replies receive identical v2 scoring; sparse enablement requires both measured runtime value and strict quality preservation. |
| Phase 8 | Sealed release audit requires v2, strict response control, minimum mean/pass/tail/semantic/completion quality, and zero hard, critical, major, and category-hard failures. |
| Playground | Version 1.1 refuses a collapsed/stale reference and produces an exploratory PASS/FAIL plus exact rejection reasons for every variant. |

## Evidence migration rule

Legacy summaries are intentionally rejected by official gates. This is a measurement change, not a training change:

- Existing adapters and checkpoints do **not** require retraining solely because the evaluator changed.
- Existing models that will be used for decisions must be re-evaluated with v2.
- Old Phase 1–8 or playground `summary.json` files must not be copied forward as v2 evidence.
- Resume logic rebuilds stale generation summaries where implemented.
- Use `python .\scripts\common\audit_quality_evidence.py --strict` to locate stale evidence.

## Verification boundary

The repository includes deterministic tests for healthy replies, Unicode corruption, repetition loops, mixed-script corruption, fragmented output, prompt leakage, and missing clarification. These tests validate code behavior only. Actual CUDA generation, model quality, latency, VRAM, and human preference must still be measured on the target Windows/RTX environment.
