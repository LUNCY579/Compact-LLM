# Phase 2 Secondary Semantic Audit

## Scope and status

The official exact-match score remains the primary benchmark metric. This
secondary audit asks only whether exact scoring hides semantically correct
answers in a controlled sample of errors.

- Population: full_k7 official exact-match failures on v2.2 TEST
- Sample: **100** exact-match failures, 50 per fusion method
- Selection: deterministic and stratified by decision and reasoning family
- Raw artifact SHA256: `d3733391b53dc53e7d9ca899d7015ba6ff0b2faa7c28e52a2778a0827c45c32d`
- Review: all prompt/gold/output triples manually inspected after selection was frozen

## Result

| Method | Audited failures | Semantic rescues | Rate | Exact 95% CI | Wrong content | False abstention | Failure to abstain | Malformed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hard K=7 | 50 | 0 | 0.00% | [0.00%, 7.11%] | 22 | 12 | 8 | 8 |
| Probability K=7 | 50 | 0 | 0.00% | [0.00%, 7.11%] | 8 | 26 | 8 | 8 |
| Combined | 100 | 0 | 0.00% | [0.00%, 3.62%] | 30 | 38 | 16 | 16 |

No audited exact-match failure was a formatting-only or paraphrase-only
success. For the combined sample, the zero-success exact-binomial interval
places the 95% upper bound at 3.62%.
This supports, but does not prove for all errors, that the reported hard/probability
difference is not an artifact of benign surface-form mismatches.

## Limitations

- This is a sample of failures, not a relabeling of the complete benchmark.
- One analysis agent performed the review; no independent inter-rater agreement is available.
- The sample is balanced by method, not weighted to estimate a pooled benchmark correction.
- The synthetic answer space makes many judgments clear, but namespace fragments such as
  `Atlas` versus `System-Atlas` were treated as incomplete/wrong because the prompt defines
  the prefixed system identifier as the requested value.
