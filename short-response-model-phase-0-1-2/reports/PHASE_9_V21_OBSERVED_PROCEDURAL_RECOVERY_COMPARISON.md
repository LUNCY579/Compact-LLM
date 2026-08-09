# v2.1 Procedural Recovery Comparison

**Status:** `PROMISING_RECOVERY`

## Original Phase 9 dev decomposition

- Legacy Real-RAG procedure correctness: `0.0`
- Same weights + query-aware v2.1 retrieval: `0.9533333333333334`
- Recovered weights + query-aware retrieval: `0.22666666666666666`
- Retrieval-policy gain: `0.9533333333333334`
- Training-only gain: `-0.7266666666666667`
- Combined gain vs historical legacy path: `0.22666666666666666`
- Zero-correct-procedure blocker resolved: `True`

## Preservation cost

| Metric | Before | After | Relative loss |
|---|---:|---:|---:|
| `supported_answer_correctness` | 0.6339622641509434 | 0.48867924528301887 | 0.229167 |
| `correct_abstention_rate` | 0.8129496402877698 | 0.8057553956834532 | 0.008850 |
| `conflict_safe_handling_rate` | 1.0 | 1.0 | 0.000000 |
| `evidence_faithfulness` | 0.979381443298969 | 0.8392857142857143 | 0.143045 |
| `social_usable_rate` | 1.0 | 1.0 | 0.000000 |

Mean preservation relative loss: `0.0762122673941491`
Mean positive procedural training gain: `0.07638888888888888`
Relative training opportunity margin: `0.00017662149473977873`

## v2.1 procedural challenge dev

- Baseline: `0.0015432098765432098`
- Recovered: `0.15432098765432098`
- Training gain: `0.15277777777777776`

The report deliberately separates retrieval and learned-weight gains. A positive retrieval-only gain is useful engineering evidence, but it is not counted as learned procedural recovery.
