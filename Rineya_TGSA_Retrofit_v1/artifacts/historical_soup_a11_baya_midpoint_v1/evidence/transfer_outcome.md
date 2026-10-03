# Historical soup transfer outcome

Result: **TRANSFER_POSITIVE**

| Split | Parent hits | Candidate hits | Delta | Parent all-four | Candidate all-four |
|---|---:|---:|---:|---:|---:|
| showcase | 45 | 44 | -1 | 7 | 7 |
| stability | 125 | 129 | +4 | 13 | 20 |
| all256 | 170 | 173 | +3 | 20 | 27 |

## Frozen transfer checks

- strict_all256_gain: **PASS**
- stability_preserved: **PASS**
- all_four_packs_preserved: **PASS**
- coverage_and_identity: **PASS**
- candidate_generation_health: **PASS**
- portable_policy_equivalence: **PASS**
- fresh_process_reload: **PASS**
- artifact_provenance_complete: **PASS**

## Production warning

Aggregate gain is driven by unanswerable behavior; answerable base/paraphrase regressions require the separate no-blocking-regression shipping gate.
