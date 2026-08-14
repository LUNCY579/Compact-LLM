# Phase 2 Data Independence and Template-Family Audit

**Status: complete descriptive sensitivity audit.**

The dataset contains unique IDs and no duplicate result keys, but deterministic generation creates repeated task structures. This report distinguishes row uniqueness from statistical independence and adds cluster-bootstrap sensitivity analyses. It does not estimate a single definitive effective sample size.

## Exact and normalized duplication

| Split | Rows | Unique IDs | Exact normalized prompts | Duplicate exact-prompt rows | Value-normalized prompt forms | Duplicate value-normalized rows | Largest normalized cluster |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 32,400 | 32,400 | 32,400 | 0 | 3,079 | 29,321 | 80 |
| DEV | 4,050 | 4,050 | 4,050 | 0 | 2,701 | 1,349 | 10 |
| TEST | 4,050 | 4,050 | 4,050 | 0 | 2,755 | 1,295 | 10 |

Exact normalized prompt overlap across splits: **0** forms. Value-normalized template overlap across all three splits: **0** forms.

## Lexical nearest-neighbor similarity

Prompts are lowercased and numeric/scenario values are replaced before word 1–2 gram TF-IDF similarity. This intentionally detects structural lexical reuse and is not semantic equivalence.

| Search | Mean | Median | P90 | P95 | P99 | ≥0.95 | ≥0.99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TEST → TRAIN/DEV | 0.305 | 0.302 | 0.424 | 0.589 | 0.594 | 0 | 0 |
| TEST → other TEST | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 | 3,214 | 2,057 |

## Paired effect sensitivity to clustered resampling

The original example bootstrap samples rows. The sensitivity intervals below resample entire clusters with replacement and retain all examples within a sampled cluster.

| Comparison | Cluster unit | Clusters | Size range | Micro difference | Macro-cluster difference | 95% cluster-bootstrap CI |
|---|---|---:|---:|---:|---:|---:|
| hard_k7 − track_a | reasoning_family | 15 | 270–270 | +2.67 pp | +2.67 pp | [-4.52 pp, +13.70 pp] |
| hard_k7 − probability_k7 | reasoning_family | 15 | 270–270 | +3.16 pp | +3.16 pp | [-1.63 pp, +6.89 pp] |
| hard_k7 − track_a | domain | 27 | 150–150 | +2.67 pp | +2.67 pp | [+1.53 pp, +3.75 pp] |
| hard_k7 − probability_k7 | domain | 27 | 150–150 | +3.16 pp | +3.16 pp | [+2.35 pp, +4.00 pp] |
| hard_k7 − track_a | reasoning×difficulty×decision×verification-shape | 114 | 1–135 | +2.67 pp | -2.19 pp | [-2.62 pp, +9.26 pp] |
| hard_k7 − probability_k7 | reasoning×difficulty×decision×verification-shape | 114 | 1–135 | +3.16 pp | +8.45 pp | [+0.33 pp, +5.80 pp] |

## Interpretation

- Row-level uniqueness is verified; independence is not.
- Numeric/value-normalized prompts reveal repeated lexical structures by design.
- Reasoning-family and structural-template cluster intervals quantify how sensitive the paired headline is to treating task families, rather than examples, as the resampling unit.
- A cluster interval containing zero does not negate the observed finite-corpus difference; it limits generalization to a population of unseen task/template families.
- Structural signatures are audit heuristics, not ground-truth latent-task labels. The appendix should report results under multiple plausible cluster definitions.

## Provenance

- Raw result SHA256: `d3733391b53dc53e7d9ca899d7015ba6ff0b2faa7c28e52a2778a0827c45c32d`
- TEST dataset SHA256: `768107b579126063d4dd304b80ffc88a57e1bc346a7adfe4b69771e1b499a8f4`
- Analysis script SHA256: `5e0f4772643144615f1ac2bd88df3d66cc3f622010fcf58bca980fbf9dd47917`
