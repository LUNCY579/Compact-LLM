# Phase 2 Cornerstone Verification

**Status: PASS**

This report independently re-audits the complete v2.2 TEST result from the sealed JSONL, the official reviewed split, frozen configuration, run manifest, source manifests, implementation, and derived statistical artifacts. It does not treat the prior narrative report as evidence.

## Verification conclusion

The cornerstone passes the mandatory verification gate. The raw artifact contains exactly **4,050 unique TEST examples × 8 frozen conditions = 32,400 unique rows**, with no missing condition, no duplicate result key, exact dataset/config hash agreement, and zero independent scoring mismatches. Mechanistic analysis may proceed from this verified observation.

## Verified full-TEST result

| Condition | Correct | N | Exact accuracy |
|---|---:|---:|---:|
| Backbone K=1 | 1 | 4,050 | 0.025% |
| Abstention K=1 | 527 | 4,050 | 13.012% |
| Probability K=7 | 571 | 4,050 | 14.099% |
| Track A K=1 | 591 | 4,050 | 14.593% |
| Hard K=7 | 699 | 4,050 | 17.259% |

The independently recovered percentage-point differences are:

- Hard K=7 − Backbone: **+17.235 pp**
- Hard K=7 − Track A: **+2.667 pp**
- Hard K=7 − Abstention: **+4.247 pp**
- Hard K=7 − Probability K=7: **+3.160 pp**

## Paired full-TEST statistics

Exact McNemar p-values and Holm adjustment were recomputed from the raw paired correctness vectors using an independent implementation. Percentile intervals were independently checked with an equivalent multinomial row bootstrap (100,000 resamples); they agree with the stored 20,000-resample intervals within 3/N.

| Comparison (A − B) | Rescue A-only | Harm B-only | Difference | Independent 95% bootstrap CI | Exact McNemar p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| hard_k7 vs backbone | 699 | 1 | +17.235 pp | [+16.074, +18.420] pp | 2.665e-208 | 1.866e-207 |
| hard_k7 vs Track A | 328 | 220 | +2.667 pp | [+1.531, +3.802] pp | 4.543e-06 | 1.363e-05 |
| hard_k7 vs abstention specialist | 444 | 272 | +4.247 pp | [+2.963, +5.531] pp | 1.355e-10 | 5.421e-10 |
| hard_k7 vs probability mean | 226 | 98 | +3.160 pp | [+2.296, +4.025] pp | 8.715e-13 | 4.358e-12 |
| probability mean vs backbone | 571 | 1 | +14.074 pp | [+13.012, +15.160] pp | 7.414e-170 | 4.448e-169 |
| probability mean vs Track A | 307 | 327 | -0.494 pp | [-1.704, +0.741] pp | 0.450523 | 0.450523 |
| probability mean vs abstention specialist | 285 | 241 | +1.086 pp | [-0.025, +2.198] pp | 0.060705 | 0.121410 |

All four planned Hard-K7 comparisons shown above remain significant after Holm correction. Statistical significance is not treated as a substitute for practical effect size; the Track A and probability-mean gains are 2.667 pp and 3.160 pp, respectively.

## Completeness, scoring, decoding, and EOS audit

- Raw JSONL SHA256: `d3733391b53dc53e7d9ca899d7015ba6ff0b2faa7c28e52a2778a0827c45c32d`
- Rows / unique keys: **32,400 / 32,400**
- Unique TEST examples: **4,050**
- Every example has all 8 frozen committee-method conditions; no extra condition is present.
- Every raw prompt hash, gold field, domain, reasoning family, difficulty, and decision label matches the official TEST row.
- Independent exact-answer, abstention, and repetition scoring produced zero mismatches.
- K=1 hard and probability aggregation have identical token sequences on all 4,050 examples for backbone, Track A, and abstention controls.
- Raw step reconstruction found zero trace/vote/EOS mismatches across the run; Hard K=7 contains 23,204 generated steps, including 1,583 plurality-tie steps.
- All ordinary-EOS rows stop at the first winning EOS; maximum generated length is 32 tokens.
- Tie traces verify plurality counts, tied-token membership, and the frozen `sum_log_probability_then_smallest_token_id` rule label. Exact tie-score arithmetic is verified by implementation tests because full tied-candidate log probabilities are not retained in the compressed trace.

## Frozen configuration and committee

- Canonical parsed config SHA256: `e72042083e464f5c043ee190e590f92f97e8a907632db471371f3df551adffce`
- Frozen YAML SHA256: `cbef7f7d828da3cb1c8b17ae94427685ea9f43ad6553f698b154115ae3696b36`
- Frozen manifest SHA256: `bdfe566c051228137f85e0a8161d2e7390c8e7d000d62d910d6510d906e470b1`
- Decoder: full-prefix no-cache reference decoder.
- EOS: ordinary token; no override.
- Max new tokens: 32.
- Hard K=7 members: Track A, Track B, General, Grounding, Procedure, Abstention, Social.

| Expert | Rank | PEFT | Adapter config SHA256 | Adapter weights SHA256 |
|---|---:|---|---|---|
| track_a | 16 | LORA | `3a89fc081824fcf511a0c9c83128bd5b0212ba6ecd35a05c997f53cc2b2b1072` | `e09e4f05fc69e2b4ed446178d137e6e7feb89500c02860a6a92fc15d574dfcfc` |
| track_b | 16 | LORA | `7ec285e8308a6194e30ef20789ee393cd664516786c0c2bcd8bb3b27835515b7` | `54cdaf75f0902a959c4521d3624b86136f3575fdee5654da642a4cf4b17c8eee` |
| general | 16 | LORA | `ea17ef3ad52bc8bcb962cf13a868886f23349b082f1c9004134c380009b2eddb` | `d002fbb4b3548c59ad955a27a3100d4024296dca00ba49df8e63c8d176fd1c7b` |
| grounding | 16 | LORA | `4f58d0755f160ac639278a1929e841cc85e67ec8b0688c6a0605aa9d2209e6c8` | `138e54b0efa35c3be19f1380a21629541db8b34659651dc1b6fb74a2319a1b02` |
| procedure | 16 | LORA | `494acf500deebb60da3dcfc7f7fd5d4b1636a9b24eaf6fc4524909b9b0c66ff2` | `c76a4a0bd25cb7d276145c15460205a80a9e2badbdb28a9987073d3d9b4cd19b` |
| abstention | 16 | LORA | `33ed7b1707f08e1d1bd83f3d8f29a435ffb83c1556665167895c97bfaebead46` | `5ccf002233bfd32b426f3808236dd9b728fd69b5afc9a380bbe53a3b2a967ace` |
| social | 16 | LORA | `79352ff036fb0c3056904301c33cbea8394ba417cf000d407a7d59754c91e947` | `7a52ae04cc99050431fa35b5e2046e7d6b74f6e16db788679692264abea72d87` |

All seven adapters share rank, PEFT type, target modules, and declared base; their config and weight hashes match the frozen source manifest.

## Dataset provenance

| Split | Role in Phase 2 | Rows | Unique IDs | SHA256 |
|---|---|---:|---:|---|
| TRAIN | TRAIN MECHANISTIC / EXPLORATORY | 32,400 | 32,400 | `8f4cc601c7c7f3c414643d8ef1dfec1f63da48b0ddfe327e6ad6d006c1da6d87` |
| DEV | DEV MODEL-SELECTION / CALIBRATION | 4,050 | 4,050 | `11f58cc066e5650a5d4aac448f7a565a7dde007884110652726e58fee04ddb7a` |
| TEST | TEST CONFIRMATORY (with prior n=32 exposure caveat) | 4,050 | 4,050 | `768107b579126063d4dd304b80ffc88a57e1bc346a7adfe4b69771e1b499a8f4` |

The three split ID sets are disjoint. The frozen TEST SHA256 matches the official reviewed TEST file exactly.

## Pilot conclusions versus full-corpus conclusions

| Condition | n=32 pilot | Full N=4,050 | Change |
|---|---:|---:|---:|
| Backbone K=1 | 0.000% | 0.025% | +0.025 pp |
| Track A K=1 | 15.625% | 14.593% | -1.032 pp |
| Abstention K=1 | 50.000% | 13.012% | -36.988 pp |
| Probability K=7 | 37.500% | 14.099% | -23.401 pp |
| Hard K=7 | 25.000% | 17.259% | -7.741 pp |

The feasibility pilot ranked Probability K=7 above Hard K=7 (37.5% vs 25.0%) and made the abstention specialist appear strongest (50.0%). The full corpus reverses both impressions: Hard K=7 reaches 17.259%, probability mean 14.099%, and abstention 13.012%. The pilot remains preserved at `results/archive/sealed_n32/`; it is evidence about feasibility and small-sample instability, not a result to ridicule or erase.

## Selection-exposure and provenance caveats

1. The same canonical method/config hash was used for the n=32 and N=4,050 runs, so the committee, aggregation methods, decoding, and EOS policy did not change after the pilot. The full-run manifest was frozen before the full artifact completed.
2. Nevertheless, the full TEST split was not wholly untouched: the earlier 32-example sealed sample came from this TEST split and was analyzed before the full-corpus run. Paper language must disclose this prior TEST exposure and should reserve newly selected mechanisms/thresholds for DEV before any further TEST confirmation.
3. `reports/FINAL_REPORT.md` is the historical n=32 report and is not authoritative for the full-corpus ranking. The raw full artifact and the full `sealed_statistics.json` are authoritative for the cornerstone verified here.
4. The before/after source snapshots bracketed the original work rather than the later 4,050 completion. This audit rehashed every one of their 24 protected source entries against the current official files; all still match exactly.
5. Example-level tests assume independent pairs, but the dataset is deterministic/templated. Phase 2 must add template-family and near-duplicate sensitivity analyses before translating nominal N into claims about effective independent sample size.

## Check ledger

| Check | Status |
|---|---|
| `source_manifest_internal_hashes` | PASS |
| `source_before_after_identical` | PASS |
| `source_manifest_matches_current_files` | PASS |
| `official_git_commit_context_recorded` | PASS |
| `protected_repositories_tracked_clean` | PASS |
| `frozen_config_hash_chain` | PASS |
| `pilot_and_full_method_config_identical` | PASS |
| `full_k7_members_exact` | PASS |
| `decoder_and_eos_frozen` | PASS |
| `adapter_hashes_and_compatibility` | PASS |
| `full_manifest_frozen_before_result_completion` | PASS |
| `dataset_train_count_and_ids` | PASS |
| `dataset_dev_count_and_ids` | PASS |
| `dataset_test_count_and_ids` | PASS |
| `dataset_split_ids_disjoint` | PASS |
| `configured_test_path_exact` | PASS |
| `frozen_test_hash_exact` | PASS |
| `raw_row_count_complete` | PASS |
| `raw_keys_unique_and_conditions_complete` | PASS |
| `raw_example_ids_match_test` | PASS |
| `raw_frozen_metadata_exact` | PASS |
| `independent_output_scoring_exact` | PASS |
| `trace_vote_and_eos_consistency` | PASS |
| `k1_hard_probability_invariant` | PASS |
| `raw_result_hash_matches_completion_audit` | PASS |
| `paired_statistics_independently_reproduced` | PASS |
| `statistics_raw_hash_exact` | PASS |
| `derived_summary_matches_raw` | PASS |
| `n32_pilot_preserved_and_complete` | PASS |
| `implementation_static_contract` | PASS |

## Reproduction

From the lab root:

```powershell
$Python = "..\.venv\Scripts\python.exe"
& $Python experiments\phase2_mechanism\verify_cornerstone.py
& $Python scripts\analyze_sealed.py
& $Python -m pytest -q
```

Machine-readable audit: `reports/phase2_cornerstone_verification.json`.
