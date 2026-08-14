# Phase 2 Cross-Benchmark Verification

Status: **PASS_WITH_CAVEATS** (27/27 checks)

The v2.1.2 TEST split is treated as an independent cross-benchmark holdout.
No Phase 2 inference may begin unless all mandatory checks pass. Caveats
limit the eventual claim but do not invalidate fixed-method cross-corpus evaluation.

| Split | Rows | Unique IDs | SHA256 |
|---|---:|---:|---|
| TRAIN | 10,560 | 10,560 | `b2eacf4435bfa7ef617639ed3d9626891fecae012bd4265ec5e0e9e7f4cdab5b` |
| DEV | 1,320 | 1,320 | `c6619bdc3c492355d8e14a4a2b4350f9bae41a060649a6f0834ad7c8f7910f90` |
| TEST | 1,320 | 1,320 | `9333a3ae97c3c70f81a8998f88611d4224fc23e979edae0f69f860fc9c7c6cce` |

## Independence checks

- TEST-v2.2 ID overlap: **0**
- TEST-v2.2 exact normalized prompt overlap: **0**
- TEST-v2.2 value-normalized template overlap: **0** (descriptive only)
- Prior lab references to v2.1.2 TEST: **0**
- TEST rows repeating an exact v2.1.2 TRAIN prompt+answer: **1,008/1,320**
- Historical official TEST use by another model: **True**
- Seven adapter training manifests point to v2.2 data: **PASS**

## Claim boundary

The permitted claim is *fixed-method cross-corpus transfer to v2.1.2 TEST*.
Do not describe this split as globally untouched, template-independent, or a test
of generalization beyond all v2.1.2 training forms.

## Caveats

- v2.1.2 TEST contains exact prompt-and-answer repetitions from v2.1.2 TRAIN; this does not leak into the seven v2.2-trained adapters, but prevents a template-independent holdout claim.
- The official project previously evaluated a different selected D_greedy dense model on v2.1.2 TEST; the split is therefore not globally untouched.

## Check ledger

| Check | Result | Evidence |
|---|---|---|
| manifest schema | PASS | `"2.1.2"` |
| manifest deterministic seed | PASS | `21242` |
| manifest split counts | PASS | `{"dev": 1320, "test": 1320, "train": 10560}` |
| manifest canonical row count | PASS | `13200` |
| manifest family count | PASS | `33` |
| train row count | PASS | `10560` |
| train frozen hash | PASS | `"b2eacf4435bfa7ef617639ed3d9626891fecae012bd4265ec5e0e9e7f4cdab5b"` |
| train unique IDs | PASS | `10560` |
| train nonempty answers | PASS | `10560` |
| train final assistant message | PASS | `10560` |
| dev row count | PASS | `1320` |
| dev frozen hash | PASS | `"c6619bdc3c492355d8e14a4a2b4350f9bae41a060649a6f0834ad7c8f7910f90"` |
| dev unique IDs | PASS | `1320` |
| dev nonempty answers | PASS | `1320` |
| dev final assistant message | PASS | `1320` |
| test row count | PASS | `1320` |
| test frozen hash | PASS | `"9333a3ae97c3c70f81a8998f88611d4224fc23e979edae0f69f860fc9c7c6cce"` |
| test unique IDs | PASS | `1320` |
| test nonempty answers | PASS | `1320` |
| test final assistant message | PASS | `1320` |
| train/dev ID disjoint | PASS | `0` |
| train/test ID disjoint | PASS | `0` |
| dev/test ID disjoint | PASS | `0` |
| v2.1.2 TEST/v2.2 all ID disjoint | PASS | `0` |
| v2.1.2 TEST/v2.2 all exact-prompt disjoint | PASS | `0` |
| no prior lab TEST reference | PASS | `[]` |
| all seven adapter training files are v2.2, not v2.1.2 | PASS | `{"track_a_broad_reasoning": "data\\phase_9_student_distillation\\multidomain_reasoning_v22\\curriculum_views\\track_a_broad_general.reviewed.jsonl", "track_b_anchored_reasoning": "data\\phase_9_student_distillation\\multidomain_reasoning_v22\\curriculum_views\\track_b_anchored_mixed.reviewed.jsonl", "track_c_general": "data\\phase_9_student_distillation\\multidomain_reasoning_v22\\curriculum_views\\specialist_general.reviewed.jsonl", "track_c_grounding": "data\\phase_9_student_distillation\\multidomain_reasoning_v22\\curriculum_views\\specialist_grounding.reviewed.jsonl", "track_c_procedure": "data\\phase_9_student_distillation\\multidomain_reasoning_v22\\curriculum_views\\specialist_procedure.reviewed.jsonl", "track_c_abstention": "data\\phase_9_student_distillation\\multidomain_reasoning_v22\\curriculum_views\\specialist_abstention.reviewed.jsonl", "track_c_social": "data\\phase_9_student_distillation\\multidomain_reasoning_v22\\curriculum_views\\specialist_social.reviewed.jsonl"}` |
