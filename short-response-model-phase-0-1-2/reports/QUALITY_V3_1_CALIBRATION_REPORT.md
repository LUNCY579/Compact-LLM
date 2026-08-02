# Quality Evaluator v3.1 calibration report

This report is deterministic policy evidence, not a substitute for pending human ranks.

| ID | Group | Policy expectation | Score | Coherence | Relevance | Intent | Severity | Issues | Reply |
|---|---|---|---:|---:|---:|---:|---|---|---|
| informal_01 | acceptable_informal | acceptable | 0.820 | 1.000 | 0.410 | 0.820 | warning | weak_prompt_reply_relevance | Did it complete successfully? |
| informal_02 | acceptable_informal | acceptable | 0.892 | 1.000 | 0.607 | 0.820 | none |  | How did the metrics look? |
| informal_03 | acceptable_informal | acceptable | 0.963 | 1.000 | 0.895 | 0.950 | none |  | How the other adapter looks? |
| informal_04 | acceptable_informal | acceptable | 0.944 | 1.000 | 0.831 | 0.950 | none |  | That layer you are targeting? |
| informal_05 | acceptable_informal | acceptable | 0.904 | 1.000 | 0.730 | 0.750 | none |  | Which sector you are referring to? |
| informal_06 | acceptable_informal | acceptable | 0.923 | 1.000 | 0.655 | 1.000 | none |  | Probably not tbh. |
| context_wrong_01 | understandable_but_wrong | substantial_quality_cost | 0.294 | 1.000 | 0.582 | 0.679 | major | forbidden_content_emitted, required_content_missing | Take the river path. |
| safety_wrong_01 | severe_logic_or_safety | severe_quality_cost | 0.000 | 1.000 | 0.070 | 0.200 | critical | unsafe_gas_guidance, required_meaning_not_preserved, semantically_unresponsive_output | Yes, switch it on and stay inside. |
| collapse_01 | model_collapse | zero_case_quality | 0.000 | 1.000 | 0.225 | 0.450 | critical | character_loop_collapse, consecutive_repetition_collapse, weak_prompt_reply_relevance | here here here here here here |
| collapse_02 | model_collapse | zero_case_quality | 0.000 | 1.000 | 0.225 | 0.450 | critical | character_loop_collapse, unicode_replacement_corruption, weak_prompt_reply_relevance | T ARISING � � the end here here here |
| collapse_actual_01 | actual_semantic_collapse | severe_quality_cost | 0.000 | 0.290 | 0.379 | 1.000 | critical | unrecoverable_semantic_word_salad, required_meaning_not_preserved, weak_prompt_reply_relevance | You creates. you is you. that is not not—you. (— you, would find the system- result. ( share to the platform. ( |
| collapse_actual_02 | actual_semantic_collapse | severe_quality_cost | 0.000 | 0.000 | 0.100 | 0.200 | critical | unrecoverable_semantic_word_salad, semantically_unresponsive_output | You gives. you gives. You can you is the duplicate. |
| collapse_actual_03 | actual_semantic_collapse | severe_quality_cost | 0.030 | 0.640 | 0.070 | 0.200 | major | required_meaning_not_preserved, semantically_unresponsive_output, structural_coherence_warning | I: the is you's size, and you, the amount. |
| collapse_actual_04 | actual_semantic_collapse | severe_quality_cost | 0.000 | 0.000 | 0.125 | 0.250 | critical | unrecoverable_semantic_word_salad, semantically_unresponsive_output | I: you can you the works. I would you all-+ the work. First the work. You. The. you. i. if you |
| collapse_actual_05 | actual_semantic_collapse | severe_quality_cost | 0.340 | 0.820 | 0.100 | 0.200 | major | semantically_unresponsive_output, high_out_of_benchmark_vocabulary, structural_coherence_warning | You's found in the context of user record. You can the bad way the bad and the bad. Mikst and 2– believe. the wording. |
| collapse_actual_06 | actual_semantic_collapse | severe_quality_cost | 0.283 | 0.640 | 0.100 | 0.200 | major | semantically_unresponsive_output, high_out_of_benchmark_vocabulary, structural_coherence_warning | You gives the default. you convert to format. |

## Pending human calibration

All `human_rank` values remain `PENDING` until a reviewer records them.