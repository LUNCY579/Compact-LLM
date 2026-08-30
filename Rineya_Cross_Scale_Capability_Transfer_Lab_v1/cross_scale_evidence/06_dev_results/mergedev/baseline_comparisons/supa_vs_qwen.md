# supa_ungated vs qwen06

- qwen06: 0.712
- supa_ungated: 0.797
- Delta (supa_ungated - qwen06): +0.085
- Paired discordance: qwen06-only 7, supa_ungated-only 12
- Exact McNemar p: 0.3593
- Scenario bootstrap 95% CI: [-0.170, +0.102]
- Responses changed: 41 / 59

## Families

| Family | Items | qwen06 | supa_ungated | Delta |
|---|---:|---:|---:|---:|
| common_sense | 4 | 0.500 | 0.500 | +0.000 |
| counterfactual_reasoning | 4 | 1.000 | 0.500 | -0.500 |
| general_reasoning | 4 | 0.750 | 1.000 | +0.250 |
| grounded_qa_rag | 4 | 1.000 | 1.000 | +0.000 |
| instruction_following | 4 | 1.000 | 1.000 | +0.000 |
| long_context_retrieval | 2 | 1.000 | 1.000 | +0.000 |
| lost_in_the_middle | 1 | 0.000 | 1.000 | +1.000 |
| mathematics | 4 | 0.000 | 0.000 | +0.000 |
| multi_hop_binding | 4 | 1.000 | 1.000 | +0.000 |
| normal_conversation | 4 | 1.000 | 0.000 | -1.000 |
| paraphrase_robustness | 12 | 0.250 | 1.000 | +0.750 |
| science | 4 | 1.000 | 1.000 | +0.000 |
| short_response | 4 | 1.000 | 1.000 | +0.000 |
| unanswerable_abstention | 4 | 1.000 | 1.000 | +0.000 |
