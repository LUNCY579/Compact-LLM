# projected_8B vs qwen06

- qwen06: 0.712
- projected_8B: 0.729
- Delta (projected_8B - qwen06): +0.017
- Paired discordance: qwen06-only 0, projected_8B-only 1
- Exact McNemar p: 1.0000
- Scenario bootstrap 95% CI: [+0.000, +0.061]
- Responses changed: 9 / 59

## Families

| Family | Items | qwen06 | projected_8B | Delta |
|---|---:|---:|---:|---:|
| common_sense | 4 | 0.500 | 0.500 | +0.000 |
| counterfactual_reasoning | 4 | 1.000 | 1.000 | +0.000 |
| general_reasoning | 4 | 0.750 | 1.000 | +0.250 |
| grounded_qa_rag | 4 | 1.000 | 1.000 | +0.000 |
| instruction_following | 4 | 1.000 | 1.000 | +0.000 |
| long_context_retrieval | 2 | 1.000 | 1.000 | +0.000 |
| lost_in_the_middle | 1 | 0.000 | 0.000 | +0.000 |
| mathematics | 4 | 0.000 | 0.000 | +0.000 |
| multi_hop_binding | 4 | 1.000 | 1.000 | +0.000 |
| normal_conversation | 4 | 1.000 | 1.000 | +0.000 |
| paraphrase_robustness | 12 | 0.250 | 0.250 | +0.000 |
| science | 4 | 1.000 | 1.000 | +0.000 |
| short_response | 4 | 1.000 | 1.000 | +0.000 |
| unanswerable_abstention | 4 | 1.000 | 1.000 | +0.000 |
