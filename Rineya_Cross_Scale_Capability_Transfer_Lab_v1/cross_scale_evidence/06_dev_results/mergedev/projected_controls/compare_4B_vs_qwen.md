# projected_4B vs qwen06

- qwen06: 0.712
- projected_4B: 0.712
- Delta (projected_4B - qwen06): +0.000
- Paired discordance: qwen06-only 2, projected_4B-only 2
- Exact McNemar p: 1.0000
- Scenario bootstrap 95% CI: [-0.082, +0.054]
- Responses changed: 17 / 59

## Families

| Family | Items | qwen06 | projected_4B | Delta |
|---|---:|---:|---:|---:|
| common_sense | 4 | 0.500 | 0.500 | +0.000 |
| counterfactual_reasoning | 4 | 1.000 | 1.000 | +0.000 |
| general_reasoning | 4 | 0.750 | 1.000 | +0.250 |
| grounded_qa_rag | 4 | 1.000 | 1.000 | +0.000 |
| instruction_following | 4 | 1.000 | 1.000 | +0.000 |
| long_context_retrieval | 2 | 1.000 | 1.000 | +0.000 |
| lost_in_the_middle | 1 | 0.000 | 0.000 | +0.000 |
| mathematics | 4 | 0.000 | 0.000 | +0.000 |
| multi_hop_binding | 4 | 1.000 | 0.750 | -0.250 |
| normal_conversation | 4 | 1.000 | 0.750 | -0.250 |
| paraphrase_robustness | 12 | 0.250 | 0.333 | +0.083 |
| science | 4 | 1.000 | 1.000 | +0.000 |
| short_response | 4 | 1.000 | 1.000 | +0.000 |
| unanswerable_abstention | 4 | 1.000 | 1.000 | +0.000 |
