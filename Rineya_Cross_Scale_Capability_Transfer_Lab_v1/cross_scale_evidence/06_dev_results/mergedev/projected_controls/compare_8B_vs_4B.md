# projected_8B vs projected_4B

- projected_4B: 0.712
- projected_8B: 0.729
- Delta (projected_8B - projected_4B): +0.017
- Paired discordance: projected_4B-only 1, projected_8B-only 2
- Exact McNemar p: 1.0000
- Scenario bootstrap 95% CI: [-0.014, +0.102]
- Responses changed: 15 / 59

## Families

| Family | Items | projected_4B | projected_8B | Delta |
|---|---:|---:|---:|---:|
| common_sense | 4 | 0.500 | 0.500 | +0.000 |
| counterfactual_reasoning | 4 | 1.000 | 1.000 | +0.000 |
| general_reasoning | 4 | 1.000 | 1.000 | +0.000 |
| grounded_qa_rag | 4 | 1.000 | 1.000 | +0.000 |
| instruction_following | 4 | 1.000 | 1.000 | +0.000 |
| long_context_retrieval | 2 | 1.000 | 1.000 | +0.000 |
| lost_in_the_middle | 1 | 0.000 | 0.000 | +0.000 |
| mathematics | 4 | 0.000 | 0.000 | +0.000 |
| multi_hop_binding | 4 | 0.750 | 1.000 | +0.250 |
| normal_conversation | 4 | 0.750 | 1.000 | +0.250 |
| paraphrase_robustness | 12 | 0.333 | 0.250 | -0.083 |
| science | 4 | 1.000 | 1.000 | +0.000 |
| short_response | 4 | 1.000 | 1.000 | +0.000 |
| unanswerable_abstention | 4 | 1.000 | 1.000 | +0.000 |
