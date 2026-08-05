# MODEL SOUP REPORT

## Result

- Accepted type: `greedy_uniform_soup`
- Selected ingredients: `seed_37, seed_49`
- Full checkpoint: `C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2\artifacts\phase_4_model_soups_phase2c_balanced_training\accepted_model`
- Mean quality score: `0.701315`
- Quality pass rate: `0.54`
- Worst-decile quality: `0.2684545454545454`
- Hard-failure cases: `0`
- Critical failures: `0`

## Greedy trials

| Added ingredient | Accepted | Trial score | Decision reason |
|---|---:|---:|---|
| seed_49 | YES | 0.7013154545454544 | positive net relative quality change |
| seed_23 | NO | 0.7013154545454544 | relative quality gains do not exceed relative quality losses; net 0.000000 |
| phase3_primary | NO | 0.7013154545454544 | relative quality gains do not exceed relative quality losses; net 0.000000 |

## Operational interpretation

The accepted checkpoint is a normal full Hugging Face model, not a PEFT adapter. Phase 4B should consume this exact directory after the Phase 4 human gate is approved.