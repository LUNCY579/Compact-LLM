# Registered endpoint validation v2 result

Status: **local prospective validation complete**.

The frozen registration preceded every compressed model run and passed all
post-execution hash, cluster, condition, and endpoint audits. This is stronger
than the earlier post-hoc pilot application, but it is a local 12-cluster
validation rather than a public preregistration or population claim.

## Registered design

- 12 independent B0-correct clusters: four each from R1, R3, and R4;
- difficulty-calibrated v2 templates with two R1 facts and three R3 updates;
- matched B128 retained history and C128 chunks;
- identical query access across B2, B3, and B4;
- primary endpoint: `layer_fact_exact_completion_rate`;
- sole primary contrast: B3 minus B2;
- exact cluster sign-flip test and cluster bootstrap interval;
- answer correctness and B3-B4 were secondary.

Registration:
`results/registrations/halo_endpoint_validation_v2.json`  
Execution audit:
`results/registrations/halo_endpoint_validation_v2_execution.json`

## Results

| Condition | Correct / clusters | Mean frozen endpoint |
|---|---:|---:|
| B2 raw accumulation | 6/12 | 0.172619 |
| B3 observation normalization | 5/12 | 0.212798 |
| B4 positional | 1/12 | 0.000000 |

Primary registered result:

```text
B3 - B2 endpoint = +0.0401786
95% cluster bootstrap interval = [0.0133929, 0.0758929]
exact two-sided cluster sign-flip p = 0.01953125
```

The primary endpoint therefore prospectively separates B3 from B2 in this
registered local sample.

Secondary answer result:

```text
B3 - B2 correctness = -0.0833333
B3 5/12, B2 6/12
one B3 loss, zero wins
exact two-sided p = 1.0
```

Thus better final evidence retention did not translate into better answers.
The strict every-token/every-layer diagnostic remained saturated at 0/12 for
all three policies.

## Interpretation

This result supports a narrow claim: observation normalization changes Halo's
retained evidence state in the predicted direction on an independent,
pre-frozen validation set. It does not establish answer-quality improvement,
causal mediation, broad task generalization, or a fundamental architecture
gain. The result is also conditioned on easier task templates introduced after
the first prospective dataset failed its B0 gate.

The execution was internally paired under Python 3.11.15,
Torch 2.13.0+cpu, and Transformers 5.15.0. The current project lock and later
Experiments B-D use Torch 2.12.1+cu126 and Transformers 5.14.1. This local
result has not been reproduced under that locked runtime; no cross-version
trajectory or quality equivalence is claimed.
