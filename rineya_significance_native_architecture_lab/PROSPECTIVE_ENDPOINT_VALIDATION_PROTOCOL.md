# Prospective endpoint validation v1

Status: **stopped at the pre-compression full-cache capability gate**.

This validation was designed to test frozen endpoint
`layer-fact-retention-v1` on latent clusters that were absent from both the
endpoint-design pilot and the R1-R8 ladder. It did not reach B2/B3/B4 execution.

## Frozen pre-compression design

- model: published Rineya Halo Qwen3-0.6B core;
- families: R1, R3, R4;
- new latent seeds: 401, 409, 419, 421, 431, 433, 439, 443;
- evidence depth: 0.50;
- independent clusters: 24, eight per family;
- exact Halo-token evidence annotations;
- inference and answer/evidence views stored separately;
- B0 gate: at least 12 correct overall and at least 3 correct per family;
- intended primary endpoint, if the gate passed:
  `layer_fact_exact_completion_rate`;
- intended primary contrast: B3 minus B2 under matched B128+C128 execution;
- intended controls: B4 positional, identical query access, cluster sign-flip,
  and cluster bootstrap interval.

Authoritative design files:

- `configs/halo_endpoint_validation_dataset.yaml`
- `configs/halo_endpoint_validation_capability.yaml`
- `results/benchmarks/halo_endpoint_validation_v1/`

No compressed outcome was inspected before the gate decision.

## Gate result

| Family | B0 correct / clusters |
|---|---:|
| R1 | 1/8 |
| R3 | 1/8 |
| R4 | 7/8 |
| Overall | 9/24 |

The gate required 12 overall and 3 per family. It failed both the overall rule
and the R1/R3 family minima. Consequently:

```text
B2/B3/B4 NOT RUN
endpoint v1 NOT prospectively validated
no R4-only rescue analysis
```

Running compression only on the seven R4 successes would change the frozen
target population after observing capability and would not provide the planned
balanced validation.

The machine audit is
`results/statistics/halo_endpoint_validation_gate_v1.json`.

## Subsequent v2 outcome

V2 changed task difficulty/templates, used new latent clusters, and repeated
the B0 gate before any compressed outcome. Halo passed at `18/24`; 12 B0-
correct clusters then entered a timestamped local registration. Its B3-B2
endpoint contrast was `+0.0401786`, interval `[0.0133929, 0.0758929]`,
`p=0.01953125`; answers were B3 5/12 versus B2 6/12 (`p=1.0`). See
[ENDPOINT_VALIDATION_V2_RESULT.md](ENDPOINT_VALIDATION_V2_RESULT.md).

This v1 stop remains important provenance: v2 was difficulty-calibrated after
observing v1 capability, so the local result is not an untouched population or
public preregistration.
