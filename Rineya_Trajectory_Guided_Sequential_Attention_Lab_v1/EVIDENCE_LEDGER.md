# Evidence ledger

This ledger records what has actually been demonstrated. It must not be used to list intended outcomes as though they were results.

## Current evidence

| Evidence ID | Question | Artifact/procedure | Result | Claim scope | Status |
|---|---|---|---|---|---|
| E0-TEST | Does the reference package satisfy its unit contracts? | `PYTHONPATH=. pytest -q` | 22 tests pass locally | small PyTorch reference only | passed |
| E0-CAUSAL | Can future suffix changes alter prefix outputs? | Phase-0 smoke suffix perturbation | maximum observed prefix error `0.0` in configured float32 run | configured synthetic shapes/seed only | passed |
| E0-GRAD | Does gradient reach the learned trajectory path? | backward test and smoke manifest | finite, nonzero trajectory-path gradient observed | configured reference operator only | passed |
| E0-ARMS | Do all registered engineering arms execute? | `run_ablation_matrix.py` | parallel, cascade, history-control, raw-delta, learned-delta, QKV, predictor–corrector, and full arms finite | execution only; no quality claim | passed |
| E0-CAP | Are history-control and learned-delta parameter matched? | direct trainable-parameter count unit test | exact equality | reference configuration family | passed |
| E0-LAT | Is the unfused reference competitive in latency? | local CPU smoke timing | no; approximately an order-of-magnitude slower in the observed tiny run | non-benchmark engineering warning | negative/expected |

Generated smoke artifacts are intentionally not committed as paper evidence because the local workspace has no authoritative repository commit identity. Re-run after checkout of the remote commit so the manifest records the exact SHA.

## Evidence levels not yet reached

| Level | Missing evidence |
|---|---|
| Mechanism benefit | trained tiny decoder, history/raw/recurrent/deeper controls, five seeds |
| Language-model benefit | held-out NLL/perplexity with matched data/compute |
| Long-context benefit | independently authored sealed bank and primary statistics |
| Efficiency benefit | optimized kernel and real prefill/decode Pareto measurements |
| Explanatory trajectory | held-out telemetry prediction plus causal interventions |
| Generalization | multiple scales, model families, and task families |

## Append-only experiment row template

```text
Evidence ID:
Hypothesis ID:
Branch / commit:
Resolved config SHA-256:
Base model / revision / tensor hash:
Dataset / split / content hash:
Tokenizer / revision:
Seed(s):
Arm and fairness view:
Trainable / total parameters:
Tokens / steps / training FLOPs:
Hardware / software / precision:
Primary metric and confidence interval:
Guardrails:
Prefill / decode latency:
Peak memory / KV bytes:
Raw artifact location:
Deviations or failures:
Conclusion and claim unlocked/falsified:
```
