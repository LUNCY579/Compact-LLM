# Native v2.3 GPU qualification: independent saved-logit audit

Result: **PASS for all 40 saved-logit comparisons and their three frozen numeric
gates**, covering 80 hash-verified FP32 vectors across five arms and eight
calibration fixtures. This is a numerical evidence audit, not an efficacy result
or an independent model rerun.

Audited qualification:
`native_armr/evidence/00_runtime_qualification/20260905T235200_982255Z_36998242`.
Its terminal receipt is COMPLETE and its qualification flag is true. The frozen
registration contract is
`86e768e08bf3576dd8ce80ec939b7e876fb5fb7449d41a3aa51c497ce6607b44`.
The audit verifies registration/config/report hash bindings and matches the
report's thresholds to the registered configuration. Git commitment is outside
this audit's scope.

| Recomputed quantity | Observed bound | Frozen gate |
|---|---:|---:|
| Maximum absolute logit error | 0.000041961669921875 | <= 0.002 |
| Maximum relative logit L2 | 0.000002490055806825606 | <= 0.0002 |
| Minimum logit cosine | 0.9999999999969049 | >= 0.999999 |

All 40 recorded metric triples were reproduced. The largest discrepancy between
recorded and independently recomputed metrics was exactly zero for maximum
absolute error, 4.573977915173222e-20 for relative L2, and
4.474198789239381e-14 for cosine. The prior audit's fixed FP64 reduction-rounding
tolerances were reused unchanged. They apply only when matching recorded metrics;
the scientific gates receive no tolerance allowance.

The parser reads each PyTorch ZIP's FP32 storage using `zipfile` and a restricted
pickle metadata reader that allows only three inert constructors. No torch code
is imported. NumPy converts one CPU/GPU pair at a time to FP64 and independently
computes maximum absolute error, error L2/reference L2, and normalized dot product.
Python 3.11.15 and NumPy 2.4.6 were used. Environment thread limits and the loaded
OpenBLAS pool were verified at one CPU thread. No model, GPU, checkpoint, or
generation execution occurred. All 129 byte-hashed inputs were unchanged when
the audit completed.

The earlier attempt
`native_armr/evidence/00_runtime_qualification/20260905T233921_233855Z_41832612`
remains **FAILED**, with `qualification_pass=false` and an `AcceleratorError`
receipt. Its registration/config match the successful retry. All 40 available
CPU logit files are exactly byte-identical to the retry's corresponding files.
This is recovery evidence only; it does not qualify the failed attempt. The saved
driver-event receipt `00_audit/gpu_driver_events_20260905T2347.json` was hash-bound
as context, without a new driver-event collection or causal diagnosis.

Hidden-state agreement, answer NLL, full greedy/repeat identity, gradients, and
checkpoint integrity were **not independently recomputed** here. Those remain
claims of the execution receipts and other programme verification. No
architecture or confirmatory efficacy result is established by this audit.

Artifacts in this directory:

- `recompute_native_logits.py`: standalone audit implementation, SHA256
  `114955e161e8dc68c9c0b7797d9182603f6473517787ccbff5521f0d2a917f1f`.
- `native_logit_recomputation.json`: complete input hash inventory, 40 comparison
  records, 40 prior CPU file comparisons, thread evidence, gates and limitations;
  SHA256 `0b14669f74b1c91f788c8f3b306cc70b39a676c8dab9be816e7e93700883a4be`.

Executed command from the repository root:

```powershell
& 'C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe' -B experiments/supa_adaptive_compute_v2/00_audit/native_v2_3_gpu_independent_review/recompute_native_logits.py
```

The process exited 0. Output creation is exclusive; the completed receipt cannot
be overwritten by rerunning the command. No existing source, configuration,
qualification evidence, or historical audit file was changed.
