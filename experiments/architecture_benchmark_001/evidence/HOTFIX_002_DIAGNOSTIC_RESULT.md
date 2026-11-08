# HOTFIX-002 diagnostic result — high-precision shared-memory read does not work

Implementation `fc44701e`. Pinned runtime: python 3.11.15 / Windows-10-10.0.26200-SP0 /
torch 2.10.0+cpu / transformers 4.51.3. Official Supa 1.7.0-devfreeze weights,
`audit_digest 97cd76af…`. CPU, float32, eager, 8 open pilot rows.

This records a **negative result**. It authorizes nothing, changes no tolerance, and does
not qualify any arm. The registered option remains default `inherit`.

## 1. The intervention fails

`diagnose-numerics --arm {C28,C40} --precision float32 --memory-precision {inherit,float64}`
on the C28/C40 checkpoints from `outputs/ladder-8d54f540-v1`:

| arm | memory-precision | max abs err | total elementwise violations | failing paths |
|---|---|---:|---:|---:|
| C28 | inherit | 8.23e-04 | 401,863 | 6 |
| C28 | float64 | 1.00e-03 | 399,698 | 6 |
| C40 | inherit | 5.89e-03 | 726,228 | 5 |
| C40 | float64 | 4.72e-03 | 803,104 | 5 |

C28 improves 0.5% while its max error worsens; C40 worsens 10.6%. The same paths fail.
`causal_suffix_isolation_exact` and `decode_choices_exact` remain True throughout.

The hypothesis under test — that the amplification is *created* in the decoder's read
contraction over the shared bank — is **falsified**.

## 2. Divergence starts in the encoder, and the bank is already out of tolerance

Per-layer hidden-state comparison, full forward vs chunked, on the C28 checkpoint:

| schedule | enc00 | enc13 (final encoder) | shared bank dK | shared bank dV |
|---|---:|---:|---:|---:|
| chunk 7 | 3.815e-06 | 1.678e-04 | 2.012e-04 | 4.597e-04 |
| chunk 1 | 3.576e-06 | 3.906e-03 | 1.073e-04 | 2.565e-04 |

Divergence begins at **encoder layer 0** and amplifies roughly 1000x through encoder depth.
The shared K/V bank inherits it at **dV = 4.6e-04 — already 4.6x outside the 1e-4
tolerance before any decoder read occurs.** Precision applied at the read cannot recover
information that is already wrong in the bank, which is precisely why §1 failed.

## 3. The ordinary control shows CED creates no extra drift

Same probe on the unmodified official B28, whose first 14 layers *are* C28's encoder:

| schedule | layer00 | layer13 | layer27 (final) | growth 00→27 |
|---|---:|---:|---:|---:|
| chunk 7 | 3.815e-06 | 1.678e-04 | 8.545e-04 | 224x |
| chunk 1 | 3.576e-06 | 1.001e-05 → 3.906e-03 | 8.789e-03 | 2458x |

B28's drift matches C28's encoder drift digit-for-digit at layer 00 and layer 13, and by
its final layer B28 drifts **more** (8.8e-03) than C28's encoder does (3.9e-03). Yet
B28's logits show 4 violating elements against C28's 401,863.

Both architectures carry ~1e-3 mid-network chunk-order drift in FP32. That is ordinary
FP32 transformer behaviour, not a CED defect.

## 4. Mechanism

What differs is how that drift reaches the logits.

- **Plain arm:** drift propagates to the final RMSNorm, which suppresses its correlated
  component. 8.8e-03 of hidden drift becomes 1.7e-04 of logit error — inside tolerance.
- **CED arm:** the *unnormalized mid-network* encoder output is frozen into a shared K/V
  bank and re-injected into all 28 decoder layers coherently, before the final norm. The
  bank is already at 4.6e-04 and never receives that suppression.

So the strict 1e-4 elementwise logit gate is passed by plain models substantially *because*
of final-norm suppression, and CED forfeits that suppression by construction. This is an
architectural property, not a rounding artifact and not a bug: §3 shows the arithmetic is
the same, and FP64 (recorded in `ladder-eab80e9f-v1`) shows both arms clean at ~1e-12.

## 5. What follows

No cheap numerical remedy exists. Raising the bank's *projection* precision cannot help
either, because its input `h` is already perturbed. The remaining routes are larger
decisions, none taken here:

1. Construct the shared bank from a full-context forward rather than incrementally —
   changes incremental-decoding semantics and cost; needs its own design and registration.
2. Run the encoder at higher precision — that is most of the model; a throughput and memory
   decision, not a tolerance tweak.
3. Prospectively amend the equivalence policy for architectures that expose a mid-network
   tensor to cache-order comparison. Must be written without citing this failure as its
   justification, and reviewed separately.

Until one of those is registered and qualified, C28 and C40 remain unqualified at FP32 and
their optimizer lanes stay closed. `gates.py` enforces this unaided.

## Reproduction

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
SRC=C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze
E=experiments/architecture_benchmark_001
$PY -m experiments.architecture_benchmark_001 diagnose-numerics --source $SRC --arm C28 \
  --checkpoint $E/outputs/ladder-8d54f540-v1/G1_CED_C28/attempt-0001 \
  --precision float32 --memory-precision float64 --rows 8 --continuation-steps 8 \
  --out $E/outputs/diagnostics-memprec/C28-float64 --device cpu
```

The per-layer probes in §2 and §3 were ad-hoc forward hooks over `model.encoder` and
`model.memory`, not a registered CLI stage. Promoting them to one would change the
implementation digest and require another requalification; that is deliberately deferred.
