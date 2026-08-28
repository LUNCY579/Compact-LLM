# Re-baseline — corrected reference numbers for the abstention work

Supersedes the headline figures in `RESULTS.md` and `CHECKPOINT_RESULTS.md`.
Evidence: `NUMERICAL_ROBUSTNESS.md`. Every number below comes from the
unmodified public runner (`Rineya_Long_Context_Demo_v1/scripts/run_local_hf.py`)
under the frozen protocol, scored by the published
`evaluate_predictions.py`.

## 1. Corrected reference table

| artifact | form | showcase64 | **stability192** | file |
|---|---|---:|---:|---|
| `nimbus_low` | base + adapter, BF16 | 42 | 121 | core + 40 MB adapter |
| Supa-A1 (published) | dense BF16 + baked S4 offset | 43 | 125 | 1.19 GB |
| FP16 dense, no gate | dense FP16 | 41 | 121 | 1.19 GB |
| **FP16 dense + baked S4** | dense FP16 + baked offset | **45** | **125** | 1.19 GB |
| FP16 dense + S1 processor | dense FP16 + logits processor | 44 | **128** | 1.19 GB |

**The abstention gate is worth a stable +4 to +7 on stability192.** It is not
statistically significant on any realisation that survives a precision change.

## 2. What changed from the earlier reporting

| claim as previously stated | corrected |
|---|---|
| S1 reaches 130/192 at p = 0.0490 | 130 only on the BF16 base+adapter path; FP32 and both dense merges give 128 at p = 0.12–0.14 |
| "the only configuration that clears p < 0.05" | true as measured, but not robust; do not cite as established |
| S4 is "baked into the model", "a property of the model" | the offset is applied in `forward`, not stored in the weights; Qwen3's head is `bias=False` with tied embeddings |
| the gate transfers | it does not — on the Baya-1.7.0 core the same threshold makes things worse (40 → 35 on showcase) |

## 3. FP16 is the better dense core

Same 1.19 GB, no overflow risk, and it stores the merged LoRA delta 8× more
faithfully than BF16 (2.19 % rounding error against 17.25 %).

| dense core | showcase64 | stability192 | baked-vs-offline gate disagreements |
|---|---:|---:|---:|
| BF16 store (Supa-A1) | 43 | 125 | 3/64 showcase, 2/192 stability |
| **FP16 store** | **45** | **125** | **0/64, 0/192** |

The FP16 core reproduces the offline gate exactly on all 256 rows and scores two
points higher on showcase for the same stability result. If a dense abstention
core is published, it should be the FP16 one.

## 4. Recommended artifact

`checkpoint/Rineya-Nimbus-Abstain-FP16/` — one directory, 1.19 GB, hardlinked to
the no-gate FP16 core so the pair costs one copy of the weights.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
path = "artifacts/rineya_abstention_gate_v1/checkpoint/Rineya-Nimbus-Abstain-FP16"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype="float16", trust_remote_code=True)
```

Set `config.abstain_logit_bias = 0.0` for the ungated core. The portable
equivalent, for stacks that will not execute remote code, is `logit_bias` of
`+5.15625` on token `51378` — identical behaviour, no custom code.

**Loader warning.** Without `trust_remote_code=True` this directory silently
loads as plain `Qwen3ForCausalLM` with no gate and no warning — 121 instead of
125. Any serving path should assert
`type(model).__name__ == "Qwen3AbstainForCausalLM"` at startup.

## 5. Still open

- **No sealed split exists.** The corpus is 64 showcase + 192 stability and
  nothing else. Stability has now absorbed the confirmatory run, a post-hoc
  gate, a pack-level sensitivity analysis, the threshold frontier and this
  precision audit. Nothing further can be validated without new data.
- The best achievable pure-bias gate on stability is 133/192 at `b = 8.28`;
  showcase gives no way to select it. Selection, not representation, is the
  binding constraint.
- Four thresholds tie at the showcase optimum and land between 125 and 129 on
  stability. The tie-break rule moves the result as much as the signal does.
