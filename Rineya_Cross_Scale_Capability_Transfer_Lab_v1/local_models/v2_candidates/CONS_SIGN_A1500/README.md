---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- rineya
- qwen3
- long-context
- abstention
---

# Rineya Supa-A1.1

Single-pass long-context core with a **baked abstention offset**, stored in
float16. One directory, one forward pass, no adapter, no second model.

Supersedes **Supa-A1**, which is the same recipe stored in bfloat16.

## Results (public runner, frozen protocol)

| | Showcase64 | Stability192 | all256 |
|---|---:|---:|---:|
| no gate | 41 | 121 | 162 |
| **with the baked gate** | **45** | **125** | **170** |

Exact McNemar vs the ungated core on Stability: **p = 0.2188**.
**No statistical significance is claimed.** See `evidence/RE_BASELINE.md`.

## Why float16

Folding a scale-0.8125 LoRA into bfloat16 discards 17.25% of the delta as
rounding noise. Float16 discards 2.19% at the **same file size** and with no
overflow risk. Against Supa-A1 this is worth +2 on Showcase, equal on Stability,
and it reproduces the offline gate on 256/256 rows where A1 disagreed on 5.

**Load in float16.** A bfloat16 load re-rounds the weights and throws away the
precision this core exists to preserve.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "Rineya-Supa-A1.1-1.1.0-devfreeze"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path, dtype="float16", trust_remote_code=True
)
assert type(model).__name__ == "Qwen3AbstainForCausalLM"   # see the warning below
```

Greedy decoding. Set `config.abstain_logit_bias = 0.0` for the ungated core.

**Loader warning.** Without `trust_remote_code=True` this directory loads as
plain `Qwen3ForCausalLM` with the gate silently disabled and no warning — 121
instead of 125 on Stability. Assert the class at startup.

The abstention is **not in the weights**. Qwen3's output head is
`nn.Linear(..., bias=False)` with tied embeddings, so no checkpoint of this
architecture can store a constant. The portable equivalent, for stacks that will
not execute remote code, is a `logit_bias` of `+5.15625` on token `51378`.

## Honest limits

- No sealed TEST split exists. The corpus is 64 Showcase + 192 Stability.
- Stability192 has been read repeatedly and is no longer clean.
- The threshold is model-specific: on Rineya-Baya-1.7.0 the same gate lowers
  Showcase from 40 to 35.
- Power at n=192 is 0.31; roughly 576 rows would be needed for 80% power.

Core id `rineya.core.supa-a1.1.v1.1.0-devfreeze.28L.qwen3-nimbus-lcr-step200-scale-0p8125-fp16-abstain-b5p15625`.
