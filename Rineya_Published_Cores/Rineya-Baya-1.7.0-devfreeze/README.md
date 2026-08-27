---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3-0.6B
---
> Formerly published internally as **Lumina**. Weight and parity evidence are unchanged.


# Rineya Baya 1.7.0-devfreeze

Status: **dense validation recorded**.

Baya is a BF16 dense reconstruction of Nimbus 1.6 plus the disjoint Halo long-context Stage-A checkpoint-200 LoRA at effective scale `1.25` (rank 16, encoded alpha `40`).

The LoRA was trained on Halo 1.2 and transferred to Nimbus; it was not trained on Nimbus. Supa and Baya are separate dense cores. The answer-blind UNKNOWN gate is a two-model inference recipe, not a property of either core alone.

## Frozen evaluation protocol

BF16, greedy decoding, seed 17, thinking disabled, 24 generated tokens, 4096-token context cap, and no truncation.

## Dense Demo results

- Showcase (selection): 40/64
- Stability (frozen validation): 112/192
- Complete public Demo: 152/256

These are public Demo results. Showcase was used during model selection and is not an untouched test result.
