# LAYER IMPORTANCE REPORT

## Scope

Compare static selective-layer LoRA configurations against the untouched base
model and the all-layer SFT reference.

## Inputs

- Repository revision:
- Base model and SHA:
- Dataset manifests:
- Benchmark version:
- LoRA rank / alpha / dropout:
- Target modules:
- Seeds:
- Steps per variant:
- Hardware:
- Software versions:

## Required Variants

| Variant | Trainable layer placement | Purpose |
|---|---|---|
| BASE | No adapter | Original behavior |
| ALL | Adapters in all layers | Quality reference |
| SINGLE-MID | One middle layer | Paper-inspired prior |
| SINGLE-BEST | Best layer from scan | Main one-layer candidate |
| PAIR | Best adjacent or complementary pair | Test one-layer limitation |
| TOP4 | Four selected layers | Robust efficiency-quality candidate |
| RANDOM4 | Four random layers | Parameter-count control |
| BOTTOM4 | Four low-ranked layers | Ranking-validity control |

## Coarse Scan Results

| Variant | Layers | Raw score | Contribution ratio | Peak train VRAM | Train time | Notes |
|---|---|---:|---:|---:|---:|---|
| TBD | TBD | TBD | TBD | TBD | TBD | |

## Contribution Metric

For a score where larger is better:

`C_short(K) = (S_K - S_base) / (S_all - S_base)`

Always report raw scores. Do not use the ratio when the denominator is too
small to be stable.

## Multi-Seed Finalists

| Variant | Seed | Raw score | Category floors passed | Peak VRAM | Notes |
|---|---:|---:|---|---:|---|
| TBD | TBD | TBD | TBD | TBD | |

## Interpretation Boundary

Training-layer importance is not the same as inference-layer importance. A
layer that is effective for adaptation does not imply that other layers can be
skipped during inference.

## Gate Status

`NOT RUN`

Advance only when a one-, two-, or four-layer configuration matches the
all-layer reference within the predefined tolerance and materially reduces
training memory or time.
