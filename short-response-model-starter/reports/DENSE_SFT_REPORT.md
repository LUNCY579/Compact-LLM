# DENSE SFT REPORT

## Scope

Train and evaluate the all-layer LoRA or QLoRA SFT behavior reference.
Inference remains dense. This stage tests whether the desired short-response
behavior is learnable before selective-layer or sparse-runtime experiments.

## Inputs

- Repository revision:
- Base model:
- Model revision / SHA:
- Tokenizer revision / SHA:
- Training manifest:
- Development manifest:
- Sealed evaluation manifest:
- LoRA configuration:
- Seed:
- Training steps / epochs:
- Hardware:
- Software versions:

## Work Completed

- [ ] Validate train/dev separation and leakage checks.
- [ ] Attach LoRA to attention and MLP projections across all layers.
- [ ] Train the dense-inference SFT reference.
- [ ] Save adapter, tokenizer, config, logs, and manifest.
- [ ] Evaluate on development data and ShortChatBench.
- [ ] Compare directly with the untouched base model.
- [ ] Perform human review of correctness, naturalness, context, and safety.

## Results

| Metric | Base | All-layer SFT | Difference |
|---|---:|---:|---:|
| Median generated tokens | TBD | TBD | TBD |
| P90 generated tokens | TBD | TBD | TBD |
| Complete-utterance rate | TBD | TBD | TBD |
| Mean heuristic score | TBD | TBD | TBD |
| Human preference rate | TBD | TBD | TBD |
| Factual correctness | TBD | TBD | TBD |
| Context consistency | TBD | TBD | TBD |
| Safety critical regressions | TBD | TBD | TBD |
| Peak training VRAM | N/A | TBD | N/A |

## Failure Analysis

Record generic replies, excessive brevity, echoing, factual simplification,
template collapse, context misuse, and safety regressions.

## Gate Status

`NOT RUN`

Do not begin the layer-selection study until the dense SFT model produces
concise, complete, correct, and varied replies on the development suite.
