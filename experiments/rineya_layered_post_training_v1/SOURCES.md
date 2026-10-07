# Source audit — 2026-10-07

## What is actually verified

1. User-provided Figure 3 screenshot: its caption describes off-policy distillation at
   65K, averaging five checkpoints, DPO, two RL stages, two long-trace SFT/soup stages,
   and final critic RL at 131K. This is the source of the stage ordering. The complete
   technical report and its exact optimizer/data/compute details were not recovered
   during this implementation. Do not describe RLP-001 as an exact reproduction.
2. Paradigma release blog, primary source:
   https://paradigma.inc/blog/limite-1b-violetto/
   Confirms the mathematical specialization and explicitly illustrates instruction-
   following/scope limitations. It motivates Rineya's protected generalism suites,
   not an inference that Violetto's recipe guarantees retention.
3. Released critic details, primary source:
   https://huggingface.co/paradigma-inc/limite-1b-value-model/blob/main/MODEL_DETAILS.md
   Reports an offline-fitted, then online-updated value model, a linear scalar head,
   preceding-action positions, terminal binary correctness, half-MSE and gamma/lambda=1.
   Explicitly says outputs are raw estimates, not calibrated probabilities. RLP uses a
   smaller detached-feature head, not that released full critic backbone or weights.
4. Official serving/evaluation repository:
   https://github.com/paradigma-inc/limite-violetto
   This is primarily serving and evaluation code, not the complete training pipeline.
   Cross-model compute plots are not a matched Rineya training budget; the repository
   notes that RL is excluded from its estimated training compute figure.

## Primary foundations

- Rafailov et al., Direct Preference Optimization, arXiv:2305.18290v3:
  https://arxiv.org/abs/2305.18290v3
  Basis for response-sequence likelihood-ratio preference loss.
- Wortsman et al., Model soups, arXiv:2203.05482v3:
  https://arxiv.org/abs/2203.05482v3
  Basis for weight averaging. Benefits are empirical, not universal guarantees for
  unrelated models, unrelated bases or catastrophic-forgetting prevention.
- Ilharco et al., Editing Models with Task Arithmetic, arXiv:2212.04089v3:
  https://arxiv.org/abs/2212.04089v3
  Basis for common-base task deltas. Convex checkpoint soups and independently scaled
  deltas are distinct operations and have separate APIs in this experiment.
- Official Qwen3 Transformers model documentation:
  https://huggingface.co/docs/transformers/model_doc/qwen3
  Basis for the local native-model API. Actual 0.6B checkpoint/CUDA qualification has
  not been executed in this delivery.

## Corrections to the motivating discussion

A critic is not automatically a calibrated success probability. A soup is not a
parameter-wise proof of retained skills. DPO does not automatically preserve generalism.
A lower-dimensional model can still need substantial rollout/training compute. The
4K profile, critic thresholds, replay weight, coefficient grid and promotion gates here
are preregistered Rineya proposals, not recovered Limite hyperparameters.

No published benchmark score is used as a test result for this branch. No teacher,
Limite, MiMo or Rineya model weights/data were downloaded or committed.
