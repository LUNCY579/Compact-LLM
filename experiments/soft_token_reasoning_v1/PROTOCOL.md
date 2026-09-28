# Soft-token reasoning post-training v1

Status: **STR-QUAL-001 mechanical qualification only. Training not run. Sealed evaluation not accessed.**

This branch isolates one scientific variable: whether reasoning post-training can use continuous token states while the deployed model still performs ordinary discrete autoregressive inference. It does not add GFlowRL, replay, KL, clipping, model soups, or another reward/objective change.

## Source grounding

The primary source is [Soft Tokens, Hard Truths, arXiv:2509.19170](https://arxiv.org/abs/2509.19170), especially Sections 3–4 and Appendix A.

The implemented mechanics are:

1. From next-token logits, compute `p_t = softmax(logits / tau)`.
2. Construct the continuous token state `z_t = p_t @ E`, the full-vocabulary probability mixture of input embeddings.
3. During soft training, sample `z_t_tilde = z_t + sigma * N(0, I)`.
4. Use the Gaussian log density of each noisy state, summed across reasoning steps, plus ordinary sampled-answer log-probability in the REINFORCE objective.
5. Stop-gradient the reward and per-prompt RLOO advantage. No straight-through estimator is used; the mixture and Gaussian mean remain differentiable.
6. For the initial S1 deployment arm, use ordinary discrete generation at evaluation time. The inference path never consumes the saved soft states.

The paper's default values are frozen for S1: `tau = 0.5` and `sigma = 0.33 * RMS(E)`. The paper's ablations motivate later, separately registered studies of noise scale, noise placement, temperature, and soft/fuzzy/hard inference; they are not tuned here.

## Current ChatBubble contracts

The model/tokenizer contract is the current research-teacher contract documented in `RESEARCH_PAPER_DATA.md`: `Qwen/Qwen3-0.6B` with the matching tokenizer. The Rineya published-core registry contains downstream Qwen3-derived checkpoints, but this qualification package does not silently substitute a downstream core or a historical adapter.

The dataset binding is `Rineya-Puzzle-Reasoning-51k` and its repository manifest `Rineya-Puzzle-Reasoning-51k/data/manifest.json`:

| split | rows | manifest SHA-256 |
|---|---:|---|
| train | 40,800 | `9e48cb5c207605209aa642438fc6bbb186f76cea22bd3fcef78807495175a703` |
| validation | 4,080 | `f717d9f19a904a0278ba932afca76976a331dd4e184fd8a997af93843cd93b95` |
| test | 4,080 | `ae951ceda2c0a464464183ca699b54cec2615a184ab68b8b4c70d117b0c4ab66` |

The qualification path binds train and validation metadata only. The test split is named in the configs for future frozen evaluation but is not read or used for selection.

## Arms

| arm | training reasoning state | evaluation reasoning state | other factors |
|---|---|---|---|
| D0 | ordinary discrete tokens | ordinary discrete tokens | current canonical discrete control |
| S1 | full-vocabulary soft state plus paper Gaussian exploration | ordinary discrete tokens | same model, data, split, updates/tokens, optimizer family, seeds, decoder, retention and stopping policy |

The initial implementation is deliberately tensor-level and model-agnostic. A future native adapter may wrap the Qwen3 `inputs_embeds` forward, but it must bind the instantiated model class, tokenizer files/template, dtype/runtime, parameter inventory, and host envelope before any launch.

## STR-QUAL-001 gates

The tests cover:

- disabled-soft equivalence with the discrete path;
- hand-computed mixture and one-hot recovery;
- gradient coverage;
- deterministic noise under a fixed seed;
- Gaussian density and RLOO/stop-gradient behavior;
- explicit train-soft/infer-hard switching and ordinary hard generation;
- tiny optimization behavior;
- NaN/Inf rejection;
- D0/S1 factor equality and GFlowRL/sealed-path rejection;
- exact manifest/split binding;
- frozen pass@1/pass@8, correct-solution diversity, paired repair/damage, and retention definitions.

The exact full-vocabulary path is used even in the tiny qualification case. There is no top-k, sparse, sampled-vocabulary, projected, or other approximation. Any later optimization must be separately named, configured, and qualified.

## Evaluation schema and preregistered gate

Every future per-example output must include the example ID, arm, seed, sample index, correctness, normalized answer, reasoning length, mean token-distribution entropy, stop reason, and train/inference mode. Receipts must include checkpoint/model and tokenizer identities, dataset/split hashes, config and code hashes, seed, dtype/runtime, package versions, and resource measurements.

Correct-solution diversity is frozen as the mean, over prompt groups, of `unique(correct answer_norm) / number(correct samples)`; groups with no correct sample contribute zero. Pass@k is the fraction of prompt groups with at least one correct sample among the first `k` registered samples. Repairs and damages are reported as separate paired counts, never netted.

Before any confirmatory evaluation, the initial gate is fixed as:

- pass@1 degradation no greater than 1 percentage point;
- at least one meaningful positive signal: pass@8 improvement at least 2 percentage points **or** correct-solution diversity improvement at least 15%;
- base-model retention comparable to D0 under a preregistered uncertainty criterion.

No metric may be selected after results are observed.

## Resource and access policy

No full training is launched by default. The resource probe runs a bounded exact soft-token forward/backward tensor path only. If CUDA is unavailable, VRAM is recorded as unavailable and the configuration is not reinterpreted as passing a GPU envelope. Resource failures are blockers, not evidence against the method.

The qualification package does not load a checkpoint, construct an optimizer for a native model, mutate a checkpoint, access sealed/ANY_READ data, or provide a `--train`/`--force`/`--resume` path. The tiny optimization test is a synthetic tensor test only.

See `HISTORY_AUDIT.json` for the exact historical refs, anchor SHAs, successor selection, and exclusions.
