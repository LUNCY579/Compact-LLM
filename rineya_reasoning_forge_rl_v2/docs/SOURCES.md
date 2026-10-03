# Design references

Consulted September 14, 2026. These are primary research or official project documentation. No passages, benchmark questions, source datasets, or paper-provided training rows were copied into the generated corpus. The package is not an implementation or reproduction of any paper's complete training method.

1. **DeepSeek-AI, DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.** arXiv:2501.12948, version 2 dated January 4, 2026. https://arxiv.org/abs/2501.12948 . Relevance: reasoning-oriented RL on verifiable tasks motivates explicit outcome oracles; it does not establish that this new small synthetic corpus improves Rineya.

2. **Hugging Face TRL, GRPO Trainer documentation.** https://huggingface.co/docs/trl/grpo_trainer . Relevance: online grouped completion sampling, custom reward callable arguments, extra dataset-column forwarding, completion lengths, truncated-sample metrics, and group reward variance. The docs can evolve; pin and validate your actual trainer version. Native TRL integration was not executed in this release.

3. **Yu et al., DAPO: An Open-Source LLM Reinforcement Learning System at Scale.** arXiv:2503.14476. https://arxiv.org/abs/2503.14476 . Relevance: token-level loss and overlong-reward considerations make stopping/truncation behavior an explicit training boundary. This bundle does not claim to reproduce DAPO.

4. **Dai, Liu, and Si, Stable Reinforcement Learning for Efficient Reasoning.** arXiv:2505.18086. https://arxiv.org/abs/2505.18086 . Relevance: the paper reports accuracy-instability risks from length penalties and motivates treating efficiency shaping as a gated experiment, not an unconditional default. Its reported improvements are not transferred to this corpus or model family.

5. **Qwen Team, Qwen3: Think Deeper, Act Faster.** April 29, 2025. https://qwenlm.github.io/blog/qwen3/ . Relevance: official examples distinguish thinking/non-thinking template behavior and parse thinking content separately from final content. Modified Rineya models still require their own tokenizer/template qualification; textual tags alone do not enforce private rendering.
