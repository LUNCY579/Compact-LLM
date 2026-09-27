# Primary references inspected on 2026-09-27

1. XiaomiMiMo dataset card and schema:
   https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/blob/main/README.md
   Used to distinguish agent environment specifications from completed demonstrations.
   The import command requires the operator to pin an exact revision; `main` is not
   silently treated as an immutable dataset version. No environment was downloaded.
2. Hugging Face TRL GRPO documentation:
   https://huggingface.co/docs/trl/grpo_trainer
   Used for grouped on-policy rewards, zero-variance diagnostics, sampler/trainer
   mismatch concerns, response masking, and explicit loss-normalization distinctions.
   This package does not wrap or claim runtime compatibility with a particular TRL
   release; it implements its own narrow PyTorch objective.
3. Uni-Agent repository:
   https://github.com/verl-project/uni-agent
   Used to identify the separate agent-harness/trajectory collection boundary. This
   commit does not claim that Uni-Agent or all MiMo tool environments are integrated.
4. Existing ChatBubble objective, accessed through the authorized GitHub connection:
   `experiments/rola_ngu_factorial_v1/objective.py`, blob
   `9ba6b17307bb84af9e0980ab9d9362a863e9f048`, on the reconciliation branch.
   The sequence-mean clipped-loss convention is preserved; separate interventions
   and prior native authorizations are not inherited.

All thresholds, stage fractions, and resource limits in the supplied profiles are
initial engineering hypotheses. They are not values validated on the user's GPU.
