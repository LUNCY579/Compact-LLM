# Source reconnaissance and integration boundaries

Implementation parent is `research/post-training-mimo-centralized-v1` at
`5e5fe36ebc6c2725c61c4c4cf78d3f85344cba80`, root tree
`80c2255126c4a5d429ca989187846fa253f7d88d`.
The remote README and source were inspected before creating this branch.

The parent's centralized README reports exact Titan identity and an E0 host-RAM
block (6.9113 GiB available versus 7.4 GiB admission), with no native training in
that centralized lane. This is a historical inspected receipt, not a measurement
of the current host. Earlier native results elsewhere in Rineya remain distinct.

Runtime integrations, without editing upstream files:

- `post_training_mimo_centralized_v1/scripts/qualify_native.py`: import unchanged
  `validate_config` and `install_lora`. Exact 28-layer/56-site q/v rank8, alpha16,
  FP32 factors and BF16 frozen base. Its Git blob was
  `6b41493033cdeeae39cdae653210e6eea56bf7b2` at inspection.
- `opallo_native_collector_v1/supervisor.py`: import unchanged `run_bounded`;
  Windows Job Object, independent memory/RSS/wall watchdog and GO handshake.
  Git blob `dc8f25bfbed2f7f2159ada79a9fec55622f8010d`. Its `contract.py` dependency
  is hashed into every native source identity as well.
- Shared `git-common-dir/rineya-central-gpu0.lock` coordinates with centralized
  workers across worktrees. It is not an OS-wide GPU scheduler for unrelated code.
- `progressive_rl_curriculum_v1/schema.py`: optional explicit task/public-prompt
  bridge; inspection blob `110844a75e1c3a82748e7e83569ee8c80ec3c042`. Existing
  GRPO token log-probability objectives are **not** repurposed as full-vocabulary KL.

New code, comparisons, native states and inputs all live in this package and new
artifact directories. E0/E1 plans, method inventories, promotion state, weights,
production inference and other branches are not modified. The package is not
implicitly enrolled in centralized E1. The optional ordinary-SFT parent stage is
an RSD-specific preparation lane with its own evidence, not an E1 efficacy claim.

## External primary sources

1. Yin et al., *Recursive Self-Improvement via On-Policy Distillation for Reasoning*,
   arXiv:2609.30652v1. https://arxiv.org/abs/2609.30652
   Full text inspected: https://arxiv.org/html/2609.30652v1
   Relevant ideas: full-distribution on-policy self-distillation, detached evolving
   privileged teacher, and gold-free concise rewrites accepted using verification.
   Our matched prompt placement, LoRA, short budgets, complete-trajectory filtering,
   replay and stopping rules are explicit adaptations, not paper hyperparameters.
2. The pinned repository source and contracts above supply the native interface,
   identity, optimizer scope and containment boundaries. No framework documentation
   is treated as proof that this particular Titan host was qualified.

The CPU workflow pins actions/checkout to `11d5960a326750d5838078e36cf38b85af677262`
and actions/setup-python to `a26af69be951a213d495a4c3e4e4022e16d87065`, resolved from
those repositories' v4/v5 tag references during implementation. CI installs only
CPU test dependencies, disables LFS smudging, and never downloads model weights.
Local tests cover this new package; the entire inherited repository suite and the
Windows watchdog were not executed locally. Remote CI status is separate evidence.
