# Progressive RL curriculum

PRLC-001 lives in `experiments/progressive_rl_curriculum_v1/`.
Start with its README for offline planning, the synthetic end-to-end demo,
native qualification, training/resume, teacher-cache bootstrap, and run reports.

It adds capability-gated dataset unlocking, explicit mixtures, independent
retention rehearsal/evaluation, bounded training, and atomic recovery without
changing existing experiment launchers or production model loaders.

Status: CPU-tested controller and PyTorch primitives. Native Rineya and live MiMo
agent environments remain unqualified. This document is not a training grant.
