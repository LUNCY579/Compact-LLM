# Repository integration and non-duplication

Remote owner/repository: `LUNCY579/ChatBubble`.
Pinned integration parent: `03db218ceebc61a2bc04f2482cf7e094045e01c1`.
Parent tree: `6a1e3b9b16789f7ce211e443654912859dd37ed0`.
All additions are confined to `experiments/rineya_layered_post_training_v1/`.
Neither main nor post-training-Mimo is modified or merged by this delivery.

## Reused code

`experiments/progressive_rl_curriculum_v1/objectives.py` is imported directly for
`advantages` and `policy_loss`. Its pinned Git blob is
`b43663f69c21318ff4281e6627cfd1da1c21f550`. A byte-identical local copy of this existing
file was used for scoped testing; that copy is NOT a new or modified branch file.
The upstream schema and TorchBackend were inspected to retain response masks,
completion validity, version checks, local loading and distribution-parity conventions.

RLP introduces an additive stage graph rather than changing the existing progressive
curriculum engine. The older engine selects tasks/difficulty; RLP selects post-training
objectives and weight-space consolidation boundaries. Its runtime is distinct because
it adds DPO, effective-dense checkpoint interpolation and a detached-feature critic.
The underlying GRPO mathematical primitive remains unchanged.

## Links to the existing research agenda

Harness/MiMo distillation supplies externally curated local TRAIN traces and preferences;
this branch does not silently assume those releases are already converted or licensed.
Independent-coefficient and model-soup experiments motivate common-base delta APIs and
explicit coefficient grids. CSS and sequential complementary-skill research motivate
protected-suite measurement, but no cross-architecture projection or CSS algorithm is
claimed to be executed by this branch. Retention-dose studies map naturally to the
actor-step/replay-weight ablations. Reward-vector/reverse-world ideas remain a future
separate experiment after a scalar prefix critic is measured on native generations.

No existing research branch is deleted, rebased or force-pushed. No application/game
code or shared repository configuration is changed. Native quality, CUDA behavior,
Windows execution, full repository regression and remote CI remain separate validation
levels from the scoped CPU tests in this delivery.
