# Sources and attribution

Checked 2026-10-07. This is an original Rineya implementation, not copied author
code or a claim to reproduce the paper's model or scores.

1. Li et al., **Scaling Trajectories for Complex Tasks through Recursive
   Self-Rewrite**, arXiv:2610.02826v1 (2026-10-02).
   https://arxiv.org/abs/2610.02826
   https://arxiv.org/html/2610.02826v1
   Sections 2.1–2.2 motivate multi-harness discovery, procedural runbooks,
   planner/critic revision, fresh execution and public-trajectory training.
   Sections 3.2–3.3 distinguish discovery harnesses and coverage from rollout
   success. We adapt the mechanism to a fixed small-student action space;
   public-only H2 feedback and finite-vocabulary runbooks are deliberate departures.
   The original paper's 27B results are not treated as evidence at 0.6B.

2. Rineya Harness Distillation (RHD-001), local repository implementation.
   Original commit: `a4f15e00cffe68df1c86c9c7cf21e31c724d7fc7`.
   Inspected integrated parent: `19551ddebc87c93161c303865ce6d0f130de0093`.
   https://github.com/LUNCY579/ChatBubble/tree/19551ddebc87c93161c303865ce6d0f130de0093/experiments/rineya_harness_distillation_v1
   Reused `contracts.py` and `environment.py`, including RHD review request binding.
   Exact Git blob hashes are enforced by `core.source_hash()`.

3. Hugging Face Transformers official model API documentation.
   https://huggingface.co/docs/transformers/main_classes/model
   The adapter expects an already-created causal model with `generate`; it does
   not call `from_pretrained`, download checkpoints or enable remote model code.
   Native tokenizer/model compatibility still needs host qualification.

4. PyTorch official cross-entropy documentation.
   https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
   Reference trainer uses shifted causal targets, `ignore_index=-100` and summed
   target loss normalized by a fixed supervised-token quota.

Research-specific margins, task counts, seed choices and budgets in this package
are proposed Rineya protocol decisions, not findings or recommendations attributed
to the paper. No source trajectories, model weights or copyrighted paper text are
redistributed with this experiment.
