# Repository reconnaissance and non-interference

Repository inspected through the authenticated GitHub connection:
`LUNCY579/ChatBubble`. Base `post-training-Mimo` resolves to
`03db218ceebc61a2bc04f2482cf7e094045e01c1`, root tree
`6a1e3b9b16789f7ce211e443654912859dd37ed0`.
No root or `experiments/AGENTS.md` was present at that ref.

## Existing relevant work read

- `research/rineya-recursive-self-distillation-v1`:
  `experiments/rineya_recursive_self_distillation_v1/README.md`, `backend.py`,
  `objectives.py`, and the native admission/E0 portion of `native.py`.
  Read blob SHAs: README `c659becd570082ea114ccf664816b8b0fa1c1248`,
  backend `15e3a8fe5073dd392ebf4f9c339fab2865470f78`,
  objectives `db60e2c9821cd28c300bab4cac046e68ccdbb96a`,
  native `0038b9445ac6e76654318f21e2887bc0b9291cb2`.
- `research/rineya-generalization-dynamics-v1`:
  `experiments/rineya_generalization_dynamics_v1/integration.py`, blob
  `89f80e6f25ab96cc59487153ea0754cd7dc1839e`.
  Its receipt hook accepts explicit HF/PEFT exports and does not itself qualify
  native model compatibility or change the trainer.
- The previous review also located independent retention/rehearsal evidence and
  the RHD-001 student-first interaction trajectory program. These are context,
  not claimed replicas or imported training datasets in this implementation.

## Integration decisions

RSD's existing objective uses a shared frozen output head, forward KL and
privileged/evolving guidance in its research design. Reusing that experiment's
policy switches directly would entangle teacher conditioning and teacher refresh
with the rollout-source question. RDD therefore implements a separate controlled
trainer and generalizes the same first-order chunked-projection principle to
FKL/RKL with different frozen heads and hidden widths. Numerical reference and
finite-difference tests cover the new implementation. No RSD source is edited.

Native receipt requirements and conservative floors are retained explicitly,
not treated as inherited model-load authorization. This branch uses only reviewed
standard HF exports; original custom artifact conversion and exact-native export
parity remain upstream prerequisites. This is intentionally not advertised as
an already qualified replacement for RSD's custom exact-Titan loader.

GD handoff is explicit PEFT format serialization plus provenance. That sibling
branch is not automatically merged or imported, and a serializer is not evidence
of real PEFT/native parity. RDD includes its own narrowly scoped paired evaluation
and all-seed selection so the controlled experiment is independently executable.

## Scope

Only `experiments/rineya_distillation_dynamics_v1/` is added. Existing research
branches, production files, model assets, native loaders, protected banks,
repository-wide workflows and automation schedules remain untouched. No models,
private datasets, runtime receipts or GPU jobs are committed or launched.
This experiment complements skill composition, RHD/RSD, GD and centralized
reasoning; it does not supersede their results, gates, or retirement decisions.
