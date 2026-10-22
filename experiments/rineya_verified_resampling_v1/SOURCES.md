# Sources and evidence provenance

Recorded 2026-09-29. This is a source register, not a claim of independent experimental replication. The preceding conversation's recommendations are preserved as proposed RVS research; implementation and scientific outcomes remain separate.

## Primary external sources

### P1 — Large Language Monkeys

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Re, and Azalia Mirhoseini. *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling*.

- Version inspected: arXiv:2407.21787v3, dated 2024-12-30.
- Version-pinned text: https://arxiv.org/html/2407.21787v3
- Abstract record: https://arxiv.org/abs/2407.21787
- Relevant sections: 2 (sampling and coverage estimation), 2.2 (model-family boundary), 4 (selection and verifier limitations), Appendix B (SWE-bench procedure).

Use this source for the historical paper findings, not for Titan's expected effect or current frontier rankings. RVS task counts, stage gates, deployment boundaries, and resource proposals are this project's design, not the paper's prescribed configuration.

### P2 — Authors' implementation

https://github.com/ScalingIntelligence/large_language_monkeys

Public reference implementation inspected for provenance and the separation of sampling from evaluation. No source files or dependencies from this repository are copied or executed by this delivery. A future reuse must pin a concrete upstream commit, inspect its license, and qualify its runtime assumptions. No upstream code SHA or license compatibility is asserted here.

## Immutable Rineya repository references

All R1-R4 links below are pinned to `LUNCY579/ChatBubble` commit:

`ba683ac0fcadbf04e4e179200d035b6a980e388f`

This was the live parent-branch head when the new RVS branch was created. The source branch is `research/post-training-mimo-centralized-v1`. These snapshots are not refreshed implicitly when another branch advances.

### R1 — Centralized experiment report

Path: `experiments/post_training_mimo_centralized_v1/CENTRALIZED_EXPERIMENT_REPORT.md`

https://github.com/LUNCY579/ChatBubble/blob/ba683ac0fcadbf04e4e179200d035b6a980e388f/experiments/post_training_mimo_centralized_v1/CENTRALIZED_EXPERIMENT_REPORT.md

Supports recorded Titan identity/geometry, prepared acquisition and preservation inputs, long-context paired-world construction, unqualified language grading, and host-RAM/native boundaries. Its statement that no pretrained model was loaded or trained is scoped to that centralized campaign, not every historical Rineya experiment.

Recorded source identities, carried without loading model binaries:

| Asset | SHA256 from R1 |
| --- | --- |
| Weights | `537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7` |
| Config | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| Tokenizer | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| Template | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |

### R2 — Centralized experiment registry

Path: `experiments/post_training_mimo_centralized_v1/EXPERIMENT_REGISTRY.json`

https://github.com/LUNCY579/ChatBubble/blob/ba683ac0fcadbf04e4e179200d035b6a980e388f/experiments/post_training_mimo_centralized_v1/EXPERIMENT_REGISTRY.json

Supports E0 `BLOCKED_HOST_RAM`, E1's unmet gates, E2D WSD, E3 matched RSFT/GRPO/MaxRL, E4 fixed sampling versus bounded NGU, E5 competent modular donors/composition, and the requirement for separate full registration of inference/architecture side tracks. RVS does not alter this registry or inherit its training grants.

### R3 — RHD-001 documented contracts

Path: `experiments/rineya_harness_distillation_v1/README.md`

https://github.com/LUNCY579/ChatBubble/blob/ba683ac0fcadbf04e4e179200d035b6a980e388f/experiments/rineya_harness_distillation_v1/README.md

Supports the documented student-first PASS/REPLACE, independent-attempt, restricted-tool, replay-audit, TRAIN-only compilation, and reviewer-free evaluation interfaces. It explicitly separates mechanical implementation from native qualification and describes the missing standalone native trainer/live teacher adapter. Future integration still requires reading and testing the actual implementation.

### R4 — PRLC documented contracts

Path: `experiments/progressive_rl_curriculum_v1/README.md`

https://github.com/LUNCY579/ChatBubble/blob/ba683ac0fcadbf04e4e179200d035b6a980e388f/experiments/progressive_rl_curriculum_v1/README.md

Supports documented prompt/task/episode integration boundaries, distinct training/calibration/preservation roles, explicit native permissions, and local-loader limitations. Inherited loss/update code is not an RVS implementation or permission to run optimization.

## Interpretation register

- Paper-reported result: historical external observation, not reproduced here.
- Repository-recorded fact: attributable to the pinned source; model binaries and historical runs were not re-executed here.
- Analytic illustration: probability and KV-memory calculations under explicitly stated assumptions; not a benchmark or a fit certificate.
- Proposed design: RVS pilot counts, future interfaces, tests, and decision rules; no implementation or run implied.
- Unknown: exact-host RVS qualification, launch budgets, selected parents, finalized decoding/selection policy, and all native RVS outcomes.

The ordinary BF16 KV estimate is derived from R1's recorded geometry, not measured memory. The parent report's 40960 configured context is not a qualified operating envelope. RVS initially considers separately qualified short tasks; the 2048/4096 extension is conditional.

No private model binaries, private outputs, protected payloads, upstream implementation files, or credentials are included in this planning package.
