# RHD-001 exact-host handoff

## Current authorization

This branch authorizes implementation and offline protocol tests only. It does not
authorize a model load, paid teacher call, native training step, or use of a sealed
bank. Previous grants from unrelated experiments do not carry over. The current
native status is BLOCKED_NOT_QUALIFIED, not a claim that native work failed.

## Gate 0 — inspect and reproduce offline

Check out `research/rineya-harness-distillation-v1`. Verify the parent is
`5517361ec09e02308eae0352b2e53a70853c5245`; inspect the additive diff. Run the scoped
unit suite, plan, demo, and manifest verification commands in README. No dependency
installation, model download, GPU, cloud credentials, or Docker is needed.

## Gate 1 — fresh, bounded native qualification, no training

After obtaining a new operator grant, bind an exact local student checkpoint and
hash all model/tokenizer files. Record loader commit, source hash, chat template,
EOS/PAD IDs, model config, dtype, library/CUDA versions, GPU, and local file paths in
a private receipt. No placeholder name or synthetic receipt can satisfy this gate.
The first candidate is the existing ~344M student with fixed rank-8 q/v LoRA and a
frozen base, subject to loader compatibility; do not automatically change capacity.

Use the inherited PRLC host admission policy: two consecutive readings with at
least 7.4 GiB free host RAM and the separate 6.0 GiB worker floor; bind disk/VRAM
thresholds and refuse allocation if they fail. These are project admission defaults,
not evidence of measured fit. Creating a PRLC backend may also allocate adapters,
a reference copy, and optimizer state; qualify the actual path, not just forward VRAM.

Explicitly approve at most 32 **action-generation calls** for the first host session,
with separately frozen per-call token/context and total wall/token budgets. An
action-generation call can contain many autoregressive model forwards; do not label
32 calls as 32 forwards. No teacher service or optimizer is needed at this gate.

Compare collector/SFT prompt IDs on the same public views. Check actual action JSON,
EOS, prompt masking, exclusion of tool/reviewer text from targets, no truncation,
termination, and stable policy identity. Measure peak CPU/GPU memory. The receipt
must say whether the exact backend supports this envelope; tests on byte tokens or
the scripted expert cannot substitute. Stop on drift, invalid actions, or resource
breach. No native runner CLI is supplied in v1; use a separately reviewed host driver
around the PRLCStudent protocol, without silently calling a training command.

## Gate 2 — teacher/environment qualification and data collection

Bind the teacher version, private procedural reference, safe deployment action
space, verifier version, source revisions/licenses, split groups, and attempt seeds.
The teacher receives the public interaction view and reference only, never graders,
held-out examples, record-store dumps, or hidden answers. A live service adapter
needs explicit approval, enforced currency/token limits, per-call deadlines and a
durable request ledger. None of these live adapters is enabled by this branch.

Begin with a small training-only pilot. Audit corrections, observation replay, rare
invalid responses, leakage, and collection cost before collecting up to 256 parent
tasks twice. The demo's 256 tasks are generated fixtures, not a research dataset.
Retain failed-attempt counts; freeze the successful-data selection policy before
looking at evaluation. Export arms separately and match retained training budgets.

## Gate 3 — separately authorize one bounded SFT smoke step

Only after Gates 1–2 pass, authorize a native backward/optimizer qualification with
an immutable pre-step checkpoint, fixed adapter targets, finite gradient checks,
and explicit token/step/memory limits. Reuse PRLC's guarded optimization machinery;
this package only exports its data contracts. A native training driver and receipt
must bind the RHD action data and include independent retention-training rehearsal.
No layer unlocking, WSD change, RL, DPO, merging, or additional dataset in this test.

After a smoke step, verify save/reload equality, unchanged frozen base parameters,
and reviewer-free baseline versus trained evaluation. Action imitation is secondary;
primary outcomes are task success and separate retention banks under the same h.
A full experiment needs matched SFT controls, frozen budgets, multiple seeds, and an
untouched confirmation panel. The code currently provides descriptive paired
intervals; adaptive calibration intervals are not confirmatory significance tests.

## Follow-ups, not activated

Only after demonstrating real behavioral transfer: harness-SFT versus ordinary
teacher-SFT; SFT then RL/MaxRL; per-skill adapters and functional interference; and
harness-derived delta composition. Geometric orthogonality alone does not establish
functional independence, and reviewer replacements are not on-policy RL samples.
