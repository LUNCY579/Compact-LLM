# Future Work

## Immediate evidence gaps

1. Evaluate the frozen methods on a fresh, preregistered corpus that was not
   used for training, exploration, method selection, or prior model evaluation.
2. Repeat the study with public backbones at several parameter scales while
   holding the committee construction and evaluation protocol as constant as
   possible.
3. Train matched specialist and same-task multi-seed committees to separate
   specialization, random seed, competence, calibration, and training-mixture
   effects.
4. Predeclare class-conditional utilities for incorrect answers, false
   abstentions, and failures to abstain, then compare policies under several
   realistic class mixtures.
5. Compare hard voting with learned routing, learned or validation-weighted
   voting, and weight-space merging using the same source specialists and
   compute-aware reporting.
6. Extend evaluation beyond templated short responses to natural prompts and
   longer generation, where prefix divergence and error accumulation may differ.
7. Improve expert batching or cache sharing under exact decision-parity tests,
   then remeasure quality, latency, memory, and energy on desktop and server
   hardware.

## Controlled causal experiments

- Train matched same-task multi-seed LoRAs to separate generic diversity from
  specialization.
- Train matched specialist and seed-only committees with equal data, steps,
  rank, and hyperparameters.
- Duplicate experts and replace specialists with competence-matched weak or
  irrelevant controls.
- Vary LoRA dose per expert and record vote identity, calibration, and coalition
  changes.
- Compare ordinary and alternative tie/EOS policies only where a causal
  question warrants the ablation.

## Generalization

- repeat on different frozen backbones and model sizes;
- compare matched LoRA, DoRA, PiSSA, and AdaLoRA committees;
- evaluate procedural recovery, grounding/RAG, memory reconstruction, evidence
  conflict, and conversational/social holdouts through their official scoring
  pathways;
- compare weight-space and decision-space composition using identical source
  specialists.

## Adaptive and learned inference

- calibrate disagreement, vote entropy, margin, and pairwise JS as escalation
  signals on disjoint DEV folds;
- learn cost-sensitive consultation while retaining a fixed hard-consensus
  fallback;
- test weighted voting, learned voting, routers, and verifiers against simple
  hard/rank baselines;
- investigate latent/recurrent thinking only with explicit training-time and
  compute controls.

## Systems

- exact shared-cache or adapter-batched inference with decision-parity tests;
- parallel expert execution;
- quantized backbones and adapters;
- hardware-aware scheduling and adapter residency;
- compressed top-k communication between expert workers;
- quality/latency/energy frontiers across desktop and server GPUs.

Long-term items are hypotheses, not promised paper contributions.
