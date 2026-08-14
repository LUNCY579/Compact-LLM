# Future Work

## Immediate evidence gaps

1. Finish the frozen same-split free-running DEV comparison for hard, raw and
   calibrated probability pooling, and reciprocal-rank aggregation.
2. Finish the frozen adaptive K3→K7 free-running measurement and report its
   actual accuracy/latency frontier.
3. Finish the frozen best-at-K/leave-one-out DEV committee panel and report
   trajectory, answer-class, latency, VRAM, and accuracy-per-forward effects.
4. Finish the frozen one-condition v2.2 calibrated TEST confirmation without
   changing its DEV-fitted temperatures or separate two-comparison family.
5. Finish the frozen v2.1.2 fixed-method cross-corpus run and retain its
   training-overlap and prior-use caveats.
6. Compare teacher-forced and free-running ordering on identical DEV examples,
   then update the trajectory and calibration-sufficiency claims.
7. Train new matched controls only if a later causal-specialization study is
   separately authorized; the current bounded proxy is not sufficient.

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
