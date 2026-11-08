# Sources and implementation relationship

Reviewed on 2026-09-27 (local date).

1. Ye, H., Lu, Y., Dong, H., Su, Z., and Song, G. **Harness-Zero: Harness Distillation
   via Agent-as-Harness**, arXiv:2609.24974v1 (21 September 2026).
   https://arxiv.org/abs/2609.24974
   https://arxiv.org/html/2609.24974v1
   Sections 3.2–3.3 motivate student-first review, target-action-space corrections,
   private reference guidance, and SFT on accepted actions with reviewer reasoning
   excluded. This implementation does not claim to reproduce the paper's benchmarks.

2. Official repository, `metaevo-ai/harness-zero`:
   https://github.com/metaevo-ai/harness-zero
   Inspected its public README to check the collection/filter/SFT boundaries.
   No source files or prompt text are vendored. This is an independently written,
   constrained Rineya research adaptation, not the upstream Harbor/Tinker pipeline.

3. Inherited Rineya PRLC-001 at commit
   `5517361ec09e02308eae0352b2e53a70853c5245`, repository `LUNCY579/ChatBubble`:
   `experiments/progressive_rl_curriculum_v1/README.md`, `schema.py`, `backends.py`.
   The actual Prompt, Episode, and Task definitions informed the transport bridge.
   The local schema used for bridge tests has verified Git blob SHA
   `110844a75e1c3a82748e7e83569ee8c80ec3c042`. Upstream source is not modified.

## Deliberate departures

The paper uses a minimal Bash-based harness and a much larger student. RHD-001 uses
four typed local actions and generated protocol fixtures, with no shell or live
benchmark environment. The procedural reference here is hand-authored, not evolved.
Native transfer to a compact Rineya model is an open experimental question.

The paper-inspired mechanism is implemented; its scale, teacher, optimizer recipe,
benchmark scores, behavioral recovery figures, and deployment cost benefits are
not reproduced or claimed. Oracle supervision is disabled by default. Preference
pair export and parent-disjoint retention controls are Rineya-specific extensions.
