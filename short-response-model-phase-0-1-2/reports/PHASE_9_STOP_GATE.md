# Phase 9 — Student distillation stop gate

## Automated status

- Recommended candidate: `none`
- Human decision: `PENDING`
- The recommendation uses the shared proportional resource-gain versus quality-opportunity-loss policy.
- No universal grounding or unsupported-answer threshold is invented. A product safety cap is enforced only when explicitly calibrated and supplied.

## Candidate assessments

### sft

- Weight reduction versus teacher: `0.240782`
- Quality opportunity loss: `0.418667`
- Opportunity margin: `-0.177885`
- Proportionally worthwhile: `False`
- Real-RAG supported correctness: `0.000000`
- Real-RAG evidence faithfulness: `0.000000`
- Real-RAG unsupported confident answer rate: `1.000000`
- Product safety blockers: `none`

### kd

- Weight reduction versus teacher: `0.240782`
- Quality opportunity loss: `0.418667`
- Opportunity margin: `-0.177885`
- Proportionally worthwhile: `False`
- Real-RAG supported correctness: `0.000000`
- Real-RAG evidence faithfulness: `0.000000`
- Real-RAG unsupported confident answer rate: `1.000000`
- Product safety blockers: `none`

## Same-architecture training comparisons

- `kd_vs_sft`: decision `NO_NET_QUALITY_IMPROVEMENT`, net quality change `0.000000`

## Human review required

Review raw oracle/real/adversarial outputs, retrieval misses, false abstentions, unsupported claims, role-play behavior, latency, total process memory, and quantized-runtime evidence before recording PASS.
A PASS should name the exact student artifact and whether it is approved as a production candidate, research-only control, or intermediate teacher.
