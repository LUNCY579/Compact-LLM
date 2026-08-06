# Phase 9 — Student distillation stop gate

## Automated status

- Recommended production candidate: `none`
- Best research-control candidate: `none`
- Human decision: `PENDING`
- The recommendation uses the shared proportional resource-gain versus quality-opportunity-loss policy.
- Narrow random-init profiles are research controls and are never auto-recommended for production.
- Production recommendation also requires non-degenerate functional evidence: applicable cases, at least one correct supported answer, and at least one correct abstention when those case types exist.
- Cross-candidate quality comparisons use bounded absolute deltas, preventing division-by-zero explosions when a reference metric is zero.
- No universal grounding or unsupported-answer threshold is invented. A product safety cap is enforced only when explicitly calibrated and supplied.

## Candidate assessments

### sft

- Track class: `production_eligible_track`
- Weight reduction versus teacher: `0.225059`
- Quality opportunity loss: `0.000000`
- Opportunity margin: `0.225059`
- Proportionally worthwhile: `True`
- Real-RAG supported correctness: `0.000000`
- Real-RAG evidence faithfulness: `n/a`
- Real-RAG unsupported confident answer rate: `0.000000`
- Real-RAG supported successes/cases: `0` / `4`
- Real-RAG safe abstentions/cases: `1` / `1`
- Evaluation research-ready: `True`
- Functional readiness blockers: `['real_rag demonstrates zero correct supported answers']`
- Product safety blockers: `none`

### progressive_kd

- Track class: `production_eligible_track`
- Weight reduction versus teacher: `0.225059`
- Quality opportunity loss: `0.000393`
- Opportunity margin: `0.224666`
- Proportionally worthwhile: `True`
- Real-RAG supported correctness: `0.000000`
- Real-RAG evidence faithfulness: `n/a`
- Real-RAG unsupported confident answer rate: `0.000000`
- Real-RAG supported successes/cases: `0` / `4`
- Real-RAG safe abstentions/cases: `1` / `1`
- Evaluation research-ready: `True`
- Functional readiness blockers: `['real_rag demonstrates zero correct supported answers']`
- Product safety blockers: `none`

## Training and cross-profile comparisons

- `progressive_kd_vs_sft`: decision `NO_NET_QUALITY_IMPROVEMENT`, net quality change `-0.000021` via `bounded_absolute_delta_v1`

## Human review required

Review raw oracle/real/adversarial outputs, retrieval misses, false abstentions, unsupported claims, role-play behavior, latency, total process memory, and quantized-runtime evidence before recording PASS.
A PASS should name the exact student artifact and whether it is approved as a production candidate, research-only control, or intermediate teacher.
Do not promote a narrow random-init student as a release model without separate language-readiness evidence.
