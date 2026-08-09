# Phase 9 — Student distillation stop gate

## Automated status

- Recommended production candidate: `none`
- Best research-control candidate: `none`
- Human decision: `PENDING`
- The recommendation uses the shared proportional resource-gain policy plus a Phase 9 task-contract deficiency cost.
- Core RAG outcomes cannot be hidden by an equally broken teacher or by generic conversational quality scores.
- Narrow random-init profiles are research controls and are never auto-recommended for production.
- Production recommendation also requires non-degenerate functional evidence: applicable cases, at least one correct supported answer, and at least one correct abstention when those case types exist.
- Initialization→SFT, SFT→KD, initialization→KD, and candidate→teacher evidence are kept separate.
- Cross-candidate quality comparisons use bounded absolute deltas, preventing division-by-zero explosions when a reference metric is zero.
- No universal grounding or unsupported-answer threshold is invented. A product safety cap is enforced only when explicitly calibrated and supplied.

## Candidate assessments

### sft

- Track class: `production_eligible_track`
- Weight reduction versus teacher: `0.520189`
- Generic relative quality loss: `0.006214`
- Phase 9 functional deficiency: `1.000000`
- Combined Phase 9 quality loss: `1.000000`
- Opportunity margin: `-0.479811`
- Proportionally worthwhile: `False`
- Real-RAG supported correctness: `0.359615`
- Real-RAG evidence faithfulness: `0.960905`
- Real-RAG correct abstention: `0.764706`
- Real-RAG unsupported confident answer rate: `0.235294`
- Real-RAG conflict detection: `0.645161`
- Real-RAG procedure correctness: `0.000000`
- Real-RAG social usable rate: `1.000000`
- Real-RAG prompt support recall: `1.000000`
- Real-RAG prompt/decode integrity failure rates: `0.000000` / `0.000000`
- Filtered/passthrough injection resistance: `1.000000` / `1.000000`
- Real-RAG supported successes/cases: `187` / `520`
- Real-RAG safe abstentions/cases: `104` / `136`
- Real-RAG detected conflicts/cases: `40` / `62`
- Real-RAG correct procedures/cases: `0` / `150`
- Real-RAG usable social replies/cases: `20` / `20`
- Evaluation research-ready: `True`
- Functional readiness blockers: `['real_rag demonstrates zero correct procedures']`
- Product safety blockers: `no configured cap; diagnostic only`

## Training and cross-profile comparisons

- `sft_vs_teacher`: decision `NET_QUALITY_IMPROVEMENT`, net quality change `0.064214` via `bounded_absolute_delta_v1`

## Human review required

Review raw oracle/real/adversarial outputs, retrieval misses, false abstentions, unsupported claims, role-play behavior, latency, total process memory, and quantized-runtime evidence before recording PASS.
A PASS should name the exact student artifact and whether it is approved as a production candidate, research-only control, or intermediate teacher.
Do not promote a narrow random-init student as a release model without separate language-readiness evidence.
