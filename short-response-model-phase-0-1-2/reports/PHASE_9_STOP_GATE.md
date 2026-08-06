# Phase 9 — Student distillation stop gate

## Automated status

- Recommended production candidate: `none`
- Best research-control candidate: `sft_narrow_100m`
- Human decision: `PENDING`
- The recommendation uses the shared proportional resource-gain versus quality-opportunity-loss policy.
- Narrow random-init profiles are research controls and are never auto-recommended for production.
- No universal grounding or unsupported-answer threshold is invented. A product safety cap is enforced only when explicitly calibrated and supplied.

## Candidate assessments

### sft_warm_344m

- Track class: `production_eligible_track`
- Weight reduction versus teacher: `0.240782`
- Quality opportunity loss: `0.418667`
- Opportunity margin: `-0.177885`
- Proportionally worthwhile: `False`
- Real-RAG supported correctness: `0.000000`
- Real-RAG evidence faithfulness: `0.000000`
- Real-RAG unsupported confident answer rate: `1.000000`
- Product safety blockers: `none`

### kd_warm_344m

- Track class: `production_eligible_track`
- Weight reduction versus teacher: `0.240782`
- Quality opportunity loss: `0.418667`
- Opportunity margin: `-0.177885`
- Proportionally worthwhile: `False`
- Real-RAG supported correctness: `0.000000`
- Real-RAG evidence faithfulness: `0.000000`
- Real-RAG unsupported confident answer rate: `1.000000`
- Product safety blockers: `none`

### sft_narrow_100m

- Track class: `research_control`
- Weight reduction versus teacher: `0.770892`
- Quality opportunity loss: `0.000000`
- Opportunity margin: `0.770892`
- Proportionally worthwhile: `True`
- Real-RAG supported correctness: `0.000000`
- Real-RAG evidence faithfulness: `1.000000`
- Real-RAG unsupported confident answer rate: `1.000000`
- Product safety blockers: `none`

### kd_narrow_100m

- Track class: `research_control`
- Weight reduction versus teacher: `0.770892`
- Quality opportunity loss: `0.430000`
- Opportunity margin: `0.340892`
- Proportionally worthwhile: `True`
- Real-RAG supported correctness: `0.000000`
- Real-RAG evidence faithfulness: `1.000000`
- Real-RAG unsupported confident answer rate: `1.000000`
- Product safety blockers: `none`

### sft_narrow_130m

- Track class: `research_control`
- Weight reduction versus teacher: `0.709489`
- Quality opportunity loss: `0.430000`
- Opportunity margin: `0.279489`
- Proportionally worthwhile: `True`
- Real-RAG supported correctness: `0.000000`
- Real-RAG evidence faithfulness: `1.000000`
- Real-RAG unsupported confident answer rate: `1.000000`
- Product safety blockers: `none`

### kd_narrow_130m

- Track class: `research_control`
- Weight reduction versus teacher: `0.709489`
- Quality opportunity loss: `0.430000`
- Opportunity margin: `0.279489`
- Proportionally worthwhile: `True`
- Real-RAG supported correctness: `0.000000`
- Real-RAG evidence faithfulness: `1.000000`
- Real-RAG unsupported confident answer rate: `1.000000`
- Product safety blockers: `none`

### sft_narrow_170m

- Track class: `research_control`
- Weight reduction versus teacher: `0.626303`
- Quality opportunity loss: `0.440889`
- Opportunity margin: `0.185414`
- Proportionally worthwhile: `True`
- Real-RAG supported correctness: `0.000000`
- Real-RAG evidence faithfulness: `0.000000`
- Real-RAG unsupported confident answer rate: `1.000000`
- Product safety blockers: `none`

### kd_narrow_170m

- Track class: `research_control`
- Weight reduction versus teacher: `0.626303`
- Quality opportunity loss: `0.000000`
- Opportunity margin: `0.626303`
- Proportionally worthwhile: `True`
- Real-RAG supported correctness: `0.000000`
- Real-RAG evidence faithfulness: `0.000000`
- Real-RAG unsupported confident answer rate: `1.000000`
- Product safety blockers: `none`

## Training and cross-profile comparisons

- `kd_warm_344m_vs_sft_warm_344m`: decision `NO_NET_QUALITY_IMPROVEMENT`, net quality change `0.000000`
- `kd_narrow_100m_vs_sft_narrow_100m`: decision `NO_NET_QUALITY_IMPROVEMENT`, net quality change `-0.611333`
- `kd_narrow_130m_vs_sft_narrow_130m`: decision `NO_NET_QUALITY_IMPROVEMENT`, net quality change `0.000000`
- `kd_narrow_170m_vs_sft_narrow_170m`: decision `NET_QUALITY_IMPROVEMENT`, net quality change `226331658291.457306`
- `sft_narrow_130m_vs_sft_narrow_100m`: decision `NO_NET_QUALITY_IMPROVEMENT`, net quality change `-0.611333`
- `sft_narrow_170m_vs_sft_narrow_100m`: decision `NO_NET_QUALITY_IMPROVEMENT`, net quality change `-0.622222`
- `sft_warm_344m_vs_sft_narrow_100m`: decision `NO_NET_QUALITY_IMPROVEMENT`, net quality change `-0.600000`
- `sft_narrow_170m_vs_sft_narrow_130m`: decision `NO_NET_QUALITY_IMPROVEMENT`, net quality change `-0.020069`
- `sft_warm_344m_vs_sft_narrow_130m`: decision `NET_QUALITY_IMPROVEMENT`, net quality change `0.022711`
- `sft_warm_344m_vs_sft_narrow_170m`: decision `NET_QUALITY_IMPROVEMENT`, net quality change `25125628140.703518`
- `kd_narrow_130m_vs_kd_narrow_100m`: decision `NO_NET_QUALITY_IMPROVEMENT`, net quality change `0.000000`
- `kd_narrow_170m_vs_kd_narrow_100m`: decision `NET_QUALITY_IMPROVEMENT`, net quality change `201206030150.776520`
- `kd_warm_344m_vs_kd_narrow_100m`: decision `NET_QUALITY_IMPROVEMENT`, net quality change `0.022711`
- `kd_narrow_170m_vs_kd_narrow_130m`: decision `NET_QUALITY_IMPROVEMENT`, net quality change `201206030150.776520`
- `kd_warm_344m_vs_kd_narrow_130m`: decision `NET_QUALITY_IMPROVEMENT`, net quality change `0.022711`
- `kd_warm_344m_vs_kd_narrow_170m`: decision `NO_NET_QUALITY_IMPROVEMENT`, net quality change `-0.418667`

## Human review required

Review raw oracle/real/adversarial outputs, retrieval misses, false abstentions, unsupported claims, role-play behavior, latency, total process memory, and quantized-runtime evidence before recording PASS.
A PASS should name the exact student artifact and whether it is approved as a production candidate, research-only control, or intermediate teacher.
Do not promote a narrow random-init student as a release model without separate language-readiness evidence.
