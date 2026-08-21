# Phase 9 RAG evaluation

- Model: `C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_R2_CrossDepth_Soup_Mini_v1\artifacts\hybrids\r2_x_v1_5e4_v2`
- Adapter: `None`
- Benchmark: `C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2\data\phase_9_student_distillation\rag_reasoning_v3_large\reviewed_splits\dev.reviewed.jsonl`
- Model/checkpoint bytes: `1899269231`
- Peak allocated VRAM bytes: `0`
- Process RSS bytes: `None`

## Conditions

### oracle_rag

- Cases: `752`
- Mean quality: `0.6591362830431978`
- Supported correctness: `0.8471698113207548`
- Evidence faithfulness: `1.0` (n=593)
- Correct abstention: `0.6618705035971223` (n=139)
- Unsupported confident answer rate: `0.3381294964028777`
- Conflict safe handling: `1.0` (n=63)
- Conflict explicit acknowledgment (diagnostic): `1.0`
- Procedure correctness: `0.7133333333333334` (n=150)
- Social usable rate: `1.0` (n=20)
- Qualified/ambiguous evidence calibration: `0.3684210526315789` (n=76)
- Stale/current precedence: `1.0` (n=38)
- Prompt-injection task success: `1.0` (n=38)
- Retrieval recall@k: `None` (n=0)
- Prompt support recall@k: `1.0` (n=530)
- Prompt integrity failure rate: `0.0`
- Decode integrity failure rate: `0.0`
- Injection resistance: `None` (n=0)
- Median/P90 completion tokens: `7.0` / `14`

### real_rag

- Cases: `752`
- Mean quality: `0.6459747340425531`
- Supported correctness: `0.6283018867924528`
- Evidence faithfulness: `1.0` (n=485)
- Correct abstention: `0.7266187050359713` (n=139)
- Unsupported confident answer rate: `0.2733812949640288`
- Conflict safe handling: `1.0` (n=63)
- Conflict explicit acknowledgment (diagnostic): `0.6031746031746031`
- Procedure correctness: `0.7` (n=150)
- Social usable rate: `1.0` (n=20)
- Qualified/ambiguous evidence calibration: `0.5` (n=76)
- Stale/current precedence: `1.0` (n=38)
- Prompt-injection task success: `1.0` (n=38)
- Retrieval recall@k: `1.0` (n=530)
- Prompt support recall@k: `1.0` (n=530)
- Prompt integrity failure rate: `0.0`
- Decode integrity failure rate: `0.0`
- Injection resistance: `None` (n=0)
- Median/P90 completion tokens: `8.0` / `14`
