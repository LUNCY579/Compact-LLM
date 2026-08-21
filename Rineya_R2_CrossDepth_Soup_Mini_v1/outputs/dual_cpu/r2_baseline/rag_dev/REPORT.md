# Phase 9 RAG evaluation

- Model: `C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_R2_CrossDepth_Soup_Mini_v1\scaffolds\rineya-core-en32k-28L`
- Adapter: `None`
- Benchmark: `C:\Users\clclc\BusinessProject\Chat-Bubble\short-response-model-phase-0-1-2\data\phase_9_student_distillation\rag_reasoning_v3_large\reviewed_splits\dev.reviewed.jsonl`
- Model/checkpoint bytes: `952067915`
- Peak allocated VRAM bytes: `0`
- Process RSS bytes: `None`

## Conditions

### oracle_rag

- Cases: `752`
- Mean quality: `0.6587618874919406`
- Supported correctness: `0.7679245283018868`
- Evidence faithfulness: `1.0` (n=512)
- Correct abstention: `0.9928057553956835` (n=139)
- Unsupported confident answer rate: `0.007194244604316547`
- Conflict safe handling: `1.0` (n=63)
- Conflict explicit acknowledgment (diagnostic): `1.0`
- Procedure correctness: `0.43333333333333335` (n=150)
- Social usable rate: `1.0` (n=20)
- Qualified/ambiguous evidence calibration: `0.5` (n=76)
- Stale/current precedence: `1.0` (n=38)
- Prompt-injection task success: `1.0` (n=38)
- Retrieval recall@k: `None` (n=0)
- Prompt support recall@k: `1.0` (n=530)
- Prompt integrity failure rate: `0.0`
- Decode integrity failure rate: `0.0`
- Injection resistance: `None` (n=0)
- Median/P90 completion tokens: `10.0` / `14`

### real_rag

- Cases: `752`
- Mean quality: `0.5247729388297872`
- Supported correctness: `0.3867924528301887`
- Evidence faithfulness: `1.0` (n=396)
- Correct abstention: `0.7266187050359713` (n=139)
- Unsupported confident answer rate: `0.2733812949640288`
- Conflict safe handling: `1.0` (n=63)
- Conflict explicit acknowledgment (diagnostic): `0.6031746031746031`
- Procedure correctness: `0.35333333333333333` (n=150)
- Social usable rate: `1.0` (n=20)
- Qualified/ambiguous evidence calibration: `0.5` (n=76)
- Stale/current precedence: `1.0` (n=38)
- Prompt-injection task success: `1.0` (n=38)
- Retrieval recall@k: `1.0` (n=530)
- Prompt support recall@k: `1.0` (n=530)
- Prompt integrity failure rate: `0.0`
- Decode integrity failure rate: `0.0`
- Injection resistance: `None` (n=0)
- Median/P90 completion tokens: `10.0` / `19`
