# Phase 9 v2.1.2 Dataset Audit

**Status: PASS**

- Canonical general-reasoning rows: **13,200**
- Reasoning families: **33**
- TRAIN / DEV / sealed TEST: **10,560 / 1,320 / 1,320**
- Answerable / abstention: **12,000 / 1,200**
- Easy / moderate / hard: **4,000 / 6,400 / 2,800**
- Stage A reasoning view: **4,800** rows
- Stage B mixed view: **6,000** rows
  - general reasoning: **3,600**
  - original Phase 9 train-only replay: **1,200**
  - v2.1.1 train-only replay: **1,200**
- Duplicate generated IDs: **0**
- Split-ID overlap: **0**
- Assistant targets containing chain-of-thought rationale: **0 by design**

Static validation command:

```powershell
python .\scripts\phase_9_student_distillation\validate_general_reasoning_v212.py `
  --corpus-root .\data\phase_9_student_distillation\general_reasoning_v212
```

The corpus is synthetic/deterministic but intentionally real-world-reflected. It is designed to train transferable operations rather than external factual knowledge.
