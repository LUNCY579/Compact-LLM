# Phase 11 — production RAG runtime and student release

Phase 11 converts the accepted Phase 10 student into a complete measured local runtime: immutable knowledge snapshot, calibrated answerability gate, optional quantization, route/retrieve/gate/generate/ground benchmark, persistent JSONL process, release registry, human gate, and reproducible release package.

Main runner:

```powershell
.\scripts\phase_11_production_rag\run_phase_11.ps1 -Mode Full -QuantizationBackend dense_copy
```

Runtime process:

```powershell
python .\scripts\phase_11_production_rag\serve_production_rag.py --model <model> --knowledge-db <db> --answerability-calibration <json>
```
