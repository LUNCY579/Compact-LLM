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


## Optional Phase 9B reconstruction

The backward-compatible default is `--memory-reconstruction-mode off`. Use `prompted` to test the source-state protocol with an ordinary approved checkpoint, or `trained` only with a human-approved Phase 9B checkpoint:

```powershell
python .\scripts\phase_11_production_rag\serve_production_rag.py `
  --model <model> `
  --knowledge-db <db> `
  --answerability-calibration <json> `
  --memory-reconstruction-mode trained `
  --reconstruction-max-new-tokens 128
```

Requests may include `"current_state": {...}`. Knowledge records may provide top-level `source_state`, `validity_conditions`, and `memory_type`; snapshot construction preserves them in SQLite metadata. The response reports the reconstruction payload, JSON validity, memory IDs, extra tokens, and two-pass latency.
