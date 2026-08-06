# Phase 11 file responsibilities

| File | Responsibility |
|---|---|
| `run_phase_11.ps1` | Preparation, quantization, dense/deployment benchmarks, gate, packaging. |
| `phase11_common.py` | Auditable routing, answerability features/calibration, conservative grounding policy. |
| `build_knowledge_snapshot.py` | Validate, deduplicate, hash, and index immutable knowledge/procedure inputs. |
| `calibrate_answerability.py` | Select feature weights and threshold from development data and declared costs. |
| `quantize_student.py` | Dense, bitsandbytes NF4/INT8, or caller-supplied llama.cpp GGUF artifact creation. |
| `benchmark_production_runtime.py` | Complete end-to-end quality, latency, RSS/VRAM, retrieval, gate, and grounding benchmark. |
| `serve_production_rag.py` | Persistent local JSONL/interactive production process. |
| `check_phase_11_gate.py` | Compare measured total process memory against proportional quality loss and write pending release gate. |
| `package_phase_11_release.py` | Hash and package the exact human-approved runtime candidate with evidence and rollback. |
