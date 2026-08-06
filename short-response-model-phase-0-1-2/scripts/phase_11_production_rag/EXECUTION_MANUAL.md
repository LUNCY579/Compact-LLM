# Phase 11 execution manual

## Preconditions

- Phase 10 human gate is `PASS`.
- `artifacts/references/phase_10_student_reference.json` exists.
- Production knowledge and procedures have stable unique IDs and trusted source metadata.
- Development/test records remain disjoint from training documents.

## Prepare

```powershell
.\scripts\phase_11_production_rag\run_phase_11.ps1 -Mode Prepare -Knowledge <knowledge.jsonl> -Procedures <procedures.jsonl>
```

## Benchmark and gate

```powershell
.\scripts\phase_11_production_rag\run_phase_11.ps1 -Mode Full -QuantizationBackend bitsandbytes_nf4 -ReuseExisting
```

For GGUF, provide the exact llama.cpp conversion, quantizer, and runtime executables. Runtime arguments vary by llama.cpp release; use repeated `--llama-cli-arg` when invoking `benchmark_production_runtime.py` directly if your build requires additional flags.

## Production process

Use `serve_production_rag.py` for one-shot, interactive, input JSONL, or persistent stdin JSONL operation. It returns route, provenance, gate, grounding, timing, and memory evidence. Application-specific IPC wrappers may sit around this stable process boundary.

## Release

After human `PASS`:

```powershell
.\scripts\phase_11_production_rag\run_phase_11.ps1 -Mode Package -PackageCandidate <registry-name> -PackageMode copy
```
