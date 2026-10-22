# EFE-001: Evidence, Function, Equilibrium

**Research programme + runnable CPU references. Native model experiments NOT RUN.**

One dedicated branch for three related but distinct questions in Rineya:

| Track | Question | Priority / entry condition |
|---|---|---|
| EFE-FUNC-001 | Can better allocation of existing update directions improve composition? | First quality-improvement track; confirm independent coefficients before growing capacity |
| EFE-KV-001 | Does informed selection beat random retention under identical protection? | Near-term inference control, only where compressible generated KV is substantial |
| EFE-DEQ-001 | Does useful finite refinement converge toward a useful stable state? | Conditional architecture research, not a default decoder replacement |

Repository: `LUNCY579/ChatBubble`  
Branch: `research/rineya-evidence-function-equilibrium-v1`  
Base / PR target: `post-training-Mimo` at `03db218ceebc61a2bc04f2482cf7e094045e01c1`  
Created: September 29, 2026.

## Read first

- [Research plan](RESEARCH_PLAN.md): hypotheses, controls, falsification, cumulative experiments.
- [Host handoff](HOST_HANDOFF.md): native integration boundaries and required receipts.
- [Source ledger](SOURCES.md): papers, inspected repository evidence, and transcript-only claims.
- [Machine-readable plan](plan.json): workstream order and unresolved native prerequisites.
- [Validation](VALIDATION.md): what actually ran in this implementation session.

## Run the bounded references

Use an isolated Python environment, not an existing native research environment.
The commands below are issued from the repository root. Python 3.13.5 was tested.

```bash
python -m venv .venv-efe
# Activate this environment using the command appropriate to your shell.
python -m pip install -r experiments/rineya_evidence_function_equilibrium_v1/requirements.txt
python -m unittest discover -s experiments/rineya_evidence_function_equilibrium_v1 -p 'test_*.py' -v
python experiments/rineya_evidence_function_equilibrium_v1/efe_reference.py plan
python experiments/rineya_evidence_function_equilibrium_v1/efe_reference.py smoke --out efe-smoke-001.json
```

Activate before installing. POSIX: `source .venv-efe/bin/activate`.
PowerShell: `.venv-efe\Scripts\Activate.ps1`.
The output path must be new and its parent must exist; existing results are never overwritten.
Do not commit local smoke files or the virtual environment.

The smoke uses invented arrays and logical positions. Its coefficients are not the
historical INDEP-001 coefficients. It is not a checkpoint evaluation, paper replication,
cache-memory benchmark, or demonstration of cumulative skill gains.

```bash
python experiments/rineya_evidence_function_equilibrium_v1/efe_reference.py native
# Expected exit code: 2. Native adapters are not implemented in EFE-001.
```

## Implemented versus planned

`efe_reference.py` implements per-KV-head seeded logical-position selection, matched
recency/scored reference controls, a total protected-token budget check, finite
least-squares coefficient fitting on supplied output directions, a bounded forward
fixed-point solver with undamped stopping, a small dense implicit-adjoint check,
and declared split-identity/parent/hash/spent guards.

It does **not** implement native JVP extraction, checkpoint composition/export,
physical KV compaction, attention kernels, a content-scoring algorithm, sidecar
training, adaptive representation growth, an LLM implicit backward pass, or native
promotion. The generic scored reference consumes supplied scores; it does not
reproduce BM25, R-KV, or any existing selector. The dense adjoint is only a
small-matrix qualification reference, not a scalable DEQ training implementation.

No production files, existing experiment results, checkpoints, quarantine policies,
or sibling branches are changed. No model downloads, GPU jobs, paid calls, native
training, or benchmark selection are launched.
