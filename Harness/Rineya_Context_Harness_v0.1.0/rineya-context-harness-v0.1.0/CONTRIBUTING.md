# Contributing to Rineya Context Harness

Thank you for improving the Rineya Context Harness. Version 0.1 is deliberately small-model-first, local-first, and conservative about side effects. Changes should make the system easier to verify, not merely more feature-rich.

## Engineering principles

Contributions must preserve these boundaries:

1. **The model proposes; code validates and commits.** Permissions, budgets, revisions, invariant checks, evidence provenance, and irreversible actions belong to deterministic code.
2. **Active context is not the archive.** Removing an item from a prompt must not silently destroy evidence or traceability.
3. **Replacement precedes deletion.** A superseded fact may leave active context only after its replacement is present and the candidate state passes validation.
4. **Small tool surfaces beat noisy tool catalogs.** Add a model-facing tool only when a high-level existing operation cannot express the task safely.
5. **Every loop is bounded.** New loops require explicit budgets, stop predicates, retry behavior, and an escalation path.
6. **Claims require outcome evidence.** A model saying it succeeded is not a passing test.

## Development setup

Python 3.11–3.13 is supported.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[api,dev]'
python scripts/export_schemas.py
python scripts/validate_repository.py --with-tests
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[api,dev]"
python scripts\export_schemas.py
python scripts\validate_repository.py --with-tests
```

## Required checks

Before opening a change, run:

```bash
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m mypy src
python -m pytest --cov=rineya_harness --cov-report=term-missing
python scripts/export_schemas.py --check
python scripts/validate_repository.py
```

Generated schemas are committed artifacts. A contract change must update `schemas/generated/` in the same change.

## Change categories

### State or memory changes

Include tests for:

- stale expected revisions;
- repeated idempotency keys;
- pin protection;
- active dependencies;
- replacement sufficiency;
- rollback/checkpoint behavior;
- prompt rebuild signaling;
- event and trace records.

### Tool changes

Document purpose, arguments, return type, errors, permissions, side effects, idempotency, and postconditions. Prefer a high-level semantic operation over exposing internal parser or storage mechanics.

### Retrieval or web changes

Treat all fetched text as untrusted data. Preserve source URI, retrieval query, content hashes, and passage identity. Do not weaken private-address blocking, redirect limits, size limits, or content-type checks without a threat-model update.

### Workflow changes

Update the workflow graph, legal transitions, budgets, failure taxonomy, and ablation plan. New model calls or search rounds must be measured in evaluation reports.

## Tests and evaluation data

- Unit tests should be deterministic and network-free.
- Integration tests should use fixtures unless the test is explicitly marked and isolated.
- Do not tune on sealed evaluation data.
- Convert discovered failures into stable regression cases.
- Preserve negative results and failed traces when they are scientifically informative.

## Documentation and ADRs

Create an architecture decision record under `docs/adrs/` when a change alters a trust boundary, persistence model, deletion semantics, public contract, security posture, or workflow topology.

## Commit and review expectations

A reviewable change should contain:

- one clearly stated problem;
- the smallest coherent implementation;
- tests proving behavior and failure handling;
- updated schemas/docs where applicable;
- migration and rollback notes for persisted data;
- no secrets, model weights, runtime databases, or private evaluation artifacts.

The project is alpha research infrastructure. Backward-incompatible changes are possible before v1.0, but they must be explicit, versioned, and documented.
