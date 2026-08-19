# Release Procedure

## 1. Freeze the candidate

- Choose the exact commit.
- Confirm version consistency in `pyproject.toml`, `_version.py`, API metadata, schemas, changelog, and package prefix.
- Freeze configuration and evaluation artifacts.
- Do not tune against sealed results after the candidate is chosen.

## 2. Run quality checks

```bash
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m mypy src
python -m pytest --cov=rineya_harness --cov-report=term-missing
python scripts/export_schemas.py --check
python scripts/validate_repository.py --with-tests
```

Run all three offline demos and inspect their state transitions:

```bash
rineya-harness demo restaurant
rineya-harness demo hanoi
rineya-harness demo research
```

## 3. Review the artifact boundary

Confirm the package contains no:

- `.env` files;
- API keys or tokens;
- model weights;
- runtime SQLite databases or WAL files;
- local traces;
- test caches or bytecode;
- private DEV/TEST data;
- machine-specific absolute paths.

## 4. Generate release artifacts

```bash
python scripts/package_release.py \
  --output ../Rineya_Context_Harness_v0.1.0.zip
```

This regenerates schemas, runs validation and tests, creates a repository manifest, normalizes ZIP timestamps, verifies ZIP integrity, and writes a SHA-256 sidecar.

## 5. Verify from a clean extraction

Extract into a new directory, create a fresh virtual environment, install the package, run `rineya-harness doctor`, execute the fixture demos, and rerun the test suite. Record OS, Python version, commands, and output hashes.

## 6. Publish with limitations

Release notes must distinguish:

- implemented and tested behavior;
- architecture-only extension points;
- fixture/offline evidence;
- real Rineya checkpoint evidence;
- security limitations;
- migration and rollback instructions.
