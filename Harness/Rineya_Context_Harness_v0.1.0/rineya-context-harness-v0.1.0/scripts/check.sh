#!/usr/bin/env sh
set -eu
export PYTHONPATH=src
python -m compileall -q src
python scripts/export_schemas.py --check
pytest --cov=rineya_harness --cov-report=term-missing
python scripts/validate_repository.py
