#!/usr/bin/env sh
set -eu
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[api,dev]'
python scripts/export_schemas.py
rineya-harness doctor --config configs/default.yaml
