#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: scripts/run_stage.sh CONFIG RUN_DIR STAGE" >&2
  exit 2
fi
rineya-soup stage "$3" --config "$1" --run-dir "$2"
