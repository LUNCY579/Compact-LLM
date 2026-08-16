#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: scripts/run_all.sh CONFIG [RUN_DIR] [--skip-sealed]" >&2
  exit 2
fi
CONFIG="$1"
shift
RUN_DIR=""
SKIP_SEALED=0
if [[ $# -gt 0 && "$1" != "--skip-sealed" ]]; then
  RUN_DIR="$1"
  shift
fi
if [[ $# -gt 0 && "$1" == "--skip-sealed" ]]; then
  SKIP_SEALED=1
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "unexpected arguments: $*" >&2
  exit 2
fi
ARGS=(run --config "$CONFIG")
if [[ -n "$RUN_DIR" ]]; then ARGS+=(--run-dir "$RUN_DIR"); fi
if [[ "$SKIP_SEALED" -eq 1 ]]; then ARGS+=(--skip-sealed); fi
rineya-soup "${ARGS[@]}"
