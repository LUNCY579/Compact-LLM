from __future__ import annotations

"""Command-line adapter for resolving the Phase 4B selected checkpoint.

PowerShell runners call this script and capture its single-line stdout. Keeping
resolution in Python avoids duplicating JSON parsing and validation in every
phase runner.
"""

import argparse
from pathlib import Path

from scripts.common.reference_selection import (
    DEFAULT_REFERENCE_MANIFEST,
    resolve_selected_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the checkpoint selected and approved in Phase 4B."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_REFERENCE_MANIFEST,
        help="Phase 4B selected-reference JSON manifest.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional explicit model for a labeled control/ablation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(resolve_selected_model(args.model, args.manifest))


if __name__ == "__main__":
    main()
