from __future__ import annotations

"""Compatibility entry point for the authoritative Evaluator v4.3 verifier."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.verify_quality_evaluation_v4_3 import main


if __name__ == "__main__":
    main()
