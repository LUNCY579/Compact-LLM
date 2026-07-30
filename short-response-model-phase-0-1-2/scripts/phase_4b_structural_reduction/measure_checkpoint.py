# Purpose: Backward-compatible Phase 4B wrapper for the shared footprint measurer.
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.measure_model_footprint import main

if __name__ == "__main__":
    main()
