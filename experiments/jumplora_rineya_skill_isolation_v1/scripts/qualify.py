"""Repository-root-friendly wrapper for the JLR-QUAL-001 CLI."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.jumplora_rineya_skill_isolation_v1.run_qualification import main


if __name__ == "__main__":
    raise SystemExit(main())
