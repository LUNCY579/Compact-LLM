from pathlib import Path
import sys

PROGRAM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROGRAM))
