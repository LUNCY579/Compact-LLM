import sys
from pathlib import Path

# Keep the lab runnable standalone without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
