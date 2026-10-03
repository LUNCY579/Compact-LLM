"""Direct-script wrapper; equivalent to python -m forge_rl.build."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forge_rl.build import main
if __name__ == "__main__":
    main()
