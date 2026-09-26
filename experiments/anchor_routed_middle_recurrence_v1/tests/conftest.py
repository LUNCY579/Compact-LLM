from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parents[1] / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src")]
