import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path[:0] = [str(ROOT), str(REPO / "experiments/anchor_routed_middle_recurrence_v1/src"),
               str(REPO / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src")]

import torch
torch.set_num_threads(2)
