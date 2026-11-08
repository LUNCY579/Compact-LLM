"""Import preserved implementations rather than copying their scientific code."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ARMR = REPO / 'experiments/anchor_routed_middle_recurrence_v1'
LAB = REPO / 'Rineya_Cross_Scale_Capability_Transfer_Lab_v1'
GRAFT = REPO / 'experiments/cross_scale_layer_graft_v1'
for directory in (ARMR / 'src', ARMR / 'scripts', LAB / 'src'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


@lru_cache(maxsize=1)
def legacy_graft():
    spec = importlib.util.spec_from_file_location('supa_historical_graft_v1', GRAFT / 'graft.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
