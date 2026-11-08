"""One heavy model process at a time on this host; no background resource races."""
from contextlib import contextmanager
import os
from .legacy import ROOT
from .hashing import write_json


@contextmanager
def machine_lease():
    path = ROOT / '.runtime.lock'
    write_json(path, {'pid':os.getpid(), 'purpose':'exclusive CPU model execution'})
    try:
        yield
    finally:
        path.unlink()
