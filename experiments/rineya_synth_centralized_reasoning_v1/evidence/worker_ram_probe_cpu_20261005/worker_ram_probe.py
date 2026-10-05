"""Stage-only worker available-RAM candidate; no integration or launch authority.

The unchanged Native7 helper calls GlobalMemoryStatusEx on every invocation.
Its cached API wrapper is not a numeric RAM cache. Separate volatile readings
are not required to be bitwise equal; no psutil fallback or retained RAM value.
"""
from pathlib import Path
import sys

WORK=Path(__file__).resolve().parent;PACKAGE=WORK.parents[1]
sys.path.insert(0,str(PACKAGE))
import native_process


def available_bytes(*,status_call=None):
    """Return fresh available physical bytes, retaining API/errors verbatim."""
    status_call=native_process.system_memory_status if status_call is None else status_call
    if not callable(status_call):raise TypeError('fresh memory status callback required')
    status=status_call()
    if type(status) is not dict or set(status)!={'available_bytes','total_bytes'}:
        raise ValueError('exact existing native_process memory-status fields required')
    available,total=status['available_bytes'],status['total_bytes']
    if type(available) is not int or type(total) is not int or not 0<=available<=total<=2**64-1 or total==0:
        raise ValueError('valid nonboolean UInt64 physical memory bytes required')
    return available
