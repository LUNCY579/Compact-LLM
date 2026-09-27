"""A host-memory budget, so a secondary job yields instead of thrashing the machine.

This exists because of a measured incident. With a training run already going, 21 of 32 logical
cores were idle, so a second model job looked free. It was not: the host has 15.2 GB and the
training run alone commits ~16.7 GB, so it was already paging. Adding the second job drove the
pagefile to 13.9 GB and degraded that job from 62 s to 670 s per unit of work -- a 10x
slowdown that looks like CPU contention and is actually thrashing -- while slowing the training
run too.

Idle cores are not evidence of headroom. Free physical memory is. This module makes that check
cheap enough that every secondary job can do it, and gives such a job a clean way to stand down
mid-flight rather than dragging the whole machine into swap.

Two calls:

    require(min_free_gib, label)   at startup  -- refuse to begin
    guard(min_free_gib, label)     between units of work -- raise LowMemoryAbort

A job that catches `LowMemoryAbort` should write whatever it has with a `completed: false`
marker and exit, so a partial run is never mistakable for a finished one.

Uses GlobalMemoryStatusEx directly: no subprocess, microseconds per call, so calling it in a
loop is free.
"""
from __future__ import annotations
import ctypes
import sys

GIB = 1024 ** 3


class LowMemoryAbort(RuntimeError):
    """Raised by `guard` when free physical memory has fallen below the floor."""


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def snapshot() -> dict:
    """Free/total physical and commit, in GiB. Returns zeros off Windows rather than raising."""
    if not sys.platform.startswith('win'):
        return {'available': False}
    m = _MemoryStatusEx()
    m.dwLength = ctypes.sizeof(_MemoryStatusEx)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    committed = (m.ullTotalPageFile - m.ullAvailPageFile) / GIB
    return {'available': True, 'memory_load_percent': int(m.dwMemoryLoad),
            'total_phys_gib': m.ullTotalPhys / GIB, 'free_phys_gib': m.ullAvailPhys / GIB,
            'commit_limit_gib': m.ullTotalPageFile / GIB, 'committed_gib': committed,
            # Commit beyond physical is what actually pages. This is the thrash signal.
            'commit_over_phys_gib': max(0.0, committed - m.ullTotalPhys / GIB)}


def free_gib() -> float:
    s = snapshot()
    return s['free_phys_gib'] if s.get('available') else float('inf')


def describe() -> str:
    s = snapshot()
    if not s.get('available'):
        return 'host memory introspection unavailable on this platform'
    return (f"free {s['free_phys_gib']:.2f} / {s['total_phys_gib']:.1f} GiB | "
            f"committed {s['committed_gib']:.1f} GiB "
            f"(over physical by {s['commit_over_phys_gib']:.1f}) | "
            f"load {s['memory_load_percent']}%")


def require(min_free_gib: float, label: str = 'job') -> dict:
    """Refuse to start when there is not enough free physical memory.

    Exits rather than raising: a job that cannot safely start has nothing to clean up, and a
    SystemExit reads clearly in a chained shell script.
    """
    s = snapshot()
    if not s.get('available'):
        return s
    if s['free_phys_gib'] < min_free_gib:
        raise SystemExit(
            f'REFUSING TO START {label}: needs {min_free_gib:.1f} GiB free physical memory, '
            f'host has {s["free_phys_gib"]:.2f} GiB. {describe()}. Starting anyway would page, '
            f'and a paging second job is slower than running the two sequentially -- measured '
            f'10x on this host. Wait for the other job to finish.')
    return s


def guard(min_free_gib: float, label: str = 'job') -> None:
    """Stand down mid-flight if the host runs short. Call between units of work."""
    s = snapshot()
    if s.get('available') and s['free_phys_gib'] < min_free_gib:
        raise LowMemoryAbort(
            f'{label} standing down: free physical memory {s["free_phys_gib"]:.2f} GiB is below '
            f'the {min_free_gib:.1f} GiB floor. {describe()}')


if __name__ == '__main__':
    print(describe())
    s = snapshot()
    for k, v in s.items():
        print(f'  {k}: {v}')
