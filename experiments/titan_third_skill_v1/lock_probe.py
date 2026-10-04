"""Read-only Windows Restart Manager query for a path's user-mode holders."""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path


class FILETIME(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


class UNIQUE_PROCESS(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint32), ("started", FILETIME)]


class PROCESS_INFO(ctypes.Structure):
    _fields_ = [("process", UNIQUE_PROCESS), ("app", ctypes.c_wchar * 256),
                ("service", ctypes.c_wchar * 64), ("app_type", ctypes.c_uint32),
                ("status", ctypes.c_uint32), ("session", ctypes.c_uint32),
                ("restartable", ctypes.c_int)]


def main() -> int:
    target = Path(sys.argv[1]).resolve(strict=True)
    dll = ctypes.WinDLL("rstrtmgr")
    session = ctypes.c_uint32()
    key = ctypes.create_unicode_buffer(33)
    rc = dll.RmStartSession(ctypes.byref(session), 0, key)
    if rc:
        raise SystemExit(f"RmStartSession rc={rc}")
    try:
        files = (ctypes.c_wchar_p * 1)(str(target))
        rc = dll.RmRegisterResources(session, 1, files, 0, None, 0, None)
        if rc:
            raise SystemExit(f"RmRegisterResources rc={rc}")
        needed = ctypes.c_uint32()
        count = ctypes.c_uint32()
        reasons = ctypes.c_uint32()
        rc = dll.RmGetList(session, ctypes.byref(needed), ctypes.byref(count), None, ctypes.byref(reasons))
        if rc not in (0, 234):
            raise SystemExit(f"RmGetList size rc={rc}")
        print(f"needed={needed.value} reboot_reasons={reasons.value}")
        if needed.value:
            items = (PROCESS_INFO * needed.value)()
            count.value = needed.value
            rc = dll.RmGetList(session, ctypes.byref(needed), ctypes.byref(count), items, ctypes.byref(reasons))
            if rc:
                raise SystemExit(f"RmGetList data rc={rc}")
            for item in items[:count.value]:
                print(f"pid={item.process.pid} app={item.app!r} service={item.service!r} type={item.app_type}")
    finally:
        dll.RmEndSession(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
