"""Synchronous Windows direct file I/O for pre-sized Adam moment files.

NO_BUFFERING bypasses the Windows data cache; WRITE_THROUGH requests durable
writes. Sizes/offsets and owned NumPy buffer addresses must satisfy the actual
file's logical/physical sector alignment. No cached fallback occurs here.

References:
https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering
https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_storage_info
https://learn.microsoft.com/en-us/windows/win32/api/minwinbase/ne-minwinbase-file_info_by_handle_class
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from functools import lru_cache
from pathlib import Path
import sys

NO_BUFFERING = 0x20000000
WRITE_THROUGH = 0x80000000
NORMAL = 0x80
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
FILE_SHARE_READ = 1
FILE_STORAGE_INFO_CLASS = 16
FILE_ALIGNMENT_INFO_CLASS = 17


class _StorageInfo(ctypes.Structure):
    _fields_ = [(name, wintypes.DWORD) for name in (
        'LogicalBytesPerSector', 'PhysicalBytesPerSectorForAtomicity',
        'PhysicalBytesPerSectorForPerformance',
        'FileSystemEffectivePhysicalBytesPerSectorForAtomicity', 'Flags',
        'ByteOffsetForSectorAlignment', 'ByteOffsetForPartitionAlignment')]


class _AlignmentInfo(ctypes.Structure):
    _fields_ = [('AlignmentRequirement', wintypes.DWORD)]


@lru_cache(maxsize=1)
def _api():
    if sys.platform != 'win32':
        raise OSError('Windows direct I/O requires Windows')
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                               ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    api.CreateFileW.restype = wintypes.HANDLE
    api.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                               ctypes.c_void_p, wintypes.DWORD]
    api.GetFileInformationByHandleEx.restype = wintypes.BOOL
    api.GetFileSizeEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong)]
    api.GetFileSizeEx.restype = wintypes.BOOL
    api.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong,
                                    ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD]
    api.SetFilePointerEx.restype = wintypes.BOOL
    for name in ('ReadFile', 'WriteFile'):
        function = getattr(api, name)
        function.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                             ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        function.restype = wintypes.BOOL
    api.FlushFileBuffers.argtypes = [wintypes.HANDLE]
    api.FlushFileBuffers.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    return api


def _error(operation, path):
    code = ctypes.get_last_error()
    return OSError(code, f'{operation} failed: {ctypes.FormatError(code).strip()}', str(path))


def _open(path: Path, direct: bool):
    flags = NORMAL | (NO_BUFFERING | WRITE_THROUGH if direct else 0)
    handle = _api().CreateFileW(str(path.resolve()), GENERIC_READ | GENERIC_WRITE,
                               FILE_SHARE_READ, None, OPEN_EXISTING, flags, None)
    if handle == ctypes.c_void_p(-1).value:
        raise _error('CreateFileW', path)
    return handle


def _geometry(handle, path: Path) -> dict:
    storage, alignment = _StorageInfo(), _AlignmentInfo()
    for kind, info in ((FILE_STORAGE_INFO_CLASS, storage), (FILE_ALIGNMENT_INFO_CLASS, alignment)):
        if not _api().GetFileInformationByHandleEx(handle, kind, ctypes.byref(info), ctypes.sizeof(info)):
            raise _error('GetFileInformationByHandleEx', path)
    result = {name: int(getattr(storage, name)) for name, _ in storage._fields_}
    result['AlignmentRequirement'] = int(alignment.AlignmentRequirement)
    logical = storage.LogicalBytesPerSector
    physical = [storage.PhysicalBytesPerSectorForAtomicity, storage.PhysicalBytesPerSectorForPerformance,
                storage.FileSystemEffectivePhysicalBytesPerSectorForAtomicity]
    values = [logical, *physical]
    if not logical or any(value and (value & (value - 1)) for value in values):
        raise ValueError(f'unsupported sector geometry: {result}')
    # Windows alignment information commonly uses an alignment mask (e.g. 511).
    # Accept a power-of-two byte value too; reported sector geometry dominates.
    requirement = alignment.AlignmentRequirement
    if requirement == 0 or (requirement & (requirement + 1)) == 0:
        requirement += 1
    elif requirement & (requirement - 1):
        raise ValueError(f'unsupported file alignment requirement: {result}')
    chosen = max(logical, *physical, requirement)
    if chosen > 1024 * 1024 or chosen % logical:
        raise ValueError(f'unsupported direct-I/O alignment: {result}')
    result['io_alignment'] = int(chosen)
    return result


def storage_geometry(path: Path) -> dict:
    """Metadata-only inspection; no file data is read through this cached handle."""
    handle = _open(Path(path), False)
    try:
        return _geometry(handle, Path(path))
    finally:
        if not _api().CloseHandle(handle):
            raise _error('CloseHandle', path)


def aligned_float32(elements: int, alignment: int):
    """Owned, aligned writable array. The view retains its overallocation owner."""
    import numpy as np
    if elements <= 0 or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError('positive element count and power-of-two alignment required')
    owner = np.empty(elements * 4 + alignment - 1, dtype=np.uint8)
    offset = (-owner.ctypes.data) % alignment
    result = owner[offset:offset + elements * 4].view(np.float32)
    if result.ctypes.data % alignment:
        raise AssertionError('aligned allocation failed')
    return result


class DirectFile:
    """Existing, fixed-size binary file; synchronous transfers or fail closed."""
    direct_io = True

    def __init__(self, path: Path, expected_size: int):
        self.path, self.name = Path(path), str(path)
        self._handle = _open(self.path, True)
        self.position = 0
        try:
            self.geometry = _geometry(self._handle, self.path)
            self.alignment = self.geometry['io_alignment']
            actual = ctypes.c_longlong()
            if not _api().GetFileSizeEx(self._handle, ctypes.byref(actual)):
                raise _error('GetFileSizeEx', self.path)
            if expected_size <= 0 or expected_size % self.alignment or actual.value != expected_size:
                raise ValueError(f'direct file size/alignment mismatch: {self.path}')
            self.size = expected_size
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, value, traceback):
        self.close()

    def _require_open(self):
        if self._handle is None:
            raise ValueError('direct file is closed')

    def close(self):
        if self._handle is not None:
            handle, self._handle = self._handle, None
            if not _api().CloseHandle(handle):
                raise _error('CloseHandle', self.path)

    def seek(self, offset: int):
        self._require_open()
        if type(offset) is not int or not 0 <= offset <= self.size or offset % self.alignment:
            raise ValueError('direct file offset must be aligned and in range')
        actual = ctypes.c_longlong()
        if not _api().SetFilePointerEx(self._handle, offset, ctypes.byref(actual), 0):
            raise _error('SetFilePointerEx', self.path)
        if actual.value != offset:
            raise OSError('direct file seek did not reach requested offset')
        self.position = offset
        return offset

    def _perform(self, write, address, size, transferred):
        function = _api().WriteFile if write else _api().ReadFile
        return function(self._handle, address, size, ctypes.byref(transferred), None)

    def _transfer(self, buffer, write):
        self._require_open()
        view = memoryview(buffer)
        if view.readonly or not view.c_contiguous:
            raise ValueError('direct I/O requires a writable contiguous owned buffer')
        view = view.cast('B')
        size = len(view)
        if (not size or size >= 2 ** 32 or size % self.alignment
                or self.position + size > self.size):
            raise ValueError('direct I/O transfer size must be aligned and within the file')
        address = ctypes.addressof(ctypes.c_char.from_buffer(view))
        if address % self.alignment:
            raise ValueError('direct I/O buffer address is not sector aligned')
        transferred = wintypes.DWORD()
        if not self._perform(write, address, size, transferred):
            raise _error('WriteFile' if write else 'ReadFile', self.path)
        if transferred.value != size:
            raise OSError(f'short direct {"write" if write else "read"}: '
                          f'{transferred.value}/{size} bytes in {self.path}; no retry/fallback')
        self.position += size
        return size

    def readinto(self, target):
        return self._transfer(target, False)

    def write(self, source):
        return self._transfer(source, True)

    def flush(self):
        self._require_open()
        if not _api().FlushFileBuffers(self._handle):
            raise _error('FlushFileBuffers', self.path)
