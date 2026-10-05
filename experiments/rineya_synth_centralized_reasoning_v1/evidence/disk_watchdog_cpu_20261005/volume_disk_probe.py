"""CPU-only staged volume-bound disk probe; no qualification or launch authority.

Use the containing mount point, not a guessed drive-letter anchor. Every sample
performs a new GetDiskFreeSpaceExW call and keeps its third output, exactly the
existing shutil.disk_usage.free policy. Caller quota availability is separate
evidence. Output ancestry/mount/GUID binding is checked on every sample and its
latency remains inside the complete sample freshness window.
"""
from __future__ import annotations
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import time


def require(ok,message):
    if not ok:raise ValueError(message)


class WindowsAPI:
    def __init__(self):
        require(os.name=='nt','Windows read-only disk API required')
        self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        u64=ctypes.POINTER(ctypes.c_ulonglong)
        signatures={
            'GetVolumePathNameW':([wintypes.LPCWSTR,wintypes.LPWSTR,wintypes.DWORD],wintypes.BOOL),
            'GetVolumeNameForVolumeMountPointW':([wintypes.LPCWSTR,wintypes.LPWSTR,wintypes.DWORD],wintypes.BOOL),
            'GetDriveTypeW':([wintypes.LPCWSTR],wintypes.UINT),
            'GetDiskFreeSpaceExW':([wintypes.LPCWSTR,u64,u64,u64],wintypes.BOOL)}
        for name,(args,result) in signatures.items():
            getattr(self.api,name).argtypes=args;getattr(self.api,name).restype=result
    def volume_path(self,path):
        buffer=ctypes.create_unicode_buffer(32768)
        if not self.api.GetVolumePathNameW(str(path),buffer,len(buffer)):raise ctypes.WinError(ctypes.get_last_error())
        return buffer.value
    def volume_id(self,root):
        buffer=ctypes.create_unicode_buffer(1024)
        if not self.api.GetVolumeNameForVolumeMountPointW(str(root),buffer,len(buffer)):raise ctypes.WinError(ctypes.get_last_error())
        return buffer.value
    def drive_type(self,root):return int(self.api.GetDriveTypeW(str(root)))
    def space(self,path):
        available,total,free=(ctypes.c_ulonglong() for _ in range(3))
        if not self.api.GetDiskFreeSpaceExW(str(path),ctypes.byref(available),ctypes.byref(total),ctypes.byref(free)):
            raise ctypes.WinError(ctypes.get_last_error())
        return {'caller_available_bytes':available.value,'caller_total_bytes':total.value,'total_free_bytes':free.value}


class VolumeBoundDiskProbe:
    def __init__(self,path,allowed_root,*,api=None,clock=time.perf_counter):
        self.path=Path(path).absolute();self.allowed_root=Path(allowed_root).absolute()
        self.api=api or WindowsAPI();self.clock=clock
        self._check_path()
        self.root=self.api.volume_path(self.path)
        require(self.root.endswith('\\') and self.api.drive_type(self.root)==3,'fixed local containing volume required')
        self.volume_id=self.api.volume_id(self.root)
        require(bool(self.volume_id) and self.api.volume_path(self.root)==self.root,'exact containing mount root')
        self.verify_binding()
    def _check_path(self):
        require(self.allowed_root.is_dir() and self.path.is_dir(),'existing output and allowed root')
        require(self.allowed_root.resolve(strict=True)==self.allowed_root
                and self.path.resolve(strict=True)==self.path,'resolved path binding; no redirected root/output')
        require(self.path==self.allowed_root or self.allowed_root in self.path.parents,'output path cannot escape allowed root')
        for entry in (self.path,*self.path.parents):
            require(not (getattr(entry.stat(),'st_file_attributes',0)&0x400),'no reparse point in output ancestry')
    def verify_binding(self):
        self._check_path()
        require(self.api.volume_path(self.path)==self.root and self.api.volume_id(self.root)==self.volume_id,'output/root volume binding drift')
    def sample(self):
        started=self.clock()
        self.verify_binding()
        values=self.api.space(self.root)
        require(set(values)=={'caller_available_bytes','caller_total_bytes','total_free_bytes'}
                and all(type(v) is int and v>=0 for v in values.values()),'complete fresh unsigned64 disk values')
        return {'sample_started':started,'sample_completed':self.clock(),
                'disk_free_bytes':values['total_free_bytes'],**values,'volume_id':self.volume_id,
                'query_path':self.root,'free_policy':'GetDiskFreeSpaceExW lpTotalNumberOfFreeBytes (existing shutil.free)',
                'native_qualified':False}
