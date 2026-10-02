"""Read-only Windows resource observations and native permit-bound barriers.

Samples are observations at declared barriers, not a continuous free-memory
claim. OS peak working set and the CUDA allocator peak are recorded separately.
No sampler, input flag or reduction here grants native execution or PASS.
"""
from __future__ import annotations

import csv
import copy
import ctypes
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

from . import ce_native_authority as authority

SCHEMA = 'CE8-RESOURCE-OBSERVATION-1'
SPEC = 'CE8-RESOURCE-PROBE-SPEC-1'
GIB = 1024 ** 3
CHUNK = 1024 ** 2
MAX_OUTPUT = 4096
UUID_PATTERN = r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
require = authority.require


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def parse_gpu(raw, expected_uuid):
    require(type(raw) is str and len(raw.encode('utf8')) <= MAX_OUTPUT, 'bounded GPU query output')
    lines = list(csv.reader(raw.splitlines()))
    require(len(lines) == 1 and len(lines[0]) == 6, 'one exact GPU query row')
    index, identity, total, free, used, reserved = [cell.strip() for cell in lines[0]]
    require(index == '0' and identity == expected_uuid
            and re.fullmatch(UUID_PATTERN, identity) is not None,
            'exact declared physical GPU0 identity')
    require(all(re.fullmatch(r'[0-9]+', value) for value in (total, free, used, reserved)), 'integer GPU MiB observations')
    total, free, used, reserved = [int(value) * 1024 ** 2 for value in (total, free, used, reserved)]
    require(total > 0 and all(0 <= value <= total for value in (free, used, reserved))
            and abs(total - free - used - reserved) <= 2 * 1024 ** 2, 'coherent GPU memory observations')
    return {'gpu_uuid': identity, 'gpu_total_bytes': total, 'gpu_free_bytes': free,
            'gpu_used_bytes': used, 'gpu_driver_reserved_bytes': reserved,
            'gpu_memory_units': 'nvidia-smi integer MiB, <=2 MiB total rounding allowance'}


def validate_spec(spec):
    require(type(spec) is dict and spec.get('schema') == SPEC, 'explicit resource probe specification')
    tool = spec.get('nvidia_smi_ref')
    require(type(tool) is dict and set(tool) == {'path', 'sha256', 'bytes'}
            and type(tool['path']) is str and Path(tool['path']).is_absolute()
            and authority._hex(tool['sha256']) and type(tool['bytes']) is int and 0 < tool['bytes'] <= 64 * CHUNK,
            'bounded pinned nvidia-smi binary reference')
    require(type(spec.get('output_root')) is str and Path(spec['output_root']).is_absolute()
            and type(spec.get('gpu_uuid')) is str
            and re.fullmatch(UUID_PATTERN, spec['gpu_uuid']) is not None,
            'explicit output volume and physical GPU identity')
    require(spec.get('cuda_visible_devices') in (None, '0')
            and type(spec.get('query_timeout_seconds')) in (int, float)
            and 0 < spec['query_timeout_seconds'] <= 10, 'bounded query and physical/logical GPU0 mapping')
    bounds = spec.get('probe_bounds')
    require(type(bounds) is dict and set(bounds) == {'read_bytes_upper', 'identity_hash_attempts_upper',
            'identity_hash_bytes_upper', 'wall_s_upper'}
            and all(type(bounds[key]) is int and bounds[key] >= 0 for key in bounds if key != 'wall_s_upper')
            and _number(bounds['wall_s_upper']) and bounds['wall_s_upper'] > 0
            and bounds['read_bytes_upper'] >= tool['bytes'] + 1
            and bounds['identity_hash_attempts_upper'] >= 1
            and bounds['identity_hash_bytes_upper'] >= tool['bytes'] + 1,
            'resource binary verification must be prepaid')
    return spec


class _Memory(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
        ('total', ctypes.c_ulonglong), ('available', ctypes.c_ulonglong),
        ('page_total', ctypes.c_ulonglong), ('page_available', ctypes.c_ulonglong),
        ('virtual_total', ctypes.c_ulonglong), ('virtual_available', ctypes.c_ulonglong),
        ('extended', ctypes.c_ulonglong)]


class _ProcessMemory(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('page_faults', ctypes.c_ulong),
        ('peak_working_set', ctypes.c_size_t), ('working_set', ctypes.c_size_t),
        ('peak_paged', ctypes.c_size_t), ('paged', ctypes.c_size_t),
        ('peak_nonpaged', ctypes.c_size_t), ('nonpaged', ctypes.c_size_t),
        ('pagefile', ctypes.c_size_t), ('peak_pagefile', ctypes.c_size_t)]


def windows_memory():
    require(os.name == 'nt', 'actual Windows resource API required')
    memory = _Memory(); memory.length = ctypes.sizeof(memory)
    require(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)), 'host memory observation failed')
    process = _ProcessMemory(); process.cb = ctypes.sizeof(process)
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = ctypes.c_void_p
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = (ctypes.c_void_p, ctypes.POINTER(_ProcessMemory), ctypes.c_ulong)
    query.restype = ctypes.c_int
    require(query(current(), ctypes.byref(process), process.cb), 'current process memory observation failed')
    require(memory.available <= memory.total and process.working_set <= process.peak_working_set,
            'coherent actual Windows memory values')
    return {'host_total_bytes': int(memory.total), 'host_free_bytes': int(memory.available),
            'worker_id': os.getpid(), 'worker_working_set_bytes': int(process.working_set),
            'worker_peak_working_set_bytes': int(process.peak_working_set),
            'worker_memory_primitive': 'GetProcessMemoryInfo WorkingSetSize/PeakWorkingSetSize'}


def _failed(operation, identifier, start, error, **fields):
    try:
        operation('resource_probe', 'failed', operation_id=identifier,
                  wall_s=time.monotonic() - start, error_type=type(error).__name__, **fields)
    except BaseException as accounting:
        error.accounting_failure = {'type': type(accounting).__name__, 'message': str(accounting)[:1000]}
        raise error from accounting


def readonly_sample(spec, operation, *, barrier=None):
    """Current-process-only OS observation; caller must retain prepayments.

    This also supports an explicitly non-native metadata probe. It imports no
    framework, initializes no CUDA and starts only the pinned query utility.
    """
    validate_spec(spec); require(callable(operation), 'resource pre-reservation observer required')
    require(barrier is None or type(barrier) is str and 0 < len(barrier) <= 128, 'bounded explicit resource barrier label')
    identifier, start = uuid.uuid4().hex, time.monotonic()
    operation('resource_probe', 'started', operation_id=identifier, barrier=barrier, **spec['probe_bounds'])
    try:
        require(os.name == 'nt' and os.environ.get('CUDA_VISIBLE_DEVICES') == spec['cuda_visible_devices'],
                'Windows and declared GPU visibility required')
        reference = spec['nvidia_smi_ref']; path = Path(reference['path'])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == reference['bytes'], 'pinned GPU utility path/size')
        hashed, read = hashlib.sha256(), 0
        with path.open('rb') as stream:
            while block := stream.read(min(CHUNK, reference['bytes'] + 1 - read)):
                read += len(block); require(read <= reference['bytes'], 'GPU utility grew'); hashed.update(block)
        require(read == reference['bytes'] and hashed.hexdigest() == reference['sha256'], 'GPU utility raw hash drift')
        result = subprocess.run([str(path), '--id=0', '--query-gpu=index,uuid,memory.total,memory.free,memory.used,memory.reserved',
            '--format=csv,noheader,nounits'], capture_output=True, text=True,
            timeout=spec['query_timeout_seconds'], shell=False, creationflags=subprocess.CREATE_NO_WINDOW)
        require(result.returncode == 0 and len(result.stderr.encode('utf8')) <= MAX_OUTPUT, 'GPU query failed')
        sample = windows_memory()
        sample.update(parse_gpu(result.stdout, spec['gpu_uuid']))
        root = Path(spec['output_root'])
        while not root.exists():
            parent = root.parent; require(parent != root, 'output volume unavailable'); root = parent
        sample.update(schema=SCHEMA, perf=time.monotonic(), barrier=barrier, resource_probe_operation=identifier,
            disk_free_bytes=shutil.disk_usage(root).free,
            output_volume_path=str(root.resolve()), native_qualified=False,
            classification='actual read-only current-process resource sample; no model/native fit proof')
        elapsed = time.monotonic() - start
        require(elapsed <= spec['probe_bounds']['wall_s_upper'], 'resource probe wall upper exceeded')
    except BaseException as error:
        _failed(operation, identifier, start, error); raise
    operation('resource_probe', 'returned', operation_id=identifier, barrier=barrier, wall_s=elapsed,
              bytes_read=read, bytes_written=0, identity_hash_attempts_actual=1, identity_hash_bytes_actual=read,
              io_observation_scope='explicit Python file IO; external query utility OS IO is unmeasured',
              resource_observation=sample)
    return sample


def enforce(sample, limits, *, launch=False):
    """Exact barriers; a failed gate is never weakened or repaired here."""
    require(type(sample) is dict and sample.get('schema') == SCHEMA, 'actual resource sample shape')
    require(type(limits) is dict and set(limits) == {'launch_host_gib', 'ongoing_host_gib', 'max_worker_gib',
            'gpu_free_gib', 'allocator_fraction', 'reserved_fraction_limit', 'disk_floor_bytes',
            'launch_disk_bytes', 'active_storage_reserve_bytes'}, 'all resource barriers explicitly declared')
    require(all(_number(limits[key]) for key in ('launch_host_gib', 'ongoing_host_gib', 'max_worker_gib',
            'gpu_free_gib', 'allocator_fraction', 'reserved_fraction_limit'))
            and all(type(sample.get(key)) is int and sample[key] >= 0 for key in
                ('host_free_bytes', 'worker_peak_working_set_bytes', 'gpu_free_bytes', 'disk_free_bytes')),
            'strict finite resource barrier values and integer OS bytes')
    require(limits['launch_host_gib'] >= 5.5 and limits['ongoing_host_gib'] >= 1
            and 0 < limits['max_worker_gib'] <= 4.5 and limits['gpu_free_gib'] >= .5
            and limits['allocator_fraction'] == .78 and limits['reserved_fraction_limit'] == .95
            and type(limits['disk_floor_bytes']) is int and limits['disk_floor_bytes'] >= 14 * GIB
            and type(limits['launch_disk_bytes']) is int and limits['launch_disk_bytes'] >= 32 * GIB
            and type(limits['active_storage_reserve_bytes']) is int and limits['active_storage_reserve_bytes'] >= 0,
            'registered resource barriers cannot be lowered')
    threshold = limits['launch_host_gib'] if launch else limits['ongoing_host_gib']
    require(sample['host_free_bytes'] >= threshold * GIB
            and sample['worker_peak_working_set_bytes'] <= limits['max_worker_gib'] * GIB,
            'host/worker peak memory barrier')
    free_lower = gpu_free_lower(sample)
    require(free_lower >= limits['gpu_free_gib'] * GIB, 'GPU free-memory barrier')
    disk = limits['disk_floor_bytes'] + limits['active_storage_reserve_bytes']
    if launch: disk = max(limits['launch_disk_bytes'], disk)
    require(sample['disk_free_bytes'] >= disk, 'disk floor and active reservation barrier')
    if 'cuda_peak_reserved_bytes' in sample:
        require(type(sample['cuda_peak_reserved_bytes']) is int and sample['cuda_peak_reserved_bytes'] >= 0
                and type(sample.get('cuda_total_bytes')) is int and sample['cuda_total_bytes'] > 0,
                'integer actual CUDA allocator bytes')
        require(sample['cuda_peak_reserved_bytes'] <= .95 * .78 * sample['cuda_total_bytes'], 'CUDA allocator peak ceiling')
    return sample


def gpu_free_lower(sample):
    if 'cuda_free_bytes' in sample:
        require(type(sample['cuda_free_bytes']) is int and sample['cuda_free_bytes'] >= 0, 'integer exact CUDA free bytes')
        return sample['cuda_free_bytes']
    # One MiB of display uncertainty keeps a rounded512 MiB from admitting
    # less than the exact .5 GiB barrier.
    require(type(sample.get('gpu_free_bytes')) is int and sample['gpu_free_bytes'] >= 0, 'integer displayed GPU free bytes')
    return max(0, sample['gpu_free_bytes'] - 1024 ** 2)


class NativeResourceRecorder:
    """Permit-bound actual sampling; no native authority or service registry."""
    def __init__(self, context, permit, operation, *, samples=None):
        authority.require_execution_permit(permit, context, 'resource_io_measurement', operation=operation)
        spec = context['bindings']['scope'].get('qualifier_resources')
        validate_spec(spec)
        self.context, self.permit, self.operation, self.spec = context, permit, operation, spec
        require(samples is None or type(samples) is list, 'actual shared resource sample list')
        self.framework = None; self.samples = [] if samples is None else samples

    @classmethod
    def fixture_recorder(cls, spec, operation, samples=None):
        validate_spec(spec); require(callable(operation) and (samples is None or type(samples) is list), 'explicit pure recorder fixture')
        value = object.__new__(_FixtureResourceRecorder)
        value.context, value.permit, value.operation, value.spec = None, None, operation, copy.deepcopy(spec)
        value.framework, value.samples = None, [] if samples is None else samples
        return value

    def _admit(self):
        authority.require_execution_permit(self.permit, self.context, 'resource_io_measurement', operation=self.operation)
        require(self.spec == self.context['bindings']['scope']['qualifier_resources'], 'frozen resource specification changed')

    def retain(self, sample):
        require(type(sample) is dict and type(sample.get('barrier')) is str and sample.get('resource_probe_operation'),
                'actual labeled returned resource probe required')
        evidence = getattr(self.operation, 'resource_evidence', None)
        if callable(evidence):
            sample['operation_links'] = [evidence(sample[key]) for key in ('resource_probe_operation', 'resource_cuda_operation') if key in sample]
        self.samples.append(copy.deepcopy(sample))
        return copy.deepcopy(sample)

    def sample(self, stage):
        self._admit()
        require(type(stage) is str and 0 < len(stage) <= 128, 'explicit resource barrier label')
        sample = readonly_sample(self.spec, self.operation, barrier=stage)
        sample['stage'] = stage
        framework = self.framework if self.framework is not None else sys.modules.get('torch')
        require(framework is None or framework is sys.modules.get('torch'), 'actual already pinned framework module required')
        if framework is not None and framework.cuda.is_initialized():
            require(framework is sys.modules.get('torch'), 'actual already pinned framework module required')
            bounds = self.spec.get('cuda_probe_bounds')
            require(type(bounds) is dict and set(bounds) == {'wall_s_upper'}
                    and _number(bounds['wall_s_upper']) and bounds['wall_s_upper'] > 0, 'explicit CUDA observation wall reservation')
            identifier, start = uuid.uuid4().hex, time.monotonic()
            self.operation('resource_cuda_probe', 'started', operation_id=identifier, barrier=stage, **bounds)
            try:
                free, total = framework.cuda.mem_get_info(0)
                require(framework.cuda.device_count() == 1, 'qualified single-visible-GPU runtime required')
                sample.update(cuda_free_bytes=free, cuda_total_bytes=total,
                    cuda_reserved_bytes=framework.cuda.memory_reserved(0),
                    cuda_peak_reserved_bytes=framework.cuda.max_memory_reserved(0), resource_cuda_operation=identifier)
                elapsed = time.monotonic() - start
                require(elapsed <= bounds['wall_s_upper'], 'CUDA observation wall reservation exceeded')
            except BaseException as error:
                try: self.operation('resource_cuda_probe', 'failed', operation_id=identifier, error_type=type(error).__name__)
                except BaseException as accounting:
                    error.accounting_failure = {'type': type(accounting).__name__, 'message': str(accounting)[:1000]}
                    raise error from accounting
                raise
            self.operation('resource_cuda_probe', 'returned', operation_id=identifier, barrier=stage, wall_s=elapsed,
                           bytes_read=0, bytes_written=0, identity_hash_attempts_actual=0,
                           identity_hash_bytes_actual=0, io_observation_scope='explicit Python file IO and hashes only',
                           resource_observation=sample)
        return self.retain(sample)

    def observations(self):
        self._admit()
        require(self.samples and any('cuda_peak_reserved_bytes' in sample for sample in self.samples), 'actual CUDA-phase resource samples required')
        # Callback IO/timing/worker verification and raw publication come from
        # the surrounding service. Samples alone cannot satisfy native PASS.
        return {'samples': copy.deepcopy(self.samples), 'peak_worker_working_set_gib': max(s['worker_peak_working_set_bytes'] for s in self.samples) / GIB,
            'peak_reserved_gib': max(s.get('cuda_peak_reserved_bytes', 0) for s in self.samples) / GIB,
            'min_sampled_gpu_free_gib': min(gpu_free_lower(s) for s in self.samples) / GIB,
            'gpu_observation_scope': 'declared barrier-sampled lower bounds; no continuous minimum assertion',
            'sample_operation_links': [copy.deepcopy(s.get('operation_links', [])) for s in self.samples],
            'native_qualified': False, 'execution_verified': False}


class _FixtureResourceRecorder(NativeResourceRecorder):
    """Pure controls only; strict native composition rejects this subclass."""
    def _admit(self):
        require(self.context is None and self.permit is None, 'explicit non-native resource recorder fixture')
