"""Pure resource parsing/barrier/admission controls; no actual query or ML."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_resource_evidence as r
from experiments.skill_acquisition_native_v1 import ce_native_authority as a

GPU = 'GPU-12345678-1234-1234-1234-123456789abc'
def spec():
    return {'schema': r.SPEC, 'nvidia_smi_ref': {'path': str(Path(tempfile.gettempdir()).resolve()/'nvidia-smi.exe'),
        'sha256': 'a'*64, 'bytes': 100}, 'output_root': tempfile.gettempdir(), 'gpu_uuid': GPU,
        'cuda_visible_devices': None, 'query_timeout_seconds': 5,
        'probe_bounds': {'read_bytes_upper': 101, 'identity_hash_attempts_upper': 1,
            'identity_hash_bytes_upper': 101, 'wall_s_upper': 10}}
def limits():
    return {'launch_host_gib': 5.5, 'ongoing_host_gib': 1, 'max_worker_gib': 4.5,
        'gpu_free_gib': .5, 'allocator_fraction': .78, 'reserved_fraction_limit': .95,
        'disk_floor_bytes': 14*r.GIB, 'launch_disk_bytes': 32*r.GIB, 'active_storage_reserve_bytes': 16*r.GIB}
def sample():
    return {'schema': r.SCHEMA, 'host_free_bytes': int(5.5*r.GIB), 'worker_peak_working_set_bytes': int(4.5*r.GIB),
        'gpu_free_bytes': r.GIB//2 + 1024**2, 'disk_free_bytes': 32*r.GIB}
class Controls(unittest.TestCase):
    def test_exact_gpu_identity_and_integer_memory(self):
        value = r.parse_gpu('0, '+GPU+', 8192, 4096, 4096, 0\n', GPU)
        self.assertEqual(value['gpu_total_bytes'], 8*r.GIB)
        actual_format = r.parse_gpu('0, '+GPU+', 8188, 6917, 1032, 240\n', GPU)
        self.assertEqual(actual_format['gpu_driver_reserved_bytes'], 240*1024**2)
        for raw in ('0, '+GPU+', 8192, NaN, 4096', '0, '+GPU+', 8192, 4096.0, 4096',
                    '1, '+GPU+', 8192, 4096, 4096', '0, '+GPU+', 8192, 8193, 0',
                    '0, '+GPU+', 8192, 1, 1', '0, '+GPU+', 8192, 4096, 4096\n0, '+GPU+', 1, 0, 1'):
            with self.subTest(raw=raw), self.assertRaises(ValueError): r.parse_gpu(raw, GPU)
        with self.assertRaises(ValueError): r.parse_gpu('0, '+GPU+', 8192, 4096, 4096', 'GPU-other')

    def test_binary_read_and_hash_bounds_have_no_defaults(self):
        r.validate_spec(spec())
        for mutate in (lambda s: s['probe_bounds'].update(read_bytes_upper=100),
                       lambda s: s['probe_bounds'].update(identity_hash_bytes_upper=100),
                       lambda s: s['probe_bounds'].update(identity_hash_attempts_upper=True),
                       lambda s: s.update(query_timeout_seconds=11), lambda s: s.update(cuda_visible_devices='1')):
            value=spec(); mutate(value)
            with self.assertRaises(ValueError): r.validate_spec(value)

    def test_pre_reservation_denial_occurs_before_paths_or_process(self):
        def denied(*args, **kwargs): raise RuntimeError('denied before query')
        with (patch.object(Path, 'is_file', side_effect=AssertionError('path before reservation')),
              patch.object(r.subprocess, 'run', side_effect=AssertionError('process before reservation'))):
            with self.assertRaisesRegex(RuntimeError, 'denied'): r.readonly_sample(spec(), denied)

    def test_native_constructor_refuses_before_context_or_objects(self):
        class Trap:
            def __getattribute__(self, name): raise AssertionError('unadmitted object access')
        for permit in (None, True, {'approved': True}, object.__new__(a.NativeExecutionPermit)):
            with self.assertRaises(ValueError): r.NativeResourceRecorder(Trap(), permit, Trap())
        self.assertNotIn('torch', sys.modules)

    def test_exact_memory_disk_and_allocator_barriers(self):
        r.enforce(sample(), limits(), launch=True)
        for key in ('host_free_bytes','worker_peak_working_set_bytes','gpu_free_bytes','disk_free_bytes'):
            value=sample(); value[key] += 1 if key=='worker_peak_working_set_bytes' else -1
            with self.subTest(key=key), self.assertRaises(ValueError): r.enforce(value, limits(), launch=True)
        value=sample(); value.update(cuda_total_bytes=8*r.GIB, cuda_peak_reserved_bytes=int(.95*.78*8*r.GIB)+1)
        with self.assertRaises(ValueError): r.enforce(value, limits())
        value=sample(); value['host_free_bytes']=r.GIB
        r.enforce(value, limits())
        with self.assertRaises(ValueError): r.enforce(value, limits(), launch=True)
        value=sample(); value['gpu_free_bytes']=r.GIB//2
        with self.assertRaisesRegex(ValueError, 'GPU free'): r.enforce(value, limits())
        value['cuda_free_bytes']=r.GIB//2
        r.enforce(value, limits())

    def test_lowered_or_nonfinite_barriers_cannot_pass(self):
        for key, bad in (('launch_host_gib',5.4),('ongoing_host_gib',.99),('max_worker_gib',4.6),
                         ('gpu_free_gib',.49),('allocator_fraction',.8),('reserved_fraction_limit',.99),
                         ('max_worker_gib',True),('max_worker_gib',float('nan'))):
            value=limits(); value[key]=bad
            with self.subTest(key=key,bad=bad), self.assertRaises(ValueError): r.enforce(sample(), value)

    def test_launch_disk_covers_larger_active_reservation(self):
        value=limits(); value['active_storage_reserve_bytes']=20*r.GIB
        with self.assertRaisesRegex(ValueError, 'disk floor'): r.enforce(sample(), value, launch=True)
        observed=sample(); observed['disk_free_bytes']=34*r.GIB
        r.enforce(observed, value, launch=True)

    def test_gpu_reduction_uses_exact_or_conservative_same_barrier_value(self):
        displayed=sample()
        self.assertEqual(r.gpu_free_lower(displayed), r.GIB//2)
        exact=dict(displayed,cuda_free_bytes=r.GIB//4)
        self.assertEqual(r.gpu_free_lower(exact),r.GIB//4)
        with self.assertRaisesRegex(ValueError, 'GPU free'): r.enforce(exact,limits())

if __name__=='__main__': unittest.main()
