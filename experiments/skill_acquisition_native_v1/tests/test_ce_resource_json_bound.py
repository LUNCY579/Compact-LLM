"""Large resource-only retained evidence controls; no native sampling or ML."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_qualification_artifacts as raw
from experiments.skill_acquisition_native_v1 import ce_qualification_accounting as accounting
from experiments.skill_acquisition_native_v1 import ce_resource_qualification as resource
from experiments.skill_acquisition_native_v1 import ce_execution_verifier as verifier
from . import test_ce_resource_qualification as controls


class ResourceJSONBounds(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='resource-json-bound-pure-')
        self.addCleanup(temporary.cleanup); self.root = Path(temporary.name)
        self.args = controls.isolated_writer(self.root)
        self.spec = self.args[0]['bindings']['scope']['resource_measurement']
        self.spec.update(json_bytes_upper=16 * raw.CHUNK, reducer_work_bytes_upper=512 * raw.CHUNK)

    def test_legacy64_mib_work_refuses_before_resource_publication(self):
        self.spec['reducer_work_bytes_upper']=64 * raw.CHUNK
        with self.assertRaisesRegex(ValueError,'copy/reduction work'):
            self.artifact()
        self.assertEqual(list(self.root.glob('resource.*.json')), [])

    def reader(self):
        self.events = []
        bounds = dict(operation_attempts_upper=1000, io_read_bytes_upper=64 * raw.CHUNK,
            identity_hash_attempts_upper=10000, identity_hash_bytes_upper=128 * raw.CHUNK,
            reducer_work_bytes_upper=256 * raw.CHUNK)
        return raw.RawEvidenceReader(lambda kind, phase, **fields: self.events.append((kind, phase, fields)),
                                     bounds, self.root)

    def artifact(self):
        recorder = self.args[0]['native_services']['resource_recorder']
        original = controls.samples(); sample = original[1]
        recorder.samples = original + [dict(sample, perf=index + 3) for index in range(17000)]
        with patch.object(controls.a, 'require_execution_permit', return_value=None):
            result = resource.resource_adapter(*self.args)
        return result['raw_artifacts'][0]

    def test_17002_samples_retained_over_4_mib_resource_reads_exact_bytes(self):
        reference = self.artifact()
        self.assertGreater(reference['bytes'], raw.MAX_JSON)
        self.assertLessEqual(reference['bytes'], self.spec['json_bytes_upper'])
        reader = self.reader()
        payload = reader._envelope(reference, self.args[-1], 'resource_io', 'resource_io',
                                   json_bytes_upper=self.spec['json_bytes_upper'])
        self.assertEqual(len(payload['resource_samples']), 17002)
        self.assertEqual(payload['resource_samples'], self.args[0]['native_services']['resource_recorder'].samples)
        self.assertEqual(payload['raw_operations'], self.args[4].operations)
        reads = [fields for kind, phase, fields in self.events
                 if kind == 'qualification_artifact_readback' and phase == 'returned']
        self.assertEqual(reads[0]['bytes_read'], reference['bytes'])
        self.assertEqual(reads[0]['identity_hash_bytes_actual'], reference['bytes'])
        self.assertEqual(payload['json_bytes_upper'], self.spec['json_bytes_upper'])
        started = self.args[3].operation.events[0][2]
        for key in ('read_bytes_upper', 'write_bytes_upper', 'identity_hash_attempts_upper', 'identity_hash_bytes_upper'):
            self.assertEqual(started[key], 0)
        self.assertEqual(started['reducer_work_bytes_upper'], self.spec['reducer_work_bytes_upper'])
        maximum = dict(read_bytes_upper=0, write_bytes_upper=0, identity_hash_attempts_upper=10000,
            identity_hash_bytes_upper=128 * 1024 ** 3, wall_s_upper=60, reducer_work_bytes_upper=512 * 1024 ** 3)
        reservation = accounting.reservation_profile(dict(operation_profile_rule=accounting.EXPLICIT_RESERVATION_RULE,
            explicit_reservation_kinds=list(accounting.EXPLICIT_RESERVATION_KINDS),
            operation_profiles={'qualification_reducer_overhead': maximum}), 'qualification_reducer_overhead', started)
        self.assertEqual(reservation['identity_hash_bytes_upper'], 0)
        self.assertEqual(reservation['wall_s_upper'], 60)

    def test_same_large_reference_cannot_expand_numeric_or_default_json_bound(self):
        reference = self.artifact(); reader = self.reader()
        with patch.object(Path, 'open', side_effect=AssertionError('oversized numeric IO')):
            for kind, role in (('masks', 'masks'), ('likelihood', 'likelihood'), ('gradient', 'gradient_manifest')):
                with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'raw JSON byte ceiling'):
                    reader._envelope(reference, self.args[-1], kind, role)
                with self.subTest(override=kind), self.assertRaisesRegex(ValueError, 'only for resource'):
                    reader._envelope(reference, self.args[-1], kind, role, json_bytes_upper=raw.MAX_RESOURCE_JSON)
            with self.assertRaisesRegex(ValueError, 'bounded numeric JSON'):
                with reader._numeric_budget(reference, 'mask_reduction', hash_attempts=1, work_multiplier=32):
                    self.fail('numeric budget must refuse')
            with self.assertRaisesRegex(ValueError, 'raw JSON byte ceiling'):
                reader._envelope(reference, self.args[-1], 'resource_io', 'resource_io')
        self.assertEqual(self.events, [])

    def test_bad_resource_caps_roles_and_stages_refuse_before_readback(self):
        reader = self.reader(); reference = dict(path=str(self.root / 'not-read.json'), sha256='a' * 64,
            bytes=raw.MAX_JSON + 1, role='resource_io', binding_sha256=raw.digest(self.args[-1]))
        with patch.object(Path, 'open', side_effect=AssertionError('unbounded resource IO')):
            for ceiling in (0, -1, True, float('inf'), raw.MAX_RESOURCE_JSON + 1):
                with self.subTest(ceiling=ceiling), self.assertRaisesRegex(ValueError, 'bounded resource JSON'):
                    reader._envelope(reference, self.args[-1], 'resource_io', 'resource_io', json_bytes_upper=ceiling)
            for kind, role, binding in (('resource_io', 'masks', self.args[-1]),
                ('masks', 'resource_io', self.args[-1]),
                ('resource_io', 'resource_io', dict(self.args[-1], stage='longest_masks'))):
                with self.subTest(kind=kind, role=role, stage=binding['stage']), self.assertRaisesRegex(ValueError, 'only for resource'):
                    reader._envelope(reference, binding, kind, role, json_bytes_upper=raw.MAX_RESOURCE_JSON)
            with self.assertRaisesRegex(ValueError, 'raw JSON byte ceiling'):
                reader._envelope(reference, self.args[-1], 'resource_io', 'resource_io', json_bytes_upper=raw.MAX_JSON)
        self.assertEqual(self.events, [])

    def test_changed_frozen_caps_and_zero_or_unbounded_work_refuse(self):
        reference = self.artifact(); envelope = json.loads(Path(reference['path']).read_bytes())
        payload = envelope['payload']; self.spec['json_bytes_upper'] = 8 * raw.CHUNK
        reader = self.reader()
        with self.assertRaisesRegex(ValueError, 'ceilings differ from frozen'):
            verifier._resource_readback(self.args[0], None, payload, self.args[-1], reader=reader,
                                        reference_bytes=reference['bytes'])
        self.assertEqual(self.events, [])
        for cap, work in ((raw.MAX_RESOURCE_JSON + 1, 4 * raw.MAX_RESOURCE_JSON),
                         (raw.MAX_RESOURCE_JSON, 0), (raw.MAX_RESOURCE_JSON, True),
                         (raw.MAX_RESOURCE_JSON, 4 * raw.MAX_RESOURCE_JSON + 1)):
            self.spec.update(json_bytes_upper=cap, reducer_work_bytes_upper=work)
            with self.subTest(cap=cap, work=work), patch.object(controls.a, 'require_execution_permit', return_value=None), \
                 patch.object(Path, 'open', side_effect=AssertionError('invalid resource bounds IO')):
                with self.assertRaises(ValueError): resource.resource_adapter(*self.args)

    def test_execution_reductions_supply_frozen_resource_bound(self):
        reference = self.artifact(); context = self.args[0]
        context['prepared_receipt'] = {'prepared_row_bindings': []}
        context['bindings']['scope']['qualifier_execution'] = {'all_witness_ids': [], 'gradient_batch_ids': []}
        attempt = dict(stage='resource_io_measurement', attempt_id=self.args[-1]['attempt_id'],
                       status='RETURNED', evidence={'raw_refs': [reference]})
        state = SimpleNamespace(attempts=[attempt]); reader = self.reader()
        with patch.object(reader, '_envelope', wraps=reader._envelope) as read, \
             patch.object(verifier, '_resource_readback', return_value={'fixture': True}):
            result, _ = verifier._reductions(context, state, {attempt['attempt_id']: controls.WORKER}, reader, {}, {})
        self.assertEqual(read.call_args.kwargs['json_bytes_upper'], self.spec['json_bytes_upper'])
        self.assertEqual(result['resource_io_measurement'], {'fixture': True})
        self.assertEqual(raw.MAX_JSON, 4 * raw.CHUNK)
        self.spec['reducer_work_bytes_upper'] = 0
        with patch.object(Path, 'open', side_effect=AssertionError('unbounded resource verifier IO')):
            with self.assertRaisesRegex(ValueError, 'positive bounded resource reducer'):
                verifier._reductions(context, state, {attempt['attempt_id']: controls.WORKER}, self.reader(), {}, {})
        self.assertEqual(self.events, [])


if __name__ == '__main__': unittest.main()
