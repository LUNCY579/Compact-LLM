"""Actual metadata intake fixtures; no registry/model/process execution."""
import json
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_qualification_configuration as c


def declaration(root):
    return {'schema': c.DECLARATION, 'configuration_path': str(root / 'config.json'),
        'owned_directory': str(root / 'output' / 'intake'), 'source_checkout': str(root / 'source'),
        'output_root': str(root / 'output'), 'choice_sha256': 'a' * 64, 'static_recipe_sha256': 'b' * 64,
        'bounds': {'read_bytes': c.MAX_CONFIGURATION + 1, 'write_bytes': 65536, 'hash_attempts': 16,
            'hash_bytes': 4 * c.MAX_CONFIGURATION, 'metadata_bytes': 65536, 'operation_attempts': 8, 'wall_seconds': 120}}


def configuration(policy):
    value = {'schema': 'CE8-QUALIFICATION-SOURCE-CONFIGURATION-1',
        'request': {'phase': 'qualification', 'arm': 'canonical', 'training_authorized': False,
            'selectedoperatorchoice_ref': {'sha256': policy['choice_sha256']}},
        'service_spec': {'scope_adapters': {'static_recipe_sha256': policy['static_recipe_sha256'],
                'worker': {'source_checkout': policy['source_checkout']}},
            'resource': {'resource_probe': {'output_root': policy['output_root']}},
            'bootstrap': {'owned_directory': str(Path(policy['output_root']) / 'services')},
            'constructor_prepayment': {'owned_directory': str(Path(policy['output_root']) / 'services')}}}
    return dict(value, configuration_sha256=c.digest(value))


class IntakeTests(unittest.TestCase):
    def test_closed_source_adoption_denies_before_fs(self):
        with patch.object(Path, 'open', side_effect=AssertionError('IO before source adoption')):
            with self.assertRaisesRegex(ValueError, 'remains closed'): c.load_source_configuration()
        self.assertIsNone(c._LOADED)

    def test_actual_permanent_reservation_precedes_config_read_and_failure_retains_charge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); policy = declaration(root)
            value = configuration(policy); value['configuration_sha256'] = 'c' * 64
            (root / 'config.json').write_bytes(c.canonical(value) + b'\n')
            original = Path.open; observed = []
            def opened(path, mode='r', *args, **kwargs):
                if path == root / 'config.json' and mode == 'rb':
                    histories = list((root / 'output' / 'intake').glob('*.jsonl'))
                    first = json.loads(histories[0].read_bytes().splitlines()[0])
                    observed.append(first['kind'])
                return original(path, mode, *args, **kwargs)
            with patch.object(c, '_SOURCE_ADOPTION', MappingProxyType(policy)), \
                 patch.object(c, '_LOADED', None), patch.object(c, '_EVIDENCE', None), patch.object(Path, 'open', opened):
                with self.assertRaisesRegex(ValueError, 'digest drift'): c.load_source_configuration()
            rows = [json.loads(line) for line in next((root / 'output' / 'intake').glob('*.jsonl')).read_bytes().splitlines()]
            self.assertEqual(observed, ['PERMANENT_RESERVATION'])
            self.assertEqual([row['kind'] for row in rows], ['PERMANENT_RESERVATION', 'FAILED'])
            self.assertFalse(rows[-1]['payload']['charges_refunded'])

    def test_success_is_metadata_only_cached_and_has_no_request_source_hash_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); policy = declaration(root); value = configuration(policy)
            (root / 'config.json').write_bytes(c.canonical(value) + b'\n')
            with patch.object(c, '_SOURCE_ADOPTION', MappingProxyType(policy)), \
                 patch.object(c, '_LOADED', None), patch.object(c, '_EVIDENCE', None):
                actual, evidence = c.load_source_configuration()
                with patch.object(Path, 'open', side_effect=AssertionError('cached configuration reread')):
                    again, retained = c.load_source_configuration()
                self.assertEqual(actual, again); self.assertEqual(evidence, retained)
                self.assertTrue(evidence['metadata_only']); self.assertFalse(evidence['native_qualified'])
                self.assertFalse(evidence['execution_verified'])
            self.assertNotIn('request_sha256', policy); self.assertNotIn('code_sha256', policy)

    def test_training_request_cannot_be_adopted_even_with_self_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); policy = declaration(root); value = configuration(policy)
            value['request']['phase'] = 'training'
            value['configuration_sha256'] = c.digest({k: v for k, v in value.items() if k != 'configuration_sha256'})
            (root / 'config.json').write_bytes(c.canonical(value) + b'\n')
            with patch.object(c, '_SOURCE_ADOPTION', MappingProxyType(policy)), \
                 patch.object(c, '_LOADED', None), patch.object(c, '_EVIDENCE', None):
                with self.assertRaisesRegex(ValueError, 'training or noncanonical'): c.load_source_configuration()
