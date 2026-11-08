"""Real bounded metadata intake fixtures, no registry/OS/model/child activation."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_checkpoint_configuration as c
from .test_ce_checkpoint_entry_provider import fixture


class Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='child-configuration-pure-'); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        configuration, common, request, proof, ledger = fixture(self.root)
        configuration['service_spec']['resource'] = {'resource_probe': {'output_root': str(self.root)}}
        self.configuration = configuration
        self.body = {'schema': c.SCHEMA, 'branch': 'uninterrupted', 'request_ref': configuration['launch_spec']['request_ref'],
            'choice_sha256': 'a' * 64, 'static_recipe_sha256': common['static_recipe_sha256'], 'configuration': configuration}
        self.body['configuration_sha256'] = c.digest(self.body)
        self.path = self.root / 'checkpoint' / 'children' / 'uninterrupted' / 'CHILD_CONFIGURATION.json'
        self.write_body()
        self.declaration = {'schema': c.DECLARATION, 'configuration_root': str(self.root / 'checkpoint'),
            'owned_directory': str(self.root / 'intake'), 'source_checkout': str(self.root), 'output_root': str(self.root),
            'choice_sha256': 'a' * 64, 'static_recipe_sha256': common['static_recipe_sha256'],
            'bounds': {'read_bytes': 64 * 1024 ** 2, 'write_bytes': 65536, 'hash_attempts': 1000,
                'hash_bytes': 256 * 1024 ** 2, 'metadata_bytes': 65536, 'operation_attempts': 16, 'wall_seconds': 120}}
        # Source checkout and output are separate in the declared admission.
        source_root = self.root / 'checkout'; source_root.mkdir()
        refs = configuration['launch_spec']['source_files']
        for name, ref in refs.items():
            old = Path(ref['path']); new = source_root / name; new.parent.mkdir(parents=True, exist_ok=True)
            new.write_bytes(old.read_bytes()); ref['path'] = str(new)
        configuration['launch_spec']['source_checkout'] = configuration['launch_spec']['cwd'] = str(source_root)
        self.declaration['source_checkout'] = str(source_root)
        self.body['configuration_sha256'] = c.digest({k: v for k, v in self.body.items() if k != 'configuration_sha256'})
        self.write_body()
    def write_body(self):
        blob = c.canonical(self.body) + b'\n'; self.path.write_bytes(blob)
        self.reference = {'path': str(self.path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}
        self.environment = {c.MARKER: c.canonical(self.reference).decode('ascii')}

    def test_default_source_declaration_refuses_before_environment_args_config_or_registry(self):
        with patch.object(Path, 'open', side_effect=AssertionError('unselected intake IO')):
            with self.assertRaisesRegex(ValueError, 'adoption remains closed'): c.install_child_configuration()
        self.assertIsNone(c._SOURCE_ADOPTION); self.assertIsNone(c._LOADED)

    def test_config_ref_is_outside_immutable_request_and_final_spec_reconstructed_without_cycle(self):
        configuration, evidence = c.fixture_read_configuration(self.declaration, self.environment)
        final = configuration['launch_spec']
        self.assertEqual(final['request_ref'], self.body['request_ref'])
        self.assertNotIn(c.MARKER, self.body['configuration']['launch_spec']['environment'])
        self.assertEqual(c.raw._json(final['environment'][c.MARKER]), self.reference)
        self.assertTrue(evidence['source_files_verified']); self.assertFalse(evidence['native_qualified'])
        rows = [c.raw._json(row) for row in Path(evidence['journal_path']).read_bytes().splitlines()]
        self.assertEqual([row['kind'] for row in rows], ['PERMANENT_RESERVATION', 'RETURNED'])
        self.assertEqual(evidence['permanent_bounds'], self.declaration['bounds'])

    def test_reservation_exists_before_environment_access_and_three_slots_never_refund(self):
        class Environment:
            def get(inner, key):
                self.assertTrue(any((self.root / 'intake').glob('child-*.jsonl')))
                raise ValueError('environment transport denied')
        for _ in range(3):
            with self.assertRaisesRegex(ValueError, 'transport denied'): c.fixture_read_configuration(self.declaration, Environment())
        with self.assertRaisesRegex(ValueError, 'slots already retained'): c.fixture_read_configuration(self.declaration, self.environment)
        for path in (self.root / 'intake').glob('child-*.jsonl'):
            rows = [c.raw._json(row) for row in path.read_bytes().splitlines()]
            self.assertEqual(rows[-1]['kind'], 'FAILED'); self.assertFalse(rows[-1]['payload']['charges_refunded'])

    def test_selfdigest_rawhash_or_current_source_drift_refuses_and_retains_failure(self):
        source = next(iter(self.configuration['launch_spec']['source_files'].values()))
        Path(source['path']).write_bytes(b'changed current source\n')
        with self.assertRaisesRegex(ValueError, 'source origin/size|source file raw hash'):
            c.fixture_read_configuration(self.declaration, self.environment)
        journal = next((self.root / 'intake').glob('*.jsonl'))
        self.assertEqual(c.raw._json(journal.read_bytes().splitlines()[-1])['kind'], 'FAILED')

    def test_json_configuration_cannot_call_source_installer_before_adoption(self):
        from experiments.skill_acquisition_native_v1 import ce_qualification_entry as entry
        before = entry.services._REGISTERED_ROLE_FACTORIES
        with self.assertRaisesRegex(ValueError, 'source-adopted paid child'): entry.install_child_source_registries(self.configuration)
        self.assertIs(entry.services._REGISTERED_ROLE_FACTORIES, before)


if __name__ == '__main__': unittest.main()
