"""Byte/field controls for inert envelope, not full workflow qualification."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from outer_context_fixture import OuterContextFixture, HERE
from native_fixture_support import producer_frame
from full_native_cpu_fixture_support import (PRODUCER_NAMES, LEASE_NAMES,
    materialize_full_native_CPU_fixture, verify_full_native_CPU_fixture_conservation)


class FullWorkflowFixtureSupportControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='CPU_full_envelope_control_', dir=HERE)
        self.root = Path(self.temporary.name).resolve()
        self.fixture = OuterContextFixture(self.root)
        base = self.root / 'actual_stage_producer'
        base.mkdir()
        self.produced = producer_frame(base)

    def tearDown(self):
        self.temporary.cleanup()

    def test_actual_producer_byte_and_selected_field_conservation_with_honest_stage_diagnostic(self):
        original = {p.name: p.read_bytes() for p in self.produced['raw'].iterdir()}
        frame = materialize_full_native_CPU_fixture(self.fixture, self.produced['raw'])
        self.assertTrue(verify_full_native_CPU_fixture_conservation(self.fixture, frame))
        for name in PRODUCER_NAMES:
            self.assertEqual((frame['raw'] / name).read_bytes(), original[name])
        self.assertEqual(frame['supervisor']['config_sha256'], self.fixture.anchors['expected_config_file'])
        self.assertEqual(frame['anchors'], {'config_path': str(self.fixture.config_path), 'anchors': self.fixture.anchors})
        self.assertEqual({p.name for p in frame['lease_root'].iterdir()}, set(LEASE_NAMES))
        self.assertFalse((frame['raw'] / 'PRESERVATION.json').exists())
        proof = frame['proof']
        self.assertTrue(proof['fixture_only'])
        for key in ('native_PASS', 'native_authority_issued', 'native_fit_established', 'current_runtime_qualified',
                'process_execution_performed', 'GPU_operations_performed', 'preservation_created'):
            self.assertIs(proof[key], False)
        self.assertFalse(proof['workflow_source_map_compatible'])
        self.assertTrue(proof['compatibility_diagnostics'])
        self.assertEqual(proof['source_map_snapshot_metadata'], self.fixture.config['source_files'])
        (frame['raw'] / 'OWNED_WORKER_RETURN.json').write_bytes(b'{}\n')
        with self.assertRaisesRegex(ValueError, 'producer original/envelope bytes'):
            verify_full_native_CPU_fixture_conservation(self.fixture, frame)

    def test_complete_declared_child_FILETIME_topology_is_carried_into_every_CPU_observation(self):
        membership = deepcopy(self.produced['container']['owned_membership_verified'])
        # A standalone normalized topology control uses the existing membership;
        # the extra descendant is declared CPU metadata, never a process query.
        topology = {'job_name': membership['job_name'], 'membership': membership,
            'owned_birth_roster': [
                {'pid': membership['launcher_root_pid'], 'parent_pid': membership['supervisor_pid'],
                    'creation_filetime_100ns': 134000000000000001},
                {'pid': 12346, 'parent_pid': membership['launcher_root_pid'],
                    'creation_filetime_100ns': 134000000000000002}],
            'supervisor_identity': {'pid': membership['supervisor_pid'], 'creation_filetime_100ns': 134000000000000000}}
        frame = materialize_full_native_CPU_fixture(self.fixture, self.produced['raw'], topology=topology)
        self.assertEqual({row['pid'] for row in frame['owned_identities']}, {12345, 12346})
        observations = self.fixture.C.parse((frame['raw'] / 'RESOURCE_OBSERVATIONS.json').read_bytes())
        for row in observations:
            if row['kind'] == 'CPU':
                self.assertEqual(row['owned_process_identities'], frame['owned_identities'])
        self.assertFalse(frame['proof']['topology_metadata_injected_by_envelope'])
        self.assertTrue(verify_full_native_CPU_fixture_conservation(self.fixture, frame))

    def test_missing_parent_and_boolean_birth_metadata_fail_before_output(self):
        membership = self.produced['container']['owned_membership_verified']
        topology = {'job_name': membership['job_name'], 'membership': membership,
            'owned_birth_roster': [
                {'pid': 12345, 'parent_pid': 12344, 'creation_filetime_100ns': 134000000000000001},
                {'pid': 12346, 'parent_pid': 777, 'creation_filetime_100ns': 134000000000000002}],
            'supervisor_identity': {'pid': 12344, 'creation_filetime_100ns': 134000000000000000}}
        with self.assertRaises(ValueError):
            materialize_full_native_CPU_fixture(self.fixture, self.produced['raw'], topology=topology)
        self.assertFalse((self.fixture.package / 'artifacts/prefix_input_owned_run_CPU').exists())
        topology['owned_birth_roster'][1].update(parent_pid=12345, creation_filetime_100ns=True)
        with self.assertRaises(ValueError):
            materialize_full_native_CPU_fixture(self.fixture, self.produced['raw'], topology=topology)


if __name__ == '__main__':
    unittest.main()
