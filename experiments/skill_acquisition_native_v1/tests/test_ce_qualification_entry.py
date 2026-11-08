"""Pure parent RPC/source-closed controls, no registry activation or OS/model."""
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_qualification_entry as entry
from .test_ce_checkpoint_local_issuer import policy


class Tests(unittest.TestCase):
    def setUp(self):
        self.binding = {'session_id': 'parent-fixture', 'request_sha256': 'a' * 64, 'scope_sha256': 'b' * 64}
        self.proof = {'fresh_process': False, 'source_verified': False, 'native_qualified': False, 'execution_verified': False}
        self.events = []
    def control(self, seq, command, data):
        return entry.raw.canonical({'schema': entry.local.launch.RPC, **self.binding, 'kind': 'control',
            'sequence': seq, 'worker_pid': 41, 'payload': {'command': command, 'data': data}}) + b'\n'
    def pipe(self, command='qualify', data=None, finish=True):
        content = self.control(1, command, {} if data is None else data)
        if finish: content += self.control(2, 'finish', {})
        output = io.BytesIO()
        pipe = entry.local.BoundedRPC(policy(), self.binding, 41, io.BytesIO(content), output,
            lambda kind, phase, **fields: self.events.append((kind, phase, fields)))
        return pipe, output
    def result(self):
        return {'seal': {'status': 'UNQUALIFIED_BODY_SEAL', 'body_ref': {'path': str(Path.cwd() / 'fixture-parent.json'),
            'sha256': 'c' * 64, 'bytes': 100, 'role': 'parent_body', 'binding_sha256': 'd' * 64}}}

    def test_unselected_entry_rejects_before_argv_environment_import_files_or_registry_mutation(self):
        class Trap:
            def __iter__(self): raise AssertionError('argv touched before source config')
        before = entry.controller._REGISTERED_NATIVE_FACTORIES
        with patch.object(Path, 'open', side_effect=AssertionError('unapproved file')), \
                patch.object(entry, '_modules', side_effect=AssertionError('unapproved dependency import')):
            with self.assertRaisesRegex(ValueError, 'configuration remains closed'): entry.main(Trap())
            with self.assertRaisesRegex(ValueError, 'configuration remains closed'): entry.install_source_registries()
        self.assertIs(entry.controller._REGISTERED_NATIVE_FACTORIES, before)
        self.assertIsNone(entry._SOURCE_CONFIGURATION); self.assertIsNone(entry._INSTALLED_SHA256)
        self.assertNotIn('torch', sys.modules); self.assertNotIn('transformers', sys.modules)

    def test_source_callback_flags_cannot_install_or_issue_metadata_scope(self):
        with self.assertRaises(TypeError): entry.install_source_registries({'approved': True})
        for issuer in (True, {'approved': True}, object()):
            with self.assertRaisesRegex(ValueError, 'issuer identity'): entry.service_metadata_verifier(issuer, 'a' * 64, 'b' * 64)
        self.assertFalse(entry.services._REGISTERED_SERVICE_VERIFIERS)
        self.assertIn('child_launch_observation', entry.registry_manifest()['child'])

    def test_parent_rpc_retains_body_ref_waits_finish_and_never_claims_terminal_or_native(self):
        pipe, output = self.pipe(); called = []
        def qualify():
            print('[lease] acquired', flush=True)
            called.append('qualify'); return self.result()
        diagnostic = io.StringIO()
        with patch.object(entry.sys, 'stderr', diagnostic):
            self.assertEqual(entry.serve_parent(pipe, self.proof, qualify), 0)
        self.assertEqual(diagnostic.getvalue(), '[lease] acquired\n')
        rows = [entry.raw._json(row) for row in output.getvalue().splitlines()]
        self.assertEqual([(row['kind'], row['sequence']) for row in rows], [('ready', 0), ('response', 1), ('response', 2)])
        self.assertEqual(rows[0]['payload']['worker_proof'], self.proof)
        self.assertEqual(set(rows[0]['payload']), {'worker_proof'})
        self.assertEqual(rows[1]['payload']['raw_artifact'], {key: self.result()['seal']['body_ref'][key] for key in ('path', 'sha256', 'bytes')})
        self.assertFalse(rows[1]['payload']['os_terminal_verified']); self.assertFalse(rows[1]['payload']['native_qualified'])
        self.assertEqual(called, ['qualify'])

    def test_early_finish_scope_data_or_missing_finish_cannot_complete_parent_rpc(self):
        for command, data in (('finish', {}), ('qualify', {'approved': True}), ('qualify', {'scope': 'foreign'})):
            pipe, output = self.pipe(command, data)
            with self.subTest(command=command, data=data), self.assertRaises(ValueError):
                entry.serve_parent(pipe, self.proof, lambda: self.fail('invalid control reached model body'))
        pipe, output = self.pipe(finish=False)
        with self.assertRaises(ValueError): entry.serve_parent(pipe, self.proof, self.result)
        self.assertEqual(self.events[-1][1], 'failed')

    def test_body_failure_or_missing_retained_ref_emits_no_success_response(self):
        pipe, output = self.pipe(); original = RuntimeError('retained Q body failed')
        def fail(): raise original
        with self.assertRaises(RuntimeError) as caught: entry.serve_parent(pipe, self.proof, fail)
        self.assertIs(caught.exception, original)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        pipe, output = self.pipe()
        with self.assertRaisesRegex(ValueError, 'retained parent body'): entry.serve_parent(pipe, self.proof, lambda: {'body_completed': True})

    def test_canonical_entry_identity_accepts_actual_main_view_without_changing_source_symbol(self):
        self.assertEqual(entry.ENTRY_MODULE, 'experiments.skill_acquisition_native_v1.ce_qualification_entry')
        self.assertIs(entry.source_issuer_identity(), entry)
        with patch.object(entry, '__name__', '__main__'), patch.dict(sys.modules, {'__main__': entry}):
            self.assertIs(entry.source_issuer_identity(), entry)
        self.assertEqual(entry.ENTRY_MODULE, 'experiments.skill_acquisition_native_v1.ce_qualification_entry')


if __name__ == '__main__': unittest.main()
