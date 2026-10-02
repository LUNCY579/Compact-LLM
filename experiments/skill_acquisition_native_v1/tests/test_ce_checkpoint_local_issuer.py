"""Streams/files only: no framework, child process, native permit or registry."""
import copy
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_checkpoint_local_issuer as local
from experiments.skill_acquisition_native_v1 import ce_checkpoint_worker as child, ce_worker_evidence as worker
from . import test_ce_qualification_accounting as qa


def policy():
    return {'schema': local.SPEC, 'deadline_seconds': 20, 'rpc_seconds': 1,
        'max_message_bytes': 4096, 'max_messages': 5, 'max_request_bytes': 65536,
        'operation_bounds': {'read_bytes_upper': 65537, 'write_bytes_upper': 4096,
            'identity_hash_attempts_upper': 64, 'identity_hash_bytes_upper': 64 * 65536, 'wall_s_upper': 20}}


def fixture(root):
    context, proof, receipt = qa.fixture(root)
    context['bindings']['scope']['checkpoint_qualification'] = {'owned_directory': str(root / 'checkpoint')}
    context['bindings']['scope']['checkpoint_branch_publication'] = {'schema': 'CE8-CHECKPOINT-BRANCH-PUBLICATION-SPEC-1'}
    context['bindings']['scope']['checkpoint_lease'] = {'lock_path': str(root / 'rineya-gpu-0.lock')}
    helper = qa.Tests(); helper.root, helper.context, helper.proof, helper.receipt = root, context, proof, receipt
    payload = helper.child_payload('uninterrupted')
    identity = {'schema': worker.IDENTITY, 'pid': 41, 'creation_filetime_100ns': 120,
        'boot_identity': 'windows-boot-guid-fixture', 'image_path': str(root / 'python.exe'), 'argv': ['fixture']}
    parent = payload['parent_proof']
    parent.update(request_sha256=context['request_sha256'], scope_sha256=context['scope_sha256'])
    parent['os_identity'] = dict(identity, pid=40, creation_filetime_100ns=110)
    reference = {'path': str(root / 'request.json'), 'sha256': '7' * 64, 'bytes': 120}
    source_ref = {'path': str(root / 'fixture.py'), 'sha256': 'c' * 64, 'bytes': 10}
    sources = {'python_ref': source_ref, 'process_image_ref': source_ref, 'request_ref': reference,
        'source_files': {'synthetic-fixture': source_ref}, 'entry': 'fixture', 'entry_script_ref': None}
    proof = dict(proof, os_identity=identity, sources=sources, request_sha256=context['request_sha256'],
        scope_sha256=context['scope_sha256'], fresh_process=False, source_verified=False, native_qualified=False,
        execution_verified=False)
    proof['worker_identity_sha256'] = local.digest(worker.worker_identity_preimage(identity, sources,
        context['request_sha256'], context['scope_sha256']))
    from experiments.skill_acquisition_native_v1 import ce_qualification_accounting as accounting
    declaration = accounting.checkpoint_child_declaration(payload)
    request = {'schema': child.REQUEST, 'branch': 'uninterrupted', 'study_request_sha256': context['request_sha256'],
        'scope_sha256': context['scope_sha256'], 'static_recipe_sha256': context['static_recipe_sha256'],
        'code_sha256': context['bindings']['source']['code_sha256'],
        'common_anchor_sha256': context['bindings']['anchors']['common_anchor_sha256'],
        'ingestion_receipt_sha256': receipt['receipt_sha256'], 'lineage_declaration': declaration,
        'plan_sha256': local.digest({'qualification_batches': declaration['qualification_batches'], 'branch': 'uninterrupted'}),
        'initializer_inputs': {'parent_binding': {'synthetic': True}},
        'lease_delegation_path': str(root / 'checkpoint' / 'children' / 'uninterrupted' / 'LEASE_DELEGATION.json')}
    delegation = {'schema': 'CE8-CHECKPOINT-GPU-DELEGATION-1', 'branch': request['branch'],
        'request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256'],
        'child_worker_identity_sha256': proof['worker_identity_sha256'], 'child_os_identity': identity,
        'child_request_ref': reference, 'parent_worker_identity_sha256': parent['worker_identity_sha256'],
        'parent_os_identity': parent['os_identity'], 'lock_path': context['bindings']['scope']['checkpoint_lease']['lock_path']}
    content = local.canonical(delegation) + b'\n'
    delegation_ref = {'path': request['lease_delegation_path'], 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    return context, request, reference, proof, delegation, delegation_ref


class Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='local-child-rpc-pure-')
        self.addCleanup(temporary.cleanup); self.root = Path(temporary.name)
        self.context, self.request, self.reference, self.proof, self.delegation, self.delegation_ref = fixture(self.root)
        self.binding = {'session_id': 'fixture', 'request_sha256': self.context['request_sha256'], 'scope_sha256': self.context['scope_sha256']}
        self.events = []
    def operation(self, kind, phase, **fields): self.events.append((kind, phase, fields))
    def message(self, sequence, command, data):
        return local.canonical({'schema': local.launch.RPC, **self.binding, 'kind': 'control', 'sequence': sequence,
            'worker_pid': 41, 'payload': {'command': command, 'data': data}}) + b'\n'
    def stream(self, first='checkpoint_qualification', data=None, finish=True):
        if data is None: data = {'schema': child.COMMAND, 'branch': self.request['branch'], 'plan_sha256': self.request['plan_sha256']}
        content = self.message(1, first, data)
        if finish: content += self.message(2, 'finish', {})
        return io.BytesIO(content)
    def body(self, command):
        self.assertEqual(child.validate_command(command, self.request), command)
        return {'raw_artifact': {'path': str(self.root / 'branch.json'), 'bytes': 100, 'sha256': 'a' * 64,
            'role': 'checkpoint_branch', 'binding_sha256': 'b' * 64}, 'body_completed': True, 'native_qualified': False}
    def serve(self, source, body=None, output=None):
        request = dict(self.request, rpc_spec=policy())
        output = io.BytesIO() if output is None else output
        result = local.serve_fixture({'schema': local.FIXTURE, 'native_qualified': False, 'execution_verified': False},
            request, self.binding, source, output, self.operation, self.body if body is None else body)
        return result, output

    def test_closed_native_entry_denies_before_path_environment_or_object_access(self):
        class Trap:
            def __getattribute__(self, key): raise AssertionError('unapproved object access')
        before = set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'}
        with patch.object(Path, 'open', side_effect=AssertionError('unapproved IO')):
            with self.assertRaisesRegex(ValueError, 'entry provider remains closed'):
                local.LocalCheckpointIssuer().run_frozen_request(Trap(), Trap(), Trap())
        self.assertIsNone(local._SOURCE_ENTRY_PROVIDER)
        self.assertFalse(child._REGISTERED_LOCAL_ISSUERS)
        self.assertEqual(before, set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'})

    def test_actual_stream_protocol_returns_only_ref_then_awaits_explicit_finish(self):
        def logged_body(command):
            print('[lease] child runtime', flush=True)
            return self.body(command)
        diagnostic = io.StringIO()
        with patch.object(local.sys, 'stderr', diagnostic):
            result, output = self.serve(self.stream(), body=logged_body)
        self.assertEqual(diagnostic.getvalue(), '[lease] child runtime\n')
        self.assertEqual(result, 0)
        rows = [local.raw._json(line) for line in output.getvalue().splitlines()]
        self.assertEqual([(row['kind'], row['sequence']) for row in rows], [('ready', 0), ('response', 1), ('response', 2)])
        self.assertEqual(set(rows[1]['payload']), {'raw_artifact', 'body_completed', 'os_terminal_verified', 'native_qualified'})
        self.assertFalse(rows[1]['payload']['os_terminal_verified'])
        self.assertEqual([event[1] for event in self.events], ['started', 'returned'] * 5)

    def test_early_finish_foreign_plan_grant_duplicate_key_or_reordered_sequence_cannot_start_body(self):
        bad = [self.stream('finish', {}), self.stream(data={'schema': child.COMMAND, 'branch': 'prefix', 'plan_sha256': '0' * 64}),
            self.stream(data={'schema': child.COMMAND, 'branch': 'uninterrupted', 'plan_sha256': self.request['plan_sha256'], 'native_permit': True})]
        encoded = self.stream().getvalue()
        bad += [io.BytesIO(encoded.replace(b'"sequence":1', b'"sequence":2', 1)),
                io.BytesIO(encoded.replace(b'"kind":"control"', b'"kind":"control","kind":"control"', 1))]
        for source in bad:
            with self.subTest(content=source.getvalue()[:100]), self.assertRaises(ValueError):
                self.serve(source, body=lambda command: self.fail('bad command reached branch body'))

    def test_missing_finish_or_branch_error_retains_failed_cost_and_never_acknowledges_finish(self):
        with self.assertRaisesRegex(ValueError, 'complete RPC control'):
            self.serve(self.stream(finish=False))
        self.assertEqual(self.events[-1][1], 'failed')
        original = RuntimeError('branch failed')
        def fail(command): raise original
        with self.assertRaises(RuntimeError) as caught: self.serve(self.stream(), body=fail)
        self.assertIs(caught.exception, original)

    def test_source_reads_prepaid_denial_hash_drift_and_short_rpc_write(self):
        path = self.root / 'source.json'; content = b'{"source":true}\n'; path.write_bytes(content)
        ref = {'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
        def deny(*args, **kwargs): raise ValueError('price denied')
        with patch.object(Path, 'open', side_effect=AssertionError('unpaid source read')):
            with self.assertRaisesRegex(ValueError, 'price denied'):
                local.read_json(ref, self.root, deny, policy()['operation_bounds'], 65536)
        self.assertEqual(local.read_json(ref, self.root, self.operation, policy()['operation_bounds'], 65536), {'source': True})
        path.write_bytes(b'{"source":null}\n')
        with self.assertRaises(ValueError): local.read_json(ref, self.root, self.operation, policy()['operation_bounds'], 65536)
        class Short(io.BytesIO):
            def write(self, value): return 0
        with self.assertRaisesRegex(ValueError, 'short/failed write'): self.serve(self.stream(), output=Short())
        self.assertEqual(self.events[-1][1], 'failed')

    def test_complete_derived_context_binds_observed_identity_without_mutating_shared_scope(self):
        original = copy.deepcopy(self.context)
        derived = local.derive_context(self.context, self.request, self.reference, self.proof, self.delegation, self.delegation_ref)
        self.assertEqual(self.context, original)
        self.assertEqual(derived['bindings']['scope'], original['bindings']['scope'])
        self.assertEqual(derived['checkpoint_child_observed_lineage']['child_request_sha256'], self.reference['sha256'])
        self.assertNotIn('worker_identity_sha256', self.request['lineage_declaration'])
        self.assertEqual(derived['checkpoint_child_delegation_ref'], self.delegation_ref)
        self.assertFalse(derived['checkpoint_child_observed_lineage']['child_proof']['execution_verified'])
        self.assertTrue(derived['checkpoint_child_accounting']['specification']['owned_directory'].endswith('uninterrupted\\ledger'))

    def test_child_accounting_keeps_actual_runtime_setup_refs_in_derived_owner(self):
        common = copy.deepcopy(self.context)
        scope = common['bindings']['scope']['qualifier_accounting']
        scope['ingestion_bootstrap_ref'] = scope['outer_setup_reservation_ref'] = None
        common['qualifier_setup_refs'] = {name: {'path': str(self.root / (name + '.json')), 'sha256': str(index) * 64, 'bytes': 12}
            for index, name in enumerate(('ingestion_bootstrap_ref', 'outer_setup_reservation_ref'), 1)}
        frozen = copy.deepcopy(common)
        derived = local.derive_context(common, self.request, self.reference, self.proof, self.delegation, self.delegation_ref)
        spec = derived['checkpoint_child_accounting']['specification']
        self.assertEqual(common, frozen)
        self.assertEqual(derived['bindings']['scope'], frozen['bindings']['scope'])
        self.assertEqual(derived['qualifier_setup_refs'], common['qualifier_setup_refs'])
        for name, reference in common['qualifier_setup_refs'].items(): self.assertEqual(spec[name], reference)
        self.assertEqual(derived['checkpoint_child_accounting']['owner']['accounting_spec_sha256'], local.digest(spec))

    def test_foreign_worker_source_parent_rawrequest_or_delegation_ref_denied(self):
        for change in ('worker', 'source', 'parent', 'request', 'ref', 'path'):
            context, request, reference, proof, delegation, delegation_ref = copy.deepcopy(
                (self.context, self.request, self.reference, self.proof, self.delegation, self.delegation_ref))
            if change == 'worker': proof['os_identity']['pid'] = 40
            elif change == 'source': proof['sources']['source_files']['synthetic-fixture']['sha256'] = '0' * 64
            elif change == 'parent': delegation['parent_os_identity']['pid'] = 99
            elif change == 'request': reference['sha256'] = '0' * 64
            elif change == 'ref': delegation_ref['sha256'] = '0' * 64
            else: request['lease_delegation_path'] = str(self.root / 'foreign.json')
            with self.subTest(change=change), self.assertRaises(ValueError):
                local.derive_context(context, request, reference, proof, delegation, delegation_ref)

    def test_parent_view_uses_exact_os_source_identity_with_informational_view_changes(self):
        retained = copy.deepcopy(self.proof); observed = copy.deepcopy(retained)
        retained['sources'].update(runtime_launcher={'path': 'parent-current'}, launcher_provenance='current runtime')
        observed['sources'].update(runtime_launcher=None, launcher_provenance='supervisor/remote')
        self.assertIs(local.validate_parent_view(observed, retained), observed)
        for change in ('os', 'source', 'scope'):
            altered = copy.deepcopy(observed)
            if change == 'os': altered['os_identity']['creation_filetime_100ns'] += 1
            elif change == 'source': altered['sources']['process_image_ref']['sha256'] = '0' * 64
            else: altered['scope_sha256'] = '0' * 64
            with self.subTest(change=change), self.assertRaises(ValueError): local.validate_parent_view(altered, retained)

    def test_entry_failure_preserves_original_when_accounting_also_fails(self):
        original = ValueError('source read failed')
        class Payment:
            def begin(self): pass
            def complete(self): raise AssertionError('failed entry cannot complete')
            def fail(self, error): raise RuntimeError('outer append failed')
        def body(): raise original
        with self.assertRaises(ValueError) as caught: local.entry_lifetime(Payment(), body)
        self.assertIs(caught.exception, original)
        self.assertEqual(original.accounting_failure['message'], 'outer append failed')
        self.assertIsInstance(original.__cause__, RuntimeError)

    def test_observed_delegation_is_prepaid_read_from_exact_branch_path(self):
        path = Path(self.request['lease_delegation_path']); path.parent.mkdir(parents=True)
        path.write_bytes(local.canonical(self.delegation) + b'\n')
        context = dict(self.context, checkpoint_child_declaration=self.request['lineage_declaration'])
        record, reference = local.read_delegation(str(path), context, self.operation, policy()['operation_bounds'], 65536)
        self.assertEqual(record, self.delegation); self.assertEqual(reference, self.delegation_ref)
        with patch.object(Path, 'open', side_effect=AssertionError('foreign delegation read')):
            with self.assertRaisesRegex(ValueError, 'source-derived child delegation path'):
                local.read_delegation(str(self.root / 'foreign.json'), context, self.operation, policy()['operation_bounds'], 65536)

    def test_body_error_cleanup_and_failed_accounting_preserve_original(self):
        original, events = ValueError('branch failed'), []
        class Payment:
            def begin(self): events.append('begin')
            def complete(self): raise AssertionError('failure completed')
            def fail(self, error): events.append('failed'); self.error = error
        payment = Payment()
        def body(): events.append('body'); raise original
        def cleanup(): events.append('cleanup'); raise RuntimeError('handle release failed')
        with self.assertRaises(ValueError) as caught: local.entry_lifetime(payment, body, cleanup)
        self.assertIs(caught.exception, original); self.assertIs(payment.error, original)
        self.assertEqual(events, ['begin', 'body', 'cleanup', 'failed'])
        self.assertEqual(original.entry_cleanup_failure['message'], 'handle release failed')
        self.assertIsInstance(original.__cause__, RuntimeError)


if __name__ == '__main__': unittest.main()
