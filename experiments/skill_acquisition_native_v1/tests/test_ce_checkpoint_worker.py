"""Closed child CLI and pure frozen branch contracts; no kernel/process/ML."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_checkpoint_worker as w
from experiments.skill_acquisition_native_v1 import ce_qualification_artifacts as raw


class ChildContracts(unittest.TestCase):
    def test_real_module_trampoline_uses_canonical_main_without_adopting_source_or_models(self):
        import subprocess
        script = '''import runpy,sys
from experiments.skill_acquisition_native_v1 import ce_checkpoint_worker as canonical,ce_qualification_adoption as adoption
called=[]
adoption.adopt_source_paths=lambda:called.append("source-closed")
def closed_main(argv=None):
    called.append("canonical-main")
    assert not canonical._REGISTERED_LOCAL_ISSUERS
    assert not any(name in sys.modules for name in ("torch","transformers","numpy","safetensors"))
    return 7
canonical.main=closed_main
try:runpy.run_module(canonical.__name__,run_name="__main__")
except SystemExit as result:
    assert result.code==7 and called==["source-closed","canonical-main"],(result.code,called)
    print("canonical closed trampoline verified")
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('canonical closed trampoline verified', result.stdout)

    def test_service_requires_exact_registered_constructor_identity(self):
        class Registered: pass
        class Subclass(Registered): pass
        value = Registered()
        self.assertIs(w._service(value, (Registered,), 'pure constructor'), value)
        for replacement in (Subclass(), object(), value):
            registry = (Registered,) if replacement is not value else (value,)
            with self.subTest(kind=type(replacement).__name__), self.assertRaises(w.MissingCheckpointWorkerService):
                w._service(replacement, registry, 'pure constructor')
        self.assertFalse(w._REGISTERED_CHILD_INITIALIZERS)
        self.assertFalse(w._REGISTERED_CHILD_LEASE_SERVICES)
        self.assertFalse(w._REGISTERED_CHILD_PUBLICATION_SERVICES)

    def setUp(self):
        batches = [['r%d' % i for i in range(8)], ['r%d' % i for i in range(8, 16)]]
        self.context = {'phase': 'qualification', 'request_sha256': '1' * 64, 'scope_sha256': '2' * 64,
            'static_recipe_sha256': '3' * 64, 'bindings': {'source': {'code_sha256': {'worker.py': '4' * 64}},
                'anchors': {'common_anchor_sha256': '5' * 64}, 'scope': {
                    'qualifier_execution': {'adam_batches': batches},
                    'qualifier_accounting': {'ingestion_receipt_sha256': '6' * 64}}}}

    def request(self, branch):
        batches = self.context['bindings']['scope']['qualifier_execution']['adam_batches']
        pred = {'schema': w.DECLARATION, 'branch': branch,
            'qualification_batches': batches, 'inherited_U': 1 if branch == 'resumed' else 0,
            'study_request_sha256': self.context['request_sha256']}
        if branch == 'resumed':
            pred.update(prefix_manifest_ref={'path': str(Path.cwd() / 'fixture-prefix.json'), 'sha256': '8' * 64, 'bytes': 1},
                prefix_checkpoint_bindings_sha256='9' * 64, inherited_history=[{'fixture': True}])
        self.context['checkpoint_child_declaration'] = copy.deepcopy(pred)
        return {'schema': w.REQUEST, 'branch': branch, 'plan_sha256': raw.digest({'qualification_batches': batches, 'branch': branch}),
            'study_request_sha256': self.context['request_sha256'], 'scope_sha256': self.context['scope_sha256'],
            'static_recipe_sha256': self.context['static_recipe_sha256'],
            'code_sha256': self.context['bindings']['source']['code_sha256'], 'common_anchor_sha256': '5' * 64,
            'ingestion_receipt_sha256': '6' * 64, 'lineage_declaration': pred}

    def test_cli_denies_before_request_path_environment_or_framework(self):
        before = set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'}
        with patch.object(Path, 'open', side_effect=AssertionError('unadmitted path')):
            with self.assertRaisesRegex(w.MissingCheckpointWorkerService, 'issuer is not registered'):
                w.main(['--request', 'missing.json'])
        self.assertEqual(before, set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'})
        self.assertFalse(w._REGISTERED_LOCAL_ISSUERS)

    def test_handler_denies_before_traversing_objects_without_execution_permit(self):
        class Trap:
            def __getattribute__(self, key): raise AssertionError('unadmitted traversal')
        with self.assertRaises(ValueError):
            w.CheckpointChildHandler(Trap(), None, Trap(), Trap(), Trap(), Trap())
        self.assertFalse(w._REGISTERED_CHILD_INITIALIZERS)
        self.assertFalse(w._REGISTERED_CHILD_LEASE_SERVICES)
        self.assertFalse(w._REGISTERED_CHILD_PUBLICATION_SERVICES)

    def test_three_branch_requests_have_exact_source_batch_and_start_contracts(self):
        for branch in w.cp.BRANCHES:
            request = self.request(branch)
            self.assertEqual(w.validate_request(request, self.context), request)
            command = {'schema': w.COMMAND, 'branch': branch, 'plan_sha256': request['plan_sha256']}
            self.assertEqual(w.validate_command(command, request), command)

    def test_json_capability_flags_or_foreign_source_recipe_anchor_census_rejected(self):
        for key in ('native_permit', 'approved', 'qualification_authorized', 'observation_permit'):
            request = self.request('prefix'); request[key] = True
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'cannot supply execution'):
                w.validate_request(request, self.context)
        for key in ('study_request_sha256', 'scope_sha256', 'static_recipe_sha256', 'common_anchor_sha256', 'ingestion_receipt_sha256'):
            request = self.request('prefix'); request[key] = 'a' * 64
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'binding'):
                w.validate_request(request, self.context)
        request = self.request('prefix'); request['code_sha256'] = {'foreign.py': 'b' * 64}
        with self.assertRaisesRegex(ValueError, 'binding'): w.validate_request(request, self.context)

    def test_resumed_branch_requires_u1_history_and_original_binding_manifest(self):
        for change in (lambda p: p.update(inherited_U=True), lambda p: p.update(inherited_U=0),
                       lambda p: p.update(inherited_history=[]), lambda p: p.pop('prefix_manifest_ref'),
                       lambda p: p.update(prefix_checkpoint_bindings_sha256='not-a-hash')):
            request = self.request('resumed'); change(request['lineage_declaration'])
            with self.assertRaises(ValueError): w.validate_request(request, self.context)

    def test_rpc_branch_or_plan_change_cannot_select_another_kernel(self):
        request = self.request('prefix')
        for command in ({'schema': w.COMMAND, 'branch': 'resumed', 'plan_sha256': request['plan_sha256']},
                        {'schema': w.COMMAND, 'branch': 'prefix', 'plan_sha256': 'a' * 64},
                        {'schema': w.COMMAND, 'branch': 'prefix', 'plan_sha256': request['plan_sha256'], 'native_permit': True}):
            with self.assertRaisesRegex(ValueError, 'branch command'): w.validate_command(command, request)

    def test_production375_plan_or_undeclared_more_updates_refuses(self):
        context = copy.deepcopy(self.context); context['phase'] = 'training'
        with self.assertRaises(ValueError): w.validate_request(self.request('prefix'), context)

    def test_static_request_excludes_future_identity_and_observed_envelope_binds_actual_raw_source(self):
        request = self.request('resumed')
        frozen_scope = copy.deepcopy(self.context['bindings']['scope'])
        self.assertNotIn('worker_identity_sha256', request)
        self.assertNotIn('child_request_sha256', request['lineage_declaration'])
        encoded = raw.canonical(request) + b'\n'
        import hashlib
        reference = {'path': str(Path.cwd() / 'immutable-child-request.json'),
            'sha256': hashlib.sha256(encoded).hexdigest(), 'bytes': len(encoded)}
        proof = {'schema': 'CE8-QUALIFICATION-WORKER-PROOF-1', 'worker_identity_sha256': '7' * 64,
                 'sources': {'request_ref': reference}}
        observed = dict(request['lineage_declaration'], schema='CE8-CHECKPOINT-CHILD-PREDECESSOR-1',
            worker_identity_sha256=proof['worker_identity_sha256'], child_request_sha256=reference['sha256'],
            child_request_ref=reference, child_proof=proof,
            declaration_sha256=raw.digest(request['lineage_declaration']))
        self.context['checkpoint_child_observed_lineage'] = observed
        self.assertEqual(w.validate_observed_lineage(request, self.context, proof), observed)
        self.assertNotEqual(observed['child_request_sha256'], self.context['request_sha256'])
        self.assertEqual(self.context['bindings']['scope'], frozen_scope)
        wrong = copy.deepcopy(proof); wrong['sources']['request_ref']['sha256'] = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'actual observed child proof'):
            w.validate_observed_lineage(request, self.context, wrong)

    def test_future_worker_identity_or_own_file_hash_cannot_appear_in_immutable_request(self):
        for key in ('worker_identity_sha256', 'child_request_sha256', 'request_sha256'):
            request = self.request('prefix'); request[key] = 'a' * 64
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'cannot supply execution'):
                w.validate_request(request, self.context)
        request = self.request('prefix'); request['lineage_declaration']['worker_identity_sha256'] = '7' * 64
        with self.assertRaisesRegex(ValueError, 'future OS identity'):
            w.validate_request(request, self.context)
        context = copy.deepcopy(self.context)
        context['bindings']['scope']['qualifier_execution']['adam_batches'] *= 3
        with self.assertRaises(ValueError): w.validate_request(self.request('prefix'), context)

    def test_negative_classifier_refuses_infrastructure_and_decorated_errors(self):
        message = 'latest checkpoint payload bytes corrupt'
        self.assertEqual(w.expected_negative_error(ValueError(message), message),
                         {'type': 'ValueError', 'message': message})
        class ResourceFailure(ValueError): pass
        failures = [ResourceFailure(message), RuntimeError(message), ValueError('budget exceeded'),
                    ValueError('exact ancestor nested restore fence')]
        for field in ('accounting_failure', 'service_failure', 'cleanup_failure'):
            error = ValueError(message); setattr(error, field, {'message': 'infrastructure error'})
            failures.append(error)
        error = ValueError(message); error.__cause__ = RuntimeError('observer failed'); failures.append(error)
        for error in failures:
            with self.subTest(error=error), self.assertRaises(type(error)) as caught:
                w.expected_negative_error(error, message)
            self.assertIs(caught.exception, error)

    def test_corrupt_rejection_requires_new_failed_restore_with_exact_worker_attempt_error(self):
        binding = {'worker_identity_sha256': '7' * 64, 'attempt_id': 'attempt-1', 'stage': 'checkpoint_resume'}
        operation = {'kind': 'checkpoint_restore', 'phase': 'failed', 'worker_id': '7' * 64,
            'attempt_id': 'attempt-1', 'stage': 'checkpoint_resume', 'started_seq': 4, 'returned_seq': 5,
            'result': {'error': {'type': 'ValueError', 'message': 'latest checkpoint payload bytes corrupt'}}}
        self.assertEqual(w.failed_corruption_operation({'old': operation, 'new': operation}, {'old'}, binding),
                         {'operation_id': 'new', 'operation': operation})
        changes = [lambda op: op.update(phase='started'), lambda op: op.update(worker_id='a' * 64),
            lambda op: op.update(attempt_id='foreign'), lambda op: op.update(stage='delta_roundtrip'),
            lambda op: op['result']['error'].update(type='RuntimeError'),
            lambda op: op['result']['error'].update(message='resource floor'),
            lambda op: op.update(result={'error_type': 'ValueError', 'error': 'latest checkpoint payload bytes corrupt'}),
            lambda op: op.update(returned_seq=4)]
        for change in changes:
            op = copy.deepcopy(operation); change(op)
            with self.assertRaisesRegex(ValueError, 'newly retained'):
                w.failed_corruption_operation({'old': operation, 'new': op}, {'old'}, binding)
        for operations, previous in (({'old': operation}, {'old'}),
                                     ({'one': operation, 'two': operation}, set())):
            with self.assertRaisesRegex(ValueError, 'newly retained'):
                w.failed_corruption_operation(operations, previous, binding)

    def test_stale_control_attempt_is_exact_current_endpoint_and_nonfuture_generation(self):
        current = {'C': 2, 'U': 2, 'generation': 3, 'history': [{'updated': True}, {'updated': True}]}
        progress = dict(current, generation=2, snapshot='a' * 64, symbolic_snapshot='a' * 64)
        record = dict(progress, schema=w.cp.RESTORE_SCHEMA, progress=copy.deepcopy(progress),
                      manifest_sha256='b' * 64, install_fence_required=True)
        self.assertEqual(w.validate_stale_attempt(record, current), record)
        changes = [lambda r: r.update(U=True), lambda r: r['progress'].update(U=True),
            lambda r: r['progress'].update(C=1, U=1, history=[{'updated': True}]),
            lambda r: r['progress'].update(generation=4), lambda r: r['progress'].update(snapshot='c' * 64),
            lambda r: r.update(schema='foreign'), lambda r: r.update(install_fence_required=False)]
        for change in changes:
            attempted = copy.deepcopy(record); change(attempted)
            with self.assertRaises(ValueError): w.validate_stale_attempt(attempted, current)

    def lifetime_fixture(self, root, mode, order):
        from . import test_ce_qualification_accounting as qa
        context, proof, receipt = qa.fixture(root)
        owner = qa.a.QualificationAccounting(context, None, proof, receipt)
        path = root / 'retained.json'; blob = b'{"fixture":true}\n'; path.write_bytes(blob)
        ref = {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}
        for stage in qa.a.q.STAGES[1:qa.a.q.STAGES.index('checkpoint_resume')]:
            owner.begin_stage(stage, stage); owner.finish_stage([ref])
        owner.begin_stage('checkpoint_resume', 'lifetime')
        class Lease:
            def __enter__(self):
                order.append('enter'); owner.operation('qualification_lease', 'started', 'enter')
                if mode == 'enter':
                    owner.operation('qualification_lease', 'failed', 'enter'); raise ValueError('enter failed')
                owner.operation('qualification_lease', 'returned', 'enter'); return self
            def __exit__(self, typ, error, tb):
                order.append('exit')
                # Actual Q semantic guard requires open worker AND active stage.
                owner.operation('qualification_lease', 'started', 'exit')
                if mode == 'exit':
                    owner.operation('qualification_lease', 'failed', 'exit'); raise RuntimeError('exit failed')
                owner.operation('qualification_lease', 'returned', 'exit'); return mode == 'suppress'
        def cleanup(): order.append('cleanup')
        def finish(primary, result):
            order.append('finish')
            self.assertTrue(owner.worker_open); self.assertIsNotNone(owner.active)
            if primary is not None: owner.fail_stage(primary)
            else: owner.finish_stage([ref])
        return owner, Lease, cleanup, finish

    def test_actual_q_lease_exit_is_priced_before_stage_finish_and_worker_end(self):
        with tempfile.TemporaryDirectory(prefix='checkpoint-lifetime-pure-') as directory:
            order = []; owner, lease, cleanup, finish = self.lifetime_fixture(Path(directory), 'success', order)
            def body(): order.append('body'); return {'fixture': True}
            self.assertEqual(w.run_child_lifetime(owner, lease, body, cleanup, finish), {'fixture': True})
            self.assertEqual(order, ['enter', 'body', 'cleanup', 'exit', 'finish'])
            self.assertFalse(owner.worker_open); self.assertIsNone(owner.active)
            self.assertEqual(owner.operations['exit']['phase'], 'returned')
            exit_seq = owner.operations['exit']['returned_seq']
            self.assertLess(exit_seq, next(row['seq'] for row in owner.events if row['kind'] == 'finish' and row['payload']['stage'] == 'checkpoint_resume'))
            self.assertEqual(owner.events[-1]['kind'], 'end')

    def test_enter_exit_or_suppressed_body_failure_retains_original_and_closes_q_clock(self):
        for mode in ('enter', 'exit', 'suppress'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix='checkpoint-lifetime-failure-pure-') as directory:
                order = []; owner, lease, cleanup, finish = self.lifetime_fixture(Path(directory), mode, order)
                original = ValueError('body failed')
                def body(): order.append('body'); raise original
                with self.assertRaises(ValueError) as caught:
                    w.run_child_lifetime(owner, lease, body, cleanup, finish)
                if mode == 'enter': self.assertEqual(str(caught.exception), 'enter failed'); self.assertNotIn('cleanup', order)
                else: self.assertIs(caught.exception, original)
                self.assertFalse(owner.worker_open); self.assertIsNone(owner.active)
                self.assertEqual(owner.attempts[-1]['status'], 'FAILED')
                self.assertEqual(owner.events[-1]['kind'], 'end')
                if mode == 'exit': self.assertEqual(original.context_exit_failure['message'], 'exit failed')


if __name__ == '__main__': unittest.main()
