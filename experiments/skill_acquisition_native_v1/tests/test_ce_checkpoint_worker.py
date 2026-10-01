"""Closed child CLI and pure frozen branch contracts; no kernel/process/ML."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_checkpoint_worker as w
from experiments.skill_acquisition_native_v1 import ce_qualification_artifacts as raw


class ChildContracts(unittest.TestCase):
    def setUp(self):
        batches = [['r%d' % i for i in range(8)], ['r%d' % i for i in range(8, 16)]]
        self.context = {'phase': 'qualification', 'request_sha256': '1' * 64, 'scope_sha256': '2' * 64,
            'static_recipe_sha256': '3' * 64, 'bindings': {'source': {'code_sha256': {'worker.py': '4' * 64}},
                'anchors': {'common_anchor_sha256': '5' * 64}, 'scope': {
                    'qualifier_execution': {'adam_batches': batches},
                    'qualifier_accounting': {'ingestion_receipt_sha256': '6' * 64}}}}

    def request(self, branch):
        batches = self.context['bindings']['scope']['qualifier_execution']['adam_batches']
        pred = {'schema': 'CE8-CHECKPOINT-CHILD-PREDECESSOR-1', 'branch': branch,
            'qualification_batches': batches, 'inherited_U': 1 if branch == 'resumed' else 0,
            'worker_identity_sha256': '7' * 64}
        if branch == 'resumed':
            pred.update(prefix_manifest_ref={'path': str(Path.cwd() / 'fixture-prefix.json'), 'sha256': '8' * 64, 'bytes': 1},
                prefix_checkpoint_bindings_sha256='9' * 64, inherited_history=[{'fixture': True}])
        return {'schema': w.REQUEST, 'branch': branch, 'plan_sha256': raw.digest({'qualification_batches': batches, 'branch': branch}),
            'request_sha256': self.context['request_sha256'], 'scope_sha256': self.context['scope_sha256'],
            'static_recipe_sha256': self.context['static_recipe_sha256'],
            'code_sha256': self.context['bindings']['source']['code_sha256'], 'common_anchor_sha256': '5' * 64,
            'ingestion_receipt_sha256': '6' * 64, 'worker_identity_sha256': '7' * 64, 'predecessor': pred}

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
        for key in ('request_sha256', 'scope_sha256', 'static_recipe_sha256', 'common_anchor_sha256', 'ingestion_receipt_sha256'):
            request = self.request('prefix'); request[key] = 'a' * 64
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'binding'):
                w.validate_request(request, self.context)
        request = self.request('prefix'); request['code_sha256'] = {'foreign.py': 'b' * 64}
        with self.assertRaisesRegex(ValueError, 'binding'): w.validate_request(request, self.context)

    def test_resumed_branch_requires_u1_history_and_original_binding_manifest(self):
        for change in (lambda p: p.update(inherited_U=True), lambda p: p.update(inherited_U=0),
                       lambda p: p.update(inherited_history=[]), lambda p: p.pop('prefix_manifest_ref'),
                       lambda p: p.update(prefix_checkpoint_bindings_sha256='not-a-hash')):
            request = self.request('resumed'); change(request['predecessor'])
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
        context = copy.deepcopy(self.context)
        context['bindings']['scope']['qualifier_execution']['adam_batches'] *= 3
        with self.assertRaises(ValueError): w.validate_request(self.request('prefix'), context)


if __name__ == '__main__': unittest.main()
