"""Closed production adapters and real metadata bytes; no model/OS execution."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_training_runtime as t


class Trap:
    def __getattribute__(self, name): raise AssertionError('unadmitted runtime access')


class Tests(unittest.TestCase):
    def test_public_native_entries_deny_before_models_paths_or_lease(self):
        before = set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'}
        calls = (lambda: t.create_training_backend(Trap()),
                 lambda: t.publication_role(Trap(), Trap(), Trap(), Trap(), Trap()),
                 lambda: t.completion_seal_role(Trap(), Trap(), Trap(), Trap(), Trap()),
                 lambda: t.outer_verifier(Trap(), Trap(), Trap(), Trap()),
                 lambda: t.bootstrap_outer(Trap(), Trap(), Trap()))
        with patch.object(Path, 'open', side_effect=AssertionError('unpaid runtime IO')):
            for call in calls:
                with self.assertRaises(ValueError): call()
            with self.assertRaises(ValueError):
                with t.training_lease(Trap(), Trap()): raise AssertionError('unadmitted lease body')
        self.assertEqual(set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'}, before)

    def test_qualification_or_unverified_pair_cannot_use_training_factory(self):
        context = {'phase': 'qualification', 'qualifiedreceipt_verified': True, 'production_training_authorized': True,
                   'admission_evidence_verified': True, 'bindings': {'scope': {'schema': t.admission.SCOPE, 'role': 'production_training'}}}
        with self.assertRaisesRegex(ValueError, 'separate source production'): t._training(context)
        context['phase'] = 'training'; context['qualifiedreceipt_verified'] = False
        with self.assertRaises(ValueError): t._training(context)

    def completion(self):
        snapshot = {'progress': {'C': 375, 'U': 375}, 'synthetic_cpu': False, 'fatal': False,
            'unknown_wall': False, 'requires_restore': False, 'active_attempt': None, 'pending_operations': 0}
        context = {'bindings': {'anchors': {'common_anchor_sha256': 'a' * 64, 'own_start_sha256': 'a' * 64}}}
        donor = {'schema': 'SKILL-ACQUISITION-DONOR-DELTA-1', 'common_anchor_sha256': 'a' * 64,
            'own_start_sha256': 'a' * 64, 'reference': 'EXACT_REBUILT_COMMON_WARM_ANCHOR',
            **dict.fromkeys(('file_sha256', 'data_section_sha256', 'state_sha256', 'final_master_sha256'), 'b' * 64)}
        return snapshot, donor, context

    def test_completion_requires_full_dose_cleared_cost_state_and_common_anchor(self):
        snapshot, donor, context = self.completion()
        self.assertIs(t.validate_completion(snapshot, donor, context), snapshot)
        for change in (lambda value: value['progress'].update(C=374), lambda value: value['progress'].update(U=True),
                       lambda value: value.update(pending_operations=1), lambda value: value.update(unknown_wall=True),
                       lambda value: value.update(requires_restore=True), lambda value: value.update(active_attempt='failed-step')):
            wrong = copy.deepcopy(snapshot); change(wrong)
            with self.assertRaises(ValueError): t.validate_completion(wrong, donor, context)
        wrong = dict(donor, common_anchor_sha256='f' * 64)
        with self.assertRaisesRegex(ValueError, 'common warm'): t.validate_completion(snapshot, wrong, context)

    def test_streamed_readback_rejects_changed_bytes_and_bound_before_open(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'delta.bin'; content = b'actual-finite-payload'; path.write_bytes(content)
            expected = hashlib.sha256(content).hexdigest()
            ref = t._stream_reference(path, len(content), expected)
            self.assertEqual(ref['bytes'], len(content)); self.assertEqual(ref['sha256'], expected)
            path.write_bytes(content[:-1] + b'x')
            with self.assertRaisesRegex(ValueError, 'bytes/hash changed'): t._stream_reference(path, len(content), expected)
            with patch.object(Path, 'open', side_effect=AssertionError('over-limit bytes read')):
                with self.assertRaisesRegex(ValueError, 'bounded retained'): t._stream_reference(path, len(content) - 1)

    def test_native_journal_resource_probe_keeps_original_bootstrap_owner(self):
        calls = []; bootstrap = SimpleNamespace(operation=lambda kind, phase, **fields: calls.append((kind, phase)))
        dispatch = t.services.StableDispatcher(bootstrap)
        journal = object.__new__(t.ce_journal.CeJournal); journal.synthetic_cpu = False
        journal.operation = lambda *args, **fields: (_ for _ in ()).throw(AssertionError('resource kind sent to CE kernel ledger'))
        dispatch.ledger = journal
        for index, kind in enumerate(('resource_probe', 'resource_cuda_probe')):
            identifier = 'probe-%d' % index
            dispatch(kind, 'started', identifier)
            dispatch.ledger = None
            dispatch(kind, 'returned', identifier)
            dispatch.ledger = journal
        self.assertEqual(calls, [('resource_probe', 'started'), ('resource_probe', 'returned'),
                                 ('resource_cuda_probe', 'started'), ('resource_cuda_probe', 'returned')])


if __name__ == '__main__': unittest.main()
