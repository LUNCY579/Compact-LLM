"""Metadata fixtures and failure controls; no native execution authority."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_native_services as s
from experiments.skill_acquisition_native_v1 import ce_service_prepayment as p
from .test_ce_native_services import Trap, bootstrap_spec


def configuration(root):
    request = {'scope_authorization_ref': {}, 'fixture': 'metadata-only'}
    descriptor = {'module': p.__name__, 'symbol': 'bootstrap_constructor_prepayment', 'source_sha256': 'a' * 64}
    bounds = dict.fromkeys(s.COUNTERS, 1000000); bounds['operation_attempts'] = 4; bounds['hash_bytes'] = 256 * 1024 ** 2
    spec = {'bootstrap': bootstrap_spec(root),
            'constructor_prepayment': {'schema': p.SCHEMA, 'owned_directory': str(root),
                'bounds': bounds, 'source_verifier_bounds': {'hash_attempts': 5, 'hash_bytes': 32 * 1024 ** 2}},
            'role_descriptors': {'bootstrap_constructor_prepayment': descriptor},
            'approved_code_sha256': {p.__name__.replace('.', '/') + '.py': 'a' * 64}}
    return request, spec


class ConstructorPaymentTests(unittest.TestCase):
    def test_unregistered_factory_denies_before_paths(self):
        with self.assertRaises(s.worker.WorkerEvidenceError):
            p.bootstrap_constructor_prepayment(Trap(), Trap(), None)
        self.assertFalse(s._REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES)

    def test_permanent_one_shot_reservation_and_exact_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'paid'; request, spec = configuration(root)
            # Isolate concrete IO/reservation mechanics. The original native
            # service capability is deliberately mocked; no permit is minted.
            with patch.object(s, '_admit', return_value=None), \
                 patch.object(s, '_REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES', frozenset((p.ServiceConstructorPrepayment,))), \
                 patch.object(s, '_REGISTERED_ROLE_FACTORIES', frozenset((p.bootstrap_constructor_prepayment,))):
                paid = p.bootstrap_constructor_prepayment(request, spec, object())
                saved = json.loads(paid.path.read_bytes())
                self.assertEqual(saved['status'], 'PERMANENTLY_RESERVED')
                self.assertFalse(saved['execution_verified']); self.assertFalse(saved['native_qualified'])
                with self.assertRaisesRegex(s.worker.WorkerEvidenceError, 'one exact'):
                    paid.require_constructor('b' * 64, paid.request_sha256)
                self.assertFalse(paid.consumed)
                ref = paid.require_constructor(s.digest(spec['bootstrap']), paid.request_sha256)
                self.assertEqual(ref, paid.evidence['reservation_ref']); self.assertTrue(paid.consumed)
                with self.assertRaisesRegex(s.worker.WorkerEvidenceError, 'one exact'):
                    paid.require_constructor(s.digest(spec['bootstrap']), paid.request_sha256)
                with self.assertRaisesRegex(s.worker.WorkerEvidenceError, 'create-only'):
                    p.bootstrap_constructor_prepayment(request, spec, object())

    def test_insufficient_self_metadata_denies_before_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'uncreated'; request, spec = configuration(root)
            spec['constructor_prepayment']['bounds']['metadata_bytes'] = 1
            with patch.object(s, '_admit', return_value=None), \
                 patch.object(s, '_REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES', frozenset((p.ServiceConstructorPrepayment,))):
                with self.assertRaisesRegex(s.worker.WorkerEvidenceError, 'own metadata'):
                    p.ServiceConstructorPrepayment(request, spec, object())
            self.assertFalse(root.exists())

    def test_mutated_spec_cannot_reuse_original_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            request, spec = configuration(Path(tmp) / 'paid')
            with patch.object(s, '_admit', return_value=None), \
                 patch.object(s, '_REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES', frozenset((p.ServiceConstructorPrepayment,))):
                paid = p.ServiceConstructorPrepayment(request, spec, object())
                paid.spec['bootstrap']['setup_bounds']['read_bytes_upper'] += 1
                with self.assertRaisesRegex(s.worker.WorkerEvidenceError, 'one exact'):
                    paid.require_constructor(paid.bootstrap_spec_sha256, paid.request_sha256)

    def test_previous_self_only_write_envelope_cannot_pay_dependent_setup_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); request, spec = configuration(root / 'first')
            with patch.object(s, '_admit', return_value=None), \
                 patch.object(s, '_REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES', frozenset((p.ServiceConstructorPrepayment,))):
                paid = p.ServiceConstructorPrepayment(request, spec, object())
                request, underpaid = configuration(root / 'second')
                underpaid['constructor_prepayment']['bounds']['write_bytes'] = \
                    underpaid['bootstrap']['setup_bounds']['write_bytes_upper'] + paid.reference['bytes']
                with self.assertRaisesRegex(s.worker.WorkerEvidenceError, 'own metadata'):
                    p.ServiceConstructorPrepayment(request, underpaid, object())
                self.assertFalse((root / 'second').exists())
