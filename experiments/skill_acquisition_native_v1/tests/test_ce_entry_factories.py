"""Full synthetic TRAIN/teacher chains only; no native/tokenizer/framework run."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_entry_factories as f
from . import test_ce_ingestion as source_fixture


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_fixture.PureTests.setUpClass.__func__(cls)
        helper = source_fixture.PureTests()
        for name in ('root', 'dataset', 'out', 'target_records', 'preparation', 'prep_ref', 'plan_refs'):
            setattr(helper, name, getattr(cls, name))
        helper.setUp()
        prepared = f.ingestion.PreparedRows(helper.verify(), 'fixture', outer_admission=helper.admit,
            operation=helper.operation, gate=helper.gates.append, renderer=source_fixture.fake_render)
        cls.common = {'mode': 'synthetic_fixture', 'prepared_receipt': prepared.receipt}
    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()
    def setUp(self):
        self.helper = source_fixture.PureTests()
        for name in ('root', 'dataset', 'out', 'target_records', 'preparation', 'prep_ref', 'plan_refs'):
            setattr(self.helper, name, getattr(self, name))
        self.helper.setUp()
        self.request = {'arm': 'canonical', 'preparation_ref': self.prep_ref, 'plan_ref': self.plan_refs['canonical']}
    def prepare(self, common=None, operation=None):
        return f.fixture_prepare_entry_rows({'schema': f.FIXTURE, 'native_qualified': False, 'execution_verified': False},
            self.common if common is None else common, self.request, self.helper.admit,
            self.helper.operation if operation is None else operation, self.helper.gates.append,
            tokenizer='fixture', renderer=source_fixture.fake_render)

    def test_closed_native_factory_denies_before_flags_callbacks_paths_or_framework(self):
        class Trap:
            def __getattribute__(self, key): raise AssertionError('unapproved entry input traversal')
            def __call__(self, *args, **kwargs): raise AssertionError('unapproved entry callback')
        before = set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'}
        with patch.object(Path, 'open', side_effect=AssertionError('unapproved path')):
            with self.assertRaisesRegex(ValueError, 'metadata verifier remains unregistered'):
                f.prepare_checkpoint_entry_rows(Trap(), True, Trap(), Trap(), {'approved': True},
                    request=Trap(), service_spec=Trap(), worker_handle=Trap(), observation_permit=Trap())
        self.assertFalse(f.controller._REGISTERED_OUTER_AUTHORITY_VERIFIERS)
        self.assertFalse(f.services._REGISTERED_ROLE_FACTORIES)
        self.assertEqual(before, set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'})

    def test_actual_3000_source_teacher_census_receipt_reproduced_without_model_grant(self):
        prepared = self.prepare()
        self.assertIs(type(prepared), f.ingestion.PreparedRows)
        self.assertEqual(prepared.receipt, self.common['prepared_receipt'])
        self.assertEqual(prepared.census['rows'], 3000)
        self.assertEqual(len(prepared.receipt['prepared_row_bindings']), 3000)
        self.assertEqual(prepared.cached_rows, 0)
        rows = prepared.iter_boundary(0)
        self.assertEqual(len(rows), 8); self.assertEqual(prepared.cached_rows, 8)
        self.assertEqual(prepared.receipt['model_loads'], 0); self.assertFalse(prepared.receipt['native_qualified'])

    def test_retained_receipt_source_tokenizer_census_or_row_hash_mismatch_refuses(self):
        # These are complete receipt differences, not caller PASS flags.
        for field in ('source_receipt_sha256', 'tokenizer_receipt_sha256', 'census', 'prepared_row_bindings'):
            common = copy.deepcopy(self.common)
            if field == 'census': common['prepared_receipt'][field]['total_prompt_tokens'] += 1
            elif field == 'prepared_row_bindings': common['prepared_receipt'][field][0]['prompt_ids_sha256'] = '0' * 64
            else: common['prepared_receipt'][field] = '0' * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'differs from retained entry receipt'):
                self.prepare(common)

    def test_paid_read_denial_precedes_source_bytes_and_cannot_be_overridden_by_metadata_flags(self):
        def deny(kind, phase, **fields):
            if phase == 'started': raise ValueError('source price denied')
        with patch.object(Path, 'open', side_effect=AssertionError('unpaid source bytes')):
            with self.assertRaisesRegex(ValueError, 'source price denied'): self.prepare(operation=deny)
        self.assertEqual([stage for stage, _ in self.helper.admissions], ['ingestion_sources'])

    def test_boolean_metadata_authority_cannot_replace_fixed_source_outer_wrapper(self):
        for value in (True, {'approved': True}, object()):
            with self.subTest(kind=type(value).__name__), self.assertRaisesRegex(ValueError, 'exact entry metadata authority'):
                f._verify_entry_outer_authority(value, 'a' * 64, 'b' * 64, 'qualification')


if __name__ == '__main__': unittest.main()
