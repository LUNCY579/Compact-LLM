"""Tiny real-file finalizer replay fixtures; no native proof or model execution."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_qualified_receipt as consumer
from experiments.skill_acquisition_native_v1 import ce_finalizer_accounting as costs
from .test_ce_finalizer_accounting import Clock, spec


class Trap:
    def __getattribute__(self, name): raise AssertionError('work before source pin')
    def __call__(self, *args, **kwargs): raise AssertionError('work before source pin')


class Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name); self.clock = Clock(); self.spec = spec(self.root)
        self.selection = self.build_pair()

    def reference(self, path):
        blob = path.read_bytes()
        return {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}

    def write(self, name, body):
        path = self.root / name; path.write_bytes(consumer.canonical(body) + b'\n')
        return self.reference(path)

    def build_pair(self):
        raw = consumer.raw
        recipe = {'loss': 'fixture-equal-row-CE', 'optimizer': 'fixture-Adam'}
        recipe_sha = raw.digest(recipe)
        claim = copy.deepcopy(consumer.finalizer.CLAIM_SCOPE)
        scope = {'static_recipe_sha256': recipe_sha, 'qualification_claim_scope': claim,
                 'fixture_limits': {'rows': 2, 'model_loads': 0}}
        scope_ref = self.write('SCOPE.json', scope)
        # Use the real source owner's explicit non-native initializer with the
        # tiny fixture scope. Every retained header/event/seal byte is original.
        self.owner = object.__new__(costs.FinalizerCostOwner)
        self.owner._initialize(self.spec, self.root, 'a' * 64, scope_ref['sha256'], None,
                               native=False, clock=self.clock)
        row = lambda rid: {'id': rid, 'prompt_tokens': 3, 'labeled_tokens': 2, 'full_sequence_tokens': 5,
            'prompt_ids_sha256': raw.digest([1, 2, 3]), 'completion_ids_sha256': raw.digest([4, 5]),
            'prepared_row_sha256': raw.digest(rid), 'render_sha256': raw.digest(rid + '-render'),
            'target_sha256': raw.digest(rid + '-target')}
        rows = [row('r0'), row('r1')]
        census = {'rows': 2, 'total_prompt_tokens': 6, 'total_supervised_tokens': 4, 'total_sequence_positions': 10,
                  'max_prompt_tokens': 3, 'max_labeled_tokens': 2, 'max_sequence_positions': 5}
        plan = {'plan_sha256': 'c' * 64, 'ordered_plan': [{'row_ids': ['r0', 'r1']}],
                'label_binding': {'boundaries': [{'row_target_pairs': [[item['id'], item['target_sha256']] for item in rows]}]}}
        preparation_ref = {'path': str(self.root / 'unread-fixture-preparation.json'), 'sha256': '1' * 64, 'bytes': 1}
        plan_ref = {'path': str(self.root / 'unread-fixture-plan.json'), 'sha256': '2' * 64, 'bytes': 1}
        anchors = {'tokenizer_receipt_sha256': 'd' * 64, 'common_anchor_sha256': 'e' * 64}
        runtime = {'schema': 'explicit-synthetic-runtime', 'cuda_initialized': False}
        prepared_body = {'schema': 'CE8-INGESTION-1', 'tokenizer_rendered': True, 'model_loads': 0, 'optimizer_steps': 0,
            'native_qualified': False, 'training_authorized': False, 'arm': 'canonical', 'plan_sha256': plan['plan_sha256'],
            'static_recipe_sha256': recipe_sha, 'tokenizer_receipt_sha256': anchors['tokenizer_receipt_sha256'],
            'preparation_ref': preparation_ref, 'plan_ref': plan_ref, 'census': census, 'prepared_row_bindings': rows,
            'witnesses': {key: {'value': census[key], 'row_ids': ['r0', 'r1']} for key in
                          ('max_prompt_tokens', 'max_labeled_tokens', 'max_sequence_positions')}}
        prepared = dict(prepared_body, receipt_sha256=raw.digest(prepared_body))
        prepared_ref = self.write('PREPARED.json', prepared)
        binding = {'stage': 'qualification_finalizer', 'request_sha256': 'a' * 64, 'scope_sha256': scope_ref['sha256'],
            'static_recipe_sha256': recipe_sha, 'code_sha256': {'fixture-only.py': 'f' * 64},
            'worker_identity_sha256': '9' * 64, 'attempt_id': 'independent-post-exit-finalization'}
        context = {'request_sha256': 'a' * 64, 'scope_sha256': scope_ref['sha256'], 'static_recipe_sha256': recipe_sha,
            'arm': 'canonical', 'refs': {'preparation_ref': preparation_ref, 'plan_ref': plan_ref},
            'bindings': {'scope': scope, 'source': {'code_sha256': binding['code_sha256']}, 'runtime': runtime,
                'anchors': anchors, 'recipe': {'recipe': recipe}, 'plan': plan, 'ingestion': {'expected_census': census}}}
        context_ref = self.write('CONTEXT.json', context)
        proof = {'request_sha256': 'a' * 64, 'scope_sha256': scope_ref['sha256'], 'worker_identity_sha256': '9' * 64,
                 'exit_code': 0, 'status_file_used': False, 'fixture_only': True}
        # Retain a successful synthetic metadata observation using the actual
        # permanent owner operation method. No OS handle or native grant exists.
        self.clock.advance(1)
        self.owner.operation('qualification_artifact_readback', 'started', 'bounded-read', io_read_bytes_upper=17,
                             identity_hash_attempts_upper=1, identity_hash_bytes_upper=17)
        self.owner.operation('qualification_artifact_readback', 'returned', 'bounded-read', bytes_read=3)
        self.owner.operation('finalizer_terminal_observation', 'started', 'terminal')
        self.owner.operation('finalizer_terminal_observation', 'returned', 'terminal', terminal_proof=proof,
                             observed_before=101., observed_after=101.)
        preclose = {'schema': consumer.finalizer.SCHEMA, 'request_sha256': 'a' * 64, 'scope_sha256': scope_ref['sha256'],
            'static_recipe_sha256': recipe_sha, 'arm': 'canonical', 'qualification_gaps': [],
            'qualification_claim_scope': claim, 'raw_numeric_reductions': {stage: {'synthetic_fixture': True}
                for stage in consumer.qualification.STAGES[1:]}, 'parent_terminal_proof': proof,
            'evidence_refs': {'context': context_ref, 'prepared': prepared_ref}, 'native_qualified': False, 'execution_verified': False}
        preclose_ref = self.owner.write_json('PRE_CLOSE_QUALIFICATION_REPLAY.json', preclose)
        snapshot = {'clock_ended': True, 'fatal': False, 'pending': [], 'request_sha256': 'a' * 64,
                    'last_monotonic': 101., 'end_reservation': {'bounds': {'wall_s_upper': 120}}}
        self.owner.operation('finalizer_service_end', 'started', 'service-end')
        self.owner.operation('finalizer_service_end', 'returned', 'service-end', observed_before=101., observed_after=101.,
                             evidence={'snapshot': snapshot})
        receipt_body = {'schema': consumer.qualification.RECEIPT, 'phase': 'qualification',
            'status': 'PROOF_COMPLETE_REQUIRES_FINALIZER_SEAL', 'passed': True, 'native_actual': True,
            'native_qualified': True, 'execution_verified': True, 'production_training_authorized': False,
            'request_sha256': 'a' * 64, 'scope_sha256': scope_ref['sha256'], 'static_recipe_sha256': recipe_sha,
            'arm': 'canonical', 'preclose_replay_ref': preclose_ref, 'finalizer_header_ref': self.owner.header_ref,
            'expected_finalizer_seal_path': str(self.owner.seal_path), 'cost_owner_binding_sha256': raw.digest(self.owner.header),
            'qualification_claim_scope': claim, 'qualification_gaps': [],
            'supervisor_service_replay': {'recorded_clock_ended': True, 'derived_snapshot': snapshot},
            'supervisor_terminal_suffix': {'recorded_end': 101., 'observed_after': 101., 'suffix_upper_seconds': 120,
                'observed_suffix_seconds_upper': 0., 'native_observation': True}}
        receipt = dict(receipt_body, receipt_sha256=raw.digest(receipt_body))
        receipt_ref = self.owner.write_json('QUALIFICATION_RECEIPT.json', receipt)
        final_costs = self.owner.close_and_seal()
        joint = {'receipt_ref': receipt_ref, 'proof_complete': True, 'native_qualified': True,
                 'completion_pending_cost_seal': False, 'qualification_gaps': [], 'status': 'NATIVE_QUALIFIED',
                 'passed': True, 'finalizer_cost_evidence': final_costs}
        completion_ref = self.write('JOINT_COMPLETION.json', joint)
        return {'schema': consumer.SCHEMA, 'owned_root': str(self.root), 'arm': 'canonical', 'binding': binding,
            'references': {'completion': completion_ref, 'receipt': receipt_ref, 'preclose': preclose_ref,
                'header': final_costs['header_ref'], 'journal': final_costs['journal_ref'], 'seal': final_costs['seal_ref'],
                'context': context_ref, 'prepared': prepared_ref, 'scope': scope_ref}, 'finalizer_specification': self.spec,
            'claim_scope': claim, 'runtime_sha256': raw.digest(runtime), 'anchors_sha256': raw.digest(anchors),
            'prepared_receipt_sha256': prepared['receipt_sha256']}

    def reader(self, observer=None):
        bounds = {key: 16 * 1024 ** 3 for key in consumer.raw.BUDGET_FIELDS}
        return consumer.raw.RawEvidenceReader(observer or (lambda *args, **kwargs: None), bounds, self.root)

    def verify(self, selection=None):
        return consumer.verify_fixture_pair(selection or self.selection, reader=self.reader(), outer_preflight=lambda *args: None)

    def replace_json(self, selection, role, mutate):
        ref = selection['references'][role]; path = Path(ref['path'])
        body = consumer.raw._json(path.read_bytes()); mutate(body)
        path.write_bytes(consumer.canonical(body) + b'\n'); selection['references'][role] = self.reference(path)
        return body

    def test_actual_small_owner_files_replay_without_native_authority(self):
        result = self.verify()
        self.assertFalse(result['completed_pair_verified']); self.assertTrue(result['synthetic_pair_consistent'])
        self.assertFalse(result['execution_reverified']); self.assertFalse(result['production_training_authorized'])
        self.assertEqual(result['census']['rows'], 2); self.assertFalse(result['unit_prices_measured'])
        self.assertEqual(result['derived_finalizer_costs']['derived_consumed_bounds'], self.owner.counts)
        self.assertNotIn('torch', sys.modules); self.assertNotIn('transformers', sys.modules)

    def test_source_absent_arbitrary_flags_and_fixture_native_header_refuse(self):
        with patch.object(Path, 'open', side_effect=AssertionError('IO before exact source pin')):
            with self.assertRaisesRegex(ValueError, 'consumer closed'): consumer.selected_qualification()
            for value in (True, {'native_qualified': True}, MappingProxyType(self.selection)):
                with self.assertRaisesRegex(ValueError, 'consumer closed'):
                    consumer.verify_completed_qualification(value, reader=Trap(), outer_preflight=Trap())
        pinned = MappingProxyType(self.selection)
        with (patch.object(consumer, '_SOURCE_SELECTION', pinned),
              patch.object(consumer, '_SOURCE_SELECTION_SHA256', consumer.digest(self.selection))):
            with self.assertRaisesRegex(ValueError, 'original native finalizer header'):
                consumer.verify_completed_qualification(pinned, reader=self.reader(), outer_preflight=lambda *args: None)
            self.selection['binding']['code_sha256']['fixture-only.py']='8'*64
            with self.assertRaisesRegex(ValueError, 'source selection changed'):
                consumer.verify_completed_qualification(pinned, reader=self.reader(), outer_preflight=lambda *args: None)

    def test_preflight_and_read_reservation_denial_precede_file_access(self):
        def deny(*args, **kwargs): raise RuntimeError('read payment denied')
        with patch.object(Path, 'open', side_effect=AssertionError('IO before payment')):
            with self.assertRaisesRegex(RuntimeError, 'read payment denied'):
                consumer.verify_fixture_pair(self.selection, reader=self.reader(), outer_preflight=deny)
            with self.assertRaisesRegex(RuntimeError, 'read payment denied'):
                consumer.verify_fixture_pair(self.selection, reader=self.reader(deny), outer_preflight=lambda *args: None)

    def test_rehashed_joint_refund_does_not_pass(self):
        selected = copy.deepcopy(self.selection)
        self.replace_json(selected, 'completion', lambda joint: joint['finalizer_cost_evidence']['consumed_declared_bounds'].__setitem__(
            'io_read_bytes_upper', joint['finalizer_cost_evidence']['consumed_declared_bounds']['io_read_bytes_upper'] - 1))
        with self.assertRaisesRegex(ValueError, 'costs changed or refunded'): self.verify(selected)

    def test_rehashed_finalizer_clock_interval_does_not_pass(self):
        selected = copy.deepcopy(self.selection); path = Path(selected['references']['journal']['path'])
        rows = [consumer.raw._json(line) for line in path.read_bytes().splitlines()]
        wall = next(row for row in rows if row['kind'] == 'wall'); wall['payload']['elapsed_delta'] += 1
        previous = None
        for row in rows:
            row['previous'] = previous; row['sha256'] = consumer.digest({key: value for key, value in row.items() if key != 'sha256'})
            previous = row['sha256']
        path.write_bytes(b''.join(consumer.canonical(row) + b'\n' for row in rows))
        selected['references']['journal'] = self.reference(path)
        self.replace_json(selected, 'completion', lambda joint: joint['finalizer_cost_evidence'].__setitem__('journal_ref', selected['references']['journal']))
        with self.assertRaisesRegex(ValueError, 'wall interval'): self.verify(selected)

    def test_missing_seal_changed_raw_receipt_and_unqualified_joint_refuse(self):
        path = Path(self.selection['references']['seal']['path']); blob = path.read_bytes(); path.unlink()
        with self.assertRaisesRegex(ValueError, 'path/size/ownership'): self.verify()
        path.write_bytes(blob)
        receipt_path = Path(self.selection['references']['receipt']['path']); original = receipt_path.read_bytes()
        receipt_path.write_bytes(original.replace(b'"passed":true', b'"passed":false'))
        with self.assertRaisesRegex(ValueError, 'path/size/ownership|bytes/hash mismatch'): self.verify()
        receipt_path.write_bytes(original)
        selected = copy.deepcopy(self.selection)
        self.replace_json(selected, 'completion', lambda joint: joint.__setitem__('native_qualified', False))
        with self.assertRaisesRegex(ValueError, 'completed joint qualification'): self.verify(selected)

    def test_pending_rehashed_seal_changed_caps_and_foreign_source_refuse(self):
        selected = copy.deepcopy(self.selection)
        seal_path = Path(selected['references']['seal']['path']); original = seal_path.read_bytes()
        completion_path = Path(selected['references']['completion']['path']); completion_original = completion_path.read_bytes()
        self.replace_json(selected, 'seal', lambda seal: seal['pending'].append('unfinished'))
        self.replace_json(selected, 'completion', lambda joint: joint['finalizer_cost_evidence'].__setitem__('seal_ref', selected['references']['seal']))
        with self.assertRaisesRegex(ValueError, 'seal differs'): self.verify(selected)
        seal_path.write_bytes(original)
        completion_path.write_bytes(completion_original)
        changed = copy.deepcopy(self.selection); changed['finalizer_specification']['self_bounds']['io_read_bytes_upper'] += 1
        with self.assertRaisesRegex(ValueError, 'header/source scope'): self.verify(changed)
        changed = copy.deepcopy(self.selection); changed['binding']['code_sha256']['fixture-only.py'] = '8' * 64
        with self.assertRaisesRegex(ValueError, 'preclose source/runtime/anchor'): self.verify(changed)


if __name__ == '__main__': unittest.main()
