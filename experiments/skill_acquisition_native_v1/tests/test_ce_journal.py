"""Pure durable CE controls. All bounds/data are illustrative synthetic fixtures."""
import copy
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_journal as cj, ce_plan, ce_update
from .test_ce_update import recipe, rows


def fixture():
    prepared = []
    for boundary in range(2):
        for i, row in enumerate(rows()):
            row['id'] = 'fixture-%02d' % (boundary * 8 + i)
            row['target_sha256'] = cj.digest(row['completion_ids'])
            prepared.append(row)
    plan = ce_plan.build_synthetic([{'id': r['id'], 'target_sha256': r['target_sha256']} for r in prepared])
    index = {r['id']: r for r in prepared}
    prepared = [index[rid] for boundary in plan['ordered_plan'] for rid in boundary['row_ids']]
    static = recipe()
    for key in ('boundary_id', 'generation', 'symbolic_snapshot', 'start_policy_id', 'U_before'): static.pop(key)
    bounds = {'read_bytes_upper': 10 ** 9, 'write_bytes_upper': 10 ** 9,
              'identity_hash_attempts_upper': 100, 'identity_hash_bytes_upper': 10 ** 9, 'wall_s_upper': 10.0}
    bindings = {'scope': cj.SCOPE, 'recipe': static,
        'anchors': {'own_start_sha256': 'a' * 64, 'common_anchor_sha256': 'a' * 64},
        'source': {'synthetic-fixture': 'c' * 64},
        'runtime': {'device': 'cpu', 'dtype': 'float32', 'cuda_initialized': False},
        'prepared_rows': prepared, 'operation_bounds': {kind: dict(bounds) for kind in cj.KINDS}, 'setup_bounds': dict(bounds)}
    bindings['setup_bounds']['identity_hash_attempts_upper'] = 10000
    caps = dict.fromkeys(cj.COUNTERS, 10 ** 12)
    caps.update(schema=cj.CAP_SCHEMA, synthetic_cpu=True, wall_seconds=10 ** 6, resumes=8,
                operation_starts={kind: 10000 for kind in cj.KINDS})
    return plan, bindings, caps


class Controls(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'ledger'
        self.plan, self.bindings, self.caps = fixture()
    def make(self, resume=False):
        return cj.CeJournal(self.root, self.plan, self.bindings, self.caps,
                            synthetic_cpu=True, resume=resume, clock=lambda: 1.0)
    def active(self):
        ledger = self.make(); ledger.start_invocation(boot_id='synthetic-boot'); return ledger
    def meta(self, ledger):
        return {'boundary_id': ledger.progress['C'], 'generation': ledger.generation,
                'symbolic_snapshot': ledger.symbolic_snapshot, 'start_policy_id': 'a' * 64,
                'static_recipe_sha256': ledger.static_recipe_sha256,
                'objective': 'SUPERVISED_EQUAL_ROW_TOKEN_MEAN_CE'}
    def pre(self, ledger):
        meta = dict(self.meta(ledger), reason='ce_pre_boundary_identity', state_bytes=100)
        ledger.operation('fingerprint', 'started', 'pre', **meta)
        ledger.operation('fingerprint', 'returned', 'pre', **dict(meta, policy_sha256='a' * 64, U=0, frozen_state_sha256='f' * 64))
    def successful_result(self, ledger):
        meta = self.meta(ledger); ordinal = 0
        def span(kind, fields, output=None):
            nonlocal ordinal
            oid = 'op-%d' % ordinal; ordinal += 1
            ledger.operation(kind, 'started', oid, **fields)
            record = dict(fields, **(output or {}), wall_s=0.0)
            ledger.operation(kind, 'returned', oid, **record)
            return dict(record, kind=kind, operation_id=oid)
        pre = span('fingerprint', dict(meta, reason='ce_pre_boundary_identity', state_bytes=100),
                   {'policy_sha256': 'a' * 64, 'U': 0, 'frozen_state_sha256': 'f' * 64})
        span('checkpoint_configuration', meta); span('gradient_reset', meta)
        result_rows = []
        for rid in ledger.plan_rows[0]:
            row = ledger.census[rid]
            fields = dict(meta, prompt_id=rid, selected_tokens=row['labeled_tokens'], full_sequence_tokens=row['full_sequence_tokens'],
                **{k: row[k] for k in ('prompt_ids_sha256', 'completion_ids_sha256', 'prepared_row_sha256')})
            forward = span('loss_forward', fields); backward = span('backward', fields)
            result_rows.append(dict(row, forward_operation=forward, backward_operation=backward,
                selected_predictor_positions=list(range(row['prompt_tokens'] - 1, row['full_sequence_tokens'] - 1)),
                target_positions=list(range(row['prompt_tokens'], row['full_sequence_tokens'])),
                eos_positions_selected=1, prompt_positions_selected=0, pad_positions_selected=0))
        validation = span('gradient_validation', meta, {'finite_confined': True, 'raw_gradient_norm_float64': .5})
        clipping = span('gradient_clipping', dict(meta, **self.bindings['recipe']['gradient_clipping']),
                        {'clipped_gradient_norm_float64': .5})
        step = span('optimizer', dict(meta, U_before=0, expected_U_after=1))
        post = span('fingerprint', dict(meta, reason='ce_post_boundary_identity', state_bytes=100),
                    {'policy_sha256': 'b' * 64, 'U': 1, 'frozen_state_sha256': 'f' * 64})
        span('gradient_cleanup', dict(meta, optimizer_attempted=True))
        return dict(meta, schema='SKILL-ACQUISITION-CE-BOUNDARY-RESULT-1', synthetic_cpu=True,
            loss_reduction=ce_update.REDUCTION, accumulation_rows=8, U_before=0, U_after=1, updated=True, requires_restore=False,
            rows=result_rows, identity_operation=pre, final_identity_operation=post, final_policy_id='b' * 64,
            frozen_state_sha256='f' * 64, optimizer_operation=step, gradient_validation_operation=validation,
            clipping_operation=clipping, raw_gradient_norm_float64=.5, clipped_gradient_norm_float64=.5,
            optimizer_recipe=self.bindings['recipe']['optimizer'], gradient_clipping_recipe=self.bindings['recipe']['gradient_clipping'])
    def test_native_proposal_denied_before_path_and_no_framework(self):
        framework_names = ('torch', 'numpy', 'transformers', 'safetensors')
        before = {name for name in framework_names if name in sys.modules}
        class PathTrap:
            def __fspath__(self): raise AssertionError('path touched')
        for flag in (False, 1, None):
            with self.assertRaisesRegex(ValueError, 'scope denied'):
                cj.CeJournal(PathTrap(), self.plan, {}, {}, synthetic_cpu=flag)
        bad = dict(self.plan, mode='unapproved_proposal')
        with self.assertRaisesRegex(ValueError, 'scope denied'):
            cj.CeJournal(PathTrap(), bad, {}, {}, synthetic_cpu=True)
        self.assertEqual({name for name in framework_names if name in sys.modules}, before)
        tree = ast.parse(Path(cj.__file__).read_text())
        top = [alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
        self.assertFalse(set(top).intersection(framework_names))
    def test_create_only_binding_drift_and_corrupt_tail_fail_closed(self):
        ledger = self.active(); ledger.end_invocation('INCOMPLETE')
        with self.assertRaisesRegex(ValueError, 'create-only'): self.make()
        saved = copy.deepcopy(self.bindings); self.bindings['source']['synthetic-fixture'] = 'd' * 64
        with self.assertRaisesRegex(ValueError, 'binding drift'): self.make(True)
        self.bindings = saved
        original = ledger.path.read_bytes(); ledger.path.write_bytes(original + b'{partial')
        with self.assertRaisesRegex(ValueError, 'partial/corrupt'): self.make(True)
        self.assertEqual(ledger.path.read_bytes(), original + b'{partial')
    def test_setup_combined_cap_denies_before_any_files_and_oversized_header_before_read(self):
        self.caps['io_write_bytes'] = 1
        with self.assertRaisesRegex(ValueError, 'ceiling'): self.make()
        self.assertFalse(self.root.exists())
        self.caps['io_write_bytes'] = 10 ** 12
        ledger = self.make(); ledger.header_path.write_bytes(b'x' * (cj.MAX_LINE + 1))
        with patch.object(Path, 'open', side_effect=AssertionError('read forbidden')):
            with self.assertRaisesRegex(ValueError, 'bounded resume'): self.make(True)
    def test_frozen_same_length_prompt_target_mismatch_denied_before_forward(self):
        ledger = self.active(); self.pre(ledger)
        row = ledger.census[ledger.plan_rows[0][0]]
        meta = dict(self.meta(ledger), prompt_id=row['id'], selected_tokens=row['labeled_tokens'],
                    full_sequence_tokens=row['full_sequence_tokens'], prompt_ids_sha256=row['prompt_ids_sha256'],
                    completion_ids_sha256='0' * 64, prepared_row_sha256=row['prepared_row_sha256'])
        count = len(ledger.events)
        with self.assertRaisesRegex(ValueError, 'content/token'): ledger.operation('loss_forward', 'started', 'wrong', **meta)
        self.assertEqual(ledger.starts['loss_forward'], 0); self.assertTrue(ledger.requires_restore)
        with self.assertRaisesRegex(ValueError, 'previous CE'): ledger.operation('fingerprint', 'started', 'pre2', **dict(self.meta(ledger), reason='ce_pre_boundary_identity', state_bytes=100))
        self.assertGreaterEqual(len(ledger.events), count)
    def test_metadata_exact_external_and_own_row_costs_retained_after_replay(self):
        ledger = self.active(); before = ledger.counters['io_write_bytes']
        ledger.reserve_metadata('checkpoint_json', 50, write_id='missing-file', path=ledger.directory / 'not-written.partial')
        self.assertGreaterEqual(ledger.counters['io_write_bytes'] - before, 50)
        self.assertFalse((ledger.directory / 'not-written.partial').exists())
        ledger.end_invocation('INCOMPLETE'); costs = dict(ledger.counters)
        reopened = self.make(True)
        self.assertTrue(reopened.requires_restore)
        self.assertTrue(all(reopened.counters[k] >= v for k, v in costs.items()))
        reopened.start_invocation(resume=True, boot_id='synthetic-boot')
        with self.assertRaisesRegex(ValueError, 'unique metadata'):
            reopened.reserve_metadata('checkpoint_json', 50, write_id='missing-file', path=ledger.directory / 'not-written.partial')
    def test_CP_fingerprint_extra_passes_priced_before_work(self):
        self.bindings['operation_bounds']['fingerprint']['identity_hash_attempts_upper'] = 2
        ledger = self.active()
        with self.assertRaisesRegex(ValueError, 'fingerprint profile'):
            ledger.operation('fingerprint', 'started', 'hash', state_bytes=100,
                             identity_hash_attempts_upper=2, identity_hash_bytes_upper=0, checkpoint_hash_scope=True)
        self.assertNotIn('hash', ledger.operations)
    def test_failed_and_pending_attempt_never_refunds_or_restarts_without_restore(self):
        ledger = self.active(); self.pre(ledger)
        ledger.operation('checkpoint_configuration', 'started', 'cfg', **self.meta(ledger))
        ledger.operation('checkpoint_configuration', 'failed', 'cfg', **self.meta(ledger))
        costs = dict(ledger.counters)
        self.assertTrue(ledger.requires_restore)
        self.assertEqual((ledger.progress['C'], ledger.progress['U']), (0, 0))
        self.assertTrue(all(ledger.counters[k] >= v for k, v in costs.items()))
        ledger.end_invocation('INCOMPLETE')
        reopened = self.make(True); reopened.start_invocation(resume=True, boot_id='synthetic-boot')
        with self.assertRaisesRegex(ValueError, 'previous CE'):
            reopened.operation('fingerprint', 'started', 'newpre', **dict(self.meta(reopened), reason='ce_pre_boundary_identity', state_bytes=100))
    def test_unknown_boot_wall_blocks_physical_work(self):
        ledger = self.active()
        reopened = self.make(True); reopened.start_invocation(resume=True, boot_id='different-boot')
        self.assertTrue(reopened.unknown_wall)
        with self.assertRaisesRegex(ValueError, 'resolved invocation'):
            reopened.operation('checkpoint_construct', 'started', 'cp')
    def test_input_plan_alias_does_not_mutate_admitted_order_or_header(self):
        ledger = self.make(); admitted = copy.deepcopy(ledger.plan_rows); header = ledger.header_path.read_bytes()
        self.plan['ordered_plan'][0]['row_ids'].reverse()
        self.assertEqual(ledger.plan_rows, admitted); self.assertEqual(ledger.header_path.read_bytes(), header)
    def test_false_is_not_boundary_or_generation_zero(self):
        ledger = self.active(); meta = dict(self.meta(ledger), boundary_id=False, reason='ce_pre_boundary_identity', state_bytes=100)
        with self.assertRaisesRegex(ValueError, 'boundary/policy/fence'):
            ledger.operation('fingerprint', 'started', 'bool-pre', **meta)
        self.assertNotIn('bool-pre', ledger.operations)
    def test_gradient_stage_denied_before_last_backward_work(self):
        ledger = self.active(); self.pre(ledger)
        with self.assertRaisesRegex(ValueError, 'all eight backwards'):
            ledger.operation('gradient_clipping', 'started', 'early-clip', **self.meta(ledger))
        self.assertNotIn('early-clip', ledger.operations); self.assertTrue(ledger.requires_restore)
    def test_complete_seal_denies_pending_restore_and_unknown_wall_without_closing_invocation(self):
        ledger = self.active()
        ledger.progress.update(C=len(ledger.plan_rows), U=len(ledger.plan_rows))
        for name in ('requires_restore', 'restore_pending', 'unknown_wall'):
            setattr(ledger, name, True)
            with self.assertRaisesRegex(ValueError, 'consistent CE'):
                ledger.end_invocation('COMPLETE')
            self.assertIsNotNone(ledger.current_invocation)
            setattr(ledger, name, False)
    def test_complete_observed_result_commits_once_and_replay_retains_progress(self):
        ledger = self.active(); result = self.successful_result(ledger)
        self.assertEqual((ledger.progress['C'], ledger.progress['U']), (0, 0))
        ledger.commit_boundary(result)
        self.assertEqual((ledger.progress['C'], ledger.progress['U']), (1, 1)); self.assertFalse(ledger.requires_restore)
        with self.assertRaisesRegex(ValueError, 'complete nonfailed'): ledger.commit_boundary(result)
        ledger.end_invocation('INCOMPLETE'); reopened = self.make(True)
        self.assertEqual(reopened.progress['history'], ledger.progress['history'])
        self.assertEqual((reopened.progress['C'], reopened.progress['U']), (1, 1))
        self.assertTrue(reopened.requires_restore)


if __name__ == '__main__': unittest.main()
