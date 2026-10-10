"""Isolated candidate integration fixtures; no actual native/model execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_operation_contracts as c
from experiments.skill_acquisition_native_v1 import ce_qualification_accounting as q
from experiments.skill_acquisition_native_v1 import ce_native_services as s
from experiments.skill_acquisition_native_v1 import ce_contract_registration as registration
from .test_ce_qualification_accounting import fixture
from .test_ce_service_bootstrap_verifier import specification, Clock


def manifest():
    # The deterministic source catalog is tracked with this package. External
    # proposal paths are historical evidence and supply no fixture authority.
    return registration.manifest()


def q_context(root):
    context, proof, receipt = fixture(root)
    spec = context['bindings']['scope']['qualifier_accounting']; spec['operation_contracts'] = manifest()
    spec['operation_profile_rule'] = q.EXPLICIT_RESERVATION_RULE_V2
    spec['explicit_reservation_kinds'] = list(q.EXPLICIT_RESERVATION_KINDS)
    for row in spec['operation_contracts']['contracts'].values():
        if 'qualification' in row['owners'] and row['kind'] in spec['operation_profiles']:
            profile = spec['operation_profiles'][row['kind']]
            profile['wall_s_upper'] = max(profile['wall_s_upper'], row['wall_s_upper'])
    # Exact candidate kind ceilings are part of the root's frozen manifest.
    for role in c.OPAQUE_ROLES:
        row = spec['operation_contracts']['contracts'][role]
        if 'qualification' in row['owners']: spec['operation_profiles'][row['kind']]['wall_s_upper'] = row['wall_s_upper']
    spec['operation_profiles']['fingerprint']['wall_s_upper'] = 240
    return context, proof, receipt


class ContractIntegration(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='candidate-contract-pure-')
        self.addCleanup(temporary.cleanup); self.root = Path(temporary.name)
        self.context, self.proof, self.receipt = q_context(self.root)
        self.spec = self.context['bindings']['scope']['qualifier_accounting']

    def make(self): return q.QualificationAccounting(self.context, None, self.proof, self.receipt, clock=lambda: 100.0)
    def raw(self, name):
        path = self.root / (name + '.json'); blob = b'{"fixture":true}\n'; path.write_bytes(blob)
        return dict(path=str(path), sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob))
    def checkpoint_stage(self, ledger):
        for stage in q.q.STAGES[1:q.q.STAGES.index('checkpoint_resume')]:
            ledger.begin_stage(stage, 'prior/' + stage); ledger.finish_stage([self.raw(stage)])
        ledger.begin_stage('checkpoint_resume', 'cp-contract')
    def cp(self, token=c.CP_TENSOR_PAIR_HASH, *, size=4096, extra=1):
        return dict(wall_contract=token, checkpoint_hash_scope=True, reason=c.CP_REASONS[str(token)],
            state_bytes=size, identity_hash_attempts_upper=extra, identity_hash_bytes_upper=0)
    def bootstrap(self):
        spec = specification(self.root / 'service'); spec['operation_contracts'] = manifest()
        spec['operation_profile_rule'] = s.INPUT_RESERVATION_RULE
        spec['explicit_reservation_kinds'] = ['input_io', 'worker_source_verification']
        spec['input_io_row_wall_s_upper'] = 5
        spec['caps'].update(wall_seconds=10**9, operation_attempts=100000)
        for row in spec['operation_contracts']['contracts'].values():
            if 'bootstrap' in row['owners'] and str(row['kind']) in c.MAPPED_KINDS:
                profile = spec['operation_profiles'].setdefault(row['kind'], dict(read_bytes_upper=10**6,
                    write_bytes_upper=10**6, identity_hash_attempts_upper=10000,
                    identity_hash_bytes_upper=10**6, wall_s_upper=1))
                profile['wall_s_upper'] = max(profile['wall_s_upper'], row['wall_s_upper'])
        ledger = s.BootstrapLedger(spec, 'a'*64, synthetic_fixture=True, clock=Clock())
        return spec, ledger

    def test_manifest_exact_dag_source_aliases_and_immutable_copy(self):
        value = manifest(); self.assertEqual(len(value['contracts']), 26)
        self.assertIs(c.validate_manifest(value), value)
        frozen = c.frozen_manifest(value)
        with self.assertRaises(TypeError): frozen['contracts']['file_hash_leaf']['wall_s_upper'] = 1
        mutations = (lambda m: m['contracts']['backend_ref_frame'].update(wall_s_upper=259),
            lambda m: m['contracts']['guard'].update(children={'backend_ref_frame': 1}),
            lambda m: m['contracts']['cp_file_hash'].update(wall_s_upper=59),
            lambda m: m['contracts']['cp_restore'].update(owners=['bootstrap']),
            lambda m: m.update(publication_tail_s_upper=True))
        for mutate in mutations:
            altered = copy.deepcopy(value); mutate(altered)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): c.validate_manifest(altered)

    def test_cp_hash_aliases_cover_start_guard_and_parents_cover_return_guards(self):
        value = manifest(); rows = value['contracts']; leaf = rows['cp_hash_leaf']
        self.assertEqual(leaf['children'], {'guard': 1})
        self.assertEqual(leaf['wall_s_upper'], 240)
        self.assertEqual(leaf['local_s_upper'], 60)
        for role in c.CP_HASH_ROLES:
            self.assertEqual(rows[role]['children'], leaf['children'])
            self.assertEqual(rows[role]['wall_s_upper'], leaf['wall_s_upper'])
        self.assertEqual(rows['cp_save_fresh']['children']['guard'], 13)
        self.assertEqual(rows['cp_restore']['children']['guard'], 627)
        self.assertEqual(rows['cp_save_adapter']['children']['opaque_checkpoint_save'], 1)
        self.assertEqual(rows['cp_restore_adapter']['children']['opaque_checkpoint_restore'], 1)
        altered = copy.deepcopy(value); altered['contracts']['cp_file_hash']['children'] = {}
        with self.assertRaisesRegex(ValueError, 'composition alias'): c.validate_manifest(altered)
        profiles = self.spec['operation_profiles']
        for token, expected in ((c.CP_SAVE_ADAPTER, 32768), (c.CP_RESTORE_ADAPTER, 1048576)):
            selected = q.reservation_profile(self.spec, 'checkpoint_adapter_io', dict(wall_contract=token))
            self.assertEqual(selected['wall_s_upper'], expected)
            self.assertLessEqual(selected['wall_s_upper'], profiles['checkpoint_adapter_io']['wall_s_upper'])

    def test_tokens_survive_deepcopy_and_plain_json_cannot_select_live_role(self):
        self.assertIs(copy.deepcopy(c.FILE_HASH_LEAF), c.FILE_HASH_LEAF)
        self.assertIs(copy.copy(c.MODEL_FINGERPRINT), c.MODEL_FINGERPRINT)
        plain = json.loads(json.dumps({'wall_contract': c.FILE_HASH_LEAF}))
        self.assertIs(type(plain['wall_contract']), str)
        ledger = self.make(); ledger.begin_stage('runtime_anchor', 'runtime')
        old_size = ledger.path.stat().st_size; before = copy.deepcopy(ledger.cumulative)
        with patch.object(Path, 'open', side_effect=AssertionError('forged source role IO')):
            with self.assertRaisesRegex(ValueError, 'live immutable'):
                ledger.operation('input_io', 'started', 'forged', **plain)
        self.assertEqual(ledger.path.stat().st_size, old_size); self.assertEqual(ledger.cumulative, before)
        with self.assertRaises(AttributeError): c.FILE_HASH_LEAF.kind = 'forged'

    def test_source_kind_owner_missing_role_and_wrong_stage_refuse(self):
        ledger = self.make(); ledger.begin_stage('runtime_anchor', 'runtime')
        for kind, fields in (('input_io', {}), ('input_io', dict(wall_contract=c.CP_FILE_HASH)),
            ('fingerprint', self.cp()), ('input_io', dict(wall_contract=c.LEASE_PARENT_CHECK))):
            with self.subTest(kind=kind, fields=fields), self.assertRaises(ValueError):
                ledger.operation(kind, 'started', 'invalid', **fields)
        with self.assertRaisesRegex(ValueError, 'kind/owner'):
            c.reservation_profile(self.spec, 'input_io', dict(wall_contract=c.BACKEND_REF_FRAME),
                owner='bootstrap', maximum=self.spec['operation_profiles']['input_io'])

    def test_q_same_kind_leaf_and_outer_inclusive_source_walls(self):
        ledger = self.make(); ledger.begin_stage('runtime_anchor', 'runtime')
        fields = dict(read_bytes_upper=17, write_bytes_upper=0, identity_hash_attempts_upper=1,
                      identity_hash_bytes_upper=17)
        ledger.operation('input_io', 'started', 'outer', wall_contract=c.BACKEND_REF_FRAME, **fields)
        ledger.operation('input_io', 'started', 'leaf', wall_contract=c.FILE_HASH_LEAF, **fields)
        ledger.operation('input_io', 'returned', 'leaf', wall_contract=c.FILE_HASH_LEAF, wall_s=1, bytes_read=17)
        ledger.operation('input_io', 'returned', 'outer', wall_contract=c.BACKEND_REF_FRAME, wall_s=2)
        self.assertEqual(ledger.operations['outer']['profile']['wall_s_upper'], 300)
        self.assertEqual(ledger.operations['leaf']['profile']['wall_s_upper'], 60)
        self.assertEqual(ledger.by_stage['runtime_anchor']['wall_seconds_upper'], 360)
        shadow = q.replay_state(self.context, ledger.header, self.receipt)
        for row in ledger.events:
            shadow._apply(row['kind'], row['payload'], replay=True); shadow.events.append(copy.deepcopy(row))
        self.assertEqual(shadow.cumulative, ledger.cumulative)

    def test_cp_extra_two_zero_and623_exact_extents_preserve_ce_builtin_fields(self):
        ledger = self.make(); self.checkpoint_stage(ledger)
        before_hash = ledger.cumulative['identity_hash_bytes_upper']; expected = 0
        for index in range(623):
            token = c.CP_TENSOR_PAIR_HASH if index < 618 else c.CP_PUBLICATION_HASH
            extra = 1 if index < 618 else 2; size = 4096 + index
            ledger.operation('fingerprint', 'started', 'cp-%d' % index, **self.cp(token, size=size, extra=extra))
            ledger.operation('fingerprint', 'returned', 'cp-%d' % index, wall_contract=token, wall_seconds=.01,
                identity_hash_attempts_actual=1 + extra, identity_hash_bytes_actual=size)
            expected += size
        self.assertEqual(ledger.operations['cp-622']['profile']['identity_hash_attempts_upper'], 3)
        self.assertEqual(ledger.operations['cp-622']['fields']['identity_hash_attempts_upper'], 2)
        self.assertEqual(ledger.operations['cp-622']['fields']['identity_hash_bytes_upper'], 0)
        # Event/framing hashes are separately and permanently charged.
        self.assertGreaterEqual(ledger.cumulative['identity_hash_bytes_upper'] - before_hash, expected)
        self.assertLess(ledger.cumulative['identity_hash_bytes_upper'] - before_hash, 16 * 1024**2)
        self.assertEqual(ledger.operations['cp-0']['profile']['wall_s_upper'], 240)
        self.assertFalse(ledger.snapshot()['native_qualified'])

    def test_cp_extent_negative_boolean_unknown_reason_and_frozen_maximum(self):
        maximum = self.spec['operation_profiles']['fingerprint']
        for change in (dict(state_bytes=True), dict(state_bytes=0), dict(state_bytes=float('nan')),
            dict(identity_hash_attempts_upper=True), dict(identity_hash_bytes_upper=-1),
            dict(identity_hash_attempts_upper=maximum['identity_hash_attempts_upper']),
            dict(state_bytes=maximum['identity_hash_bytes_upper']+1), dict(reason='unknown')):
            fields = dict(self.cp(), **change)
            with self.subTest(change=change), self.assertRaises(ValueError): q.reservation_profile(self.spec, 'fingerprint', fields)
        fields = self.cp(); fields.pop('identity_hash_bytes_upper')
        with self.assertRaises(ValueError): q.reservation_profile(self.spec, 'fingerprint', fields)

    def test_model_and_metadata_opaque_full_profiles_keep_all_maxima(self):
        fields = dict(wall_contract=c.MODEL_FINGERPRINT, state_bytes=4096,
                      identity_hash_attempts_upper=2, identity_hash_bytes_upper=0)
        selected = q.reservation_profile(self.spec, 'fingerprint', fields)
        self.assertEqual(selected, self.spec['operation_profiles']['fingerprint'])
        fields = dict(wall_contract=c.OPAQUE_INPUT_FRAME, read_bytes_upper=17,
                      identity_hash_attempts_upper=1, identity_hash_bytes_upper=17)
        self.assertEqual(q.reservation_profile(self.spec, 'input_io', fields), self.spec['operation_profiles']['input_io'])

    def test_existing_declared_wall_cannot_discount_or_reprice_source_contract(self):
        for token, declared, expected in ((c.FILE_HASH_LEAF, 1, 60), (c.BACKEND_REF_FRAME, 5, 300),
                                          (c.OPAQUE_INPUT_FRAME, 60, 360)):
            fields = dict(wall_contract=token, wall_s_upper=declared)
            selected = q.reservation_profile(self.spec, 'input_io', fields)
            self.assertEqual(selected['wall_s_upper'], expected)
        for value in (True, float('nan'), 361):
            with self.subTest(value=value), self.assertRaises(ValueError):
                q.reservation_profile(self.spec, 'input_io', dict(wall_contract=c.FILE_HASH_LEAF, wall_s_upper=value))

    def test_compiled_manifest_matches_frozen26_and_opaque_adapter_full_price(self):
        from experiments.skill_acquisition_native_v1 import ce_contract_registration as registration
        value = registration.manifest()
        self.assertEqual(value, manifest()); c.validate_manifest(value)
        self.assertEqual(value['contracts']['model_fingerprint']['owners'], ['qualification'])
        fields = dict(wall_contract=c.OPAQUE_CP_ADAPTER_FRAME, read_bytes_upper=17,
                      identity_hash_attempts_upper=1, identity_hash_bytes_upper=17)
        selected = q.reservation_profile(self.spec, 'checkpoint_adapter_io', fields)
        self.assertEqual(selected, self.spec['operation_profiles']['checkpoint_adapter_io'])
        self.assertEqual(selected['wall_s_upper'], 1048576)
        with self.assertRaisesRegex(ValueError, 'owner'):
            c.validate_live_fields(self.spec, 'fingerprint', 'started', dict(wall_contract=c.MODEL_FINGERPRINT),
                                   owner='bootstrap')

    def test_selected_wall_alias_overreturn_is_fatal_no_refund(self):
        for returned in (dict(wall_s=61), dict(wall_seconds=61), dict(wall_s=1, wall_seconds=2),
                         dict(wall_s=True), dict(wall_seconds=float('nan'))):
            with tempfile.TemporaryDirectory() as subdirectory:
                context, proof, receipt = q_context(Path(subdirectory))
                ledger = q.QualificationAccounting(context, None, proof, receipt, clock=lambda: 100)
                ledger.begin_stage('runtime_anchor', 'runtime')
                ledger.operation('input_io', 'started', 'leaf', wall_contract=c.FILE_HASH_LEAF)
                charged = copy.deepcopy(ledger.cumulative)
                with self.subTest(returned=returned), self.assertRaises(ValueError):
                    ledger.operation('input_io', 'returned', 'leaf', wall_contract=c.FILE_HASH_LEAF, **returned)
                self.assertTrue(ledger.fatal); self.assertEqual(ledger.cumulative, charged)
                self.assertEqual(ledger.operations['leaf']['phase'], 'started')

    def test_failed_charge_replay_and_terminal_repricing_reject(self):
        ledger = self.make(); ledger.begin_stage('runtime_anchor', 'runtime')
        ledger.operation('input_io', 'started', 'leaf', wall_contract=c.FILE_HASH_LEAF)
        before = copy.deepcopy(ledger.cumulative)
        with self.assertRaisesRegex(ValueError, 'terminal source role'):
            ledger.operation('input_io', 'returned', 'leaf', wall_contract=c.OPAQUE_INPUT_FRAME, wall_s=1)
        ledger.operation('input_io', 'failed', 'leaf', wall_contract=c.FILE_HASH_LEAF, wall_s=1)
        self.assertGreaterEqual(ledger.cumulative['wall_seconds_upper'], before['wall_seconds_upper'])
        ledger.fail_stage(RuntimeError('discarded fixture')); ledger.end()
        events = copy.deepcopy(ledger.events)
        next(row for row in events if row['kind']=='operation' and row['payload']['phase']=='started')['payload']['reservation_profile']['wall_s_upper'] = 1
        head = None
        for row in events:
            row['previous'] = head; row.pop('sha256'); row['sha256'] = q.digest(row); head = row['sha256']
        ledger.path.write_bytes(b''.join(q.canonical(row)+b'\n' for row in events))
        with self.assertRaisesRegex(ValueError, 'reservation profile drift'): self.make()

    def test_manifest_terminal_role_cannot_be_omitted_live_or_raw_replay(self):
        ledger = self.make(); ledger.begin_stage('runtime_anchor', 'runtime')
        ledger.operation('input_io', 'started', 'leaf', wall_contract=c.FILE_HASH_LEAF)
        before = copy.deepcopy(ledger.cumulative)
        with self.assertRaisesRegex(ValueError, 'terminal source role'):
            ledger.operation('input_io', 'returned', 'leaf', wall_s=1)
        self.assertEqual(ledger.cumulative, before); self.assertEqual(ledger.operations['leaf']['phase'], 'started')
        shadow = q.replay_state(self.context, ledger.header, self.receipt)
        for row in ledger.events:
            shadow._apply(row['kind'], row['payload'], replay=True); shadow.events.append(copy.deepcopy(row))
        with self.assertRaisesRegex(ValueError, 'terminal source role'):
            shadow._prepare('operation', dict(kind='input_io', phase='returned', id='leaf', fields={'wall_s':1}))
        _, service = self.bootstrap()
        service.operation('input_io', 'started', 'leaf', wall_contract=c.FILE_HASH_LEAF)
        charged = copy.deepcopy(service.counters)
        with self.assertRaisesRegex(ValueError, 'terminal source role'):
            service.operation('input_io', 'returned', 'leaf', wall_s=1)
        self.assertEqual(service.counters, charged); self.assertEqual(service.operations['leaf']['phase'], 'started')

    def test_bootstrap_token_selection_and_raw_replay_are_separate(self):
        spec, ledger = self.bootstrap()
        fields = dict(wall_contract=c.TEACHER_ROW, wall_s_upper=60, row_id='known-row', read_bytes_upper=17, write_bytes_upper=0,
                      identity_hash_attempts_upper=1, identity_hash_bytes_upper=17)
        ledger.operation('input_io', 'started', 'row', **fields)
        self.assertEqual(ledger.operations['row']['profile']['wall_s_upper'], 240)
        ledger.operation('input_io', 'returned', 'row', wall_contract=c.TEACHER_ROW, wall_s=1)
        start = ledger.events[ledger.operations['row']['started_seq']]['payload']
        serialized = json.loads(s.canonical(start['fields']))
        with self.assertRaisesRegex(ValueError, 'live immutable'): s.reservation_profile(spec, 'input_io', serialized)
        selected = s.reservation_profile(spec, 'input_io', serialized, contract_replay=True)
        self.assertEqual(selected, start['reservation_profile'])
        with self.assertRaisesRegex(ValueError, 'live immutable'):
            ledger.operation('input_io', 'started', 'forged', **serialized)

    def test_bootstrap_lease_only_exact_source_parent_action_and_alias_bounds(self):
        _, ledger = self.bootstrap()
        with self.assertRaisesRegex(ValueError, 'exact parent check'):
            ledger.operation('supervisor_lease', 'started', 'wrong', wall_contract=c.LEASE_PARENT_CHECK, lease_action='delegate_child')
        ledger.operation('supervisor_lease', 'started', 'parent', wall_contract=c.LEASE_PARENT_CHECK, lease_action='check_parent_use')
        self.assertEqual(ledger.operations['parent']['profile']['wall_s_upper'], 60)
        before = copy.deepcopy(ledger.counters)
        with self.assertRaises(ValueError): ledger.operation('supervisor_lease', 'returned', 'parent',
            wall_contract=c.LEASE_PARENT_CHECK, lease_action='check_parent_use', wall_seconds=61)
        self.assertTrue(ledger.fatal); self.assertEqual(ledger.counters, before)

    def test_no_marker_retains_old_five_seconds_and_no_role_authority(self):
        context, proof, receipt = fixture(self.root / 'legacy')
        context['bindings']['scope']['qualifier_accounting']['operation_profiles']['input_io']['wall_s_upper'] = 5
        ledger = q.QualificationAccounting(context, None, proof, receipt, clock=lambda: 100)
        ledger.begin_stage('runtime_anchor', 'runtime')
        ledger.operation('input_io', 'started', 'row', wall_contract=c.TEACHER_ROW)
        self.assertEqual(ledger.operations['row']['profile']['wall_s_upper'], 5)
        self.assertNotIn('wall_contract', ledger.operations['row']['fields'])
        ledger.operation('input_io', 'returned', 'row', wall_s=4)
        legacy = copy.deepcopy(self.spec); legacy.pop('operation_contracts'); legacy['operation_profile_rule'] = q.EXPLICIT_RESERVATION_RULE
        with self.assertRaisesRegex(ValueError, 'coherent'):
            q.reservation_profile(legacy, 'fingerprint', {key:value for key,value in self.cp().items() if key!='wall_contract'})


if __name__ == '__main__': unittest.main()
