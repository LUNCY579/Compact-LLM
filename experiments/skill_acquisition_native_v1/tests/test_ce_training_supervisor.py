"""Original durable CE byte replay and closed supervisor; no model/OS jobs."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_training_supervisor as s
from . import test_ce_journal as original


class Trap:
    def __getattribute__(self, name): raise AssertionError('before source admission')


class Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='training-supervisor-pure-'); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.plan, self.bindings, self.caps = original.fixture()
        self.ledger = original.cj.CeJournal(self.root / 'ledger', self.plan, self.bindings, self.caps,
                                          synthetic_cpu=True, clock=lambda: 1.)
        self.ledger.start_invocation(boot_id='fixture-boot')
        self.binding = {'stage': 'training_supervisor', 'request_sha256': 'a' * 64, 'scope_sha256': 'b' * 64,
            'static_recipe_sha256': s.digest(self.bindings['recipe']), 'code_sha256': self.bindings['source'],
            'worker_identity_sha256': 'c' * 64, 'attempt_id': 'pure-raw-replay'}
        self.artifact = {'owned_directory': str(self.ledger.directory), 'plan': self.plan, 'bindings': self.bindings, 'caps': self.caps}
        self.calls = []
    def ref(self, path):
        blob = path.read_bytes(); return {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}
    def replay(self, snapshot=None, preflight=None):
        reader = s.raw.RawEvidenceReader(lambda *args, **fields: self.calls.append((args, fields)),
            {key: 10 ** 12 for key in s.raw.BUDGET_FIELDS}, self.root)
        return s.replay_journal(self.artifact, self.ref(self.ledger.header_path), self.ref(self.ledger.path),
            self.ledger.snapshot() if snapshot is None else snapshot, self.binding, reader=reader,
            preflight=preflight or (lambda *args: self.calls.append(('preflight', args))), fixture=True)
    def test_raw_causal_commit_and_ended_invocation_replay_no_execution_claim(self):
        helper = object.__new__(original.Controls); helper.bindings = self.bindings
        self.ledger.commit_boundary(helper.successful_result(self.ledger))
        self.ledger.end_invocation('INTERRUPTED')
        state = self.replay()
        self.assertEqual(state.progress_record()['C'], 1)
        self.assertEqual(state.snapshot(), self.ledger.snapshot())
        self.assertFalse(state.snapshot()['native_qualified'])
    def test_failed_and_pending_costs_cannot_refund_or_claim_cleared_progress(self):
        self.ledger.operation('input_io', 'started', 'failed'); self.ledger.operation('input_io', 'failed', 'failed')
        self.ledger.operation('input_io', 'started', 'pending')
        state = self.replay(); self.assertEqual(state.counters, self.ledger.counters)
        self.assertEqual(state.snapshot()['pending_operations'], 1)
        wrong = copy.deepcopy(self.ledger.snapshot()); wrong['cumulative']['io_read_bytes'] = 0
        with self.assertRaisesRegex(ValueError, 'snapshot refund'): self.replay(wrong)
    def test_foreign_header_chain_partial_tail_and_census_are_rejected(self):
        before = self.ledger.path.read_bytes(); self.ledger.path.write_bytes(before[:-1])
        with self.assertRaisesRegex(ValueError, 'partial CE tail'): self.replay()
        self.ledger.path.write_bytes(before.replace(b'"seq":0', b'"seq":9', 1))
        with self.assertRaisesRegex(ValueError, 'hash/order'): self.replay()
        self.ledger.path.write_bytes(before)
        header = s.raw._json(self.ledger.header_path.read_bytes()); first = next(iter(header['token_census']))
        header['token_census'][first]['prompt_tokens'] += 1
        self.ledger.header_path.write_bytes(s.canonical(header) + b'\n')
        with self.assertRaisesRegex(ValueError, 'token census'): self.replay()
    def test_preflight_denial_and_counterfeit_native_supervisor_before_body(self):
        def deny(*args): raise RuntimeError('replay prepaid scope denied')
        with self.assertRaisesRegex(RuntimeError, 'scope denied'): self.replay(preflight=deny)
        self.assertEqual(self.calls, [])
        with patch.object(s.launch, 'launch_native', side_effect=AssertionError('unadmitted child launch')):
            with self.assertRaises(ValueError): s.supervise_training(Trap(), Trap(), Trap())

    def native_metadata_transcript(self):
        """Simulated causal records only; no native permit, model or graphs."""
        from experiments.skill_acquisition_native_v1.ce_update import prepared_row_binding
        plan_rows, prepared = [], []
        for i in range(3000):
            row = {'id': 'native-metadata-%04d' % i, 'prompt_ids': [2, 3], 'completion_ids': [4, 151645]}
            target = s.digest(row['completion_ids'])
            plan_rows.append({'id': row['id'], 'target_sha256': target})
            prepared.append(dict(prepared_row_binding(row), target_sha256=target, render_sha256='e' * 64,
                                 prepared_hash_bytes_upper=1024))
        plan = s.ce_plan.build_synthetic(plan_rows)
        index = {row['id']: row for row in prepared}; prepared = [index[rid] for ids in s.ce_plan.validate_plan(plan) for rid in ids]
        bindings = copy.deepcopy(self.bindings); bindings.update(scope=s.journal.NATIVE_SCOPE,
            recipe=dict(bindings['recipe'], synthetic_cpu=False), prepared_rows=prepared, metadata_roots=[str(self.root)])
        profile = dict(read_bytes_upper=10**8, write_bytes_upper=10**8, identity_hash_attempts_upper=1000,
                       identity_hash_bytes_upper=10**9, wall_s_upper=1)
        bindings['operation_bounds'] = {kind: dict(profile) for kind in s.journal.NATIVE_KINDS}
        bindings['setup_bounds'] = dict(profile)
        caps = dict.fromkeys(s.journal.COUNTERS, 10**15)
        caps.update(schema=s.journal.NATIVE_CAP_SCHEMA, synthetic_cpu=False,
                    operation_starts=dict.fromkeys(s.journal.NATIVE_KINDS, 10000))
        census = {row['id']: {key: row[key] for key in ('id','prompt_tokens','labeled_tokens','full_sequence_tokens',
                    'prompt_ids_sha256','completion_ids_sha256','prepared_row_sha256')} for row in prepared}
        header = {'schema':s.journal.SCHEMA,'classification':'explicit source-free metadata simulation; never execution proof',
                  'plan':plan,'bindings':bindings,'caps':caps,'token_census':census}
        directory = self.root / 'native-metadata'; directory.mkdir()
        hp, jp = directory / s.journal.HEADER, directory / s.journal.JOURNAL
        encoded = s.canonical(header) + b'\n'; hp.write_bytes(encoded)
        state = s._shadow(header, directory, fixture=False)
        state._charge({'io_write_bytes':len(encoded),'metadata_reserved_bytes':len(encoded),
                      'identity_hash_attempts':1,'identity_hash_bytes':len(s.canonical(header))})
        serial = 0
        def event(kind, payload):
            body, line = state._encoded(kind, payload)
            cost, action = state._prepare(kind, body['payload']); state._charge(state._combined(cost,len(line),len(s.canonical(body))))
            action(); state.events.append(s.raw._json(line))
            with jp.open('ab') as stream:stream.write(line)
        def span(kind, fields, output=None):
            nonlocal serial
            serial += 1; identifier = 'simulated-%d' % serial
            event('operation', {'operation':kind,'phase':'started','id':identifier,'fields':fields})
            returned = dict(fields, **(output or {}), wall_s=0.)
            event('operation', {'operation':kind,'phase':'returned','id':identifier,'fields':returned})
            return dict(returned, kind=kind, operation_id=identifier)
        def save():
            p = state.progress_record()
            fields = {'C':p['C'],'U':p['U'],'generation':p['generation'],'symbolic_snapshot':p['snapshot']}
            span('checkpoint_save', fields, {'manifest_sha256':s.digest(p)})
        event('setup', {'resume':False}); event('invocation_start', {'id':'metadata-invocation','resume':False,'boot_id':'fixture-boot','perf_started':1.,'reason':'no execution'})
        save()
        schedule = [0,1,75,150,225,300,375]
        for C in range(375):
            start, final = state._expected_policy(), s.digest(['simulated-policy',C+1])
            meta = {'boundary_id':C,'generation':state.generation,'symbolic_snapshot':state.symbolic_snapshot,
                'start_policy_id':start,'static_recipe_sha256':state.static_recipe_sha256,'objective':'SUPERVISED_EQUAL_ROW_TOKEN_MEAN_CE'}
            before = span('fingerprint',dict(meta,reason='ce_pre_boundary_identity',state_bytes=100),
                          {'policy_sha256':start,'U':C,'frozen_state_sha256':'f'*64})
            span('checkpoint_configuration',meta);span('gradient_reset',meta);rows=[]
            for rid in state.plan_rows[C]:
                row = state.census[rid]
                fields = dict(meta,prompt_id=rid,selected_tokens=row['labeled_tokens'],full_sequence_tokens=row['full_sequence_tokens'],
                              **{key:row[key] for key in ('prompt_ids_sha256','completion_ids_sha256','prepared_row_sha256')})
                forward, backward = span('loss_forward',fields), span('backward',fields)
                rows.append(dict(row,forward_operation=forward,backward_operation=backward,
                    selected_predictor_positions=list(range(row['prompt_tokens']-1,row['full_sequence_tokens']-1)),
                    target_positions=list(range(row['prompt_tokens'],row['full_sequence_tokens'])),
                    eos_positions_selected=1,prompt_positions_selected=0,pad_positions_selected=0))
            validation = span('gradient_validation',meta,{'finite_confined':True,'raw_gradient_norm_float64':.5})
            clip = span('gradient_clipping',dict(meta,**bindings['recipe']['gradient_clipping']),{'clipped_gradient_norm_float64':.5})
            step = span('optimizer',dict(meta,U_before=C,expected_U_after=C+1))
            after = span('fingerprint',dict(meta,reason='ce_post_boundary_identity',state_bytes=100),
                         {'policy_sha256':final,'U':C+1,'frozen_state_sha256':'f'*64})
            span('gradient_cleanup',dict(meta,optimizer_attempted=True))
            result = dict(meta,schema='SKILL-ACQUISITION-CE-BOUNDARY-RESULT-1',synthetic_cpu=False,
                loss_reduction=bindings['recipe']['loss_reduction'],accumulation_rows=8,U_before=C,U_after=C+1,
                updated=True,requires_restore=False,rows=rows,identity_operation=before,final_identity_operation=after,
                final_policy_id=final,frozen_state_sha256='f'*64,optimizer_operation=step,gradient_validation_operation=validation,
                clipping_operation=clip,raw_gradient_norm_float64=.5,clipped_gradient_norm_float64=.5,
                optimizer_recipe=bindings['recipe']['optimizer'],gradient_clipping_recipe=bindings['recipe']['gradient_clipping'])
            event('commit', {'result':result})
            if C+1 in schedule:save()
        event('invocation_end', {'id':'metadata-invocation','outcome':'COMPLETE'})
        return state, {'owned_directory':str(directory),'plan':plan,'bindings':bindings,'caps':caps}, self.ref(hp), self.ref(jp), schedule

    def test_full375_native_mode_record_replay_seven_saves_and_complete_without_execution(self):
        state, artifact, hp, jp, schedule = self.native_metadata_transcript()
        reader = s.raw.RawEvidenceReader(lambda *args,**fields:None,{key:10**12 for key in s.raw.BUDGET_FIELDS},self.root)
        replayed = s.replay_journal(artifact,hp,jp,state.snapshot(),self.binding,reader=reader,preflight=lambda *args:None)
        execution = {'boundaries':375,'checkpoint_boundaries':schedule}
        saves = s.validate_checkpoint_schedule(replayed,execution)
        self.assertEqual([value['C'] for value in saves],schedule)
        self.assertEqual(replayed.progress_record()['C'],375);self.assertEqual(replayed.progress_record()['U'],375)
        self.assertEqual(replayed.snapshot(),state.snapshot());self.assertFalse(replayed.snapshot()['native_qualified'])
        save = next(operation for operation in replayed.operations.values() if operation['kind']=='checkpoint_save' and operation['result']['C']==75)
        save['phase']='failed'
        with self.assertRaisesRegex(ValueError,'missing scheduled'):s.validate_checkpoint_schedule(replayed,execution)


if __name__ == '__main__': unittest.main()
