"""Sparse CPU controls only; real bound TRAIN input IDs, no model or decoder."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
STAGED=HERE.name=='long_context_qualification_work' and HERE.parent.name=='artifacts'
PACKAGE=HERE.parents[1] if STAGED else HERE.parent
CODE=HERE if STAGED else PACKAGE
sys.path.insert(0,str(CODE));sys.path.append(str(PACKAGE))
import long_split_controller as split
spec=importlib.util.spec_from_file_location("split_sparse_fixtures",PACKAGE/'tests/test_inference_component.py')
fake=importlib.util.module_from_spec(spec);spec.loader.exec_module(fake)
PACKET=json.loads((PACKAGE/'evidence/long_train_prompts_cpu_20261005/TRAIN_PROMPTS_CPU.json').read_bytes())
IDENTITY={key:split.component.digest(["CPU_FIXTURE_DECLARATION",key]) for key in split.IDENTITY_KEYS}


class Fixture:
    def __init__(self, *, tokens=None, cap=4096, stress=False):
        ids=next(x['row']['prompt_ids'] for x in PACKET['rows'] if x['context_cap']==cap)
        prefix=[20628]*31 if stress else None
        self.recipe=split.recipe(PACKET,cap,IDENTITY,prefix_stress_ids=prefix)
        self.tokens=[9]*32 if tokens is None else tokens
        self.saved={};self.sessions=[];self.mutate=None;self.clock_value=0.;self.fail_sink=None
        self.controller=split.Controller(self.recipe,self.factory,self.preserve,lambda *args:None,clock=lambda:self.clock_value)
    def preserve(self,name,value):
        if self.fail_sink:self.fail_sink(name,value)
        if name in self.saved:raise ValueError("immutable fixture receipt name reused")
        self.saved[name]=deepcopy(value)
    def factory(self,recipe,role,guard):
        torch,journal=fake.FakeTorch(),fake.Journal()
        model=fake.Model(torch,tokens=self.tokens,prompt_length=recipe['row']['prompt_tokens'])
        engine=split.component.InferenceComponent(torch,model,fake.Decoder(),journal.call,guard,journal.preserve,
            max_attempted_forwards=recipe['bulk_scope_forwards'] if role=='bulk' else recipe['trace_scope_forwards'],
            max_input_positions=recipe['bulk_scope_positions'] if role=='bulk' else recipe['trace_scope_positions'],
            context_cap=recipe['context_cap'],generation_cap=32,device='cpu')
        identity={**IDENTITY,'scope_nonce':f'CPU_SCOPE_{len(self.sessions)}','fixture':True}
        session={'component':engine,'identity':lambda:deepcopy(identity),'identity_object':identity,'model':model,'torch':torch}
        self.sessions.append(session)
        if self.mutate:self.mutate(session,role)
        return session
    def run(self,role,**kwargs):
        return self.controller.run(role,expected_latest_sha256=self.controller.latest_sha256,**kwargs)
    def complete(self):
        first=self.run('trace1');second=self.run('trace2');bulk=self.run('bulk',references=self.controller.reference())
        return first,second,bulk


class Controls(unittest.TestCase):
    def test_full32_three_fresh_scopes_exact_parity_cumulative_costs_and_all_LPs(self):
        for cap,total in ((1024,68017),(2048,134577),(4096,267697)):
            f=Fixture(cap=cap);first,second,bulk=f.complete()
            self.assertEqual(first['trace'],second['trace']);self.assertEqual(len(first['trace']),32)
            self.assertEqual(bulk['comparison']['compared_actual_steps'],32)
            self.assertEqual(bulk['comparison']['maximum_selected_LP_error'],0)
            self.assertEqual(bulk['cumulative_costs'],{'attempted_forwards':71,'input_positions':total})
            self.assertEqual([len(x['model'].calls) for x in f.sessions],[34,34,3])
            self.assertFalse(bulk['native_launch_authorized']);self.assertFalse(bulk['native_zero_B_parity_proven'])
            self.assertEqual(first['previous_capsule_sha256'],f.recipe['sha256'])
            self.assertEqual(second['previous_capsule_sha256'],first['sha256'])

    def test_early_EOS_is_preserved_and_separate_predeclared_stress_not_greedy(self):
        f=Fixture(tokens=[9,151645],stress=True);first,second,bulk=f.complete()
        self.assertEqual(len(first['trace']),2);self.assertEqual(first['trace'][-1]['token'],151645)
        self.assertEqual(first['greedy_finish_reason'],'eos')
        self.assertFalse(first['full32_greedy_coverage']);self.assertEqual(bulk['comparison']['compared_actual_steps'],2)
        self.assertEqual(bulk['cumulative_costs'],{'attempted_forwards':11,'input_positions':22837})
        stress=f.run('prefix_stress')
        self.assertEqual(len(stress['trace']),32);self.assertFalse(stress['stress_is_greedy_trace'])
        self.assertIsNone(stress['greedy_finish_reason']);self.assertTrue(stress['prefix_stress_selected_EOS_observed'])
        self.assertFalse(stress['full32_greedy_coverage']);self.assertEqual(stress['scope_costs']['attempted_forwards'],34)
        self.assertEqual(stress['cumulative_costs'],{'attempted_forwards':45,'input_positions':154219})

    def test_stress_not_predeclared_and_EOS_or_nonTRAIN_prefix_denied(self):
        f=Fixture();f.complete()
        with self.assertRaisesRegex(ValueError,'not predeclared'):f.run('prefix_stress')
        for tokens in ([151645]*31,[42]*31):
            with self.assertRaises(ValueError):split.recipe(PACKET,4096,IDENTITY,prefix_stress_ids=tokens)

    def test_rollback_wrong_reference_and_order_fail_before_new_factory(self):
        f=Fixture();first=f.run('trace1')
        with self.assertRaisesRegex(ValueError,'rollback'):
            f.controller.run('trace2',expected_latest_sha256=f.recipe['sha256'])
        self.assertEqual(len(f.sessions),1)
        f.run('trace2')
        with self.assertRaisesRegex(ValueError,'reference hashes'):f.run('bulk',references={'trace1':first['sha256']})
        self.assertEqual(len(f.sessions),2)
        with self.assertRaisesRegex(ValueError,'scope order'):f.run('trace1')

    def test_published_capsule_mutation_or_self_resigning_denied(self):
        for resign in (False,True):
            f=Fixture();f.run('trace1');cap=f.controller.capsules[0];cap['trace'][0]['token']=10
            if resign:cap['sha256']=split.component.digest({k:v for k,v in cap.items() if k!='sha256'})
            with self.assertRaisesRegex(ValueError,'capsule'):f.run('trace2')
            self.assertEqual(len(f.sessions),1)

    def test_source_model_runtime_zeroB_RNG_and_fresh_identity_denials(self):
        for key in split.IDENTITY_KEYS:
            f=Fixture();f.mutate=lambda s,r,key=key:s['identity_object'].update({key:'f'*64})
            with self.assertRaisesRegex(ValueError,'identity mismatch'):f.run('trace1')
            self.assertEqual(f.sessions[0]['model'].calls,[])
        f=Fixture();f.run('trace1');old=f.sessions[0]
        f.mutate=lambda s,r:s.update(component=old['component'])
        with self.assertRaisesRegex(ValueError,'fresh component'):f.run('trace2')

    def test_trace_repeat_bulk_numerics_mismatch_halts_and_preserves_costs(self):
        f=Fixture();f.run('trace1')
        f.tokens=[10]*32;f.run('trace2')
        with self.assertRaisesRegex(ValueError,'greedy repeats'):f.run('bulk',references=f.controller.reference())
        self.assertTrue(f.controller.halted);self.assertEqual(f.saved['bulk_failed']['actual_scope_costs']['attempted_forwards'],2)
        f=Fixture();f.run('trace1');f.run('trace2');f.tokens=[10]*32
        with self.assertRaisesRegex(ValueError,'bulk argmax'):f.run('bulk',references=f.controller.reference())
        self.assertEqual(f.saved['bulk_failed']['actual_scope_costs']['attempted_forwards'],3)

    def test_same_argmax_bulk_selected_LP_error_cannot_pass_numeric_gate(self):
        f=Fixture();f.run('trace1');f.run('trace2')
        def change(session,role):
            if role=='bulk':
                def poison(model,logits):
                    for row in logits.rows:row.selected=12.
                session['model'].fault=poison
        f.mutate=change
        with self.assertRaisesRegex(ValueError,'numerical gates'):f.run('bulk',references=f.controller.reference())
        self.assertTrue(f.controller.halted);self.assertEqual(f.saved['bulk_failed']['actual_scope_costs']['attempted_forwards'],3)

    def test_expanded_budget_native_scope_and_deadline_rejected(self):
        f=Fixture();f.mutate=lambda s,r:setattr(s['component'],'max_input_positions',134145)
        with self.assertRaisesRegex(ValueError,'scope dose'):f.run('trace1')
        changed=deepcopy(f.recipe);changed['scope']='NATIVE';changed=split.seal({k:v for k,v in changed.items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'unregistered'):split.Controller(changed,f.factory,f.preserve,lambda *args:None)
        f=Fixture();f.mutate=lambda s,r:setattr(f,'clock_value',601.)
        with self.assertRaisesRegex(ValueError,'600s'):f.run('trace1')

    def test_resealed_recipe_expansion_and_row_substitution_rejected(self):
        f=Fixture()
        for field,value in (('trace_scope_positions',134145),('per_scope_seconds',601),('short_parity_row',{**f.recipe['short_parity_row'],'prompt_sha256':'f'*64})):
            changed=deepcopy(f.recipe);changed[field]=value;changed=split.seal({k:v for k,v in changed.items() if k!='sha256'})
            with self.assertRaisesRegex(ValueError,'registered CPU recipe'):
                split.Controller(changed,f.factory,f.preserve,lambda *args:None)

    def test_hidden_factory_warmup_and_extra_forward_cannot_disappear(self):
        f=Fixture();f.mutate=lambda s,r:s['model'].calls.append([5])
        with self.assertRaisesRegex(ValueError,'hidden factory warmup'):f.run('trace1')
        self.assertEqual(f.saved['trace1_failed']['actual_observed_CPU_fixture_forwards'],1)
        f=Fixture()
        def hidden(s,role):
            old=s['component'].call;done=[False]
            def call(name,func):
                value=old(name,func)
                if name=='long_greedy_trace_forward' and not done[0]:
                    done[0]=True;torch=s['torch'];ids=torch.tensor([f.recipe['short_parity_row']['prompt_ids']],dtype=torch.long,device='cpu')
                    with torch.no_grad():s['model'](input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True)
                return value
            s['component'].call=call
        f.mutate=hidden
        with self.assertRaisesRegex(ValueError,'hidden extra calls|physical prefix roster'):f.run('trace1')
        self.assertEqual(f.saved['trace1_failed']['actual_observed_CPU_fixture_forwards'],35)
        self.assertEqual(f.saved['trace1_failed']['actual_scope_costs']['attempted_forwards'],34)

    def test_review_mutated_budget_then_extra_metered_long_forward_rejected(self):
        f=Fixture()
        def mutate(session,role):
            e=session['component'];old=e.call;done=[False]
            def wrapped(name,fn):
                value=old(name,fn)
                if name=='long_greedy_trace_forward' and not done[0]:
                    done[0]=True;e.max_attempted_forwards=35;e.max_input_positions=200000
                    e._forward(f.recipe['row']['prompt_ids'],'injected_metered_extra_forward')
                return value
            e.call=wrapped
        f.mutate=mutate
        with self.assertRaisesRegex(ValueError,'frozen role/source dose'):f.run('trace1')
        self.assertTrue(f.controller.halted);self.assertEqual(f.controller.capsules,[])
        failed=f.saved['trace1_failed']
        self.assertIn('mutable component limits differ from frozen role',failed['frozen_role_dose_violations'])
        self.assertEqual(failed['actual_observed_CPU_fixture_forwards'],3)

    def test_review_early_EOS_extra_metered_call_rejected_with_unchanged_limits(self):
        for operation in ('injected_metered_extra_forward','long_greedy_trace_forward'):
            f=Fixture(tokens=[9,151645])
            def mutate(session,role):
                e=session['component'];old=e.call;done=[False]
                def wrapped(name,fn):
                    value=old(name,fn)
                    if name=='long_greedy_trace_forward' and not done[0]:
                        done[0]=True;e._forward(f.recipe['short_parity_row']['prompt_ids'],operation)
                    return value
                e.call=wrapped
            f.mutate=mutate
            with self.assertRaisesRegex(ValueError,'frozen role/source dose|physical prefix roster'):f.run('trace1')
            self.assertTrue(f.controller.halted);self.assertEqual(f.controller.capsules,[])
            self.assertEqual(f.sessions[0]['component'].max_attempted_forwards,34)
            self.assertEqual(f.sessions[0]['component'].max_input_positions,131382)
            if operation=='long_greedy_trace_forward':
                failed=f.saved['trace1_failed'];self.assertEqual(failed['actual_scope_costs']['attempted_forwards'],5)
                self.assertEqual(failed['actual_scope_costs']['input_positions']['attempted'],9386)
                self.assertEqual(len(failed['actual_observed_CPU_fixture_prefix_hashes']),5)

    def test_guard_forward_and_sink_failures_halt_no_retry(self):
        f=Fixture()
        def fault(s,r):s['model'].fault=lambda *args:(_ for _ in ()).throw(RuntimeError('CPU_FORWARD_FAIL'))
        f.mutate=fault
        with self.assertRaisesRegex(RuntimeError,'CPU_FORWARD_FAIL'):f.run('trace1')
        self.assertEqual(f.saved['trace1_failed']['actual_scope_costs']['attempted_forwards'],1)
        with self.assertRaisesRegex(ValueError,'halted'):f.run('trace1')
        f=Fixture();f.fail_sink=lambda name,value:(_ for _ in ()).throw(OSError('CPU_SINK_FAIL')) if name=='trace1_token_01' else None
        with self.assertRaisesRegex(OSError,'CPU_SINK_FAIL'):f.run('trace1')
        self.assertTrue(f.controller.halted);self.assertEqual(f.saved['trace1_failed']['actual_scope_costs']['attempted_forwards'],3)

    def test_final_sink_deadline_and_call_ledger_mutation_cannot_commit_lineage(self):
        for deadline in (True,False):
            f=Fixture()
            def sink(name,value):
                if name=='trace1_complete':
                    if deadline:f.clock_value=601.
                    else:f.sessions[0]['model'].calls.append([5])
            f.fail_sink=sink
            with self.assertRaisesRegex(ValueError,'600s|sink changed actual|physical prefix roster'):f.run('trace1')
            self.assertTrue(f.controller.halted);self.assertEqual(f.controller.capsules,[])
            self.assertEqual(f.controller.latest_sha256,f.recipe['sha256'])

    def test_hash_conventions_and_no_decoder_or_framework(self):
        f=Fixture();f.complete()
        self.assertEqual(f.recipe['prompt_hash_conventions'],{'short':'SHA256_RENDERED_UTF8_CHAT_TEMPLATE','long':'SHA256_CANONICAL_MESSAGES_JSON'})
        self.assertTrue(all(session['component'].tokenizer.calls==[] for session in f.sessions))
        self.assertFalse(any(n in sys.modules for n in ('torch','transformers','tokenizers')))


if __name__=='__main__':unittest.main()
