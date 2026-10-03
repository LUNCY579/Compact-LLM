from __future__ import annotations
import contextlib
import copy
import io
import itertools
import json
import random
import tempfile
import unittest
from pathlib import Path
from forge_rl.common import FAMILIES, SYSTEM_FINAL, SYSTEM_REASON, Writer, digest, dumps, loads, read_jsonl, sha_file
from forge_rl.tasks import make_world, solve, trace_text
from forge_rl.text import render, parse_prompt, canonical_case, LAYOUTS
from forge_rl.oracle import audit_answer, verify_solution, replay_transition, verify_trace, check_world
from forge_rl.rewards import score_response, parse_response
from forge_rl.environment import PlanningEnvironment
from forge_rl.prepare import prepare_record, tokenizer_inventory
from forge_rl.build import build_data, negative_answer
from forge_rl.validate import validate, verify_manifest
from forge_rl.evaluate import evaluate
from forge_rl.export import export
from forge_rl.process_eval import evaluate_process
from forge_rl.adapters import VerifiableReward


def route_fixture():
    r=[]
    for i,(a,b,c) in enumerate([('N0','N1',1),('N1','N3',2),('N0','N2',1),('N2','N3',2),('N3','N0',1)],1):
        r.append(dict(kind='edge',src=a,dst=b,cost=c,tick=i,status='approved',open=True))
    return dict(family='temporal_route',query=dict(nodes=['N0','N1','N2','N3'],start='N0',goal='N3',waypoints=[],cutoff=99),records=r)


def plan_fixture():
    return dict(family='prerequisite_plan',query=dict(bits=3,initial=0,goal_on=6,goal_off=1),records=[
        dict(kind='action',name='prepare',pre_on=0,pre_off=0,add=1,delete=0,cost=1),
        dict(kind='action',name='build',pre_on=1,pre_off=0,add=2,delete=1,cost=2),
        dict(kind='action',name='finish',pre_on=2,pre_off=1,add=4,delete=0,cost=1),
        dict(kind='action',name='expensive',pre_on=0,pre_off=0,add=6,delete=1,cost=9)])


class KnownAnswers(unittest.TestCase):
    def test_ledger_signed_void_future_scale(self):
        w=dict(family='ledger_replay',query=dict(account='A',cutoff=3),records=[dict(kind='initial',account=k,amount=10 if k=='A' else 0) for k in 'ABCD']+[
            dict(kind='transfer',tick=1,account='A',other='B',amount=-3,status='posted'),
            dict(kind='scale',tick=2,account='A',other='B',amount=-2,status='posted'),
            dict(kind='add',tick=3,account='A',other='C',amount=99,status='void'),
            dict(kind='add',tick=4,account='A',other='C',amount=99,status='posted')])
        self.assertEqual(solve(w)[0],{'value':-26}); self.assertEqual(audit_answer(w),{'value':-26})

    def test_register_strict_condition_and_mod(self):
        before={'X':1,'Y':4,'Z':2}
        e=dict(kind='conditional',tick=1,register='X',other='Y',value=8,threshold=4)
        self.assertEqual(replay_transition('register_machine',before,e),{'X':90,'Y':4,'Z':2})

    def test_route_alternative_optima(self):
        w=route_fixture(); cert=audit_answer(w)
        self.assertEqual(cert,{'optimal_cost':3})
        for path in (['N0','N1','N3'],['N0','N2','N3']):
            self.assertTrue(verify_solution(w,{'path':path,'cost':3},cert)['correct'])

    def test_route_revisit_for_waypoint(self):
        w=route_fixture(); w['query'].update(start='N0',goal='N0',waypoints=['N3'])
        self.assertEqual(audit_answer(w),{'optimal_cost':4})
        self.assertTrue(verify_solution(w,solve(w)[0])['correct'])

    def test_route_drafts_and_future_do_not_replace(self):
        w=route_fixture()
        w['records'] += [dict(kind='edge',src='N0',dst='N1',cost=90,tick=20,status='draft',open=False),dict(kind='edge',src='N0',dst='N2',cost=90,tick=100,status='approved',open=False)]
        self.assertEqual(audit_answer(w),{'optimal_cost':3})

    def test_route_latest_closure(self):
        w=route_fixture()
        for i,d in enumerate(['N1','N2'],20):
            w['records'].append(dict(kind='edge',src='N0',dst=d,cost=1,tick=i,status='approved',open=False))
        self.assertEqual(audit_answer(w),{'status':'IMPOSSIBLE'})

    def test_workflow_delete_and_optimality(self):
        w=plan_fixture(); a,s=solve(w)
        self.assertEqual(a,{'actions':['prepare','build','finish'],'cost':4})
        self.assertEqual(audit_answer(w),{'optimal_cost':4}); self.assertTrue(verify_trace(w,a,s))
        r=verify_solution(w,{'actions':['expensive'],'cost':9})
        self.assertTrue(r['feasible']); self.assertFalse(r['correct'])

    def test_workflow_invalid_precondition(self):
        self.assertFalse(verify_solution(plan_fixture(),{'actions':['finish'],'cost':1})['feasible'])

    def test_workflow_initial_goal_empty(self):
        w=plan_fixture(); w['query']['initial']=6
        self.assertTrue(verify_solution(w,{'actions':[],'cost':0})['correct'])

    def test_schedule_release_precedence_deadline(self):
        w=dict(family='deadline_schedule',query=dict(start_time=0),records=[dict(kind='job',name='A',duration=2,release=3,deadline=8,after=[]),dict(kind='job',name='B',duration=1,release=0,deadline=6,after=['A'])])
        self.assertEqual(audit_answer(w),{'optimal_finish':6})
        self.assertTrue(verify_solution(w,{'order':['A','B'],'finish':6})['correct'])
        w['records'][1]['deadline']=5
        self.assertEqual(audit_answer(w),{'status':'IMPOSSIBLE'})

    def test_schedule_alternative_tie_and_duplicate_rejection(self):
        w=dict(family='deadline_schedule',query=dict(start_time=0),records=[dict(kind='job',name=n,duration=1,release=0,deadline=5,after=[]) for n in 'AB'])
        self.assertTrue(verify_solution(w,{'order':['B','A'],'finish':2})['correct'])
        self.assertFalse(verify_solution(w,{'order':['A','A'],'finish':2})['correct'])

    def test_budget_dependencies_and_tie_cost(self):
        w=dict(family='budget_selection',query=dict(cutoff=10),records=[dict(kind='item',name='A',cost=2,weight=1,value=2,requires=[],excludes=[]),dict(kind='item',name='B',cost=2,weight=1,value=7,requires=['A'],excludes=[]),dict(kind='item',name='C',cost=5,weight=1,value=9,requires=[],excludes=['A']),dict(kind='limit',key='budget',tick=1,value=5,status='approved'),dict(kind='limit',key='capacity',tick=2,value=2,status='approved')])
        self.assertEqual(audit_answer(w),{'optimal_value':9,'optimal_cost':4})
        self.assertTrue(verify_solution(w,{'items':['B','A'],'value':9,'cost':4})['correct'])
        self.assertFalse(verify_solution(w,{'items':['B'],'value':7,'cost':2})['feasible'])
        w['records'][-1]['status']='draft'
        self.assertEqual(audit_answer(w),{'status':'UNKNOWN'})

    def test_evidence_missing_is_not_impossible(self):
        w=dict(family='evidence_chain',query=dict(start='E0',path=['owner','site'],cutoff=5),records=[dict(kind='field',entity='E0',key='owner',value='E1',tick=1,status='approved'),dict(kind='field',entity='E1',key='site',value='S3',tick=2,status='draft')])
        self.assertEqual(audit_answer(w),{'status':'UNKNOWN'})
        self.assertFalse(verify_solution(w,{'status':'IMPOSSIBLE'})['correct'])
        w['records'][1]['status']='approved'
        self.assertEqual(audit_answer(w),{'answer':'S3'})

    def test_structured_trace_mutation_detected(self):
        w=plan_fixture();a,s=solve(w);s[1]['after'] ^= 1
        self.assertFalse(verify_trace(w,a,s))


class GeneratedCases(unittest.TestCase):
    pass


def make_generated_test(family,seed,tier):
    def test(self):
        world=make_world(family,random.Random(seed),tier,seed%3==0)
        answer,steps=solve(world)
        for layout in LAYOUTS:
            prompt,meta=render(world,random.Random(seed+991),tier,layout)
            parsed=parse_prompt(prompt)
            self.assertEqual(digest(canonical_case(world)),digest(canonical_case(parsed)))
            cert=audit_answer(parsed)
            self.assertTrue(verify_solution(parsed,answer,cert)['correct'])
            self.assertTrue(verify_trace(parsed,answer,steps))
            self.assertEqual(score_response(parsed,dumps(answer),certificate=cert)['reward'],1)
            self.assertEqual(score_response(parsed,negative_answer(answer),certificate=cert)['reward'],0)
            trace=trace_text(parsed,answer,steps)
            out='<think>'+trace+'</think><final>'+dumps(answer)+'</final>'
            self.assertEqual(score_response(parsed,out,mode='reason_then_final',certificate=cert)['reward'],1)
            self.assertEqual(score_response(parsed,out,mode='final_only',certificate=cert)['reward'],0)
    return test

for _f in FAMILIES:
    for _seed in range(4):
        for _tier in range(4):
            setattr(GeneratedCases,f'test_{_f}_seed{_seed}_tier{_tier}',make_generated_test(_f,_seed,_tier))


class Strictness(unittest.TestCase):
    def test_duplicate_json_key(self):
        with self.assertRaises(ValueError): loads('{"value":1,"value":2}')
    def test_nan(self):
        with self.assertRaises(ValueError): loads('{"value":NaN}')
    def test_boolean_is_not_integer(self):
        w=make_world('register_machine',random.Random(1),0)
        self.assertFalse(verify_solution(w,{'value':True})['schema_valid'])
    def test_extra_keys_are_not_concise(self):
        w=plan_fixture(); a=solve(w)[0];a['explanation']='because'
        self.assertFalse(verify_solution(w,a)['schema_valid'])
    def test_json_fence_rejected(self):
        with self.assertRaises(ValueError):parse_response('```json\n{"value":1}\n```')
    def test_multiple_final_blocks_rejected(self):
        with self.assertRaises(ValueError):parse_response('<think>x</think><final>{}</final><final>{}</final>','reason_then_final')
    def test_reasoning_suffix_contamination(self):
        with self.assertRaises(ValueError):parse_response('<think>x</think><final>{}</final> extra','reason_then_final')
    def test_empty_trace_rejected(self):
        with self.assertRaises(ValueError):parse_response('<think></think><final>{}</final>','reason_then_final')
    def test_oversized_response(self):
        with self.assertRaises(ValueError):parse_response(' '*100001)
    def test_unknown_mode(self):
        with self.assertRaises(ValueError):parse_response('{}','auto')
    def test_negative_cost_rejected(self):
        self.assertFalse(verify_solution(plan_fixture(),{'actions':[],'cost':-1})['schema_valid'])
    def test_fractional_cost_rejected(self):
        self.assertFalse(verify_solution(plan_fixture(),{'actions':[],'cost':1.0})['schema_valid'])
    def test_short_wrong_not_rewarded(self):
        self.assertEqual(score_response(plan_fixture(),'{}',efficiency_weight=.05)['reward'],0)
    def test_length_shaping_never_outweighs_correctness(self):
        w=plan_fixture();a=solve(w)[0]
        s=json.dumps(a,indent=400)
        self.assertGreaterEqual(score_response(w,s,efficiency_weight=.05)['reward'],.95)
    def test_token_counts_must_be_paired(self):
        with self.assertRaises(ValueError):score_response(plan_fixture(),'{}',trace_tokens=20)
    def test_efficiency_cap(self):
        with self.assertRaises(ValueError):score_response(plan_fixture(),'{}',efficiency_weight=.9)
    def test_prompt_duplicate_query(self):
        p,_=render(plan_fixture(),random.Random(1),0,'early');p[1]['content']+='\nQUERY {}'
        with self.assertRaises(ValueError):parse_prompt(p)
    def test_prompt_missing_scope(self):
        p,_=render(plan_fixture(),random.Random(1),0,'early');p[1]['content']='\n'.join(l for l in p[1]['content'].splitlines() if not l.startswith('SCOPE '))
        with self.assertRaises(ValueError):parse_prompt(p)
    def test_duplicate_record_ids(self):
        p,_=render(plan_fixture(),random.Random(1),0,'early');line=next(l for l in p[1]['content'].splitlines() if l.startswith('Record '));p[1]['content']+='\n'+line
        with self.assertRaises(ValueError):parse_prompt(p)
    def test_unbounded_workflow_rejected(self):
        w=plan_fixture();w['query']['bits']=30
        with self.assertRaises(ValueError):audit_answer(w)
    def test_duplicate_action_names_rejected(self):
        w=plan_fixture();w['records'].append(w['records'][0].copy())
        with self.assertRaises(ValueError):audit_answer(w)
    def test_zero_edge_cost_rejected(self):
        w=route_fixture();w['records'][0]['cost']=0
        with self.assertRaises(ValueError):audit_answer(w)
    def test_rules_mutation_rejected(self):
        p,_=render(plan_fixture(),random.Random(1),0,'early');p[1]['content']=p[1]['content'].replace('costs are positive integers','costs are negative integers')
        with self.assertRaises(ValueError):parse_prompt(p)


class EnvironmentTests(unittest.TestCase):
    def test_optimal_trajectory(self):
        env=PlanningEnvironment(plan_fixture())
        rewards=[]
        for a in ['prepare','build','finish','STOP']:
            obs,reward,done,trunc,info=env.step(a);rewards.append(reward)
        self.assertEqual(rewards,[0,0,0,1]);self.assertTrue(done);self.assertFalse(trunc)
        self.assertEqual(obs['state'],6)
    def test_reset_parity_and_no_oracle_leak(self):
        env=PlanningEnvironment(plan_fixture());first=env.reset();env.step('prepare')
        self.assertEqual(first,env.reset());self.assertNotIn('optimal_cost',first);self.assertNotIn('reference',first)
    def test_illegal_action_does_not_change_state(self):
        env=PlanningEnvironment(plan_fixture());obs,r,d,t,i=env.step('finish')
        self.assertEqual(obs['state'],0);self.assertTrue(d);self.assertEqual(r,0)
    def test_suboptimal_goal_not_rewarded(self):
        env=PlanningEnvironment(plan_fixture());env.step('expensive');self.assertEqual(env.step('STOP')[1],0)
    def test_step_budget(self):
        env=PlanningEnvironment(plan_fixture(),max_steps=1);out=env.step('prepare');self.assertTrue(out[3]);self.assertEqual(out[1],0)
    def test_terminal_requires_reset(self):
        env=PlanningEnvironment(plan_fixture());env.step('STOP')
        with self.assertRaises(RuntimeError):env.step('prepare')
    def test_false_impossibility(self):
        self.assertEqual(PlanningEnvironment(plan_fixture()).step('IMPOSSIBLE')[1],0)
    def test_true_impossibility(self):
        w=plan_fixture();w['records']=w['records'][:1]
        self.assertEqual(PlanningEnvironment(w).step('IMPOSSIBLE')[1],1)


class StubTokenizer:
    def apply_chat_template(self,messages,tokenize,add_generation_prompt,**kwargs):
        text=''.join(m['role']+':'+m['content']+'|' for m in messages if m['role']!='assistant')+'assistant:'
        prefix=[ord(x)+10 for x in text]
        if add_generation_prompt:return prefix
        return prefix+[ord(x)+10 for x in messages[-1]['content']]+[3]


class PreparationTests(unittest.TestCase):
    def row(self):
        return dict(id='a',group_id='a',split='train',messages=[dict(role='system',content='s'),dict(role='user',content='u'),dict(role='assistant',content='{}')])
    def test_mask_prompt_answer_eos(self):
        out,r=prepare_record(self.row(),StubTokenizer(),kind='sft',max_length=1000,completion_reserve=32,terminator_id=3)
        self.assertTrue(all(x==-100 for x in out['labels'][:r['prompt_tokens']]))
        self.assertEqual(out['labels'][-1],3);self.assertEqual(r['loss_tokens'],3)
    def test_overlength_not_truncated(self):
        out,r=prepare_record(self.row(),StubTokenizer(),kind='sft',max_length=2,completion_reserve=32,terminator_id=3)
        self.assertIsNone(out);self.assertFalse(r['truncated'])
    def test_prefix_mismatch_fails_closed(self):
        class Bad(StubTokenizer):
            def apply_chat_template(self,messages,tokenize,add_generation_prompt,**kwargs):
                out=super().apply_chat_template(messages,tokenize,add_generation_prompt,**kwargs)
                return out if add_generation_prompt else [0]+out
        with self.assertRaises(ValueError):prepare_record(self.row(),Bad(),kind='sft',max_length=1000,completion_reserve=32,terminator_id=3)
    def test_wrong_terminator(self):
        with self.assertRaises(ValueError):prepare_record(self.row(),StubTokenizer(),kind='sft',max_length=1000,completion_reserve=32,terminator_id=9999)
    def test_test_sft_rejected(self):
        r=self.row();r['split']='test_iid'
        with self.assertRaises(ValueError):prepare_record(r,StubTokenizer(),kind='sft',max_length=1000,completion_reserve=32,terminator_id=3)
    def test_rl_reserves_generation(self):
        r=self.row();r['prompt']=r.pop('messages')[:-1]
        out,rep=prepare_record(r,StubTokenizer(),kind='rl',max_length=64,completion_reserve=60,terminator_id=None)
        self.assertIsNone(out);self.assertGreater(rep['required_context'],64)
    def test_template_cannot_enable_truncation(self):
        with self.assertRaises(ValueError):prepare_record(self.row(),StubTokenizer(),kind='sft',max_length=1000,completion_reserve=32,terminator_id=3,template_kwargs={'truncation':True})
    def test_assets_inventory_excludes_weights(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'tokenizer.json').write_text('{}');(p/'model.safetensors').write_text('weights')
            result=tokenizer_inventory(p);self.assertEqual(set(result['files']),{'tokenizer.json'})
    def test_no_tokenizer_assets_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):tokenizer_inventory(Path(d))


class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.base=Path(cls.temp.name);cls.root=cls.base/'a'
        with contextlib.redirect_stdout(io.StringIO()):build_data(cls.root,scale=.002)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_full_smoke_validation(self):
        r=validate(self.root,emit_progress=False);self.assertEqual(r['status'],'PASS');self.assertEqual(r['independent_worlds'],64)
    def test_reproducible_bytes(self):
        root=self.base/'rebuild'
        with contextlib.redirect_stdout(io.StringIO()):build_data(root,scale=.002)
        self.assertEqual({str(p.relative_to(self.root)):sha_file(p) for p in self.root.rglob('*') if p.is_file()}, {str(p.relative_to(root)):sha_file(p) for p in root.rglob('*') if p.is_file()})
    def test_build_refuses_existing_directory(self):
        with self.assertRaises(FileExistsError):build_data(self.root,scale=.002)
    def test_build_invalid_scale(self):
        with self.assertRaises(ValueError):build_data(self.base/'never',scale=float('nan'))
    def test_manifest_detects_corruption(self):
        f=self.root/'examples/train_samples.json';original=f.read_bytes()
        try:
            f.write_bytes(original+b' ')
            with self.assertRaises(ValueError):verify_manifest(self.root)
        finally:f.write_bytes(original)
    def test_reward_adapter_batch_alignment(self):
        adapter=VerifiableReward(self.root);r=next(read_jsonl(self.root/'data/rl/train.jsonl.gz'));m=next(read_jsonl(self.root/'metadata/train.jsonl.gz'))
        self.assertEqual(adapter([m['reference_text']],[r['id']],[r['prompt']]),[1])
        with self.assertRaises(ValueError):adapter([m['reference_text']],[],[r['prompt']])
    def test_reward_adapter_wrong_prompt(self):
        adapter=VerifiableReward(self.root);r=next(read_jsonl(self.root/'data/rl/train.jsonl.gz'))
        with self.assertRaises(ValueError):adapter(['{}'],[r['id']],[[{'role':'user','content':'different'}]])
    def test_adapter_blocks_tests(self):
        with self.assertRaises(ValueError):VerifiableReward(self.root,'test_iid')
    def test_gold_selfscore_not_model_result(self):
        p=self.base/'gold.jsonl'
        with Writer(p) as out:
            for r in read_jsonl(self.root/'metadata/test_iid.jsonl.gz'):out.write({'id':r['id'],'prediction':r['reference_text']})
        rep=evaluate(self.root,p,'test_iid');self.assertEqual(rep['overall']['accuracy_missing_as_wrong'],1);self.assertTrue(rep['full_coverage'])
    def test_missing_predictions_count_as_wrong(self):
        p=self.base/'empty.jsonl';p.write_text('')
        rep=evaluate(self.root,p,'test_iid');self.assertEqual(rep['overall']['accuracy_missing_as_wrong'],0);self.assertFalse(rep['full_coverage'])
    def test_duplicate_predictions_fail(self):
        p=self.base/'dup.jsonl';r=next(read_jsonl(self.root/'metadata/test_iid.jsonl.gz'));line=dumps({'id':r['id'],'prediction':r['reference_text']})+'\n';p.write_text(line*2)
        with self.assertRaises(ValueError):evaluate(self.root,p,'test_iid')
    def test_truncated_prefix_not_counted_correct(self):
        p=self.base/'trunc.jsonl';r=next(read_jsonl(self.root/'metadata/test_iid.jsonl.gz'));p.write_text(dumps({'id':r['id'],'prediction':r['reference_text'],'finish_reason':'length'})+'\n')
        rep=evaluate(self.root,p,'test_iid');self.assertEqual(rep['overall']['correct'],0);self.assertEqual(rep['overall']['truncated'],1)
    def test_pass_at_k(self):
        p=self.base/'samples.jsonl'
        with Writer(p) as out:
            for r in read_jsonl(self.root/'metadata/test_iid.jsonl.gz'):
                for i in range(2):out.write({'id':r['id'],'sample_id':i,'prediction':r['reference_text'] if i else '{}'})
        rep=evaluate(self.root,p,'test_iid',samples_per_prompt=2)
        self.assertEqual(rep['pass_at_k']['1']['estimate_complete_groups_only'],.5);self.assertEqual(rep['pass_at_k']['2']['estimate_complete_groups_only'],1)
    def test_preferences_export_no_target_in_prompt(self):
        p=self.base/'dpo.jsonl';n=export(self.root,p,'preferences','train');self.assertEqual(n,64)
        r=next(read_jsonl(p));self.assertEqual([m['role'] for m in r['prompt']],['system','user'])
    def test_episode_export_strips_oracle(self):
        p=self.base/'episodes.jsonl';export(self.root,p,'episodes','train')
        for r in read_jsonl(p):self.assertEqual(set(r),{'id','group_id','split','stage','prompt'})
    def test_process_export_labels_not_in_prompt(self):
        p=self.base/'process.jsonl';export(self.root,p,'process','train');r=next(read_jsonl(p));self.assertNotIn('"valid":',r['prompt'][1]['content'])
    def test_test_export_requires_explicit_eval_flag(self):
        with self.assertRaises(ValueError):export(self.root,self.base/'bad.jsonl','rl','test_iid')
    def test_test_sft_export_always_blocked(self):
        with self.assertRaises(ValueError):export(self.root,self.base/'bad2.jsonl','sft_final','test_iid',True)
    def test_process_evaluation_gold(self):
        p=self.base/'process_gold.jsonl';refs=self.root/'process/train_transitions.jsonl.gz'
        with Writer(p) as out:
            for r in read_jsonl(refs):out.write({'id':r['id'],'prediction':dumps({'valid':r['valid']})})
        rep=evaluate_process(refs,p);self.assertEqual(rep['accuracy_missing_as_wrong'],1)
    def test_export_refuses_existing(self):
        p=self.base/'existing.jsonl';p.write_text('keep')
        with self.assertRaises(FileExistsError):export(self.root,p,'rl','train')
        self.assertEqual(p.read_text(),'keep')


if __name__=='__main__':unittest.main()
