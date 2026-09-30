"""Analytic synthetic controls; no tokenizer, model, GPU or capability claims."""
from copy import deepcopy
import math
import random
import unittest

from experiments.post_training_mimo_centralized_v1.scripts import e1_data, e1_schedule
from experiments.post_training_mimo_centralized_v1.scripts import e1_training_engine as e
from experiments.post_training_mimo_centralized_v1.scripts.e1_training_accounting import E1CallAccounting


def array(planned):
    completion = [151645] if planned['component']=='acquisition' else [2,3,151645]
    ids=[1,2,3]+completion;p=3;c=len(completion);n=len(ids)
    return {'row_id':planned['key'].split(':',1)[1],'parent_id':planned['parent_id'],'family':'synthetic_control',
        'role':'TRAIN','training_role_eligible':True,'prompt_tokens':p,'completion_tokens':c,'nonpadding_tokens':n,
        'padding_tokens':0,'completion_eos_id':151645,'input_ids':ids,'labels':[-100]*p+completion,
        'attention_mask':[1]*n,'completion_mask':[0]*p+[1]*c,'input_token_ids_sha256':e1_data.digest(ids),
        'tokenized_prompt_sha256':e1_data.digest(ids[:p]),'completion_token_ids_sha256':e1_data.digest(completion),
        'source_row_sha256':'a'*64}


class AnalyticBackend:
    """Scalar mean target MSE+2, with explicit Adam moments and optional RNG.

    It tests orchestration and reduction algebra, not Rineya or native CE math.
    Unequal target counts separate equal-sequence means from pooled-token means.
    """
    execution_scope='SYNTHETIC_COMPONENT_CONTROL'
    def __init__(self,stochastic=False):
        self.w=.9;self.m=self.v=self.grad=0.;self.t=0
        self.rng=random.Random(101);self.stochastic=stochastic
        self.keys=[];self.scales=[];self.norms=[];self.updates=[];self.saves=[];self.resets=0
        self.fail_forward=None;self.fail_backward=None;self.fail_save=None;self.bad_loss=False
    def forward_loss(self,row):
        self.keys.append(row['row_id'])
        if self.fail_forward==len(self.keys):raise RuntimeError('injected forward failure')
        targets=[(label%7)/10 for label in row['labels'] if label!=-100]
        noise=self.rng.random()*.001 if self.stochastic else 0.
        return {'value':sum((self.w-target)**2/2+2 for target in targets)/len(targets),
                'gradient':self.w-sum(targets)/len(targets)+noise}
    def loss_value(self,loss):return float('nan') if self.bad_loss else loss['value']
    def backward(self,loss,*,scale):
        self.scales.append(scale)
        if self.fail_backward==len(self.scales):raise RuntimeError('injected backward failure')
        self.grad+=loss['gradient']*scale
    def reset_gradients(self):self.grad=0.;self.resets+=1
    def clip_gradients(self,*,max_norm):
        norm=abs(self.grad);self.norms.append(norm)
        self.grad=max(-max_norm,min(max_norm,self.grad));return norm
    def optimizer_step(self,*,lr):
        before=self.w;old_m=self.m
        self.t+=1;self.m=.9*self.m+.1*self.grad;self.v=.999*self.v+.001*self.grad**2
        self.w-=lr*(self.m/(1-.9**self.t))/(math.sqrt(self.v/(1-.999**self.t))+1e-8)
        self.updates.append({'lr':lr,'w_before':before,'w_after':self.w,'m_before':old_m,'m_after':self.m})
    def save_checkpoint(self,cursor,accounting):
        if self.fail_save==cursor['completed_optimizer_steps']:raise OSError('injected checkpoint failure')
        self.saves.append((deepcopy(cursor),deepcopy(accounting)))
        return {'parameters':{'w':self.w},'optimizer':{'m':self.m,'v':self.v,'step':self.t},
                'training_rng':{'python':self.rng.getstate(),'stochastic':self.stochastic}}
    def restore_checkpoint(self,state):
        self.w=state['parameters']['w'];self.m=state['optimizer']['m'];self.v=state['optimizer']['v'];self.t=state['optimizer']['step']
        self.rng.setstate(state['training_rng']['python']);self.stochastic=state['training_rng']['stochastic'];self.grad=0.


class EngineControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows=[]
        for component,count in e1_schedule.COMPONENT_COUNTS.items():
            c=1 if component=='acquisition' else 3
            rows += [{'key':f'{component}:{component}_{index}','component':component,'role':'TRAIN',
                      'parent_id':f'{component}_parent_{index}','nonpadding_tokens':3+c,'completion_tokens':c}
                     for index in range(count)]
        cls.plan=e1_schedule.build_plan(rows,identities={key:'a'*64 for key in e1_schedule.IDENTITY_FIELDS})
        cls.rows={row['key']:row for row in rows}
    def ledger(self):return E1CallAccounting(lambda *args,**kwargs:None)
    def get(self,key):return array(self.rows[key])

    def test_full_order_dose_lr_and_state_advancement(self):
        backend=AnalyticBackend();result=e.run(self.plan,self.get,backend,self.ledger())
        self.assertEqual(result['status'],'COMPLETE_ENGINE_COMPONENT_UNQUALIFIED')
        expected=[key.split(':',1)[1] for step in self.plan['steps'] for key in step['row_keys']]
        self.assertEqual(backend.keys,expected);self.assertEqual(len(backend.keys),5120)
        self.assertEqual(backend.scales,[1/20]*5120);self.assertEqual(backend.resets,256);self.assertEqual(len(backend.norms),256)
        self.assertEqual([entry['lr'] for entry in backend.updates],[step['learning_rate'] for step in self.plan['steps']])
        self.assertEqual(backend.t,256);self.assertEqual(backend.updates[-1]['lr'],0.)
        self.assertEqual(backend.updates[-1]['w_before'],backend.updates[-1]['w_after'])
        self.assertNotEqual(backend.updates[-1]['m_before'],backend.updates[-1]['m_after'])
        counts=result['call_accounting'];self.assertEqual(counts['positions']['input_positions_returned'],22528)
        self.assertEqual(counts['positions']['supervised_positions_returned'],7168)
        self.assertEqual([s[0]['completed_optimizer_steps'] for s in backend.saves],[0,64,128,192,256])

    def test_equal_sequence_means_not_pooled_target_means(self):
        backend=AnalyticBackend();result=e.run(self.plan,self.get,backend,self.ledger(),stop_after_update=64)
        self.assertEqual(result['status'],'STOPPED_AT_SAVED_COMPONENT_BOUNDARY')
        # Sixteen one-target acquisition sequences have gradient .5; four
        # three-target replay sequences have mean gradient .6 at starting w=.9.
        expected=(16*.5+4*.6)/20
        wrong_pooled=(16*.5+4*3*.6)/(16+4*3)
        self.assertAlmostEqual(backend.norms[0],expected,places=14)
        self.assertNotAlmostEqual(backend.norms[0],wrong_pooled,places=6)

    def test_controlled_resume_restores_parameters_moments_rng_and_cumulative_costs(self):
        whole=AnalyticBackend(stochastic=True);full=e.run(self.plan,self.get,whole,self.ledger())
        first=AnalyticBackend(stochastic=True);partial=e.run(self.plan,self.get,first,self.ledger(),stop_after_update=64)
        checkpoint=partial['checkpoint_envelopes'][-1]
        restored=AnalyticBackend(stochastic=False);restored.w=-17;restored.rng.seed(999)
        ledger=E1CallAccounting.from_snapshot(lambda *args,**kwargs:None,checkpoint['cost_snapshot'])
        result=e.run(self.plan,self.get,restored,ledger,resume=checkpoint,latest_terminal_costs=checkpoint['cost_snapshot'])
        self.assertEqual(result['status'],'COMPLETE_ENGINE_COMPONENT_UNQUALIFIED')
        self.assertEqual((restored.w,restored.m,restored.v,restored.t,restored.rng.getstate()),
                         (whole.w,whole.m,whole.v,whole.t,whole.rng.getstate()))
        self.assertEqual(result['returned_sequence_backwards_this_invocation'],3840)
        self.assertEqual(result['call_accounting']['operations']['optimizer_step']['returned'],256)
        self.assertEqual(result['call_accounting']['operations']['checkpoint_reload']['returned'],1)
        self.assertEqual(result['call_accounting']['operations']['forward']['returned'],5120)

    def test_checkpoint_has_pre_save_inputs_and_post_save_cost_binding(self):
        backend=AnalyticBackend();result=e.run(self.plan,self.get,backend,self.ledger(),stop_after_update=0)
        envelope=result['checkpoint_envelopes'][0]
        self.assertEqual(backend.saves[0][1]['operations']['checkpoint_save']['attempted'],0)
        self.assertEqual(envelope['cost_snapshot']['operations']['checkpoint_save']['returned'],1)
        self.assertEqual(envelope['save_call_id'],envelope['cost_snapshot']['last_call_id'])
        self.assertEqual(e.validate_checkpoint(envelope,self.plan),0)

    def test_arbitrary_partial_stop_refused_before_backend_calls(self):
        backend=AnalyticBackend()
        for boundary in (True,1,63,257):
            with self.assertRaises(ValueError):e.run(self.plan,self.get,backend,self.ledger(),stop_after_update=boundary)
        self.assertEqual(backend.saves,[]);self.assertEqual(backend.keys,[])

    def test_cursor_only_or_missing_state_refused(self):
        result=e.run(self.plan,self.get,AnalyticBackend(),self.ledger(),stop_after_update=0)
        checkpoint=result['checkpoint_envelopes'][0]
        for component in e.STATE_FIELDS:
            changed=deepcopy(checkpoint);changed['backend_state'].pop(component)
            with self.assertRaises(ValueError):e.validate_checkpoint(changed,self.plan)
        with self.assertRaises(ValueError):e.validate_checkpoint(checkpoint['cursor'],self.plan)

    def test_uncontrolled_rollback_later_costs_or_missing_terminal_refused(self):
        result=e.run(self.plan,self.get,AnalyticBackend(),self.ledger(),stop_after_update=0)
        checkpoint=result['checkpoint_envelopes'][0]
        restored=E1CallAccounting.from_snapshot(lambda *args,**kw:None,checkpoint['cost_snapshot'])
        with self.assertRaises(ValueError):e.run(self.plan,self.get,AnalyticBackend(),restored,resume=checkpoint)
        later=deepcopy(checkpoint['cost_snapshot']);later['last_call_id']+=1
        with self.assertRaises(ValueError):e.run(self.plan,self.get,AnalyticBackend(),restored,resume=checkpoint,latest_terminal_costs=later)

    def test_forward_failure_preserves_partial_work(self):
        backend=AnalyticBackend();backend.fail_forward=7
        result=e.run(self.plan,self.get,backend,self.ledger())
        self.assertEqual(result['status'],'FAILED_ENGINE_COMPONENT_UNQUALIFIED')
        self.assertEqual(result['completed_optimizer_steps'],0)
        self.assertEqual(result['call_accounting']['operations']['forward'],dict(attempted=7,returned=6,failed=1,pending=0))
        self.assertEqual(result['returned_sequence_backwards_this_invocation'],6)
        self.assertIs(result['automatic_resume_authorized'],False)

    def test_backward_failure_and_nonfinite_loss_do_not_update(self):
        for mode in ('backward','loss'):
            backend=AnalyticBackend()
            if mode=='backward':backend.fail_backward=3
            else:backend.bad_loss=True
            result=e.run(self.plan,self.get,backend,self.ledger())
            self.assertEqual(result['status'],'FAILED_ENGINE_COMPONENT_UNQUALIFIED')
            self.assertEqual(result['call_accounting']['operations']['optimizer_step']['attempted'],0)
            self.assertEqual(result['call_accounting']['operations']['forward']['returned'],3 if mode=='backward' else 1)

    def test_checkpoint_failure_retains_returned_updates(self):
        backend=AnalyticBackend();backend.fail_save=64
        result=e.run(self.plan,self.get,backend,self.ledger())
        self.assertEqual(result['status'],'FAILED_ENGINE_COMPONENT_UNQUALIFIED')
        self.assertEqual(result['completed_optimizer_steps'],64)
        self.assertEqual(result['call_accounting']['operations']['checkpoint_save']['failed'],1)
        self.assertEqual(result['call_accounting']['operations']['optimizer_step']['returned'],64)

    def test_wrong_row_or_packing_rejected_before_forward(self):
        for mode in ('identity','packing'):
            def get(key):
                value=self.get(key)
                if mode=='identity':value['row_id']='wrong'
                else:value['position_ids']=list(range(value['nonpadding_tokens']))
                return value
            result=e.run(self.plan,get,AnalyticBackend(),self.ledger())
            self.assertEqual(result['status'],'FAILED_ENGINE_COMPONENT_UNQUALIFIED')
            self.assertEqual(result['call_accounting']['operations']['forward']['attempted'],0)

    def test_changed_plan_order_rejected_before_backend_calls(self):
        changed=deepcopy(self.plan);changed['steps'][0]['row_keys'].reverse()
        backend=AnalyticBackend()
        with self.assertRaises(ValueError):e.run(changed,self.get,backend,self.ledger())
        self.assertEqual(backend.keys,[])

    def test_return_marker_failure_does_not_erase_actual_optimizer_return(self):
        def sink(kind,**fields):
            if kind=='operation_returned' and fields['operation']=='optimizer_step':raise OSError('disk failure')
        backend=AnalyticBackend();result=e.run(self.plan,self.get,backend,E1CallAccounting(sink))
        self.assertEqual(result['status'],'FAILED_ENGINE_COMPONENT_UNQUALIFIED')
        self.assertEqual(result['returned_optimizer_calls_this_invocation'],1)
        self.assertEqual(result['call_accounting']['operations']['optimizer_step']['failed'],0)
        self.assertEqual(result['completed_optimizer_steps'],0)
        self.assertEqual(backend.t,1)

    def test_component_never_grants_scientific_or_native_authority(self):
        result=e.run(self.plan,self.get,AnalyticBackend(),self.ledger(),stop_after_update=0)
        for field in ('native_launch_authorized','native_resume_qualified','E1_authorized','CHECK_FINAL_access'):
            self.assertIs(result[field],False)


if __name__=='__main__':unittest.main()
