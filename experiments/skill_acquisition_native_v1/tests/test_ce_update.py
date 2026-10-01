"""Pure CE admission controls and deferred tiny actual CPU graph controls.

COMP_RS_RUN_NATIVE_CPU_TESTS=1 requires root resource clearance. This fixture
never loads a pretrained model or supplies native SFT authorization.
"""
import ast
import copy
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_update as ce


def recipe():
    return {'schema':ce.SCHEMA,'synthetic_cpu':True,'accumulation_rows':8,'microbatch_rows':1,
            'loss_reduction':ce.REDUCTION,'boundary_id':0,'generation':0,'U_before':0,
            'symbolic_snapshot':'b'*64,'start_policy_id':'a'*64,'trainable_names':['model.layers.0.linear.weight',
             'model.layers.0.linear.bias'],'eos_ids':[2],'target_eos_id':2,'pad_id':0,'max_target_tokens':8,
            'gradient_checkpointing':{'use_reentrant':False},'optimizer_origin':'FRESH_PER_ARM',
            'gradient_clipping':{'max_norm':1.0,'norm_type':2.0,'foreach':False,'error_if_nonfinite':True},
            'optimizer':{'class':'torch.optim.AdamW','lr':1e-5,'betas':[.9,.999],'eps':1e-8,
                         'weight_decay':0,'foreach':False,'capturable':False}}


def rows():
    result=[]
    for i,(P,C) in enumerate(zip((1,2,3,4,2,3,1,4),(1,2,4,7,3,6,2,5))):
        result.append({'id':'row%d'%i,'prompt_ids':list(range(7,7+P)),
                       'completion_ids':[3+(i+j)%6 for j in range(C-1)]+[2]})
    return result


class PureTests(unittest.TestCase):
    def test_native_scope_denied_before_model_access_even_with_grant_booleans(self):
        class Unreadable:
            def __getattribute__(self,key):raise AssertionError('model accessed before native refusal')
        r=recipe();r.update(synthetic_cpu=False,approved=True,training_authorized=True)
        with self.assertRaisesRegex(ValueError,'no separately admitted SFT scope'):
            ce.boundary_update(Unreadable(),Unreadable(),rows(),r,lambda *a,**k:None)

    def test_explicit_recipe_without_framework(self):
        ce.validate_recipe(recipe());ce.validate_rows(rows(),recipe())
        parsed=ast.parse(Path(ce.__file__).read_text())
        top=[name.name for node in parsed.body if isinstance(node,ast.Import) for name in node.names]
        self.assertFalse(set(top)&{'torch','transformers','numpy','safetensors'})

    def test_missing_or_ambiguous_optimizer_clip_reduction_refused(self):
        for key in ('optimizer','optimizer_origin','loss_reduction','gradient_clipping','gradient_checkpointing','U_before'):
            value=recipe();del value[key]
            with self.subTest(key=key),self.assertRaises(ValueError):ce.validate_recipe(value)
        value=recipe();value['gradient_clipping']['max_norm']=True
        with self.assertRaises(ValueError):ce.validate_recipe(value)

    def test_partial_batch_duplicate_rows_internal_and_missing_EOS_refused(self):
        for change in ('partial','duplicate','internal','missing','padding'):
            data=rows()
            if change=='partial':data.pop()
            if change=='duplicate':data[7]['id']=data[0]['id']
            if change=='internal':data[7]['completion_ids'][0]=2
            if change=='missing':data[7]['completion_ids'][-1]=3
            if change=='padding':data[7]['attention_mask']=[1]
            with self.subTest(change=change),self.assertRaises(ValueError):ce.validate_rows(data,recipe())

    def test_equal_row_token_means_differ_from_pooled_token_mean(self):
        values=[[-1.0],[-3.0]*7]+[[-2.0]*2]*6
        expected=(1+3+6*2)/8
        self.assertEqual(ce.loss_from_logprobs(values),expected)
        pooled=-sum(sum(row) for row in values)/sum(len(row) for row in values)
        self.assertNotEqual(expected,pooled)
        with self.assertRaises(ValueError):ce.loss_from_logprobs(values[:-1])

    def test_prepared_content_binding_detects_same_length_prompt_target_and_order_drift(self):
        row=rows()[3]; original=ce.prepared_row_binding(row)
        self.assertEqual(original['prepared_row_sha256'],ce.update.digest(row))
        self.assertEqual((original['prompt_tokens'],original['labeled_tokens'],original['full_sequence_tokens']),
                         (len(row['prompt_ids']),len(row['completion_ids']),len(row['prompt_ids'])+len(row['completion_ids'])))
        for change in ('prompt','target','target-order'):
            changed=copy.deepcopy(row)
            if change=='prompt':changed['prompt_ids'][0]+=1
            if change=='target':changed['completion_ids'][0]+=1
            if change=='target-order':changed['completion_ids'][:-1]=reversed(changed['completion_ids'][:-1])
            value=ce.prepared_row_binding(changed)
            with self.subTest(change=change):
                for key in ('id','prompt_tokens','labeled_tokens','full_sequence_tokens'):self.assertEqual(value[key],original[key])
                self.assertNotEqual(value['prepared_row_sha256'],original['prepared_row_sha256'])
                if change=='prompt':
                    self.assertNotEqual(value['prompt_ids_sha256'],original['prompt_ids_sha256'])
                    self.assertEqual(value['completion_ids_sha256'],original['completion_ids_sha256'])
                else:
                    self.assertEqual(value['prompt_ids_sha256'],original['prompt_ids_sha256'])
                    self.assertNotEqual(value['completion_ids_sha256'],original['completion_ids_sha256'])
        for key in ('prompt_ids','completion_ids'):
            for tokens in ([],[True],[-1],[1.0],[float('nan')],[float('inf')],[None],(1,),'1'):
                bad=copy.deepcopy(row);bad[key]=tokens
                with self.subTest(key=key,tokens=tokens),self.assertRaises(ValueError):ce.prepared_row_binding(bad)
        for identity in ('',None,True,7):
            bad=copy.deepcopy(row);bad['id']=identity
            with self.subTest(identity=identity),self.assertRaises(ValueError):ce.prepared_row_binding(bad)

    def test_static_recipe_identity_excludes_exact_live_fields_and_binds_every_static_choice(self):
        value=recipe(); original=ce.static_recipe_sha256(value)
        for key,new in (('boundary_id',7),('generation',4),('symbolic_snapshot','c'*64),
                        ('start_policy_id','d'*64),('U_before',3)):
            changed=copy.deepcopy(value);changed[key]=new;ce.validate_recipe(changed)
            with self.subTest(dynamic=key):self.assertEqual(ce.static_recipe_sha256(changed),original)
        for change in ('lr','betas','clip-literal','inventory','EOS'):
            changed=copy.deepcopy(value)
            if change=='lr':changed['optimizer']['lr']*=2
            if change=='betas':changed['optimizer']['betas'][0]=.8
            if change=='clip-literal':changed['gradient_clipping']['max_norm']=1
            if change=='inventory':changed['trainable_names']=changed['trainable_names'][:-1]
            if change=='EOS':changed['eos_ids'].append(31)
            ce.validate_recipe(changed)
            with self.subTest(static=change):self.assertNotEqual(ce.static_recipe_sha256(changed),original)
        # The helper excludes only the named live fields, without granting new
        # keys or silently omitting future static configuration.
        changed=copy.deepcopy(value);changed['future_static_field']='bound'
        self.assertNotEqual(ce.static_recipe_sha256(changed),original)
        self.assertEqual(value,recipe())
        with self.assertRaises(ValueError):ce.static_recipe_sha256(None)


@unittest.skipUnless(os.environ.get('COMP_RS_RUN_NATIVE_CPU_TESTS')=='1','actual CPU CE graphs await root clearance')
class TinyCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from experiments.maxrl_comp002_pretraining_v1 import s0_runtime,s0_generate,mechanics
        cls.torch,cls.runtime,cls.generate,cls.mechanics=torch,s0_runtime,s0_generate,mechanics

    def model(self,scale=1):
        torch=self.torch
        from functools import partial
        from types import SimpleNamespace
        from torch.utils.checkpoint import checkpoint
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(8,8);self.gradient_checkpointing=False;self.checkpoint_calls=0
            def forward(self,value):
                if self.training and self.gradient_checkpointing:
                    self.checkpoint_calls+=1
                    return torch.tanh(self._gradient_checkpointing_func(self.linear,value))
                return torch.tanh(self.linear(value))
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embed_tokens=torch.nn.Embedding(32,8);self.layers=torch.nn.ModuleList([Layer()])
                self.register_buffer('frozen_offset',torch.tensor(.125))
            def forward(self,input_ids,attention_mask=None,**kwargs):
                embedded=self.embed_tokens(input_ids)
                if attention_mask is not None:embedded=embedded*attention_mask.unsqueeze(-1)
                value=embedded.cumsum(1)+self.frozen_offset
                for layer in self.layers:value=layer(value)
                return SimpleNamespace(last_hidden_state=value)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.model=Decoder();self.lm_head=torch.nn.Linear(8,32,bias=False)
                self.config=SimpleNamespace(vocab_size=32);self.is_gradient_checkpointing=False
            def get_output_embeddings(self):return self.lm_head
            def gradient_checkpointing_enable(self,gradient_checkpointing_kwargs):
                self.is_gradient_checkpointing=True
                for layer in self.model.layers:
                    layer.gradient_checkpointing=True
                    layer._gradient_checkpointing_func=partial(checkpoint,**gradient_checkpointing_kwargs)
        torch.manual_seed(901)
        model=Model().float().eval()
        with torch.no_grad():model.lm_head.weight.mul_(scale)
        for name,param in model.named_parameters():param.requires_grad_(name.startswith('model.layers.'))
        return model

    def inputs(self,model):
        r=recipe();r['start_policy_id']=self.runtime.anchor_digest(model)
        r['trainable_names']=sorted(n for n,p in model.named_parameters() if p.requires_grad)
        return rows(),r

    def optimizer(self,model,r):
        kwargs=dict(r['optimizer']);del kwargs['class'];kwargs['betas']=tuple(kwargs['betas'])
        return self.torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],**kwargs)

    def reference(self,model,data,pooled=False,padding=None):
        torch=self.torch
        means,values=[],[]
        for row in data:
            P,C=len(row['prompt_ids']),len(row['completion_ids'])
            real=row['prompt_ids']+row['completion_ids']
            left,right=(3,0) if padding=='left' else (0,3) if padding=='right' else (0,0)
            ids=torch.tensor([[0]*left+real+[0]*right])
            attention=torch.tensor([[0]*left+[1]*len(real)+[0]*right])
            hidden=model.model(input_ids=ids,attention_mask=attention).last_hidden_state
            full=model.lm_head(hidden).float().log_softmax(-1)[0]
            selected=torch.stack([full[left+P-1+j,t] for j,t in enumerate(row['completion_ids'])])
            # Independent indexing above supplies the gradient reference. The
            # shared label helper is checked separately against literal masks.
            labels=self.mechanics.completion_labels(ids,attention.bool(),[P],[C],[2],['eos'],8)
            expected=self.torch.full_like(ids,-100);expected[0,left+P:left+P+C]=ids[0,left+P:left+P+C]
            self.assertTrue(torch.equal(labels,expected))
            self.assertEqual(int((labels==2).sum()),1)
            means.append(-selected.mean());values.append(selected)
        loss=-torch.cat(values).mean() if pooled else torch.stack(means).mean()
        loss.backward()
        return float(loss.detach()),{n:p.grad.detach().clone() for n,p in model.named_parameters() if p.requires_grad}

    def test_equal_row_gradient_independent_reference_EOS_and_padding(self):
        torch=self.torch;model=self.model();data,r=self.inputs(model)
        ref=copy.deepcopy(model);loss,raw=self.reference(ref,data)
        wrong=copy.deepcopy(model);_,pooled=self.reference(wrong,data,pooled=True)
        self.assertTrue(any(not torch.allclose(raw[n],pooled[n],atol=1e-6,rtol=1e-4) for n in raw))
        for side in ('left','right'):
            padded=copy.deepcopy(model);_,gradient=self.reference(padded,data,padding=side)
            for n in raw:torch.testing.assert_close(gradient[n],raw[n],atol=1e-6,rtol=1e-5)
        optimizer=self.optimizer(model,r);events=[];captured={}
        def observer(kind,phase,**meta):
            events.append((kind,phase,meta))
            if kind=='gradient_validation' and phase=='returned':
                captured.update({n:p.grad.detach().clone() for n,p in model.named_parameters() if p.requires_grad})
        from torch.utils import checkpoint as checkpoint_module
        head_calls=[];original_checkpoint=checkpoint_module.checkpoint
        def head_spy(function,*args,**kwargs):
            if getattr(function,'__name__',None)=='_head_logprobs':
                head_calls.append(kwargs.get('use_reentrant'))
            return original_checkpoint(function,*args,**kwargs)
        with patch.object(checkpoint_module,'checkpoint',head_spy):
            result=ce.boundary_update(model,optimizer,data,r,observer)
        self.assertAlmostEqual(result['loss'],loss,places=5)
        for n in raw:torch.testing.assert_close(captured[n],raw[n],atol=1e-6,rtol=1e-5)
        order=[(k,p) for k,p,_ in events if k in ('loss_forward','backward')]
        self.assertEqual(order,[(k,p) for _ in range(8) for k,p in
            (('loss_forward','started'),('loss_forward','returned'),('backward','started'),('backward','returned'))])
        self.assertGreater(model.model.layers[0].checkpoint_calls,0)
        self.assertEqual(head_calls,[False]*8)
        self.assertEqual((result['U_before'],result['U_after']),(0,1))
        self.assertTrue(all(p.grad is None for p in model.parameters()));self.assertFalse(model.training)

    def test_clip_after_eight_backwards_before_Adam_matches_reference_and_frozen(self):
        torch=self.torch;model=self.model(scale=1000);data,r=self.inputs(model)
        ref=copy.deepcopy(model);_,raw=self.reference(ref,data);refopt=self.optimizer(ref,r)
        refnorm=torch.nn.utils.clip_grad_norm_([p for p in ref.parameters() if p.requires_grad],**r['gradient_clipping'])
        self.assertGreater(float(refnorm),1);expected={n:p.grad.clone() for n,p in ref.named_parameters() if p.requires_grad}
        refopt.step();refopt.zero_grad(set_to_none=True)
        optimizer=self.optimizer(model,r);events=[];actual={};step=optimizer.step
        def observed_step(*args,**kwargs):
            actual.update({n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad})
            return step(*args,**kwargs)
        frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        frozen_buffers={n:p.detach().clone() for n,p in model.named_buffers()}
        with patch.object(optimizer,'step',observed_step):
            result=ce.boundary_update(model,optimizer,data,r,lambda k,p,**m:events.append((k,p,m)))
        self.assertGreater(result['raw_gradient_norm_float64'],1)
        self.assertLessEqual(result['clipped_gradient_norm_float64'],1+32*torch.finfo(torch.float32).eps)
        self.assertLess(events.index(next(e for e in events if e[0]=='gradient_clipping' and e[1]=='returned')),
                        events.index(next(e for e in events if e[0]=='optimizer' and e[1]=='started')))
        for n in raw:torch.testing.assert_close(actual[n],expected[n],atol=1e-6,rtol=1e-5)
        ref_named=dict(ref.named_parameters())
        for n,p in model.named_parameters():
            torch.testing.assert_close(p,ref_named[n],atol=1e-7,rtol=1e-5)
            if n in frozen:self.assertTrue(torch.equal(p,frozen[n]))
            if p.requires_grad:
                for key in ('step','exp_avg','exp_avg_sq'):
                    torch.testing.assert_close(optimizer.state[p][key],refopt.state[ref_named[n]][key],atol=1e-7,rtol=1e-5)
        self.assertTrue(all(torch.equal(value,frozen_buffers[n]) for n,value in model.named_buffers()))

    def test_nonfinite_raw_gradient_refused_before_clip_or_Adam_and_cleared(self):
        torch=self.torch;model=self.model();data,r=self.inputs(model);opt=self.optimizer(model,r);events=[]
        before=self.runtime.anchor_digest(model)
        def observer(kind,phase,**meta):
            events.append((kind,phase,meta))
            if kind=='gradient_validation' and phase=='started':
                next(p for p in model.parameters() if p.requires_grad).grad.reshape(-1)[0]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite trainable gradient'):
            ce.boundary_update(model,opt,data,r,observer)
        self.assertFalse(any(k in ('gradient_clipping','optimizer') for k,p,_ in events))
        self.assertFalse(opt.state);self.assertEqual(self.runtime.anchor_digest(model),before)
        self.assertTrue(all(p.grad is None for p in model.parameters()));self.assertFalse(model.training)

    def test_late_forward_failure_retains_backwards_clears_noAdam(self):
        model=self.model();data,r=self.inputs(model);opt=self.optimizer(model,r);events=[]
        before=self.runtime.anchor_digest(model);original=self.generate.teacher_forced_logprobs;calls=0
        def fail_late(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==8:raise ValueError('eighth row failure')
            return original(*args,**kwargs)
        with patch.object(self.generate,'teacher_forced_logprobs',fail_late),self.assertRaisesRegex(ValueError,'eighth') as failure:
            ce.boundary_update(model,opt,data,r,lambda k,p,**m:events.append((k,p,m)))
        self.assertFalse(failure.exception.requires_restore)
        self.assertEqual(sum(k=='backward' and p=='returned' for k,p,_ in events),7)
        self.assertFalse(any(k=='optimizer' for k,p,_ in events))
        self.assertFalse(opt.state);self.assertEqual(self.runtime.anchor_digest(model),before)
        self.assertTrue(all(p.grad is None for p in model.parameters()));self.assertFalse(model.training)

    def test_inherited_Adam_wrong_U_and_actual_vocabulary_rejected(self):
        model=self.model();data,r=self.inputs(model);opt=self.optimizer(model,r)
        ce.boundary_update(model,opt,data,r,lambda *a,**k:None)
        r['start_policy_id']=self.runtime.anchor_digest(model)
        with self.assertRaisesRegex(ValueError,'Adam applied U'):
            ce.boundary_update(model,opt,data,r,lambda *a,**k:None)
        r['U_before']=1;data[7]['completion_ids'][0]=32
        with self.assertRaisesRegex(ValueError,'vocabulary'):
            ce.boundary_update(model,opt,data,r,lambda *a,**k:None)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_returned_Adam_then_failed_identity_requires_restore_no_success_record(self):
        model=self.model();data,r=self.inputs(model);opt=self.optimizer(model,r);events=[]
        original=self.runtime.state_digests;calls=0
        def fail_after_step(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise ValueError('post-step identity failure')
            return original(*args,**kwargs)
        with patch.object(self.runtime,'state_digests',fail_after_step),self.assertRaisesRegex(ValueError,'post-step') as failure:
            ce.boundary_update(model,opt,data,r,lambda k,p,**m:events.append((k,p,m)))
        self.assertTrue(failure.exception.requires_restore)
        self.assertTrue(any(k=='optimizer' and p=='returned' for k,p,_ in events))
        self.assertTrue(any(k=='fingerprint' and p=='failed' for k,p,_ in events))
        self.assertEqual({int(s['step']) for s in opt.state.values()},{1})
        self.assertTrue(all(p.grad is None for p in model.parameters()));self.assertFalse(model.training)

    def test_partial_optimizer_failure_retains_attempt_and_marks_live_state_unusable(self):
        torch=self.torch;model=self.model();data,r=self.inputs(model);opt=self.optimizer(model,r);events=[]
        before=self.runtime.anchor_digest(model)
        def fail_partial(*args,**kwargs):
            with torch.no_grad():next(p for p in model.parameters() if p.requires_grad).add_(.01)
            raise RuntimeError('partial Adam fixture')
        with patch.object(opt,'step',fail_partial),self.assertRaisesRegex(RuntimeError,'partial Adam') as failure:
            ce.boundary_update(model,opt,data,r,lambda k,p,**m:events.append((k,p,m)))
        self.assertTrue(failure.exception.requires_restore)
        self.assertTrue(any(k=='optimizer' and p=='started' for k,p,_ in events))
        self.assertTrue(any(k=='optimizer' and p=='failed' for k,p,_ in events))
        self.assertFalse(any(k=='optimizer' and p=='returned' for k,p,_ in events))
        self.assertNotEqual(self.runtime.anchor_digest(model),before)
        self.assertTrue(all(p.grad is None for p in model.parameters()));self.assertFalse(model.training)

    def test_final_clear_failure_after_real_Adam_retains_restore_obligation_and_eval(self):
        model=self.model();data,r=self.inputs(model);opt=self.optimizer(model,r);events=[]
        original=opt.zero_grad;calls=0
        def fail_final(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('final gradient clear failure')
            return original(*args,**kwargs)
        with patch.object(opt,'zero_grad',fail_final),self.assertRaisesRegex(RuntimeError,'final gradient clear') as failure:
            ce.boundary_update(model,opt,data,r,lambda k,p,**m:events.append((k,p,m)))
        self.assertTrue(failure.exception.requires_restore)
        self.assertFalse(model.training)
        self.assertEqual({int(state['step']) for state in opt.state.values()},{1})
        self.assertTrue(any(k=='optimizer' and p=='returned' for k,p,_ in events))
        self.assertTrue(any(k=='gradient_reset' and p=='returned' for k,p,_ in events))
        self.assertTrue(any(k=='gradient_cleanup' and p=='failed' for k,p,_ in events))
        original(set_to_none=True) # disposable fixture cleanup after proven failure

    def test_primary_forward_failure_not_masked_by_failed_cleanup(self):
        model=self.model();data,r=self.inputs(model);opt=self.optimizer(model,r);events=[]
        original=opt.zero_grad;calls=0
        def fail_final(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('secondary cleanup failure')
            return original(*args,**kwargs)
        with patch.object(opt,'zero_grad',fail_final),patch.object(self.generate,'teacher_forced_logprobs',
             side_effect=ValueError('primary forward failure')),self.assertRaisesRegex(ValueError,'primary forward') as failure:
            ce.boundary_update(model,opt,data,r,lambda k,p,**m:events.append((k,p,m)))
        self.assertFalse(failure.exception.requires_restore)
        self.assertEqual(failure.exception.cleanup_failure['type'],'RuntimeError')
        self.assertFalse(model.training)
        self.assertFalse(any(k=='optimizer' for k,p,_ in events))
        self.assertTrue(any(k=='gradient_cleanup' and p=='failed' for k,p,_ in events))
        original(set_to_none=True)

    def test_earlier_second_moment_chunk_corruption_rejected_before_work_nonnegative_control(self):
        torch=self.torch;model=self.model()
        # 513*512 exceeds the 262144-FP32-element chunk. A nonnegative tail
        # cannot hide a negative entry at the first element of the first chunk.
        model.model.embed_tokens=torch.nn.Embedding(32,512)
        model.model.layers[0].linear=torch.nn.Linear(512,513)
        model.lm_head=torch.nn.Linear(513,32,bias=False)
        for name,parameter in model.named_parameters():parameter.requires_grad_(name.startswith('model.layers.'))
        data,r=self.inputs(model);opt=self.optimizer(model,r)
        for parameter in model.parameters():
            if parameter.requires_grad:parameter.grad=torch.full_like(parameter,.001)
        opt.step();opt.zero_grad(set_to_none=True)
        r['U_before']=1;r['start_policy_id']=self.runtime.anchor_digest(model)
        weight=model.model.layers[0].linear.weight
        moment=opt.state[weight]['exp_avg_sq']
        chunk=ce.update.CHUNK_BYTES//4
        self.assertGreater(moment.numel(),chunk)
        self.assertTrue(bool((moment.reshape(-1)[chunk:]>=0).all()))
        saved_first=moment.reshape(-1)[0].clone()
        moment.reshape(-1)[0]=-.25
        before_parameters={n:p.detach().clone() for n,p in model.named_parameters()}
        before_state={p:{key:value.detach().clone() for key,value in state.items()} for p,state in opt.state.items()}
        events=[]
        with self.assertRaisesRegex(ValueError,'negative Adam second moment'):
            ce.boundary_update(model,opt,data,r,lambda k,p,**m:events.append((k,p,m)))
        self.assertFalse(any(k in ('loss_forward','backward','gradient_clipping','optimizer') for k,p,_ in events))
        self.assertTrue(all(torch.equal(parameter,before_parameters[n]) for n,parameter in model.named_parameters()))
        for parameter,state in opt.state.items():
            for key,value in state.items():self.assertTrue(torch.equal(value,before_state[parameter][key]))
        self.assertEqual({int(state['step']) for state in opt.state.values()},{1})
        self.assertTrue(all(p.grad is None for p in model.parameters()));self.assertFalse(model.training)
        moment.reshape(-1)[0].copy_(saved_first)
        control=ce.boundary_update(model,opt,data,r,lambda *args,**kwargs:None)
        self.assertTrue(control['updated']);self.assertEqual((control['U_before'],control['U_after']),(1,2))


if __name__=='__main__':unittest.main()
