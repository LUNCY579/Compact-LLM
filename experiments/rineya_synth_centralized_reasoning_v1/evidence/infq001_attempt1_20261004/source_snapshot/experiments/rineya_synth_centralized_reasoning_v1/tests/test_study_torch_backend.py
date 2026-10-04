"""Random two-layer CPU Torch controls isolated from the stdlib test parent.

No published checkpoint, Transformers model, GPU, native worker or training
corpus is opened. These checks qualify neither native execution nor capability.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

PACKAGE = Path(__file__).resolve().parents[1]

CHILD = r'''
from copy import deepcopy
import hashlib, io, json, random, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path(sys.argv[1])))
import study_torch_backend as backend
import native_mechanics as mechanics
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()

class RMSNorm(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width))
    def forward(self, x):
        work = x.float()
        return self.weight * work * torch.rsqrt(work.square().mean(-1, keepdim=True) + 1e-6)

class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4,8,bias=False)
        self.k_proj = torch.nn.Linear(4,4,bias=False)
        self.v_proj = torch.nn.Linear(4,4,bias=False)
        self.o_proj = torch.nn.Linear(8,4,bias=False)
        self.q_norm, self.k_norm = RMSNorm(8), RMSNorm(4)
    def forward(self,x):
        q = self.q_norm(self.q_proj(x))
        kv = self.k_norm(self.k_proj(x)) + self.v_proj(x)
        return self.o_proj(q + kv.repeat_interleave(2,dim=-1))

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj, self.up_proj = torch.nn.Linear(4,8,bias=False), torch.nn.Linear(4,8,bias=False)
        self.down_proj = torch.nn.Linear(8,4,bias=False)
    def forward(self,x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn, self.mlp = Attention(), MLP()
        self.input_layernorm, self.post_attention_layernorm = RMSNorm(4), RMSNorm(4)
    def forward(self,x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))

class Body(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens, self.norm = torch.nn.Embedding(11,4), RMSNorm(4)
        self.layers = torch.nn.ModuleList([Layer(),Layer()])
    def forward(self,ids):
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.lm_head = Body(), torch.nn.Linear(4,11,bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.to(dtype=torch.bfloat16)
        self.requires_grad_(False)
        self.eval()
        self.forward_calls = 0
        self.draw_rng = False
    def forward(self,input_ids,attention_mask,use_cache,logits_to_keep):
        assert use_cache is False and logits_to_keep == 0
        assert torch.equal(attention_mask,torch.ones_like(input_ids))
        self.forward_calls += 1
        if self.draw_rng:
            random.random()
            torch.rand(1)
        return SimpleNamespace(logits=self.lm_head(self.model(input_ids)))

PROFILE = {'expected_linear_sites':15,'expected_layers':2,'expected_base_parameters':472,
    'expected_adapter_shapes':{f'model.layers.{i}.self_attn.{p}.lora_{f}':s for i in range(2)
        for p,f,s in [('q_proj','A',[8,4]),('q_proj','B',[8,8]),('v_proj','A',[8,4]),('v_proj','B',[4,8])]},
    'eos_id':10,'pad_id':9,'vocab_size':11}

class Journal:
    def __init__(self):
        self.events = []
        self.fail_before = self.fail_after = None
    def __call__(self,op,function):
        self.events.append(('attempt',op))
        if op == self.fail_before:
            raise OSError('injected attempted sink failure')
        try:
            value = function()
        except BaseException:
            self.events.append(('failed',op))
            raise
        self.events.append(('returned',op))
        if op == self.fail_after:
            raise OSError('injected returned sink failure')
        return value

def expect_error(function, exception=ValueError):
    try:
        function()
    except exception:
        return
    raise AssertionError('expected fail-closed error')

def make(arm='M1', journal=None, **kwargs):
    torch.manual_seed(333)
    random.seed(333)
    model = Toy()
    journal = journal or Journal()
    opts = dict(arm=arm,seed=101,recipe_sha256='a'*64,cohort_sha256='b'*64,
                execution_scope='CPU_RANDOM_TOY',toy_profile=deepcopy(PROFILE))
    opts.update(kwargs)
    result = backend.StudyTorchBackend(torch,model,journal,lambda stage:None,**opts)
    return result,journal

def row(update=1,slot=0):
    ids = [slot%8,(slot+1)%8,(slot+2)%8,3+update%3,4,10]
    return {'row_id':hashlib.sha256(f'{update}/{slot}'.encode()).hexdigest(),
            'input_ids':ids,'labels':[-100]*3+ids[3:],'attention_mask':[1]*6,
            'prompt_ids':ids[:3],'completion_ids':ids[3:],'prompt_tokens':3,
            'supervised_tokens':3,'total_tokens':6}

def train(b,u):
    b.reset_gradients(u)
    losses = []
    for slot in range(20):
        r = row(u,slot)
        out = b.forward(r)
        loss = b.completion_loss(out,r)
        assert b.last_loss_check['supervised_positions'] == 3
        assert b.last_loss_check['absolute_error'] <= 2e-6
        losses.append(float(loss.detach()))
        b.backward(loss,scale=1/20)
    norm = b.finite_clip_gradients(max_norm=1.0)
    assert norm >= 0
    b.optimizer_step(lr=backend.learning_rate(u),update_ordinal=u)
    assert all(p.grad is None for p in b.parameters)
    return losses

def binding(b):
    return {'completed_updates':b.completed_updates,'consumed_occurrences':len(b.order),
            'input_tokens':b.input_tokens,'supervised_tokens':b.supervised_tokens,
            'consumed_order_sha256':backend.digest(b.order),'next_update_ordinal':b.completed_updates+1,
            'learning_rate':backend.learning_rate(b.completed_updates)}

def witness(b):
    return mechanics.snapshot_training_state(torch,b.model,b.optimizer,require_cuda=False)

def corrupt_archive(descriptor,path,change):
    raw = Path(descriptor['path']).read_bytes()
    body = torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu')
    change(body)
    stream = io.BytesIO()
    torch.save(body,stream)
    raw = stream.getvalue()
    path.write_bytes(raw)
    out = deepcopy(descriptor)
    out.update(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return out

mode = sys.argv[2]
with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    if mode == 'continuation':
        full,journal = make()
        full.model.draw_rng = True
        first_losses = train(full,1)
        pre_save = witness(full)  # independent detached witness created before save
        cp = full.save_checkpoint(directory/'study1.pt',binding(full))
        assert mechanics._state_matches(torch,full.model,full.optimizer,pre_save,False)
        snapshot = full.snapshot_boundary(binding(full))
        assert type(snapshot.data) is bytes
        detached = snapshot.descriptor
        detached['metadata']['arm'] = 'M2'
        assert snapshot.descriptor['metadata']['arm'] == 'M1'
        with torch.no_grad():
            full.parameters[0].add_(.25)
            full.optimizer.state[full.parameters[0]]['exp_avg'].add_(.5)
        random.random();torch.rand(1)
        full.restore_checkpoint(cp,binding(full))
        assert mechanics._state_matches(torch,full.model,full.optimizer,pre_save,False)
        second_losses = train(full,2)
        completed = witness(full)
        resumed,_ = make()
        resumed.model.draw_rng = True
        resumed.restore_checkpoint(cp,cp['metadata']['binding'])
        assert mechanics._state_matches(torch,resumed.model,resumed.optimizer,pre_save,False)
        assert train(resumed,2) == second_losses
        assert mechanics._state_matches(torch,resumed.model,resumed.optimizer,completed,False)
        assert resumed.order == full.order and binding(resumed) == binding(full)
        cp2 = resumed.save_checkpoint(directory/'study2.pt',binding(resumed))
        assert cp2['metadata']['binding']['consumed_occurrences'] == 40
        counts = full.primitive_operation_snapshot()
        assert counts['training_forward']['returned'] == counts['training_backward']['returned'] == 40
        assert counts['completion_CE_selected_NLL_check']['returned'] == 40
        assert counts['optimizer_step']['returned'] == counts['consumed_gradient_clear']['returned'] == 2
        assert counts['gradient_reset']['returned'] == 2
        assert all(not cp['metadata'][key] for key in ('training_authorized','native_launch_authorized','native_resume_qualified','qualification_STATE1_used'))
        full.close();resumed.close()
    elif mode == 'loss':
        b,_ = make()
        b.reset_gradients(1)
        r = row()
        out = b.forward(r)
        loss = b.completion_loss(out,r)
        manual = torch.stack([-out['logits'][0,i-1].detach().log_softmax(-1)[r['input_ids'][i]] for i in (3,4,5)]).mean()
        assert abs(float(loss.detach())-float(manual)) <= 2e-6
        altered = out['logits'].detach().clone()
        altered[0,:2].add_(torch.arange(11)*100)
        assert torch.equal(mechanics.completion_loss(torch,altered,out['input_ids'],out['labels']).detach(),loss.detach())
        changed = deepcopy(r);changed['labels'][-1]=4
        expect_error(lambda:b.completion_loss(out,changed))
        expect_error(lambda:b.backward(loss,scale=1.0))
        b.backward(loss,scale=1/20)
        expect_error(lambda:b.finite_clip_gradients(max_norm=1.0))
        expect_error(lambda:b.snapshot_boundary({}))
        b.close()
    elif mode == 'constructor':
        expect_error(lambda:make(seed=102))
        expect_error(lambda:make(execution_scope='CPU_COMPONENT_CONTROL'))
        expect_error(lambda:make(execution_scope='UNQUALIFIED_NATIVE_IMPLEMENTATION',toy_profile=None,source_hashes={'x':'a'*64}))
        bad=deepcopy(PROFILE);bad['expected_base_parameters']+=1
        expect_error(lambda:make(toy_profile=bad))
        bad=deepcopy(PROFILE);bad['expected_adapter_shapes'][next(iter(bad['expected_adapter_shapes']))]=[4,4]
        expect_error(lambda:make(toy_profile=bad))
        left,_=make('M1');right,_=make('M2')
        assert left.fresh_start_attestation == right.fresh_start_attestation
        assert not left.fresh_start_attestation['native_output_parity_qualified']
        expect_error(lambda:backend.StudyTorchBackend(torch,left.model,Journal(),lambda s:None,arm='M1',seed=101,
            recipe_sha256='a'*64,cohort_sha256='b'*64,execution_scope='CPU_RANDOM_TOY',toy_profile=PROFILE))
        left.close();right.close()
        torch.manual_seed(333);model=Toy();next(model.parameters()).requires_grad_(True)
        expect_error(lambda:backend.StudyTorchBackend(torch,model,Journal(),lambda s:None,arm='M1',seed=101,
            recipe_sha256='a'*64,cohort_sha256='b'*64,execution_scope='CPU_RANDOM_TOY',toy_profile=PROFILE))
    elif mode == 'bad_rows':
        b,journal=make();b.reset_gradients(1)
        for kind in ('mask','EOS','padding','bool','vocabulary','length','overlength','attention'):
            bad=deepcopy(row())
            if kind=='mask':bad['labels'][0]=bad['input_ids'][0]
            elif kind=='EOS':bad['input_ids'][-1]=4
            elif kind=='padding':bad['input_ids'][0]=9
            elif kind=='bool':bad['input_ids'][0]=True
            elif kind=='vocabulary':bad['input_ids'][0]=11
            elif kind=='attention':bad['attention_mask'][0]=0
            elif kind=='overlength':
                bad.update(input_ids=[0]*512+[10],labels=[-100]*512+[10],attention_mask=[1]*513,
                    prompt_ids=[0]*512,completion_ids=[10],prompt_tokens=512,supervised_tokens=1,total_tokens=513)
            else:bad['attention_mask'].pop()
            expect_error(lambda:b.forward(bad))
        assert b.model.forward_calls == 0
        assert 'training_forward' not in b.primitive_operation_snapshot()
        b.close()
    elif mode == 'base':
        for fault in ('bytes','requires_grad','gradient','shape'):
            b,_=make();b.reset_gradients(1)
            p=b.model.model.embed_tokens.weight
            if fault=='bytes':
                with torch.no_grad():p[0,0].add_(1)
            elif fault=='requires_grad':p.requires_grad_(True)
            elif fault=='gradient':p.grad=torch.zeros_like(p)
            else:p.data=p.data[:10]
            expect_error(lambda:b.forward(row()))
            assert b.model.forward_calls==0
            b.close()
    elif mode == 'descriptor':
        b,_=make();train(b,1)
        cp=b.save_checkpoint(directory/'study.pt',binding(b))
        expect_error(lambda:b.save_checkpoint(directory/'study.pt',binding(b)),FileExistsError)
        fresh,_=make()
        for field in ('arm','recipe_sha256','cohort_sha256','seed','backend_source_sha256','training_authorized','adapter_parameter_order'):
            bad=deepcopy(cp)
            bad['metadata'][field]=('M2' if field=='arm' else 102 if field=='seed' else 0 if field=='training_authorized'
                else list(reversed(bad['metadata'][field])) if field=='adapter_parameter_order' else 'c'*64)
            bad['metadata_sha256']=backend.digest(bad['metadata'])
            expect_error(lambda:fresh.restore_checkpoint(bad,cp['metadata']['binding']))
        other,_=make('M2')
        expect_error(lambda:other.restore_checkpoint(cp,cp['metadata']['binding']))
        expect_error(lambda:fresh.restore_checkpoint({'kind':'STATE1.pt'},cp['metadata']['binding']))
        expect_error(lambda:fresh.save_checkpoint(directory/'STATE1.pt',{}))
        bad=deepcopy(cp);bad['bytes']=backend.MAX_CHECKPOINT_BYTES+1
        expect_error(lambda:fresh.restore_checkpoint(bad,cp['metadata']['binding']))
        original=Path(cp['path']).read_bytes()
        Path(cp['path']).write_bytes(original[:-1]+bytes([original[-1]^1]))
        expect_error(lambda:fresh.restore_checkpoint(cp,cp['metadata']['binding']))
        assert 'checkpoint_weights_only_load' not in fresh.primitive_operation_snapshot()
        b.close();fresh.close();other.close()
    elif mode == 'state':
        source,_=make();train(source,1)
        cp=source.save_checkpoint(directory/'source.pt',binding(source))
        name=next(iter(PROFILE['expected_adapter_shapes']))
        faults={
            'shape':lambda body:body['state']['adapters'].__setitem__(name,torch.zeros(1)),
            'moment':lambda body:body['state']['optimizer']['state'][0].__setitem__('exp_avg',torch.tensor(float('nan'))),
            'missing_moment':lambda body:body['state']['optimizer']['state'].pop(0),
            'step':lambda body:body['state']['optimizer']['state'][0].__setitem__('step',torch.tensor(2.)),
            'gradient':lambda body:body['state']['gradients'].__setitem__(name,torch.zeros_like(body['state']['adapters'][name])),
            'python_rng':lambda body:body['state']['rng'].__setitem__('python',()),
            'cpu_rng':lambda body:body['state']['rng'].__setitem__('cpu',torch.zeros(1,dtype=torch.uint8)),
            'order':lambda body:body['occurrence_order'].__setitem__(0,'changed'),
            'lr':lambda body:body['state']['optimizer']['param_groups'][0].__setitem__('lr',.0001)}
        for kind,change in faults.items():
            b,_=make()
            before={n:p.detach().clone() for n,p in b.entries}
            bad=corrupt_archive(cp,directory/(kind+'.pt'),change)
            expect_error(lambda:b.restore_checkpoint(bad,cp['metadata']['binding']))
            assert all(mechanics.bitwise_equal_tensor(torch,before[n],p) for n,p in b.entries)
            assert 'checkpoint_adapter_restore' not in b.primitive_operation_snapshot()
            b.close()
        source.close()
    elif mode == 'step_contract':
        b,_=make();b.reset_gradients(1)
        expect_error(lambda:b.optimizer_step(lr=backend.learning_rate(1),update_ordinal=1))
        for slot in range(20):
            r=row(1,slot);loss=b.completion_loss(b.forward(r),r);b.backward(loss,scale=1/20)
        expect_error(lambda:b.forward(row()))
        expect_error(lambda:b.finite_clip_gradients(max_norm=2))
        b.finite_clip_gradients(max_norm=1)
        expect_error(lambda:b.optimizer_step(lr=.0001,update_ordinal=1))
        expect_error(lambda:b.optimizer_step(lr=backend.learning_rate(2),update_ordinal=2))
        b.optimizer_step(lr=backend.learning_rate(1),update_ordinal=1)
        assert len(b.order)==20
        assert b.primitive_operation_snapshot()['optimizer_step']['returned']==1
        bad=binding(b);bad['input_tokens']+=1
        expect_error(lambda:b.snapshot_boundary(bad))
        b.close()
    elif mode == 'accounting':
        b,journal=make();journal.fail_before='training_forward';b.reset_gradients(1)
        expect_error(lambda:b.forward(row()),OSError)
        assert b.model.forward_calls==0
        assert b.primitive_operation_snapshot()['training_forward']=={'attempted':1,'returned':0,'failed':0,'pending':1}
        expect_error(lambda:b.forward(row()))
        b.close()
        b,journal=make();b.reset_gradients(1)
        for slot in range(20):
            r=row(1,slot);loss=b.completion_loss(b.forward(r),r);b.backward(loss,scale=1/20)
        b.finite_clip_gradients(max_norm=1)
        journal.fail_after='optimizer_step'
        expect_error(lambda:b.optimizer_step(lr=backend.learning_rate(1),update_ordinal=1),OSError)
        assert b.primitive_operation_snapshot()['optimizer_step']=={'attempted':1,'returned':1,'failed':0,'pending':0}
        assert 'consumed_gradient_clear' not in b.primitive_operation_snapshot()
        expect_error(lambda:b.snapshot_boundary({}))
        b.close()
    else:
        raise AssertionError(mode)
assert not torch.cuda.is_initialized()
print(json.dumps({'status':'PASS_CPU_RANDOM_TOY_COMPONENT_ONLY','mode':mode,'native_launch_authorized':False}))
'''


class StudyTorchBackendTests(unittest.TestCase):
    def child(self, mode):
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        completed = subprocess.run([sys.executable, "-c", CHILD, str(PACKAGE), mode],
                                   capture_output=True, text=True, env=environment, timeout=90)
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual("PASS_CPU_RANDOM_TOY_COMPONENT_ONLY", result["status"])
        self.assertFalse(result["native_launch_authorized"])

    def test_framework_free_module_import(self):
        source = "import sys;sys.path.insert(0,sys.argv[1]);import study_torch_backend;assert 'torch' not in sys.modules;assert 'transformers' not in sys.modules"
        completed = subprocess.run([sys.executable, "-c", source, str(PACKAGE)], capture_output=True, text=True, timeout=20)
        self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)

    def test_two_updates_save_restore_independent_witness_and_continuation(self):
        self.child("continuation")

    def test_completion_plus_EOS_masked_CE_independent_selected_NLL(self):
        self.child("loss")

    def test_fixed_initialization_inventory_scope_and_no_STATE1_warmstart(self):
        self.child("constructor")

    def test_invalid_data_rejected_before_forward(self):
        self.child("bad_rows")

    def test_nongrad_base_content_object_shape_and_gradient_changes_rejected(self):
        self.child("base")

    def test_writeonce_samebytes_hash_crossarm_recipe_cursor_descriptor_rejection(self):
        self.child("descriptor")

    def test_full_state_shape_moment_step_grad_RNG_order_corruptions_rejected_before_restore(self):
        self.child("state")

    def test_exact20_global_LR_clip_and_empty_accumulation_boundaries(self):
        self.child("step_contract")

    def test_actual_returned_and_pending_primitive_costs_survive_sink_failures(self):
        self.child("accounting")


if __name__ == "__main__":
    unittest.main()
