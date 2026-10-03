"""Tiny CPU-only child checks; no Torch import in the stdlib test parent."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

PACKAGE = Path(__file__).resolve().parents[1]

CHILD = r'''
import hashlib, io, json, os, random, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path(sys.argv[1])))
import native_precision as precision
import native_mechanics as mechanics
torch.set_num_threads(1)
torch.manual_seed(411)
random.seed(411)
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
        self.q_proj = torch.nn.Linear(4, 8, bias=False)
        self.k_proj = torch.nn.Linear(4, 4, bias=False)
        self.v_proj = torch.nn.Linear(4, 4, bias=False)
        self.o_proj = torch.nn.Linear(8, 4, bias=False)
        self.q_norm = RMSNorm(8)
        self.k_norm = RMSNorm(4)
    def forward(self, x):
        q = self.q_norm(self.q_proj(x))
        kv = self.k_norm(self.k_proj(x)) + self.v_proj(x)
        return self.o_proj(q + kv.repeat_interleave(2, dim=-1))

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 8, bias=False)
        self.up_proj = torch.nn.Linear(4, 8, bias=False)
        self.down_proj = torch.nn.Linear(8, 4, bias=False)
    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.mlp = MLP()
        self.input_layernorm = RMSNorm(4)
        self.post_attention_layernorm = RMSNorm(4)
    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))

class Body(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(11, 4)
        self.layers = torch.nn.ModuleList([Layer(), Layer()])
        self.norm = RMSNorm(4)
    def forward(self, ids):
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Body()
        self.lm_head = torch.nn.Linear(4, 11, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.to(dtype=torch.bfloat16)
        self.requires_grad_(False)
        self.eval()
        self.draw_rng = False
    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
        assert use_cache is False and logits_to_keep == 0
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        if self.draw_rng:
            torch.rand(1)
        return SimpleNamespace(logits=self.lm_head(self.model(input_ids)))

def make_toy():
    toy = Toy()
    precision.install_fp32_compute(toy, torch, expected_linear_sites=15)
    precision.install_qv_lora(toy, torch, expected_layers=2)
    return toy

def rows():
    output = []
    for index in range(40):
        prompt = [index % 7, 2, 3] + [4] * (index % 3)
        completion = [5 + index % 3] * (1 + index % 3) + [9]
        ids = prompt + completion
        output.append({'row_id':str(index), 'input_ids':ids,
                       'labels':[-100] * len(prompt) + completion,
                       'attention_mask':[1] * len(ids), 'prompt_ids':prompt,
                       'completion_ids':completion, 'prompt_tokens':len(prompt),
                       'supervised_tokens':len(completion), 'total_tokens':len(ids)})
    return output

CONFIG = {'require_cuda':False, 'adapter_tensors':8, 'adapter_parameters':320,
          'eos_id':9, 'rows_per_update':20}
mode = sys.argv[2]
if mode == 'ce_fault':
    original_loss = mechanics.completion_loss
    mechanics.completion_loss = lambda *args: original_loss(*args) + .001
operations, stages = [], []
preserved = {}
toy = make_toy()
base = {name:p for name,p in toy.named_parameters(remove_duplicate=False)
        if not name.endswith(('.lora_A','.lora_B'))}
base_values = {name:p.detach().clone() for name,p in base.items()}
base_pointers = {name:p.data_ptr() for name,p in base.items()}
native_backward_count = 0

def call(operation, function):
    global native_backward_count
    operations.append(operation)
    if mode == 'rng_fault' and operation == 'reference_forward':
        toy.draw_rng = True
    if mode == 'tamper_checkpoint' and operation == 'checkpoint_load':
        checkpoint = work / 'STATE1.pt'
        data = bytearray(checkpoint.read_bytes())
        data[40] ^= 1
        checkpoint.write_bytes(data)
    result = function()
    if mode == 'restore_snapshot_fault' and operation == 'checkpoint_load':
        first_name = next(iter(result['adapters']))
        result['adapters'][first_name].reshape(-1)[0].add_(.25)
    if mode == 'reference_optimizer_fault' and operation == 'reference_optimizer_step':
        first_state = next(iter(function.__self__.state.values()))
        first_state['exp_avg'].reshape(-1)[0].add_(.25)
    if operation == 'native_backward':
        native_backward_count += 1
        if mode == 'gradient_fault' and native_backward_count == 21:
            first = next(p for n,p in toy.named_parameters() if n.endswith('.lora_A'))
            first.grad.reshape(-1)[0].add_(.1)
    return result

def guard(stage):
    stages.append(stage)
    if mode == 'guard_failure' and stage == 'native_optimizer_step:before':
        raise ValueError('controlled external guard rejection')

def preserve(name, fragment):
    assert name not in preserved
    raw = json.dumps(fragment,allow_nan=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    assert len(raw) <= mechanics.MAX_MILESTONE_BYTES
    preserved[name] = json.loads(raw)
    folder = work / 'milestones'
    folder.mkdir(exist_ok=True)
    with (folder / (name + '.json')).open('xb',buffering=0) as stream:
        stream.write(raw)
        os.fsync(stream.fileno())
    if mode == 'preservation_failure' and name == 'mechanics_state1_restoration':
        raise ValueError('controlled milestone writer interruption')

def check_base():
    current = dict(toy.named_parameters(remove_duplicate=False))
    assert all(current[n] is p and p.data_ptr() == base_pointers[n]
               and mechanics.bitwise_equal_tensor(torch, p, base_values[n])
               and not p.requires_grad and p.grad is None and p.dtype == torch.bfloat16
               for n,p in base.items())

def expect_value_error(function, text=None):
    try:
        function()
    except ValueError as error:
        if text is not None:
            assert text in str(error), (str(error), text)
        return str(error)
    raise AssertionError('expected a fail-closed ValueError')

with tempfile.TemporaryDirectory(prefix='synth-mechanics-cpu-') as directory:
    work = Path(directory)
    if mode in ('normal','durable'):
        options = {} if mode == 'normal' else {'preserve':preserve}
        report = mechanics.qualify_training(torch, toy, rows(), work, call, guard, CONFIG, **options)
        assert report['status'] == 'MECHANICAL_CHECKS_PASSED'
        assert report['completed_native_updates'] == 2
        assert report['completed_reference_updates'] == 1
        assert report['state1_restored_bitwise_before_group2']
        assert report['copied_gradient_optimizer_continuation_bitwise']
        assert report['reference_preserved_state_and_rng']
        assert report['initial_A_gradients_exactly_zero']
        assert all(report['restore_fault_witnesses'].values())
        assert report['changed_adapters_update1'] and report['changed_adapters_update2']
        assert all(n.endswith('.lora_B') for n in report['changed_adapters_update1'])
        counts = report['operation_counts']
        expected = {'native_optimizer_construct':1, 'native_forward':40, 'native_backward':40,
                    'native_row_tensor_materialize':40, 'native_completion_loss_check':40,
                    'native_gradient_reset':3, 'native_gradient_clip':2, 'native_optimizer_step':2,
                    'reference_forward':20, 'reference_autograd_grad':20,
                    'reference_row_tensor_materialize':20, 'reference_completion_loss_check':20,
                    'reference_gradient_accumulate':20, 'reference_parameter_clone':1,
                    'reference_optimizer_construct':1, 'reference_optimizer_load_state':1,
                    'reference_gradient_copy':1, 'reference_gradient_clip':1,
                    'reference_optimizer_step':1, 'checkpoint_save':1, 'checkpoint_load':1,
                    'checkpoint_restore':1, 'initial_adapter_snapshot':1, 'state1_snapshot':1,
                    'restore_fault_adapter':1, 'restore_fault_moment':1, 'restore_fault_python_rng':1,
                    'restore_fault_cpu_rng':1, 'state1_restore_verify':1, 'reference_state_verify':1,
                    'gradient_compare':1, 'optimizer_continuation_verify':1}
        assert counts == expected, (counts, expected)
        assert len(operations) == sum(expected.values())
        assert 'restore_fault_cuda0_rng' not in operations
        for operation, count in expected.items():
            assert stages.count(operation + ':before') == count
            assert stages.count(operation + ':after') == count
        assert len(report['native_row_loss_checks']) == 40
        assert len(report['reference_row_loss_checks']) == 20
        assert all(item['absolute_error'] <= 2e-6 for item in report['native_row_loss_checks']
                   + report['reference_row_loss_checks'])
        assert all(item['passed'] and item['finite'] for item in report['gradient_comparisons'])
        assert 'decoupled_weight_decay' not in report['optimizer_group_fields']
        assert report['optimizer_settings']['foreach'] is False
        assert report['optimizer_settings']['fused'] is False
        descriptor = report['checkpoint']
        data = (work/'STATE1.pt').read_bytes()
        assert descriptor['bytes'] == len(data) < 32*1024*1024
        assert descriptor['sha256'] == hashlib.sha256(data).hexdigest()
        payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
        assert set(payload) == {'schema_version','adapters','optimizer','rng'}
        assert len(payload['adapters']) == 8 and set(payload['rng']) == {'python','cpu'}
        assert all(float(item['step']) == 1 for item in payload['optimizer']['state'].values())
        assert len(list(work.iterdir())) == (1 if mode == 'normal' else 2)
        if mode == 'normal':
            assert not preserved
        else:
            assert list(preserved) == ['mechanics_initial_A_gradient_check', 'mechanics_update1',
                'mechanics_restore_fault_witnesses', 'mechanics_state1_restoration',
                'mechanics_reference_state', 'mechanics_reference_gradient_comparison',
                'mechanics_update2', 'mechanics_optimizer_continuation']
            assert preserved['mechanics_update1']['completed_native_updates'] == 1
            assert preserved['mechanics_update2']['completed_native_updates'] == 2
            assert preserved['mechanics_state1_restoration']['before_group2'] is True
            assert preserved['mechanics_state1_restoration']['restored_bitwise_against_detached_pre_save_snapshot']
            assert preserved['mechanics_reference_gradient_comparison']['passed'] is True
            assert preserved['mechanics_optimizer_continuation']['passed'] is True
            assert len(list((work/'milestones').iterdir())) == 8
            assert all(json.loads(path.read_text()) == preserved[path.stem]
                       for path in (work/'milestones').iterdir())
        # Reusing a completed output path is rejected before any new optimizer
        # construction or mutation, rather than failing only after update 1.
        fresh = make_toy()
        calls_before = list(operations)
        expect_value_error(lambda:mechanics.qualify_training(
            torch,fresh,rows(),work,call,guard,CONFIG),'already exists')
        assert operations == calls_before
        check_base()
        print(json.dumps({'status':'CPU_TOY_TRAINING_MECHANICS_PASSED', 'torch':torch.__version__,
                          'native_updates':2, 'reference_updates':1, 'native_forwards':40,
                          'reference_forwards':20, 'checkpoint_bytes':len(data),
                          'cuda_initialized':torch.cuda.is_initialized(),
                          'pretrained_weights_loaded':False, 'native_qualification':False}))
    elif mode in ('gradient_fault','rng_fault','tamper_checkpoint','guard_failure',
                  'ce_fault','restore_snapshot_fault','reference_optimizer_fault','preservation_failure'):
        expected_error = {'gradient_fault':'independent reference',
                          'rng_fault':'reference pass changed',
                          'tamper_checkpoint':'checkpoint bytes changed',
                          'ce_fault':'independently selected log-softmax',
                          'restore_snapshot_fault':'pre-save witness bitwise',
                          'reference_optimizer_fault':'moments/steps differ bitwise',
                          'preservation_failure':'controlled milestone writer interruption',
                          'guard_failure':'controlled external guard rejection'}[mode]
        error = expect_value_error(lambda: mechanics.qualify_training(
            torch, toy, rows(), work, call, guard, CONFIG, preserve=preserve), expected_error)
        if mode in ('gradient_fault','rng_fault'):
            assert operations.count('native_optimizer_step') == 1
            assert 'reference_optimizer_step' not in operations
        if mode == 'tamper_checkpoint':
            assert 'checkpoint_restore' not in operations
            assert 'reference_forward' not in operations
        if mode == 'guard_failure':
            assert 'native_optimizer_step' not in operations
            assert not (work/'STATE1.pt').exists()
            assert preserved['mechanics_initial_A_gradient_check']['initial_A_gradients_exactly_zero'] is True
        if mode == 'ce_fault':
            assert operations.count('native_forward') == 1
            assert 'native_backward' not in operations and 'native_optimizer_step' not in operations
            measurement = preserved['mechanics_native_row_00_ce_failure']
            assert measurement['passed'] is False and measurement['absolute_error'] > 2e-6
            assert list(preserved) == ['mechanics_native_row_00_ce_failure']
        if mode == 'restore_snapshot_fault':
            assert operations.count('native_optimizer_step') == 1
            assert 'reference_forward' not in operations
            assert preserved['mechanics_state1_restoration']['restored_bitwise_against_detached_pre_save_snapshot'] is False
        if mode == 'reference_optimizer_fault':
            assert operations.count('native_optimizer_step') == 2
            assert operations.count('reference_optimizer_step') == 1
            measurement = preserved['mechanics_optimizer_continuation']
            assert measurement['passed'] is False and measurement['moments_and_steps_bitwise_equal'] is False
            assert preserved['mechanics_update2']['completed_native_updates'] == 2
        if mode == 'gradient_fault':
            measurement = preserved['mechanics_reference_gradient_comparison']
            assert measurement['passed'] is False and any(not row['passed'] for row in measurement['comparisons'])
            assert measurement['atol'] == 1e-6 and measurement['rtol'] == 1e-5
            assert 'mechanics_update2' not in preserved
        if mode == 'rng_fault':
            assert preserved['mechanics_reference_state']['parameters_moments_gradients_rng_preserved'] is False
        if mode == 'preservation_failure':
            assert preserved['mechanics_state1_restoration']['restored_bitwise_against_detached_pre_save_snapshot'] is True
            assert operations.count('native_optimizer_step') == 1
            assert 'reference_forward' not in operations
        check_base()
        print(json.dumps({'status':'EXPECTED_FAILURE_REJECTED', 'mode':mode, 'error':error,
                          'cuda_initialized':torch.cuda.is_initialized()}))
    elif mode == 'contracts':
        good_rows = rows()
        # Mask/EOS/whole-group rejection precedes any optimizer or model call.
        for mutate, message in (
            (lambda r:r[0]['attention_mask'].__setitem__(0,0), 'padding'),
            (lambda r:r[0]['labels'].__setitem__(0,r[0]['input_ids'][0]), 'masked prompt'),
            (lambda r:r[0]['labels'].__setitem__(-1,-100), 'complete supervised'),
            (lambda r:r[0]['input_ids'].__setitem__(-1,8), 'complete supervised'),
            (lambda r:r[0].__setitem__('supervised_tokens',999), 'token count'),
            (lambda r:r[0]['input_ids'].__setitem__(0,True), 'invalid token')):
            candidates = rows()
            mutate(candidates)
            expect_value_error(lambda: mechanics.qualify_training(
                torch,toy,candidates,work,call,guard,CONFIG), message)
        expect_value_error(lambda: mechanics.qualify_training(
            torch,toy,good_rows[:39],work,call,guard,CONFIG), 'exactly 40')
        for overrides in ({'rows_per_update':19}, {'rows_per_update':True},
                          {'adapter_tensors':112}, {'adapter_parameters':1146880},
                          {'require_cuda':True}):
            expect_value_error(lambda: mechanics.qualify_training(
                torch,toy,good_rows,work,call,guard,dict(CONFIG,**overrides)))
        assert not operations and not list(work.iterdir())
        expect_value_error(lambda: mechanics.qualify_training(
            torch,toy,good_rows,work,call,guard,CONFIG,preserve='not callable'),'preservation callback')
        assert not operations
        # Milestone serialization rejects tensors, tuples, nonfinite numbers,
        # nonstring keys and overlarge data before calling the writer.
        observed = []
        writer = lambda name,value:observed.append((name,value))
        for bad in ({'tensor':torch.ones(1)}, {'tuple':(1,2)}, {'bad':float('nan')},
                    {1:'integer key'}, {'large':'x' * mechanics.MAX_MILESTONE_BYTES}):
            expect_value_error(lambda:mechanics._preserve_fragment(writer,'bad',bad))
        assert not observed
        source = {'rows':[{'passed':False,'metric':mechanics._json_number(float('inf'))}]}
        mechanics._preserve_fragment(writer,'valid',source)
        observed[0][1]['rows'][0]['passed'] = True
        assert source['rows'][0]['passed'] is False and source['rows'][0]['metric'] == 'Infinity'
        # Mean completion CE includes the final EOS; prompt predictions do not.
        ids = torch.tensor([[0,1,2,9]],dtype=torch.long)
        labels = torch.tensor([[-100,-100,2,9]],dtype=torch.long)
        logits = torch.zeros((1,4,11),dtype=torch.float32,requires_grad=True)
        logits.data[0,0] = torch.arange(11,dtype=torch.float32)*100
        loss = mechanics.completion_loss(torch,logits,ids,labels)
        expected = (-logits[0,1].log_softmax(-1)[2]-logits[0,2].log_softmax(-1)[9])/2
        assert mechanics.bitwise_equal_tensor(torch,loss,expected)
        loss.backward()
        assert not logits.grad[0,0].any() and logits.grad[0,1].any() and logits.grad[0,2].any()
        assert not logits.grad[0,3].any()
        expect_value_error(lambda: mechanics.completion_loss(torch,logits.bfloat16(),ids,labels),'FP32')
        expect_value_error(lambda: mechanics.completion_loss(torch,logits,ids,torch.tensor([[0,-100,2,9]])), 'mask')
        expect_value_error(lambda: mechanics.completion_loss(torch,logits,ids,torch.tensor([[-100,-100,3,9]])), 'target')
        expect_value_error(lambda: mechanics.completion_loss(torch,logits,ids,torch.full_like(ids,-100)), 'mask')
        assert not mechanics.bitwise_equal_tensor(torch,torch.tensor(0.),torch.tensor(-0.))
        check_base()
        print(json.dumps({'status':'CPU_CONTRACT_REJECTIONS_PASSED','cuda_initialized':False}))
    elif mode == 'state_helpers':
        entries = [(n,p) for n,p in toy.named_parameters() if p.requires_grad]
        optimizer = mechanics._adam(torch,[p for _,p in entries])
        row = rows()[0]
        ids = torch.tensor([row['input_ids']],dtype=torch.long)
        labels = torch.tensor([row['labels']],dtype=torch.long)
        output = toy(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=0)
        mechanics.completion_loss(torch,output.logits,ids,labels).backward()
        optimizer.step()
        state = mechanics.snapshot_training_state(torch,toy,optimizer,require_cuda=False)
        assert any(gradient is not None for gradient in state['gradients'].values())
        assert all(value.data_ptr() != dict(entries)[name].data_ptr() for name,value in state['adapters'].items())
        with torch.no_grad():
            entries[0][1].reshape(-1)[0].add_(1)
            optimizer.state[entries[0][1]]['exp_avg'].reshape(-1)[0].add_(1)
        optimizer.zero_grad(set_to_none=True)
        random.random(); torch.rand(1)
        mechanics.restore_training_state(torch,toy,optimizer,state,require_cuda=False)
        assert mechanics._state_matches(torch,toy,optimizer,state,False)
        # Validate all state before any mutation when inventory/moments/RNG are invalid.
        for mutate in (
            lambda s:s['adapters'].__setitem__('invented',torch.ones(1)),
            lambda s:s['optimizer']['state'][0].__setitem__('invented',True),
            lambda s:s['rng'].__setitem__('cpu',torch.ones(3,dtype=torch.uint8)),
            lambda s:s['rng'].__setitem__('cpu',torch.zeros_like(s['rng']['cpu'])),
            lambda s:s['gradients'].__setitem__(entries[0][0],torch.ones(1)),
            lambda s:s['optimizer']['param_groups'][0].__setitem__('decoupled_weight_decay',True)):
            bad = mechanics._clone_tree(torch,state)
            mutate(bad)
            expect_value_error(lambda:mechanics.restore_training_state(
                torch,toy,optimizer,bad,require_cuda=False))
            assert mechanics._state_matches(torch,toy,optimizer,state,False)
        optimizer.zero_grad(set_to_none=True)
        descriptor = mechanics._save_checkpoint(torch,toy,optimizer,work/'state.pt',False)
        original_bytes = (work/'state.pt').read_bytes()
        try: mechanics._save_checkpoint(torch,toy,optimizer,work/'state.pt',False)
        except FileExistsError: pass
        else: raise AssertionError('write-once checkpoint overwritten')
        assert (work/'state.pt').read_bytes() == original_bytes
        loaded = mechanics._load_checkpoint(torch,descriptor)
        assert set(loaded) == {'schema_version','adapters','optimizer','rng'}
        check_base()
        print(json.dumps({'status':'CPU_STATE_HELPERS_PASSED','cuda_initialized':False}))
    else:
        raise AssertionError('unknown child mode')
assert not torch.cuda.is_initialized()
'''


class NativeMechanicsTests(unittest.TestCase):
    def run_child(self, mode):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                           TRANSFORMERS_OFFLINE="1", USE_HUB_KERNELS="NO", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", CHILD, str(PACKAGE), mode],
                                text=True, capture_output=True, env=environment, timeout=60)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def test_framework_free_import(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import native_mechanics; "
                "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
                "print('FRAMEWORK_FREE_IMPORT')")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(PACKAGE)],
                                text=True, capture_output=True, timeout=30, check=True)
        self.assertEqual("FRAMEWORK_FREE_IMPORT", result.stdout.strip())

    def test_two_updates_reference_restore_and_full_accounting(self):
        report = self.run_child("normal")
        self.assertEqual("CPU_TOY_TRAINING_MECHANICS_PASSED", report["status"])
        self.assertEqual(2, report["native_updates"])
        self.assertEqual(1, report["reference_updates"])
        self.assertEqual(40, report["native_forwards"])
        self.assertEqual(20, report["reference_forwards"])
        self.assertFalse(report["cuda_initialized"])
        self.assertFalse(report["pretrained_weights_loaded"])
        self.assertFalse(report["native_qualification"])

    def test_durable_milestones_have_pure_json_and_unchanged_operation_counts(self):
        self.assertEqual("CPU_TOY_TRAINING_MECHANICS_PASSED", self.run_child("durable")["status"])

    def test_milestone_writer_failure_stops_before_next_group(self):
        self.assertEqual("EXPECTED_FAILURE_REJECTED", self.run_child("preservation_failure")["status"])

    def test_independent_gradient_mismatch_stops_before_second_update(self):
        self.assertEqual("EXPECTED_FAILURE_REJECTED", self.run_child("gradient_fault")["status"])

    def test_independent_ce_mismatch_stops_before_backward(self):
        self.assertEqual("EXPECTED_FAILURE_REJECTED", self.run_child("ce_fault")["status"])

    def test_restoration_uses_independent_pre_save_witness(self):
        self.assertEqual("EXPECTED_FAILURE_REJECTED", self.run_child("restore_snapshot_fault")["status"])

    def test_copied_gradient_optimizer_moment_corruption_is_rejected(self):
        self.assertEqual("EXPECTED_FAILURE_REJECTED", self.run_child("reference_optimizer_fault")["status"])

    def test_reference_rng_mutation_stops_before_second_update(self):
        self.assertEqual("EXPECTED_FAILURE_REJECTED", self.run_child("rng_fault")["status"])

    def test_changed_checkpoint_bytes_rejected_before_restore(self):
        self.assertEqual("EXPECTED_FAILURE_REJECTED", self.run_child("tamper_checkpoint")["status"])

    def test_external_guard_rejection_cannot_hide_an_optimizer_update(self):
        self.assertEqual("EXPECTED_FAILURE_REJECTED", self.run_child("guard_failure")["status"])

    def test_whole_sequence_mask_eos_precision_and_scope_contracts(self):
        self.assertEqual("CPU_CONTRACT_REJECTIONS_PASSED", self.run_child("contracts")["status"])

    def test_detached_state_helpers_validation_and_write_once_checkpoint(self):
        self.assertEqual("CPU_STATE_HELPERS_PASSED", self.run_child("state_helpers")["status"])


if __name__ == "__main__":
    unittest.main()
