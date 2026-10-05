"""CPU toy Torch mechanics only, isolated from the parent test process.

Torch is imported exclusively in a CPU-only subprocess. No published model,
Transformers model, GPU, corpus, network or native Rineya qualification is used.
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
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(sys.argv[1])))
import native_precision as precision
torch.set_num_threads(1)
torch.manual_seed(101)
assert not torch.cuda.is_initialized()

class RMSNorm(torch.nn.Module):
    def __init__(self,width):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.ones(width))
    def forward(self,x):
        dtype=x.dtype
        work=x.float()
        work=work*torch.rsqrt(work.square().mean(-1,keepdim=True)+1e-6)
        return self.weight*work.to(dtype)

class Rotary(torch.nn.Module):
    def forward(self,x,positions):
        frequencies=positions.float().unsqueeze(-1)*torch.arange(1,5,dtype=torch.float32)/100
        return frequencies.cos().to(x.dtype),frequencies.sin().to(x.dtype)

class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj=torch.nn.Linear(4,8,bias=False)
        self.k_proj=torch.nn.Linear(4,4,bias=False)
        self.v_proj=torch.nn.Linear(4,4,bias=False)
        self.o_proj=torch.nn.Linear(8,4,bias=False)
        self.q_norm=RMSNorm(8)
        self.k_norm=RMSNorm(4)
    def forward(self,x):
        q=self.q_norm(self.q_proj(x))
        kv=self.k_norm(self.k_proj(x))+self.v_proj(x)
        return self.o_proj(q+kv.repeat_interleave(2,dim=-1))

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj=torch.nn.Linear(4,8,bias=False)
        self.up_proj=torch.nn.Linear(4,8,bias=False)
        self.down_proj=torch.nn.Linear(8,4,bias=False)
    def forward(self,x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x))*self.up_proj(x))

class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn=Attention()
        self.mlp=MLP()
        self.input_layernorm=RMSNorm(4)
        self.post_attention_layernorm=RMSNorm(4)
    def forward(self,x):
        x=x+self.self_attn(self.input_layernorm(x))
        return x+self.mlp(self.post_attention_layernorm(x))

class Body(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens=torch.nn.Embedding(8,4)
        self.layers=torch.nn.ModuleList([Layer(),Layer()])
        self.norm=RMSNorm(4)
        self.rotary_emb=Rotary()
    def forward(self,ids):
        x=self.embed_tokens(ids)
        cos,sin=self.rotary_emb(x,torch.arange(ids.shape[1]).unsqueeze(0))
        x=x*cos+sin
        for layer in self.layers:x=layer(x)
        return self.norm(x)

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model=Body()
        self.lm_head=torch.nn.Linear(4,8,bias=False)
        self.lm_head.weight=self.model.embed_tokens.weight
        self.to(dtype=torch.bfloat16)
        self.requires_grad_(False)
    def forward(self,ids):return self.lm_head(self.model(ids))

def expect_error(function):
    try:function()
    except ValueError:return
    raise AssertionError('expected fail-closed ValueError')

toy=Toy()
toy.eval()
original={name:p for name,p in toy.named_parameters(remove_duplicate=False)}
copies={name:p.detach().clone() for name,p in original.items()}
pointers={name:p.data_ptr() for name,p in original.items()}
ids=torch.tensor([[0,1,2]],dtype=torch.long)
profile=precision.install_fp32_compute(toy,torch,expected_linear_sites=15)
assert profile['linear_site_count']==15 and profile['embedding_site_count']==1
assert all(not module.training for _,module in toy.named_modules())
assert toy.lm_head.weight is toy.model.embed_tokens.weight
assert dict(toy.named_parameters(remove_duplicate=False)).keys()==original.keys()
assert all(p is original[n] and p.data_ptr()==pointers[n] and torch.equal(p,copies[n])
           for n,p in toy.named_parameters(remove_duplicate=False))
assert toy(ids).dtype==torch.float32
# Directly verify linear arithmetic and the nonzero-bias preservation case.
site=toy.model.layers[0].self_attn.k_proj
x=torch.randn(1,3,4)
assert torch.equal(site(x),torch.nn.functional.linear(x.float(),site.weight.float()))
biased=torch.nn.Sequential(torch.nn.Embedding(8,4),torch.nn.Linear(4,3,bias=True))
biased.to(dtype=torch.bfloat16);biased.requires_grad_(False)
old_bias=biased[1].bias;old_weight=biased[1].weight
precision.install_fp32_compute(biased,torch,expected_linear_sites=1)
assert biased[1].bias is old_bias and biased[1].weight is old_weight
assert torch.equal(biased[1](x),torch.nn.functional.linear(x.float(),old_weight.float(),old_bias.float()))
# Outer autocast must not secretly recast the registered compute path.
reference_linear=torch.nn.functional.linear(x.float(),site.weight.float())
with torch.autocast('cpu',dtype=torch.bfloat16):
    assert torch.equal(site(x),reference_linear)
    assert toy(ids).dtype==torch.float32
before=toy(ids).detach().clone()
adapter=precision.install_qv_lora(toy,torch,expected_layers=2)
assert adapter['trainable_tensors']==8 and adapter['trainable_parameters']==320
assert toy.lm_head.weight is toy.model.embed_tokens.weight
assert torch.equal(before,toy(ids))
assert precision.zero_lora_status(toy,torch)['all_B_zero']
expected={f'model.layers.{layer}.self_attn.{projection}.lora_{letter}':shape
          for layer in range(2)
          for projection,letter,shape in [('q_proj','A',[8,4]),('q_proj','B',[8,8]),
                                        ('v_proj','A',[8,4]),('v_proj','B',[4,8])]}
assert {item['name']:item['shape'] for item in adapter['inventory']}==expected
assert all(item['dtype']=='float32' and item['device']=='cpu' and item['requires_grad']
           for item in adapter['inventory'])
assert all(n in expected for n,p in toy.named_parameters() if p.requires_grad)
hooks=precision.install_fp32_activation_assertions(toy,torch)
toy(ids)
assert all(count>0 for count in hooks['observations'].values())
expect_error(lambda:toy.model.norm(torch.ones(1,3,4,dtype=torch.bfloat16)))
for handle in hooks['handles']:handle.remove()
trainable=[p for p in toy.parameters() if p.requires_grad]
optimizer=torch.optim.AdamW(trainable,lr=1e-4,betas=(.9,.999),eps=1e-8,
                           weight_decay=0,foreach=False,fused=False,capturable=False,differentiable=False)
# Torch2.6's actual schema intentionally lacks Torch2.10's added field.
assert 'decoupled_weight_decay' not in optimizer.param_groups[0]
loss=(toy(ids)-torch.arange(8,dtype=torch.float32)).square().mean()
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
assert all(p.grad is None for p in original.values())
torch.nn.utils.clip_grad_norm_(trainable,1.0)
optimizer.step()
assert not precision.zero_lora_status(toy,torch)['all_B_zero']
assert all(p is dict(toy.named_parameters(remove_duplicate=False))[name]
           and p.data_ptr()==pointers[name] and p.dtype==torch.bfloat16 and torch.equal(p,copies[name])
           for name,p in original.items())
assert all(state['exp_avg'].dtype==torch.float32 and state['exp_avg_sq'].dtype==torch.float32
           for state in optimizer.state.values())
expect_error(lambda:precision.install_qv_lora(toy,torch,expected_layers=2))
fresh=Toy();fresh_names=set(dict(fresh.named_parameters(remove_duplicate=False)))
expect_error(lambda:precision.install_fp32_compute(fresh,torch,expected_linear_sites=197))
assert set(dict(fresh.named_parameters(remove_duplicate=False)))==fresh_names
expect_error(lambda:precision.install_qv_lora(fresh,torch,expected_layers=2))
expect_error(lambda:precision.install_fp32_compute(fresh,torch,expected_linear_sites=True))
fresh.model.layers[0].self_attn.k_proj.weight.requires_grad_(True)
expect_error(lambda:precision.install_fp32_compute(fresh,torch,expected_linear_sites=15))
fresh=Toy();precision.install_fp32_compute(fresh,torch,expected_linear_sites=15)
expect_error(lambda:precision.install_fp32_compute(fresh,torch,expected_linear_sites=15))
expect_error(lambda:precision.install_qv_lora(fresh,torch,rank=True,expected_layers=2))
expect_error(lambda:precision.install_qv_lora(fresh,torch,alpha=8,expected_layers=2))
expect_error(lambda:precision.install_qv_lora(fresh,torch,expected_layers=28))
shared=Toy();shared.duplicate=shared.lm_head
expect_error(lambda:precision.install_fp32_compute(shared,torch,expected_linear_sites=15))
renorm=Toy();renorm.model.embed_tokens.max_norm=1.0
expect_error(lambda:precision.install_fp32_compute(renorm,torch,expected_linear_sites=15))
assert not torch.cuda.is_initialized()
print(json.dumps({'status':'CPU_TOY_COMPONENT_MECHANICS_PASSED','torch':torch.__version__,
                  'linear_sites':15,'adapter_tensors':8,'adapter_parameters':320,
                  'cuda_initialized':False,'native_Rineya_qualified':False,
                  'published_weights_loaded':False}))
'''


class NativePrecisionTests(unittest.TestCase):
    def test_module_import_requires_no_framework(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import native_precision; "
                "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
                "print('FRAMEWORK_FREE_IMPORT')")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(PACKAGE)],
                                text=True, capture_output=True, timeout=30, check=True)
        self.assertEqual("FRAMEWORK_FREE_IMPORT", result.stdout.strip())

    def test_isolated_cpu_toy_compute_optimizer_and_fail_closed_contracts(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   USE_HUB_KERNELS="NO", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", CHILD, str(PACKAGE)],
                                text=True, capture_output=True, timeout=60, env=env)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual("CPU_TOY_COMPONENT_MECHANICS_PASSED", report["status"])
        self.assertEqual(15, report["linear_sites"])
        self.assertEqual(8, report["adapter_tensors"])
        self.assertEqual(320, report["adapter_parameters"])
        self.assertFalse(report["cuda_initialized"])
        self.assertFalse(report["native_Rineya_qualified"])
        self.assertFalse(report["published_weights_loaded"])


if __name__ == "__main__":
    unittest.main()
