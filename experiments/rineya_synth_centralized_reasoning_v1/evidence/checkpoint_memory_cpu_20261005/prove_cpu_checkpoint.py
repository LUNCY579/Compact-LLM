"""Framework-free parent; one bounded random tiny-Qwen CPU child, no GPU/weights."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

WORK=Path(__file__).resolve().parent
PACKAGE=WORK.parents[1]
CHILD=r'''
from copy import deepcopy
import gc,hashlib,io,json,random,sys,weakref
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import native_precision as precision
import native_mechanics as mechanics
import checkpoint_profile as profile
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
assert not torch.cuda.is_initialized()
PROFILE={'name':'CPU_RANDOM_TINY_QWEN_NONREENTRANT_28_LAYER_112_FACTOR_CONTROL',
 'layers':28,'hidden_size':8,'intermediate_size':16,'attention_heads':2,'KV_heads':1,'head_dim':4,
 'vocab_size':32,'context_cap':64,'rank':8,'alpha':16,'dropout':0.0,'base_dtype':'bfloat16',
 'compute_dtype':'float32','adapter_tensors':112,'adapter_parameters':6272,
 'accumulation_sequences':20,'training_updates':2,'loss':'completion_plus_EOS_sequence_mean_then_mean_over20',
 'atol':1e-6,'rtol':1e-5,'native_parameters_and_budget_unchanged':True,'native_result':False}
def make(use_checkpoint):
 torch.manual_seed(333);random.seed(333)
 config=Qwen3Config(vocab_size=32,hidden_size=8,intermediate_size=16,num_hidden_layers=28,
  num_attention_heads=2,num_key_value_heads=1,head_dim=4,max_position_embeddings=64,
  attention_dropout=0.,tie_word_embeddings=True,bos_token_id=1,eos_token_id=31,pad_token_id=30)
 config._attn_implementation='eager'
 model=Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval().requires_grad_(False)
 precision.install_fp32_compute(model,torch,expected_linear_sites=197)
 torch.manual_seed(101);random.seed(101)
 inv=precision.install_qv_lora(model,torch,rank=8,alpha=16,expected_layers=28)
 assert inv['trainable_tensors']==112 and inv['trainable_parameters']==6272
 entries=mechanics._adapters(model,torch);opt=mechanics._adam(torch,[p for n,p in entries])
 handles=precision.install_fp32_activation_assertions(model,torch)
 installed=profile.install(model,torch,checkpoint) if use_checkpoint else None
 return model,opt,entries,handles,installed
def rows(u):
 return [{'row_id':str((u,i)),'input_ids':[1+(i+j+u)%29 for j in range(11)]+[31],
  'labels':[-100]*8+[1+(i+j+u)%29 for j in range(8,11)]+[31],'attention_mask':[1]*12} for i in range(20)]
def state(model,opt):return mechanics.snapshot_training_state(torch,model,opt,require_cuda=False)
def equal(a,b):return mechanics.equal_tree(torch,a,b)
def compare_state(a,b):
 assert equal(a['adapters'],b['adapters']) and equal(a['optimizer'],b['optimizer']) and equal(a['rng'],b['rng']) and equal(a['gradients'],b['gradients'])
def rng_restore(s):random.setstate(s['python']);torch.set_rng_state(s['cpu'])
def lr(u):return .00005 if u==1 else .0001
def run_group(model,opt,entries,u,*,independent=None,measure=False):
 opt.zero_grad(set_to_none=True);outcomes=[];saved={};refs=[]
 def pack(t):
  storage=t.untyped_storage();key=(storage.data_ptr(),storage.nbytes())
  saved[key]={'bytes':storage.nbytes(),'dtype':str(t.dtype),'shape':list(t.shape),'requires_grad':t.requires_grad}
  return t.detach()
 for slot,row in enumerate(rows(u)):
  mechanics._validate_row(row,31)
  ids=torch.tensor([row['input_ids']],dtype=torch.long);labels=torch.tensor([row['labels']],dtype=torch.long)
  mask=torch.ones_like(ids)
  if measure and slot==0:
   with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):
    output=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    loss,check=mechanics._checked_loss(torch,output.logits,ids,labels)
  else:
   output=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
   loss,check=mechanics._checked_loss(torch,output.logits,ids,labels)
  assert check['absolute_error']<=2e-6
  outcomes.append({'logits':output.logits.detach().clone(),'CE':float(loss.detach())})
  (loss*.05).backward()
  ref=weakref.ref(output.logits);refs.append(ref)
  del output,loss,ids,labels,mask
  gc.collect();assert ref() is None,'previous full logits retained across row boundary'
 grads={n:p.grad.detach().clone() for n,p in entries}
 if independent is not None:
  assert set(grads)==set(independent) and all(torch.allclose(grads[n],independent[n],atol=1e-6,rtol=1e-5) for n in grads)
 norm=torch.nn.utils.clip_grad_norm_([p for n,p in entries],1.,error_if_nonfinite=True,foreach=False)
 opt.param_groups[0]['lr']=lr(u);opt.step();opt.zero_grad(set_to_none=True)
 assert all(p.grad is None for n,p in entries)
 return outcomes,grads,float(norm),{'saved_unique_bytes':sum(x['bytes'] for x in saved.values()),'saved_unique_storages':len(saved),'saved_FP32_bytes':sum(x['bytes'] for x in saved.values() if x['dtype']=='torch.float32')}
def reference(model,opt,entries,u):
 before=state(model,opt);gradients={n:torch.zeros_like(p) for n,p in entries};checks=[]
 for row in rows(u):
  ids=torch.tensor([row['input_ids']],dtype=torch.long);labels=torch.tensor([row['labels']],dtype=torch.long)
  output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True)
  loss,check=mechanics._checked_loss(torch,output.logits,ids,labels);checks.append(check)
  pieces=torch.autograd.grad(loss*.05,[p for n,p in entries],retain_graph=False,create_graph=False)
  for (n,p),g in zip(entries,pieces):gradients[n].add_(g.detach())
  del output,loss,pieces,ids,labels
 compare_state(before,state(model,opt));return gradients,checks
plain,po,pe,ph,_=make(False);ckpt,co,ce,ch,installed=make(True)
assert equal({n:p for n,p in pe},{n:p for n,p in ce})
initial=state(plain,po);rng_restore(initial['rng'])
ref,checks=reference(plain,po,pe,1)
rng_restore(initial['rng']);a,ag,an,am=run_group(plain,po,pe,1,independent=ref,measure=True);a1=state(plain,po)
rng_restore(initial['rng']);b,bg,bn,bm=run_group(ckpt,co,ce,1,independent=ref,measure=True);b1=state(ckpt,co)
assert all(torch.equal(x['logits'],y['logits']) and x['CE']==y['CE'] for x,y in zip(a,b))
assert equal(ag,bg) and an==bn;compare_state(a1,b1)
assert bm['saved_unique_bytes']<am['saved_unique_bytes'] and bm['saved_FP32_bytes']<am['saved_FP32_bytes']
assert installed['counters']['recompute_regions']>0 and not ckpt.training
rng_restore(a1['rng']);a,ag,an,_=run_group(plain,po,pe,2);a2=state(plain,po)
rng_restore(b1['rng']);b,bg,bn,_=run_group(ckpt,co,ce,2);b2=state(ckpt,co)
assert all(torch.equal(x['logits'],y['logits']) and x['CE']==y['CE'] for x,y in zip(a,b));assert equal(ag,bg) and an==bn;compare_state(a2,b2)
# Restore through detached pre-update-one state, preserving current variableLR.
fresh,fo,fe,fh,fi=make(True)
with torch.no_grad():
 for n,p in fe:p.copy_(b1['adapters'][n])
fo.load_state_dict(mechanics._clone_tree(torch,b1['optimizer']));rng_restore(b1['rng'])
compare_state(b1,state(fresh,fo));c,cg,cn,_=run_group(fresh,fo,fe,2);c2=state(fresh,fo)
assert all(torch.equal(x['logits'],y['logits']) and x['CE']==y['CE'] for x,y in zip(b,c));assert equal(bg,cg) and bn==cn;compare_state(b2,c2)
installed['verify']();fi['verify']();installed['remove']();fi['remove']()
for handles in (ph,ch,fh):
 for h in handles['handles']:h.remove()
assert not torch.cuda.is_initialized()
print(json.dumps({'status':'PASS_CPU_RANDOM_TINY_QWEN_CHECKPOINT_CONTROL_NATIVE_UNQUALIFIED','profile':PROFILE,
 'output_CE_all112_grad_Adam_moments_RNG_bitwise_equal':True,'independent_gradient_rows':20,
 'continuous_vs_restored_update2_bitwise_equal':True,'saved_storage_plain':am,'saved_storage_checkpoint':bm,
 'checkpoint_recompute_regions':installed['counters'],'CUDA_initialized':False,'published_weights_opened':False,
 'native_launch_authorized':False,'study_training_qualified':False}))
'''


def main():
    assert not any(name in sys.modules for name in ('torch','transformers'))
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                     USE_TF='0',USE_FLAX='0',USE_TORCH='1',TOKENIZERS_PARALLELISM='false')
    started=datetime.now(timezone.utc)
    run=subprocess.run([sys.executable,'-I','-B','-c',CHILD,str(PACKAGE),str(WORK)],env=environment,
                       text=True,capture_output=True,timeout=180)
    target=WORK/'cpu_checkpoint_proof_20261005.json'
    result=json.loads(run.stdout) if run.returncode==0 else {'status':'FAILED_CPU_TINY_PROOF','stderr':run.stderr,'stdout':run.stdout}
    result.update({'parent_framework_free':True,'child_returncode':run.returncode,'started_at_utc':started.isoformat(),
                   'finished_at_utc':datetime.now(timezone.utc).isoformat(),
                   'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),WORK/'checkpoint_profile.py')}})
    with target.open('x',encoding='utf-8') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'child_returncode':run.returncode,'receipt':str(target)}))
    return run.returncode


if __name__=='__main__':raise SystemExit(main())
