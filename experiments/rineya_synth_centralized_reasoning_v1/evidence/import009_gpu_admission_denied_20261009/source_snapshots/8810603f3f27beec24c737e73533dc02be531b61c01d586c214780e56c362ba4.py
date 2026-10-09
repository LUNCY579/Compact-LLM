"""Framework-free parent; explicit tiny CPU final27 fixture, never real study."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

WORK=Path(__file__).resolve().parents[1]
PACKAGE=WORK


def child_code():
    tree=ast.parse((PACKAGE/'tests/test_study_torch_backend.py').read_text())
    original=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHILD' for t in n.targets))
    preamble=original.split("mode = sys.argv[2]")[0]
    return preamble+r'''
sys.path.insert(0,sys.argv[2])
import final_adapter_inference as restore
def fresh_model():
 torch.manual_seed(333);random.seed(777)
 return Toy()
def invoke_fixture(model,fixture,reader=None,journal=None,guard=None):
 return restore.restore_final(torch,model,reader or (lambda d:Path(d['path']).read_bytes()),
    journal or Journal(),guard or (lambda stage:None),arm='M1',fixture=fixture)
with tempfile.TemporaryDirectory(dir=sys.argv[2]) as temp:
 directory=Path(temp)
 b,j=make()
 for u in range(1,28):train(b,u)
 cp=b.save_checkpoint(directory/'CPU_TINY_final27.pt',binding(b))
 fixture={'kind':'CPU_RANDOM_TINY_FINAL27_RECIPE_FIXTURE_ONLY','descriptor':cp,'binding':binding(b),
          'profile':deepcopy(PROFILE),'expected_recipe_sha256':'a'*64,'cohort_sha256':'b'*64}
 fixture['expected_backend_source_sha256']=b.backend_source_sha256
 assert cp['metadata']['scope']=='CPU_RANDOM_TOY' and binding(b)['consumed_occurrences']==540
 ids=torch.tensor([[1,2,3,4]],dtype=torch.long)
 with torch.no_grad():expected=b.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=0).logits.detach().clone()
 target=fresh_model();base={n:p.detach().clone() for n,p in target.named_parameters(remove_duplicate=False)}
 random.seed(888);torch.manual_seed(888);py_before=random.getstate();cpu_before=torch.get_rng_state().clone()
 result=invoke_fixture(target,fixture)
 assert result['status']=='RESTORED_FINAL_ADAPTER_INFERENCE_UNQUALIFIED' and result['CPU_fixture'] is True
 assert result['real_final_study_chain_verified'] is False and result['study_checkpoint_qualification_established'] is False
 assert result['adapter_tensors_copied']==8 and result['optimizer_restored'] is result['training_RNG_restored'] is False
 assert all(mechanics.bitwise_equal_tensor(torch,base[n],dict(target.named_parameters(remove_duplicate=False))[n]) for n in base)
 assert random.getstate()==py_before
 assert torch.equal(torch.get_rng_state(),cpu_before) and result['caller_RNG_preserved'] is True
 assert not torch.equal(torch.get_rng_state(),torch.load(io.BytesIO(Path(cp['path']).read_bytes()),weights_only=True)['state']['rng']['cpu'])
 with torch.no_grad():observed=target(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=0).logits
 assert torch.equal(expected,observed)
 assert not any('optimizer_restore' in k or 'training_RNG_restore' in k for k in result['operation_costs'])
 assert result['operation_costs']['inference_caller_RNG_restore']=={'attempted':1,'returned':1,'failed':0,'pending':0}
 public={'row_id':'d'*64,'prompt_sha256':'e'*64,'prompt_ids':[1,2,3], 'prompt_tokens':3,'attention_mask':[1,1,1]}
 assert restore.validate_raw_input(target,public,result)==public
 badrow=dict(public,labels=[1])
 expect_error(lambda:restore.validate_raw_input(target,badrow,result))
 # Exact byte, kind, final ordinal, arm, recipe and inventory corruption all
 # fail before final adapter copy; no fixture can be laundered as a native chain.
 for name,change in (
  ('kind',lambda f:f['descriptor'].update(kind='STATE1')),
  ('C9',lambda f:f['binding'].update(completed_updates=9,consumed_occurrences=180)),
  ('arm',lambda f:f['descriptor']['metadata'].update(arm='M2')),
  ('recipe',lambda f:f.update(expected_recipe_sha256='0'*64)),
  ('source',lambda f:f.update(expected_backend_source_sha256='0'*64)),
  ('hash',lambda f:f['descriptor'].update(sha256='0'*64))):
  f=deepcopy(fixture);change(f);model=fresh_model();r=invoke_fixture(model,f)
  assert r['status']=='FAILED_RESTORE_MODEL_INVALID_DO_NOT_COLLECT' and not r['inference_allowed'],name
  assert 'final_adapter_parameter_copy' not in r['operation_costs'],name
  expect_error(lambda:restore.validate_raw_input(model,public,r))
 body=torch.load(io.BytesIO(Path(cp['path']).read_bytes()),weights_only=True,map_location='cpu')
 for name,change in (
  ('shape',lambda t:t['state']['adapters'].__setitem__(next(iter(t['state']['adapters'])),torch.zeros(1))),
  ('dtype',lambda t:t['state']['adapters'].__setitem__(next(iter(t['state']['adapters'])),next(iter(t['state']['adapters'].values())).double())),
  ('moment',lambda t:t['state']['optimizer']['state'].pop(0)),
  ('grad',lambda t:t['state']['gradients'].__setitem__(next(iter(t['state']['gradients'])),torch.zeros(1))),
  ('rng_shape',lambda t:t['state']['rng'].__setitem__('cpu',torch.zeros(1,dtype=torch.uint8))),
  ('rng_encoding',lambda t:t['state']['rng'].__setitem__('cpu',torch.zeros_like(t['state']['rng']['cpu'])))):
  corrupted=deepcopy(body);change(corrupted);stream=io.BytesIO();torch.save(corrupted,stream);data=stream.getvalue()
  f=deepcopy(fixture);f['descriptor'].update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  r=invoke_fixture(fresh_model(),f,reader=lambda d,data=data:data)
  assert r['status']=='FAILED_RESTORE_MODEL_INVALID_DO_NOT_COLLECT' and 'final_adapter_parameter_copy' not in r['operation_costs'],name
  assert 'inference_fresh_zeroB_qv_install' not in r['operation_costs'],name
 # A initialization consumes local Torch RNG transiently. Restore the caller's
 # exact state on actual initialization failure, post-return journal failure,
 # and a cleanup journal/guard refusal; archived RNG is never installed.
 original_install=restore.precision.install_qv_lora
 for mode in ('actual_install_failure','post_install_sink_failure','cleanup_sink_failure','cleanup_guard_failure'):
  model=fresh_model();torch.manual_seed(889);before=torch.get_rng_state().clone();failure_journal=Journal()
  def fail_install(*args,**kwargs):
   original_install(*args,**kwargs)
   raise RuntimeError('injected initialized-factor failure')
  if mode=='actual_install_failure':restore.precision.install_qv_lora=fail_install
  if mode=='post_install_sink_failure':failure_journal.fail_after='inference_fresh_zeroB_qv_install'
  if mode=='cleanup_sink_failure':failure_journal.fail_before='inference_caller_RNG_restore'
  def failure_guard(stage):
   if mode=='cleanup_guard_failure' and stage=='adapter_inference_before_inference_caller_RNG_restore':
    raise RuntimeError('injected cleanup guard failure')
  try:r=invoke_fixture(model,fixture,journal=failure_journal,guard=failure_guard)
  finally:restore.precision.install_qv_lora=original_install
  assert not r['inference_allowed'] and r['caller_RNG_preserved'] is True,mode
  assert torch.equal(torch.get_rng_state(),before),mode
  assert 'final_adapter_parameter_copy' not in r['operation_costs'],mode
  assert r['operation_costs']['inference_caller_RNG_restore']=={'attempted':1,'returned':1,'failed':0,'pending':0},mode
 # Returned copy is not erased by a post-return sink failure. Model becomes
 # invalid even though partial factor values really reached it.
 failing=Journal();failing.fail_after='final_adapter_parameter_copy';badmodel=fresh_model()
 r=invoke_fixture(badmodel,fixture,journal=failing)
 assert r['status']=='FAILED_RESTORE_MODEL_INVALID_DO_NOT_COLLECT' and not r['inference_allowed']
 assert r['operation_costs']['final_adapter_parameter_copy']=={'attempted':1,'returned':1,'failed':0,'pending':0}
 native=restore.restore_final(torch,fresh_model(),lambda d:b'',Journal(),lambda stage:None,arm='M1',
  registration=None,chunk_chain=[{'kind':'STATE1'}],latest_terminal_costs={},source_hashes={})
 assert not native['inference_allowed'] and 'final_adapter_parameter_copy' not in native['operation_costs']
 b.close()
assert not torch.cuda.is_initialized()
print(json.dumps({'status':'PASS_CPU_RANDOM_TINY_FINAL27_ADAPTER_INFERENCE_FIXTURE','scope':'CPU_ONLY_NOT_REAL_STUDY_CHECKPOINT',
 'training_updates':27,'whole_sequences':540,'native_tensors':112,'fixture_tensors':8,
 'final_output_numerical_parity':True,'base_unchanged':True,'optimizer_RNG_reused':False,
 'corruption_cases_before_copy':12,'caller_CPU_RNG_exact_before_after':True,'caller_RNG_failure_cases':4,
 'archived_RNG_shape_and_encoding_rejected_before_install':True,
 'returned_copy_failure_cost_preserved':True,'CUDA_initialized':False}))
'''


class FinalInferenceTests(unittest.TestCase):
    def test_framework_free_import(self):
        code="import sys;sys.path.insert(0,sys.argv[1]);import final_adapter_inference;assert not any(x in sys.modules for x in ['torch','transformers']);print('FRAMEWORK_FREE')"
        run=subprocess.run([sys.executable,'-I','-B','-c',code,str(WORK)],text=True,capture_output=True,timeout=20)
        self.assertEqual(0,run.returncode,run.stdout+run.stderr)

    def test_actual_tiny_CPU_final27_restore_raw_output_and_corruption_controls(self):
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        run=subprocess.run([sys.executable,'-I','-B','-c',child_code(),str(PACKAGE),str(WORK)],
                           env=environment,text=True,capture_output=True,timeout=120)
        self.assertEqual(0,run.returncode,run.stdout+run.stderr)
        result=json.loads(run.stdout)
        self.assertEqual('PASS_CPU_RANDOM_TINY_FINAL27_ADAPTER_INFERENCE_FIXTURE',result['status'])
        self.assertFalse(result['optimizer_RNG_reused']);self.assertFalse(result['CUDA_initialized'])


if __name__=='__main__':unittest.main()
