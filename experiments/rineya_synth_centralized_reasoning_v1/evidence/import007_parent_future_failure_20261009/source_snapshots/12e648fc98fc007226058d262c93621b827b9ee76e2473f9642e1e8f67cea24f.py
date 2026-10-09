"""Staged full27 qualification controller and owned root-only worker.

The qualification proxy retains the real backend math, adds independent first
resumed-update reference controls and detached saved-boundary witnesses. Native
dispatch remains closed until separate reviewed source/manifest integration.
"""
from copy import deepcopy
import argparse
import importlib.metadata
import os
from pathlib import Path
import random
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1] if HERE.name=='full27_qualification_work' and HERE.parent.name=='artifacts' else HERE
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(HERE))
import full27_contract as contract
import full27_chunk as chunks
from rs_contract import canonical,digest,require


class ForwardDose:
    """All model calls, including parity/reference, share fixed scope budgets."""
    OPERATIONS={'training_forward':'train_forwards','reference_forward':'reference_forwards',
        'zero_parity_base_forward':'zero_parity_forwards','zero_parity_adapter_forward':'zero_parity_forwards'}
    def __init__(self,scope):
        self.expected=scope['dose'];self.stack=[];self.calls=0;self.positions=0
        self.counts={k:0 for k in ('train_forwards','reference_forwards','zero_parity_forwards')}
    def wrap(self,call):
        def invoke(name,function):
            self.stack.append(name)
            try:return call(name,function)
            finally:self.stack.pop()
        return invoke
    def charge(self,length):
        require(self.stack and self.stack[-1] in self.OPERATIONS and type(length) is int and 0<length<=512,'registered complete physical model forward')
        key=self.OPERATIONS[self.stack[-1]]
        require(self.calls<self.expected['maximum_forward_calls'] and self.positions+length<=102400
                and self.counts[key]<self.expected[key],'exact qualification forward category/position ceiling')
        self.calls+=1;self.positions+=length;self.counts[key]+=1
    def before(self,model,args,kwargs):
        ids=kwargs.get('input_ids')
        require(hasattr(ids,'shape') and len(ids.shape)==2 and ids.shape[0]==1 and str(ids.dtype)=='torch.int64','typed single model input')
        self.charge(int(ids.shape[1]))
    def snapshot(self):return {'attempted_calls':self.calls,'input_positions':self.positions,'operations':deepcopy(self.counts)}
    def complete(self):
        require(self.calls==self.expected['maximum_forward_calls'] and self.positions==self.expected['expected_forward_input_positions']
                and self.counts=={k:self.expected[k] for k in self.counts},'every registered physical model call/position completed')


class QualificationProxy:
    """Delegates unchanged optimizer/loss/restore; qualification extras only."""
    def __init__(self,adapter,scope,preserve):
        self.adapter,self.backend,self.scope,self.preserve=adapter,adapter.backend,scope,preserve
        self.reference=None;self.compared=False;self.saved_witness=None;self.restore_witness=None;self.reference_report=None
        self.first_ordinal=(scope['chunk']-1)*9+1
    def __getattr__(self,name):return getattr(self.backend,name)
    @property
    def call(self):return self.backend.call
    @call.setter
    def call(self,value):self.backend.call=value
    def reset_gradients(self,ordinal):
        if ordinal==self.first_ordinal and self.scope['chunk']>1:
            start=(ordinal-1)*20
            self.reference=self.adapter.prepare_reference(deepcopy(self.scope['rows'][start:start+20]))
        return self.backend.reset_gradients(ordinal)
    def finite_clip_gradients(self,*,max_norm):
        if self.reference is not None and not self.compared:
            result=self.adapter.compare_reference(self.reference)
            require(result.get('passed') is True and result.get('count')==self.adapter.expected_gradient_count
                    and result.get('atol')==1e-6 and result.get('rtol')==1e-5,'all registered independent preclip gradients')
            self.preserve({'event':'reference_gradient_comparison','payload':deepcopy(result)})
            self.reference_report=deepcopy(result)
            self.reference=None
            self.compared=True
        return self.backend.finite_clip_gradients(max_norm=max_norm)
    def restore_checkpoint(self,descriptor,binding):
        result=self.backend.restore_checkpoint(descriptor,binding)
        restored=self.adapter.witness(deepcopy(binding))
        expected=self.scope['previous']['resume_descriptor']['envelope']['qualification_evidence']['witness']
        require(restored==expected,'complete restored qualification boundary witness equality')
        self.restore_witness=restored
        self.preserve({'event':'restored_boundary_witness','payload':deepcopy(restored)})
        return result
    def save_checkpoint(self,path,binding):
        before=self.adapter.witness(deepcopy(binding))
        descriptor=self.backend.save_checkpoint(path,binding)
        require(before==self.adapter.witness(deepcopy(binding)),'boundary serialization must not mutate parameters/moments/RNG/cursor')
        if hasattr(self.backend,'_read') and hasattr(self.adapter,'_state_witness'):
            payload=self.backend._read(deepcopy(descriptor),deepcopy(binding))
            saved=self.adapter._invoke('qualification_saved_archive_state_hash_witness',lambda:
                self.adapter._state_witness(payload['state'],deepcopy(binding)))
            require(before==saved,'serialized boundary bytes must match detached pre-save witness')
        self.saved_witness=before
        self.preserve({'event':'detached_boundary_witness','payload':deepcopy(before)})
        return descriptor


def run_component(scope,factory,row_get,call,guard,preserve,*,admission=None):
    """A CPU fake or externally owned native lazy factory; no admission default."""
    require(scope['purpose']==contract.PURPOSE and scope['authorization']==contract.AUTHORIZATION
            and scope['study_warmstart_allowed'] is False,'qualification-only exact purpose')
    arm,chunk=scope['arm'],scope['chunk'];registration=scope['registration']
    require(scope['rows']==contract.packet(registration,arm),'all540 frozen whole occurrences before factory/outputs')
    require(scope['recipe']==chunks.recipe(registration,arm,getattr(factory,'execution_scope',None)),'exact qualification source/arm/recipe')
    require(scope['dose']==contract.expected_work(registration,arm,chunk,scope['rows']),'exact9-update physical work budgets')
    previous=scope['previous'];latest=scope['latest_costs'];proxies=[];adapters=[];dose=ForwardDose(scope);closed=False
    def keep(adapter,event):
        name=event['event'];payload=event['payload'];shared=adapter.backend.call
        publication=shared('qualification_'+name+'_write',lambda:preserve(deepcopy(event)))
        return shared('qualification_'+name+'_verify',lambda:chunks._verify_publication(publication,payload))
    def make(**kwargs):
        adapter=factory(scope,dose=dose,**kwargs)
        adapters.append(adapter)
        require(adapter.backend.arm==arm and adapter.backend.seed==101,'fresh arm/seed')
        parity=adapter.zero_parity(deepcopy(scope['rows'][(chunk-1)*180]))
        require(parity.get('equal') is True and parity.get('max_error')==0.0 and parity.get('all_B_zero') is True,'physical fresh zeroB parity')
        keep(adapter,{'event':'zero_parity','payload':deepcopy(parity)})
        proxy=QualificationProxy(adapter,scope,lambda event:keep(adapter,event));proxies.append(proxy);return proxy
    make.execution_scope=getattr(factory,'execution_scope',None)
    resume=previous['resume_descriptor'] if previous is not None else None
    def cleanup():
        nonlocal closed
        if closed:return
        closed=True
        if callable(getattr(factory,'cleanup',None)):factory.cleanup()
        else:
            for adapter in adapters:adapter.close()
    def before_publish(backend,descriptor,binding):
        proxy=proxies[0]
        require(proxy.saved_witness is not None and (chunk==1 or proxy.compared),'complete boundary witness/reference qualification')
        dose.complete()
        decoder=None
        if callable(getattr(factory,'validate_work',None)):
            factory.validate_work(scope)
        cleanup()
        if callable(getattr(factory,'work_evidence',None)):
            decoder=factory.work_evidence(scope)
        evidence={'purpose':contract.PURPOSE,'binding':deepcopy(binding),'witness':deepcopy(proxy.saved_witness),
            'reference_gradient_comparison':deepcopy(proxy.reference_report),'physical_dose':dose.snapshot(),
            'decoder_work':deepcopy(decoder),'cleanup_completed':True,'study_warmstart_allowed':False,
            'continuous27_bitwise_parity_established':False,'CPU_fake_control':make.execution_scope==chunks.FAKE_SCOPE}
        evidence['sha256']=digest(evidence)
        return evidence
    result=chunks.run_chunk(registration,arm,chunk,make,row_get,dose.wrap(call),guard,preserve,
        resume_descriptor=resume,latest_terminal_costs=latest,admission=admission,
        before_publish=before_publish,cleanup=cleanup)
    witness=proxies[0].saved_witness if proxies else None
    result.update({'qualification_purpose':contract.PURPOSE,'qualification_witness':deepcopy(witness),
        'physical_dose':dose.snapshot(),
        'continuous27_bitwise_parity_established':False,'study_warmstart_allowed':False,'native_launch_authorized':False,
        'study_training_authorized':False,'DEV_collection_authorized':False,'scoring_authorized':False})
    result['sha256']=digest({k:v for k,v in result.items() if k!='sha256'})
    return result


class NativeFactory:
    """The only production factory: fresh pinned dense load after owned admission.

    The injected factories in run_component are CPU component seams. run() never
    accepts a supplied factory, framework, model, reader or admission callback.
    """
    execution_scope=chunks.NATIVE_SCOPE
    def __init__(self,output,journal,scope):
        self.output,self.journal,self.scope=Path(output),journal,scope
        self.torch=None;self.hooks=[];self.profiles=[];self.backends=[];self.call=None;self.cleaned=False
    def __call__(self,scope,*,call,guard,dose,arm,seed,recipe_sha256,cohort_sha256):
        from native_loader import load_titan
        from study_torch_backend import StudyTorchBackend
        from backend_qual_native import NativeAdapter
        import checkpoint_successor_profile
        require(scope is self.scope and arm==scope['arm'] and seed==101 and recipe_sha256==scope['recipe']['sha256']
                and cohort_sha256==scope['recipe']['cohort_sha256'],'exact fresh factory source/arm/seed')
        self.call=call
        for name,value in scope['offline_environment'].items():os.environ[name]=value
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        require(shutil.disk_usage(PACKAGE).free>=4*1024**3,'disk admission floor')
        def frameworks():
            import torch
            from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
            from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
            self.torch=torch
            return torch,Qwen3Config,Qwen3ForCausalLM
        torch,config_class,model_class=call('qualification_framework_import',frameworks)
        require(str(torch.__version__)==scope['runtime_versions']['torch']
                and importlib.metadata.version('transformers')==scope['runtime_versions']['transformers'],'executing pinned framework versions')
        import psutil
        rss=psutil.Process().memory_info().rss
        require(psutil.virtual_memory().available>=1.25*1024**3+max(0,2*1024**3-rss),'post-import whole growth budget')
        call('cuda_context_init',torch.cuda.init)
        require(call('qualification_GPU_preload_admission',lambda:torch.cuda.mem_get_info()[0])>=4*1024**3,'fresh GPU preload floor')
        def configure():
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
            random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
        call('qualification_precision_seed_configure',configure)
        model,loader,hashes=load_titan(torch,config_class,model_class,scope['model_dir'],call,guard)
        self.journal.evidence('loader',loader)
        def meter():
            hook=model.register_forward_pre_hook(dose.before,with_kwargs=True);self.hooks.append(hook)
        call('qualification_forward_meter_install',meter)
        def construct():
            backend=StudyTorchBackend(torch,model,call,guard,arm=arm,seed=seed,recipe_sha256=recipe_sha256,
                cohort_sha256=cohort_sha256,execution_scope=chunks.NATIVE_SCOPE,source_hashes=hashes)
            self.backends.append(backend);return backend
        backend=call('qualification_backend_construct',construct)
        def install():
            from torch.utils.checkpoint import checkpoint,set_checkpoint_early_stop
            installed=checkpoint_successor_profile.install(model,torch,checkpoint,early_stop_context=set_checkpoint_early_stop,
                expected_layers=28,call=call,guard=guard)
            self.profiles.append(installed)
        call('qualification_full_checkpoint_install',install)
        return call('qualification_native_adapter_construct',lambda:NativeAdapter(torch,backend,call,guard,dose))
    def validate_work(self,scope):
        require(len(self.backends)==len(self.profiles)==1,'one owned native backend and checkpoint profile')
        profile=self.profiles[0]
        self.call('qualification_checkpoint_attributes_verify',lambda:profile['verify'](installed=True))
        contract.prerequisite.validate_decoder_work(profile['snapshot'](),scope['checkpoint_amendment'])
        counts=self.backends[0].primitive_operation_snapshot()
        for name,count in (('native_base_content_verify',scope['dose']['native_base_content_checks']),
            ('training_forward',180),('training_backward',180),('gradient_reset',9),('optimizer_step',9),('consumed_gradient_clear',9)):
            require(counts.get(name)==dict(attempted=count,returned=count,failed=0,pending=0),'exact native primitive work '+name)
        shared=self.call.snapshot();reference=scope['dose']['reference_autograd_calls']
        for name,count in (('reference_forward',reference),('reference_autograd_grad',reference),
                           ('qualification_gradient_compare',int(reference>0))):
            require(shared.get(name,dict(attempted=0,returned=0,failed=0,pending=0))==dict(attempted=count,returned=count,failed=0,pending=0),
                    'exact independent reference primitive work '+name)
        for index in range(28):
            for phase in ('forward','recompute','no_grad'):
                count=scope['checkpoint_amendment']['counts']['per_layer_'+phase]
                require(shared.get('checkpoint_layer%02d_%s'%(index,phase))==dict(attempted=count,returned=count,failed=0,pending=0),
                        'decoder costs must match all paired actual regions')
        return True
    def cleanup(self):
        if self.cleaned:return
        self.cleaned=True;errors=[]
        for profile in self.profiles:
            try:profile['remove']()
            except BaseException as error:errors.append(error)
        def release():
            failures=[]
            for item,method in [(backend,'close') for backend in self.backends]+[(hook,'remove') for hook in self.hooks]:
                try:getattr(item,method)()
                except BaseException as error:failures.append(error)
            if failures:raise RuntimeError('owned hook removal failed') from failures[0]
        try:
            if self.call is not None:self.call('qualification_owned_hooks_remove',release)
            else:release()
        except BaseException as error:
            errors.append(error)
            # Containment cleanup still runs if a failed guard/journal blocked it.
            try:release()
            except BaseException as failure:errors.append(failure)
        if errors:raise RuntimeError('qualification owned cleanup failed') from errors[0]
    def work_evidence(self,scope):
        snapshot=self.profiles[0]['snapshot']()
        require(snapshot['removed'] is True,'checkpoint forward attributes restored before publication')
        contract.prerequisite.validate_decoder_work(snapshot,scope['checkpoint_amendment'])
        path=self.output/'decoder_regions.jsonl';records=snapshot.pop('regions')
        def write():
            with path.open('xb') as stream:
                for row in records:
                    raw=(canonical(row)+'\n').encode();require(stream.write(raw)==len(raw),'complete decoder record')
                stream.flush();os.fsync(stream.fileno())
            return {'path':str(path),'rows':len(records),'bytes':path.stat().st_size,'sha256':contract.native.filehash(path)}
        descriptor=self.call('qualification_decoder_regions_write',write)
        self.call('qualification_decoder_regions_verify',lambda:contract.verify_decoder_file(snapshot,descriptor,scope['checkpoint_amendment'],self.output))
        snapshot['region_file']=descriptor
        return snapshot


def run(config_path,output):
    """Integrated root-only owned purpose; staging rejection precedes all APIs."""
    scope=contract.preflight(config_path)
    from inference_qual_worker import make_guard
    import worker_ram_probe
    from native_worker import Journal,verify_worker_ownership,preserve_terminal
    output=Path(output).absolute()
    require(output.resolve().parent==(PACKAGE/'artifacts').resolve() and output.is_dir() and not output.is_symlink(),'owned new output leaf')
    owner=contract.native.read_json(output/'supervisor_ownership.json')
    ownership=verify_worker_ownership(owner,config_path)
    class DurableJournal(Journal):
        def emit(self,value):
            super().emit(value);os.fsync(self.stream.fileno())
    journal=DurableJournal(output);start=time.monotonic();factory=NativeFactory(output,journal,scope)
    guard=make_guard(output,owner,start,lambda:factory.torch,available=worker_ram_probe.available_bytes)
    journal.guard=guard;journal.evidence('worker_ownership_verified',ownership)
    sequence=0
    def preserve(event):
        nonlocal sequence
        if event['event']=='checkpoint_path':
            path=output/('qualification_C%d.pt'%(scope['chunk']*9));require(not path.exists() and not path.is_symlink(),'new private qualification boundary')
            return str(path)
        sequence+=1;payload=event['payload'];path=output/('%04d_%s.json'%(sequence,event['event']))
        with path.open('xb') as stream:
            raw=(canonical(payload)+'\n').encode();require(stream.write(raw)==len(raw),'complete qualification publication');stream.flush();os.fsync(stream.fileno())
        descriptor={'path':str(path),'bytes':len(raw),'sha256':contract.native.filehash(path),'payload_sha256':digest(payload),'durable':True}
        journal.emit({'event':'qualification_measurement_preserved','name':event['event'],'publication':descriptor})
        return descriptor
    def admission(request):
        guard('before_full27_framework_model_load_and_constructor');observed=time.monotonic()
        return {'binding':request,'authorized':True,'owned_supervision_verified':True,'model_load_and_constructor_in_scope':True,
                'observed_monotonic':observed,'expires_monotonic':observed+.2}
    report={'status':'FAILED_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED','resume_descriptor':None}
    try:
        rows={row['row_id']:row for row in scope['rows']}
        report=run_component(scope,factory,rows.__getitem__,journal.call,guard,preserve,admission=admission)
        if report['status']=='COMPLETE_QUALIFICATION_CHUNK_STUDY_UNAUTHORIZED':
            contract.preflight(config_path);guard('full27_qualification_completed')
    except BaseException as error:
        report.update({'status':'FAILED_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED','resume_descriptor':None,
                       'error_type':type(error).__name__,'error':str(error)})
    finally:
        # run_component owns cleanup before durable promotion, including failure.
        # Any earlier prevalidation failure has no framework/model resources.
        report.update({'config_sha256':scope['config_sha256'],'source_commit':scope['source_commit'],
            'wall_seconds':time.monotonic()-start,'operation_accounting':journal.snapshot(),
            'production_training_authorized':False,'study_training_authorized':False,'study_warmstart_allowed':False,
            'DEV_collection_authorized':False,'scoring_authorized':False,'protected_bank_access':False,'quality_gain_established':False,
            'full_resumed9_qualified':False,'full27_qualified':False,'continuous27_bitwise_parity_established':False,'automatic_retry_authorized':False})
        report['sha256']=digest({k:v for k,v in report.items() if k!='sha256'})
        preserve_terminal(output,report,journal,factory.torch)
    return 0 if report['status']=='COMPLETE_QUALIFICATION_CHUNK_STUDY_UNAUTHORIZED' else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();return run(args.config,args.output_dir)


if __name__=='__main__':raise SystemExit(main())
