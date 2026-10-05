"""Owned fresh native-long worker; staging rejects before all native APIs."""
from copy import deepcopy
from pathlib import Path
import argparse,hashlib,importlib.metadata,os,random,sys,time
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1] if HERE.name=='long_native_qualification_work' and HERE.parent.name=='artifacts' else HERE
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(HERE))
import long_native_contract as contract
import long_native_core as core
from rs_contract import canonical,digest,require


class NoDecode:
    def decode(self,*args,**kwargs):raise ValueError('long qualification never decodes or grades answers')


def verified_files(scope,call,guard):
    """Actual byte verification only after fresh owned admission."""
    maps={'source':scope['source_pins'],'runtime':scope['runtime_pins'],'data':scope['data_pins'],'model':scope['model_pins'],
          'encoder_assets':{str(Path(scope['tokenizer_binding']['model_dir'])/name):entry['sha256'] for name,entry in scope['tokenizer_binding']['assets'].items()},
          'encoder_bindings':scope['tokenizer_binding']['binding_sources']}
    actual={}
    for section,entries in maps.items():
        actual[section]={}
        for name,want in entries.items():
            path=Path(scope['model_dir'])/name if section=='model' else Path(name)
            require(path.is_file() and not path.is_symlink(),'regular admitted pin file')
            def read_hash(path=path):
                value=hashlib.sha256()
                with path.open('rb') as stream:
                    for raw in iter(lambda:stream.read(2*1024**2),b''):
                        guard('long_pin_chunk_before');value.update(raw);guard('long_pin_chunk_after')
                return value.hexdigest()
            value=call('long_%s_file_verify'%section,read_hash);require(value==want,'actual admitted '+section+' byte drift')
            actual[section][name]=value
    return actual


def verify_encoder(scope,call):
    from long_inference_inputs import LocalPromptTokenizer,TEMPLATE_KWARGS
    tokenizer=call('long_encoder_construct',lambda:LocalPromptTokenizer(scope['tokenizer_binding']))
    packet=contract.native.read_json(scope['packet_path']);require(contract.native.filehash(Path(scope['packet_path']))==scope['packet_sha256'],'actual TRAIN packet bytes')
    records=contract.native.read_jsonl(PACKAGE/'data/crossed_source2_v1_20261003/candidate_records.jsonl')
    source=[r for r in records if r['row_id']==scope['source_item']['source_row_id']]
    require(len(source)==1 and source[0]['arm']=='M1' and source[0]['split']=='train' and source[0]['skill']=='R','unique TRAIN prompt encoder source')
    messages=source[0]['messages'];short=scope['short_parity_row']
    rendered=call('long_short_template_render',lambda:tokenizer._template.render(messages=deepcopy(messages),tools=None,**TEMPLATE_KWARGS))
    require(hashlib.sha256(rendered.encode()).hexdigest()==short['prompt_sha256'] and short['prompt_tokens']==419,'typed renderedUTF8 short419 hash')
    require(call('long_short_prompt_encode',lambda:tokenizer.encode_prompt(messages))==short['prompt_ids'],'exact short419 retokenization')
    require(digest(scope['messages'])==scope['row']['prompt_sha256'] and scope['messages'][:-1]==messages[:-1]
        and scope['messages'][-1]['content'].startswith(messages[-1]['content']),'typed canonicalmessagesJSON long TRAIN task retained')
    require(call('long_full_prompt_encode',lambda:tokenizer.encode_prompt(scope['messages']))==scope['row']['prompt_ids'],'exact full untruncated long retokenization')
    require(call('long_encoder_binding_postcheck',tokenizer.verify_binding)==scope['tokenizer_binding'],'encoder assets/runtime binding unchanged')
    return {'short_hash_kind':'SHA256_RENDERED_UTF8_CHAT_TEMPLATE','long_hash_kind':'SHA256_CANONICAL_MESSAGES_JSON',
        'short_tokens':419,'long_tokens':scope['row']['prompt_tokens'],'full_retokenization_matches':True,
        'tokenizer_binding_sha256':digest(scope['tokenizer_binding']),'native_long_context_qualified':False}


class NativeFactory:
    execution_scope=contract.NATIVE
    def __init__(self,owner,journal):self.owner,self.journal=owner,journal;self.torch=None;self.hooks=[];self.call=None;self.cleaned=False
    def cleanup(self):
        if self.cleaned:return
        self.cleaned=True
        def remove():
            errors=[]
            for handle in self.hooks:
                try:handle.remove()
                except BaseException as error:errors.append(error)
            if errors:raise RuntimeError('owned native-long hook removal failed') from errors[0]
        try:
            if self.call:self.call('long_owned_hooks_remove',remove)
            else:remove()
        finally:remove()
    def __call__(self,scope,*,call,guard,dose):
        self.call=call
        # Contract sources and static metadata were frozen before dispatch; all
        # actual byte/encoder work is included inside the admitted factory.
        pins_before=verified_files(scope,call,guard)
        encoding=verify_encoder(scope,call)
        for name,value in scope['offline_environment'].items():os.environ[name]=value
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        def frameworks():
            import torch
            from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
            from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
            self.torch=torch;return torch,Qwen3Config,Qwen3ForCausalLM
        torch,config_class,model_class=call('long_framework_import',frameworks)
        require(str(torch.__version__)==scope['runtime_versions']['torch'] and importlib.metadata.version('transformers')==scope['runtime_versions']['transformers'],'actual framework versions')
        import psutil
        rss=psutil.Process().memory_info().rss
        require(psutil.virtual_memory().available>=1.25*1024**3+max(0,2*1024**3-rss),'post-import whole growth budget')
        call('cuda_context_init',torch.cuda.init)
        require(call('long_GPU_preload_admission',lambda:torch.cuda.mem_get_info()[0])>=4*1024**3,'fresh GPU preload floor')
        def configure():
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
            random.seed(101);torch.manual_seed(101);torch.cuda.manual_seed_all(101);torch.cuda.reset_peak_memory_stats()
        call('long_precision_seed_configure',configure)
        from native_loader import load_titan,verify_frozen_base
        from native_precision import install_fp32_compute,install_fp32_activation_assertions,install_qv_lora,zero_lora_status,adapter_inventory
        from long_inference_component import InferenceComponent
        model,loader,source_hashes=load_titan(torch,config_class,model_class,scope['model_dir'],call,guard)
        self.journal.evidence('loader',loader)
        require(not any(name.endswith(('.lora_A','.lora_B')) for name,_ in model.named_parameters()),'fresh dense pinned base; no checkpoint adapters')
        def actual_hook(module,args,kwargs):
            ids=kwargs.get('input_ids');require(torch.is_tensor(ids) and tuple(ids.shape[:1])==(1,) and len(ids.shape)==2 and ids.dtype==torch.long,'actual typed single-batch ids')
            values=call('long_physical_prefix_read',lambda:ids.detach().cpu().tolist()[0]);dose.charge(values)
        def install_meter():self.hooks.append(model.register_forward_pre_hook(actual_hook,with_kwargs=True))
        call('long_physical_meter_install',install_meter)
        compute=call('long_fp32_compute_install',lambda:install_fp32_compute(model,torch))
        def activation():
            value=install_fp32_activation_assertions(model,torch);self.hooks.extend(value['handles']);return value
        active=call('long_activation_hooks_install',activation)
        call('long_model_eval_set',model.eval)
        def tensor_hash(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        def rng():return {'python_sha256':digest(random.getstate()),'cpu':tensor_hash(torch.get_rng_state()),'cuda0':tensor_hash(torch.cuda.get_rng_state(0))}
        dense_rng=call('long_dense_start_RNG_snapshot',rng)
        ids=call('long_parity_ids_materialize',lambda:torch.tensor([scope['short_parity_row']['prompt_ids']],dtype=torch.long,device='cuda:0'))
        def logits(output):
            value=output.logits;require(torch.is_tensor(value) and tuple(value.shape)==(1,419,151936) and value.dtype==torch.float32
                and str(value.device)=='cuda:0' and not value.requires_grad and bool(torch.isfinite(value).all()),'full finite FP32 noGrad parity logits')
            return value
        with torch.no_grad():dense=call('long_native_dense_parity_forward',lambda:logits(model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True)))
        adapters=call('long_new_rank8_qv_install',lambda:install_qv_lora(model,torch,rank=8,alpha=16))
        zero=call('long_zeroB_before',lambda:zero_lora_status(model,torch));require(zero['all_B_zero'] is True and zero['qv_site_count']==56,'all56 fresh B zero')
        with torch.no_grad():adapted=call('long_native_zeroB_parity_forward',lambda:logits(model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True)))
        parity=call('long_physical_zeroB_compare',lambda:{'equal':bool(torch.equal(dense,adapted)),'maximum_error':float((dense-adapted).abs().max())})
        require(parity=={'equal':True,'maximum_error':0.},'actual dense-versus-zeroB parity exact');del dense,adapted,ids
        inventory=call('long_native_adapter_inventory',lambda:adapter_inventory(model,torch))
        require(len(inventory)==112 and sum(v['parameters'] for v in inventory)==1146880,'exact native FP32 qv112 profile')
        def adapter_hashes():return {name:tensor_hash(p) for name,p in model.named_parameters() if name.endswith(('.lora_A','.lora_B'))}
        rng_start=call('long_inference_start_RNG_snapshot',rng);adapter_start=call('long_adapter_bytes_before',adapter_hashes)
        common={'recipe_sha256':scope['recipe']['sha256'],'source_byte_maps_sha256':digest(pins_before),'model_source_map_sha256':digest(source_hashes),
            'encoder_proof_sha256':digest(encoding),'dense_start_rng':dense_rng,'inference_start_rng':rng_start,'initial_adapter_sha256':digest(adapter_start),'seed':101}
        start={'kind':'NATIVE_OBSERVED_LONG_START','identity_common':common,'owned_scope_identity':deepcopy(self.owner),
            'actual_pin_maps':pins_before,'encoder':encoding,'loader':loader,'compute':compute,'adapters':adapters,
            'actual_dense_zeroB_parity':parity,'zeroB':zero,'native_long_context_qualified':False}
        engine=InferenceComponent(torch,model,NoDecode(),call,guard,self.journal.evidence,
            max_attempted_forwards=1 if scope['role']=='bulk' else 32,max_input_positions=scope['dose']['maximum_forward_input_positions']-838,
            context_cap=scope['dose']['context_cap'],generation_cap=32,device='cuda:0')
        def finalize():
            engine.verify_unchanged();after_rng=rng();after_adapter=adapter_hashes()
            require(after_rng==rng_start and after_adapter==adapter_start,'actual RNG/all112 adapter bytes unchanged')
            z=call('long_zeroB_after',lambda:zero_lora_status(model,torch));require(z['all_B_zero'] is True,'zeroB unchanged')
            base=verify_frozen_base(torch,model,source_hashes,call,guard)
            require(base['verified_source_tensors']==310 and base['frozen_base_unchanged'] is True,'actual all310 frozen base contents')
            require(active['observations'] and all(value>0 for value in active['observations'].values()),'every FP32 activation site observed')
            actual_after=verified_files(scope,call,guard);require(actual_after==pins_before,'actual source/runtime/model/encoder bytes before/after')
            return {'kind':'NATIVE_OBSERVED_LONG_END','identity_common':common,'RNG_unchanged':True,'RNG_after':after_rng,
                'base310_unchanged':True,'base_postcheck':base,'zeroB_unchanged':True,'zeroB_after':z,
                'activation_observations':deepcopy(active['observations']),'actual_pin_maps_after':actual_after,'native_long_context_qualified':False}
        return {'component':engine,'attest':lambda:deepcopy(start),'finalize':finalize,'cleanup':self.cleanup}


def run(config_path,output):
    scope=contract.preflight(config_path)  # staging denial precedes all APIs
    from native_worker import Journal,verify_worker_ownership,preserve_terminal
    from inference_qual_worker import make_guard
    output=Path(output).absolute();require(output.resolve().parent==(PACKAGE/'artifacts').resolve() and output.is_dir() and not output.is_symlink(),'owned new leaf')
    owner=contract.native.read_json(output/'supervisor_ownership.json');ownership=verify_worker_ownership(owner,config_path)
    class DurableJournal(Journal):
        def emit(self,value):super().emit(value);os.fsync(self.stream.fileno())
    journal=DurableJournal(output);factory=NativeFactory(owner,journal);start=time.monotonic()
    guard=make_guard(output,owner,start,lambda:factory.torch);journal.guard=guard;journal.evidence('worker_ownership_verified',ownership)
    sequence=0
    def preserve(event):
        nonlocal sequence
        sequence+=1;path=output/('%04d_%s.json'%(sequence,event['event']));payload=event['payload'];raw=(canonical(payload)+'\n').encode()
        with path.open('xb') as stream:require(stream.write(raw)==len(raw),'complete long publication');stream.flush();os.fsync(stream.fileno())
        value={'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'payload_sha256':digest(payload),'durable':True}
        journal.emit({'event':'long_measurement_preserved','name':event['event'],'publication':value});return value
    def admission(request):
        guard('before_long_owned_factory');now=time.monotonic()
        return {'binding':request,'owned_supervision_verified':True,'model_load_and_constructor_in_scope':True,'observed_monotonic':now,'expires_monotonic':now+.2}
    result=core.run_scope(scope,factory,journal.call,guard,preserve,admission=admission)
    result.update({'config_sha256':scope['config_sha256'],'source_commit':scope['source_commit'],'wall_seconds':time.monotonic()-start,
        'operation_accounting':journal.snapshot(),'requires_independent_supervised_audit':True})
    result['sha256']=digest({k:v for k,v in result.items() if k!='sha256'});preserve_terminal(output,result,journal,factory.torch)
    return 0 if result['status']=='COMPLETE_NATIVE_LONG_SCOPE_PENDING_INDEPENDENT_AUDIT' else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();return run(args.config,args.output_dir)


if __name__=='__main__':raise SystemExit(main())
