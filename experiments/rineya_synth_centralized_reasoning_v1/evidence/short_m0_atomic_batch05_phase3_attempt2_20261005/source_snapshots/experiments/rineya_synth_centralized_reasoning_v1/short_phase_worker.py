"""Phase-observed caller of exact original M0 math; stage native entry denied."""
from pathlib import Path
import argparse,importlib,importlib.metadata,os,random,shutil,sys,time
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1] if HERE.name=='short_phase_collection_work' and HERE.parent.name=='artifacts' else HERE
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(HERE))
import short_phase_contract as contract
import short_phase_observation as observation
import short_collection_worker as original
from rs_contract import digest,require
GIB=1024**3
# These are exact imported implementations, not a second collection algorithm.
ForwardDose=original.ForwardDose
LocalGeneratedDecoder=original.LocalGeneratedDecoder
DurableRawCollection=original.DurableRawCollection
collect_loaded=original.collect_loaded

def pipeline(config,api,holder,journal,recorder,guard,report,*,collection_kwargs=None):
    """Injected CPU API or source-bound lazy native API; no added guards/queries."""
    def main(name,function,*,result=False):return observation.span(recorder,'main_'+name,function,result=result)
    guard('pre_import')
    require(main('disk_admission',api['disk_free'],result=True)>=4*GIB,'disk admission floor')
    torch=main('torch_import',api['torch_import']);holder[0]=torch
    config_class=main('Qwen3Config_symbol_import',api['Qwen3Config_symbol_import'])
    model_class=main('Qwen3ForCausalLM_symbol_import',api['Qwen3ForCausalLM_symbol_import'])
    versions=main('framework_version_read',api['framework_version_read'],result=True)
    require(versions=={key:config['runtime_versions'][key] for key in ('torch','transformers')},'framework runtime versions')
    growth=main('post_import_growth_observation',api['growth_observation'],result=True)
    require(growth['available_bytes']>=1.25*GIB+max(0,2*GIB-growth['worker_RSS_bytes']),'post-import full growth reserve')
    journal.call('cuda_context_init',torch.cuda.init)
    require(main('direct_CUDA_preload_mem_get_info',torch.cuda.mem_get_info,result=True)[0]>=4*GIB,'preload GPU admission')
    main('precision_and_thread_configuration',lambda:api['configure'](torch))
    main('Python_seed_101',lambda:api['Python_seed'](101))
    main('Torch_CPU_seed_101',lambda:torch.manual_seed(101))
    main('Torch_CUDA_seed_all_101',lambda:torch.cuda.manual_seed_all(101))
    main('CUDA_reset_peak_memory_stats',torch.cuda.reset_peak_memory_stats)
    model,loader,source_hashes=main('load_titan',lambda:api['load_titan'](torch,config_class,model_class,config,journal.call,guard))
    report['loader']=loader;journal.evidence('loader',loader)
    decoder=journal.call('generated_decoder_construct',lambda:api['decoder_construct'](config))
    dose=main('forward_dose_construct',lambda:ForwardDose(torch,config));report['forward_dose']=dose.snapshot()
    try:
        main('collect_loaded',lambda:collect_loaded(torch,model,decoder,config,source_hashes,journal,guard,report,dose,
            **(collection_kwargs or {})))
    finally:report['forward_dose']=dose.snapshot()
    main('final_config_preflight',api['final_preflight']);guard('completed')
    return torch

def lazy_api(holder,config_path):
    def symbol(module,name):return lambda:getattr(importlib.import_module(module),name)
    def versions():return {'torch':str(holder[0].__version__),'transformers':importlib.metadata.version('transformers')}
    def growth():
        import psutil
        return {'available_bytes':psutil.virtual_memory().available,'worker_RSS_bytes':psutil.Process().memory_info().rss}
    def configure(torch):
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    return {'disk_free':lambda:shutil.disk_usage(PACKAGE).free,'torch_import':lambda:importlib.import_module('torch'),
        'Qwen3Config_symbol_import':symbol('transformers.models.qwen3.configuration_qwen3','Qwen3Config'),
        'Qwen3ForCausalLM_symbol_import':symbol('transformers.models.qwen3.modeling_qwen3','Qwen3ForCausalLM'),
        'framework_version_read':versions,'growth_observation':growth,'configure':configure,'Python_seed':random.seed,
        'load_titan':lambda torch,c,m,config,call,guard:original.load_titan(torch,c,m,config['model_dir'],call,guard),
        'decoder_construct':LocalGeneratedDecoder,'final_preflight':lambda:contract.preflight(config_path)}

def run(config_path,output):
    contract.integrated()  # Before all paths/artifacts/framework/native APIs.
    output,config_path=Path(output).absolute(),Path(config_path).absolute()
    require(output.resolve().parent==(PACKAGE/'artifacts').resolve() and output.is_dir() and not output.is_symlink(),'owned new phase collection output')
    start=time.monotonic();owner=contract.native.read_json(output/'supervisor_ownership.json')
    ownership=original.verify_worker_ownership(owner,config_path)
    sink=observation.PhaseSink(output);recorder=observation.Recorder(sink);journal=observation.PhaseJournal(output,recorder)
    holder=[None];report={};status='SHORT_M0_RAW_COLLECTION_FAILED';error=terminal_error=None;config=None
    actual=original.make_guard(output,owner,start,
        lambda:None if holder[0] is None else observation.GuardTorchProxy(holder[0],recorder),
        available=original.worker_ram_probe.available_bytes)
    guard=observation.observed_guard(actual,recorder);journal.guard=guard
    try:
        journal.evidence('worker_ownership_verified',ownership)
        config=observation.span(recorder,'main_initial_config_preflight',lambda:contract.preflight(config_path))
        report.update(source_commit=config['source_commit'],config_sha256=config['config_sha256'],batch_selector=config['batch_selector'])
        for key,value in config['offline_environment'].items():os.environ[key]=value
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        pipeline(config,lazy_api(holder,config_path),holder,journal,recorder,guard,report)
        status='COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED'
    except BaseException as caught:
        error={'type':type(caught).__name__,'message':str(caught)}
        try:journal.emit({'event':'worker_failed','error':error})
        except BaseException as caught_io:report['journal_failure']={'type':type(caught_io).__name__,'message':str(caught_io)}
    finally:
        report.update(status=status,error=error,wall_seconds=time.monotonic()-start,operation_accounting=journal.snapshot(),
            phase_accounting_requires_final_receipt=observation.FINAL_NAME,scope=contract.PURPOSE,
            correctness_certified=False,retention_certified=False,scoring_authorized=False,production_training_authorized=False,
            language_qualification_established=False,long_context_qualification_established=False,protected_bank_access=False,
            paid_services=False,prior_correct_enrollment_performed=False,confirmation_eligible=False,quality_gain_established=False,
            optimizer_steps=0,backwards=0,requires_independent_supervised_audit=True)
        # Keep the unchanged helper's terminal-first ordering and exact existing
        # optional CUDA reads. This recorder remains open after Journal closes.
        try:
            observation.span(recorder,'main_preserve_terminal',lambda:original.preserve_terminal(output,report,journal,
                None if holder[0] is None else observation.TerminalTorchProxy(holder[0],recorder)),before_health=False)
        except BaseException as caught:terminal_error={'type':type(caught).__name__,'message':str(caught)}
        identity={'execution_scope':'OWNED_SHORT_M0_PHASE_OBSERVED_RAW_COLLECTION','purpose':contract.PURPOSE,
            'config_sha256':None if config is None else config['config_sha256'],
            'source_commit':None if config is None else config['source_commit'],'batch_selector':report.get('batch_selector')}
        try:final=observation.finish(output,recorder,sink,identity,primary_error=error,terminal_error=terminal_error)
        except BaseException:return 2
    return 0 if status=='COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED' and final['status']=='COMPLETE_PHASE_OBSERVATION' else 2

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();return run(args.config,args.output_dir)

if __name__=='__main__':raise SystemExit(main())
