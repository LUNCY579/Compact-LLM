"""Pinned one-prefix CPU contract. No native permission, model or probe API."""
from pathlib import Path
import hashlib,json,importlib.util
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
INPUT=PACKAGE/'evidence/exception_frame_helper_reviewed_cpu_20261005/root/ROOT_FAILURE_PREFIX_INPUT_001.json'
INPUT_SHA='195f1b3a658959ccbe517ce5aefaef1d0d2c298db0095ace08f744bd6488feb5'
INPUT_BODY='a22c5b27225fc75ae697a5f06a96da18c1cb995a7bf7c6d03eb05f0cb6f06ec9'
PREFIX_SHA='e806d781c2c21c15180b6ec1bc714ed5acc7104948c2859ec6b411761a4a19d1'
HELPER=PACKAGE/'evidence/exception_frame_helper_reviewed_cpu_20261005/exception_frames.py'
HELPER_SHA='253caa71942c300c225abef9f33d09f2ac28c4e6f169570ecbea09b926817d0b'
GIB=1024**3;CPU_SCOPE='CPU_INJECTED_PREFIX_INPUT_ONLY'
LIMITS={'maximum_markers':128,'maximum_marker_bytes':65536,'maximum_ledger_bytes':8388608,'maximum_worker_json_bytes':16*1024**2}
OPS=('initial_config_preflight','worker_ownership_verify','publication_ownership_verified','disk_admission','torch_import','torch_version_read','post_import_growth_observation',
 'cuda_context_init','pre_input_direct_CUDA_mem_get_info','original_precision_thread_configuration','Python_seed_101','Torch_CPU_seed_101',
 'Torch_CUDA_seed_all_101','CUDA_reset_peak_memory_stats','input_tensor_create','mask_ones_like_create','input_is_tensor','input_shape_matches',
 'input_dtype_long','input_device_cuda0','input_exact_tolist','mask_is_tensor','mask_shape_matches','mask_dtype_long','mask_device_cuda0','mask_exact_tolist',
 'input_reference_release','mask_reference_release','final_config_preflight','terminal_cuda_is_initialized','terminal_cuda_max_memory_allocated',
 'terminal_cuda_max_memory_reserved','terminal_cuda_mem_get_info','publication_input_check_summary','publication_worker_result')
GUARDS=('pre_import','after_import','after_growth','before_context_init','after_context_init','before_input_construction','after_input_validation','complete')
def require(v,m):
    if not v:raise ValueError(m)
def canonical(v):return json.dumps(v,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode('ascii')).hexdigest()
def pairs(items):
    v={}
    for k,x in items:require(k not in v,'duplicate input JSON key');v[k]=x
    return v
def load_input():
    raw=INPUT.read_bytes();require(len(raw)<=16384 and hashlib.sha256(raw).hexdigest()==INPUT_SHA,'exact durable registered exposed input bytes')
    v=json.loads(raw,object_pairs_hook=pairs);require(v['sha256']==INPUT_BODY==digest({k:x for k,x in v.items() if k!='sha256'}),'exact input body')
    ids=v['prefix_ids'];require(type(ids)is list and len(ids)==422 and all(type(x)is int and 0<=x<151936 for x in ids)
        and v['attention_mask']==[1]*422 and digest(ids)==PREFIX_SHA,'exact422 prefix/mask')
    require(v['native_launch_authorized'] is v['M0_population_membership_allowed'] is v['training_or_grading_allowed'] is False,'all input authority closed')
    return v
def frame_helper():
    raw=HELPER.read_bytes();require(hashlib.sha256(raw).hexdigest()==HELPER_SHA,'exact durable accepted frame helper')
    spec=importlib.util.spec_from_file_location('accepted_prefix_exception_frames',HELPER);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def dose():
    require(len(OPS)==35 and len(GUARDS)==8,'code-derived operation/guard population')
    return {'prefix_tokens':422,'logical_tensor_bytes':2*422*8,'success_operations':len(OPS),'guard_sites':len(GUARDS),
        'maximum_guard_queries':14,'optional_first_error_frame_capture':1,'maximum_phases':len(OPS)+len(GUARDS)+14+1,
        'maximum_markers':2*(len(OPS)+len(GUARDS)+14+1),'model_forwards':0,'model_positions':0,'backwards':0,'optimizer_steps':0,'decodes':0,'grades':0}
def native_entry(*args,**kwargs):raise ValueError('staged input diagnostic cannot authorize native paths, workers or probes')
