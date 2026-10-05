"""Ignored stage-only auditor, adapted from actual batch01 independent CPU audit.

No native/model/framework/GPU invocation or scoring. Heavy audit/optional evidence
write require the root's explicit terminal+preservation signal and independently
provided exact config-file and preservation-body SHA256 pins. The output is raw
collection mechanics only. Source/config/native launch gates are never edited.
"""
from pathlib import Path
from datetime import datetime,timezone
import collections,json,math,sys,os,importlib.metadata,importlib.util
HERE=Path(__file__).resolve().parent
package=HERE.parents[2]
root=package.parents[1]
sys.path.insert(0,str(package));sys.path.insert(0,str(HERE.parent))
import short_collection_contract as contract
from inference_component import validate_row,ROW_FIELDS,EOS_IDS,VERSION
from rs_contract import digest,require
SIGNAL="ROOT_TERMINAL_AND_PRESERVED"
CONTRACT_VERSION="synth-short-M0-physical-RAM-coherent-monitor-staged-2"
BATCH01_AUDIT_FILE_SHA256="0972d328fcd3fbcf7eadffadb38149448e237215e5bb5cd8c0e454b75066f44d"
BATCH01_AUDIT_BODY_SHA256="5246ff75489bacf627069034ec9f433941edbc7e05a167b591d1c92974ef5782"
ALLOWED_OPERATIONS=("checkpoint_load","cuda_context_init","generated_decoder_construct","inference_forward","install_activation_hooks","install_fp32_compute","install_qv_lora","model_construct","tensor_copy","verify_frozen_base","zero_lora_query_after","zero_lora_query_before","zero_parity_adapter_forward","zero_parity_base_forward")
PROFILE_ID="SHORT_M0_PHYSICAL_RAM_COHERENT_EXTERNAL_CHECK_V2"
REGISTRATION_ID="SYNTH_SHORT_RAM2_INDEPENDENT_AUDIT_INVENTORY_001"
SECTIONS={"source":"source_pins","runtime":"runtime_pins","decoder":"decoder_runtime_pins","data":"data_pins","model":"model_pins"}


def _registered_admission(scope,sup,registration,expected_profile_body_sha256,current_profile,current_sources,check):
 """Pure CPU admission against externally trusted frozen maps, never len-only."""
 check(registration.get("id")==REGISTRATION_ID and registration.get("schema_version")==1,"known independently frozen inventory registration")
 check(registration.get("sha256")==digest({k:v for k,v in registration.items() if k!="sha256"}),"inventory registration body checksum")
 profile=scope.get("short_RAM_monitor_profile")
 check(type(profile) is dict and profile.get("id")==PROFILE_ID and profile==current_profile
       and digest(profile)==registration.get("profile_body_sha256")==expected_profile_body_sha256,"exact trusted new-profile body/helper/callback/CPU policy")
 check(scope.get("contract")==CONTRACT_VERSION and registration.get("contract")==CONTRACT_VERSION,"new-only auditor rejects old/unknown contract")
 check(registration.get("source_commit")==sup.get("preflight",{}).get("source_commit"),"inventory frozen to independently selected source commit")
 inventories=registration.get("inventories");counts=registration.get("counts")
 check(type(inventories) is dict and set(inventories)==set(SECTIONS) and type(counts) is dict
       and set(counts)==set(SECTIONS) and all(type(value) is int for value in counts.values()),"exact declared inventory sections/counts")
 check(counts.get("runtime")==65 and counts.get("decoder")==17 and counts.get("data")==5 and counts.get("model")==7
       and type(counts.get("source")) is int and counts["source"]>0,"registered runtime65/decoder17/data5/model7 and frozen source count")
 for section,key in SECTIONS.items():
  expected=inventories[section]
  check(type(expected) is dict and len(expected)==counts[section] and all(type(path) is str and _hash(pin) for path,pin in expected.items()),"complete registered inventory "+section)
  check(scope.get(key)==expected,"every frozen path/hash retained "+section)
 for path,pin in current_sources.items():check(inventories["source"].get(path)==pin,"reviewed successor/helper source closure")
 extension=profile.get("runtime_extension_pins")
 check(type(extension) is dict and len(extension)==1 and all(Path(path).name=="ast.py" for path in extension)
       and all(inventories["runtime"].get(path)==pin for path,pin in extension.items()),"AST belongs to executing runtime65 inventory")
 core={k:v for k,v in inventories["runtime"].items() if k not in extension}
 check(len(core)==64 and digest(core)==registration.get("qualified_runtime64_sha256"),"unchanged independently frozen qualified runtime64 core")
 diagnostic=sup.get("short_RAM_monitor_diagnostic")
 check(type(diagnostic) is dict and diagnostic=={"path":"SHORT_RAM_ABORT_DIAGNOSTIC.json","attempted":False,
       "persisted":False,"write_error":None,"original_abort_reason_preserved":None},"successful terminal cannot carry abort/diagnostic contradiction")
 return counts

def _hash(value):
 return type(value) is str and len(value)==64 and all(ch in "0123456789abcdef" for ch in value)

def _early_terminal_checks(scope,worker,sup,pres,selector,check):
 check(scope.get("contract")==CONTRACT_VERSION and scope.get("authorization")==contract.AUTHORIZATION
       and scope.get("profile")=="BF16_RESIDENT_FP32_ACTIVATIONS_QV8_V1","only new registered raw collector contract")
 check(scope.get("batch_selector")=={"source_kind":selector[0],"ordinal":selector[1]},"exact independently selected frozen batch")
 check(sup.get("status")=="SUPERVISED_WORKER_EXITED_ZERO" and sup.get("returncode")==0
       and sup.get("partial_outputs_frozen") is True,"only frozen successful terminal")
 check(worker.get("status")=="COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED" and worker.get("error") is None,"only complete raw worker")
 rows=scope.get("rows");outputs=worker.get("collection",{}).get("rows")
 check(type(rows) is list and len(rows)==64 and type(outputs) is list and len(outputs)==64
       and [row.get("row_id") for row in rows]==[row.get("row_id") for row in outputs]
       and len({row.get("row_id") for row in rows})==64,"no partial/reordered/duplicate64 terminal")
 check(scope.get("source_packet_sha256")==digest(rows)==worker["collection"].get("source_packet_sha256"),"whole exact packet")
 check(all(set(row)==ROW_FIELDS for row in rows),"prompt-only rows; no target/oracle input fields")
 check(pres.get("source_commit")==worker.get("source_commit")==sup.get("preflight",{}).get("source_commit"),"preserved worker/supervisor source commit")

def audit_terminal(config_path, raw_directory, preserved_directory, *, expected_selector,
                   expected_config_file_sha256, expected_preservation_body_sha256,
                   inventory_registration_path=None, expected_inventory_registration_file_sha256=None,
                   expected_profile_body_sha256=None,
                   terminal_preservation_signal=None, output_path=None):
 """Verify one independently named64-row terminal scope; optional exclusive JSON."""
 require(terminal_preservation_signal==SIGNAL,"root terminal and preservation signal required before heavy reads")
 require(HERE.name=="audit_work" and HERE.parent.name=="short_ram_successor_work" and HERE.parent.parent.name=="artifacts","stage-only new-profile auditor")
 require(_hash(expected_config_file_sha256) and _hash(expected_preservation_body_sha256),"independent config/preservation SHA256 handshake")
 require(_hash(expected_inventory_registration_file_sha256) and _hash(expected_profile_body_sha256),"trusted independent inventory/profile handshake before heavy reads")
 # Fixture admission imports staged metadata; terminal audit must use the actual
 # reviewed integrated root contract, never a staged build_scope/source roster.
 spec=importlib.util.spec_from_file_location("short_RAM2_audit_integrated_contract",package/'short_collection_contract.py')
 contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
 require(contract.VERSION==CONTRACT_VERSION and contract.HERE.resolve()==package.resolve() and not contract.STAGED,
         "reviewed new root contract integration required before terminal/model closure")
 require(type(expected_selector) is tuple and len(expected_selector)==2
         and expected_selector[0] in ("atomic_RS_DEV","task_guard_DEV")
         and type(expected_selector[1]) is int and 1<=expected_selector[1]<=(6 if expected_selector[0]=="atomic_RS_DEV" else 12),"one preregistered frozen batch selector")
 config_path=Path(config_path).absolute();leaf=Path(raw_directory).absolute();saved=Path(preserved_directory).absolute()
 target=None if output_path is None else Path(output_path).absolute()
 require(config_path.resolve().parent==(package/"configs").resolve() and config_path.is_file() and not config_path.is_symlink(),"owned frozen config")
 require(leaf.resolve().parent==(package/"artifacts").resolve() and leaf.is_dir() and not leaf.is_symlink(),"owned completed raw leaf")
 require(saved.resolve().parent==(package/"evidence").resolve() and saved.is_dir() and not saved.is_symlink(),"owned preserved evidence leaf")
 if target is not None:
  require(target.resolve().parent==(package/"evidence").resolve() and target.name.startswith(leaf.name+"_independent_audit_")
          and target.suffix==".json" and not target.exists() and not target.is_symlink(),"new owned write-once audit receipt")
 checks=0
 def check(value,message):
  nonlocal checks
  checks+=1
  if not value:raise ValueError(message)
 def info(path):return {'bytes':path.stat().st_size,'sha256':contract.native.filehash(path)}
 def read(path,limit=16*1024**2):return contract.native.read_json(path,limit)
 def jsonl(path):
  result=[]
  with path.open('rb') as stream:
   for raw in stream:
    check(len(raw)<=65536 and raw.endswith(b'\n'),'complete bounded JSONL')
    result.append(json.loads(raw.decode(),object_pairs_hook=contract.native._pairs,parse_constant=contract.native._bad_constant))
  return result
 check(target is None or not target.exists(),'write-once audit')
 scope=read(config_path);worker=read(leaf/'worker_result.json');sup=read(leaf/'SUPERVISOR_RECEIPT.json');pres=read(saved/'preservation_receipt.json');lease=Path(sup['lease_path'])
 check(pres['sha256']==digest({k:v for k,v in pres.items() if k!='sha256'})==expected_preservation_body_sha256,'preservation canonical checksum')
 check(info(config_path)['sha256']==expected_config_file_sha256,'independent config pin before heavy closure')
 check(pres['raw_directory']==str(leaf) and pres['preserved_directory']==str(saved) and sup['output_dir']==str(leaf),'exact original/preserved/terminal leaf crosslinks')
 check(worker['batch_selector']==scope['batch_selector'],'worker exact fixed selector')
 _early_terminal_checks(scope,worker,sup,pres,expected_selector,check)
 registration_path=Path(inventory_registration_path).absolute()
 check(registration_path.resolve().parent==(package/'configs').resolve() and registration_path.is_file()
       and not registration_path.is_symlink(),"owned prospective write-once inventory registration")
 check(info(registration_path)['sha256']==expected_inventory_registration_file_sha256,"independently supplied inventory file pin")
 registration=read(registration_path)
 check(registration.get('root_path')==str(root.resolve()) and registration.get('package_path')==str(package.resolve())
       and registration.get('config_path')==str(config_path.resolve())
       and registration.get('config_file_sha256')==expected_config_file_sha256,
       "registration bound to exact root/package/scope-file path and bytes")
 current_profile=contract.successor_profile()
 current_sources={str((package/name).resolve()):contract.native.filehash(contract.HERE/name) for name in contract.SOURCE_NAMES}
 current_sources.update({str((package/name).resolve()):pin for name,pin in contract.HELPER_PINS.items()})
 counts=_registered_admission(scope,sup,registration,expected_profile_body_sha256,current_profile,current_sources,check)
 capture_n=worker['collection_costs']['attempted_forwards'];capture_p=worker['collection_costs']['input_positions']['attempted'];parity_p=2*scope['rows'][0]['prompt_tokens']
 check(type(capture_n) is int and 64<=capture_n<=2048 and type(capture_p) is int and 0<capture_p<=scope['dose']['maximum_capture_input_positions'],'bounded terminal dose')
 physical_n=capture_n+2;physical_p=capture_p+parity_p;api_n=capture_n+322
 raw_names=sorted(path.name for path in leaf.iterdir() if path.is_file());check(len(raw_names)==capture_n+77 and set(raw_names)==set(pres['raw_files']),'exact raw inventory')
 check('SHORT_RAM_ABORT_DIAGNOSTIC.json' not in raw_names,'successful completed scope has no abort diagnostic raw file')
 def maps():
  return {'original_raw':{name:info(leaf/name) for name in raw_names},'preserved_raw':{name:info(saved/name) for name in raw_names},'source':{path:info(Path(path)) for path in scope['source_pins']},'runtime':{path:info(Path(path)) for path in scope['runtime_pins']},'decoder':{path:info(Path(path)) for path in scope['decoder_runtime_pins']},'data':{path:info(Path(path)) for path in scope['data_pins']},'model':{name:info(Path(scope['model_dir'])/name) for name in scope['model_pins']},'source_snapshots':{name:info(saved/'source_snapshots'/name) for name in pres['source_snapshots']},'config':info(config_path),'preserved_config':info(saved/'config_snapshot.json'),'preservation_receipt':info(saved/'preservation_receipt.json'),'lease':{path.name:info(path) for path in lease.glob('*.json')}}
 before=maps()
 check(before['original_raw']==before['preserved_raw']==pres['raw_files'],'all original/preserved raw byte pairs')
 check(before['config']==before['preserved_config']==pres['config_snapshot'],'config byte closure')
 check(before['config']['sha256']==expected_config_file_sha256,'exact config file')
 check(scope['sha256']==digest({k:v for k,v in scope.items() if k!='sha256'}),'config body')
 for section,scope_key,count in ((section,key,counts[section]) for section,key in SECTIONS.items()):
  check(len(before[section])==count,'pin count '+section)
  for path,item in before[section].items():check(item['sha256']==scope[scope_key][path],'current '+section+' byte pin')
 check(before['source_snapshots']==pres['source_snapshots'] and len(before['source_snapshots'])==counts['source'],'complete independently registered source snapshots')
 for path,item in before['source'].items():check(before['source_snapshots'][Path(path).relative_to(root).as_posix()]==item,'snapshot joins original source')
 check(scope['runtime_pins']==contract.merge_runtime_inventory(contract.native.runtime_files(),current_profile)
       and scope['decoder_runtime_pins']==contract._tokenizer_runtime(),'actual interpreter65/decoder closure')
 expected=contract.build_scope(*expected_selector);check(scope=={**expected,'sha256':digest(expected)},'full frozen current batch contract')
 check(sup['preflight']=={**scope,'config_path':str(config_path),'config_sha256':before['config']['sha256'],'worker_path':str(package/'short_collection_worker.py'),'source_commit':worker['source_commit'],'bindings':{'source':scope['source_pins'],'runtime':scope['runtime_pins'],'decoder':scope['decoder_runtime_pins'],'data':scope['data_pins'],'model':scope['model_pins']}},'supervisor exact preflight')
 check(worker['config_sha256']==sup['config_sha256']==before['config']['sha256'] and worker['source_commit']==pres['source_commit']==sup['preflight']['source_commit'],'source/config identity')
 check(sup['status']=='SUPERVISED_WORKER_EXITED_ZERO' and sup['returncode']==0 and sup['partial_outputs_frozen'] is True,'successful supervised terminal')
 check(worker['status']=='COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED' and worker['error'] is None,'raw worker terminal')
 check(sup['elapsed_seconds']<600 and worker['wall_seconds']<600 and scope['resources']==contract.native.RESOURCE_BOUNDS and scope['tolerances']==contract.native.TOLERANCES,'unchanged resources/numerics/deadline')
 for item in sup['partial_outputs']:check(before['original_raw'][item['path']]=={'bytes':item['bytes'],'sha256':item['sha256']},'frozen supervisor inventory')
 check(len(sup['partial_outputs'])==capture_n+76,'supervisor receipt separately bound')
 events=jsonl(leaf/'events.jsonl');heartbeats=jsonl(leaf/'supervisor_heartbeat.jsonl');starts={};terminal={};operations={};measurements=[]
 for event in events:
  kind=event['event']
  if kind=='measurement_preserved':
   check(before['original_raw'][event['path']]['sha256']==event['sha256'] and event['qualification_established'] is False,'fragment byte crosslink');measurements.append(event);continue
  check(kind in ('attempt','returned','failed'),'closed event type')
  ordinal=event['id'];name=event['operation'];cost=operations.setdefault(name,dict.fromkeys(('attempted','returned','failed','pending'),0))
  if kind=='attempt':check(ordinal not in starts,'unique API attempt');starts[ordinal]=event;cost['attempted']+=1
  else:check(ordinal in starts and ordinal not in terminal and starts[ordinal]['operation']==name,'matching API terminal');terminal[ordinal]=event;cost[kind]+=1
 for cost in operations.values():cost['pending']=cost['attempted']-cost['returned']-cost['failed'];check(cost['attempted']==cost['returned'] and cost['failed']==cost['pending']==0,'complete actual call conservation')
 check(set(starts)==set(range(1,api_n+1)) and set(terminal)==set(starts),'api_n actual calls')
 check(worker['operation_accounting']['operations']==operations and worker['operation_accounting']['last_call']==api_n,'worker API ledger matches events')
 check(worker['operation_accounting']['scope']=='registered_API_boundaries;internal_ATen_operations_not_individually_metered','truthful API scope')
 check(set(operations)==set(ALLOWED_OPERATIONS),'first collector exact API inventory')
 check(all(not any(term in name.lower() for term in ('backward','optimizer','gradient','checkpoint_restore','warmstart')) for name in operations),'no training/resume operations')
 for name,count in (('inference_forward',capture_n),('zero_parity_base_forward',1),('zero_parity_adapter_forward',1),('tensor_copy',310),('verify_frozen_base',1)):
  check(operations[name]=={'attempted':count,'returned':count,'failed':0,'pending':0},'exact operation '+name)
 check(importlib.metadata.version('tokenizers')=='0.22.2','CPU decoder version')
 from tokenizers import Tokenizer
 decoder=Tokenizer.from_file(str(Path(scope['model_dir'])/'tokenizer.json'))
 check(decoder.token_to_id('<|im_end|>')==151645 and decoder.token_to_id('<|endoftext|>')==151643,'exact EOS decoder IDs')
 rows=scope['rows'];outputs=worker['collection']['rows'];check(len(rows)==len(outputs)==64 and len({row['row_id'] for row in rows})==64,'all64 fixed rows')
 check(digest(rows)==scope['source_packet_sha256']==worker['collection']['source_packet_sha256'],'source packet')
 capture_calls=capture_positions=0;finishes=collections.Counter();row_manifest=[];partial_names=[];final_names=[]
 def cost_snapshot(calls,positions):return {'attempted_forwards':calls,'input_positions':{'attempted':positions,'returned':positions,'failed':0,'pending':0},'maximum_attempted_forwards':2048,'maximum_input_positions':scope['dose']['maximum_capture_input_positions'],'operations':{'inference_forward':{'attempted':calls,'returned':calls,'failed':0,'pending':0}}}
 flags=('oracle_inputs_read','capability_correctness_scored','confirmation_eligible','native_launch_authorized','native_qualification_established','padding','packing','prompt_truncated')
 for ordinal,(row,output) in enumerate(zip(rows,outputs),1):
  check(set(row)==ROW_FIELDS and validate_row(row)==row,'whole prompt-only row')
  check(output['row_id']==row['row_id'] and output['prompt_sha256']==row['prompt_sha256'] and output['prompt_ids_sha256']==digest(row['prompt_ids']) and output['input_row_sha256']==digest(row),'exact row/prompt identity')
  check(output['source_packet_sha256']==scope['source_packet_sha256'] and output['prompt_tokens']==row['prompt_tokens'],'frozen row assignment')
  check(output['status']=='RAW_RESPONSE_COLLECTED_UNSCORED' and output['implementation']==VERSION and output['scope']=='UNQUALIFIED_NATIVE_IMPLEMENTATION','raw-only component state')
  check(output['context_cap']==512 and output['maximum_generated_tokens']==32 and output['decoding']=='greedy_no_cache_no_grad' and all(output[key] is False for key in flags),'row closed flags/context')
  ids=output['raw_generated_ids'];trace=output['trace'];lps=output['selected_logprobs'];check(1<=len(ids)<=32 and len(ids)==len(trace)==len(lps),'token/trace/LP count')
  check(all(type(token) is int and 0<=token<151936 for token in ids) and not any(token in EOS_IDS for token in ids[:-1]),'typed IDs/first EOS stop')
  eos=ids[-1] in EOS_IDS;check(output['finish_reason']==('eos' if eos else 'generation_cap') and output['truncated'] is (not eos) and output['eos_token_id']==(ids[-1] if eos else None) and (eos or len(ids)==32),'exact finish semantics')
  check(output['decoded_response']==decoder.decode(list(ids),skip_special_tokens=True),'generated-only CPU decode exact')
  for step,(token,lp,entry) in enumerate(zip(ids,lps,trace),1):
   check(set(entry)=={'token','logprob','prefix_length'} and entry['token']==token and entry['logprob']==lp and entry['prefix_length']==row['prompt_tokens']+step-1 and math.isfinite(lp) and lp<=0,'finite LP/exact growing prefix')
   capture_calls+=1;capture_positions+=entry['prefix_length']
   name='stage_raw_row%04d_token%02d.json'%(ordinal,step);partial_names.append(name);partial=read(leaf/name)
   for key in ('row_id','prompt_sha256','prompt_ids_sha256','input_row_sha256','source_packet_sha256','prompt_tokens','context_cap','maximum_generated_tokens','decoding','implementation','scope'):check(partial[key]==output[key],'partial identity')
   check(partial['status']=='RAW_TOKEN_PARTIAL_UNSCORED' and partial['row_ordinal']==ordinal and partial['step_ordinal']==step and partial['raw_generated_ids']==ids[:step] and partial['selected_logprobs']==lps[:step] and partial['trace']==trace[:step] and partial['costs']==cost_snapshot(capture_calls,capture_positions),'durable partial prefix/cost')
   partial_eos=token in EOS_IDS
   check(partial['finish_reason']==('eos' if partial_eos else ('in_progress' if step<32 else 'generation_cap')) and partial['truncated'] is (not partial_eos and step==32) and partial['eos_token_id']==(token if partial_eos else None) and all(partial[key] is False for key in flags),'partial finish/flags')
  check(output['costs']==cost_snapshot(capture_calls,capture_positions),'cumulative final-row costs')
  check(output['parameter_preservation']==worker['parameter_preservation'],'perrow parameter invariant')
  name='stage_inference_row_%04d.json'%ordinal;final_names.append(name);check(read(leaf/name)==output,'durable final row matches terminal')
  finishes[output['finish_reason']]+=1
  row_manifest.append({'ordinal':ordinal,'row_id':row['row_id'],'input_row_sha256':digest(row),'raw_output_sha256':digest(output),'generated_tokens':len(ids),'finish_reason':output['finish_reason']})
 check(capture_calls==capture_n and capture_positions==capture_p,'exact capture totals')
 check(len(list(leaf.glob('stage_raw_row*.json')))==len(partial_names)==capture_n and len(list(leaf.glob('stage_inference_row_*.json')))==64,'no missing/extra raw rows/tokens')
 check(worker['collection_costs']==worker['collection']['costs']==cost_snapshot(capture_calls,capture_positions),'component terminal costs')
 check(worker['last_partial_receipt']==outputs[-1],'last immutable receipt')
 expected_dose={'attempted_calls':physical_n,'input_positions':physical_p,'maximum_forward_calls':2050,'maximum_forward_input_positions':scope['dose']['maximum_forward_input_positions'],'operations':{'inference_forward':{'attempted_calls':capture_n,'input_positions':capture_p},'zero_parity_base_forward':{'attempted_calls':1,'input_positions':parity_p//2},'zero_parity_adapter_forward':{'attempted_calls':1,'input_positions':parity_p//2}}}
 check(worker['forward_dose']==expected_dose,'actual physical prehook dose')
 check(worker['zero_adapter_parity']['equal'] is True and worker['zero_adapter_parity']['maximum_error']==0 and worker['zero_adapter_parity']['capability_scored'] is False,'fresh physical zeroB parity')
 for key in ('zero_adapter_before','zero_adapter_after'):check(worker[key]['all_B_zero'] is True and worker[key]['qv_site_count']==56 and worker[key]['trainable_tensors']==112,'unchanged zeroB')
 check(worker['base_postcheck']['verified_source_tensors']==310 and worker['base_postcheck']['frozen_base_unchanged'] is True and worker['base_postcheck']['base_dtype']=='bfloat16' and worker['base_postcheck']['unique_base_parameters']==596049920 and worker['base_postcheck']['embedding_head_tie'] is True,'all310 independent native base bytes')
 check(worker['parameter_preservation']=={'independent_parameter_byte_verification_required':True,'mutation_counters_preserved':True,'objects_pointers_dtypes_flags_gradients_preserved':True,'parameter_aliases':423},'parameter object/pointer/flags/gradients')
 check(len(worker['activation_observations'])==198 and set(worker['activation_observations'].values())=={physical_n},'all198 F32 activation sites observed')
 check(worker['compute']['base_storage_dtype']=='bfloat16' and worker['compute']['activation_dtype']=='float32' and worker['compute']['base_parameter_objects_and_pointers_preserved'] is True and worker['compute']['linear_site_count']==197,'qualified BF16/F32 compute profile')
 check(worker['adapters']['rank']==8 and worker['adapters']['alpha']==16 and worker['adapters']['dropout']==0 and worker['adapters']['trainable_parameters']==1146880 and worker['adapters']['trainable_tensors']==112 and worker['adapters']['qv_site_count']==56,'exact zero adapter inventory')
 check({name:value['sha256'] for name,value in worker['loader']['asset_hashes'].items()}==scope['model_pins'] and worker['loader']['base_frozen'] is True and worker['loader']['FP32_base_copy_allocated'] is False,'exact independent model bytes/load receipt')
 check(read(leaf/'stage_loader.json')==worker['loader'] and read(leaf/'stage_zero_adapter_parity.json')==worker['zero_adapter_parity'],'loader/parity fragments')
 infer_audit=read(contract.INFERENCE_AUDIT);check(infer_audit['sha256']==digest({k:v for k,v in infer_audit.items() if k!='sha256'}) and infer_audit['status']=='PASS_SHORT_SCOPED_NATIVE_INFERENCE_MECHANICS' and infer_audit['short_scoped_native_inference_mechanics_verified'] is True,'bound prior numerical qualification')
 check(infer_audit['numerics']['all_bulk_argmax_equal'] is True and infer_audit['numerics']['two_exact_repeats_each'] is True and infer_audit['numerics']['maximum_selected_logprob_error']<=.03125 and infer_audit['numerics']['largest_row_mean_selected_logprob_error']<=.01,'unchanged prior numerical thresholds')
 resources=read(leaf/'RESOURCE_OBSERVATIONS.json');cpu=[row for row in resources if row.get('kind')=='CPU'];gpu=[row for row in resources if row.get('kind')=='GPU'];check(len(cpu)>0 and len(gpu)>0,'resource counts')
 for row in cpu:
  check(row['sample_started_monotonic']<=row['sample_completed_monotonic'] and sum(item['rss_bytes'] for item in row['owned_process_samples'])==row['worker_tree_rss_bytes'],'CPU sample/RSS accounting')
  check(row['host_available_bytes']>=1.25*1024**3 and row['worker_tree_rss_bytes']<=2*1024**3 and row['disk_free_bytes']>=4*1024**3,'recorded CPU amount gates')
 for row in gpu:check(row['gpu_sample_started_monotonic']<=row['gpu_sample_completed_monotonic'] and row['gpu_free_bytes']>=.5*1024**3 and row['gpu_uuid']==sup['dispatch_observation']['gpu_uuid'],'GPU gate/identity')
 cpu_window=max(row['sample_completed_monotonic']-row['sample_started_monotonic'] for row in cpu);cpu_gap=max(b['sample_started_monotonic']-a['sample_started_monotonic'] for a,b in zip(cpu,cpu[1:]));cpu_publish_age=max(b['sample_completed_monotonic']-a['sample_started_monotonic'] for a,b in zip(cpu,cpu[1:]));gpu_window=max(row['gpu_sample_completed_monotonic']-row['gpu_sample_started_monotonic'] for row in gpu);gpu_gap=max(b['gpu_sample_started_monotonic']-a['gpu_sample_started_monotonic'] for a,b in zip(gpu,gpu[1:]));gpu_publish_age=max(b['gpu_sample_completed_monotonic']-a['gpu_sample_started_monotonic'] for a,b in zip(gpu,gpu[1:]))
 check(cpu_publish_age<=.2 and gpu_publish_age<=2,'recorded preceding-sample ages through next publication')
 reservation=read(lease/'reservation.json');resume=read(lease/'resume.json');owned=read(lease/'worker.json')
 check(all(row['supervisor_pid']==reservation['supervisor']['pid'] for row in heartbeats) and all(b['unix_seconds']>=a['unix_seconds'] for a,b in zip(heartbeats,heartbeats[1:])),'heartbeat identity/order')
 hb_gap=max(b['unix_seconds']-a['unix_seconds'] for a,b in zip(heartbeats,heartbeats[1:]));check(hb_gap<=2,'heartbeat observed gap')
 for observation in sup['admission_observations']+[sup['dispatch_observation']]:check(observation['gpu_free_bytes']>=4*1024**3 and observation['host_available_bytes']>=3.3125*1024**3 and observation['disk_free_bytes']>=4*1024**3,'admission amount gates')
 check(owned==sup['worker_identity']==resume['worker_identity'] and owned['assigned_before_resume'] is True and owned['job_kill_on_close'] is True and owned['breakaway_allowed'] is False and resume['external_monitor_ready_before_resume'] is True and resume['resumed_monotonic']==sup['worker_resume_monotonic'],'owned identity/monitor-before-resume')
 check(before['lease']['terminal.json']==before['original_raw']['SUPERVISOR_RECEIPT.json'] and reservation['config_sha256']==before['config']['sha256'] and reservation['output_dir']==str(leaf),'lease terminal/config closure')
 check(sup['job_closed_kill_on_close'] is True and sup['owned_worker_identity_currently_alive'] is False and sup['sampler_threads_terminal_verified'] is True and sup['gpu_telemetry_cleanup_verified'] is True and sup['unjoined_sampler_threads']==[],'terminal Job/samplers/NVML')
 gpu_final=read(leaf/'worker_GPU_final.json');check(gpu_final['CUDA_initialized'] is True and gpu_final['free_bytes_final']>=.5*1024**3,'direct terminal GPU diagnostic')
 check(worker['backwards']==worker['optimizer_steps']==0 and all(worker[key] is False for key in ('correctness_certified','retention_certified','scoring_authorized','production_training_authorized','language_qualification_established','long_context_qualification_established','protected_bank_access','paid_services','prior_correct_enrollment_performed','confirmation_eligible','quality_gain_established')),'all quality/training/enrollment flags closed')
 check(scope['authorization']==contract.AUTHORIZATION and scope['scoring_authorized'] is False and scope['training_authorized'] is False and scope['prior_correct_enrollment_performed'] is False,'raw-only owned authorization')
 after=maps();check(before==after,'all original/preserved/pin bytes stable')
 check(not any(name in sys.modules for name in ('torch','transformers')),'no framework/model execution in audit')
 def compact(values):return {key:({'entries':len(value),'manifest_sha256':digest(value)} if key in ('original_raw','preserved_raw','source','runtime','decoder','data','model','source_snapshots','lease') else value) for key,value in values.items()}
 audit={'schema_version':1,'id':scope['id']+'-INDEPENDENT-AUDIT','status':'PASS_SCOPED_SHORT_M0_RAW_COLLECTION_MECHANICS_UNSCORED','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'checks_passed':checks,'source_commit':worker['source_commit'],'attempt_path':str(leaf),'preserved_path':str(saved),'config_path':str(config_path),'config_file_sha256':before['config']['sha256'],'config_body_sha256':scope['sha256'],'batch_selector':scope['batch_selector'],'source_packet_sha256':scope['source_packet_sha256'],'source_row_order_sha256':digest([row['row_id'] for row in rows]),'ordered_raw_row_receipts_sha256':digest(row_manifest),'rows':64,'per_token_fragments_verified':capture_n,'finish_counts':dict(finishes),'all_exact_row_prompt_token_trace_prefix_finish_decode_and_cost_joins':True,'supervisor_status':sup['status'],'worker_status':worker['status'],'wall_seconds':{'whole_scope':sup['elapsed_seconds'],'worker':worker['wall_seconds'],'limit':600},'accounting':{'physical_forwards':physical_n,'physical_input_positions':physical_p,'capture_forwards':capture_n,'capture_input_positions':capture_p,'parity_forwards':2,'parity_input_positions':parity_p,'API_attempted':api_n,'API_returned':api_n,'API_failed':0,'API_pending':0,'operation_accounting_scope':worker['operation_accounting']['scope'],'backwards':0,'optimizer_steps':0,'exact_component_prehook_API_and_per_token_cost_reconciliation':True,'all_measurement_file_byte_links_verified':True},'model_preservation':{'fresh_dense_M0_and_no_STATE1_or_study_checkpoint':True,'base_dtype':'bfloat16','activation_dtype':'float32','physical_zero_B_parity_exact':True,'zero_B_before_after':True,'qv_sites':56,'trainable_zero_adapter_factors':112,'parameter_aliases_preserved':423,'all310_independent_native_base_bytes_unchanged':True,'activation_sites_observed':198,'all_activation_observation_counts':physical_n,'published_model_assets_independently_CPU_byte_hashed':True},'numerics':{'prior_INFQ1_audit_sha256':scope['inference1_audit_sha256'],'same_byte_bound_qualified_helpers_and_profile':True,'prior_max_selected_LP_error':infer_audit['numerics']['maximum_selected_logprob_error'],'prior_largest_mean_LP_error':infer_audit['numerics']['largest_row_mean_selected_logprob_error'],'max_limit':.03125,'mean_limit':.01,'all_current_selected_LPs_finite_nonpositive':True,'current_physical_zero_B_output_error':0,'scope_limit':'Prior four-TRAIN repeat/prefix-bulk qualification transfers compute mechanics only. Current64 raw outputs were not independently rerun for argmax/logit correctness, graded, or enrolled.'},'resources':{'CPU_samples':len(cpu),'GPU_samples':len(gpu),'minimum_sampled_host_available_bytes':min(row['host_available_bytes'] for row in cpu),'maximum_owned_tree_RSS_bytes':max(row['worker_tree_rss_bytes'] for row in cpu),'minimum_disk_free_bytes':min(row['disk_free_bytes'] for row in cpu),'minimum_sampled_GPU_free_bytes':min(row['gpu_free_bytes'] for row in gpu),'direct_terminal_GPU':gpu_final,'maximum_CPU_sample_envelope_seconds':cpu_window,'maximum_CPU_start_gap_seconds':cpu_gap,'maximum_preceding_CPU_age_at_next_sample_publication_seconds':cpu_publish_age,'maximum_GPU_sample_envelope_seconds':gpu_window,'maximum_GPU_start_gap_seconds':gpu_gap,'maximum_preceding_GPU_age_at_next_sample_publication_seconds':gpu_publish_age,'maximum_heartbeat_gap_seconds':hb_gap,'CPU_freshness_limit_seconds':.2,'GPU_freshness_limit_seconds':2,'all_registered_amount_cadence_freshness_and_deadline_gates_verified':True,'limits':'Sampled envelopes and successful pinned monitor checks support this one completed run; no broad sustained-training/native-fit conclusion.'},'cleanup':{'owned_job_identity_sha256':digest(owned),'assigned_before_resume':True,'external_monitor_ready_before_resume':True,'job_closed_kill_on_close':True,'owned_worker_identity_currently_alive':False,'sampler_threads_terminal_verified':True,'GPU_telemetry_cleanup_verified':True,'lease_path':str(lease),'lease_terminal_same_bytes_as_supervisor':True,'partial_outputs_frozen':True},'hash_closure':{'before':compact(before),'after':compact(after),'all_equal':True,'preservation_receipt_path':str(saved/'preservation_receipt.json'),'preservation_receipt_file_sha256':before['preservation_receipt']['sha256'],'preservation_receipt_body_sha256':pres['sha256'],'independently_registered_source_runtime65_decoder17_data5_model7_and_all_raw_files_verified':True,'maps_bound_by_config_and_preservation_receipt':'Every individual path/byte/hash is retained in the immutable config/preservation receipt; compact manifest hashes here bind verified complete maps.'},'CPU_audit_decoder_version':'tokenizers0.22.2','audit_framework_model_GPU_execution':False,'correctness_grade_performed':False,'raw_collection_mechanics_verified':True,'correctness_certified':False,'retention_certified':False,'prior_correct_enrollment_performed':False,'scoring_authorized':False,'training_authorized':False,'language_qualification_established':False,'long_context_qualification_established':False,'protected_bank_access':False,'paid_services':False,'confirmation_eligible':False,'quality_gain_established':False,'M0_M1_M2_comparison_completed':False,'native_study_training_qualified':False,'conclusion':'One fixed64-row unchanged-M0 batch completed and its raw token/likelihood/finish/cost/preservation/resource evidence is independently closed. No correctness, enrollment, transfer or retention result is established.'}
 audit['registered_successor_inventory']={'path':str(registration_path),'file_sha256':expected_inventory_registration_file_sha256,
     'body_sha256':registration['sha256'],'profile_body_sha256':expected_profile_body_sha256,'exact_counts':counts,
     'authority_boundary':'externally trusted root/source/scope-byte registration anchors; supplied hashes alone do not establish independent auditor authority',
     'new_profile_only':True,'legacy_auditor_18f6_unchanged':True}
 audit['sha256']=digest(audit)
 if target is not None:
  with target.open('x',encoding='utf-8',newline='\n') as stream:
   json.dump(audit,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
  written=read(target);check(written['sha256']==digest({k:v for k,v in written.items() if k!='sha256'}),'saved audit checksum')
 return audit


def main():
 import argparse
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--config",required=True);parser.add_argument("--raw-directory",required=True)
 parser.add_argument("--preserved-directory",required=True);parser.add_argument("--kind",choices=("atomic_RS_DEV","task_guard_DEV"),required=True)
 parser.add_argument("--ordinal",type=int,required=True);parser.add_argument("--config-file-sha256",required=True)
 parser.add_argument("--preservation-body-sha256",required=True);parser.add_argument("--root-terminal-and-preserved",action="store_true")
 parser.add_argument("--inventory-registration",required=True);parser.add_argument("--inventory-registration-file-sha256",required=True)
 parser.add_argument("--profile-body-sha256",required=True)
 parser.add_argument("--output")
 args=parser.parse_args()
 result=audit_terminal(args.config,args.raw_directory,args.preserved_directory,expected_selector=(args.kind,args.ordinal),
     expected_config_file_sha256=args.config_file_sha256,expected_preservation_body_sha256=args.preservation_body_sha256,
     inventory_registration_path=args.inventory_registration,expected_inventory_registration_file_sha256=args.inventory_registration_file_sha256,
     expected_profile_body_sha256=args.profile_body_sha256,
     terminal_preservation_signal=SIGNAL if args.root_terminal_and_preserved else None,output_path=args.output)
 print(json.dumps({"status":result["status"],"checks_passed":result["checks_passed"],"sha256":result["sha256"],"output":args.output}))


if __name__=="__main__":main()
