"""Strict Stage 1 provenance admission and unchanged bridge-subset export.

This module does not license generation or choose a dose/control design. Imports
are inert. verify_upstream lazily calls the unchanged Stage 1 helpers; they may
transitively import torch, but read_capture is always metadata_only=True. Only
export_upstream deserializes the hash-verified bridge artifact, never activations,
models or checkpoints. Export destinations are exclusive and stage-local.
"""
from __future__ import annotations
import datetime
import hashlib
import importlib
import json
import pathlib
import re
import sys
from types import SimpleNamespace

STAGE = pathlib.Path(__file__).resolve().parents[1]
TRACK = STAGE.parents[1]
PROGRAMME = TRACK.parent
REPO = PROGRAMME.parents[1]
AUDIT_ROOT = TRACK/'evidence/07_independent_audit'
EXPORT_ROOT = STAGE/'evidence/00_upstream_exports'
OBJECTIVES = ('B0','B1','B2','B3','BS','BR')
PAIRS = tuple((i,j) for i in (8,12,16,19) for j in (12,16,20,24))
CONFIG_SHA256 = '68b5de95cf4eca48d8a93fb26ff7cae9148afa1636d05fc834d5a628fd9655be'
SHAPES = {'up':(1024,2560),'up_bias':(2560,),'down':(2560,1024),'down_bias':(1024,)}

class UpstreamError(RuntimeError):
    """Upstream evidence is incomplete, changed, inadmissible or out of scope."""

def _require(condition, message):
    if not condition:
        raise UpstreamError(message)

def _sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):h.update(block)
    return h.hexdigest()

def _canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def _read(path):
    def constants(value):raise UpstreamError('nonfinite JSON constant '+value)
    def pairs(items):
        result={}
        for key,value in items:
            _require(key not in result,'duplicate JSON key '+key)
            result[key]=value
        return result
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'),parse_constant=constants,object_pairs_hook=pairs)

def _within(path,root):
    path,root=pathlib.Path(path).resolve(),pathlib.Path(root).resolve()
    _require(path.is_relative_to(root),'path outside authorized subtree: '+str(path))
    return path

def _legacy_helpers():
    # Stage 1 modules require programme imports. No source or shared module is
    # rewritten. Their transitive torch import is permitted, not tensor loading.
    if str(PROGRAMME) not in sys.path:sys.path.insert(0,str(PROGRAMME))
    caller=importlib.import_module('cross_scale.scripts.run_alignment')
    manifests=importlib.import_module('shared.manifests')
    return SimpleNamespace(configure=caller.configure,
        verify_registration=manifests.verify_registration,
        read_capture=caller.read_capture,
        collect_completed_pairs=caller.collect_completed_pairs,
        select_matrix=caller.select_matrix,
        validate_receipt=caller.validate_receipt,
        require_committed=manifests._require_committed)

def _stage_sources():
    files={p for root in (STAGE/'src',STAGE/'tests') for p in root.glob('*.py')}
    if (STAGE/'README.md').is_file():files.add(STAGE/'README.md')
    return {p.relative_to(REPO).as_posix():_sha(p) for p in sorted(files)}

def _strict_matrix(reports):
    keys=[(row.get('supa_layer'),row.get('donor_layer')) for row in reports]
    _require(len(keys)==len(set(keys))==16 and set(keys)==set(PAIRS),'complete 16-pair matrix required before selection')
    _require(all(set(row.get('objectives',{}))==set(OBJECTIVES) for row in reports),'complete 96-cell matrix including BS/BR required')

def _current_attempt_inventory(root):
    """Enumerate current terminal attempts; a saved audit is not a directory lock."""
    root=pathlib.Path(root).resolve()
    _require(root.is_dir(),'missing attempt root: '+str(root))
    inventory=[]
    for entry in sorted(root.iterdir()):
        _require(entry.is_dir() and re.fullmatch('attempt_[0-9]{3,}',entry.name),'invalid attempt directory: '+str(entry))
        directory=_within(entry,root)
        _require(directory.parent==root and directory.name==entry.name,'aliased attempt directory: '+str(entry))
        _require((directory/'started.json').is_file(),'missing started receipt: '+str(directory))
        final=directory/'run_manifest.json';abandoned=directory/'abandonment.json'
        _require(not (final.exists() and abandoned.exists()),'ambiguous terminal receipts: '+str(directory))
        terminal_path=final if final.exists() else abandoned
        _require(terminal_path.is_file(),'unfinished attempt: '+str(directory))
        terminal=_read(terminal_path)
        _require(isinstance(terminal,dict) and terminal.get('status') in ('COMPLETE','FAILED','ABANDONED'),'invalid terminal status: '+str(directory))
        inventory.append({'path':str(directory),'status':terminal['status'],'manifest_sha256':_sha(terminal_path)})
    return inventory

def _current_pair_inventory(root):
    root=pathlib.Path(root).resolve()
    _require(root.is_dir(),'missing pair root: '+str(root))
    expected={f'S{i}_D{j}' for i,j in PAIRS}
    children=list(root.iterdir())
    _require(all(child.is_dir() and child.name in expected for child in children),'unexpected pair-root child')
    _require({child.name for child in children}==expected,'complete 16-pair directory inventory required')
    inventory=[]
    for i,j in PAIRS:
        pair_root=_within(root/f'S{i}_D{j}',root)
        _require(pair_root.parent==root and pair_root.name==f'S{i}_D{j}','aliased pair directory')
        inventory.extend(_current_attempt_inventory(pair_root))
    return inventory

def _audit_header(path,run_id):
    receipt=_read(path/'audit_receipt.json')
    expected={'schema':'independent_cross_scale_alignment_v2_audit_v1','status':'PASS',
        'run_id':run_id,'matrix_pairs':16,'objective_cells':96,'candidate_gate_records':48,
        'selection_recomputed':True,'models_loaded':0,'torch_imported':False,
        'programme_helpers_imported':False,'checkpoint_weight_files_opened':False,
        'bridge_activation_deserialization':False,'generation_performed':False,'git_commands_run':False}
    _require(all(type(receipt.get(key)) is type(value) and receipt[key]==value for key,value in expected.items()),'complete independent PASS audit is required')
    return receipt

def verify_upstream(run_id: str, aggregate_attempt: int=0, *, audit_directory: pathlib.Path) -> dict:
    """Return a fully verified Stage 1 winner and immutable same-pair paths.

    Requires committed aggregate/matrix/selection, all terminal pair evidence,
    source/config/data bindings, and the complete independent PASS audit. Calls
    original read_capture(metadata_only=True), collect_completed_pairs and
    select_matrix. Never deserializes capture, bridge, model or checkpoint tensors.
    """
    _require(isinstance(run_id,str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}',run_id),'invalid Stage 1 run ID')
    _require(type(aggregate_attempt) is int and aggregate_attempt>=0,'invalid aggregate attempt')
    audit_dir=_within(audit_directory,AUDIT_ROOT)
    _require(audit_dir!=AUDIT_ROOT.resolve() and not audit_dir.is_relative_to((AUDIT_ROOT/'tools').resolve()),'an audit result directory is required')
    audit=_audit_header(audit_dir,run_id)  # Cheap fail closed before legacy imports.
    capture_dir=_within(TRACK/'evidence/01_alignment_matrix'/run_id,TRACK/'evidence')
    pair_root=_within(TRACK/'evidence/02_bridge_objectives'/run_id,TRACK/'evidence')
    local=_within(TRACK/'local'/run_id,TRACK/'local')
    aggregate_root=_within(capture_dir/'aggregate',capture_dir)
    aggregate_dir=_within(aggregate_root/f'attempt_{aggregate_attempt:03d}',aggregate_root)
    current_pairs=_current_pair_inventory(pair_root)
    current_aggregates=_current_attempt_inventory(aggregate_root)
    _require([pathlib.Path(item['path']) for item in current_aggregates if item['status']=='COMPLETE']==[aggregate_dir],'exactly one matching complete aggregate required')
    aggregate=_read(aggregate_dir/'run_manifest.json')
    _require(aggregate.get('status')=='COMPLETE' and aggregate.get('arm')=='aggregate','committed complete aggregate required')
    helpers=_legacy_helpers()
    config_path=TRACK/'configs/alignment_v2.json'
    config=helpers.configure(config_path)
    _require(_canonical(config)==CONFIG_SHA256 and config['device']=='cpu' and config['generation_enabled'] is False and config['qwen8b_eligible'] is False,'only original frozen CPU Stage 1 is supported')
    registration=helpers.verify_registration('cross_scale',config_path)
    capture,bundle=helpers.read_capture(run_id,config,registration,metadata_only=True)
    _require(bundle is None,'metadata-only upstream verification must not return activations')
    _require(capture.get('status')=='COMPLETE','complete capture required')
    capture_path=capture_dir/'capture.json'
    capture_hash=_sha(capture_path)
    reports,sources=helpers.collect_completed_pairs(run_id,config,registration,capture_hash,local)
    _require(sources==current_pairs,'current pair inventory differs from collected evidence')
    _strict_matrix(reports)
    helpers.validate_receipt(aggregate_dir,aggregate,config,registration,'aggregate')
    _require(aggregate['dataset_manifest_sha256']==capture['dataset_manifest_sha256'],'aggregate dataset differs from capture')
    matrix_path=aggregate_dir/'matrix.json'
    matrix=_read(matrix_path)
    _require(_sha(matrix_path)==aggregate['matrix_sha256'] and matrix=={'pairs':reports,'sources':sources},'aggregate matrix/source inventory differs from complete pair evidence')
    selection_path=aggregate_dir/'selection.json'
    saved=_read(selection_path)
    reconstructed=helpers.select_matrix(reports,config)
    _require(saved==aggregate['selection']==reconstructed,'edited or inconsistent Stage 1 selection')
    _require(reconstructed.get('status')=='MECHANISM_GO' and reconstructed.get('large_generation_licensed') is False and reconstructed.get('small_generation_eligible_for_separate_freeze') is True,'mechanism GO with separate future freeze required')
    winners=reconstructed.get('selected',[])
    _require(len(winners)==1 and winners[0].get('go') is True,'exactly one preregistered eligible winner required')
    winner=dict(winners[0])
    pair=(winner['supa_layer'],winner['donor_layer'])
    _require(pair in PAIRS and winner['objective'] in ('B1','B2','B3'),'winner is not a registered candidate')
    source_files=dict(registration['sources'])
    disk_registration_path=_within(registration['registration_path'],TRACK/'evidence/00_preregistration')
    dataset_path=_within(capture['dataset_manifest'],PROGRAMME/'data/generated/v2/cross_scale')
    bank=_read(dataset_path)
    calibration_path=_within(dataset_path.parent/bank['datasets']['calibration']['path'],dataset_path.parent)
    _require(calibration_path.name=='calibration.jsonl','calibration path mismatch')
    _require(_sha(calibration_path)==bank['datasets']['calibration']['sha256'],'calibration input bytes changed')
    required_inputs={str(REPO/name):digest for name,digest in source_files.items()}
    required_inputs.update({str(config_path):_sha(config_path),str(disk_registration_path):_sha(disk_registration_path),
        str(dataset_path):capture['dataset_manifest_sha256'],str(calibration_path):_sha(calibration_path)})
    commitment_paths={disk_registration_path,config_path,dataset_path,calibration_path}
    for name,digest in source_files.items():
        path=_within(REPO/name,REPO)
        _require(_sha(path)==digest,'upstream source changed: '+name)
        if path.is_relative_to(PROGRAMME):commitment_paths.add(path)
    for name in ('capture.json','run_manifest.json','started.json'):
        path=capture_dir/name;required_inputs[str(path)]=_sha(path);commitment_paths.add(path)
    selected_artifact=None
    terminal_inventory=[]
    expected_bridge_paths={}
    for source in sources:
        directory=_within(source['path'],pair_root)
        _require(directory.parent.name in {f'S{i}_D{j}' for i,j in PAIRS} and re.fullmatch('attempt_[0-9]{3,}',directory.name),'invalid source attempt path')
        terminal_path=directory/'run_manifest.json'
        if not terminal_path.exists():terminal_path=directory/'abandonment.json'
        terminal=_read(terminal_path)
        _require(source['status']==terminal['status'] and source['manifest_sha256']==_sha(terminal_path),'pair source terminal binding differs')
        _require(terminal['status'] in ('COMPLETE','FAILED','ABANDONED'),'unfinished source attempt')
        terminal_inventory.append(dict(source))
        for path in directory.rglob('*'):
            if path.is_file():
                path=_within(path,directory)
                _require(path.suffix=='.json','unexpected non-JSON science evidence artifact')
                commitment_paths.add(path);required_inputs[str(path)]=_sha(path)
        if terminal['status']=='COMPLETE':
            _require(terminal['capture_sha256']==capture_hash and terminal['activation_sha256']==capture['activation_sha256'] and terminal['dataset_manifest_sha256']==capture['dataset_manifest_sha256'],'pair capture/data binding differs')
            _require(terminal['donor_identity']==capture['donor_identity'] and terminal['supa_identity']==capture['supa_identity'],'pair model/donor identity differs')
            bridge_path=_within(terminal['bridge_path'],local)
            _require(bridge_path==(local/'pairs'/directory.parent.name/directory.name/'bridges.pt').resolve(),'bridge artifact belongs to another pair/attempt')
            expected_bridge_paths[str(bridge_path)]=terminal['bridge_sha256']
            required_inputs[str(bridge_path)]=terminal['bridge_sha256']
            if directory.parent.name==f'S{pair[0]}_D{pair[1]}':
                _require(selected_artifact is None,'ambiguous winner pair artifact')
                selected_artifact={'path':str(bridge_path),'sha256':terminal['bridge_sha256'],
                    'pair_directory':str(directory),'pair_manifest_sha256':_sha(terminal_path),
                    'metrics_path':str(directory/'metrics.json'),'metrics_sha256':terminal['metrics_sha256']}
    _require(selected_artifact is not None and len(expected_bridge_paths)==16,'all 16 pair bridge artifacts required')
    activation_path=local/'activations.pt'
    required_inputs[str(activation_path)]=capture['activation_sha256']
    for source in current_aggregates:
        directory=_within(source['path'],aggregate_root)
        terminal_path=directory/'run_manifest.json'
        if not terminal_path.exists():terminal_path=directory/'abandonment.json'
        helpers.validate_receipt(directory,_read(terminal_path),config,registration,'aggregate')
        for path in directory.rglob('*'):
            if path.is_file():
                path=_within(path,directory)
                _require(path.suffix=='.json','unexpected non-JSON aggregate evidence artifact')
                commitment_paths.add(path);required_inputs[str(path)]=_sha(path)
    _require(audit['selection']==reconstructed and pathlib.Path(audit['aggregate_directory']).resolve()==aggregate_dir and audit['capture_sha256']==capture_hash,'audit belongs to a different selection/capture/aggregate')
    _require(audit['registration_contract_sha256']==registration['registration_sha256'] and audit['embedded_registration_sha256']==_canonical(registration) and audit['config_canonical_sha256']==_canonical(config),'audit registration/config binding differs')
    audit_script=AUDIT_ROOT/'tools/audit_alignment_matrix_v2.py'
    _require(audit['script_sha256']==_sha(audit_script),'independent auditor source changed since PASS')
    commitment_paths.add(audit_script)
    inventory_path=audit_dir/'attempt_inventory.json';input_path=audit_dir/'input_hashes.json'
    _require(_sha(inventory_path)==audit['attempt_inventory_sha256'] and _sha(input_path)==audit['input_hashes_sha256'],'audit output hash mismatch')
    inventory=_read(inventory_path);inputs=_read(input_path)
    _require([{key:item[key] for key in ('path','status','manifest_sha256')} for item in inventory['pairs']]==sources,'audit omitted/changed failed or completed pair attempts')
    audit_aggregates=inventory['aggregates']
    _require([{key:item[key] for key in ('path','status','manifest_sha256')} for item in audit_aggregates]==current_aggregates,'audit omitted/changed current aggregate attempts')
    _require(inventory['capture']['capture_sha256']==capture_hash and pathlib.Path(inventory['capture']['path']).resolve()==capture_dir,'audit capture inventory differs')
    fixed_paths={pathlib.Path(name).resolve() for name in required_inputs}
    donor_record=REPO/'Rineya_Cross_Scale_Capability_Transfer_Lab_v1/cross_scale_evidence/02_architecture/qwen4b.json'
    fixed_paths.add(donor_record.resolve())
    normalized_inputs={}
    for name,expected in inputs.items():
        path=_within(name,REPO)
        _require(path not in normalized_inputs,'duplicate normalized audit input path')
        # No audit-supplied path may cause arbitrary checkpoint reads.
        allowed=path in fixed_paths or (path.suffix=='.json' and (path.is_relative_to(capture_dir) or path.is_relative_to(pair_root)))
        _require(allowed,'unregistered file in audit input inventory: '+str(path))
        _require(isinstance(expected,str) and re.fullmatch('[0-9a-f]{64}',expected) and _sha(path)==expected,'audit input bytes changed: '+str(path))
        normalized_inputs[path]=expected
        if path.suffix=='.json' and (path.is_relative_to(capture_dir) or path.is_relative_to(pair_root)):
            commitment_paths.add(path)
    _require(all(normalized_inputs.get(pathlib.Path(name).resolve())==digest for name,digest in required_inputs.items()),'independent audit input inventory is incomplete or differently bound')
    for name in ('audit_receipt.json','attempt_inventory.json','input_hashes.json'):commitment_paths.add(audit_dir/name)
    stage_sources=_stage_sources()
    _require(stage_sources,'generation-stage code inventory is empty')
    for relative,digest in stage_sources.items():
        path=_within(REPO/relative,STAGE);commitment_paths.add(path)
        _require(_sha(path)==digest,'generation-stage source changed')
    for path in sorted(commitment_paths):helpers.require_committed(path)
    _require(_current_pair_inventory(pair_root)==current_pairs and _current_attempt_inventory(aggregate_root)==current_aggregates,'attempt inventory changed during verification')
    context={'schema_version':'cross_scale_generation_upstream_context_v1',
        'supa_hidden_size':1024,'donor_hidden_size':2560,'supa_layer':pair[0],'donor_layer':pair[1],
        'selected_objective':winner['objective'],'supa_files':capture['supa_files'],
        'supa_identity':capture['supa_identity'],'donor_identity':capture['donor_identity'],
        'checkpoint_sha256':config['checkpoint_sha256'],'donor_revision':config['donor_revision'],
        'stage1_config':config,'stage1_config_sha256':_canonical(config),
        'stage1_registration_id':registration['registration_sha256'],
        'stage1_embedded_registration_sha256':_canonical(registration),
        'stage1_capture_sha256':capture_hash,'stage1_sources':source_files,
        'generation_stage_source_files':stage_sources,'generation_stage_source_sha256':_canonical(stage_sources),
        'generation_licensed':False,'dose_frozen':False,'matched_control_frozen':False}
    return {'schema_version':'cross_scale_upstream_verified_v1','status':'VERIFIED_MECHANISM_GO',
        'run_id':run_id,'aggregate_attempt':aggregate_attempt,'selected':winner,'context':context,
        'registration_path':str(disk_registration_path),'registration_file_sha256':_sha(disk_registration_path),
        'capture_directory':str(capture_dir),'capture_sha256':capture_hash,
        'aggregate_directory':str(aggregate_dir),'matrix_sha256':_sha(matrix_path),
        'selection_sha256':_sha(selection_path),'aggregate_manifest_sha256':_sha(aggregate_dir/'run_manifest.json'),
        'audit_directory':str(audit_dir),'audit_receipt_sha256':_sha(audit_dir/'audit_receipt.json'),
        'audit_input_hashes_sha256':_sha(input_path),'audit_attempt_inventory_sha256':_sha(inventory_path),
        'original_bridges':selected_artifact,'export_source_objectives':{'XG':winner['objective'],'XS':'BS','XR':'BR'},
        'pair_attempt_inventory':terminal_inventory,'committed_file_count':len(commitment_paths),
        'upstream_input_hashes':{str(path):digest for path,digest in normalized_inputs.items()},
        'generation_licensed':False,
        'limitations':['Verification calls unchanged Stage 1 helpers and may transitively import torch, but never deserializes activations/models/checkpoints.',
            'Independent PASS certifies metadata/scalar/ranking consistency, not model or bridge numerical recomputation.',
            'This adapter supplies unchanged bridges only; dose, matched control and generation protocol remain unfrozen.']}

def _validate_bridge(bridge,torch):
    _require(isinstance(bridge,dict) and set(bridge)==set(SHAPES),'bridge must contain exactly up/up_bias/down/down_bias')
    for key,shape in SHAPES.items():
        tensor=bridge[key]
        _require(isinstance(tensor,torch.Tensor) and tensor.device.type=='cpu' and tensor.dtype==torch.float32 and tensor.layout==torch.strided and tuple(tensor.shape)==shape,'invalid CPU FP32 bridge tensor: '+key)
        _require(not tensor.requires_grad and tensor.grad is None and bool(torch.isfinite(tensor).all()),'bridge tensor is nonfinite/trainable/has gradient: '+key)

def _tensor_hash(tensor):
    value=tensor.detach().contiguous()
    h=hashlib.sha256(str((value.dtype,tuple(value.shape))).encode())
    h.update(memoryview(value.numpy()).cast('B'))
    return h.hexdigest()

def _write_json(path,value):
    with pathlib.Path(path).open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(value,handle,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);handle.write('\n')

def export_upstream(run_id: str, aggregate_attempt: int=0, *, audit_directory: pathlib.Path,
                    output_directory: pathlib.Path) -> dict:
    """Export unchanged XG/XS/XR tensor dictionaries after strict admission.

    Each exported map has exact keys up/up_bias/down/down_bias. XG comes from the
    unique selected objective; XS/XR come from BS/BR in the same original pair
    artifact. No refit, precision conversion, scaling, dose choice or generation.
    """
    destination=_within(output_directory,EXPORT_ROOT)
    _require(destination!=EXPORT_ROOT.resolve() and not destination.exists(),'export directory must be new and below the stage export root')
    upstream=verify_upstream(run_id,aggregate_attempt,audit_directory=audit_directory)
    original=pathlib.Path(upstream['original_bridges']['path'])
    expected=upstream['original_bridges']['sha256']
    _require(_sha(original)==expected,'original bridge artifact changed before export')
    import torch  # Only this function deserializes bridge tensors.
    bridges=torch.load(original,map_location='cpu',weights_only=True)
    _require(isinstance(bridges,dict) and set(bridges)==set(OBJECTIVES),'original artifact must contain all six objectives including null controls')
    exported={}
    tensor_hashes={}
    for arm,objective in upstream['export_source_objectives'].items():
        _validate_bridge(bridges[objective],torch)
        exported[arm]={key:value.detach().clone().contiguous() for key,value in bridges[objective].items()}
        tensor_hashes[arm]={key:_tensor_hash(value) for key,value in bridges[objective].items()}
        _require({key:_tensor_hash(value) for key,value in exported[arm].items()}==tensor_hashes[arm],'export clone changed tensor bytes')
    _require(_sha(original)==expected,'original bridge artifact changed while loading')
    _require(_stage_sources()==upstream['context']['generation_stage_source_files'],'stage code changed after admission')
    destination.mkdir(parents=True,exist_ok=False)
    tensor_path=destination/'bridges.pt'
    try:
        with tensor_path.open('xb') as handle:torch.save(exported,handle)
        reloaded=torch.load(tensor_path,map_location='cpu',weights_only=True)
        _require(set(reloaded)=={'XG','XS','XR'},'exported role set changed on reload')
        for arm,values in reloaded.items():
            _validate_bridge(values,torch)
            _require({key:_tensor_hash(value) for key,value in values.items()}==tensor_hashes[arm],'exported tensors differ on reload')
        _require(_sha(original)==expected,'original bridge artifact changed after export')
        _require(_stage_sources()==upstream['context']['generation_stage_source_files'],'stage source changed during export')
        manifest={'schema_version':'cross_scale_generation_bridge_export_v1','status':'COMPLETE',
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'upstream':upstream,'context':upstream['context'],'source_objectives':upstream['export_source_objectives'],
            'tensor_file':'bridges.pt','tensor_file_sha256':_sha(tensor_path),'tensor_sha256':tensor_hashes,
            'original_artifact_sha256_before':expected,'original_artifact_sha256_after':_sha(original),
            'subset_byte_identical_on_reload':True,'dtype':'float32','device':'cpu',
            'dimensions':{'supa':1024,'donor':2560},'frozen':True,'trainable_parameters':0,'gradient_tensors':0,
            'refit_performed':False,'weights_changed':False,'generation_licensed':False,
            'dose_frozen':False,'matched_control_frozen':False}
        _write_json(destination/'manifest.json',manifest)
    except BaseException as error:
        _write_json(destination/'failure.json',{'status':'FAILED','type':type(error).__name__,'message':str(error),
            'upstream':upstream,'generation_licensed':False})
        raise
    return {'directory':str(destination),'manifest_path':str(destination/'manifest.json'),
        'manifest_sha256':_sha(destination/'manifest.json'),'tensor_path':str(tensor_path),
        'tensor_sha256':manifest['tensor_file_sha256'],'context':manifest['context'],'generation_licensed':False}
