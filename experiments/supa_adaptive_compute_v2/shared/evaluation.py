"""Per-arm immutable shards shared by both scientific callers.

Resumption means skipping completed compatible shards and creating new exclusive
attempts for incomplete ones. Historical evidence is never appended or repaired.
"""
from pathlib import Path
import json
import time
import traceback
from .legacy import ROOT, REPO
from .hashing import canonical_sha256, sha256_file, source_files, write_json
from .guards import require_track, require_within, require_runtime
from .manifests import new_run, utc_now, verify_registration
from .datasets import load_frozen_dataset
from .model_loading import load_supa
from .integrity import assert_unchanged, frozen_inventory, state_hash, validate_inference_receipt
from .timing import machine_lease


def shard_cases(rows, index, count):
    if type(index) is not int or type(count) is not int or count < 1 or not 0 <= index < count:
        raise ValueError('invalid deterministic shard')
    # Scenario ownership also keeps linked variants in the same shard.
    return [row for row in rows if int(canonical_sha256(row['scenario_id']),16) % count == index]


def _read_complete(path, track):
    path = require_within(path, ROOT / track / 'evidence')
    final = json.loads((path / 'run_manifest.json').read_text(encoding='utf-8'))
    if final['status'] != 'COMPLETE' or final['track'] != track or not final.get('canonical_unchanged'):
        raise RuntimeError('failed/incomplete/integrity-failed evidence cannot be aggregated')
    rows_path = path / 'results.json'
    if sha256_file(rows_path) != final['results_sha256']:
        raise RuntimeError('evidence results changed')
    rows = json.loads(rows_path.read_text(encoding='utf-8'))
    if [r['example_id'] for r in rows] != final['example_ids'] or any(r.get('safety_stopped') for r in rows):
        raise RuntimeError('incomplete or reordered results')
    return final, rows


def _compatible(manifest, contract, baseline=False):
    keys = ('registration_sha256','dataset_manifest_sha256','role','config_sha256', 'shard_index','shard_count','example_ids')
    return all(manifest.get(k) == contract.get(k) for k in keys)


def _find_baseline(track, contract, config):
    baseline = 'A0' if track == 'native_armr' else 'X0'
    found = []
    for path in (ROOT / track / 'evidence').glob('*/*/run_manifest.json'):
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest.get('arm') == baseline and manifest.get('status') == 'COMPLETE' and _compatible(manifest,contract):
            _read_complete(path.parent,track)
            validate_inference_receipt(manifest,config)
            if manifest.get('sources') != contract['sources']:
                raise RuntimeError('baseline source identity differs from registered execution')
            found.append(path.parent)
    if len(found) != 1:
        raise RuntimeError('exactly one complete compatible A0/X0 shard required before candidate execution')
    return found[0]


def qualify_native_architecture(config, dataset_manifest, registration):
    """All frozen architecture cases and arms, never efficacy-based pruning."""
    expected = load_frozen_dataset(dataset_manifest,track='native_armr',role='architecture_dev')
    ids = {row['example_id'] for row in expected}
    by_arm = {arm:{} for arm in config['arms']}
    identities = set()
    for path in (ROOT/'native_armr/evidence/01_architecture_dev').glob('*/run_manifest.json'):
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('status') != 'COMPLETE' or value.get('role') != 'architecture_dev':
            continue
        if value.get('registration_sha256') != registration['registration_sha256'] or value.get('dataset_manifest_sha256') != sha256_file(Path(dataset_manifest)):
            continue
        value, rows = _read_complete(path.parent,'native_armr')
        validate_inference_receipt(value,config)
        if value.get('sources') != registration['sources']:
            raise RuntimeError('architecture source binding mismatch')
        identities.add(canonical_sha256({k:value.get(k) for k in ('state_sha256_before','checkpoint_files_before','model_class','qwen_implementation_sha256')}))
        if value['arm'] not in by_arm:
            raise RuntimeError('unregistered architecture arm')
        for row in rows:
            target = by_arm[value['arm']]
            if row['example_id'] in target:
                raise RuntimeError('duplicate architecture evidence; resolve provenance before confirming')
            target[row['example_id']] = row
    if any(set(rows) != ids for rows in by_arm.values()):
        raise RuntimeError('complete frozen architecture-development matrix is required before confirmatory execution')
    if len(identities) != 1:
        raise RuntimeError('architecture model identity differs between paired arms')
    from native_armr.src.analysis import audit_rows
    audit_rows(expected,{arm:list(rows.values()) for arm,rows in by_arm.items()},config,'architecture_dev')
    base_invalid = sum(bool(r.get('invalid')) for r in by_arm['A0'].values())
    for arm, rows in by_arm.items():
        if any(r.get('safety_stopped') for r in rows.values()) or sum(bool(r.get('invalid')) for r in rows.values()) > base_invalid:
            raise RuntimeError(f'architecture validity prerequisite failed for {arm}; do not prune or tune on confirmation')
    return {'status':'QUALIFIED', 'n':len(ids),'arms':list(by_arm), 'selection':'none; preregistered arms retained'}


def run_shard(track, arm, config, dataset_manifest, role, shard_index, shard_count,
              operator_factory, model_path=None):
    from .generation import evaluate_case
    from armr.hooks import frozen_inference
    import torch
    require_track(track)
    require_runtime(config)
    config_path = ROOT / track / 'configs/protocol.json'
    registration = verify_registration(track,config_path)
    if registration['config'] != config:
        raise RuntimeError('unregistered runtime configuration')
    if arm not in config.get('arms', {}):
        raise ValueError('unregistered generation arm')
    dataset_manifest = require_within(dataset_manifest,ROOT / 'data')
    relative = dataset_manifest.relative_to(REPO).as_posix()
    if registration['dataset_manifests'].get(relative) != sha256_file(dataset_manifest):
        raise RuntimeError('dataset not bound by track registration')
    if role not in ('architecture_dev','confirmatory_dev','confirmatory_diagnostics'):
        raise ValueError('generation role forbidden')
    if track == 'cross_scale' and not config.get('generation_enabled',False):
        raise RuntimeError('Track B generation is not licensed by alignment-only registration')
    if role.startswith('confirmatory') and track == 'native_armr':
        qualify_native_architecture(config,dataset_manifest,registration)
    rows = load_frozen_dataset(dataset_manifest,track=track,role=role)
    cases = shard_cases(rows,shard_index,shard_count)
    if not cases:
        raise ValueError('empty shard; choose a smaller shard count')
    contract = {'arm':arm,'role':role,'config_sha256':canonical_sha256(config),
        'dataset_manifest_sha256':sha256_file(dataset_manifest),'dataset_manifest':str(dataset_manifest),
        'registration_sha256':registration['registration_sha256'], 'sources':source_files(track),
        'shard_index':shard_index, 'shard_count':shard_count, 'example_ids':[r['example_id'] for r in cases],
        'seed':config['seed'],'threads':config['threads'],'dtype':config['dtype'],
        'device':config['device'],'cache_mode':'full_prefix_no_cache', 'max_prompt_seconds':config['max_prompt_seconds']}
    base_arm = 'A0' if track == 'native_armr' else 'X0'
    baseline_path = None if arm == base_arm else _find_baseline(track,contract,config)
    contract['baseline_run_directory'] = None if baseline_path is None else str(baseline_path)
    contract['baseline_manifest_sha256'] = None if baseline_path is None else sha256_file(baseline_path/'run_manifest.json')
    stage = '02_confirmatory_run' if role.startswith('confirmatory') and track == 'native_armr' else '01_architecture_dev' if track == 'native_armr' else '04_small_generation'
    output, manifest = new_run(track,stage,contract)
    print(f'RUN_DIR={output}',flush=True)
    start = time.perf_counter()
    model = None
    identity = None
    wrapper = None
    extra_before = None
    results = []
    failure = None
    try:
        with machine_lease():
            model, tokenizer, identity = load_supa(config,model_path)
            manifest.update(identity)
            write_json(output / 'model_identity.json',identity)
            if baseline_path is not None:
                if sha256_file(baseline_path/'run_manifest.json') != contract['baseline_manifest_sha256']:
                    raise RuntimeError('baseline manifest changed after selection')
                baseline_identity = json.loads((baseline_path/'run_manifest.json').read_text(encoding='utf-8'))
                for key in ('state_sha256_before','checkpoint_files_before','model_class','torch','transformers','qwen_implementation_sha256','parameters'):
                    if identity.get(key) != baseline_identity.get(key):
                        raise RuntimeError(f'candidate-loaded model differs from baseline before inference: {key}')
            wrapper = operator_factory(model,arm,config)
            extra_module = getattr(wrapper,'transform',getattr(wrapper,'expert',None))
            if extra_module is not None:
                extra_inventory = frozen_inventory(extra_module)
                manifest['operator_trainable_parameters'] = extra_inventory['trainable_parameters']
                manifest['operator_gradients'] = extra_inventory['backbone_gradients']
                extra_before = state_hash(extra_module)
            manifest['operator'] = {k:getattr(wrapper,k,0) for k in
                ('added_parameter_count','added_active_parameters','extra_matmul_flops_per_token')}
            (output / 'cases').mkdir()
            (output / 'logits').mkdir()
            with wrapper, frozen_inference(model):
                for case in cases:
                    name = canonical_sha256(case['example_id'])
                    base = None
                    if baseline_path:
                        bf = baseline_path / 'logits' / (name+'.pt')
                        bm = json.loads((baseline_path/'run_manifest.json').read_text())
                        if sha256_file(bf) != bm['logits_sha256'][bf.name]:
                            raise RuntimeError('baseline logits changed')
                        base = torch.load(bf,map_location='cpu',weights_only=True)
                    row, first = evaluate_case(model,tokenizer,case,config,wrapper,base)
                    row['arm'] = arm
                    write_json(output / 'cases' / (name+'.json'),row)
                    with (output / 'logits' / (name+'.pt')).open('xb') as handle:
                        torch.save(first,handle)
                    results.append(row)
                    print(f"{arm} {case['example_id']} score={row['score']} stop={row['termination']} elapsed={row['wall_seconds']:.2f}s",flush=True)
            if extra_module is not None:
                extra_inventory = frozen_inventory(extra_module)
                manifest['operator_trainable_parameters_after'] = extra_inventory['trainable_parameters']
                manifest['operator_gradients_after'] = extra_inventory['backbone_gradients']
                if state_hash(extra_module) != extra_before:
                    raise RuntimeError('extra operator changed during inference')
                manifest['operator_state_sha256'] = extra_before
            manifest.update(assert_unchanged(model,identity['state_sha256_before'],Path(identity['model_path']),identity['checkpoint_files_before']))
            if registration['sources'] != source_files(track):
                raise RuntimeError('source changed during execution')
            load_frozen_dataset(dataset_manifest,track=track,role=role)
            write_json(output/'results.json',results)
            manifest.update(status='COMPLETE',results_sha256=sha256_file(output/'results.json'),
                logits_sha256={p.name:sha256_file(p) for p in (output/'logits').glob('*.pt')})
    except BaseException as exc:
        failure = exc
        manifest.update(status='FAILED',failure_receipt={'type':type(exc).__name__,'message':str(exc), 'traceback':traceback.format_exc()}, completed_examples=len(results))
        if model is not None and identity is not None:
            try:
                manifest['failure_integrity'] = assert_unchanged(model,identity['state_sha256_before'],Path(identity['model_path']),identity['checkpoint_files_before'])
            except BaseException as check_exc:
                manifest['failure_integrity'] = {'canonical_unchanged':False,'error':str(check_exc)}
    finally:
        import psutil
        memory = psutil.Process().memory_info()
        manifest.update(end_utc=utc_now(),wall_seconds=time.perf_counter()-start,
                        peak_memory_bytes=getattr(memory,'peak_wset',memory.rss))
        write_json(output/'run_manifest.json',manifest)
    if failure is not None:
        raise failure
    return output


def aggregate_runs(track,config,dataset_manifest,role,run_dirs):
    from .statistics import paired_analysis
    config_path = ROOT/track/'configs'/('protocol.json' if track == 'native_armr' else 'alignment_v2.json')
    registration = verify_registration(track,config_path)
    if registration['config'] != config:
        raise RuntimeError('aggregation configuration differs from registered contract')
    expected = load_frozen_dataset(dataset_manifest,track=track,role=role)
    ids = {r['example_id'] for r in expected}
    cases = {r['example_id']:r for r in expected}
    by_arm = {}
    source_bindings = set()
    for path in run_dirs:
        manifest,rows = _read_complete(Path(path),track)
        if manifest['config_sha256'] != canonical_sha256(config) or manifest['dataset_manifest_sha256'] != sha256_file(Path(dataset_manifest)) or manifest['role'] != role:
            raise RuntimeError('incompatible shard configuration, dataset or role')
        if manifest['registration_sha256'] != registration['registration_sha256'] or manifest.get('sources') != registration['sources']:
            raise RuntimeError('aggregation requires current registered source evidence')
        validate_inference_receipt(manifest,config)
        source_bindings.add(manifest['registration_sha256'])
        target = by_arm.setdefault(manifest['arm'],{})
        for row in rows:
            if row['example_id'] in target:
                raise RuntimeError('duplicate example/arm across shard attempts')
            if row['example_id'] not in cases:
                raise RuntimeError('unregistered evaluation example')
            for key in ('scenario_id','family','track','split','variant','template_id','template_unseen_in_engineering'):
                if row.get(key) != cases[row['example_id']].get(key):
                    raise RuntimeError(f'raw evidence metadata differs from frozen input: {key}')
            target[row['example_id']] = row
    if len(source_bindings) != 1:
        raise RuntimeError('mixed registrations')
    baseline = 'A0' if track == 'native_armr' else 'X0'
    if baseline not in by_arm or set(by_arm[baseline]) != ids:
        raise RuntimeError('incomplete baseline rejected')
    if set(by_arm) != set(config['arms']):
        raise RuntimeError('complete registered control matrix required; missing or unregistered arm')
    if any(set(values) != ids for values in by_arm.values()):
        raise RuntimeError('incomplete candidate rejected; no interim efficacy analysis')
    base = [by_arm[baseline][r['example_id']] for r in expected]
    analysis_role = 'diagnostic' if role == 'confirmatory_diagnostics' else 'confirmatory' if role == 'confirmatory_dev' else 'exploratory'
    comparisons = {name:paired_analysis(base,[values[r['example_id']] for r in expected],
                                       seed=config.get('analysis',{}).get('bootstrap_seed',20260905),
                                       bootstrap_samples=config.get('analysis',{}).get('bootstrap_replicates',10000),
                                       analysis_role=analysis_role,expected_example_ids=[r['example_id'] for r in expected])
                   for name,values in by_arm.items() if name != baseline}
    return {'track':track,'role':role,'n':len(ids),'comparisons':comparisons,
            'status':'COMPLETE', 'registration_sha256':next(iter(source_bindings)),
            'run_dirs':[str(Path(p).resolve()) for p in run_dirs], 'decision':'analysis required under track-specific frozen protocol'}
