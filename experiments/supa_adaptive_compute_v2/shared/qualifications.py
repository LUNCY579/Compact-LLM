"""Admission receipts for prospectively registered CPU-to-GPU qualification."""
import json
import math
from pathlib import Path
from .legacy import ROOT, REPO
from .hashing import canonical_sha256,sha256_file,source_files
from .guards import require_within,require_track
from .manifests import verify_registration,_require_committed
from .runtime import configure_runtime,runtime_identity


def validate_native_qualification_report(report,config,calibration_rows):
    """Recompute coverage/gates from the numerical report, ignoring claimed pass."""
    families = sorted({r['family'] for r in calibration_rows})
    chosen = [min((r for r in calibration_rows if r['family']==family),key=lambda r:r['example_id']) for family in families]
    ids = [r['example_id'] for r in chosen]
    if len(ids) != 8 or report.get('fixture_ids') != ids or report.get('thresholds') != config['qualification']:
        raise RuntimeError('native GPU qualification fixture/threshold mismatch')
    if report.get('fixture_prompt_sha256') != {r['example_id']:r['prompt_sha256'] for r in chosen}:
        raise RuntimeError('native GPU qualification prompt identity mismatch')
    rows = report.get('comparisons',[])
    expected = {(arm,identity) for arm in config['arms'] for identity in ids}
    keys = [(row.get('arm'),row.get('example_id')) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError('native GPU qualification comparison matrix incomplete')
    q = config['qualification']
    def finite(value):
        return isinstance(value,(int,float)) and math.isfinite(value)
    def metric(value,relative,cosine,absolute=None):
        return (all(finite(value.get(k)) for k in ('max_abs','relative_l2','cosine')) and
                0 <= value['relative_l2'] <= relative and cosine <= value['cosine'] <= 1.000000000001 and
                value['max_abs'] >= 0 and (absolute is None or value['max_abs'] <= absolute))
    for row in rows:
        if not metric(row.get('logits',{}),q['max_relative_logit_l2'],q['minimum_logit_cosine'],q['max_abs_logit_error']):
            raise RuntimeError('native CPU/GPU logit agreement gate failed')
        hidden = row.get('hidden',{})
        if set(hidden) != {'8','12','13','16','19'} or not all(metric(v,q['max_relative_hidden_l2'],q['minimum_hidden_cosine']) for v in hidden.values()):
            raise RuntimeError('native CPU/GPU hidden agreement gate failed')
        error = row.get('absolute_nll_error')
        if not finite(error) or not 0 <= error <= q['max_abs_nll_error']:
            raise RuntimeError('native CPU/GPU NLL gate failed')
        if not row.get('cpu_token_ids') or row['cpu_token_ids'] != row.get('gpu_token_ids') or row.get('gpu_repeat_identity') is not True:
            raise RuntimeError('native CPU/GPU behavior or repeat identity gate failed')
    left,right = report.get('cpu_identity',{}),report.get('gpu_identity',{})
    for value in (left,right):
        if value.get('model_class') != 'Qwen3AbstainForCausalLM' or value.get('canonical_unchanged') is not True or value.get('trainable_parameters') != 0 or value.get('backbone_gradients') != 0:
            raise RuntimeError('native qualification frozen canonical identity failed')
        if not value.get('state_sha256_before') or value['state_sha256_before'] != value.get('state_sha256_after') or value.get('checkpoint_files_before') != value.get('checkpoint_files_after'):
            raise RuntimeError('native qualification model state changed')
        if value.get('checkpoint_files_before',{}).get('model.safetensors') != config['checkpoint_sha256']:
            raise RuntimeError('native qualification checkpoint mismatch')
    for key in ('state_sha256_before','checkpoint_files_before','operator_hashes'):
        if left.get(key) != right.get(key):
            raise RuntimeError('native CPU/GPU weight identity mismatch')
    if set(left.get('operator_hashes',{})) != {'AM'} or not left['operator_hashes']['AM'] or report.get('same_device_disabled_restored_identity') is not True:
        raise RuntimeError('native qualification control/restoration identity absent')
    return True


def verify_runtime_qualification(config,dataset_manifest,registration=None):
    if config.get('device') == 'cpu':
        return {'status':'CPU_REFERENCE','receipt_sha256':None,'directory':None}
    track = require_track(config['track'])
    if registration is None:
        name = 'protocol_gpu_v2_2.json' if track == 'native_armr' else 'alignment_gpu_v2_1.json'
        registration = verify_registration(track,ROOT/track/'configs'/name)
    dataset_manifest = require_within(dataset_manifest,ROOT/'data')
    if registration['config'] != config or registration['sources'] != source_files(track):
        raise RuntimeError('GPU admission source/config registration mismatch')
    dataset_hash = sha256_file(dataset_manifest)
    if registration['dataset_manifests'].get(dataset_manifest.relative_to(REPO).as_posix()) != dataset_hash:
        raise RuntimeError('GPU qualification dataset not registered for this track')
    configure_runtime(config)
    actual_runtime = runtime_identity(config)
    matches = []
    for path in (ROOT/track/'evidence/00_runtime_qualification').glob('*/run_manifest.json'):
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('registration_sha256') != registration['registration_sha256']:
            continue
        if value.get('status') != 'COMPLETE' or value.get('qualification_pass') is not True:
            continue
        if (value.get('runtime_qualification') != config['runtime_qualification'] or
            value.get('config_sha256') != canonical_sha256(config) or value.get('track') != track or
            value.get('dataset_manifest_sha256') != dataset_hash or
            value.get('sources') != registration['sources'] or value.get('runtime_identity') != actual_runtime or
            value.get('canonical_unchanged') is not True):
            raise RuntimeError('GPU qualification receipt mismatches source, data, runtime or integrity')
        report = path.parent/'report.json'
        if sha256_file(report) != value.get('report_sha256'):
            raise RuntimeError('GPU qualification report hash mismatch')
        report_value = json.loads(report.read_text(encoding='utf-8'))
        if report_value.get('qualification_pass') is not True:
            raise RuntimeError('GPU qualification report did not pass')
        from .datasets import load_frozen_dataset
        calibration = load_frozen_dataset(dataset_manifest,track=track,role='calibration')
        if track == 'native_armr':
            validate_native_qualification_report(report_value,config,calibration)
        else:
            from cross_scale.src.gpu_qualification import validate_qualification_report
            validate_qualification_report(report_value,config,calibration)
        for relative,expected_hash in report_value.get('artifacts',{}).items():
            artifact = require_within(path.parent/relative,path.parent)
            if sha256_file(artifact) != expected_hash:
                raise RuntimeError('qualification tensor artifact changed')
        _require_committed(path)
        _require_committed(report)
        matches.append({'status':'QUALIFIED','receipt_sha256':sha256_file(path),
                        'directory':str(path.parent),'receipt':value})
    if len(matches) != 1:
        raise RuntimeError('exactly one committed passing GPU qualification receipt is required')
    return matches[0]
