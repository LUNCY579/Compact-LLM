"""Content-bound evidence with recursive parent validation; not signed attestations.

Schema-1 receipts are preserved as historical evidence, not upgraded in place.
Every schema-2 official stage binds the complete current implementation/runtime and
requires its declared parent DAG, artifacts, and mandatory check names.
"""
from __future__ import annotations
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import torch
from .source import ROOT, canonical_hash, sha256_file, source_lock


class GateClosed(ValueError):
    pass


def protocol() -> dict:
    return json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))


def implementation_manifest() -> dict[str, str]:
    files = list(ROOT.rglob('*.py')) + [ROOT/'protocol.json', ROOT/'source_lock.json',
                                       ROOT/'requirements.txt', ROOT/'requirements-native.txt']
    for name in protocol().get('amendments', []):
        amendment = ROOT/name
        if not amendment.resolve().is_relative_to(ROOT.resolve()) or not amendment.is_file():
            raise GateClosed('missing or escaping protocol amendment')
        files.append(amendment)
    return {p.relative_to(ROOT).as_posix(): sha256_file(p)
            for p in sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())}


def implementation_digest() -> str:
    return canonical_hash(implementation_manifest())


def runtime() -> dict:
    result = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ('torch', 'transformers', 'safetensors', 'numpy', 'tokenizers'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'not_installed'
    result['numerics'] = {
        'cpu_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cuda_build': torch.version.cuda,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
    }
    return result


def context(audit: dict, data_path: Path, device: str) -> dict:
    identity = {'device': device}
    if torch.device(device).type == 'cuda':
        if not torch.cuda.is_available():
            raise GateClosed('CUDA unavailable; no device fallback')
        identity.update(name=torch.cuda.get_device_name(device),
                        capability=list(torch.cuda.get_device_capability(device)))
    return {'context_schema': 2, 'experiment_id': protocol()['experiment_id'],
            'implementation_sha256': implementation_digest(),
            'source_audit_digest': audit['audit_digest'],
            'weights_sha256': audit['assets']['model.safetensors'],
            'data_sha256': sha256_file(data_path), 'runtime': runtime(), 'device': device,
            'device_identity': identity, 'precision': protocol()['baseline_precision'],
            'backend': protocol()['attention_backend']}


def write_once(path: Path, data: dict) -> dict:
    """Encode before creating the destination; never truncate an existing receipt."""
    if 'receipt_sha256' in data:
        raise GateClosed('do not re-sign or relabel an existing receipt')
    body = dict(data)
    body['receipt_sha256'] = canonical_hash(body)
    encoded = json.dumps(body, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())
    return body


# The first parent is the data/model donor. Additional parents are matched controls.
STAGES = {
    'G0': {'parents': [], 'artifacts': ['predictions.json'], 'checks': [
        'official_source_unchanged', 'native_reference_logits', 'native_reference_finite',
        'parser_usable', 'termination_usable', 'diagnostic_not_floor_or_ceiling',
        'registered_context_executed', 'unmodified_native_readout']},
    'G1_IDENTITY': {'parents': ['G0'], 'artifacts': ['lineage.json'], 'checks': [
        'identity_logits_exact', 'chunked_cache_parity', 'source_unchanged',
        'retained_original_layer_order', 'exact_parameter_count']},
    'G1_WIDTH': {'parents': ['G1_IDENTITY'],
        'artifacts': ['lineage.json', 'calibration_scores.json', 'config.json', 'model.safetensors'],
        'checks': ['registered_dimensions', 'all_tensors_accounted', 'finite_converted_logits',
                   'save_reload_exact', 'source_unchanged', 'calibration_unchanged', 'exact_parameter_count']},
    'G2': {'parents': ['G1_WIDTH'], 'artifacts': ['config.json', 'model.safetensors'],
        'checks': ['25_steps_completed', 'all_intended_parameters_reached', 'save_reload_exact',
                   'source_unchanged', 'training_data_unchanged']},
}
for arm, parent in [('C28', 'G0'), ('C40', 'G1_WIDTH')]:
    STAGES[f'G1_CED_{arm}'] = {
        'parents': [parent], 'artifacts': ['lineage.json', 'config.json', 'model.safetensors', 'numerics.json'],
        'checks': ['registered_dimensions', 'exact_parameter_count', 'registered_donor_layer',
                   'shared_encoder_final_memory', 'decoder_local_attention_absent',
                   'finite_converted_logits', 'save_reload_exact', 'strict_tail_only_parity',
                   'strict_chunked_cache_parity', 'causal_prefix_parity', 'decode_choices_exact',
                   'donor_state_unmodified', 'source_unchanged', 'pilot_unchanged', 'calibration_unchanged']}
# Output-quality gate. Deliberately separate from construction: an arm can be structurally
# perfect and still emit token salad, which every other stage accepts.
for arm, parent in [('B28', 'G0'), ('D40', 'G1_WIDTH'),
                    ('C28', 'G1_CED_C28'), ('C40', 'G1_CED_C40')]:
    STAGES[f'G1_QUALITY_{arm}'] = {
        'parents': [parent], 'artifacts': ['quality.json'],
        'checks': ['ce_increase_within_margin', 'kl_to_source_within_margin',
                   'top1_agreement_above_floor', 'generation_terminates',
                   'generation_not_degenerate', 'source_unchanged',
                   'calibration_unchanged', 'pilot_unchanged']}


for arm, parents in [('B28', ['G0']), ('D40', ['G1_WIDTH']),
                     ('C28', ['G1_CED_C28', 'G2_CACHED_B28']),
                     ('C40', ['G1_CED_C40', 'G2_CACHED_D40'])]:
    STAGES[f'G2_CACHED_{arm}'] = {
        'parents': parents, 'artifacts': ['config.json', 'model.safetensors', 'teacher_cache_manifest.json'],
        'checks': ['25_steps_completed', 'all_intended_parameters_reached', 'teacher_frozen',
                   'save_reload_exact', 'source_unchanged', 'training_data_unchanged',
                   'teacher_cache_unchanged', 'matched_control_verified', 'optimizer_steps_changed_state']}


def read_hashed_json(path: Path) -> dict:
    """Check only byte-independent JSON content hash; NOT stage authorization."""
    try:
        body = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(body, dict):
            raise GateClosed('receipt must be an object')
        digest = body.get('receipt_sha256')
        payload = {k: v for k, v in body.items() if k != 'receipt_sha256'}
        if not isinstance(digest, str) or digest != canonical_hash(payload):
            raise GateClosed('receipt content hash mismatch')
        return body
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise GateClosed(f'missing/invalid evidence at {path}: {exc}') from exc


def validate_artifacts(path: Path, artifacts: dict) -> None:
    if not isinstance(artifacts, dict) or not artifacts:
        raise GateClosed('evidence must bind actual output artifacts')
    for name, expected in artifacts.items():
        if not isinstance(name, str) or '\\' in name:
            raise GateClosed('artifact names must be relative POSIX paths')
        artifact = path.parent/name
        resolved = artifact.resolve()
        if (artifact.is_symlink() or not resolved.is_relative_to(path.parent.resolve())
                or not resolved.is_file() or sha256_file(resolved) != expected):
            raise GateClosed(f'missing, changed, or escaping receipt artifact: {name}')


def validate_receipt(path: Path, stage: str, expected_context: dict, *,
                     data_by_stage: dict[str, str] | None = None,
                     _seen: frozenset[Path] = frozenset()) -> dict:
    """Validate the whole donor/control DAG, not just a parent hash-shaped string.

    data_by_stage binds ancestor inputs to the *current* supplied pilot/calibration/train
    files. Parent contexts may differ only in data_sha256; all runtime/code/source fields
    must match. Old schema-1 observations can be inspected with audit-evidence, never reused.
    """
    try:
        path = path.resolve()
        if path in _seen or len(_seen) > 32:
            raise GateClosed('cyclic or excessively deep receipt ancestry')
        spec = STAGES.get(stage)
        if spec is None:
            raise GateClosed(f'unregistered stage: {stage}')
        body = read_hashed_json(path)
        if (body.get('schema') != 2 or body.get('stage') != stage or body.get('status') != 'PASS'
                or body.get('evidence_kind') != 'official_model_qualification'
                or body.get('context') != expected_context
                or expected_context.get('context_schema') != 2
                or expected_context.get('implementation_sha256') != implementation_digest()
                or expected_context.get('weights_sha256') != source_lock()['assets']['model.safetensors']
                or body.get('licenses_registered_training') is not False
                or body.get('licenses_efficacy_claim') is not False):
            raise GateClosed('wrong stage/schema, stale code/context, synthetic evidence, or failed receipt')
        if data_by_stage and stage in data_by_stage and body['context']['data_sha256'] != data_by_stage[stage]:
            raise GateClosed(f'current input data differs from {stage} evidence')
        checks = body.get('checks')
        if (not isinstance(checks, dict) or not set(spec['checks']).issubset(checks)
                or any(value is not True for value in checks.values())):
            raise GateClosed('mandatory named qualification checks must be booleans and all pass')
        if not set(spec['artifacts']).issubset(body.get('artifacts', {})):
            raise GateClosed('mandatory stage artifact missing from manifest')
        validate_artifacts(path, body['artifacts'])
        parents = body.get('parent_receipts')
        if not isinstance(parents, list) or [p.get('stage') for p in parents] != spec['parents']:
            raise GateClosed('missing or wrong-stage parent receipt DAG')
        want_first = parents[0]['receipt_sha256'] if parents else None
        if body.get('parent_receipt_sha256') != want_first:
            raise GateClosed('legacy parent digest disagrees with declared ancestry')
        for link in parents:
            parent_path = (path.parent/link['path']).resolve()
            parent_body = read_hashed_json(parent_path)
            if parent_body['receipt_sha256'] != link['receipt_sha256']:
                raise GateClosed('parent receipt digest mismatch')
            parent_context = dict(expected_context)
            parent_context['data_sha256'] = (data_by_stage or {}).get(
                link['stage'], parent_body['context']['data_sha256'])
            validate_receipt(parent_path, link['stage'], parent_context,
                             data_by_stage=data_by_stage, _seen=_seen | {path})
        return body
    except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise GateClosed(f'missing/invalid {stage} evidence: {exc}') from exc


def parent_links(folder: Path, paths: list[Path]) -> list[dict]:
    result = []
    for path in paths:
        body = read_hashed_json(path)
        result.append({'stage': body['stage'], 'receipt_sha256': body['receipt_sha256'],
                       'path': Path(os.path.relpath(path.resolve(), folder.resolve())).as_posix()})
    return result


def data_bindings(pilot: Path, calibration: Path | None = None, training: Path | None = None) -> dict:
    result = {stage: sha256_file(pilot) for stage in ('G0', 'G1_IDENTITY', 'G1_CED_C28', 'G1_CED_C40')}
    if calibration is not None:
        result['G1_WIDTH'] = sha256_file(calibration)
        # Quality is scored on calibration; the pilot split is used only for the
        # nondegeneracy probe and is bound separately inside the stage.
        for arm in ('B28', 'D40', 'C28', 'C40'):
            result[f'G1_QUALITY_{arm}'] = result['G1_WIDTH']
    if training is not None:
        for stage in ('G2', *(f'G2_CACHED_{a}' for a in ('B28', 'D40', 'C28', 'C40'))):
            result[stage] = sha256_file(training)
    return result


def require_registered_stage(stage: str) -> None:
    raise GateClosed(f'{stage} remains closed: preregistration and a separately reviewed recovery/evaluation runner are required')
