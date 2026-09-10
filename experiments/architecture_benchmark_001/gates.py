"""Content-bound, append-only evidence. Hashes detect drift, not malicious forgery."""
from __future__ import annotations
import importlib.metadata
import json
import platform
from pathlib import Path
from .source import ROOT, canonical_hash, sha256_file, source_lock


class GateClosed(ValueError):
    pass


def protocol() -> dict:
    return json.loads((ROOT/'protocol.json').read_text())


def implementation_digest() -> str:
    files = list(ROOT.rglob('*.py')) + [ROOT/'protocol.json', ROOT/'source_lock.json']
    return canonical_hash({str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(files)})


def runtime() -> dict:
    result = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ('torch', 'transformers', 'safetensors'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'not_installed'
    return result


def context(audit: dict, data_path: Path, device: str) -> dict:
    return {'experiment_id': protocol()['experiment_id'], 'implementation_sha256': implementation_digest(),
            'source_audit_digest': audit['audit_digest'], 'weights_sha256': audit['assets']['model.safetensors'],
            'data_sha256': sha256_file(data_path), 'runtime': runtime(), 'device': device,
            'precision': protocol()['baseline_precision'], 'backend': protocol()['attention_backend']}


def write_once(path: Path, data: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(data)
    body['receipt_sha256'] = canonical_hash(body)
    with path.open('x', encoding='utf-8') as f:
        json.dump(body, f, indent=2, allow_nan=False)
        f.write('\n')
    return body


def validate_receipt(path: Path, stage: str, expected_context: dict) -> dict:
    try:
        body = json.loads(path.read_text())
        digest = body.pop('receipt_sha256')
        if digest != canonical_hash(body):
            raise GateClosed('receipt content hash mismatch')
        if (body.get('stage') != stage or body.get('status') != 'PASS'
                or body.get('evidence_kind') != 'official_model_qualification'
                or body.get('context') != expected_context
                or expected_context['weights_sha256'] != source_lock()['assets']['model.safetensors']):
            raise GateClosed('wrong stage, synthetic evidence, changed source/code/data/runtime, or failed receipt')
        if not body.get('checks') or any(value is not True for value in body['checks'].values()):
            raise GateClosed('all declared qualification checks must be actual booleans and pass')
        for name, expected in body.get('artifacts', {}).items():
            artifact = (path.parent/name).resolve()
            if not artifact.is_relative_to(path.parent.resolve()) or sha256_file(artifact) != expected:
                raise GateClosed('missing, changed, or escaping receipt artifact')
        if not body.get('artifacts'):
            raise GateClosed('qualification receipt must bind actual output artifacts')
        body['receipt_sha256'] = digest
        return body
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise GateClosed(f'missing/invalid {stage} evidence: {exc}') from exc


def require_registered_stage(stage: str) -> None:
    # No switch or edited JSON can silently license an unimplemented campaign.
    raise GateClosed(f'{stage} is closed in foundation v1: preregistration and a reviewed source-bound runner are required')
