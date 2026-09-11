"""Read-only historical evidence audit. Never upgrades an old receipt or opens gates."""
from __future__ import annotations
import json
from pathlib import Path
from .gates import GateClosed, implementation_digest, read_hashed_json
from .source import sha256_file

# Named strict-parity checks emitted by the schema-2 CED writer. Schema-1 CED receipts
# instead carried per-path counts under metrics.cache_order_parity, so an audit that reads
# only the legacy key reports zero failures for every schema-2 FAIL receipt.
STRICT_PARITY_CHECKS = ('strict_tail_only_parity', 'strict_chunked_cache_parity',
                        'causal_prefix_parity', 'decode_choices_exact')


def _legacy_parity_failures(body: dict) -> dict:
    """Per-path violation counts as recorded by schema-1 CED receipts."""
    parity = body.get('metrics', {}).get('cache_order_parity', {})
    if not isinstance(parity, dict):
        return {}
    return {name: value['elementwise_g0_violations'] for name, value in parity.items()
            if isinstance(value, dict) and value.get('elementwise_g0_violations', 0) > 0}


def _numerics_violation_totals(path: Path, body: dict) -> dict:
    """Per-path violation totals from a schema-2 receipt's own numerics.json artifact.

    Read only when the artifact is declared and its recorded hash still matches, so an
    unbound or edited sidecar never contributes numbers to this audit.
    """
    expected = body.get('artifacts', {}).get('numerics.json')
    item = path.parent/'numerics.json'
    resolved = item.resolve()
    if (not isinstance(expected, str) or item.is_symlink()
            or not resolved.is_relative_to(path.parent.resolve())
            or not resolved.is_file() or sha256_file(resolved) != expected):
        return {}
    try:
        data = json.loads(resolved.read_text(encoding='utf-8'))
    except (ValueError, OSError):
        return {}
    totals: dict[str, int] = {}
    for row in data.get('rows', []) if isinstance(data.get('rows'), list) else []:
        paths = row.get('paths', {}) if isinstance(row, dict) else {}
        if not isinstance(paths, dict):
            continue
        chunks = paths.get('chunks') if isinstance(paths.get('chunks'), dict) else {}
        entries = [('tail_only', paths.get('tail_only')), ('prefix', paths.get('prefix'))]
        entries += [(f'chunk_{label}', value) for label, value in chunks.items()]
        for label, entry in entries:
            count = entry.get('elementwise_violations', 0) if isinstance(entry, dict) else 0
            if isinstance(count, int) and count > 0:
                totals[label] = totals.get(label, 0) + count
    return totals


def audit_evidence(root: Path) -> dict:
    if not root.is_dir():
        raise GateClosed('evidence directory is missing')
    records, by_hash = [], {}
    for path in sorted(root.rglob('G*.json')):
        try:
            body = read_hashed_json(path)
        except ValueError as exc:
            records.append({'path': path.relative_to(root).as_posix(), 'integrity': 'INVALID', 'reason': str(exc)})
            continue
        artifacts = {}
        for name, expected in body.get('artifacts', {}).items():
            item = (path.parent/name).resolve()
            if not item.is_relative_to(path.parent.resolve()):
                status = 'ESCAPING_PATH'
            elif not item.is_file():
                status = 'MISSING'
            else:
                status = 'MATCH' if sha256_file(item) == expected else 'HASH_MISMATCH'
            artifacts[name] = status
        failures = _legacy_parity_failures(body)
        checks = body.get('checks', {}) if isinstance(body.get('checks'), dict) else {}
        failed_checks = sorted(name for name in STRICT_PARITY_CHECKS if checks.get(name) is False)
        violation_totals = _numerics_violation_totals(path, body) if failed_checks else {}
        record = {'path': path.relative_to(root).as_posix(), 'integrity': 'JSON_HASH_MATCH',
                  'stage': body.get('stage'), 'declared_status': body.get('status'),
                  'receipt_sha256': body['receipt_sha256'], 'parent_receipt_sha256': body.get('parent_receipt_sha256'),
                  'schema': body.get('schema'), 'artifacts': artifacts,
                  'implementation_matches_current': body.get('context', {}).get('implementation_sha256') == implementation_digest(),
                  'recorded_strict_parity_failures': failures,
                  'failed_strict_parity_checks': failed_checks,
                  'strict_parity_violation_totals': violation_totals,
                  'any_recorded_strict_parity_failure': bool(failures or failed_checks),
                  'reusable_as_current_qualification': False}
        records.append(record)
        by_hash[body['receipt_sha256']] = record
    for row in records:
        parent = row.get('parent_receipt_sha256')
        row['parent_hash_present_in_scan'] = parent is None or parent in by_hash
    return {'schema': 2, 'evidence_kind': 'historical_evidence_audit_only', 'records': records,
            'status': 'AUDIT_COMPLETE' if records else 'NO_RECEIPTS_FOUND',
            'licenses_registered_training': False, 'licenses_efficacy_claim': False,
            'warning': 'Historical PASS is a recorded claim, not new execution. Missing weights prevent complete artifact verification. '
                       'Argmax/global-scale PASS does not erase recorded elementwise failures; causal explanations are unproven. '
                       'recorded_strict_parity_failures covers schema-1 receipts only; failed_strict_parity_checks and '
                       'strict_parity_violation_totals cover schema-2. Read any_recorded_strict_parity_failure for either generation.'}
