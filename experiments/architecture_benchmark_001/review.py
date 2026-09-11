"""Read-only historical evidence audit. Never upgrades an old receipt or opens gates."""
from __future__ import annotations
from pathlib import Path
from .gates import GateClosed, implementation_digest, read_hashed_json
from .source import sha256_file


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
        parity = body.get('metrics', {}).get('cache_order_parity', {})
        failures = {name: value['elementwise_g0_violations'] for name, value in parity.items()
                    if value.get('elementwise_g0_violations', 0) > 0}
        record = {'path': path.relative_to(root).as_posix(), 'integrity': 'JSON_HASH_MATCH',
                  'stage': body.get('stage'), 'declared_status': body.get('status'),
                  'receipt_sha256': body['receipt_sha256'], 'parent_receipt_sha256': body.get('parent_receipt_sha256'),
                  'schema': body.get('schema'), 'artifacts': artifacts,
                  'implementation_matches_current': body.get('context', {}).get('implementation_sha256') == implementation_digest(),
                  'recorded_strict_parity_failures': failures,
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
                       'Argmax/global-scale PASS does not erase recorded elementwise failures; causal explanations are unproven.'}
