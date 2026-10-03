"""Verify saved qualification provenance without loading or comparing tensors.

The generation auditor already evaluated numerical gates. This helper checks
that its embedded qualification result still names the same committed receipts
and exact artifact inventories, including unsuccessful preserved attempts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def verify_qualification(result, expected_cell, stage_root, stage_id,
                         expected_sources, expected_config_hash, reader, hasher,
                         committed_checker):
    """Return absolute file paths and hashes for the complete cell inventory.

Terminal/start/report/failure JSON receipts must be committed. All artifacts,
including untracked bulk tensors, are verified by their terminal receipt hashes.
No inference, tensor deserialization, or numerical-threshold replay occurs.
"""
    def check(condition, message):
        if not condition:
            raise ValueError('qualification provenance: ' + message)

    def canonical(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True,
                              separators=(',', ':'), ensure_ascii=False,
                              allow_nan=False).encode()).hexdigest()

    check(isinstance(expected_cell, str) and bool(re.fullmatch(r'[A-Za-z0-9_.]+', expected_cell))
          and expected_cell not in ('.', '..'), 'invalid expected cell')
    check(isinstance(stage_id, str) and bool(re.fullmatch('[0-9a-f]{64}', stage_id)),
          'invalid stage registration')
    root = Path(stage_root).resolve() / 'evidence/01_runtime_qualification' / stage_id / expected_cell
    check(root.resolve() == root, 'aliased qualification root')
    verified = {}

    def safe(path, parent=root):
        path = Path(path)
        check(path.is_absolute(), 'absolute saved paths required')
        check(path.resolve() == path.absolute() and path.resolve().is_relative_to(parent),
              'aliased or foreign qualification path')
        return path.resolve()

    def record(path, *, committed=False):
        path = safe(path)
        check(path.is_file(), 'missing qualification file: ' + str(path))
        digest = hasher(path)
        if committed:
            committed_checker(path)
        check(str(path) not in verified or verified[str(path)] == digest,
              'qualification file changed during verification')
        verified[str(path)] = digest
        return digest

    check(result.get('phase') == 'qualification' and result.get('status') in ('PASS', 'FAILED'),
          'terminal qualification result required')
    directory = safe(result['directory'])
    check(root.is_dir(), 'qualification cell directory missing')
    check({p.name for p in root.iterdir()} <= {'cpu', 'gpu', 'comparison.json'},
          'unknown cell-level qualification artifact')
    phases = {}
    source_hash = canonical(expected_sources)
    for phase in ('cpu', 'gpu'):
        phase_root = safe(root / phase)
        entries = []
        if phase_root.exists():
            check(phase_root.is_dir(), 'phase path is not a directory')
        for attempt in sorted(phase_root.iterdir()) if phase_root.exists() else []:
            attempt = safe(attempt, phase_root)
            check(attempt.is_dir() and bool(re.fullmatch('attempt_[0-9]{3,}', attempt.name)),
                  'unknown phase attempt')
            index = int(attempt.name[8:])
            check(attempt.name == f'attempt_{index:03d}', 'noncanonical phase attempt')
            terminals = [p for p in (attempt / 'run_manifest.json', attempt / 'abandonment.json') if p.exists()]
            check(len(terminals) == 1, 'unfinished or ambiguous phase terminal')
            terminal = safe(terminals[0], attempt)
            terminal_hash = record(terminal, committed=True)
            receipt = reader(terminal)
            start_path = attempt / 'started.json'
            record(start_path, committed=True)
            started = reader(start_path)
            fixed = {'schema_version': 'cross_scale_generation_cell_qualification_phase_v1',
                     'track': 'cross_scale', 'purpose': 'runtime_qualification_only',
                     'registration_sha256': stage_id, 'config_sha256': expected_config_hash,
                     'source_sha256': source_hash, 'sources': expected_sources,
                     'cell_id': expected_cell, 'phase': phase,
                     'device': 'cpu' if phase == 'cpu' else 'cuda:0',
                     'generation_role': 'calibration', 'attempt': index}
            check(started.get('status') == 'STARTED', 'invalid phase start status')
            check(all(type(started.get(k)) is type(v) and started[k] == v
                      and type(receipt.get(k)) is type(v) and receipt[k] == v
                      for k, v in fixed.items()), 'phase contract changed')
            check(all(receipt.get(k) == v for k, v in started.items() if k != 'status'),
                  'terminal rewrote started contract')
            status = receipt.get('status')
            check(status in ('COMPLETE', 'FAILED', 'ABANDONED'), 'invalid phase terminal status')
            check((terminal.name == 'abandonment.json') == (status == 'ABANDONED'),
                  'terminal filename/status disagree')
            artifacts = {}
            for artifact in sorted(attempt.rglob('*')):
                artifact = safe(artifact, attempt)
                if artifact.is_file() and artifact != terminal:
                    name = artifact.relative_to(attempt).as_posix()
                    artifacts[name] = record(artifact)
            check(receipt.get('artifacts_sha256') == artifacts,
                  'phase artifact inventory/hash changed')
            if status == 'ABANDONED':
                check(receipt.get('original_process_absent') is True,
                      'abandonment lacks original-process verification')
            elif status == 'FAILED':
                record(attempt / 'failure.json', committed=True)
                check(receipt.get('failure_class') in
                      ('cell_rejection', 'resource_admission', 'runtime_resource', 'fatal_integrity'),
                      'unknown phase failure class')
            else:
                report_path = attempt / 'report.json'
                check(record(report_path, committed=True) == receipt.get('report_sha256'),
                      'phase report hash changed')
                check(reader(report_path).get('cell_id') == expected_cell,
                      'phase report names another cell')
            check(record(terminal, committed=True) == terminal_hash,
                  'phase terminal changed during verification')
            entries.append({'directory': attempt, 'path': terminal, 'sha256': terminal_hash,
                            'receipt': receipt})
        complete = [e for e in entries if e['receipt']['status'] == 'COMPLETE']
        hard = [e for e in entries if e['receipt']['status'] == 'FAILED'
                and e['receipt'].get('failure_class') == 'cell_rejection']
        check(len(complete) <= 1, 'duplicate complete qualification phase')
        check(len(hard) <= 1 and not (hard and complete),
              'terminal qualification rejection was retried')
        phases[phase] = {'entries': entries, 'complete': complete, 'hard': hard}

    comparison_path = root / 'comparison.json'
    if directory == root:
        check(comparison_path.is_file(), 'comparison receipt missing')
        check(record(comparison_path, committed=True) == result.get('receipt_sha256'),
              'referenced comparison hash changed')
        comparison = reader(comparison_path)
        check(comparison == result.get('receipt'), 'embedded comparison differs from saved receipt')
        fixed = {'status': result['status'], 'phase': 'qualification', 'cell_id': expected_cell,
                 'registration_sha256': stage_id, 'config_sha256': expected_config_hash}
        check(all(type(comparison.get(k)) is type(v) and comparison[k] == v
                  for k, v in fixed.items()), 'comparison contract changed')
        for phase in ('cpu', 'gpu'):
            check(len(phases[phase]['complete']) == 1 and not phases[phase]['hard'],
                  'comparison lacks a complete phase reference')
            check(comparison.get(phase + '_receipt_sha256') == phases[phase]['complete'][0]['sha256'],
                  'comparison phase receipt changed')
        check(type(comparison.get('result', {}).get('pass')) is bool
              and comparison['result']['pass'] == (result['status'] == 'PASS'),
              'recorded comparison outcome disagrees with status')
    else:
        matches = [(phase, entry) for phase, inventory in phases.items()
                   for entry in inventory['entries'] if entry['directory'] == directory]
        check(len(matches) == 1, 'result does not name an exact phase attempt')
        phase, selected = matches[0]
        check(selected['path'].name == 'run_manifest.json'
              and selected['sha256'] == result.get('receipt_sha256'),
              'referenced phase receipt hash changed')
        check(selected['receipt'] == result.get('receipt'),
              'embedded phase receipt differs from saved receipt')
        # This diagnostic is bound to the frozen CUDA screen. Successful cells
        # require a CPU/GPU comparison, whereas a hard phase failure closes one.
        check(result['status'] == 'FAILED' and selected in phases[phase]['hard'],
              'CUDA qualification result lacks comparison or hard phase rejection')
        check(not comparison_path.exists(), 'comparison exists after phase rejection')
        if phase == 'gpu':
            check(len(phases['cpu']['complete']) == 1 and not phases['cpu']['hard'],
                  'GPU rejection lacks its complete CPU reference')
        else:
            check(not phases['gpu']['entries'], 'GPU phase followed CPU rejection')
    return dict(sorted(verified.items()))
