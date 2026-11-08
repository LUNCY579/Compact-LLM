"""Reuse audited canonical hashing; create evidence exclusively, never replace."""
import json
import os
from pathlib import Path
from . import legacy
from rineya_cross_scale.hashing import canonical_json_bytes, canonical_sha256, sha256_file, bundle_sha256


def write_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def source_files(track=None):
    """Bind transitive implementation, not mutable status/evidence or other track."""
    roots = [legacy.ROOT / 'shared', legacy.ARMR / 'src', legacy.LAB / 'src',
             legacy.REPO / 'rineya_significance_native_architecture_lab/src']
    if track:
        roots += [legacy.ROOT / track / 'src', legacy.ROOT / track / 'scripts']
    files = {p for root in roots for p in root.rglob('*.py')}
    files.add(legacy.ARMR / 'scripts/run_experiment.py')
    files.add(legacy.REPO / 'short-response-adapter-consensus-lab/src/consensus_lab/analysis/statistics.py')
    files.add(legacy.ARMR / 'evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json')
    files.update((legacy.ROOT / 'scripts').glob('*.py'))
    if track:
        files.update((legacy.ROOT / track / 'protocol').rglob('*.md'))
        files.update((legacy.ROOT / track / 'protocol').rglob('*.json'))
    if track == 'cross_scale':
        files.add(legacy.GRAFT / 'graft.py')
        files.add(legacy.LAB / 'cross_scale_evidence/02_architecture/qwen4b.json')
    return {p.relative_to(legacy.REPO).as_posix(): sha256_file(p) for p in sorted(files)}
