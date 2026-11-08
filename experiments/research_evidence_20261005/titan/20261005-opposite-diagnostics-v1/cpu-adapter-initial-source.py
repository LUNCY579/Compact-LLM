"""Reuse the byte-pinned, measured CPU guardian without editing its reviewed package.

Outputs are fresh external artifacts. The pending native proposal and all forty
reviewed source files remain immutable. This driver has no native entry or permit.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINAL = REPO / 'experiments/titan_quality_repair_v1'
ARTIFACT_ROOT = Path('C:/Users/clclc/BusinessProject/ChatBubble-agent-prompts-20261002/run-artifacts/titan-quality-repair-v1')
REVIEWED = ARTIFACT_ROOT / '20261005-research-v2/NATIVE_QUALIFICATION_REVIEWED.json'
REVIEWED_SHA = '5052bf797cc89af24b767ce51db6738353e523c2addf5139149e56278798c073'


def snapshot():
    return {f'{root.name}/{path.name}': hashlib.sha256(path.read_bytes()).hexdigest()
            for root in (ORIGINAL, HERE) for path in sorted(root.iterdir()) if path.is_file()}


def load_helpers():
    body = REVIEWED.read_bytes()
    if hashlib.sha256(body).hexdigest() != REVIEWED_SHA:
        raise ValueError('STOP_PENDING_NATIVE_PROPOSAL_CHANGED')
    manifest = json.loads(body)['source_manifest']
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ORIGINAL.iterdir() if p.is_file()}
    if actual != manifest:
        raise ValueError('STOP_PENDING_NATIVE_SOURCE_CHANGED')
    path = ORIGINAL / 'offline_supervise.py'
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != manifest[path.name]:
        raise ValueError('STOP_UNBOUND_CPU_GUARDIAN_BYTES')
    name = '_titan_diagnostics_byte_pinned_cpu_helpers'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    # The original supervisor starts its own file as a gate-waiting child. This
    # scoped adapter redirects only that child entry and captures both packages.
    # Original source bytes on disk and all native admission rules stay intact.
    module.__file__ = __file__
    module.HERE, module.REPO = HERE, REPO
    module.source_snapshot = snapshot
    return module


def payload(stage, root, gate, helpers):
    deadline = time.monotonic() + 30
    while not gate.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('STOP_DIAGNOSTICS_WITHOUT_JOB_GATE')
        time.sleep(.02)
    # Job assignment and a one-CPU affinity precede every experiment import.
    sys.meta_path.insert(0, helpers.NoModelImports())
    sys.path.insert(0, str(REPO))
    if stage == 'tests':
        import unittest
        suite = unittest.defaultTestLoader.discover(str(HERE), pattern='test_*.py', top_level_dir=str(REPO))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        helpers.write(root / 'CPU_FIXTURES.json', {
            'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
            'passed': result.wasSuccessful(), 'profile': 'LIGHTWEIGHT_NO_TORCH',
            'model_loads': 0, 'model_forwards': 0, 'capability_evidence': False})
        return 0 if result.wasSuccessful() else 1
    if stage == 'cases':
        from experiments.titan_quality_repair_diagnostics_v1.cases import audit_prepared_inputs
        result = audit_prepared_inputs(ARTIFACT_ROOT / '20261005-offline/packet-002', root)
        from experiments.titan_quality_repair_diagnostics_v1.gain_attribution import analyze_gain_attribution
        helpers.write(root / 'GAIN_ATTRIBUTION_UNMEASURED.json', analyze_gain_attribution())
        return 0 if result['status'] == 'VERIFIED_INPUT_CASE_PATTERN_NO_MODEL_GAIN' else 1
    raise ValueError('STOP_UNKNOWN_DIAGNOSTICS_STAGE')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('tests', 'cases'))
    parser.add_argument('--payload', choices=('tests', 'cases'))
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--gate', type=Path)
    args = parser.parse_args()
    # A fresh prefix prevents reads of pre-existing repository bytecode; -B and
    # inherited PYTHONDONTWRITEBYTECODE also prevent new source bytecode writes.
    os.environ['PYTHONPYCACHEPREFIX'] = str(args.output_dir / 'pycache')
    helpers = load_helpers()
    if args.payload:
        if args.gate is None:
            parser.error('--gate required for the supervised child')
        return payload(args.payload, args.output_dir, args.gate, helpers)
    if args.stage is None:
        parser.error('--stage required')
    return helpers.supervise(args.stage, args.output_dir)


if __name__ == '__main__':
    raise SystemExit(main())
