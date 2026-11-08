"""Resumable local qualification ladder, without GitHub Actions or background jobs.

Each stage is a foreground child process with isolated memory and an append-only
attempt directory. --resume reuses only current fully validated PASS evidence;
failed/interrupted stages get new attempts. Independent C28/C40 siblings still run
when the other fails. No optimizer executes unless --include-smoke is explicit.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from .gates import (GateClosed, context, data_bindings, read_hashed_json,
                    validate_receipt, write_once)
from .source import ROOT, audit_source


DEPENDENCIES = {'G0': [], 'G1_IDENTITY': ['G0'], 'G1_WIDTH': ['G1_IDENTITY'],
                'G1_CED_C28': ['G0'], 'G1_CED_C40': ['G1_WIDTH'], 'G2': ['G1_WIDTH']}


def commands(source: Path, pilot: Path, calibration: Path, training: Path | None,
             device: str, stage: str, out: Path, parents: dict[str, Path]) -> list[str]:
    common = ['--source', str(source), '--pilot', str(pilot), '--device', device, '--out', str(out)]
    if stage == 'G0': tail = ['qualify-source', *common]
    elif stage == 'G1_IDENTITY': tail = ['qualify-identity', *common, '--g0', str(parents['G0'])]
    elif stage == 'G1_WIDTH':
        tail = ['calibrate-d40', *common, '--g1', str(parents['G1_IDENTITY']), '--calibration', str(calibration)]
    elif stage == 'G1_CED_C28':
        tail = ['convert-ced', *common, '--arm', 'C28', '--g0', str(parents['G0'])]
    elif stage == 'G1_CED_C40':
        tail = ['convert-ced', *common, '--arm', 'C40', '--g1', str(parents['G1_WIDTH']), '--calibration', str(calibration)]
    elif stage == 'G2' and training is not None:
        tail = ['smoke', *common, '--g1', str(parents['G1_WIDTH']), '--calibration', str(calibration), '--train', str(training)]
    else:
        raise GateClosed(f'unknown or incomplete ladder stage: {stage}')
    return [sys.executable, '-m', 'experiments.architecture_benchmark_001', *tail]


def _journal(root: Path, data: dict) -> None:
    record = {'utc': datetime.now(timezone.utc).isoformat(), **data}
    with (root/'JOURNAL.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())


def _launch(argv: list[str], log: Path) -> int:
    # Run from repository root even when called elsewhere; no shell expansion.
    with log.open('x', encoding='utf-8') as stream:
        completed = subprocess.run(argv, cwd=str(ROOT.parents[1]), stdout=stream, stderr=subprocess.STDOUT,
                                   text=True, check=False)
    return completed.returncode


def execute(source: Path, pilot: Path, calibration: Path, training: Path | None,
            out: Path, device: str, *, resume: bool = False, include_smoke: bool = False,
            _launcher=None) -> dict:
    if include_smoke and training is None:
        raise GateClosed('--include-smoke needs --train; the optimizer is never implicit')
    if not include_smoke and training is not None:
        raise GateClosed('--train requires --include-smoke; omit both for qualification only')
    source, pilot, calibration, out = (p.resolve() for p in (source, pilot, calibration, out))
    training = training.resolve() if training else None
    if out.is_relative_to(source) or 'Rineya_Published_Cores' in out.parts:
        raise GateClosed('ladder output must be outside published/source directories')
    audit = audit_source(source)
    bindings = data_bindings(pilot, calibration, training)
    base = context(audit, pilot, device)
    session = {'schema': 2, 'evidence_kind': 'local_ladder_configuration_only',
               'context': base, 'data_bindings': bindings, 'include_smoke': include_smoke,
               'licenses_registered_training': False, 'licenses_efficacy_claim': False}
    if out.exists():
        if not resume:
            raise GateClosed('output already exists; use --resume only for this exact session')
        prior = read_hashed_json(out/'RUN.json')
        if {k: v for k,v in prior.items() if k != 'receipt_sha256'} != session:
            raise GateClosed('session code/runtime/data changed; use a new output root, not an old PASS')
    else:
        if resume:
            raise GateClosed('cannot resume a missing session')
        out.mkdir(parents=True, exist_ok=False)
        write_once(out/'RUN.json', session)
    lock = out/'ACTIVE.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise GateClosed('session locked; verify no process is running before removing a stale ACTIVE.lock') from exc
    with os.fdopen(fd, 'w') as stream: stream.write(str(os.getpid()))
    stages = list(DEPENDENCIES) if include_smoke else [s for s in DEPENDENCIES if s != 'G2']
    parents, results = {}, []
    launcher = _launcher or _launch
    try:
        for stage in stages:
            missing = [s for s in DEPENDENCIES[stage] if s not in parents]
            if missing:
                record = {'stage': stage, 'status': 'BLOCKED', 'missing_passing_parents': missing}
                results.append(record); _journal(out, record); continue
            folder = out/stage; folder.mkdir(exist_ok=True)
            attempts = sorted(p for p in folder.glob('attempt-*') if p.is_dir())
            expected = {**base, 'data_sha256': bindings[stage]}
            reused = None
            for attempt in reversed(attempts):
                receipt = attempt/f'{stage}.json'
                if not receipt.exists():
                    continue  # Interrupted attempt remains intact, not clean evidence.
                body = read_hashed_json(receipt)
                if body.get('status') != 'PASS':
                    continue
                # A purported PASS whose code/artifacts/ancestors changed is not silently
                # ignored or repaired: reject the session and request a new root.
                validate_receipt(receipt, stage, expected, data_by_stage=bindings)
                reused = receipt; break
            if reused is not None:
                parents[stage] = reused
                record = {'stage': stage, 'status': 'PASS', 'reused': True, 'receipt': reused.relative_to(out).as_posix()}
                results.append(record); _journal(out, record); continue
            next_number = max([int(p.name.split('-')[-1]) for p in attempts] + [0])+1
            attempt = folder/f'attempt-{next_number:04d}'
            argv = commands(source, pilot, calibration, training, device, stage, attempt, parents)
            log = folder/f'attempt-{next_number:04d}.log'
            _journal(out, {'stage': stage, 'status': 'STARTED', 'attempt': attempt.relative_to(out).as_posix(), 'argv': argv})
            print(f'{stage}: starting {attempt.name}', flush=True)
            code = launcher(argv, log)
            receipt = attempt/f'{stage}.json'
            record = {'stage': stage, 'status': 'FAIL', 'exit_code': code, 'reused': False,
                      'attempt': attempt.relative_to(out).as_posix(), 'log': log.relative_to(out).as_posix()}
            if code == 0:
                try:
                    validate_receipt(receipt, stage, expected, data_by_stage=bindings)
                except (ValueError, OSError) as exc:
                    record['reason'] = f'zero exit is insufficient: {exc}'
                else:
                    record.update(status='PASS', receipt=receipt.relative_to(out).as_posix())
                    parents[stage] = receipt
            else:
                record['reason'] = 'child failed; log or failure receipt is not a clean review'
            results.append(record); _journal(out, record)
            print(f'{stage}: {record["status"]}', flush=True)
        report = {'schema': 2, 'evidence_kind': 'local_ladder_summary_only', 'expected_stages': stages,
                  'status': 'PASS' if len(results)==len(stages) and all(r['status']=='PASS' for r in results) else 'FAIL',
                  'results': results, 'licenses_registered_training': False, 'licenses_efficacy_claim': False}
        number = len(list(out.glob('LADDER-*.json')))+1
        return write_once(out/f'LADDER-{number:04d}.json', report)
    except BaseException as exc:
        _journal(out, {'status': 'INTERRUPTED_OR_ERROR', 'reason': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pilot',type=Path,default=ROOT/'data/pilot.jsonl')
    p.add_argument('--calibration',type=Path,required=True); p.add_argument('--train',type=Path)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--resume',action='store_true'); p.add_argument('--include-smoke',action='store_true')
    a=p.parse_args(argv)
    try:
        report=execute(a.source,a.pilot,a.calibration,a.train,a.out,a.device,resume=a.resume,include_smoke=a.include_smoke)
        print(json.dumps(report,indent=2)); return 0 if report['status']=='PASS' else 2
    except (ValueError,OSError,RuntimeError) as exc:
        print(json.dumps({'status':'BLOCKED_OR_FAILED','reason':str(exc),'licenses_training':False})); return 2


if __name__=='__main__': raise SystemExit(main())
