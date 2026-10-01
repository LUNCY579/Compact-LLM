"""S0 churn floor, stage S0-E (COMP-RS-001 Amendment 1, section 16 'Enrollment and churn'; section 8.3).

python -m experiments.maxrl_comp002_pretraining_v1.s0_churn --run1 RECORDS\\S0_ENROLL_RUN1.json \
    --run2 RECORDS\\S0_ENROLL_RUN2.json --runtime-freeze RECORDS\\S0_RUNTIME_FREEZE.json [--out RECORDS\\S0_CHURN.json]

Churn is the number of rows whose generated token IDs differ between enrollment run 1 and run 2. It must be 0;
otherwise STOP_NONDETERMINISTIC, because churn enters every discordance count directly (section 8.3). This stage
reads two finished s0_enroll receipts and their journals and decodes nothing.

Comparability checks, all made before anything is counted:
- each receipt is a complete S0_ENROLL_RUN<n> receipt of this schema;
- each journal's bytes hash to the sha its receipt recorded, end with LF (no partial line), and re-verify line by
  line (s0_generate.ResumableJsonl: canonical bytes, row hash, unique ids, bind fields);
- the row count equals the receipt's;
- the two binds are equal except stage and run, so arm, weights, decoding_sha256, generation contract, plan,
  tokenizer receipt and code bundle all match;
- both journals hold the plan's ids in the same order, and each row has the same prompt_hash and rendered prompt
  IDs in both runs.
Process separation (section 16: run 2 is 'a separate process'): no launch session appears in both runs. When
registered, no OS process (pid + creation time) appears in both either; test mode records any overlap, because its
runs share the pytest process.
Consistency: where the token IDs agree, the stop reason and the response text must agree too (the response rule is
a function of the IDs). With zero churn, the runs' strict scores must agree as well. Otherwise the stage raises.
Registered mode also asserts the runtime freeze: its decoding_sha256 and scorer bundle equal both runs', and both
runs' bound anchor is the one it names (s0_competence.anchor_matches; section 18). The receipt S0_CHURN is
create-only; for an anchored (theta0') freeze its registered name is S0_CHURN_WARM.json. Exit status: 0 for
CHURN_ZERO, 2 for STOP_NONDETERMINISTIC.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import sys
import time

from . import s0_competence as comp
from . import s0_enroll as enroll
from . import s0_runtime as s0
from .preflight import digest, require

SCHEMA = 'COMP-RS-001-S0-CHURN-1'
STAGE = 'S0_CHURN'
REGISTRATION = 'COMP_RS_001_AMENDMENT_1.md section 16, enrollment and churn (S0-E); section 8.3 churn floor'
BIND_EXEMPT = ('stage', 'run')     # the only bind fields allowed to differ between run 1 and run 2
CODE_FILES = comp.code_files(comp.PACKAGE + 's0_enroll.py', comp.PACKAGE + 's0_churn.py')
_IMPORT_CODE = comp.code_sha256(CODE_FILES)


def _enroll_receipt(path, run: int, registered: bool) -> dict:
    receipt, receipt_sha = comp.json_file(path, 'S0_ENROLL_RUN%d receipt' % run)
    require(receipt.get('stage') == enroll.stage_name(run) and receipt.get('schema') == enroll.SCHEMA
            and receipt.get('run') == run, 'not an S0_ENROLL_RUN%d receipt' % run)
    require(receipt.get('completed') is True, 'enrollment run %d is not complete' % run)
    require(receipt.get('registered_run') is registered and receipt.get('test_mode') is (not registered),
            'enrollment run %d is not a %s run' % (run, 'registered' if registered else 'test-mode'))
    return {'receipt': receipt, 'path': str(Path(path).resolve()), 'sha256': receipt_sha}


def _journal_rows(entry: dict, run: int) -> list:
    """The run's journal, verified against its receipt without writing anything."""
    g = comp.generate_module()
    record = entry['receipt']['journal']
    path = Path(record['path'])
    require(path.is_file(), 'run %d journal missing: %s' % (run, path))
    raw = path.read_bytes()
    require(comp._sha256(raw) == record['sha256'] and len(raw) == record['bytes'],
            'run %d journal bytes differ from its receipt' % run)
    require(raw.endswith(b'\n'), 'run %d journal has a partial last line' % run)
    bind = record['bind']
    require(type(bind) is dict and bind.get('run') == run and bind.get('stage') == enroll.stage_name(run),
            'run %d journal bind' % run)
    rows = g.ResumableJsonl(path, key='id', bind=bind).rows()
    require(comp._sha256(path.read_bytes()) == record['sha256'], 'run %d journal changed while read' % run)
    require(len(rows) == record['rows'] == entry['receipt']['inputs']['planned'],
            'run %d journal row count differs from its receipt' % run)
    return rows


def _processes(rows: list) -> tuple:
    sessions = {row['process']['session'] for row in rows}
    processes = {(row['process']['pid'], row['process']['process_created']) for row in rows}
    return sessions, processes


def _first_difference(a: list, b: list):
    for position, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return position
    return None if len(a) == len(b) else min(len(a), len(b))


def _strict_ids(receipt: dict, key: str) -> list:
    return sorted(receipt[key]['scoring']['strict_correct_ids'])


def compare_runs(run1, run2, *, registered: bool, freeze: dict | None = None) -> dict:
    """Churn between two complete enrollment runs (paths to their receipts); every comparability check first."""
    first, second = _enroll_receipt(run1, 1, registered), _enroll_receipt(run2, 2, registered)
    rows1, rows2 = _journal_rows(first, 1), _journal_rows(second, 2)
    bind1, bind2 = first['receipt']['journal']['bind'], second['receipt']['journal']['bind']
    require(set(bind1) == set(bind2), 'the runs bind different fields')
    require(bind1.get('test_mode') is (not registered), 'the journals were not written in the %s mode'
            % ('registered' if registered else 'test'))
    differing = sorted(key for key in bind1 if key not in BIND_EXEMPT and bind1[key] != bind2[key])
    require(not differing, 'the runs are not comparable; bind fields differ: ' + ', '.join(differing))
    ids1, ids2 = [row['id'] for row in rows1], [row['id'] for row in rows2]
    require(ids1 == ids2 and digest(ids1) == bind1['plan_sha256'], 'the runs did not read the same plan in order')
    for a, b in zip(rows1, rows2):
        require(a['prompt_hash'] == b['prompt_hash'] and a['prompt_ids_sha256'] == b['prompt_ids_sha256']
                and a['bank'] == b['bank'], 'the runs rendered different prompts for ' + a['id'])
    if freeze is not None:
        require(freeze['decoding_sha256'] == bind1['decoding_sha256'],
                'decoding_sha256 differs from the runtime freeze')
        require(all(comp.anchor_matches(bind.get('anchor'), freeze.get('anchor')) for bind in (bind1, bind2)),
                'a run read another anchor than the runtime freeze names')
        for entry in (first, second):
            require(entry['receipt']['scorer_bundle']['expected'] == freeze['scoring_code_bundle_sha256'],
                    'a run scored with another scorer bundle than the runtime freeze')
    sessions1, processes1 = _processes(rows1)
    sessions2, processes2 = _processes(rows2)
    require(not sessions1 & sessions2, 'a launch session wrote rows of both runs')
    shared = sorted(processes1 & processes2)
    if registered:
        require(not shared, 'run 2 was not decoded in a separate process (section 16)')
    changed = []
    for a, b in zip(rows1, rows2):
        if a['completion_ids'] == b['completion_ids']:
            require(a['stop_reason'] == b['stop_reason'] and a['response'] == b['response']
                    and a['generated_tokens'] == b['generated_tokens'],
                    'equal token IDs but different derived fields: ' + a['id'])
            continue
        changed.append({'id': a['id'], 'bank': a['bank'], 'group': a['group'], 'index': a['index'],
                        'first_difference': _first_difference(a['completion_ids'], b['completion_ids']),
                        'generated_tokens': [a['generated_tokens'], b['generated_tokens']],
                        'stop_reason': [a['stop_reason'], b['stop_reason']],
                        'completion_sha256': [digest(a['completion_ids']), digest(b['completion_ids'])]})
    scores = {key: _strict_ids(first['receipt'], key) == _strict_ids(second['receipt'], key)
              for key in ('guard_dev_scoring', 'calibration_scoring')}
    if not changed:
        require(all(scores.values()), 'zero token churn but the runs scored differently')
    churn = len(changed)
    return {'churn': churn, 'rows_compared': len(rows1), 'differing_ids': [c['id'] for c in changed],
            'differing': changed, 'by_bank': dict(sorted(Counter(c['bank'] for c in changed).items())),
            'by_group': dict(sorted(Counter(c['group'] for c in changed).items())),
            'scores_equal': scores, 'outcome': 'CHURN_ZERO' if churn == 0 else 'STOP_NONDETERMINISTIC',
            'rule': 'churn = number of rows whose generated token IDs differ between run 1 and run 2; '
                    'it must be 0, otherwise STOP_NONDETERMINISTIC (section 8.3)',
            'runs': {str(n): {'receipt': e['path'], 'receipt_sha256': e['sha256'],
                              'journal': e['receipt']['journal']['path'],
                              'journal_sha256': e['receipt']['journal']['sha256'],
                              'sessions': sorted(s), 'processes': sorted([list(p) for p in ps])}
                     for n, e, s, ps in ((1, first, sessions1, processes1), (2, second, sessions2, processes2))},
            'bind': {k: v for k, v in bind1.items() if k not in BIND_EXEMPT},
            'separate_processes': not shared, 'shared_processes': [list(p) for p in shared],
            'separate_processes_required': registered}


def registered_out(out, freeze: dict) -> Path:
    """The registered receipt path: RECORDS/S0_CHURN.json, or S0_CHURN_WARM.json when the freeze has an anchor.
    An explicit --out must carry the _WARM suffix exactly when the freeze is anchored."""
    anchored = freeze['anchor'] is not None
    target = Path(out) if out is not None else s0.RECORDS / s0.anchored_name(STAGE + '.json', anchored)
    if anchored != target.stem.endswith(s0.WARM_SUFFIX):
        raise SystemExit("a theta0' (anchored) churn writes to a *%s receipt name and a Supa churn does not: %s"
                         % (s0.WARM_SUFFIX, target.name))
    return target


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='s0_churn', description='COMP-RS-001 S0-E churn floor.')
    ap.add_argument('--run1', type=Path, required=True)
    ap.add_argument('--run2', type=Path, required=True)
    ap.add_argument('--runtime-freeze', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--repo', type=Path, default=comp.REPO)
    ap.add_argument('--test-mode', action='store_true')
    return ap


def main(argv=None) -> int:
    t0 = time.perf_counter()
    args = _parser().parse_args(argv)
    registered = not args.test_mode
    if registered:
        if args.runtime_freeze is None:
            raise SystemExit('the registered churn asserts the S0 runtime freeze (--runtime-freeze)')
        if not comp._same_path(args.repo, comp.REPO):
            raise SystemExit('the registered churn binds the checkout that holds this code: ' + str(comp.REPO))
        if args.out is not None and not comp._same_path(Path(args.out).resolve().parent, s0.RECORDS):
            raise SystemExit('registered records are written to ' + str(s0.RECORDS))
        freeze = comp.read_freeze(args.runtime_freeze, registered=True)
        out = registered_out(args.out, freeze)
    else:
        if args.out is None:
            raise SystemExit('--test-mode needs --out')
        out = args.out
        freeze = comp.read_freeze(args.runtime_freeze, registered=False) if args.runtime_freeze else None
    target = s0.assert_outside_checkout(out, args.repo)
    if target.exists():
        raise SystemExit('receipt exists (create-only): ' + str(target))
    if registered:
        runtime = dict(s0.assert_runtime(), asserted=True)
        checkout = dict(s0.assert_clean_checkout(args.repo), checked=True)
    else:
        runtime = {'python_executable': sys.executable, 'python': platform.python_version(), 'asserted': False}
        checkout = {'checked': False, 'reason': 'test mode'}
    result = compare_runs(args.run1, args.run2, registered=registered, freeze=freeze)
    receipt = {'experiment': 'COMP-RS-001', 'stage': STAGE, 'schema': SCHEMA, 'registration': REGISTRATION,
               'test_mode': not registered, 'registered_run': registered,
               'argv': [str(a) for a in (sys.argv[1:] if argv is None else argv)], 'finished_utc': comp._now(),
               'wall_s': time.perf_counter() - t0, 'runtime': runtime, 'checkout': checkout,
               'runtime_freeze': freeze, 'code_sha256': _IMPORT_CODE, **result,
               'native_outputs_scored': False, 'decoded': False}
    require(comp.code_sha256(CODE_FILES) == _IMPORT_CODE, 'churn source changed during the run')
    written = s0.write_receipt(target, receipt, repo=args.repo)
    print(json.dumps({'stage': STAGE, 'test_mode': not registered, 'outcome': result['outcome'],
                      'churn': result['churn'], 'rows_compared': result['rows_compared'], 'receipt': str(target),
                      'sha256': written}), flush=True)
    return 0 if result['outcome'] == 'CHURN_ZERO' else 2


if __name__ == '__main__':
    raise SystemExit(main())
