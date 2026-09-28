"""S0-F runtime freeze (COMP_RS_001_AMENDMENT_1.md section 16): bind what every later stage asserts.

Inputs are the passed S0-Q1 receipt and the adopted S0-P pilot receipt (plus the A receipt when B was adopted).
No model, tokenizer or GPU: the decoding preimage is the one S0-Q1 made on cuda:0, and the pilot must have
run under the same decoding_sha256, so both native stages are shown to share the frozen runtime.

Output fields (the reader is s0_competence.read_freeze): stage S0_RUNTIME_FREEZE, decoding_preimage,
decoding_sha256, host_floor_gib (the S0-Q1 peak worker RSS + 1.0 GiB), scoring_code_bundle_sha256 (the live
scoped_scorer bundle), inventory (the adopted pilot inventory), plus provenance. Create-only, outside the checkout.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as s0
from experiments.maxrl_comp002_pretraining_v1 import scoped_scorer
from experiments.maxrl_comp002_pretraining_v1.preflight import digest, require

STAGE = 'S0_RUNTIME_FREEZE'
SCHEMA = 'COMP-RS-001-S0-FREEZE-1'
REGISTRATION = 'COMP_RS_001_AMENDMENT_1.md section 16 (S0-F) and Clarifications C10-C12'
Q1_STAGE = 'S0_Q1_FP32'
PILOT_STAGES = {'A': 'S0_PILOT_A', 'B': 'S0_PILOT_B'}
REGISTERED_DEVICE = 'cuda:0'
HOST_MARGIN_GIB = 1.0
INVENTORY_BLOCKS = {'A': (8, 27), 'B': (14, 27)}
INVENTORY_NUMEL = {'A': 314_618_880, 'B': 220_233_216}
CODE_FILES = ('s0_freeze.py', 's0_runtime.py', 's0_generate.py', 's0_q1_fp32.py', 's0_pilot.py',
              's0_competence.py', 's0_enroll.py', 's0_churn.py', 'mechanics.py', 'preflight.py')


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def code_sha256() -> dict:
    return {name: _sha((s0.HERE / name).read_bytes()) for name in CODE_FILES}


_IMPORT_CODE = code_sha256()


def load_receipt(path) -> tuple:
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode('utf-8'))
    require(type(value) is dict, 'receipt is not an object: %s' % path)
    return value, _sha(raw)


def _registered(value: dict, what: str, test_mode: bool) -> None:
    if test_mode:
        return
    require(value.get('test_mode') is False and value.get('registered_run') is True,
            what + ' is not a registered run')


def check_q1(q1: dict, *, test_mode: bool) -> dict:
    require(q1.get('stage') == Q1_STAGE, 'not an S0_Q1_FP32 receipt')
    _registered(q1, 'the S0-Q1 receipt', test_mode)
    require(q1.get('passed') is True and not q1.get('failed_checks'), 'S0-Q1 did not pass')
    preimage, decoding = q1.get('decoding_preimage'), q1.get('decoding_sha256')
    require(type(preimage) is dict and s0.decoding_sha256(preimage) == decoding,
            'S0-Q1 decoding_sha256 is not the digest of its preimage')
    if not test_mode:
        require(preimage.get('device') == REGISTERED_DEVICE, 'the S0-Q1 preimage is not for cuda:0')
    floor = (q1.get('host_floor') or {}).get('peak_worker_rss_gib')
    require(type(floor) in (int, float) and math.isfinite(floor) and floor > 0, 'S0-Q1 peak worker RSS missing')
    return {'decoding_preimage': preimage, 'decoding_sha256': decoding,
            'peak_worker_rss_gib': float(floor), 'host_floor_gib': float(floor) + HOST_MARGIN_GIB}


def check_pilot(pilot: dict, decoding: str, *, test_mode: bool, a_receipt: dict | None) -> dict:
    stage = pilot.get('stage')
    require(stage in PILOT_STAGES.values(), 'not an S0 pilot receipt')
    _registered(pilot, 'the pilot receipt', test_mode)
    inventory = 'A' if stage == PILOT_STAGES['A'] else 'B'
    decision = pilot.get('decision') or {}
    require(pilot.get('decoding_sha256') == decoding, 'the pilot ran under a different decoding_sha256 than S0-Q1')
    if inventory == 'A' and decision.get('outcome') == 'UNMEASURED_NO_UPDATE' and a_receipt is None:
        # C12: no mixed pilot group; S0-C runs with the inventory pending, the pilot re-runs if S0-C passes
        return {'name': 'PENDING', 'blocks': None, 'registered_numel': None, 'decision': decision,
                'rule': 'C12: UNMEASURED_NO_UPDATE defers the inventory decision to after S0-C'}
    require(decision.get('outcome') == 'ADOPT_' + inventory and decision.get('inventory') == inventory,
            'the pilot did not adopt its inventory: %s' % decision.get('outcome'))
    if inventory == 'B':
        require(a_receipt is not None, 'inventory B needs the A pilot receipt')
        require(a_receipt.get('stage') == PILOT_STAGES['A'], 'the A receipt is not an S0_PILOT_A receipt')
        _registered(a_receipt, 'the A pilot receipt', test_mode)
        require((a_receipt.get('decision') or {}).get('outcome') == 'RERUN_WITH_B',
                'the A pilot did not end RERUN_WITH_B')
    else:
        require(a_receipt is None, 'an A receipt is only given when B was adopted')
    first, last = INVENTORY_BLOCKS[inventory]
    return {'name': inventory, 'blocks': [first, last], 'registered_numel': INVENTORY_NUMEL[inventory],
            'decision': decision}


def build(q1_path, pilot_path, *, a_path=None, test_mode: bool = False) -> dict:
    q1, q1_sha = load_receipt(q1_path)
    pilot, pilot_sha = load_receipt(pilot_path)
    a_receipt, a_sha = (None, None) if a_path is None else load_receipt(a_path)
    frozen = check_q1(q1, test_mode=test_mode)
    inventory = check_pilot(pilot, frozen['decoding_sha256'], test_mode=test_mode, a_receipt=a_receipt)
    require(code_sha256() == _IMPORT_CODE, 's0 source changed since import')
    return {'experiment': 'COMP-RS-001', 'stage': STAGE, 'schema': SCHEMA, 'registration': REGISTRATION,
            'test_mode': bool(test_mode), 'registered_run': not test_mode,
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
            'decoding_preimage': frozen['decoding_preimage'], 'decoding_sha256': frozen['decoding_sha256'],
            'host_floor_gib': frozen['host_floor_gib'],
            'host_floor': {'peak_worker_rss_gib': frozen['peak_worker_rss_gib'], 'margin_gib': HOST_MARGIN_GIB,
                           'source': 'S0-Q1 host_floor.peak_worker_rss_gib'},
            'scoring_code_bundle_sha256': digest(scoped_scorer.code_sha256()),
            'inventory': inventory,
            'inputs': {'q1': {'path': str(Path(q1_path).resolve()), 'sha256': q1_sha},
                       'pilot': {'path': str(Path(pilot_path).resolve()), 'sha256': pilot_sha},
                       'a_pilot': None if a_path is None else {'path': str(Path(a_path).resolve()),
                                                               'sha256': a_sha}},
            'code_sha256': _IMPORT_CODE, 'code_bundle_sha256': digest(_IMPORT_CODE)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--q1-receipt', required=True, type=Path)
    ap.add_argument('--pilot-receipt', required=True, type=Path)
    ap.add_argument('--a-receipt', type=Path, default=None, help='the S0_PILOT_A receipt, when B was adopted')
    ap.add_argument('--out', type=Path, default=s0.RECORDS / 'S0_RUNTIME_FREEZE.json')
    ap.add_argument('--repo', type=Path, default=s0.HERE.parents[1])
    ap.add_argument('--test-mode', action='store_true', help='synthetic receipts; never a registered freeze')
    args = ap.parse_args(argv)
    if not args.test_mode:
        s0.assert_clean_checkout(args.repo)
    value = build(args.q1_receipt, args.pilot_receipt, a_path=args.a_receipt, test_mode=args.test_mode)
    value['checkout'] = None if args.test_mode else s0.assert_clean_checkout(args.repo)
    written = s0.write_receipt(args.out, value, repo=args.repo)
    print(json.dumps({'stage': STAGE, 'inventory': value['inventory']['name'],
                      'host_floor_gib': value['host_floor_gib'], 'decoding_sha256': value['decoding_sha256'][:16],
                      'out': str(args.out), 'sha256': written[:16]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
