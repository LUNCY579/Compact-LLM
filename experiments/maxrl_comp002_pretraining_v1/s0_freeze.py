"""S0-F runtime freeze (COMP_RS_001_AMENDMENT_1.md section 16): bind what every later stage asserts.

Three modes, one output stage (S0_RUNTIME_FREEZE):
- Supa (section 16, C12). Inputs are the passed S0-Q1 receipt and the adopted S0-P pilot receipt (plus the A
  receipt when B was adopted). The decoding preimage is the one S0-Q1 made on cuda:0, and the pilot must have run
  under the same decoding_sha256, so both native stages are shown to share the frozen runtime. anchor is None.
  Default output S0_RUNTIME_FREEZE.json.
- theta0' (section 18, Amendment 2, step 3). --anchor-receipt S0_WARMSTART.json with --q1-receipt and no pilot
  receipt. The S0-Q1 decoding preimage and host floor are carried over (the numeric path is unchanged); the
  inventory is PENDING (C12: the pilot ladder is re-run on theta0' after S0-C passes). The warm-start receipt must
  be stage S0_WARMSTART, a registered run (test_mode False, registered_run True), and hold an 'anchor' object with
  WARMSTART_ANCHOR_KEYS (= s0_runtime.ANCHOR_KEYS: base_weights_sha256, delta_path, delta_sha256 = the delta file's
  data-section digest, arm_sha256 = theta0' under s0_runtime.ANCHOR_DIGEST_RULE). base_weights_sha256 must be the
  asset receipt's model.safetensors sha, and the delta file must exist (outside the checkout when registered) and
  reproduce delta_sha256 (arm_build.delta_data_digest, streamed). The arm digest itself is verified by every later
  stage's rebuild (s0_runtime.load_anchor), not here: no model is loaded. The output anchor is those four fields
  plus warmstart_receipt_sha256. Default output S0_RUNTIME_FREEZE_WARM.json.
- theta0' after the pilot ladder (sections 16/18). --initial-warm-freeze with --competence-receipt and
  --pilot-receipt (plus --a-receipt for adopted B) creates S0_RUNTIME_FREEZE_PILOT_WARM.json. The competence
  must be complete SIGNAL on that exact initial freeze/anchor; both participating pilots must bind it too.
  The original anchor and decoding identity are preserved, inventory is adopted, and the host floor is the
  maximum of the carried Q1 floor and each participating pilot's measured peak worker RSS + 1.0 GiB.
  The initial warm freeze remains the separate pre-competence record; neither record is overwritten.

No model, tokenizer or GPU in either mode. Output fields (the reader is s0_competence.read_freeze): stage,
decoding_preimage, decoding_sha256, host_floor_gib (the S0-Q1 peak worker RSS + 1.0 GiB),
scoring_code_bundle_sha256 (the live scoped_scorer bundle), inventory, anchor, plus provenance. Create-only,
outside the checkout.
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
from experiments.maxrl_comp002_pretraining_v1.preflight import digest, require, sha

STAGE = 'S0_RUNTIME_FREEZE'
SCHEMA = 'COMP-RS-001-S0-FREEZE-1'
SCHEMA_WARM = 'COMP-RS-001-S0-FREEZE-WARM-1'
SCHEMA_WARM_PILOT = 'COMP-RS-001-S0-FREEZE-WARM-PILOT-1'
REGISTRATION = 'COMP_RS_001_AMENDMENT_1.md section 16 (S0-F) and Clarifications C10-C12'
REGISTRATION_WARM = ("COMP_RS_001_AMENDMENT_1.md section 18 (Amendment 2, step 3: the runtime freeze for theta0'), "
                     'on section 16 (S0-F) and Clarifications C10-C12')
REGISTRATION_WARM_PILOT = ('COMP_RS_001_AMENDMENT_1.md sections 16/18: adopted donor inventory after SIGNAL on '
                           "theta0', preserving the initial warm runtime freeze")
WARMSTART_STAGE = 'S0_WARMSTART'
WARMSTART_ANCHOR_KEYS = s0.ANCHOR_KEYS
DEFAULT_OUT = {False: 'S0_RUNTIME_FREEZE.json', True: s0.anchored_name('S0_RUNTIME_FREEZE.json', True)}
DEFAULT_WARM_PILOT_OUT = 'S0_RUNTIME_FREEZE_PILOT_WARM.json'
PENDING_WARM = {'name': 'PENDING', 'blocks': None, 'registered_numel': None, 'decision': None,
                'rule': ("C12 / section 18: no pilot has run on theta0'; the section 16 pilot ladder is re-run on "
                         "theta0' after S0-C on theta0' passes")}
Q1_STAGE = 'S0_Q1_FP32'
PILOT_STAGES = {'A': 'S0_PILOT_A', 'B': 'S0_PILOT_B'}
REGISTERED_DEVICE = 'cuda:0'
HOST_MARGIN_GIB = 1.0
INVENTORY_BLOCKS = {'A': (8, 27), 'B': (14, 27)}
INVENTORY_NUMEL = {'A': 314_618_880, 'B': 220_233_216}
CODE_FILES = ('s0_freeze.py', 's0_runtime.py', 's0_generate.py', 's0_q1_fp32.py', 's0_pilot.py',
              's0_competence.py', 's0_enroll.py', 's0_churn.py', 'mechanics.py', 'preflight.py', 'arm_build.py')


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hex64(value) -> bool:
    return type(value) is str and len(value) == 64 and not set(value) - set('0123456789abcdef')


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


def check_pilot(pilot: dict, decoding: str, *, test_mode: bool, a_receipt: dict | None,
                anchor=None, allow_pending: bool = True) -> dict:
    stage = pilot.get('stage')
    require(stage in PILOT_STAGES.values(), 'not an S0 pilot receipt')
    _registered(pilot, 'the pilot receipt', test_mode)
    require(pilot.get('anchor') == anchor, 'the pilot anchor differs from the freeze anchor')
    inventory = 'A' if stage == PILOT_STAGES['A'] else 'B'
    decision = pilot.get('decision') or {}
    require(pilot.get('decoding_sha256') == decoding, 'the pilot ran under a different decoding_sha256 than S0-Q1')
    if allow_pending and inventory == 'A' and decision.get('outcome') == 'UNMEASURED_NO_UPDATE' and a_receipt is None:
        # C12: no mixed pilot group; S0-C runs with the inventory pending, the pilot re-runs if S0-C passes
        return {'name': 'PENDING', 'blocks': None, 'registered_numel': None, 'decision': decision,
                'rule': 'C12: UNMEASURED_NO_UPDATE defers the inventory decision to after S0-C'}
    require(decision.get('outcome') == 'ADOPT_' + inventory and decision.get('inventory') == inventory,
            'the pilot did not adopt its inventory: %s' % decision.get('outcome'))
    if inventory == 'B':
        require(a_receipt is not None, 'inventory B needs the A pilot receipt')
        require(a_receipt.get('stage') == PILOT_STAGES['A'], 'the A receipt is not an S0_PILOT_A receipt')
        _registered(a_receipt, 'the A pilot receipt', test_mode)
        require(a_receipt.get('anchor') == anchor, 'the A pilot anchor differs from the freeze anchor')
        require(a_receipt.get('decoding_sha256') == decoding,
                'the A pilot ran under a different decoding_sha256 than the freeze')
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
            'inventory': inventory, 'anchor': None,
            'inputs': {'q1': {'path': str(Path(q1_path).resolve()), 'sha256': q1_sha},
                       'pilot': {'path': str(Path(pilot_path).resolve()), 'sha256': pilot_sha},
                       'a_pilot': None if a_path is None else {'path': str(Path(a_path).resolve()),
                                                               'sha256': a_sha}},
            'code_sha256': _IMPORT_CODE, 'code_bundle_sha256': digest(_IMPORT_CODE)}


def _asset_weights_sha(asset_receipt, *, test_mode: bool) -> str:
    """The base weights the anchor must be built on: the asset receipt's model.safetensors sha (pinned file)."""
    path = Path(asset_receipt)
    if not test_mode:
        require(path.resolve() == s0.ASSET_RECEIPT.resolve() and sha(path) == s0.ASSET_RECEIPT_SHA,
                'the registered freeze binds the pinned asset receipt')
    receipt, _ = load_receipt(path)
    require(receipt.get('stage') == 'Q1_ASSET_CANDIDATE', 'not a Q1_ASSET_CANDIDATE receipt')
    weights = ((receipt.get('assets') or {}).get('model.safetensors') or {}).get('sha256')
    require(_hex64(weights), 'the asset receipt has no model.safetensors sha256')
    return weights


def check_warmstart(warm: dict, *, test_mode: bool, base_weights_sha256: str, repo) -> dict:
    """The theta0' anchor a registered warm-start receipt names (its delta file re-hashed, streamed)."""
    from experiments.maxrl_comp002_pretraining_v1 import arm_build
    require(warm.get('stage') == WARMSTART_STAGE, 'not an ' + WARMSTART_STAGE + ' receipt')
    _registered(warm, 'the warm-start receipt', test_mode)
    require(warm.get('completed') is True and warm.get('outcome') == 'WARMSTART_COMPLETE',
            'the warm-start receipt did not complete successfully')
    plan, training, steps = warm.get('plan'), warm.get('training'), warm.get('steps')
    require(type(plan) is dict and type(training) is dict and type(steps) is list,
            'warm-start completion counts are missing')
    rows, planned_steps, committed_steps = plan.get('rows'), plan.get('steps'), training.get('steps')
    require(type(rows) is int and type(planned_steps) is int and type(committed_steps) is int
            and rows > 0 and planned_steps > 0 and rows == 8 * planned_steps
            and committed_steps == planned_steps == len(steps), 'warm-start completion counts disagree')
    if not test_mode:
        require(rows == 600 and planned_steps == 75, 'the registered warm-start completion is not 600 rows / 75 steps')
    anchor = warm.get('anchor')
    require(type(anchor) is dict, "the warm-start receipt has no 'anchor' object")
    missing = [key for key in WARMSTART_ANCHOR_KEYS if key not in anchor]
    require(not missing, 'the warm-start anchor lacks: ' + ', '.join(missing))
    for key in ('base_weights_sha256', 'delta_sha256', 'arm_sha256'):
        require(_hex64(anchor[key]), 'warm-start anchor %s is not a sha256 hex digest' % key)
    path = anchor['delta_path']
    require(type(path) is str and path and Path(path).is_absolute(), 'warm-start anchor delta_path must be absolute')
    require(anchor['base_weights_sha256'] == base_weights_sha256,
            'the warm-start anchor was built on other base weights than the asset receipt names')
    if not test_mode:
        s0.assert_outside_checkout(path, repo)
    require(Path(path).is_file(), 'the anchor delta file is missing: ' + path)
    require(arm_build.delta_data_digest(path) == anchor['delta_sha256'],
            'the anchor delta file does not reproduce its data-section digest')
    return {key: anchor[key] for key in WARMSTART_ANCHOR_KEYS}


def build_warm(q1_path, anchor_path, *, test_mode: bool = False, asset_receipt=None, repo=None) -> dict:
    """The theta0' runtime freeze (section 18 step 3): the S0-Q1 runtime carried over, the anchor added."""
    repo = s0.HERE.parents[1] if repo is None else repo
    q1, q1_sha = load_receipt(q1_path)
    warm, warm_sha = load_receipt(anchor_path)
    frozen = check_q1(q1, test_mode=test_mode)
    base = _asset_weights_sha(s0.ASSET_RECEIPT if asset_receipt is None else asset_receipt, test_mode=test_mode)
    anchor = check_warmstart(warm, test_mode=test_mode, base_weights_sha256=base, repo=repo)
    require(code_sha256() == _IMPORT_CODE, 's0 source changed since import')
    return {'experiment': 'COMP-RS-001', 'stage': STAGE, 'schema': SCHEMA_WARM, 'registration': REGISTRATION_WARM,
            'test_mode': bool(test_mode), 'registered_run': not test_mode,
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
            'decoding_preimage': frozen['decoding_preimage'], 'decoding_sha256': frozen['decoding_sha256'],
            'host_floor_gib': frozen['host_floor_gib'],
            'host_floor': {'peak_worker_rss_gib': frozen['peak_worker_rss_gib'], 'margin_gib': HOST_MARGIN_GIB,
                           'source': 'S0-Q1 host_floor.peak_worker_rss_gib (carried over, section 18)'},
            'scoring_code_bundle_sha256': digest(scoped_scorer.code_sha256()),
            'inventory': dict(PENDING_WARM),
            'anchor': dict(anchor, warmstart_receipt_sha256=warm_sha),
            'anchor_digest_rule': s0.ANCHOR_DIGEST_RULE,
            'inputs': {'q1': {'path': str(Path(q1_path).resolve()), 'sha256': q1_sha},
                       'warmstart': {'path': str(Path(anchor_path).resolve()), 'sha256': warm_sha},
                       'pilot': None, 'a_pilot': None},
            'code_sha256': _IMPORT_CODE, 'code_bundle_sha256': digest(_IMPORT_CODE)}


def _same_initial_freeze(receipt: dict, initial: dict, initial_sha: str, what: str) -> None:
    bound = receipt.get('runtime_freeze')
    require(type(bound) is dict and bound.get('sha256') == initial_sha
            and bound.get('anchor') == initial['anchor'], what + ' did not bind the initial warm freeze/anchor')


def build_warm_postpilot(initial_path, pilot_path, competence_path, *, a_path=None, test_mode: bool = False) -> dict:
    """Create the post-pilot theta0' freeze as a distinct successor; retain the initial anchor and runtime."""
    from experiments.maxrl_comp002_pretraining_v1 import s0_competence as comp
    initial, initial_sha = load_receipt(initial_path)
    parsed = comp.read_freeze(initial_path, registered=not test_mode)
    require(parsed['sha256'] == initial_sha, 'the initial warm freeze changed while read')
    require(initial.get('schema') == SCHEMA_WARM and parsed['anchor'] is not None
            and (parsed['inventory'] or {}).get('name') == 'PENDING', 'not an initial pre-competence warm freeze')
    require(initial.get('anchor_digest_rule') == s0.ANCHOR_DIGEST_RULE, 'initial warm anchor digest rule differs')
    require(parsed['scoring_code_bundle_sha256'] == digest(scoped_scorer.code_sha256()),
            'the initial warm freeze scorer bundle differs from the live bundle')
    competence, competence_sha = load_receipt(competence_path)
    require(competence.get('stage') == 'S0_COMPETENCE', 'not an S0_COMPETENCE receipt')
    _registered(competence, 'the competence receipt', test_mode)
    require(competence.get('completed') is True and competence.get('outcome') == 'SIGNAL',
            'the warm competence receipt did not complete with SIGNAL')
    _same_initial_freeze(competence, parsed, initial_sha, 'the competence receipt')
    require(comp.anchor_matches(competence.get('anchor'), parsed['anchor'])
            and competence.get('decoding_sha256') == parsed['decoding_sha256'],
            'the competence receipt used another anchor/decoding identity')
    pilot, pilot_sha = load_receipt(pilot_path)
    a_receipt, a_sha = (None, None) if a_path is None else load_receipt(a_path)
    inventory = check_pilot(pilot, parsed['decoding_sha256'], test_mode=test_mode, a_receipt=a_receipt,
                            anchor=parsed['anchor'], allow_pending=False)
    require(pilot.get('completed') is True and pilot.get('error') is None and pilot.get('oom') is None
            and pilot.get('abort') is None, 'the adopted warm pilot did not complete successfully')
    if a_receipt is not None:
        linkage = pilot.get('after_a_pilot')
        require(type(linkage) is dict and linkage.get('sha256') == a_sha
                and linkage.get('outcome') == 'RERUN_WITH_B'
                and linkage.get('anchor') == parsed['anchor']
                and linkage.get('registered_run') is a_receipt.get('registered_run'),
                'the B pilot did not bind the supplied A fallback receipt')
    requirements = []
    for receipt, path, file_sha in ((pilot, pilot_path, pilot_sha), (a_receipt, a_path, a_sha)):
        if receipt is None:
            continue
        _same_initial_freeze(receipt, parsed, initial_sha, 'the ' + receipt['stage'] + ' receipt')
        peak = (receipt.get('decision') or {}).get('peak_worker_rss_gib')
        require(type(peak) in (int, float) and math.isfinite(peak) and peak > 0,
                'the participating pilot has no measured peak worker RSS')
        requirements.append({'stage': receipt['stage'], 'path': str(Path(path).resolve()), 'sha256': file_sha,
                             'peak_worker_rss_gib': float(peak), 'floor_gib': float(peak) + HOST_MARGIN_GIB})
    require(code_sha256() == _IMPORT_CODE, 's0 source changed since import')
    host_floor = max([parsed['host_floor_gib']] + [entry['floor_gib'] for entry in requirements])
    return {'experiment': 'COMP-RS-001', 'stage': STAGE, 'schema': SCHEMA_WARM_PILOT,
            'registration': REGISTRATION_WARM_PILOT, 'test_mode': bool(test_mode), 'registered_run': not test_mode,
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
            'decoding_preimage': initial['decoding_preimage'], 'decoding_sha256': parsed['decoding_sha256'],
            'host_floor_gib': host_floor,
            'host_floor': {'source': 'maximum of initial Q1 floor and participating pilot peak RSS + 1.0 GiB',
                           'initial_floor_gib': parsed['host_floor_gib'], 'margin_gib': HOST_MARGIN_GIB,
                           'pilot_requirements': requirements},
            'scoring_code_bundle_sha256': parsed['scoring_code_bundle_sha256'], 'inventory': inventory,
            'anchor': dict(parsed['anchor']), 'anchor_digest_rule': initial['anchor_digest_rule'],
            'inputs': {'initial_warm_freeze': {'path': str(Path(initial_path).resolve()), 'sha256': initial_sha},
                       'competence': {'path': str(Path(competence_path).resolve()), 'sha256': competence_sha},
                       'pilot': {'path': str(Path(pilot_path).resolve()), 'sha256': pilot_sha},
                       'a_pilot': None if a_path is None else {'path': str(Path(a_path).resolve()), 'sha256': a_sha}},
            'code_sha256': _IMPORT_CODE, 'code_bundle_sha256': digest(_IMPORT_CODE)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--q1-receipt', type=Path, help='required for the Supa and initial warm freezes')
    ap.add_argument('--pilot-receipt', type=Path, default=None, help='the adopted S0 pilot receipt (Supa mode)')
    ap.add_argument('--a-receipt', type=Path, default=None, help='the S0_PILOT_A receipt, when B was adopted')
    ap.add_argument('--anchor-receipt', type=Path, default=None,
                    help="the S0_WARMSTART receipt (theta0' mode; no pilot receipt)")
    ap.add_argument('--initial-warm-freeze', type=Path,
                    help='the initial warm freeze to preserve in a post-pilot successor')
    ap.add_argument('--competence-receipt', type=Path,
                    help='complete SIGNAL on the initial warm freeze (post-pilot mode only)')
    ap.add_argument('--asset-receipt', type=Path, default=None, help='test mode only (default: the pinned one)')
    ap.add_argument('--out', type=Path, default=None,
                    help='default: RECORDS/S0_RUNTIME_FREEZE.json, _WARM.json, or _PILOT_WARM.json by mode')
    ap.add_argument('--repo', type=Path, default=s0.HERE.parents[1])
    ap.add_argument('--test-mode', action='store_true', help='synthetic receipts; never a registered freeze')
    args = ap.parse_args(argv)
    postpilot = args.initial_warm_freeze is not None
    warm = postpilot or args.anchor_receipt is not None
    if postpilot:
        if args.pilot_receipt is None or args.competence_receipt is None:
            raise SystemExit('post-pilot mode needs --pilot-receipt and --competence-receipt')
        if args.q1_receipt is not None or args.anchor_receipt is not None or args.asset_receipt is not None:
            raise SystemExit('post-pilot mode preserves --initial-warm-freeze; no Q1, anchor or asset overrides')
    else:
        if args.competence_receipt is not None:
            raise SystemExit('--competence-receipt belongs to post-pilot mode with --initial-warm-freeze')
        if warm == (args.pilot_receipt is not None):
            raise SystemExit("give exactly one of --pilot-receipt (Supa) or --anchor-receipt (theta0'); the initial "
                             "warm freeze takes no pilot receipt (C12: the pilot re-runs after S0-C)")
        if args.q1_receipt is None:
            raise SystemExit('the Supa and initial warm freezes need --q1-receipt')
    if args.anchor_receipt is not None and args.a_receipt is not None:
        raise SystemExit('--a-receipt belongs to a pilot freeze')
    if args.asset_receipt is not None and not args.test_mode:
        raise SystemExit('--asset-receipt is test-mode only')
    default = DEFAULT_WARM_PILOT_OUT if postpilot else DEFAULT_OUT[warm]
    out = s0.RECORDS / default if args.out is None else args.out
    if postpilot:
        if Path(out).resolve() == args.initial_warm_freeze.resolve():
            raise SystemExit('the post-pilot warm freeze must have a distinct successor output')
        if not args.test_mode and not Path(out).stem.endswith(s0.WARM_SUFFIX):
            raise SystemExit('the registered post-pilot warm freeze output needs the _WARM suffix')
    if not args.test_mode:
        s0.assert_clean_checkout(args.repo)
    if postpilot:
        value = build_warm_postpilot(args.initial_warm_freeze, args.pilot_receipt, args.competence_receipt,
                                     a_path=args.a_receipt, test_mode=args.test_mode)
    elif warm:
        value = build_warm(args.q1_receipt, args.anchor_receipt, test_mode=args.test_mode,
                           asset_receipt=args.asset_receipt, repo=args.repo)
    else:
        value = build(args.q1_receipt, args.pilot_receipt, a_path=args.a_receipt, test_mode=args.test_mode)
    value['checkout'] = None if args.test_mode else s0.assert_clean_checkout(args.repo)
    written = s0.write_receipt(out, value, repo=args.repo)
    print(json.dumps({'stage': STAGE, 'inventory': value['inventory']['name'],
                      'anchor_arm_sha256': None if value['anchor'] is None else value['anchor']['arm_sha256'][:16],
                      'host_floor_gib': value['host_floor_gib'], 'decoding_sha256': value['decoding_sha256'][:16],
                      'out': str(out), 'sha256': written[:16]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
