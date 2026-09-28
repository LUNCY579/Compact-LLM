"""Load every panel arm with full per-row predictions, rescore it, and COLLIDE with its own record.

An arm is admitted (collision_ok True) only if the rescored correct sets reproduce everything
its artifact recorded: totals on both policies, lost_ids per class on both policies, per-row
flags where present, and the raw text + termination of every stored sample. The reference side
is checked too: frozen digest a329eb7b, sizes 47/41, and every stored sample's `source` text
equals Supa's raw output for that row.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import core as C  # noqa: E402

REC = Path('C:/Users/clclc/BusinessProject/rineya-dataset-retention-records')
PILOT_EV = Path('C:/Users/clclc/BusinessProject/ChatBubble-retention-pilot-20260917/experiments/architecture_benchmark_001/evidence/retention-design-v1')
AB_EV = C.AB / 'evidence' / 'retention-design-v1'
POLS = C.POLS


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


PROBE_INDEX = {}
for root in (REC, AB_EV, PILOT_EV):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d != '.git']
        for f in fn:
            if f.startswith('RETENTION_PROBE') and f.endswith('.json'):
                PROBE_INDEX.setdefault(f, []).append(Path(dp) / f)
# eval-dir result.json files are RETENTION_PROBE-shaped too (doseint001, highdose, binding-*)
RESULT_INDEX = {}
for dp, dn, fn in os.walk(REC):
    dn[:] = [d for d in dn if d != '.git']
    if 'result.json' in fn:
        RESULT_INDEX[Path(dp)] = Path(dp) / 'result.json'


def probe_copies_agree(paths):
    """All copies must carry the same label, weight and by_bare_policy content."""
    ds = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    keys = [(d.get('label'), (d.get('hf_candidate_identity') or {}).get('weight_sha256'),
             json.dumps(d['by_bare_policy'], sort_keys=True), json.dumps(d.get('samples'), sort_keys=True)) for d in ds]
    return len(set(keys)) == 1, len({sha(p) for p in paths})


def recorded_from_probe(d):
    rec = {}
    for pol in POLS:
        b = d['by_bare_policy'][pol]
        lost = sorted(i for c in b['per_class'].values() for i in c['lost_ids'])
        rec[pol] = {'total': b['arm_correct_total'], 'lost_ids': lost, 'source_total': b['source_correct_total'],
                    'per_row': ({i: r['correct'] for i, r in b['per_row'].items()} if 'per_row' in b else None)}
    rec['samples'] = d.get('samples') or []
    rec['weight'] = (d.get('hf_candidate_identity') or {}).get('weight_sha256')
    rec['probe_sha'] = d.get('probe_sha256')
    rec['ref_digest'] = d.get('reference_digest')
    rec['ref_sizes'] = d.get('frozen_reference_sizes')
    return rec


def collide(preds, recorded):
    detail, ok = {}, True
    if 'ref_digest' in recorded:
        detail['ref'] = f"{(recorded['ref_digest'] or '')[:8]} {recorded['ref_sizes']}"
        ok &= (recorded['ref_digest'] or '').startswith('a329eb7b') and recorded['ref_sizes'] == {'permissive': 47, 'strict': 41}
        ok &= (recorded.get('probe_sha') or '').startswith('5f6df1c1')
    for pol in POLS:
        if pol not in recorded:
            continue
        c = C.score(preds, pol)
        r = recorded[pol]
        lost = sorted(C.SUPA_REF[pol] - c)
        d = {'rescored': len(c), 'recorded': r.get('total')}
        if r.get('total') is not None:
            ok &= (r['total'] == len(c))
        if r.get('source_total') is not None:
            ok &= (r['source_total'] == len(C.SUPA_REF[pol]))
        if r.get('lost_ids') is not None:
            d['lost_match'] = (lost == r['lost_ids'])
            ok &= d['lost_match']
        if r.get('lost_total') is not None:
            d['lost_total_match'] = (len(lost) == r['lost_total'])
            ok &= d['lost_total_match']
        if r.get('gained_total') is not None:
            d['gained_total_match'] = (len(c - C.SUPA_REF[pol]) == r['gained_total'])
            ok &= d['gained_total_match']
        if r.get('per_row') is not None:
            d['per_row_match'] = ({i for i, v in r['per_row'].items() if v} == c)
            ok &= d['per_row_match']
        detail[pol] = d
    samples = recorded.get('samples') or []
    if samples:
        m = sum(1 for s in samples if preds[s['id']]['text'] == s['arm'] and preds[s['id']]['terminated'] == s['arm_terminated'])
        ms = sum(1 for s in samples if C.SUPA_PREDS[s['id']]['text'] == s['source'] and C.SUPA_PREDS[s['id']]['terminated'] == s['source_terminated'])
        detail['samples_arm_text'] = f'{m}/{len(samples)}'
        detail['samples_source_text'] = f'{ms}/{len(samples)}'
        ok &= (m == len(samples)) and (ms == len(samples))
    return bool(ok), detail


FAMILY = {
    # Titan = RRP-015 control delta (transition rows, seed-17 draw) and its re-carvings
    'rrp015_control': 'TITAN_STORED', 'int001_a100': 'TITAN_DENSE', 'panel001_gate_a100': 'TITAN_DENSE',
    'int001_s25d': 'TITAN_SPARSE',
    'int001_a075': 'TITAN_xform', 'int001_a050': 'TITAN_xform', 'int001_a025': 'TITAN_xform',
    'int001_s25': 'TITAN_xform', 'int001_s10': 'TITAN_xform', 'int001_s10d': 'TITAN_xform', 'int001_s15': 'TITAN_xform',
    'int001_s20': 'TITAN_xform', 'int001_s50': 'TITAN_xform', 'int001_s50d': 'TITAN_xform',
    'int001_rank001_r16': 'TITAN_xform', 'int001_rank001_r4': 'TITAN_xform', 'int001_rank001_r64': 'TITAN_xform',
    'mag001_T16flat': 'TITAN_xform', 'mag001_sm05': 'TITAN_xform', 'mag001_sm10': 'TITAN_xform',
    # independent training draws of the SAME Titan recipe
    'int001_rep_s23_ctrl': 'TITAN_REP', 'int001_rep_s29_ctrl': 'TITAN_REP',
    'int001_rep_s23_k25': 'TITAN_REP_xform', 'int001_rep_s29_k25': 'TITAN_REP_xform',
    # other transition-trained arms
    'rrp015_rehearsal': 'TRANSITION_OTHER',
    'panel001_d1_s17': 'D1', 'panel001_d1_s23': 'D1', 'panel001_d1_s29': 'D1',
    'panel001_dose_a025': 'DOSEINT', 'panel001_dose_a050': 'DOSEINT', 'panel001_dose_a075': 'DOSEINT', 'panel001_dose_a100': 'DOSEINT',
    'doseint001_dual899_a025_retention': 'DOSEINT', 'doseint001_dual899_a050_retention': 'DOSEINT',
    'doseint001_dual899_a075_retention': 'DOSEINT', 'doseint001_dual899_a100_retention': 'DOSEINT',
}

ARMS = {}


def add(name, preds, path, recorded, family, weight=None, note=''):
    if recorded:
        ok, detail = collide(preds, recorded)
    else:
        ok, detail = None, {'note': 'no recorded numbers to collide with'}
    ARMS[name] = {'preds': preds, 'path': str(path), 'sha8': sha(path)[:8], 'weight8': (weight or '')[:8],
                  'collision_ok': ok, 'detail': detail, 'family': family, 'note': note}


def load_pred_list(path):
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    return {p['id']: {'text': p['text'], 'terminated': p['terminated']} for p in d['predictions']}


def binding_for(pred_path):
    for cand in (pred_path.parent / 'probe_binding.json', pred_path.parent.parent / 'probe_binding.json'):
        if cand.is_file():
            return json.loads(cand.read_text(encoding='utf-8'))
    return None


# 1) eval-dir retention_predictions.json (raw worker output) -> collide with its RETENTION_PROBE / result.json
for dp, dn, fn in os.walk(REC):
    dn[:] = [d for d in dn if d != '.git']
    if 'retention_predictions.json' not in fn:
        continue
    p = Path(dp) / 'retention_predictions.json'
    b = binding_for(p)
    if b is None:
        continue
    label = b['arguments'][0]
    base = b['result'].replace('\\', '/').split('/')[-1]
    cands = PROBE_INDEX.get(base, [])
    rec, weight, note = None, None, ''
    if cands:
        agree, nh = probe_copies_agree(cands)
        d = json.loads(cands[0].read_text(encoding='utf-8'))
        if not (d.get('probe_sha256') or '').startswith('5f6df1c1'):
            continue
        rec = recorded_from_probe(d)
        weight = rec['weight']
        note = f'collided with {base}: {len(cands)} copies, {nh} distinct hashes, content_agree={agree}'
        if not agree:
            rec = None
            note += ' -> REFUSED'
    else:
        # eval dir with result.json sibling (worker/ layout)
        rj = RESULT_INDEX.get(p.parent.parent) or RESULT_INDEX.get(p.parent)
        if rj is not None:
            d = json.loads(rj.read_text(encoding='utf-8'))
            if (d.get('probe_sha256') or '').startswith('5f6df1c1') and 'by_bare_policy' in d:
                rec = recorded_from_probe(d)
                weight = rec['weight']
                note = f'collided with {rj.parent.name}/result.json'
        if rec is None:
            note = f'NO record found for {base}'
    add(label, load_pred_list(p), p, rec, FAMILY.get(label, 'OTHER'), weight, note)

# 2) RETENTION_PROBE files carrying scored per_row (panel001, indep, depthprof, ...)
for base, paths in PROBE_INDEX.items():
    d = json.loads(paths[0].read_text(encoding='utf-8'))
    if not (d.get('probe_sha256') or '').startswith('5f6df1c1'):
        continue
    bp = d['by_bare_policy']['permissive']
    if 'per_row' not in bp:
        continue
    agree, nh = probe_copies_agree(paths)
    label = d['label']
    pr_p, pr_s = bp['per_row'], d['by_bare_policy']['strict']['per_row']
    assert all(pr_p[i]['raw'] == pr_s[i]['raw'] and pr_p[i]['terminated'] == pr_s[i]['terminated'] for i in C.IDS)
    preds = {i: {'text': pr_p[i]['raw'], 'terminated': pr_p[i]['terminated']} for i in C.IDS}
    name = label if label not in ARMS else label + '__scored'
    add(name, preds, paths[0], recorded_from_probe(d) if agree else None, FAMILY.get(label, 'OTHER'),
        (d.get('hf_candidate_identity') or {}).get('weight_sha256'), f'{len(paths)} copies, {nh} hashes, agree={agree}')

# 3) RAB-001 ARM_DOSE final checkpoints (Opallo recipe + its seed replicates + other RAB arms) and the BF16 core
for f in sorted(AB_EV.glob('ARM_DOSE_*.json')):
    d = json.loads(f.read_text(encoding='utf-8'))
    if not d.get('completed'):
        continue
    c = d['checkpoints'][-1]
    tag = f.stem.replace('ARM_DOSE_', '')
    preds = {i: {'text': v['text'], 'terminated': v['terminated']} for i, v in c['outputs'].items()}
    reg = d.get('registered_bare_policy')
    rec = {pol: {'total': c['retention'][pol]['correct_total'], 'lost_ids': None,
                 'lost_total': c['retention'][pol].get('lost_total'), 'gained_total': c['retention'][pol].get('gained_total'),
                 'source_total': c['retention'][pol].get('reference_n'),
                 'per_row': (c['retention']['per_row'] if (pol == reg and isinstance(c['retention'].get('per_row'), dict)) else None)}
           for pol in POLS}
    if tag == 'B28last20_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95':
        fam = 'OPALLO'
    elif tag.startswith('B28last20_lr1e-05_b5200_seed') and tag.endswith('wsd0.2-0.2_b20.95'):
        fam = 'OPALLO_REP'
    else:
        fam = 'RAB_OTHER'
    add('rab_' + tag, preds, f, rec, fam, note=f'registered_bare_policy={reg}')

add('opallo_bf16', C.BF16_PREDS, C.OPALLO_BF16,
    {pol: {'total': len(C.BF16_RECORDED[pol]), 'lost_ids': None,
           'per_row': {i: (i in C.BF16_RECORDED[pol]) for i in C.IDS}} for pol in POLS}, 'OPALLO')

if __name__ == '__main__':
    bad = [n for n, a in ARMS.items() if a['collision_ok'] is False]
    none = [n for n, a in ARMS.items() if a['collision_ok'] is None]
    print('arms loaded', len(ARMS), '| collision FAIL', len(bad), bad, '| no record', len(none), none)
    rows = []
    for n in sorted(ARMS, key=lambda n: (ARMS[n]['family'], n)):
        a = ARMS[n]
        cp, cs = len(C.score(a['preds'], 'permissive')), len(C.score(a['preds'], 'strict'))
        rows.append({'name': n, 'family': a['family'], 'sha8': a['sha8'], 'weight8': a['weight8'], 'perm': cp, 'strict': cs,
                     'collision_ok': a['collision_ok'], 'detail': a['detail'], 'note': a['note'], 'path': a['path']})
        print(f"{a['family']:16s} {n[:46]:46s} {a['sha8']} w={a['weight8']:8s} perm {cp:3d} strict {cs:3d} ok={a['collision_ok']} "
              f"{json.dumps(a['detail'])[:170]}")
    (HERE / 'arms_admission.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')
