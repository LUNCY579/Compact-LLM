"""Q2: held-out rows of Opallo's 24 training classes, and overlap with Titan's training data.

Overlap detector (text-level), with POSITIVE and NEGATIVE controls reported with counts.
"""
import collections
import glob
import json
import os
import random
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AB, ARM_DOSE, DEV, PROBE, SPL, TBANK, TITAN_DIR, TRAIN, dump, jl, sha  # noqa

arm = json.loads(ARM_DOSE.read_text(encoding='utf-8'))
ids = arm['train_row_ids_consumed']
idset = set(ids)
train = jl(TRAIN)
byid = {r['id']: r for r in train}
sel = [byid[i] for i in ids]
classes = sorted({r['reply_class'] for r in sel})
fams41 = {r['group_id'] for r in sel}
famsize = collections.Counter(r['group_id'] for r in train)
print('train.jsonl families of the 41 consumed rows: size distribution',
      dict(collections.Counter(famsize[f] for f in fams41)))

# ---------------------------------------------------------------- held-out pools, per class
rest = [r for r in train if r['id'] not in idset]
same = [r for r in rest if r['reply_class'] in classes]
sib = [r for r in same if r['group_id'] in fams41]
fd = [r for r in same if r['group_id'] not in fams41]
pos = {r['id']: k for k, r in enumerate(train)}
print(f'train.jsonl rest {len(rest)}: same-class {len(same)}, of which family-siblings of a consumed row '
      f'{len(sib)}, family-disjoint {len(fd)}')
per_class = {c: {'consumed': sum(r['reply_class'] == c for r in sel),
                 'train_rest_same_class': sum(r['reply_class'] == c for r in same),
                 'train_rest_family_disjoint': sum(r['reply_class'] == c for r in fd)} for c in classes}

# Other B28-family arms consumed larger prefixes of the same shuffle; record how deep the
# held-out pool is before it collides with ANY recorded arm's consumption (not only Opallo's).
sh = list(train)
random.Random(17).shuffle(sh)
order = {r['id']: k for k, r in enumerate(sh)}
print('shuffle positions of the 41 consumed ids: max', max(order[i] for i in ids))

files = sorted(set(glob.glob(str(SPL / '*.jsonl')) + glob.glob(str(AB / 'data' / '*.jsonl')) +
                   glob.glob(str(AB / 'outputs' / 'splits' / '*.jsonl')) +
                   glob.glob(str(AB / 'outputs' / 'splits_curated_*' / '*.jsonl')) +
                   glob.glob(str(AB / 'outputs' / 'splits_mixed_*' / '*.jsonl'))))
other = {}
for p in files:
    rs = jl(p)
    rel = os.path.relpath(p, AB).replace('\\', '/')
    rcf = lambda r: r.get('reply_class') or (r.get('metadata') or {}).get('reply_class')
    gf = lambda r: r.get('group_id') or (r.get('metadata') or {}).get('template_family')
    rc = collections.Counter(rcf(r) for r in rs)
    in_cls = [r for r in rs if rcf(r) in classes]
    other[rel] = {'sha8': sha(p)[:8], 'rows': len(rs), 'n_classes': len(rc),
                  'rows_in_opallo_classes': len(in_cls),
                  'id_overlap_with_41': sum(r.get('id') in idset for r in rs),
                  'family_overlap_with_41': sum(gf(r) in fams41 for r in rs),
                  'id_overlap_with_train_jsonl': sum(r.get('id') in byid for r in rs),
                  'source_row_id_in_41': sum(r.get('source_row_id') in idset for r in rs),
                  'source_row_id_in_train_jsonl': sum(r.get('source_row_id') in byid for r in rs),
                  'per_opallo_class': {c: sum(rcf(r) == c for r in in_cls) for c in classes}}
    o = other[rel]
    print(f"{rel:62s} {o['sha8']} rows {o['rows']:5d} classes {o['n_classes']:2d} in-Opallo-classes "
          f"{o['rows_in_opallo_classes']:5d} id-ovl41 {o['id_overlap_with_41']} fam-ovl41 "
          f"{o['family_overlap_with_41']} id-ovl-train {o['id_overlap_with_train_jsonl']} srcid-in41 "
          f"{o['source_row_id_in_41']} srcid-in-train {o['source_row_id_in_train_jsonl']}")
dev = jl(DEV)
dev40 = dev[:40]
dev_cls = collections.Counter(r['reply_class'] for r in dev)
dev40_cls = collections.Counter(r['reply_class'] for r in dev40)
print('dev150 classes', len(dev_cls), '| dev150 rows in Opallo classes', sum(dev_cls[c] for c in classes),
      '| dev[:40] (the recorded dev_ce rows) in Opallo classes', sum(dev40_cls[c] for c in classes))
for c in classes:
    per_class[c]['dev150'] = dev_cls[c]
    per_class[c]['dev40'] = dev40_cls[c]
print(f"{'class':30s} consumed train_rest fam_disj dev150 dev40")
for c in classes:
    v = per_class[c]
    print(f"{c:30s} {v['consumed']:8d} {v['train_rest_same_class']:10d} {v['train_rest_family_disjoint']:8d} "
          f"{v['dev150']:6d} {v['dev40']:5d}")

# ---------------------------------------------------------------- Titan's training corpus
TC = TITAN_DIR / 'control-prepared' / 'microfit-train.jsonl'
TR = TITAN_DIR / 'rehearsal-prepared' / 'microfit-train.jsonl'
titan = jl(TC)
rehe = jl(TR)
print('Titan control corpus', TC.name, sha(TC)[:8], len(titan), dict(collections.Counter(r['family'] for r in titan)),
      '| supervised tokens', sum(r['supervised_tokens'] for r in titan),
      '| objective: assistant_response_plus_eos (labels != -100):',
      sum(sum(1 for x in r['labels'] if x != -100) for r in titan))
tb = json.loads(TBANK.read_text(encoding='utf-8'))['rows']
assert sha(TBANK).startswith('fc094136')


def rin(r):
    t = r['text']
    s = re.search(r'<\|im_start\|>system\n(.*?)<\|im_end\|>', t, re.S).group(1)
    u = re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', t, re.S)
    a = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', t, re.S)[-1]
    a = a.split('</think>\n\n', 1)[-1]
    return s, u, a


def msg(prompt, ans):
    return (' '.join(m['content'] for m in prompt if m['role'] == 'system'),
            [m['content'] for m in prompt if m['role'] == 'user'], ans)


def probe_parts(r):
    ms = r.get('messages') or r.get('prompt')
    return msg(ms, r.get('answer', ''))


norm = lambda s: re.sub(r'\s+', ' ', s.lower()).strip()


def grams(s, n=8):
    w = re.findall(r'\w+|[^\w\s]', norm(s))
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def detect(A, B):
    """Rows of A that (exact) share a normalized last-user turn with some row of B;
    (ug) share >=1 user 8-gram; (sys) share >=1 system-prompt 8-gram; (pay) exact (user,answer)."""
    ub = {norm(u[-1]) for _, u, _ in B}
    gb = set().union(*[grams(' '.join(u)) for _, u, _ in B]) if B else set()
    sb = set().union(*[grams(s) for s, _, _ in B]) if B else set()
    pb = {norm(u[-1]) + '||' + norm(a) for _, u, a in B}
    return {'exact_user': sum(norm(u[-1]) in ub for _, u, _ in A),
            'user_8gram': sum(bool(grams(' '.join(u)) & gb) for _, u, _ in A),
            'sys_8gram': sum(bool(grams(s) & sb) for s, _, _ in A),
            'exact_payload': sum(norm(u[-1]) + '||' + norm(a) in pb for _, u, a in A),
            'n': len(A)}


P41 = [rin(r) for r in sel]
Pfd = [rin(r) for r in fd]
Psame = [rin(r) for r in same]
Pdev = [rin(r) for r in dev]
Ptitan = [msg(r['prompt'], r['reference_text']) for r in titan]
Prehe = [msg(r['prompt'], r['reference_text']) for r in rehe]
Prehe_conv = [msg(r['prompt'], r['reference_text']) for r in rehe if r['family'].startswith('rehearsal_')]
Ptb = [msg(r['prompt'], r['reference']) for r in tb]
Pprobe = [probe_parts(r) for r in jl(PROBE)]
Ptrain800 = [rin(r) for r in jl(AB / 'outputs' / 'splits' / 'train.jsonl')]
n41_in_800 = sum(pos[i] < 800 for i in ids)

ctl = {}
# POSITIVE controls: known overlaps must be found.
ctl['POS dev(splits_large) vs dev(splits) [bit-identical files]'] = detect(Pdev, [rin(r) for r in jl(AB / 'outputs' / 'splits' / 'dev.jsonl')])
ctl[f'POS opallo41 vs splits/train 800 prefix [expect {n41_in_800} by id]'] = detect(P41, Ptrain800)
rng = random.Random(0)
plant = rng.sample(Ptitan, 20)
ctl['POS Opallo family-disjoint pool + 20 planted Titan rows vs Titan [expect exact_user>=20]'] = detect(Pfd + plant, Ptitan)
ctl['POS T_BANK_V2 vs Titan, SYSTEM 8-grams [verbatim shared system prompts, expect 192]'] = detect(Ptb, Ptitan)
ctl['POS rehearsal conversational rows vs splits_large train (v3 preservation slice)'] = detect(Prehe_conv, [rin(r) for r in train])
# NEGATIVE controls: must not fire on exact payload.
shuf_ans = [a for _, _, a in Ptitan]
rng.shuffle(shuf_ans)
Ptitan_sh = [(s, u, a) for (s, u, _), a in zip(Ptitan, shuf_ans)]
ctl['NEG Titan payloads with answers shuffled vs Titan [exact_payload expect ~0]'] = detect(Ptitan_sh, Ptitan)
ctl['NEG monitor_probe vs splits_large train [family-disjoint by manifest]'] = detect(Pprobe, [rin(r) for r in train])
# QUESTIONS
q = {}
q['Titan384 vs Opallo41'] = detect(Ptitan, P41)
q['Titan384 vs Opallo held-out same-class (train rest)'] = detect(Ptitan, Psame)
q['Opallo held-out same-class vs Titan384'] = detect(Psame, Ptitan)
q['dev150 vs Titan384'] = detect(Pdev, Ptitan)
q['T_BANK_V2 vs Opallo41'] = detect(Ptb, P41)
q['T_BANK_V2 vs Opallo held-out same-class'] = detect(Ptb, Psame)
q['T_BANK_V2 vs Titan384 (user/payload; disjoint by construction)'] = detect(Ptb, Ptitan)
q['dev150 vs Opallo41'] = detect(Pdev, P41)
q['Opallo held-out family-disjoint vs Opallo41'] = detect(Pfd, P41)
q['rehearsal conversational (648) vs Opallo41'] = detect(Prehe_conv, P41)
for name, d in list(ctl.items()) + list(q.items()):
    print(f'{name}: {d}')
titan_fams = sorted({r['family'] for r in titan})
res = {'classes': classes, 'per_class': per_class, 'train_rest': len(rest), 'same_class': len(same),
       'family_siblings': len(sib), 'family_disjoint': len(fd), 'other_files': other,
       'titan_corpus': {'path': str(TC), 'sha256': sha(TC), 'rows': len(titan),
                        'families': dict(collections.Counter(r['family'] for r in titan)),
                        'class_overlap_with_opallo': sorted(set(titan_fams) & set(classes))},
       'rehearsal_families': dict(collections.Counter(r['family'] for r in rehe)),
       'controls': ctl, 'questions': q, 'opallo41_in_800_prefix': n41_in_800}
print('wrote q2_heldout.json', dump('q2_heldout.json', res))
