# COMPOSE-REQ-001 independent re-score (reimplements score_composereq.py classify/correct;
# that module imports torch at top level so it is NOT imported)
import json, re, collections, math, hashlib, os
D = 'C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/composereq001-20260918'
OUT = os.path.dirname(os.path.abspath(__file__))


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


for f in ['COMPOSEREQ_SCOUT.jsonl', 'COMPOSEREQ_HINTED.jsonl', 'score_composereq.py', 'build_composereq.py',
          'SCORED_frozen_supa.json', 'SCORED_supa_HINTED.json', 'SCORED_H16.json', 'SCORED_TH16s_rebuild.json',
          'JOINT_ENDPOINT_POSITIVE_CONTROL.json', 'COMPOSEREQ_SCOUT_BUILD.json']:
    print('sha', f, sha(D + '/' + f)[:8])
bank = [json.loads(l) for l in open(D + '/COMPOSEREQ_SCOUT.jsonl', encoding='utf-8') if l.strip()]
hint = [json.loads(l) for l in open(D + '/COMPOSEREQ_HINTED.jsonl', encoding='utf-8') if l.strip()]
B = {r['id']: r for r in bank}
print('bank rows', len(bank), 'packs', len({r['pack'] for r in bank}), 'members', dict(collections.Counter(r['member'] for r in bank)))
print('families (rows)', dict(collections.Counter(r['family'] for r in bank)),
      'forms (INS)', dict(collections.Counter(r['form'] for r in bank if r['member'] == 'INSUFFICIENT')))
print('parents distinct', len({r['parent_id'] for r in bank}), 'source views distinct', len({r['source_view_id'] for r in bank}))
print('distinct system prompts', len({r['prompt'][0]['content'] for r in bank}))
packs_all = {r['pack'] for r in bank}
print('pack members share system prompt', all(B[p + '-suf']['prompt'][0] == B[p + '-ins']['prompt'][0] for p in packs_all))
H = {(r['id'][:-5] if r['id'].endswith('-hint') else r['id']): r for r in hint}
print('hinted ids map', set(H) == set(B), 'user identical', all(H[i]['prompt'][1] == B[i]['prompt'][1] for i in B),
      'system differs', sum(H[i]['prompt'][0] != B[i]['prompt'][0] for i in B))
ex = bank[0]['prompt'][0]['content']; hx = H[bank[0]['id']]['prompt'][0]['content']
cp = os.path.commonprefix([ex, hx])
print('HINT system = base system prefix len', len(cp), 'of', len(ex), '| hint-only text:', repr(hx[len(cp):]))
rk = collections.Counter()
for r in bank:
    if r['reference'] == 'UNKNOWN':
        rk['sentinel'] += 1; continue
    o = json.loads(r['reference']); rk['json_' + type(o).__name__] += 1
print('reference kinds', dict(rk))
SENT = 'unknown'


def norm(s): return re.sub(r"\s+", " ", str(s).strip())
def canon(s): return re.sub(r"[.\s]+$", "", norm(s).lower())


def bkeys(r):
    b = json.loads(r['prompt'][1]['content']).get('before')
    return set(b) if isinstance(b, dict) else set()


def classify(out, bk):
    if canon(out).startswith(SENT):
        return 'ABSTAIN'
    t = norm(out); m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            o = json.loads(m.group(0))
            if isinstance(o, dict) and set(o) == bk:
                return 'STATE'
        except Exception:
            pass
    if re.fullmatch(r"-?\d+", t):
        return 'STATE'
    return 'OTHER'


def correct(pred, r, mode='strict'):
    em = classify(pred, bkeys(r))
    if r['member'] == 'INSUFFICIENT':
        return em == 'ABSTAIN'
    if em != 'STATE':
        return False
    f = norm if mode == 'strict' else canon
    try:
        return json.loads(re.search(r"\{.*\}", pred, re.S).group(0)) == json.loads(r['reference'])
    except Exception:
        return f(pred) == f(r['reference'])


def joint(preds, rows=B):
    pk = collections.defaultdict(list)
    for i, p in preds.items():
        pk[rows[i]['pack']].append(correct(p, rows[i]))
    return sum(1 for v in pk.values() if len(v) == 2 and all(v)), len(pk)


res = {}
orc = {i: r['reference'] for i, r in B.items()}
res['POS_oracle_joint'] = joint(orc); print('POS oracle joint', res['POS_oracle_joint'])
ins = [i for i in B if B[i]['member'] == 'INSUFFICIENT']; suf = [i for i in B if B[i]['member'] == 'SUFFICIENT']
print('POS abstain-detector on oracle INS', sum(classify(orc[i], bkeys(B[i])) == 'ABSTAIN' for i in ins), '/', len(ins),
      '| NEG abstain-detector on oracle SUF refs', sum(classify(orc[i], bkeys(B[i])) == 'ABSTAIN' for i in suf), '/', len(suf))
print('POS state-detector on oracle SUF refs', sum(classify(orc[i], bkeys(B[i])) == 'STATE' for i in suf), '/', len(suf))
ss = sorted(suf); rot = ss[1:] + ss[:1]; neg = dict(orc)
for i, j in zip(ss, rot):
    neg[i] = B[j]['reference']
res['NEG_rotated_suf_refs_joint'] = joint(neg); print('NEG rotated SUF references across packs joint', res['NEG_rotated_suf_refs_joint'])
neg2 = dict(orc)
for i in ins:
    neg2[i] = B[i[:-4] + '-suf']['reference']
res['NEG_state_on_ins_joint'] = joint(neg2); print('NEG own-pack state emitted on INS joint', res['NEG_state_on_ins_joint'])
print('NEG always-abstain joint', joint({i: 'UNKNOWN' for i in B}))


def fisher2(a, b, c, d):
    r1 = a + b; c1 = a + c; n = a + b + c + d
    def p(x): return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    p0 = p(a); lo = max(0, c1 - (n - r1)); hi = min(r1, c1)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9))


def mcnemar(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l); return min(1.0, 2 * sum(math.comb(n, x) for x in range(0, k + 1)) / 2 ** n)


BH = dict(B); BH.update({r['id']: r for r in hint})
arms = {}
for f, lab in [('SCORED_frozen_supa.json', 'frozen_supa'), ('SCORED_supa_HINTED.json', 'supa_HINTED'),
               ('SCORED_H16.json', 'H16'), ('SCORED_TH16s_rebuild.json', 'TH16s_rebuild')]:
    d = json.load(open(D + '/' + f, encoding='utf-8'))
    rows = {p['id']: BH[p['id']] for p in d['predictions']}
    preds = {p['id']: p['prediction'] for p in d['predictions']}
    base = {i: (i[:-5] if i.endswith('-hint') else i) for i in preds}
    mism = sum(correct(preds[p['id']], rows[p['id']]) != p['correct_strict'] for p in d['predictions'])
    emm = sum(classify(preds[p['id']], bkeys(rows[p['id']])) != p['emission'] for p in d['predictions'])
    S = [i for i in preds if rows[i]['member'] == 'SUFFICIENT']; I = [i for i in preds if rows[i]['member'] == 'INSUFFICIENT']
    sc = sum(correct(preds[i], rows[i]) for i in S)
    abI = sum(classify(preds[i], bkeys(rows[i])) == 'ABSTAIN' for i in I)
    abS = sum(classify(preds[i], bkeys(rows[i])) == 'ABSTAIN' for i in S)
    anyunk = sum('unknown' in preds[i].lower() for i in preds)
    em = {m: dict(collections.Counter(classify(preds[i], bkeys(rows[i])) for i in preds if rows[i]['member'] == m))
          for m in ('SUFFICIENT', 'INSUFFICIENT')}
    jn = joint(preds, rows)
    trunc = sum(1 for i in preds if classify(preds[i], bkeys(rows[i])) == 'OTHER' and '{' in preds[i] and '}' not in preds[i])
    maxlen = max(len(preds[i]) for i in preds)
    sufok = {base[i] for i in S if correct(preds[i], rows[i])}
    # packs where INS emitted a STATE equal to own SUF reference (confabulated the determined answer)
    same = sum(1 for i in I if classify(preds[i], bkeys(rows[i])) == 'STATE' and
               correct(preds[i], B[base[i][:-4] + '-suf']))
    arms[lab] = {'sufok': sufok}
    fp = fisher2(abI, len(I) - abI, abS, len(S) - abS)
    print(f"{lab:14s} rows {len(preds)} | row-verdict mismatches vs recorded {mism}, emission mismatches {emm} | "
          f"SUF correct {sc}/{len(S)} | abstain INS {abI} SUF {abS} Fisher p {fp:.3f} | joint {jn} | "
          f"rows containing 'unknown' {anyunk} | OTHER open-brace-no-close {trunc} | max chars {maxlen} | "
          f"INS rows emitting own SUF reference {same}")
    print('    emission by member', em)
    res[lab] = {'suf_correct': sc, 'abstain_ins': abI, 'abstain_suf': abS, 'fisher_p': fp, 'joint': jn[0],
                'row_mismatch_vs_recorded': mism, 'emission_mismatch_vs_recorded': emm, 'emission': em,
                'unterminated_json_other': trunc, 'max_chars': maxlen, 'ins_emits_own_suf_reference': same,
                'suf_correct_ids': sorted(sufok)}
for a, b in [('TH16s_rebuild', 'frozen_supa'), ('TH16s_rebuild', 'H16'), ('supa_HINTED', 'frozen_supa'), ('H16', 'frozen_supa')]:
    w = len(arms[a]['sufok'] - arms[b]['sufok']); l = len(arms[b]['sufok'] - arms[a]['sufok'])
    print(f"paired SUF correctness {a} minus {b}: net {w - l} wins {w} losses {l} exact McNemar p {mcnemar(w, l):.4g}")
    res[f'paired_{a}_minus_{b}'] = {'net': w - l, 'wins': w, 'losses': l, 'p': mcnemar(w, l)}


def anomaly(r):
    b = json.loads(r['prompt'][1]['content']); ev = b['event']; before = b['before']
    if isinstance(ev.get('amount'), str):
        return True
    if isinstance(ev.get('kind'), str) and ' or ' in ev['kind']:
        return True
    for k in ('account', 'other', 'register'):
        if isinstance(ev.get(k), str) and ev[k] not in before:
            return True
    return False


fi = sum(anomaly(B[i]) for i in ins); fs = sum(anomaly(B[i]) for i in suf)
print('structural answer-free predicate: fires on INS', fi, '/64 (POS)  on SUF', fs, '/64 (NEG)')
res['structural_predicate'] = {'INS': fi, 'SUF': fs}
for lab in arms:
    print(lab, 'SUF correct by family', dict(collections.Counter(B[i]['family'] for i in arms[lab]['sufok'])))
print('SUF rows by family', dict(collections.Counter(B[i]['family'] for i in suf)))
json.dump(res, open(OUT + '/crq_results.json', 'w'), indent=1, default=list)
