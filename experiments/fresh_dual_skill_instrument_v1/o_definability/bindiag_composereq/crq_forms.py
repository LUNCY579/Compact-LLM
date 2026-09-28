# COMPOSE-REQ-001: by-form decomposition of INSUFFICIENT behaviour, and a check of the
# "default-recovers-the-twin" leak (range midpoint == original amount; first-listed kind == original kind)
import json, re, collections, os
D = 'C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/composereq001-20260918'
OUT = os.path.dirname(os.path.abspath(__file__))
bank = {json.loads(l)['id']: json.loads(l) for l in open(D + '/COMPOSEREQ_SCOUT.jsonl', encoding='utf-8') if l.strip()}
def body(r): return json.loads(r['prompt'][1]['content'])
# construction leak check (answer-free, on the bank only)
mid_ok = first_ok = n_rng = n_kind = 0
for i, r in bank.items():
    if r['member'] != 'INSUFFICIENT':
        continue
    s = body(bank[i[:-4] + '-suf'])['event']; e = body(r)['event']
    if r['form'] == 'amount_range':
        n_rng += 1
        lo, hi = map(float, re.findall(r'-?\d+(?:\.\d+)?', e['amount']))
        mid_ok += ((lo + hi) / 2 == s['amount'])
    if r['form'] == 'kind_ambiguous':
        n_kind += 1
        first_ok += (e['kind'].split(' or ')[0] == s['kind'])
print(f'LEAK amount_range: midpoint == sufficient amount on {mid_ok}/{n_rng}; kind_ambiguous: first-listed kind == sufficient kind on {first_ok}/{n_kind}')
# negative control for the leak check: compare against a DIFFERENT pack's sufficient twin
ins_r = sorted(i for i, r in bank.items() if r['member'] == 'INSUFFICIENT' and r['form'] == 'amount_range')
rot = dict(zip(ins_r, ins_r[1:] + ins_r[:1])); negm = 0
for i in ins_r:
    lo, hi = map(float, re.findall(r'-?\d+(?:\.\d+)?', body(bank[i])['event']['amount']))
    negm += ((lo + hi) / 2 == body(bank[rot[i][:-4] + '-suf'])['event']['amount'])
print(f'NEG leak control (midpoint vs another pack\'s amount): {negm}/{len(ins_r)}')
ins_k = sorted(i for i, r in bank.items() if r['member'] == 'INSUFFICIENT' and r['form'] == 'kind_ambiguous')
rotk = dict(zip(ins_k, ins_k[1:] + ins_k[:1]))
negk = sum(body(bank[i])['event']['kind'].split(' or ')[0] == body(bank[rotk[i][:-4] + '-suf'])['event']['kind'] for i in ins_k)
print(f'NEG leak control (first-listed kind vs another pack\'s kind): {negk}/{len(ins_k)}  (kind vocabulary has 3 values, so chance agreement is expected)')
def canon(s): return re.sub(r"[.\s]+$", "", re.sub(r"\s+", " ", str(s).strip()).lower())
def as_state(pred):
    m = re.search(r"\{.*\}", re.sub(r"\s+", " ", pred.strip()), re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None
res = {}
for f in ['SCORED_frozen_supa.json', 'SCORED_H16.json', 'SCORED_TH16s_rebuild.json']:
    d = json.load(open(D + '/' + f, encoding='utf-8'))
    t = collections.defaultdict(collections.Counter)
    for p in d['predictions']:
        r = bank[p['id']]
        if r['member'] != 'INSUFFICIENT':
            if canon(p['prediction']).startswith('unknown'):
                t['SUF_abstain_by_form'][r['form']] += 1
            continue
        st = as_state(p['prediction'])
        twin = json.loads(bank[p['id'][:-4] + '-suf']['reference'])
        if canon(p['prediction']).startswith('unknown'): t['ABSTAIN'][r['form']] += 1
        elif st == twin: t['EMITS_SUF_TWIN_ANSWER'][r['form']] += 1
        elif st is not None: t['OTHER_STATE'][r['form']] += 1
        else: t['NONSTATE'][r['form']] += 1
    print(f, {k: dict(v) for k, v in t.items()})
    res[f] = {k: dict(v) for k, v in t.items()}
json.dump({'leak': {'midpoint': [mid_ok, n_rng], 'first_kind': [first_ok, n_kind], 'neg_midpoint': [negm, len(ins_r)], 'neg_kind': [negk, len(ins_k)]}, 'by_form': res},
          open(OUT + '/crq_forms.json', 'w'), indent=1)
