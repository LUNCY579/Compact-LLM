# BINDING-DIAGNOSTIC-001 fresh arithmetic bank: structure + independent re-score with the frozen typed scorer
import hashlib, importlib.util, json, collections, os, math
AB = 'C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/experiments/architecture_benchmark_001'
BD = 'C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/binding-diagnostic001-20260920'
OUT = os.path.dirname(os.path.abspath(__file__))
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
RP = AB + '/retention.py'; h = sha(RP); assert h.startswith('c52b6a48'), h
spec = importlib.util.spec_from_file_location('_r2_fresh_retention', RP)
ret = importlib.util.module_from_spec(spec); spec.loader.exec_module(ret)
files = ['candidate-v1/DIAGNOSTIC_INPUTS.jsonl', 'candidate-v1/GOLD.jsonl', 'candidate-v1/METADATA.json', 'candidate-v1/RAW_INPUTS.jsonl',
         'RESULTS.json', 'eval/source/GENERATIONS.json', 'eval/lora/GENERATIONS.json', 'eval/mixed/GENERATIONS.json']
for f in files:
    print('sha', f, sha(BD + '/' + f)[:8])
gold = [json.loads(l) for l in open(BD + '/candidate-v1/GOLD.jsonl', encoding='utf-8') if l.strip()]
inp = [json.loads(l) for l in open(BD + '/candidate-v1/DIAGNOSTIC_INPUTS.jsonl', encoding='utf-8') if l.strip()]
G = {g['id']: g for g in gold}
print('total views', len(gold), 'sections', dict(collections.Counter(g['section'] for g in gold)),
      'inputs', len(inp), 'input ids == gold ids', [r['id'] for r in inp] == [g['id'] for g in gold])
print('input fields', sorted(inp[0]))
ar = {i: g for i, g in G.items() if g['section'] == 'arithmetic'}
bi = {i: g for i, g in G.items() if g['section'] == 'binding'}
print('ARITH views', len(ar), 'parents', len({g['parent_id'] for g in ar.values()}))
print('  family (views)', dict(collections.Counter(g['family'] for g in ar.values())))
print('  family (parents)', dict(collections.Counter(g['family'] for g in {g['parent_id']: g for g in ar.values()}.values())))
print('  subtype (parents)', dict(collections.Counter(g['subtype'] for g in {g['parent_id']: g for g in ar.values()}.values())))
print('  response_type x numeric_variant', dict(collections.Counter((g['response_type'], g['numeric_variant']) for g in ar.values())))
print('  target kinds strict', dict(collections.Counter((g['response_type'], g['targets']['strict']['kind']) for g in ar.values())))
print('BINDING views', len(bi), 'parents', len({g['parent_id'] for g in bi.values()}))
sysp = collections.Counter(r['prompt'][0]['content'] for r in inp if r['id'] in ar)
print('arith system prompts', dict(sysp))
res = {}
Rj = json.load(open(BD + '/RESULTS.json', encoding='utf-8'))
for m in ('source', 'lora', 'mixed'):
    Gm = json.load(open(BD + f'/eval/{m}/GENERATIONS.json', encoding='utf-8'))
    assert Gm['numerics']['max_new_tokens'] == 24, Gm['numerics']
    pred = {p['id']: p for p in Gm['predictions']}
    assert set(ar) <= set(pred)
    sc = {pol: {i: ret.score_prediction(pred[i]['prediction'], pred[i]['terminated'], g['targets'][pol])['correct'] for i, g in ar.items()}
          for pol in ('permissive', 'strict')}
    anchors = [i for i, g in ar.items() if g['numeric_variant'] == 0 and g['response_type'] == 'unit_explicit']
    assert len(anchors) == 96
    a_s = sorted(ar[i]['parent_id'] for i in anchors if sc['strict'][i])
    a_p = sorted(ar[i]['parent_id'] for i in anchors if sc['permissive'][i])
    # identity vs recorded row verdicts
    rows = Rj['endpoints'][m]['rows']
    mism = sum(sc[pol][i] != rows[i][pol + '_correct'] for i in ar for pol in ('permissive', 'strict'))
    tot = Rj['endpoints'][m]['totals']['arithmetic']
    # NEG control: rotate targets by 7 within sorted arithmetic ids
    ids = sorted(ar); rot = dict(zip(ids, ids[7:] + ids[:7]))
    neg = {pol: sum(ret.score_prediction(pred[i]['prediction'], pred[i]['terminated'], ar[rot[i]]['targets'][pol])['correct'] for i in ids)
           for pol in ('permissive', 'strict')}
    tab = collections.defaultdict(lambda: [0, 0, 0])
    for i, g in ar.items():
        k = f"{g['family']}|{g['response_type']}"; tab[k][0] += sc['permissive'][i]; tab[k][1] += sc['strict'][i]; tab[k][2] += 1
    term = sum(pred[i]['terminated'] for i in ar)
    maxgen = max(pred[i]['generated_tokens'] for i in ar)
    atcap = sum(pred[i]['generated_tokens'] >= 24 and not pred[i]['terminated'] for i in ar)
    # parent-level any-view (permissive) and numeric-only both-variants per family
    par = collections.defaultdict(list)
    for i, g in ar.items(): par[g['parent_id']].append(sc['permissive'][i])
    print(f"{m:7s} anchor strict {len(a_s)}/96 permissive {len(a_p)}/96 | all-384 perm {sum(sc['permissive'].values())} strict {sum(sc['strict'].values())} "
          f"(recorded {tot['permissive_correct']}/{tot['strict_correct']}) | row mismatches vs RESULTS {mism}/768 | NEG rotated {neg} | "
          f"terminated {term}/384 | max gen tokens {maxgen} | nonterminated-at-cap {atcap} | parents with >=1 correct view {sum(any(v) for v in par.values())}/96")
    for k in sorted(tab): print('     ', k, 'perm/strict/n', tab[k])
    print('      strict anchor parents', a_s)
    res[m] = {'anchor_strict': a_s, 'anchor_permissive': a_p, 'row_mismatch': mism, 'neg_rotated': neg, 'table': dict(tab),
              'terminated': term, 'nonterminated_at_cap': atcap}
# recorded source anchor cohort
coh = Rj['source_conditioned_retention_descriptive']
print('recorded retention cohort keys', list(coh))
print('recorded primary arithmetic n', Rj['primary']['arithmetic']['n'], 'status', Rj['primary']['arithmetic']['reference_cohort_status'])
# per-family source anchor permissive & numeric-only by family (source only) for O-endpoint headroom
json.dump(res, open(OUT + '/fresh_results.json', 'w'), indent=1)
