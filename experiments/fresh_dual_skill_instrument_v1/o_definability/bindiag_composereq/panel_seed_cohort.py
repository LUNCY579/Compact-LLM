# Stability of the Opallo-specific (Opallo correct, Supa wrong) panel cohort across the four b5200 seeds
import hashlib, importlib.util, json, collections, os
AB = 'C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/experiments/architecture_benchmark_001'
OUT = os.path.dirname(os.path.abspath(__file__))
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
RP = AB + '/retention.py'; assert sha(RP).startswith('c52b6a48')
spec = importlib.util.spec_from_file_location('_r2_seed_retention', RP)
ret = importlib.util.module_from_spec(spec); spec.loader.exec_module(ret)
audit = json.load(open(AB + '/evidence/retention-design-v1/PROBE_LABEL_AUDIT.json', encoding='utf-8'))
ov = {n: ret.build_overlay(audit, n)['targets'] for n in ret.BARE_POLICIES}
ids = set(ov['strict'])
ident = json.load(open(OUT + '/identity_ids.json'))
S = {p: set(v) for p, v in ident['supa_B28'].items()}
cohorts = {}
for seed in (17, 23, 29, 41):
    f = AB + f'/evidence/retention-design-v1/ARM_DOSE_B28last20_lr1e-05_b5200_seed{seed}_wsd0.2-0.2_b20.95.json'
    d = json.load(open(f, encoding='utf-8')); outs = d['checkpoints'][-1]['outputs']
    assert set(outs) == ids
    c = {p: {i for i in ids if ret.score_prediction(outs[i]['text'], outs[i]['terminated'], ov[p][i])['correct']} for p in ov}
    cohorts[seed] = {p: c[p] - S[p] for p in ov}
    print('seed', seed, sha(f)[:8], {p: len(c[p]) for p in ov}, 'gains vs Supa', {p: len(cohorts[seed][p]) for p in ov})
for p in ov:
    inter = set.intersection(*(cohorts[s][p] for s in cohorts)); union = set.union(*(cohorts[s][p] for s in cohorts))
    k3 = {i for i in union if sum(i in cohorts[s][p] for s in cohorts) >= 3}
    print(p, 'gain in ALL 4 seeds', len(inter), dict(collections.Counter(i.split('__')[1] for i in inter)),
          '| >=3 seeds', len(k3), '| any seed', len(union), '| seed17 cohort', len(cohorts[17][p]))
