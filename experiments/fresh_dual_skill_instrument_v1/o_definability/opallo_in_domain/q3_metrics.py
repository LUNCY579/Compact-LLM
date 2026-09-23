"""Q3: every recorded in-domain metric for B28 arms, and what it can / cannot say.

Recorded JSON only. Arithmetic on recorded aggregates is marked VERIFIED only when every input
is a recorded number; cross-recipe interpolation is INFERRED.
"""
import glob
import json
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AB, ARM_DOSE, TITAN_DIR, dump, sha  # noqa: E402

out = {}
# ---------------------------------------------------------------- (a) every ARM_DOSE B28* file
rows = []
for p in sorted(glob.glob(str(AB / 'evidence' / 'retention-design-v1' / 'ARM_DOSE_B28*.json'))):
    d = json.loads(Path(p).read_text(encoding='utf-8'))
    h = d.get('headline') or {}
    last = d['checkpoints'][-1] if d.get('checkpoints') else {}
    rows.append({'file': Path(p).name, 'sha8': sha(p)[:8], 'arm': d.get('arm'), 'completed': d.get('completed'),
                 'tokens': d.get('supervised_target_tokens'), 'steps': d.get('optimizer_steps'),
                 'rows': d.get('train_rows_consumed'), 'split': (d.get('data_sha256') or {}).get('train', '')[:8],
                 'dev_ce_start': h.get('dev_ce_start'), 'dev_ce_end': h.get('dev_ce_end'),
                 'panel_perm_end': h.get('correct_end'), 'retained_end': h.get('retained_end'),
                 'strict_end': ((last.get('retention') or {}).get('strict') or {}).get('correct_total')})
print(f"{'file':78s} arm        done tokens  steps rows split    devCE0  devCE1  panel ret strict")
for r in rows:
    f = lambda x, n=4: ('%.*f' % (n, x)) if isinstance(x, float) else str(x)
    print(f"{r['file'][:78]:78s} {str(r['arm']):10s} {str(r['completed'])[0]} {str(r['tokens']):>7s} {str(r['steps']):>5s} "
          f"{str(r['rows']):>5s} {r['split']} {f(r['dev_ce_start'])} {f(r['dev_ce_end'])} {str(r['panel_perm_end']):>5s} "
          f"{str(r['retained_end']):>3s} {str(r['strict_end'])}")
out['arm_dose_b28'] = rows
done = [r for r in rows if r['completed'] and isinstance(r['dev_ce_end'], float) and r['panel_perm_end'] is not None]


def spearman(x, y):
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            for k in range(i, j + 1):
                r[s[k]] = (i + j) / 2
            i = j + 1
        return r
    a, b = rk(x), rk(y)
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** 0.5
    return num / den if den else float('nan')


rho = spearman([r['dev_ce_end'] for r in done], [r['panel_perm_end'] for r in done])
print(f'completed B28* ARM_DOSE arms with dev_ce_end and panel: n={len(done)}; Spearman(dev_ce_end, panel) = {rho:.3f}')
out['spearman_devce_panel'] = {'n': len(done), 'rho': rho}

# ---------------------------------------------------------------- (b) Opallo's own trace
arm = json.loads(ARM_DOSE.read_text(encoding='utf-8'))
tr = arm['step_trace']
ce = [s['train_ce'] for s in tr]
tok = [s['tokens'] for s in tr]
per = arm['per_step_target_tokens']
assert len(ce) == 41 and len(per) == 41
first, last5 = statistics.mean(ce[:5]), statistics.mean(ce[-5:])
# token-weighted
w_first = sum(c * n for c, n in zip(ce[:10], per[:10])) / sum(per[:10])
w_last = sum(c * n for c, n in zip(ce[-10:], per[-10:])) / sum(per[-10:])
rho_t = spearman(list(range(41)), ce)
print(f"Opallo step_trace train_ce (full-sequence CE on each row at its FIRST and only exposure, i.e. "
      f"online held-out-until-seen): steps1-5 mean {first:.3f}, steps37-41 mean {last5:.3f}; token-weighted "
      f"first10 {w_first:.3f} last10 {w_last:.3f}; Spearman(step, ce) {rho_t:.3f}")
print('   recorded dev_ce (full-sequence, splits_large/dev.jsonl[:40]):', arm['headline']['dev_ce_start'], '->',
      arm['headline']['dev_ce_end'], '| checkpoints dev_ce_nats:', [c['dev_ce_nats'] for c in arm['checkpoints']])
out['opallo_trace'] = {'train_ce': ce, 'first5': first, 'last5': last5, 'tw_first10': w_first, 'tw_last10': w_last,
                       'spearman_step_ce': rho_t, 'dev_ce_start': arm['headline']['dev_ce_start'],
                       'dev_ce_end': arm['headline']['dev_ce_end']}

# ---------------------------------------------------------------- (c) response vs boilerplate bound
scan = json.loads((AB / 'outputs' / 'response_ce' / 'RESPONSE_CE_SCAN.json').read_text(encoding='utf-8'))
src = scan['arms']['source_control']
f_r = scan['dev_response_fraction']
drop = arm['headline']['dev_ce_start'] - arm['headline']['dev_ce_end']
max_resp_contrib = f_r * src['dev_response_ce_nats']   # response CE cannot go below 0
print(f"RESPONSE_CE_SCAN (source, dev40): overall {src['dev_ce_nats']:.4f}, response {src['dev_response_ce_nats']:.4f} "
      f"over {scan['dev_response_target_tokens']} of {scan['dev_target_tokens']} positions (f={f_r:.4f}), boilerplate "
      f"{src['dev_boilerplate_ce_nats']:.4f}")
print(f"   Opallo dev_ce drop {drop:.4f} nats; the reply positions can contribute at most f*respCE_src = "
      f"{max_resp_contrib:.4f} nats (if reply CE fell to 0) -> at least {100 * (1 - max_resp_contrib / drop):.1f}% of the "
      f"recorded dev_ce improvement is NON-reply (system/user/template) positions")
out['bound'] = {'dev_ce_drop': drop, 'resp_fraction': f_r, 'src_resp_ce': src['dev_response_ce_nats'],
                'max_resp_contrib': max_resp_contrib, 'min_boilerplate_share': 1 - max_resp_contrib / drop,
                'dev_ce_start_scan_vs_armdose_diff': src['dev_ce_nats'] - arm['headline']['dev_ce_start']}

# ---------------------------------------------------------------- (d) LONGBUDGET: response CE recorded for a trained arm
lb = json.loads((AB / 'outputs' / 'longbudget' / 'LONGBUDGET_source_control_seed17_b500000_lr1e-05.json').read_text(encoding='utf-8'))
assert lb['train_fingerprint'].startswith('cd57e650') and lb['dev_target_tokens'] == scan['dev_target_tokens']
traj = [(c['tokens'], c['steps'], c['dev_ce_nats'], c['dev_response_ce_nats'], c['dev_response_top1_agreement'],
         c['dev_response_kl_nats']) for c in lb['checkpoints']]
print('LONGBUDGET source_control (596M all trainable, AdamW 1e-5, accum 8, cosine, CPU, SAME train.jsonl & dev40):')
for t in traj:
    share = (f_r * (traj[0][3] - t[3]) / (traj[0][2] - t[2])) if t[2] != traj[0][2] else float('nan')
    print(f'   tokens {t[0]:7d} steps {t[1]:4d} devCE {t[2]:.4f} respCE {t[3]:.4f} (drop {traj[0][3] - t[3]:+.4f}) '
          f'resp top1-vs-src {t[4]:.3f} respKL {t[5]:.4f} | reply share of devCE drop {share:.3%}')
# INFERRED: place Opallo on this curve by its dev CE (cross-recipe; different trainable set, schedule, steps).
de = arm['headline']['dev_ce_end']
seg = next((a, b) for a, b in zip(traj, traj[1:]) if a[2] >= de >= b[2])
fr = (seg[0][2] - de) / (seg[0][2] - seg[1][2])
resp_interp = seg[0][3] + fr * (seg[1][3] - seg[0][3])
tok_interp = seg[0][0] + fr * (seg[1][0] - seg[0][0])
print(f'   INFERRED: at Opallo\'s dev CE {de:.4f} the source_control curve sits at ~{tok_interp:.0f} tokens with reply CE '
      f'~{resp_interp:.3f} (drop ~{traj[0][3] - resp_interp:.3f} nats/token = {100 * (traj[0][3] - resp_interp) / traj[0][3]:.1f}%); '
      f'at Opallo\'s actual token count 5233 the curve reply-CE drop is between {traj[0][3] - traj[1][3]:.3f} and '
      f'{traj[0][3] - traj[2][3]:.3f}')
out['longbudget'] = {'traj': traj, 'interp_resp_ce_at_opallo_devce': resp_interp, 'interp_tokens': tok_interp}

# ---------------------------------------------------------------- (e) Titan's recorded response CE (held-in fixed probe)
mf = TITAN_DIR / 'control-journal' / 'runs' / 'train-001' / 'microfit'
bl = json.loads((mf / 'baseline.json').read_text(encoding='utf-8'))
md = json.loads((mf / 'midpoint.json').read_text(encoding='utf-8'))
fn = json.loads((mf / 'final.json').read_text(encoding='utf-8'))
for n, d in [('baseline', bl), ('midpoint', md), ('final', fn)]:
    s = sum(float(r['response_ce_sum']) for r in d['records'])
    t = sum(int(r['response_ce_targets']) for r in d['records'])
    print(f"Titan fixed TRAIN probe {n}: recorded response_ce {d['response_ce']:.4f} | recomputed from records "
          f"{s / t:.4f} over {t} targets | canonical_correct {d['canonical_correct']}/{d['rows']}")
b = {r['id']: float(r['response_ce_sum']) / int(r['response_ce_targets']) for r in bl['records']}
f_ = {r['id']: float(r['response_ce_sum']) / int(r['response_ce_targets']) for r in fn['records']}
fam = {r['id']: r['family'] for r in fn['records']}
gains = [b[i] - f_[i] for i in b]
imp = sum(g > 0 for g in gains)
print(f'   per-row paired gain (baseline - final, nats/token): mean {statistics.mean(gains):.4f} sd {statistics.stdev(gains):.4f} '
      f'min {min(gains):.4f} max {max(gains):.4f}; rows improved {imp}/{len(gains)}; '
      f'paired t = {statistics.mean(gains) / (statistics.stdev(gains) / len(gains) ** 0.5):.2f}')
by = {}
for i, g in zip(b, gains):
    by.setdefault(fam[i], []).append(g)
print('   by family:', {k: (len(v), round(statistics.mean(v), 4)) for k, v in by.items()})
out['titan_fixed_probe'] = {'baseline': bl['response_ce'], 'midpoint': md['response_ce'], 'final': fn['response_ce'],
                            'gains': gains, 'improved': imp, 'n': len(gains)}

# ---------------------------------------------------------------- (f) the official ladder's B28 arm
q = json.loads((AB / 'outputs' / 'quality-a329eb7b' / 'B28' / 'G1_QUALITY_B28.json').read_text(encoding='utf-8'))
g2 = json.loads((AB / 'outputs' / 'engineering-a329eb7b' / 'b28' / 'G2_CACHED_B28.json').read_text(encoding='utf-8'))
g2r = json.loads((AB / 'outputs' / 'retention-a329eb7b' / 'B28-trained' / 'retention.json').read_text(encoding='utf-8'))
chg = g2r.get('arm_matches_source_verbatim')
print(f"RAB-001 ladder B28: G1_QUALITY_B28 {q['status']} calibration CE source {q['metrics']['mean_ce_source_nats']:.4f} "
      f"arm {q['metrics']['mean_ce_arm_nats']:.4f} (identity); G2_CACHED_B28 {g2['status']} engineering_only="
      f"{g2['metrics']['engineering_only']} {g2['metrics']['recipe']['steps']} SGD steps KD alpha {g2['metrics']['recipe']['alpha']} "
      f"losses {g2['metrics']['losses'][0]:.4f} -> {g2['metrics']['losses'][-1]:.4f}; B28-trained retention "
      f"arm_matches_source_verbatim={chg}")
out['ladder_b28'] = {'g1q_ce': q['metrics']['mean_ce_source_nats'], 'g2_losses': [g2['metrics']['losses'][0], g2['metrics']['losses'][-1]],
                     'b28_trained_matches_source_verbatim': chg}
print('wrote q3_metrics.json', dump('q3_metrics.json', out))
