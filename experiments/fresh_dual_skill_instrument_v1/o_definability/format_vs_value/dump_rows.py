"""Eyeball dump: every row, target, Supa text/reason, Opallo BF16/FP32 texts. Stdout only."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
from common import load_all, score_all  # noqa: E402

R, hashes, probe, audit, targets, supa, panel, fp32, arms = load_all()
scored, correct = score_all(R, arms, targets)
aud = {r['id']: r for r in audit['rows']}
want_cls = sys.argv[1] if len(sys.argv) > 1 else None
for row in probe:
    i = row['id']
    if want_cls and row['reply_class'] != want_cls:
        continue
    t = targets['permissive'][i]
    tv = t.get('minute_of_day', t.get('value'))
    s = scored[('supa', 'permissive')][i]
    ss = scored[('supa', 'strict')][i]
    b = scored[('opallo_bf16', 'permissive')][i]
    f = scored[('opallo_fp32', 'permissive')][i]
    pt = aud[i]['proposed_typed_target']
    disp = pt.get('unit') or pt.get('display_unit') or ''
    tag = ('S' if s['correct'] else '-') + ('s' if ss['correct'] else '-') + '|' + \
          ('B' if b['correct'] else '-') + ('F' if f['correct'] else '-')
    print(f"{i.split('__')[-1]} {row['reply_class'][:4]} {tag} tgt={t['kind']}:{tv} {disp} "
          f"nt={arms['supa'][i].get('new_tokens')} term={s['terminated']}")
    print(f"    SUPA: {s['raw']!r}  [{s['reason']}]")
    print(f"    BF16: {b['raw']!r}  [{b['reason']}]")
    if f['raw'] != b['raw']:
        print(f"    FP32: {f['raw']!r}  [{f['reason']}]")
