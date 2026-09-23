import json, sys
from collections import defaultdict
from pathlib import Path
sys.dont_write_bytecode = True
inv = json.loads(Path(__file__).with_name('inventory.json').read_text(encoding='utf-8'))
g = defaultdict(lambda: {'kinds': set(), 'weights': set(), 'probe': set(), 'n': 0, 'where': set()})
other = defaultdict(int)
for r in inv:
    p = r['path'].replace(chr(92), '/')
    if 'error' in r: other['ERR'] += 1; continue
    if '/outputs/arm-runs/' in p or '/ladder-' in p: other['arm-runs/ladder'] += 1; continue
    lab = r.get('label') or r.get('binding_label') or Path(p).stem
    x = g[lab]; x['kinds'] |= set(r['kinds']); x['n'] += 1
    if r.get('weight8'): x['weights'].add(r['weight8'])
    if r.get('probe_sha8'): x['probe'].add(r['probe_sha8'])
    x['where'].add('REC' if 'retention-records' in p else ('PILOT' if 'retention-pilot' in p else 'AB'))
rich = lambda ks: any(k.startswith(('PRED_FULL', 'SCORED_FULL', 'BOOL_FULL', 'ROWS_FULL')) for k in ks)
for lab in sorted(g, key=lambda l: (not rich(g[l]['kinds']), l)):
    x = g[lab]
    print(('FULL ' if rich(x['kinds']) else 'part ') + lab[:52].ljust(52), ','.join(sorted(x['kinds']))[:60].ljust(60),
          'w=' + ','.join(sorted(x['weights'])), 'p=' + ','.join(sorted(x['probe'])), ','.join(sorted(x['where'])), x['n'])
print(dict(other))
