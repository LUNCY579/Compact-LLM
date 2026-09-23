"""H16 (abstention-rows delta, rank 16) Jaccard against every independent Opallo / Titan / D1 draw."""
import json, sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import overlap as O
out = {}
for p in O.POLS:
    g = O.gained('compose001_H16', p)
    r = {}
    for fam, d in (('opallo', O.OPALLO_SEEDS), ('titan', O.TITAN_SEEDS), ('d1', O.D1_SEEDS)):
        js = {k: round(O.overlap_stats(g, O.gained(v, p), O.WRONG[p])['jaccard'], 4) for k, v in d.items()}
        r[fam] = {'per_draw': js, 'mean': round(sum(js.values()) / len(js), 4)}
    out[p] = r
(HERE / 'h16_pairs.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps(out, indent=1))
