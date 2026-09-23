"""Third-donor check: arms trained on data that is neither transition rows nor reasoning SFT.
compose001_H16 = Supa + rank16(delta_H), delta_H trained on abstention rows only (COMPOSE001_PREREG)."""
import json, sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import core as C
import arms as A
import overlap as O
ARMS = A.ARMS
out = {}
for p in C.POLS:
    W = O.WRONG[p]
    G = lambda n: O.gained(n, p)
    core = {i for i in W if all(i in G(o) for o in O.OPALLO_SEEDS.values()) and all(i in G(t) for t in O.TITAN_SEEDS.values())}
    res = {'core_n': len(core)}
    for n in ('compose001_H16', 'int001_lora596', 'int001_lora596b', 'lwf001_l0', 'rab_H_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95',
              'rab_P_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95', 'rab_B28last20_lr1e-05_b1300_seed17_wsd0.2-0.2_b20.95',
              'rab_B28last8_lr1e-05_b5200_seed17', 'doselora001_norm_retention', 'int001_s25d', 'int001_a100', O.OPALLO['fp32']):
        g = G(n)
        so = O.overlap_stats(g, G(O.OPALLO['fp32']), W)
        st = O.overlap_stats(g, G('int001_a100'), W)
        res[n] = {'correct': len(O.CORRECT[n][p]), 'gained': len(g), 'lost': len(O.lost(n, p)), 'core12_gained': len(g & core),
                  'vs_opallo': (so['inter'], so['jaccard'], so['hg_expected'], f"{so['hg_p_upper']:.2e}"),
                  'vs_titan_dense': (st['inter'], st['jaccard'], st['hg_expected'], f"{st['hg_p_upper']:.2e}")}
    out[p] = res
(HERE / 'third_donor.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
for p in out:
    print('==', p, 'core', out[p]['core_n'])
    for k, v in out[p].items():
        if k != 'core_n': print(k[:50].ljust(50), v)
