"""Step 6: false-negative scan of VALUE calls -- texts with no readable value, or number words."""
import json, re, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
from common import load_all, score_all, OUT, sk  # noqa: E402
from extract import presence  # noqa: E402
R, hashes, probe, audit, targets, supa, panel, fp32, arms = load_all()
scored, correct = score_all(R, arms, targets)
ids = [r['id'] for r in probe]
WORDS = re.compile(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|hundred|half|quarter|dozen)\b', re.I)
out = {}
for arm in ('supa', 'opallo_bf16', 'opallo_fp32'):
    no_val, words = [], []
    for i in ids:
        s = scored[(arm, 'permissive')][i]
        if s['correct'] or not s['terminated']:
            continue
        p = presence(s['raw'], targets['permissive'][i], R)
        if p['loose']:
            continue
        if p['n_values'] == 0:
            no_val.append((sk(i), s['raw']))
        if WORDS.search(s['raw']):
            words.append((sk(i), s['raw']))
    out[arm] = {'value_rows_with_no_readable_value': no_val, 'value_rows_with_number_words': words}
    print(arm, 'VALUE rows with 0 readable values:', len(no_val), no_val)
    print(arm, 'VALUE rows with number words:', len(words), words)
(OUT / 'step6_fn_scan.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
