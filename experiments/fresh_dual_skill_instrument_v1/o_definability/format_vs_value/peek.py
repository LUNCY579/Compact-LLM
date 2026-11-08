import json, sys
sys.dont_write_bytecode = True
sys.path.insert(0, '.')
from common import load_all
R, hashes, probe, audit, targets, supa, panel, fp32, arms = load_all()
aud = {r['id']: r for r in audit['rows']}
for suf in ['00828', '02213', '02677', '00982', '02842', '00280', '02025', '01429', '02552']:
    i = [k for k in aud if k.endswith(suf)][0]
    a = aud[i]
    print(suf, json.dumps(a['proposed_typed_target']))
    print('   ', a['operative_prompt'])
    print('   ', a['original_answer'], '|', a['independent_recomputation'].get('expression'))
from collections import Counter
print(Counter((r['reply_class'], r['proposed_typed_target']['type']) for r in audit['rows']))
