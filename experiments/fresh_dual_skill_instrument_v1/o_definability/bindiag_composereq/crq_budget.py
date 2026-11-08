# Decode-budget audit for COMPOSE-REQ-001: re-encode each recorded prediction with the arm's tokenizer.json
# (tokenizers library, CPU, no model) and count how many predictions sit at the 32-token default budget.
import json, re, hashlib, collections, os
from tokenizers import Tokenizer
D = 'C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/composereq001-20260918'
OUT = os.path.dirname(os.path.abspath(__file__))
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
toks = {a: D + f'/arm-{a}/tokenizer.json' for a in ('H16', 'TH16s', 'T16s-solo', 'alpha0836')}
hs = {a: sha(p)[:8] for a, p in toks.items()}
print('tokenizer.json sha by arm dir', hs)
tk = Tokenizer.from_file(toks['H16'])
src = open(D + '/score_composereq.py', encoding='utf-8').read()
print('scorer default max_new_tokens:', re.search(r'--max-new-tokens", type=int, default=(\d+)', src).group(1))
bank = {json.loads(l)['id']: json.loads(l) for l in open(D + '/COMPOSEREQ_SCOUT.jsonl', encoding='utf-8') if l.strip()}
# POSITIVE control for the "at budget" detector: the longest reference state, and a synthetic 40-token string
refs = [r['reference'] for r in bank.values() if r['reference'] != 'UNKNOWN']
rl = [len(tk.encode(x, add_special_tokens=False).ids) for x in refs]
print('reference (compact JSON) token lengths: min', min(rl), 'max', max(rl))
spaced = [json.dumps(json.loads(x)) for x in refs]  # the ", " / ": " spacing the models actually emit
sl = [len(tk.encode(x, add_special_tokens=False).ids) for x in spaced]
print('reference re-serialised with model-style spacing: min', min(sl), 'max', max(sl), '| >=32 tokens:', sum(v >= 32 for v in sl))
out = {}
for f in ['SCORED_frozen_supa.json', 'SCORED_supa_HINTED.json', 'SCORED_H16.json', 'SCORED_TH16s_rebuild.json']:
    d = json.load(open(D + '/' + f, encoding='utf-8'))
    L = [(p, len(tk.encode(p['prediction'], add_special_tokens=False).ids)) for p in d['predictions']]
    hist = collections.Counter(n for _, n in L)
    at = [(p, n) for p, n in L if n >= 31]
    at_suf_other = [p for p, n in at if p['member'] == 'SUFFICIENT' and p['emission'] == 'OTHER']
    # restate-then-patch pattern: duplicate key inside the JSON prefix
    def dupkey(s):
        ks = re.findall(r'"([A-Z])"\s*:', s); return len(ks) != len(set(ks))
    dk = [p for p, n in at if dupkey(p['prediction'])]
    print(f, '| max tokens', max(n for _, n in L), '| rows >=31 tokens', len(at), '(SUF OTHER', len(at_suf_other), ') | at-budget rows with a duplicated key', len(dk))
    out[f] = {'max_tokens': max(n for _, n in L), 'rows_ge31': len(at), 'suf_other_ge31': len(at_suf_other), 'dupkey_ge31': len(dk)}
json.dump(out, open(OUT + '/crq_budget.json', 'w'), indent=1)
