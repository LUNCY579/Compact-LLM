"""Q1: the 41 rows Opallo was trained on -- classes, families, format, 3 verbatim examples."""
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_DOSE, PROBE, TRAIN, dump, jl, sha  # noqa: E402

assert sha(TRAIN).startswith('cd57e650')
arm = json.loads(ARM_DOSE.read_text(encoding='utf-8'))
ids = arm['train_row_ids_consumed']
assert len(ids) == 41 == arm['train_rows_consumed'] == arm['optimizer_steps'], (len(ids), arm['optimizer_steps'])
train = jl(TRAIN)
byid = {r['id']: r for r in train}
assert len(byid) == len(train) == 10000

# CANARY 1: the consumed ids are the first 41 of random.Random(17).shuffle(train), as arm_dose.py does.
sh = list(train)
random.Random(17).shuffle(sh)
canary_shuffle = [r['id'] for r in sh[:41]] == ids
# CANARY 2: train_row_ids_sha256 recipe
recipes = {name: hashlib.sha256(s.encode()).hexdigest() for name, s in
           [('join_nl', '\n'.join(ids)), ('json', json.dumps(ids)), ('json_compact', json.dumps(ids, separators=(',', ':'))),
            ('join_nl_trail', '\n'.join(ids) + '\n')]}
sha_match = [k for k, v in recipes.items() if v == arm['train_row_ids_sha256']]
print('canary: consumed ids == Random(17).shuffle(train)[:41]:', canary_shuffle,
      '| train_row_ids_sha256 recipe matched:', sha_match or 'none of ' + str(list(recipes)))

sel = [byid[i] for i in ids]
rc = collections.Counter(r['reply_class'] for r in sel)
dom = collections.Counter(r['id'].split('__')[2].split('_')[0] for r in sel)
fam = collections.Counter(r['group_id'] for r in sel)
print(f'41 rows: {len(rc)} reply classes, {len(fam)} families (max {max(fam.values())}/family); '
      f'domain prefix {dict(dom)}')
cls_dom = collections.defaultdict(set)
for r in train:
    cls_dom[r['reply_class']].add(r['id'].split('__')[2].split('_')[0])
for k, v in sorted(rc.items(), key=lambda x: (-x[1], x[0])):
    print(f'   {v:2d}  {k}  [{",".join(sorted(cls_dom[k]))}]')


def parts(t):
    sysm = re.search(r'<\|im_start\|>system\n(.*?)<\|im_end\|>', t, re.S).group(1)
    users = re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', t, re.S)
    asst = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', t, re.S)
    final = asst[-1].split('</think>\n\n', 1)[1] if '</think>\n\n' in asst[-1] else asst[-1]
    return sysm, users, asst, final


P = [parts(r['text']) for r in sel]
sysc = collections.Counter(p[0] for p in P)
print('distinct system prompts among 41:', len(sysc))
for s, n in sysc.most_common():
    print(f'   {n:2d}  {s[:110]!r}')
print('user turns per row:', dict(collections.Counter(len(p[1]) for p in P)),
      '| assistant turns per row:', dict(collections.Counter(len(p[2]) for p in P)))
print('final assistant turn has empty <think>\\n\\n</think>\\n\\n scaffold:',
      sum(p[2][-1].startswith('<think>\n\n</think>\n\n') for p in P), 'of 41')
fl = sorted(len(p[3]) for p in P)
wl = sorted(len(p[3].split()) for p in P)
print('final answer chars min/median/max', fl[0], fl[20], fl[-1], '| words', wl[0], wl[20], wl[-1])
digits = sum(bool(re.search(r'\d', p[3])) for p in P)
print('final answers containing a digit:', digits, 'of 41')
# Whole-corpus system prompt distribution (is the system prompt a per-row variable or boilerplate?)
allsys = collections.Counter(parts(r['text'])[0] for r in train)
print('train.jsonl distinct system prompts:', len(allsys), '| top-3 share:',
      round(sum(n for _, n in allsys.most_common(3)) / len(train), 3))

probe_classes = collections.Counter(r['reply_class'] for r in jl(PROBE))
print('probe classes', dict(probe_classes), '| overlap with the 41 rows classes:',
      sorted(set(probe_classes) & set(rc)), '| with all train.jsonl classes:',
      sorted(set(probe_classes) & {r['reply_class'] for r in train}))

# Three verbatim examples, chosen deterministically: first row (by consumed order) of the three
# most frequent domain prefixes.
ex = []
for d in ['reas', 'comm', 'inte']:
    k = next(j for j, r in enumerate(sel) if r['id'].split('__')[2].startswith(d + '_'))
    s, u, a, f = P[k]
    ex.append({'order': k, 'id': sel[k]['id'], 'reply_class': sel[k]['reply_class'],
               'system': s, 'user_last_head': u[-1][:400], 'n_user_turns': len(u),
               'assistant_final': f})
for e in ex:
    print(f"=== EX consumed#{e['order']} {e['id']} [{e['reply_class']}] user_turns={e['n_user_turns']}")
    print('SYSTEM:', e['system'][:160])
    print('USER  :', e['user_last_head'])
    print('ASSIST:', repr(e['assistant_final']))
res = {'ids': ids, 'canary_shuffle': canary_shuffle, 'ids_sha_recipe': sha_match,
       'reply_class_counts': dict(rc), 'domain_prefix': dict(dom), 'families': len(fam),
       'system_prompts_41': dict(sysc), 'examples': ex,
       'final_answer_chars': fl, 'final_answers_with_digit': digits}
print('wrote q1_rows.json', dump('q1_rows.json', res))
