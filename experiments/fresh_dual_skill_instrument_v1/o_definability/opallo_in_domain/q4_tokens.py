"""Q4: tokenizer-only accounting (no model) for a teacher-forced gold-NLL O-endpoint and S-endpoint.

Canaries (must collide with recorded numbers before anything downstream is read):
  C1 per_step_target_tokens of the 41 Opallo rows == len(enc(text)) - 1        (ARM_DOSE)
  C2 dev[:40] target positions == 4979, reply positions == 381                (RESPONSE_CE_SCAN)
  C3 Titan microfit ids re-rendered from prompt + reference_text == recorded ids, and the
     supervised count == recorded supervised_tokens (sum 5254)                (control-PLAN)
Partition controls:
  O side: format positions (<|im_end|>, trailing newline, a final '.'-ending token) vs content.
  S side: gold tokens that the COPY baseline already contains (difflib 'equal' blocks) vs changed.
          POS control: the 13 answer==input rows must have 0 changed tokens.
          NEG control: aligning each gold to ANOTHER row's copy baseline must raise changed tokens.
"""
import collections
import difflib
import json
import random
import re
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_DOSE, DEV, TBANK, TITAN_DIR, TOKENIZER, TRAIN, dump, jl, sha  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

assert sha(TOKENIZER).startswith('aeb13307'), sha(TOKENIZER)
tok = Tokenizer.from_file(str(TOKENIZER))
enc = lambda t: tok.encode(t, add_special_tokens=False).ids
MARKER = enc('<|im_start|>assistant\n')
THINK_END = enc('</think>\n\n')
IM_END = enc('<|im_end|>')
NL = enc('\n')
assert len(IM_END) == 1 and len(NL) == 1


def reply_cut(seq):
    """Index of the first reply token: after the LAST assistant marker and its empty think block
    (same rule as rab001_exploration/response_ce_scan.py:response_mask)."""
    cut = -1
    for i in range(len(seq) - len(MARKER), -1, -1):
        if seq[i:i + len(MARKER)] == MARKER:
            cut = i + len(MARKER)
            break
    if cut >= 0:
        for i in range(cut, len(seq) - len(THINK_END) + 1):
            if seq[i:i + len(THINK_END)] == THINK_END:
                cut = i + len(THINK_END)
                break
    return cut


def account(text):
    seq = enc(text)
    cut = reply_cut(seq)
    resp = seq[cut:]
    body = list(resp)
    fmt = 0
    if body and body[-1] == NL[0]:
        fmt += 1
        body = body[:-1]
    if body and body[-1] == IM_END[0]:
        fmt += 1
        body = body[:-1]
    period = int(bool(body) and tok.decode([body[-1]]).endswith('.'))
    return {'targets': len(seq) - 1, 'reply': len(resp), 'content': len(body), 'fmt_end': fmt,
            'final_period_tok': period, 'final_char': (tok.decode(body)[-1:] if body else '')}


arm = json.loads(ARM_DOSE.read_text(encoding='utf-8'))
ids = arm['train_row_ids_consumed']
train = jl(TRAIN)
byid = {r['id']: r for r in train}
sel = [byid[i] for i in ids]
A41 = [account(r['text']) for r in sel]
c1 = [a['targets'] for a in A41] == arm['per_step_target_tokens']
dev = jl(DEV)
D40 = [account(r['text']) for r in dev[:40]]
c2 = (sum(a['targets'] for a in D40), sum(a['reply'] for a in D40))
print(f'C1 per_step_target_tokens reproduced: {c1} (sum {sum(a["targets"] for a in A41)} vs recorded '
      f'{arm["supervised_target_tokens"]})')
print(f'C2 dev[:40] targets/reply = {c2} (recorded 4979/381): {c2 == (4979, 381)}')
assert c1 and c2 == (4979, 381), 'tokenizer canary failed -- stop'


def summ(name, A):
    T = sum(a['targets'] for a in A)
    R = sum(a['reply'] for a in A)
    C = sum(a['content'] for a in A)
    F = sum(a['fmt_end'] for a in A) + sum(a['final_period_tok'] for a in A)
    fc = collections.Counter(a['final_char'] for a in A)
    med = statistics.median(a['reply'] for a in A)
    print(f'{name:34s} rows {len(A):5d} targets {T:8d} reply {R:7d} ({R / T:6.2%}) reply/row median {med:4.1f} | '
          f'format (im_end + nl + final-period token) {F} = {F / R:6.2%} of reply | content {C - sum(a["final_period_tok"] for a in A)} '
          f'| final char {dict(fc.most_common(4))}')
    return {'rows': len(A), 'targets': T, 'reply': R, 'reply_frac': R / T, 'format': F, 'format_frac_of_reply': F / R,
            'reply_median': med, 'final_char': dict(fc)}


idset = set(ids)
classes = {r['reply_class'] for r in sel}
held = [r for r in train if r['id'] not in idset and r['reply_class'] in classes]
other = [r for r in train if r['id'] not in idset and r['reply_class'] not in classes]
res = {'canary_C1': c1, 'canary_C2': c2}
res['opallo41'] = summ('Opallo 41 trained rows', A41)
res['dev40'] = summ('dev[:40] (recorded dev_ce rows)', D40)
res['dev150'] = summ('dev150', [account(r['text']) for r in dev])
res['dev150_opallo_classes'] = summ('dev150 in Opallo classes', [account(r['text']) for r in dev if r['reply_class'] in classes])
Ah = [account(r['text']) for r in held]
res['heldout_same_class'] = summ('train rest, Opallo classes', Ah)
res['heldout_other_class'] = summ('train rest, the 12 other classes', [account(r['text']) for r in other])

# Canned-answer rate: can the O reply be predicted from class alone?
final = lambda r: r['text'].rsplit('</think>\n\n', 1)[1].replace('<|im_end|>\n', '').strip()
norm = lambda s: re.sub(r'\s+', ' ', s.lower()).strip()
ans41 = {norm(final(r)) for r in sel}
dup41 = sum(norm(final(r)) in ans41 for r in held)
cnt = collections.Counter(norm(final(r)) for r in held)
top = cnt.most_common(8)
distinct = len(cnt)
print(f'held-out same-class answers identical to one of the 41 trained answers: {dup41}/{len(held)}; '
      f'distinct answers {distinct}/{len(held)}; top repeated {top[:5]}')
percls = {}
for c in sorted(classes):
    cc = collections.Counter(norm(final(r)) for r in held if r['reply_class'] == c)
    n = sum(cc.values())
    percls[c] = {'n': n, 'distinct': len(cc), 'top_share': cc.most_common(1)[0][1] / n, 'top': cc.most_common(1)[0][0]}
worst = sorted(percls.items(), key=lambda kv: -kv[1]['top_share'])[:6]
print('classes with the most canned answers (top-answer share):',
      [(k, round(v['top_share'], 3), v['top']) for k, v in worst])
res['canned'] = {'dup_with_41': dup41, 'n': len(held), 'distinct': distinct, 'top': top, 'per_class': percls}

# ------------------------------------------------------------------ S side
TC = TITAN_DIR / 'control-prepared' / 'microfit-train.jsonl'
titan = jl(TC)


def render(prompt, answer):
    s = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in prompt)
    p = s + '<|im_start|>assistant\n<think>\n\n</think>\n\n'
    return enc(p), enc(answer) + IM_END


ok_ids = ok_sup = 0
for r in titan:
    p, a = render(r['prompt'], r['reference_text'])
    ok_ids += (p == r['prompt_ids'] and p + a + NL == r['ids'])
    ok_sup += (len(a) == r['supervised_tokens'] == sum(1 for x in r['labels'] if x != -100))
c3 = (ok_ids, ok_sup, sum(r['supervised_tokens'] for r in titan))
print(f'C3 Titan microfit re-render: ids identical {ok_ids}/384, supervised count identical {ok_sup}/384, '
      f'sum supervised {c3[2]} (plan 5254)')
assert ok_ids == 384 and ok_sup == 384 and c3[2] == 5254, 'S-side template canary failed -- stop'
res['canary_C3'] = c3

tb = json.loads(TBANK.read_text(encoding='utf-8'))['rows']


def s_account(r, copy):
    p, a = render(r['prompt'], r['reference'])
    gold = a[:-1]
    cp = enc(copy)
    sm = difflib.SequenceMatcher(a=gold, b=cp, autojunk=False)
    eq = sum(bl.size for bl in sm.get_matching_blocks())
    return {'prompt': len(p), 'sup': len(a), 'content': len(gold), 'copy_equal': eq,
            'changed': len(gold) - eq, 'eos': 1}


S = [s_account(r, r['copy_baseline']) for r in tb]
eqrows = [s for s, r in zip(S, tb) if r['answer_equals_input']]
rng = random.Random(0)
perm = list(range(len(tb)))
rng.shuffle(perm)
# NEG control: pair each gold with another row's copy baseline of the SAME family (same format)
fam_idx = collections.defaultdict(list)
for k, r in enumerate(tb):
    fam_idx[r['family']].append(k)
neg = []
for k, r in enumerate(tb):
    pool = [j for j in fam_idx[r['family']] if j != k]
    neg.append(s_account(r, tb[rng.choice(pool)]['copy_baseline']))
tot = lambda L, k: sum(x[k] for x in L)
print(f"T_BANK_V2: rows {len(S)}, supervised (answer+EOS) {tot(S, 'sup')} = {tot(S, 'sup') / len(S):.2f}/row, "
      f"of which EOS {tot(S, 'eos')}; content tokens {tot(S, 'content')}: copy-equal {tot(S, 'copy_equal')} "
      f"({tot(S, 'copy_equal') / tot(S, 'content'):.1%}), CHANGED {tot(S, 'changed')} ({tot(S, 'changed') / tot(S, 'content'):.1%}); "
      f"prompt tokens {tot(S, 'prompt')}")
print(f"   POS control: answer==input rows {len(eqrows)} -> changed tokens {tot(eqrows, 'changed')} (must be 0)")
print(f"   NEG control: gold vs another same-family row's copy baseline -> changed {tot(neg, 'changed')} "
      f"vs {tot(S, 'changed')} with the true copy baseline (must be larger)")
byfam = {}
for s, r in zip(S, tb):
    b = byfam.setdefault(r['family'], collections.Counter())
    for k in ('sup', 'content', 'copy_equal', 'changed'):
        b[k] += s[k]
    b['rows'] += 1
for f, b in byfam.items():
    print(f"   {f:18s} rows {b['rows']} sup {b['sup']} content {b['content']} changed {b['changed']} "
          f"({b['changed'] / b['content']:.1%}) changed/row {b['changed'] / b['rows']:.2f}")
res['tbank'] = {'rows': len(S), 'sup': tot(S, 'sup'), 'content': tot(S, 'content'), 'copy_equal': tot(S, 'copy_equal'),
                'changed': tot(S, 'changed'), 'pos_ctrl_changed': tot(eqrows, 'changed'), 'pos_ctrl_rows': len(eqrows),
                'neg_ctrl_changed': tot(neg, 'changed'), 'by_family': {f: dict(b) for f, b in byfam.items()}}
# Titan's own training rows, same partition (the S skill's training signal)
Tt = [s_account({'prompt': r['prompt'], 'reference': r['reference_text']},
                json.dumps(json.loads(r['prompt'][1]['content'])['before'], separators=(',', ':'))) for r in titan]
print(f"Titan 384 training rows: supervised {tot(Tt, 'sup')}, content {tot(Tt, 'content')}, changed-vs-input "
      f"{tot(Tt, 'changed')} ({tot(Tt, 'changed') / tot(Tt, 'content'):.1%})")
res['titan_train_partition'] = {'sup': tot(Tt, 'sup'), 'content': tot(Tt, 'content'), 'changed': tot(Tt, 'changed')}
print('wrote q4_tokens.json', dump('q4_tokens.json', res))
