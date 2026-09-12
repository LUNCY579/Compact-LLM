"""Where does the displaced-layer penalty actually land: in the answer, or in the boilerplate?

Measured on TRAIN, only ~10% of a row's tokens are the assistant reply. The other ~90% is the
system prompt, the user turn and chat-template scaffolding. Every cross-entropy number in this
programme so far is an average over ALL target positions, so ~90% of it measures how well a
model reproduces text it has seen in every single row.

That matters because this programme has repeatedly measured CE decoupling from capability in
both directions, and has never had a mechanism for it. A candidate mechanism: a converted model
can recover the boilerplate cheaply -- it is high-frequency and low-entropy -- and post a large
CE improvement while learning nothing about answering. If so, the response-restricted CE should
separate the arms far more sharply than the overall CE does.

This tests that, at INITIALIZATION only. No training, no optimizer, one arm resident at a time.

Exploratory. Outside the hashed root; no receipt, no gate. Open splits only.

usage: response_ce_scan.py [n_dev] [arms_csv]
"""
import gc, json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification')
from experiments.architecture_benchmark_001.run import _load, _restore, _load_tokenizer

E = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification'
         r'\experiments\architecture_benchmark_001')
SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble'
           r'\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')
LAD = E / 'outputs/ladder-fc44701e-v1'
SPL = E / 'outputs/splits'

N_DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 40
ARMS = (sys.argv[2].split(',') if len(sys.argv) > 2
        else ['source_control', 'ced_last4', 'ced_last8', 'C28'])
DISPLACED = {'source_control': 0, 'ced_last4': 3, 'ced_last8': 7, 'C28': 13, 'C40': 13}

tok = _load_tokenizer(SRC)
jl = lambda p: [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]
enc = lambda t: tok(t, return_tensors='pt', add_special_tokens=False)['input_ids']
dev = [x for x in (enc(r['text']) for r in jl(SPL / 'dev.jsonl')[:N_DEV]) if x.shape[1] >= 2]

MARKER = tok('<|im_start|>assistant' + chr(10), add_special_tokens=False)['input_ids']
# The assistant turn opens with an empty `<think></think>` scaffold, which is as much
# boilerplate as the system prompt. Excluding it takes the 'reply' region from 10.9%
# of DEV positions down to 7.7% -- the part that actually carries the answer.
THINK_END = tok('</think>' + chr(10) + chr(10), add_special_tokens=False)['input_ids']


def response_mask(ids):
    """True over TARGET positions inside the assistant reply.

    Located in the row's own token sequence, not by re-tokenising a text prefix: a prefix can
    split differently at the boundary and would misalign the mask by a token or two.
    """
    seq = ids[0].tolist()
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
    mask = torch.zeros(len(seq) - 1, dtype=torch.bool)
    if cut >= 0:
        mask[max(0, cut - 1):] = True
    return mask


masks = [response_mask(x) for x in dev]
NTOK = sum(len(m) for m in masks)
NRESP = int(sum(int(m.sum()) for m in masks))
assert NRESP, 'no assistant-reply positions found; the marker or chat template changed'
print(f'dev {len(dev)} rows | {NTOK} target tokens | {NRESP} in the reply = {NRESP / NTOK:.1%}\n',
      flush=True)


def build(arm):
    if arm in ('C28', 'C40'):
        return _restore(LAD / f'G1_CED_{arm}/attempt-0001', 'cpu')
    if arm.startswith('ced_last'):
        from dataclasses import replace as _replace
        from experiments.architecture_benchmark_001.transfer import to_ced
        n = int(arm[len('ced_last'):])
        s, _, _ = _load(SRC, 'cpu')
        e = s.config.total_layers - n
        m, _ = to_ced(s, _replace(s.config, architecture='ced', encoder_layers=e,
                                  decoder_layers=n), donor_layer=e)
        del s; gc.collect()
        return m
    if arm == 'source_control':
        return _load(SRC, 'cpu')[0]
    raise SystemExit(f'unknown arm {arm}')


@torch.inference_mode()
def measure(model):
    ce = rce = bce = 0.0
    model.eval()
    for ids, m in zip(dev, masks):
        lp = F.log_softmax(model(ids).logits.float()[:, :-1], -1)[0]
        per = F.nll_loss(lp, ids[0, 1:], reduction='none')
        ce += float(per.sum()); rce += float(per[m].sum()); bce += float(per[~m].sum())
        del lp, per
    return {'dev_ce_nats': ce / NTOK, 'dev_response_ce_nats': rce / NRESP,
            'dev_boilerplate_ce_nats': bce / (NTOK - NRESP)}


results = {}
base = None
print(f"{'arm':16}{'disp':>5}{'overall':>10}{'reply':>10}{'boiler':>10}"
      f"{'d overall':>11}{'d reply':>10}{'concentration':>15}")
for arm in ARMS:
    t0 = time.perf_counter()
    model = build(arm)
    r = measure(model)
    r['displaced_layers'] = DISPLACED.get(arm)
    if base is None:
        base = r
    d_all = r['dev_ce_nats'] - base['dev_ce_nats']
    d_resp = r['dev_response_ce_nats'] - base['dev_response_ce_nats']
    r['penalty_overall_nats'] = d_all
    r['penalty_response_nats'] = d_resp
    # >1 means the damage is worse in the answer than the average position suggests.
    r['response_concentration'] = (d_resp / d_all) if abs(d_all) > 1e-9 else None
    results[arm] = r
    conc = f"{r['response_concentration']:.2f}x" if r['response_concentration'] else '-'
    print(f"{arm:16}{str(r['displaced_layers']):>5}{r['dev_ce_nats']:>10.4f}"
          f"{r['dev_response_ce_nats']:>10.4f}{r['dev_boilerplate_ce_nats']:>10.4f}"
          f"{d_all:>+11.4f}{d_resp:>+10.4f}{conc:>15}   ({time.perf_counter() - t0:.0f}s)",
          flush=True)
    del model; gc.collect()

OUT = E / 'outputs/response_ce'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'RESPONSE_CE_SCAN.json').write_text(json.dumps({
    'schema': 1, 'evidence_kind': 'exploratory_position_resolved_ce_only',
    'note': 'Cross-entropy split by target position: assistant reply vs prompt/boilerplate. '
            'At INITIALIZATION, no training. Tests whether the displaced-layer penalty is '
            'uniform over positions or concentrated in the answer. Open splits only; no '
            'receipt, no gate, no efficacy claim.',
    'baseline_arm': ARMS[0],
    'dev_rows': len(dev), 'dev_target_tokens': NTOK,
    'dev_response_target_tokens': NRESP, 'dev_response_fraction': NRESP / NTOK,
    'arms': results,
    'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}, indent=2) + '\n', encoding='utf-8')
print('\nwrote', OUT / 'RESPONSE_CE_SCAN.json')
