"""Initialization quality of every arm against the frozen official source.

Exploratory diagnostic run OUTSIDE the experiment package so the implementation digest and
every existing receipt stay valid. Teacher-forced next-token CE and KL-to-source on the
already-open calibration split. Not a receipt; authorizes nothing.
"""
import gc, json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification')
from experiments.architecture_benchmark_001.run import _load, _restore, _load_tokenizer, _text, load_rows

E = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification\experiments\architecture_benchmark_001')
SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')
LAD = E / 'outputs/ladder-fc44701e-v1'
ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 156

ARMS = [
    ('B28_source', None),
    ('D40_width', LAD / 'G1_WIDTH/attempt-0001'),
    ('C28_ced', LAD / 'G1_CED_C28/attempt-0001'),
    ('C40_ced', LAD / 'G1_CED_C40/attempt-0001'),
]

tok = _load_tokenizer(SRC)
rows = load_rows(E / 'data/calibration.jsonl', 'calibration')[:ROWS]
batches = []
for r in rows:
    ids = _text(tok, r, 'cpu')['input_ids']
    if ids.shape[1] >= 2:
        batches.append((r['id'], ids))
print(f'rows scored: {len(batches)}  tokens: {sum(b.shape[1] for _, b in batches)}', flush=True)

source_logprobs = {}
results = {}

for name, ckpt in ARMS:
    t0 = time.perf_counter()
    if ckpt is None:
        model, _, _ = _load(SRC, 'cpu')
    else:
        model = _restore(ckpt, 'cpu')
    model.eval()
    tot_ce, tot_tok, tot_kl, top1_agree, top1_n = 0.0, 0, 0.0, 0, 0
    with torch.inference_mode():
        for rid, ids in batches:
            logits = model(ids).logits.float()
            lp = F.log_softmax(logits[:, :-1], dim=-1)
            tgt = ids[:, 1:]
            ce = F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='sum')
            tot_ce += ce.item(); tot_tok += tgt.numel()
            if name == 'B28_source':
                source_logprobs[rid] = lp.clone()
            else:
                s = source_logprobs[rid]
                tot_kl += (s.exp() * (s - lp)).sum().item()
                top1_agree += (s.argmax(-1) == lp.argmax(-1)).sum().item(); top1_n += tgt.numel()
            del logits, lp
    del model; gc.collect()
    ce_mean = tot_ce / tot_tok
    results[name] = {
        'mean_ce_nats': ce_mean,
        'perplexity': float(torch.exp(torch.tensor(ce_mean))),
        'mean_kl_to_source_nats': (tot_kl / tot_tok) if name != 'B28_source' else 0.0,
        'top1_agreement_with_source': (top1_agree / top1_n) if top1_n else 1.0,
        'scored_tokens': tot_tok,
        'wall_seconds': time.perf_counter() - t0,
    }
    r = results[name]
    print(f"{name:12} CE={r['mean_ce_nats']:.4f} ppl={r['perplexity']:.2f} "
          f"KL_to_source={r['mean_kl_to_source_nats']:.4f} top1_agree={r['top1_agreement_with_source']:.4f} "
          f"({r['wall_seconds']:.0f}s)", flush=True)

out = E / 'outputs/arm-quality'
out.mkdir(parents=True, exist_ok=True)
(out / 'ARM_QUALITY.json').write_text(json.dumps({
    'schema': 1,
    'evidence_kind': 'exploratory_quality_diagnostic_only',
    'note': 'Teacher-forced CE and KL-to-source on the already-open calibration split. '
            'Run outside the experiment package; not a receipt and authorizes no stage. '
            'Measures initialization deficit before any recovery training.',
    'implementation_reference': 'fc44701e (unmodified; this script lives outside ROOT)',
    'split': 'calibration', 'rows_scored': len(batches),
    'arms': results,
    'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}, indent=2) + '\n', encoding='utf-8')
print('\nwrote', out / 'ARM_QUALITY.json')
