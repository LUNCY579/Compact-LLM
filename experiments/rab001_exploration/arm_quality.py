"""Initialization quality of every arm against the frozen official source.

Teacher-forced next-token CE, perplexity, KL-to-source and top-1 agreement on the
already-open calibration split. Exploratory diagnostic run OUTSIDE the hashed experiment
root so the implementation digest and every existing receipt stay valid. Not a receipt.

Memory note: an earlier version cached the source log-probabilities for every row, which is
rows x tokens x 151936 x 4 bytes -- about 12 GB at 156 rows, and it exhausted this machine.
The source is now held resident and paired with one arm at a time, streaming row by row, so
peak memory is two models plus a single row regardless of row count.
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
ARMS = [('D40_width', LAD / 'G1_WIDTH/attempt-0001'),
        ('C28_ced', LAD / 'G1_CED_C28/attempt-0001'),
        ('C40_ced', LAD / 'G1_CED_C40/attempt-0001')]

tok = _load_tokenizer(SRC)
rows = load_rows(E / 'data/calibration.jsonl', 'calibration')[:ROWS]
batches = [_text(tok, r, 'cpu')['input_ids'] for r in rows]
batches = [b for b in batches if b.shape[1] >= 2]
print(f'rows scored: {len(batches)}  target tokens: {sum(b.shape[1]-1 for b in batches)}', flush=True)

source, _, _ = _load(SRC, 'cpu')
source.eval()


def ce_of(model):
    tot, n = 0.0, 0
    with torch.inference_mode():
        for ids in batches:
            lp = F.log_softmax(model(ids).logits.float()[:, :-1], -1)
            tgt = ids[:, 1:]
            tot += F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='sum').item()
            n += tgt.numel()
            del lp
    return tot / n, n


t0 = time.perf_counter()
ce_src, ntok = ce_of(source)
results = {'B28_source': {'mean_ce_nats': ce_src, 'perplexity': float(torch.exp(torch.tensor(ce_src))),
                          'mean_kl_to_source_nats': 0.0, 'top1_agreement_with_source': 1.0,
                          'scored_target_tokens': ntok, 'wall_seconds': time.perf_counter() - t0}}
r = results['B28_source']
print(f"B28_source   CE={r['mean_ce_nats']:.4f} ppl={r['perplexity']:.2f} "
      f"KL=0.0000 top1=1.0000 ({r['wall_seconds']:.0f}s)", flush=True)

for name, ckpt in ARMS:
    t0 = time.perf_counter()
    arm = _restore(ckpt, 'cpu').eval()
    tot_ce, tot_kl, agree, n = 0.0, 0.0, 0, 0
    with torch.inference_mode():
        for ids in batches:
            s = F.log_softmax(source(ids).logits.float()[:, :-1], -1)
            a = F.log_softmax(arm(ids).logits.float()[:, :-1], -1)
            tgt = ids[:, 1:]
            tot_ce += F.nll_loss(a.reshape(-1, a.shape[-1]), tgt.reshape(-1), reduction='sum').item()
            tot_kl += (s.exp() * (s - a)).sum().item()
            agree += (s.argmax(-1) == a.argmax(-1)).sum().item()
            n += tgt.numel()
            del s, a
    del arm; gc.collect()
    ce = tot_ce / n
    results[name] = {'mean_ce_nats': ce, 'perplexity': float(torch.exp(torch.tensor(ce))),
                     'mean_kl_to_source_nats': tot_kl / n, 'top1_agreement_with_source': agree / n,
                     'scored_target_tokens': n, 'wall_seconds': time.perf_counter() - t0}
    r = results[name]
    print(f"{name:12} CE={r['mean_ce_nats']:.4f} ppl={r['perplexity']:.2f} "
          f"KL={r['mean_kl_to_source_nats']:.4f} top1={r['top1_agreement_with_source']:.4f} "
          f"({r['wall_seconds']:.0f}s)", flush=True)

out = E / 'outputs/arm-quality'
out.mkdir(parents=True, exist_ok=True)
(out / f'ARM_QUALITY_{len(batches)}rows.json').write_text(json.dumps({
    'schema': 1, 'evidence_kind': 'exploratory_quality_diagnostic_only',
    'note': 'Teacher-forced CE and KL-to-source on the already-open calibration split. '
            'Outside the experiment package; not a receipt and authorizes no stage. '
            'Measures the initialization deficit before any recovery training.',
    'implementation_reference': 'fc44701e (unmodified; this script lives outside ROOT)',
    'split': 'calibration', 'rows_scored': len(batches),
    'uniform_baseline_ce_nats': float(torch.log(torch.tensor(151936.0))),
    'arms': results,
    'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}, indent=2) + '\n', encoding='utf-8')
print('\nwrote', out / f'ARM_QUALITY_{len(batches)}rows.json')
