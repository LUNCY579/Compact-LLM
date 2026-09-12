"""Head-to-head: CED global sharing vs local cross-layer KV sharing, at equal KV budget.

Forward-only conversion of the official weights, so every number is an initialisation
quality, not a trained result. Exploratory; outside the hashed experiment root.
"""
import gc, json, sys, time
from dataclasses import replace
from pathlib import Path
import torch
import torch.nn.functional as F

# This host's console is cp1252. A model emitting non-ASCII token salad would otherwise
# kill a run inside a print AFTER the expensive compute was done -- which is exactly how
# three C28 runs were lost. Output formatting must never destroy a measurement.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification')
from experiments.architecture_benchmark_001.run import _load, _load_tokenizer, _text, _prompt, load_rows
from experiments.architecture_benchmark_001.budget import budget
from experiments.architecture_benchmark_001.transfer import to_ced
from experiments.rab001_exploration.local_shared_kv import LocalSharedKVReference

E = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification\experiments\architecture_benchmark_001')
SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')
NROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 24
GROUPS = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['1', '2', '4', '7', '14'])]
EOS = {151643, 151645}

source, tok, _ = _load(SRC, 'cpu'); source.eval()
cfg = source.config
rows = load_rows(E / 'data/calibration.jsonl', 'calibration')[:NROWS]
batches = [_text(tok, r, 'cpu')['input_ids'] for r in rows]
batches = [b for b in batches if b.shape[1] >= 2]
pilot = load_rows(E / 'data/pilot.jsonl', 'pilot')[:8]
base_kv = budget(cfg)['kv_bytes_per_token_batch1']
print(f'source {cfg.encoder_layers} layers, kv/token {base_kv}, {len(batches)} rows', flush=True)


def evaluate(fwd, ref=None):
    tot_ce, tot_kl, agree, n = 0.0, 0.0, 0, 0
    store = {}
    with torch.inference_mode():
        for idx, ids in enumerate(batches):
            lp = F.log_softmax(fwd(ids).float()[:, :-1], -1)
            tgt = ids[:, 1:]
            tot_ce += F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='sum').item()
            n += tgt.numel()
            if ref is None:
                store[idx] = lp.clone()
            else:
                s = ref[idx]
                tot_kl += (s.exp() * (s - lp)).sum().item()
                agree += (s.argmax(-1) == lp.argmax(-1)).sum().item()
            del lp
    ce = tot_ce / n
    return {'mean_ce_nats': ce, 'perplexity': float(torch.exp(torch.tensor(ce))),
            'mean_kl_to_source_nats': (tot_kl / n) if ref else 0.0,
            'top1_agreement_with_source': (agree / n) if ref else 1.0}, store


def gen(fwd, limit=8):
    hits, term = 0, 0
    with torch.inference_mode():
        for row in pilot[:limit]:
            ids, out = _prompt(tok, row, 'cpu'), []
            for _ in range(24):
                nxt = int(fwd(ids)[0, -1].argmax())
                if nxt in EOS:
                    term += 1; break
                out.append(nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
            if tok.decode(out, skip_special_tokens=True).strip() == row['answer'].strip():
                hits += 1
    return {'exact_hits': hits, 'of': limit, 'terminated': term}


results = {}
base, ref = evaluate(lambda x: source(x).logits)
bg = gen(lambda x: source(x).logits)
results['source_plain'] = {**base, **bg, 'kv_bytes_per_token': base_kv, 'kv_ratio': 1.0,
                           'family': 'baseline'}
print(f"source          CE={base['mean_ce_nats']:8.4f} kv=1.000x hits={bg['exact_hits']}/{bg['of']}", flush=True)

for g in GROUPS:
    t0 = time.perf_counter()
    m = LocalSharedKVReference(cfg, g)
    info = m.load_from_plain(source)
    m.eval()
    met, _ = evaluate(lambda x: m(x), ref)
    gg = gen(lambda x: m(x))
    ratio = m.kv_ratio()
    kv = int(round(base_kv * ratio))
    results[f'local_kv_group{g}'] = {**met, **gg, 'kv_bytes_per_token': kv,
                                     'kv_ratio': ratio, 'family': 'local_cross_layer',
                                     'group': g, 'kv_producing_layers': m.kv_producing_layers(),
                                     'info': info}
    print(f"local group {g:<2}  CE={met['mean_ce_nats']:8.4f} ppl={met['perplexity']:10.1f} "
          f"KL={met['mean_kl_to_source_nats']:7.4f} top1={met['top1_agreement_with_source']:.3f} "
          f"kv={ratio:.3f}x hits={gg['exact_hits']}/{gg['of']} term={gg['terminated']} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
    del m; gc.collect()

# CED variants at comparable KV ratios, for the head-to-head.
for n in [int(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['4', '8', '14'])]:
    enc = cfg.total_layers - n
    target = replace(cfg, architecture='ced', encoder_layers=enc, decoder_layers=n)
    t0 = time.perf_counter()
    m, _ = to_ced(source, target, donor_layer=enc); m.eval()
    met, _ = evaluate(lambda x: m(x).logits, ref)
    gg = gen(lambda x: m(x).logits)
    kv = budget(target)['kv_bytes_per_token_batch1']
    results[f'ced_last{n}'] = {**met, **gg, 'kv_bytes_per_token': kv, 'kv_ratio': kv / base_kv,
                               'family': 'ced_global', 'converted_layers': n}
    print(f"ced last {n:<2}    CE={met['mean_ce_nats']:8.4f} ppl={met['perplexity']:10.1f} "
          f"KL={met['mean_kl_to_source_nats']:7.4f} top1={met['top1_agreement_with_source']:.3f} "
          f"kv={kv/base_kv:.3f}x hits={gg['exact_hits']}/{gg['of']} term={gg['terminated']} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
    del m; gc.collect()

out = E / 'outputs/arm-quality'
out.mkdir(parents=True, exist_ok=True)
(out / 'ARCHITECTURE_COMPARISON.json').write_text(json.dumps({
    'schema': 1, 'evidence_kind': 'exploratory_architecture_comparison_only',
    'note': 'Zero-training conversions of the official weights. Initialization quality at a '
            'given KV budget, NOT a trained result and NOT an efficacy claim. Outside the '
            'experiment package; no receipt, no gate opened.',
    'split': 'calibration', 'rows_scored': len(batches),
    'uniform_baseline_ce_nats': float(torch.log(torch.tensor(float(cfg.vocab_size)))),
    'variants': results,
    'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}, indent=2) + '\n', encoding='utf-8')
print('\nwrote', out / 'ARCHITECTURE_COMPARISON.json')
