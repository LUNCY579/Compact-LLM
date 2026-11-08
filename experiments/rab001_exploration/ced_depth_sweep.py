"""How much CED conversion can the model absorb without training?

Converts only the last N layers of the official 28-layer source into shared-memory
ReadBlocks and measures the quality/efficiency trade-off. N=28 minus encoder depth.
Forward-only, no training. Outside the hashed experiment root; not a receipt.
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

E = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification\experiments\architecture_benchmark_001')
SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')
NROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 24
CONVERTED = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['1', '2', '4', '7', '14'])]
EOS = {151643, 151645}

source, tok, _ = _load(SRC, 'cpu')
source.eval()
base_cfg = source.config
rows = load_rows(E / 'data/calibration.jsonl', 'calibration')[:NROWS]
batches = [(r['id'], _text(tok, r, 'cpu')['input_ids']) for r in rows]
batches = [(i, b) for i, b in batches if b.shape[1] >= 2]
pilot = load_rows(E / 'data/pilot.jsonl', 'pilot')
base_kv = budget(base_cfg)['kv_bytes_per_token_batch1']
print(f'source: {base_cfg.total_layers} layers, kv/token {base_kv}, scoring {len(batches)} rows', flush=True)


def score(model, ref=None):
    tot_ce, tot_tok, tot_kl, agree, n = 0.0, 0, 0.0, 0, 0
    store = {}
    with torch.inference_mode():
        for rid, ids in batches:
            lp = F.log_softmax(model(ids).logits.float()[:, :-1], dim=-1)
            tgt = ids[:, 1:]
            tot_ce += F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='sum').item()
            tot_tok += tgt.numel()
            if ref is None:
                store[rid] = lp.clone()
            else:
                s = ref[rid]
                tot_kl += (s.exp() * (s - lp)).sum().item()
                agree += (s.argmax(-1) == lp.argmax(-1)).sum().item(); n += tgt.numel()
    ce = tot_ce / tot_tok
    return {'mean_ce_nats': ce, 'perplexity': float(torch.exp(torch.tensor(ce))),
            'mean_kl_to_source_nats': (tot_kl / tot_tok) if ref else 0.0,
            'top1_agreement_with_source': (agree / n) if n else 1.0}, store


def generate(model, limit=8):
    hits, term = 0, 0
    with torch.inference_mode():
        for row in pilot[:limit]:
            cur, cache, gen = _prompt(tok, row, 'cpu'), None, []
            for _ in range(24):
                o = model(cur, past=cache, use_cache=True, last_position_only=True)
                cache = o.cache
                nxt = int(o.logits[0, -1].argmax())
                if nxt in EOS:
                    term += 1; break
                gen.append(nxt); cur = torch.tensor([[nxt]], dtype=torch.long)
            if tok.decode(gen, skip_special_tokens=True).strip() == row['answer'].strip():
                hits += 1
    return {'exact_hits': hits, 'of': limit, 'terminated': term}


base_metrics, ref = score(source)
base_gen = generate(source)
results = {'B28_source_plain': {**base_metrics, 'converted_layers': 0,
                                'kv_bytes_per_token': base_kv, 'kv_ratio': 1.0, **base_gen}}
print(f"plain source      CE={base_metrics['mean_ce_nats']:.4f} ppl={base_metrics['perplexity']:.1f} "
      f"kv=1.000x hits={base_gen['exact_hits']}/{base_gen['of']}", flush=True)

for n in CONVERTED:
    enc = base_cfg.total_layers - n
    target = replace(base_cfg, architecture='ced', encoder_layers=enc, decoder_layers=n)
    t0 = time.perf_counter()
    model, _ = to_ced(source, target, donor_layer=enc)
    model.eval()
    m, _ = score(model, ref)
    g = generate(model)
    kv = budget(target)['kv_bytes_per_token_batch1']
    results[f'ced_last{n}'] = {**m, 'converted_layers': n, 'encoder_layers': enc,
                               'kv_bytes_per_token': kv, 'kv_ratio': kv / base_kv, **g}
    print(f"ced last {n:<2} enc={enc:<2} CE={m['mean_ce_nats']:8.4f} ppl={m['perplexity']:12.1f} "
          f"KL={m['mean_kl_to_source_nats']:7.4f} top1={m['top1_agreement_with_source']:.3f} "
          f"kv={kv/base_kv:.3f}x hits={g['exact_hits']}/{g['of']} term={g['terminated']} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
    del model; gc.collect()

out = E / 'outputs/arm-quality'
out.mkdir(parents=True, exist_ok=True)
(out / 'CED_DEPTH_SWEEP.json').write_text(json.dumps({
    'schema': 1, 'evidence_kind': 'exploratory_quality_diagnostic_only',
    'note': 'Converts only the last N of 28 layers to shared-memory ReadBlocks, no training. '
            'Quality/efficiency trade-off at initialization. Outside the experiment package; '
            'not a receipt and authorizes nothing.',
    'split': 'calibration', 'rows_scored': len(batches),
    'uniform_baseline_ce_nats': float(torch.log(torch.tensor(float(base_cfg.vocab_size)))),
    'variants': results,
    'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}, indent=2) + '\n', encoding='utf-8')
print('\nwrote', out / 'CED_DEPTH_SWEEP.json')
