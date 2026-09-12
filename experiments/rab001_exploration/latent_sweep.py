"""Quality-versus-KV frontier for per-layer low-rank compression, against the sharing frontier.

Sharing (CED and local cross-layer) prices at ~0.5 nats per removed bank, because removing a
bank displaces a layer *by construction*. Per-layer low-rank compression displaces nothing,
so the law predicts the sharing price should not apply to it. That is a prediction, and this
measures it rather than assuming it.

Design decisions that the result depends on, stated before the numbers:

* **Exactness control.** `W_KV` is (2*n_kv*head_dim, hidden) = (2048, 1024) here, so its rank
  is at most 1024 and r = 1024 must reproduce the source to floating-point noise. If it does
  not, the implementation is wrong and no other number in the sweep may be believed. This
  point is also not a discovery -- at r = hidden the latent is an invertible map of the layer
  input, i.e. the standard "cache h instead of K,V" trick for GQA models where
  2*n_kv*head_dim > hidden. It is reported as a control, not as a finding.
* **Comparison points are budget-matched and pre-declared**, chosen to land on the KV ratios
  the sharing frontier was already measured at (0.500x = local group 2, 0.250x = group 4,
  0.143x = group 7, 0.071x = group 14) -- not chosen after seeing which ranks did well.
* **Calibration for the whitened fit uses TRAIN rows only.** Fitting the compression on DEV
  would contaminate the split it is scored on.

Stage A sweeps every rank on likelihood only (cheap). Stage B runs the generation probe at
the budget-matched points, because likelihood alone is already known in this programme to
decouple from capability in both directions.

Exploratory. Outside the hashed root; no receipt, no gate, no sealed data.

usage: latent_sweep.py [dev_rows] [task_rows] [calib_rows]
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
from experiments.architecture_benchmark_001.run import _load, _load_tokenizer
from experiments.rab001_exploration.latent_kv import (LatentKVModel, collect_input_covariance,
                                                     covariance_health)
from experiments.rab001_exploration.scoring import answer_matches, self_test
self_test()

E = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification\experiments\architecture_benchmark_001')
SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')
SPL = E / 'outputs/splits'
NDEV = int(sys.argv[1]) if len(sys.argv) > 1 else 40
NTASK = int(sys.argv[2]) if len(sys.argv) > 2 else 60
NCAL = int(sys.argv[3]) if len(sys.argv) > 3 else 256
EOS = {151643, 151645}

# Pre-declared. Stage B (generation) runs only at the budget-matched points.
SWEEP_RANKS = [1024, 768, 640, 512, 384, 293, 256, 192, 146, 128, 64]
MATCHED = {1024: 'local group 2 (0.500x)', 512: 'local group 4 (0.250x)',
           293: 'local group 7 (0.143x)', 146: 'local group 14 (0.071x)'}
FITS = ['whitened', 'plain']

tok = _load_tokenizer(SRC)
jl = lambda p: [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]
enc = lambda t: tok(t, return_tensors='pt', add_special_tokens=False)['input_ids']
dev = [x for x in (enc(r['text']) for r in jl(SPL / 'dev.jsonl')[:NDEV]) if x.shape[1] >= 2]
calib = [x for x in (enc(r['text']) for r in jl(SPL / 'train.jsonl')[:NCAL]) if x.shape[1] >= 2]
task_rows = jl(SPL / 'task_probe.jsonl')[:NTASK]

source, _, _ = _load(SRC, 'cpu'); source.eval()
cfg = source.config
kv_per_layer = 2 * cfg.num_key_value_heads * cfg.head_dim
print(f'source: {cfg.encoder_layers}L hidden {cfg.hidden_size}, {cfg.num_key_value_heads} kv heads '
      f'x {cfg.head_dim} = {kv_per_layer} cached values/token/layer', flush=True)
print(f'W_KV is ({kv_per_layer} x {cfg.hidden_size}) so rank <= {min(kv_per_layer, cfg.hidden_size)}; '
      f'r={min(kv_per_layer, cfg.hidden_size)} is exact at '
      f'{kv_per_layer / min(kv_per_layer, cfg.hidden_size):.1f}x reduction (a control, not a result)',
      flush=True)
print(f'dev {len(dev)} rows | calib {len(calib)} TRAIN rows | task {len(task_rows)} prompts\n', flush=True)


def ce_and_kl(fwd):
    """Teacher-forced CE, plus KL and top-1 agreement against the source.

    The source is recomputed row by row rather than cached: caching every row's full
    log-softmax is ~76 MB per row at this vocabulary and OOMed an earlier harness.
    """
    tot = kl = 0.0
    agree = n = 0
    with torch.inference_mode():
        for ids in dev:
            s = F.log_softmax(source(ids).logits.float()[:, :-1], -1)
            lp = F.log_softmax(fwd(ids).float()[:, :-1], -1)
            tgt = ids[:, 1:]
            tot += F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='sum').item()
            kl += (s.exp() * (s - lp)).sum().item()
            agree += int((s.argmax(-1) == lp.argmax(-1)).sum())
            n += tgt.numel()
            del s, lp
    return {'dev_ce_nats': tot / n, 'dev_kl_to_source_nats': kl / n,
            'dev_top1_agreement': agree / n, 'dev_target_tokens': n}


def max_logit_error(fwd):
    worst = 0.0
    with torch.inference_mode():
        for ids in dev[:8]:
            worst = max(worst, float((source(ids).logits - fwd(ids)).abs().max()))
    return worst


def probe(step):
    """Greedy decode, matching `rigorous_pilot.evaluate_task` exactly.

    `step(ids, past)` must return `(logits, cache)`; decoding uses the incremental cache so
    this variant is scored under the same compute as every other arm in the programme.
    """
    hits = strict = term = 0
    distinct = set()
    with torch.inference_mode():
        for r in task_rows:
            cur = enc(tok.apply_chat_template(r['messages'], tokenize=False,
                                              enable_thinking=False, add_generation_prompt=True))
            cache, got = None, []
            for _ in range(24):
                logits, cache = step(cur, cache)
                nxt = int(logits[0, -1].argmax())
                if nxt in EOS:
                    term += 1; break
                got.append(nxt)
                cur = torch.tensor([[nxt]], dtype=torch.long)
            text = tok.decode(got, skip_special_tokens=True).strip()
            distinct.add(text)
            hits += answer_matches(text, r['answer'])
            strict += (text == r['answer'])
    return {'task_numeric_hits': hits, 'task_strict_hits': strict, 'task_n': len(task_rows),
            'task_numeric_rate': hits / len(task_rows), 'task_terminated': term,
            'task_distinct': len(distinct)}


def source_step(ids, cache):
    out = source(ids, past=cache, use_cache=True, last_position_only=True)
    return out.logits, out.cache


t0 = time.perf_counter()
covs, cal_tokens = collect_input_covariance(source, calib)
health = covariance_health(covs, cal_tokens, cfg.hidden_size)
print(f'covariance over {cal_tokens} TRAIN tokens ({time.perf_counter()-t0:.0f}s) | '
      f'{health["tokens_per_dimension"]:.1f} tokens/dim | effective rank '
      f'{health["min_effective_rank"]:.0f}-{health["mean_effective_rank"]:.0f} of {cfg.hidden_size}',
      flush=True)
if health['rank_deficient_by_construction']:
    print('  WARNING: fewer calibration tokens than hidden dimensions -- the whitened fit '
          'would report retained energy it has no evidence for.', flush=True)
print(flush=True)

variants = {}
base = ce_and_kl(lambda x: source(x).logits)
base_probe = probe(source_step)
variants['source_plain'] = {**base, **base_probe, 'rank': None, 'fit': None, 'kv_ratio': 1.0,
                            'displaced_layers': 0, 'family': 'baseline',
                            'max_abs_logit_error_vs_source': 0.0}
print(f"source          CE={base['dev_ce_nats']:8.4f} kv=1.000x "
      f"task={base_probe['task_numeric_hits']}/{base_probe['task_n']}\n", flush=True)

for fit in FITS:
    for rank in SWEEP_RANKS:
        t0 = time.perf_counter()
        m = LatentKVModel(source, rank)
        info = m.fit(covs if fit == 'whitened' else None)
        m.eval()
        rec = {**ce_and_kl(lambda x: m(x)), 'rank': rank, 'fit': fit,
               'kv_ratio': m.kv_ratio(), 'displaced_layers': 0, 'family': 'latent_lowrank',
               'spectral_energy_retained': info['mean_spectral_energy_retained'],
               'min_spectral_energy_retained': info['min_spectral_energy_retained'],
               'exact_reparameterisation': info['exact_reparameterisation'],
               'budget_matched_to': MATCHED.get(rank)}
        if info['exact_reparameterisation']:
            rec['max_abs_logit_error_vs_source'] = max_logit_error(lambda x: m(x))
        if rank in MATCHED:
            rec.update(probe(lambda ids, cache: m(ids, past=cache, use_cache=True)))
        variants[f'latent_{fit}_r{rank}'] = rec
        tk = (f"task={rec['task_numeric_hits']}/{rec['task_n']} term={rec['task_terminated']}"
              if 'task_numeric_hits' in rec else '')
        ex = (f" maxlogiterr={rec['max_abs_logit_error_vs_source']:.2e}"
              if 'max_abs_logit_error_vs_source' in rec else '')
        print(f"{fit:9} r={rank:<5} CE={rec['dev_ce_nats']:8.4f} KL={rec['dev_kl_to_source_nats']:8.4f} "
              f"top1={rec['dev_top1_agreement']:.3f} kv={rec['kv_ratio']:.3f}x "
              f"energy={rec['spectral_energy_retained']:.5f} {tk}{ex} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        del m; gc.collect()

out = E / 'outputs/latent'
out.mkdir(parents=True, exist_ok=True)
(out / 'LATENT_KV_SWEEP.json').write_text(json.dumps({
    'schema': 1, 'evidence_kind': 'exploratory_architecture_comparison_only',
    'note': 'Per-layer low-rank KV compression fitted to the existing weights, ZERO training. '
            'Displaces no layer, so the displaced-layer law predicts the ~0.5 nats/bank '
            'sharing price should not apply. Open splits only; calibration from TRAIN; no '
            'receipt, no gate, no efficacy claim. r=1024 is an exactness control, not a result.',
    'model': {'core': 'Rineya-Supa-1.7.0-devfreeze', 'layers': cfg.encoder_layers,
              'hidden': cfg.hidden_size, 'kv_heads': cfg.num_key_value_heads,
              'head_dim': cfg.head_dim, 'cached_values_per_token_per_layer': kv_per_layer,
              'max_useful_rank': min(kv_per_layer, cfg.hidden_size)},
    'dev_rows': len(dev), 'task_rows': len(task_rows),
    'calibration': {'split': 'train', 'rows': len(calib), **health},
    'budget_matched_points': MATCHED,
    'variants': variants,
    'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}, indent=2) + '\n', encoding='utf-8')
print('\nwrote', out / 'LATENT_KV_SWEEP.json')
