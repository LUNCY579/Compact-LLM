"""Is the 4x KV result an artifact of one unswept regularisation constant?

`latent_kv.RIDGE = 1e-6` was chosen to make a Cholesky well-posed, not tuned. The measured
input covariance has a condition number of ~1.08e8 and an effective rank of 3.5-5.3 out of
1024, so the whitened fit inverts a badly conditioned matrix and the ridge could plausibly be
load-bearing. If the headline result (rank 512, 0.250x KV, 19/60 against the source's 20/60)
moves materially with the ridge, it is a tuning artifact and must be reported as one.

This sweeps the ridge across six orders of magnitude at fixed rank. It is the single largest
open hole in the only deployable finding in this programme.

Two controls, both cheap and both load-bearing:

  * **Exactness is ridge-independent.** At rank = 1024 the factorisation is
    `A = (W S) S^-1 = W` for *any* positive-definite `S`, so every ridge must reproduce the
    source. A ridge that breaks it means the whitening algebra is wrong, not that the ridge is
    badly chosen.
  * **Plain SVD is ridge-independent** by construction (it never forms a covariance). It is
    re-measured once as a fixed reference point the whitened curve can be read against.

Exploratory. Outside the hashed root; no receipt, no gate. Calibration from TRAIN, scored on
DEV and TASK.

usage: latent_ridge_sweep.py [device] [ranks_csv] [ridges_csv] [dev_rows] [task_rows] [calib_rows]
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

E = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification'
         r'\experiments\architecture_benchmark_001')
SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble'
           r'\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')
SPL = E / 'outputs/splits'
OUT = E / 'outputs/latent'

DEVICE = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
RANKS = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['1024', '512', '293'])]
RIDGES = [float(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3
                             else ['1e-8', '1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '1e-2'])]
NDEV = int(sys.argv[4]) if len(sys.argv) > 4 else 40
NTASK = int(sys.argv[5]) if len(sys.argv) > 5 else 60
NCAL = int(sys.argv[6]) if len(sys.argv) > 6 else 256
PROBE_RANK = 512          # the headline point; generation is the expensive part
EOS = {151643, 151645}

if DEVICE == 'cuda':
    # cuDNN TF32 defaults to ON on Ada and would silently drop matmul mantissas 24 -> 10 bits.
    # Both flags are set explicitly so the configuration is recorded, not inherited.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

tok = _load_tokenizer(SRC)
jl = lambda p: [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]
enc = lambda t: tok(t, return_tensors='pt', add_special_tokens=False)['input_ids']
dev = [x for x in (enc(r['text']) for r in jl(SPL / 'dev.jsonl')[:NDEV]) if x.shape[1] >= 2]
calib = [x for x in (enc(r['text']) for r in jl(SPL / 'train.jsonl')[:NCAL]) if x.shape[1] >= 2]
task_rows = jl(SPL / 'task_probe.jsonl')[:NTASK]

source, _, _ = _load(SRC, DEVICE); source.eval()
cfg = source.config
print(f'device {DEVICE} | torch {torch.__version__} | dev {len(dev)} rows | calib {len(calib)} '
      f'TRAIN rows | task {len(task_rows)}', flush=True)
if DEVICE == 'cuda':
    print(f'  tf32 matmul {torch.backends.cuda.matmul.allow_tf32} | '
          f'tf32 cudnn {torch.backends.cudnn.allow_tf32}', flush=True)


CHUNK = 32   # positions per slice; caps the full-vocab transient at ~10 MiB instead of ~470


def ce_kl(fwd):
    """Teacher-forced CE, plus KL and top-1 against the source, chunked over positions.

    A row's log-softmax is ~78 MiB at this vocabulary, and the KL needs `s`, `lp`, `s.exp()`
    and `(s - lp)` live simultaneously. Materialising all of that for a whole row peaked at
    ~470 MiB and OOMed an 8 GiB card. Slicing bounds the transient without changing the sum.
    """
    tot = kl = 0.0
    agree = n = 0
    with torch.inference_mode():
        for ids in dev:
            ids = ids.to(DEVICE)
            src_logits = source(ids).logits[:, :-1]
            arm_logits = fwd(ids)[:, :-1]
            tgt = ids[:, 1:]
            for i in range(0, tgt.shape[1], CHUNK):
                j = min(i + CHUNK, tgt.shape[1])
                s = F.log_softmax(src_logits[:, i:j].float(), -1)
                lp = F.log_softmax(arm_logits[:, i:j].float(), -1)
                t = tgt[:, i:j]
                tot += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), t.reshape(-1),
                                        reduction='sum'))
                kl += float((s.exp() * (s - lp)).sum())
                agree += int((s.argmax(-1) == lp.argmax(-1)).sum())
                del s, lp
            n += tgt.numel()
            del src_logits, arm_logits
    return {'dev_ce_nats': tot / n, 'dev_kl_to_source_nats': kl / n, 'dev_top1_agreement': agree / n}


def max_logit_err(fwd):
    worst = 0.0
    with torch.inference_mode():
        for ids in dev[:8]:
            ids = ids.to(DEVICE)
            worst = max(worst, float((source(ids).logits - fwd(ids)).abs().max()))
    return worst


def probe(step):
    hits = strict = term = 0
    distinct = set()
    with torch.inference_mode():
        for r in task_rows:
            cur = enc(tok.apply_chat_template(r['messages'], tokenize=False,
                                              enable_thinking=False,
                                              add_generation_prompt=True)).to(DEVICE)
            cache, got = None, []
            for _ in range(24):
                logits, cache = step(cur, cache)
                nxt = int(logits[0, -1].argmax())
                if nxt in EOS:
                    term += 1; break
                got.append(nxt)
                cur = torch.tensor([[nxt]], dtype=torch.long, device=DEVICE)
            text = tok.decode(got, skip_special_tokens=True).strip()
            distinct.add(text)
            hits += answer_matches(text, r['answer']); strict += (text == r['answer'])
    return {'task_numeric_hits': hits, 'task_strict_hits': strict, 'task_n': len(task_rows),
            'task_numeric_rate': hits / len(task_rows), 'task_terminated': term,
            'task_distinct': len(distinct)}


t0 = time.perf_counter()
covs, cal_tokens = collect_input_covariance(source, calib)   # moves rows itself
# `collect_input_covariance` already accumulates on CPU in float64; nothing to move.
if DEVICE == 'cuda':
    torch.cuda.empty_cache()
health = covariance_health(covs, cal_tokens, cfg.hidden_size)
print(f'covariance over {cal_tokens} TRAIN tokens ({time.perf_counter()-t0:.0f}s) | effective '
      f'rank {health["min_effective_rank"]:.1f}-{health["mean_effective_rank"]:.1f} of '
      f'{cfg.hidden_size} | condition {health["worst_condition_number"]:.2e}\n', flush=True)

base = ce_kl(lambda x: source(x).logits)
base_probe = probe(lambda i, c: (lambda o: (o.logits, o.cache))(
    source(i, past=c, use_cache=True, last_position_only=True)))
results = {'source': {**base, **base_probe, 'rank': None, 'ridge': None, 'fit': 'none',
                      'kv_ratio': 1.0}}
print(f"source              CE={base['dev_ce_nats']:8.4f} task={base_probe['task_numeric_hits']}"
      f"/{base_probe['task_n']}\n", flush=True)

print(f"{'rank':>6}{'ridge':>10}{'CE':>10}{'dCE':>10}{'KL':>9}{'top1':>7}{'energy':>9}"
      f"{'task':>9}{'exact err':>12}")
for rank in RANKS:
    for ridge in RIDGES:
        t = time.perf_counter()
        m = LatentKVModel(source, rank).to(DEVICE)
        info = m.fit(covs, ridge=ridge)
        m.eval()
        rec = {**ce_kl(lambda x: m(x)), 'rank': rank, 'ridge': ridge, 'fit': 'whitened',
               'kv_ratio': m.kv_ratio(),
               'spectral_energy_retained': info['mean_spectral_energy_retained'],
               'exact_reparameterisation': info['exact_reparameterisation']}
        if info['exact_reparameterisation']:
            rec['max_abs_logit_error_vs_source'] = max_logit_err(lambda x: m(x))
        if rank == PROBE_RANK:
            rec.update(probe(lambda i, c: m(i, past=c, use_cache=True)))
        results[f'r{rank}_ridge{ridge:g}'] = rec
        tk = (f"{rec['task_numeric_hits']:>3}/{rec['task_n']:<4}" if 'task_numeric_hits' in rec else '')
        ex = (f"{rec['max_abs_logit_error_vs_source']:>12.2e}"
              if 'max_abs_logit_error_vs_source' in rec else '')
        print(f"{rank:>6}{ridge:>10.0e}{rec['dev_ce_nats']:>10.4f}"
              f"{rec['dev_ce_nats']-base['dev_ce_nats']:>+10.4f}{rec['dev_kl_to_source_nats']:>9.4f}"
              f"{rec['dev_top1_agreement']:>7.3f}{rec['spectral_energy_retained']:>9.5f}"
              f"{tk:>9}{ex}   ({time.perf_counter()-t:.0f}s)", flush=True)
        del m; gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

# Verdict: does the headline point survive the sweep?
# A smoke run writes the same filename as a real sweep. Emitting `ridge_sensitive` from one or
# two points would be a one-point "sweep" presented as a sensitivity result -- an earlier smoke
# artifact did exactly that (ridges_tested: 1, ridge_sensitive: false) and had to be deleted.
MIN_RIDGES_FOR_VERDICT = 3
head = [v for k, v in results.items() if v.get('rank') == PROBE_RANK and 'task_numeric_hits' in v]
verdict = None
if head and len(head) < MIN_RIDGES_FOR_VERDICT:
    verdict = {'rank': PROBE_RANK, 'ridges_tested': len(head), 'ridge_sensitive': None,
               'insufficient': f'need >= {MIN_RIDGES_FOR_VERDICT} ridges for a sensitivity '
                               f'verdict; this run is a smoke test, not a sweep'}
    print(f'only {len(head)} ridge(s) at rank {PROBE_RANK} -- NO sensitivity verdict emitted')
elif head:
    ces = [v['dev_ce_nats'] for v in head]
    tasks = [v['task_numeric_hits'] for v in head]
    verdict = {'rank': PROBE_RANK, 'ridges_tested': len(head),
               'dev_ce_min': min(ces), 'dev_ce_max': max(ces), 'dev_ce_spread': max(ces) - min(ces),
               'task_min': min(tasks), 'task_max': max(tasks),
               'source_task': base_probe['task_numeric_hits'],
               'ridge_sensitive': (max(ces) - min(ces)) > 0.05 or (max(tasks) - min(tasks)) > 4}
    print(f"\nrank {PROBE_RANK} across {len(head)} ridges: CE {min(ces):.4f}-{max(ces):.4f} "
          f"(spread {max(ces)-min(ces):.4f}) | task {min(tasks)}-{max(tasks)} of "
          f"{base_probe['task_n']} vs source {base_probe['task_numeric_hits']}")
    print(f"RIDGE-SENSITIVE: {verdict['ridge_sensitive']}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / f'LATENT_RIDGE_SWEEP_{DEVICE}.json').write_text(json.dumps({
    'schema': 1, 'evidence_kind': 'exploratory_regularisation_sensitivity_only',
    'note': 'Sensitivity of the activation-whitened low-rank KV fit to the Cholesky ridge, '
            'which was set to 1e-6 for well-posedness and never tuned. At rank 1024 the '
            'factorisation is exact for ANY positive-definite whitening, so every ridge must '
            'reproduce the source; that is the control. Open splits, calibration from TRAIN, '
            'zero training, no receipt, no gate, no efficacy claim.',
    'device': DEVICE, 'torch': torch.__version__,
    'tf32_matmul': torch.backends.cuda.matmul.allow_tf32 if DEVICE == 'cuda' else None,
    'tf32_cudnn': torch.backends.cudnn.allow_tf32 if DEVICE == 'cuda' else None,
    'device_comparability': 'CUDA FP32 is not bitwise CPU FP32. These numbers are comparable '
                            'to each other and to other runs on the same device; compare them '
                            'against the CPU corpus only through evidence/device_parity.',
    'ranks': RANKS, 'ridges': RIDGES, 'probe_rank': PROBE_RANK,
    'is_smoke_run': len(RIDGES) < MIN_RIDGES_FOR_VERDICT or NTASK < 60 or NCAL < 256,
    'dev_rows': len(dev), 'task_rows': len(task_rows),
    'calibration': {'split': 'train', 'rows': len(calib), **health},
    'variants': results, 'verdict': verdict,
    'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}, indent=2) + '\n', encoding='utf-8')
print('\nwrote', OUT / f'LATENT_RIDGE_SWEEP_{DEVICE}.json')
