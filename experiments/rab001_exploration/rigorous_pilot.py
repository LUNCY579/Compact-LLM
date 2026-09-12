"""Multi-seed matched comparison on family-disjoint splits with a real task eval.

Replaces the small probe (25 train rows, 8 dev rows, 6 task prompts, one seed) that earlier
conclusions rested on. Here:

  TRAIN  800 family-disjoint non-arithmetic rows; 40 steps sees 40 DISTINCT rows, no cycling
  DEV     50 family-disjoint non-arithmetic rows, never used for a gradient step
  TASK   100 arithmetic_chain prompts, held out of TRAIN/DEV entirely, verifiable integers

arithmetic_chain is excluded from training, so TASK measures RETENTION of a capability the
frozen source already has rather than in-domain learning.

Seeds are the protocol's registered training seeds (17, 29, 43). Each (arm, seed) writes its
own JSON immediately so a killed session loses at most one run.

Scoring reports `numeric` (normalised string, or exact match of the full number sequence) and
`strict` (raw full-string equality) side by side. Strict collapses once a model adopts the
corpus's "33." style, which previously inverted a conclusion, so `numeric` is primary — but it
compares EVERY number, because leading-integer matching scored "10:55 AM" as correct against
"10:25 AM" for the 31 of 50 time_duration answers that are clock times.

Exploratory. Outside the hashed root: no receipt, no gate. Open splits only.

usage: rigorous_pilot.py <arm> <seed> [steps] [lr] [n_dev] [n_task] [eval_every]
"""
import gc, json, re, sys, time
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
from experiments.architecture_benchmark_001.run import _load, _restore, _load_tokenizer

E = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification\experiments\architecture_benchmark_001')
SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')
LAD = E / 'outputs/ladder-fc44701e-v1'
SPL = E / 'outputs/splits'

ARM = sys.argv[1] if len(sys.argv) > 1 else 'source_control'
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 17
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
N_DEV = int(sys.argv[5]) if len(sys.argv) > 5 else 50
N_TASK = int(sys.argv[6]) if len(sys.argv) > 6 else 100
EVERY = int(sys.argv[7]) if len(sys.argv) > 7 else 20
EOS = {151643, 151645}
TAG = f'{ARM}_seed{SEED}_lr{LR:g}_s{STEPS}'

torch.manual_seed(SEED)
tok = _load_tokenizer(SRC)
# BUG FIX. Seeds 17 and 29 previously produced BIT-IDENTICAL results: the data order was
# fixed, the weights are loaded from fixed checkpoints, there is no dropout and decoding is
# greedy, so nothing in the run consumed the seed. Reporting three such runs as "replicated
# across three seeds" would have been one run reported three times. The seed now controls the
# training order, so each seed sees a different subset of TRAIN and the spread across seeds is
# a real measure of run-to-run variance.
import random as _random
jl = lambda p: [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]
train_rows = jl(SPL / 'train.jsonl')
dev_rows = jl(SPL / 'dev.jsonl')[:N_DEV]
task_rows = jl(SPL / 'task_probe.jsonl')[:N_TASK]

# Family disjointness is enforced at build time; re-assert it here so a corrupted split file
# cannot silently contaminate a run.
fam = lambda rows: {r['group_id'] for r in rows}
assert not (fam(train_rows) & fam(dev_rows)), 'train/dev family overlap'
assert not (fam(train_rows) & fam(task_rows)), 'train/task family overlap'
assert not (fam(dev_rows) & fam(task_rows)), 'dev/task family overlap'
assert 'arithmetic_chain' not in {r['reply_class'] for r in train_rows}

_random.Random(SEED).shuffle(train_rows)          # the seed's only job, and it must have one
enc = lambda text: tok(text, return_tensors='pt', add_special_tokens=False)['input_ids']
train = [enc(r['text']) for r in train_rows]
train = [t for t in train if t.shape[1] >= 2]
dev = [enc(r['text']) for r in dev_rows]
dev = [t for t in dev if t.shape[1] >= 2]

# Reference log-probs from the frozen source on DEV, for KL and top-1.
source, _, _ = _load(SRC, 'cpu'); source.eval()
ref, ce_src, ntok = [], 0.0, 0
with torch.inference_mode():
    for ids in dev:
        lp = F.log_softmax(source(ids).logits.float()[:, :-1], -1)
        ref.append(lp.clone())
        tgt = ids[:, 1:]
        ce_src += F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='sum').item()
        ntok += tgt.numel()
ce_src /= ntok
del source; gc.collect()
print(f'{TAG} | train {len(train)} dev {len(dev)} task {len(task_rows)} | dev tokens {ntok} '
      f'| source dev CE {ce_src:.4f}', flush=True)

if ARM in ('C28', 'C40'):
    model = _restore(LAD / f'G1_CED_{ARM}/attempt-0001', 'cpu')
elif ARM == 'D40':
    model = _restore(LAD / 'G1_WIDTH/attempt-0001', 'cpu')
elif ARM.startswith('ced_last'):
    # Partially converted arm: N trailing layers displaced. Lets the displaced-layer law be
    # tested against TRAINING -- does the ~0.5 nats/bank exchange rate move when the weights
    # are allowed to adapt, or is it hard?
    from dataclasses import replace as _replace
    from experiments.architecture_benchmark_001.transfer import to_ced
    n = int(ARM[len('ced_last'):])
    src, _, _ = _load(SRC, 'cpu')
    enc = src.config.total_layers - n
    model, _ = to_ced(src, _replace(src.config, architecture='ced',
                                    encoder_layers=enc, decoder_layers=n), donor_layer=enc)
    del src; gc.collect()
elif ARM == 'source_control':
    model, _, _ = _load(SRC, 'cpu')
else:
    raise SystemExit(f'unknown arm {ARM}')

model.gradient_checkpointing = True
trainable = [p for p in model.parameters() if p.requires_grad]
frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
assert not frozen, f'{len(frozen)} parameters are frozen; the comparison would not be matched'
opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)

from experiments.rab001_exploration.scoring import answer_matches, self_test
self_test()   # a drifted scorer already inverted one conclusion; fail loudly instead


def evaluate_dev():
    model.eval()
    ce = kl = 0.0; agree = 0
    with torch.inference_mode():
        for lp_ref, ids in zip(ref, dev):
            lp = F.log_softmax(model(ids).logits.float()[:, :-1], -1)
            tgt = ids[:, 1:]
            ce += F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='sum').item()
            kl += (lp_ref.exp() * (lp_ref - lp)).sum().item()
            agree += int((lp_ref.argmax(-1) == lp.argmax(-1)).sum())
            del lp
    model.train()
    return {'dev_ce_nats': ce / ntok, 'dev_kl_to_source_nats': kl / ntok,
            'dev_top1_agreement': agree / ntok}


def evaluate_task():
    """Greedy decode on held-out arithmetic. Retention, not in-domain learning."""
    model.eval()
    numeric = strict = term = 0
    outs, distinct = [], set()
    with torch.inference_mode():
        for r in task_rows:
            prompt = tok.apply_chat_template(r['messages'], tokenize=False,
                                             enable_thinking=False, add_generation_prompt=True)
            cur = tok(prompt, return_tensors='pt', add_special_tokens=False)['input_ids']
            cache, got = None, []
            for _ in range(24):
                step = model(cur, past=cache, use_cache=True, last_position_only=True)
                cache = step.cache
                t = int(step.logits[0, -1].argmax())
                if t in EOS:
                    term += 1; break
                got.append(t); cur = torch.tensor([[t]], dtype=torch.long)
            text = tok.decode(got, skip_special_tokens=True).strip()
            distinct.add(text)
            want = r['answer']
            n_ok = answer_matches(text, want)
            numeric += n_ok; strict += (text == want)
            if len(outs) < 8:
                outs.append({'id': r['id'], 'expected': want, 'got': text[:50], 'numeric_match': bool(n_ok)})
    model.train()
    n = len(task_rows)
    return {'task_numeric_hits': numeric, 'task_strict_hits': strict, 'task_n': n,
            'task_numeric_rate': numeric / n, 'task_terminated': term,
            'task_distinct': len(distinct), 'samples': outs}


model.train()
checkpoints, traj, tokens, t0 = [], [], 0, time.perf_counter()


def checkpoint(step):
    d, t = evaluate_dev(), evaluate_task()
    rec = {'step': step, 'tokens': tokens, **d, **{k: v for k, v in t.items() if k != 'samples'}}
    checkpoints.append(rec)
    print(f"  step {step:3d} | devCE {d['dev_ce_nats']:7.4f} KL {d['dev_kl_to_source_nats']:6.3f} "
          f"top1 {d['dev_top1_agreement']:.3f} | task {t['task_numeric_hits']:3d}/{t['task_n']} "
          f"({t['task_numeric_rate']:.2%}) strict {t['task_strict_hits']:3d} term {t['task_terminated']:3d} "
          f"| {time.perf_counter()-t0:.0f}s", flush=True)
    for s in t['samples'][:2]:
        print(f"        expected={s['expected']!r:8} got={s['got']!r:30} ok={s['numeric_match']}", flush=True)
    return t['samples']


last_samples = checkpoint(0)
for step in range(1, STEPS + 1):
    ids = train[(step - 1) % len(train)]
    logits = model(ids).logits
    tgt = ids[:, 1:]
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    tokens += tgt.numel()
    traj.append({'step': step, 'train_ce': float(loss.detach()), 'grad_norm': float(gnorm), 'tokens': tokens})
    if step % EVERY == 0 or step == STEPS:
        last_samples = checkpoint(step)

out = E / 'outputs/rigorous'
out.mkdir(parents=True, exist_ok=True)
(out / f'RIGOROUS_{TAG}.json').write_text(json.dumps({
    'schema': 1, 'evidence_kind': 'exploratory_multiseed_matched_comparison_only',
    'note': 'Family-disjoint splits; arithmetic_chain held out of TRAIN/DEV so the task set '
            'measures retention. Open splits only; not a sealed or DEV evaluation and not an '
            'efficacy claim. Outside the experiment package; no receipt, no gate.',
    'arm': ARM, 'seed': SEED, 'steps': STEPS, 'optimizer': f'AdamW lr={LR:g} wd=0 clip=1.0',
    'seed_controls': 'training row order (shuffle of TRAIN); nothing else in the run is '
                     'stochastic, so this is the only source of between-seed variance',
    'task_scoring': 'normalised-string match OR exact equality of the full number sequence '
                    '(decimals and clock times included); leading-integer matching was too '
                    'lenient and is not used',
    'train_rows_available': len(train), 'train_rows_seen': min(STEPS, len(train)),
    'dev_rows': len(dev), 'dev_target_tokens': ntok, 'task_rows': len(task_rows),
    'source_dev_ce_nats': ce_src,
    'trainable_parameters': sum(p.numel() for p in trainable),
    'supervised_target_tokens': tokens, 'wall_seconds': time.perf_counter() - t0,
    'checkpoints': checkpoints, 'trajectory': traj, 'final_samples': last_samples,
    'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}, indent=2) + '\n', encoding='utf-8')
print(f"\n{TAG}: dev CE {checkpoints[0]['dev_ce_nats']:.4f} -> {checkpoints[-1]['dev_ce_nats']:.4f} | "
      f"task {checkpoints[0]['task_numeric_hits']}/{checkpoints[0]['task_n']} -> "
      f"{checkpoints[-1]['task_numeric_hits']}/{checkpoints[-1]['task_n']}")
print('wrote', out / f'RIGOROUS_{TAG}.json')
