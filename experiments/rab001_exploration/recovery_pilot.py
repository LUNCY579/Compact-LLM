"""Bounded recovery pilot with HELD-OUT evaluation: does a destroyed arm actually recover?

The overfitting probe showed C28 train-CE falling 11.48 -> 3.20 in 20 steps, which is
suggestive but weak: train CE can fall by memorising. This measures what matters instead —
CE and KL against the frozen source on calibration rows the optimiser NEVER sees, plus
greedy generation on the open pilot split, at checkpoints through training.

Train split: data/train-smoke.jsonl (25 rows). Held-out: data/calibration.jsonl rows,
disjoint from train by construction and never used for a gradient step. Not a sealed or
DEV evaluation, and not an efficacy claim.

Exploratory. Outside the hashed experiment root; no receipt, no gate opened. Uses AdamW,
which the registered 25-step engineering recipe deliberately does not — that recipe is an
engineering smoke and is left untouched.

usage: recovery_pilot.py <arm> <steps> [lr] [eval_every] [heldout_rows]
  arm: C28 | C40 | ced_lastN | source_control
"""
import gc, json, re, sys, time
from dataclasses import replace
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification')
from experiments.architecture_benchmark_001.run import (
    _load, _restore, _load_tokenizer, _prompt, _text, ensure_disjoint, load_rows)

E = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification\experiments\architecture_benchmark_001')
SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')
LAD = E / 'outputs/ladder-fc44701e-v1'
ARM = sys.argv[1] if len(sys.argv) > 1 else 'C28'
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 120
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
EVERY = int(sys.argv[4]) if len(sys.argv) > 4 else 20
HELD = int(sys.argv[5]) if len(sys.argv) > 5 else 8
SEED, EOS = 17, {151643, 151645}
TAG = f'{ARM}_lr{LR:g}_s{STEPS}'

torch.manual_seed(SEED)
tok = _load_tokenizer(SRC)
train_rows = load_rows(E / 'data/train-smoke.jsonl', 'train')
held_rows = load_rows(E / 'data/calibration.jsonl', 'calibration')[:HELD]
pilot = load_rows(E / 'data/pilot.jsonl', 'pilot')[:6]
ensure_disjoint(train_rows, held_rows)
ensure_disjoint(pilot, train_rows, held_rows)
train = [_text(tok, r, 'cpu')['input_ids'] for r in train_rows]
train = [b for b in train if b.shape[1] >= 2]
held = [_text(tok, r, 'cpu')['input_ids'] for r in held_rows]
held = [b for b in held if b.shape[1] >= 2]

# Frozen source reference log-probs on the held-out rows, for KL and top-1 agreement.
source, _, _ = _load(SRC, 'cpu'); source.eval()
ref, ce_src, ntok = [], 0.0, 0
with torch.inference_mode():
    for ids in held:
        lp = F.log_softmax(source(ids).logits.float()[:, :-1], -1)
        ref.append(lp.clone())
        tgt = ids[:, 1:]
        ce_src += F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='sum').item()
        ntok += tgt.numel()
ce_src /= ntok
del source; gc.collect()
print(f'arm={ARM} steps={STEPS} lr={LR} | held-out rows={len(held)} tokens={ntok} '
      f'source CE={ce_src:.4f}', flush=True)

if ARM in ('C28', 'C40'):
    model = _restore(LAD / f'G1_CED_{ARM}/attempt-0001', 'cpu')
elif ARM.startswith('ced_last'):
    from experiments.architecture_benchmark_001.transfer import to_ced
    n = int(ARM[len('ced_last'):])
    src, _, _ = _load(SRC, 'cpu')
    enc = src.config.total_layers - n
    target = replace(src.config, architecture='ced', encoder_layers=enc, decoder_layers=n)
    model, _ = to_ced(src, target, donor_layer=enc)
    del src; gc.collect()
elif ARM == 'source_control':
    model, _, _ = _load(SRC, 'cpu')
else:
    raise SystemExit(f'unknown arm {ARM}')

model.gradient_checkpointing = True
trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)


def heldout():
    model.eval()
    ce, kl, agree = 0.0, 0.0, 0
    with torch.inference_mode():
        for lp_ref, ids in zip(ref, held):
            lp = F.log_softmax(model(ids).logits.float()[:, :-1], -1)
            tgt = ids[:, 1:]
            ce += F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='sum').item()
            kl += (lp_ref.exp() * (lp_ref - lp)).sum().item()
            agree += int((lp_ref.argmax(-1) == lp.argmax(-1)).sum())
            del lp
    model.train()
    return {'heldout_ce_nats': ce / ntok, 'heldout_kl_to_source_nats': kl / ntok,
            'heldout_top1_agreement': agree / ntok}


def _leading_int(text):
    """First integer in the decoded answer, or None.

    Strict string equality is the wrong scorer here. After adapting to this corpus the model
    answers in the corpus's own style -- "3." rather than "3" -- so exact match reported 0/6
    for a run whose arithmetic was entirely correct. That understatement inverted a
    conclusion, so the scorer now records BOTH readings and the tolerant one is primary for
    the arithmetic pilot, whose answers are integers by construction.
    """
    m = re.search(r'-?\d+', text)
    return m.group(0) if m else None


def generate():
    model.eval()
    outs, term = [], 0
    with torch.inference_mode():
        for row in pilot:
            cur, cache, got = _prompt(tok, row, 'cpu'), None, []
            for _ in range(24):
                step = model(cur, past=cache, use_cache=True, last_position_only=True)
                cache = step.cache
                t = int(step.logits[0, -1].argmax())
                if t in EOS:
                    term += 1; break
                got.append(t); cur = torch.tensor([[t]], dtype=torch.long)
            text = tok.decode(got, skip_special_tokens=True)[:60]
            want = row['answer'].strip()
            outs.append({'id': row['id'], 'expected': want, 'got': text,
                         'leading_int': _leading_int(text),
                         'strict_match': text.strip() == want,
                         'numeric_match': _leading_int(text) == want})
    model.train()
    return {'exact_hits': sum(o['numeric_match'] for o in outs),
            'strict_hits': sum(o['strict_match'] for o in outs),
            'of': len(pilot), 'terminated': term,
            'distinct': len({o['got'] for o in outs}),
            'scoring': 'exact_hits is leading-integer match; strict_hits is full-string equality',
            'samples': outs}


model.train()
checkpoints, traj, tokens, t0 = [], [], 0, time.perf_counter()
e0 = heldout(); g0 = generate()
checkpoints.append({'step': 0, 'tokens': 0, **e0, **{k: v for k, v in g0.items() if k != 'samples'}})
print(f"  step   0 | held CE={e0['heldout_ce_nats']:8.4f} KL={e0['heldout_kl_to_source_nats']:7.4f} "
      f"top1={e0['heldout_top1_agreement']:.3f} | hits={g0['exact_hits']}/{g0['of']} "
      f"term={g0['terminated']} distinct={g0['distinct']}", flush=True)

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
        e = heldout(); g = generate()
        checkpoints.append({'step': step, 'tokens': tokens, **e,
                            **{k: v for k, v in g.items() if k != 'samples'}})
        print(f"  step {step:3d} | held CE={e['heldout_ce_nats']:8.4f} KL={e['heldout_kl_to_source_nats']:7.4f} "
              f"top1={e['heldout_top1_agreement']:.3f} | hits={g['exact_hits']}/{g['of']} "
              f"term={g['terminated']} distinct={g['distinct']} | train_ce={float(loss.detach()):7.4f} "
              f"tok={tokens} {time.perf_counter()-t0:.0f}s", flush=True)
        for s in g['samples'][:2]:
            print(f"        {s['id']:14} expected={s['expected']!r:6} got={s['got']!r}", flush=True)

final_gen = generate()
out = E / 'outputs/recovery-pilot'
out.mkdir(parents=True, exist_ok=True)
(out / f'RECOVERY_PILOT_{TAG}.json').write_text(json.dumps({
    'schema': 1, 'evidence_kind': 'exploratory_bounded_recovery_pilot_only',
    'note': 'HELD-OUT calibration rows never used for a gradient step, plus open pilot '
            'generation. Not a sealed or DEV evaluation and not an efficacy claim. Outside '
            'the experiment package; no receipt, no gate opened.',
    'arm': ARM, 'steps': STEPS, 'optimizer': f'AdamW lr={LR} wd=0 clip=1.0', 'seed': SEED,
    'train_rows': len(train), 'heldout_rows': len(held), 'heldout_target_tokens': ntok,
    'source_heldout_ce_nats': ce_src,
    'trainable_parameters': sum(p.numel() for p in trainable),
    'supervised_target_tokens': tokens, 'wall_seconds': time.perf_counter() - t0,
    'checkpoints': checkpoints, 'trajectory': traj, 'final_generation': final_gen,
    'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}, indent=2) + '\n', encoding='utf-8')
print(f"\nsource held-out CE {ce_src:.4f} | arm held-out CE "
      f"{checkpoints[0]['heldout_ce_nats']:.4f} -> {checkpoints[-1]['heldout_ce_nats']:.4f}")
print('wrote', out / f'RECOVERY_PILOT_{TAG}.json')
