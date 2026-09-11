"""Can these architectures learn at all? A bounded overfitting probe.

If an architecture cannot drive training CE down on a handful of rows it is not a budget
problem, it is a broken design. If it can, the question becomes how much data is needed.
This is the cheapest decisive test of the CED initialisation, and it deliberately does NOT
claim recovery: overfitting 25 rows is not evidence of generalisation.

Exploratory. Outside the hashed experiment root, so it produces no receipt and cannot open
a gate. Uses AdamW, which the registered 25-step engineering recipe does not -- that recipe
is an engineering smoke, not a recovery optimiser, and is left untouched.

usage: trainability.py <arm> <steps> [lr]
  arm: C28 | C40 | gated_branch_only | source_control
"""
import gc, json, sys, time
from dataclasses import replace
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification')
from experiments.architecture_benchmark_001.run import _load, _restore, _load_tokenizer, _text, load_rows

E = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification\experiments\architecture_benchmark_001')
SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')
LAD = E / 'outputs/ladder-fc44701e-v1'
ARM = sys.argv[1] if len(sys.argv) > 1 else 'C28'
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
SEED = 17

torch.manual_seed(SEED)
tok = _load_tokenizer(SRC)
rows = load_rows(E / 'data/train-smoke.jsonl', 'train')
batches = [_text(tok, r, 'cpu')['input_ids'] for r in rows]
batches = [b for b in batches if b.shape[1] >= 2]
target_tokens_per_step = None

if ARM in ('C28', 'C40'):
    model = _restore(LAD / f'G1_CED_{ARM}/attempt-0001', 'cpu')
elif ARM.startswith('ced_last'):
    # Partially converted variant: N trailing layers displaced. Lets trainability be
    # measured against damage level, not only at the registered arm's extreme.
    from experiments.architecture_benchmark_001.transfer import to_ced
    n = int(ARM[len('ced_last'):])
    src, _, _ = _load(SRC, 'cpu')
    enc = src.config.total_layers - n
    model, _ = to_ced(src, replace(src.config, architecture='ced',
                                   encoder_layers=enc, decoder_layers=n), donor_layer=enc)
    del src; gc.collect()
elif ARM == 'source_control':
    model, _, _ = _load(SRC, 'cpu')
elif ARM == 'gated_branch_only':
    # Function-preserving augmentation: freeze every original tensor and train ONLY the
    # added shared-memory branch. Starting CE is exactly the source's, so any reduction is
    # added capacity earning its keep rather than damage being repaired.
    from experiments.rab001_exploration.gated_memory import GatedMemoryReference
    src, _, _ = _load(SRC, 'cpu')
    model = GatedMemoryReference(src.config, 14)
    model.load_from_plain(src)
    del src; gc.collect()
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith('reads.') or name.startswith('memory.'))
else:
    raise SystemExit(f'unknown arm {ARM}')

model.train()
model.gradient_checkpointing = True
trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
print(f'arm={ARM} steps={STEPS} lr={LR} trainable_params={sum(p.numel() for p in trainable):,}', flush=True)

traj, t0, tokens = [], time.perf_counter(), 0
for step in range(STEPS):
    ids = batches[step % len(batches)]
    out = model(ids)
    logits = out if ARM == 'gated_branch_only' else out.logits
    tgt = ids[:, 1:]
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    tokens += tgt.numel()
    traj.append({'step': step, 'ce': float(loss), 'grad_norm': float(gnorm), 'tokens': tokens})
    if step % 10 == 0 or step == STEPS - 1:
        print(f'  step {step:4d}  ce={float(loss):8.4f}  |g|={float(gnorm):9.3f}  '
              f'tok={tokens:7d}  {time.perf_counter()-t0:6.0f}s', flush=True)

wall = time.perf_counter() - t0
first = sum(t['ce'] for t in traj[:5]) / min(5, len(traj))
last = sum(t['ce'] for t in traj[-5:]) / min(5, len(traj))
out = E / 'outputs/trainability'
out.mkdir(parents=True, exist_ok=True)
(out / f'TRAINABILITY_{ARM}.json').write_text(json.dumps({
    'schema': 1, 'evidence_kind': 'exploratory_trainability_probe_only',
    'note': 'Overfitting probe on 25 train-smoke rows. NOT a recovery result and NOT '
            'evidence of generalisation: the eval set is the train set. Outside the '
            'experiment package; no receipt, no gate opened.',
    'arm': ARM, 'steps': STEPS, 'optimizer': f'AdamW lr={LR} wd=0', 'seed': SEED,
    'gradient_checkpointing': True, 'rows': len(batches),
    'trainable_parameters': sum(p.numel() for p in trainable),
    'mean_ce_first5': first, 'mean_ce_last5': last, 'ce_reduction_nats': first - last,
    'supervised_target_tokens': tokens, 'wall_seconds': wall,
    'throughput_targets_per_second': tokens / wall,
    'trajectory': traj,
    'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}, indent=2) + '\n', encoding='utf-8')
print(f'\n{ARM}: CE {first:.4f} -> {last:.4f}  (drop {first-last:.4f} nats over {STEPS} steps, '
      f'{tokens} targets, {wall:.0f}s, {tokens/wall:.1f} tgt/s)')
print('wrote', out / f'TRAINABILITY_{ARM}.json')
