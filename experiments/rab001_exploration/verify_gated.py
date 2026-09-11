"""Verify the gated variant is function-preserving at initialisation, then confirm the
branch is live (nonzero gradients, and a nonzero effect once o_proj is perturbed).

Exploratory, outside the hashed experiment root. Not a receipt.
"""
import json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification')
from experiments.architecture_benchmark_001.run import _load, _load_tokenizer, _text, load_rows
from experiments.rab001_exploration.gated_memory import GatedMemoryReference

E = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification\experiments\architecture_benchmark_001')
SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')
SPLIT = int(sys.argv[1]) if len(sys.argv) > 1 else 14
NROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 8

source, tok, _ = _load(SRC, 'cpu')
source.eval()
rows = load_rows(E / 'data/calibration.jsonl', 'calibration')[:NROWS]
batches = [_text(tok, r, 'cpu')['input_ids'] for r in rows]

gated = GatedMemoryReference(source.config, SPLIT)
info = gated.load_from_plain(source)
gated.eval()
print('load_from_plain:', json.dumps(info), flush=True)

exact, max_delta, ce_src, ce_gat, ntok = True, 0.0, 0.0, 0.0, 0
with torch.inference_mode():
    for ids in batches:
        a = source(ids).logits
        b = gated(ids)
        exact &= torch.equal(a, b)
        max_delta = max(max_delta, (a - b).abs().max().item())
        tgt = ids[:, 1:]
        for tag, logits in (('src', a), ('gat', b)):
            lp = F.log_softmax(logits.float()[:, :-1], -1)
            v = F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='sum').item()
            if tag == 'src':
                ce_src += v
            else:
                ce_gat += v
        ntok += tgt.numel()

print(f'\nsplit={SPLIT}  rows={len(batches)}  tokens={ntok}')
print(f'bitwise identical logits : {exact}')
print(f'max |delta| logits       : {max_delta:.3e}')
print(f'CE source / CE gated     : {ce_src/ntok:.6f} / {ce_gat/ntok:.6f}')
print(f'added parameters         : {info["added_parameters"]:,} '
      f'({100*info["added_parameters"]/sum(p.numel() for p in source.parameters()):.2f}% of source)')

# The branch must be live: gradients must reach o_proj, and perturbing it must change output.
gated.train()
ids = batches[0]
logits = gated(ids)
loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
loss.backward()
grads = {n: float(p.grad.abs().max()) for n, p in gated.named_parameters()
         if p.grad is not None and ('reads.0.' in n or n.startswith('memory.'))}
nonzero = {n: g for n, g in grads.items() if g > 0}
print(f'\nbranch params receiving nonzero grad: {len(nonzero)}/{len(grads)}')
for n in sorted(nonzero)[:6]:
    print(f'   {n}  max|grad|={nonzero[n]:.3e}')

gated.eval()
with torch.inference_mode():
    before = gated(ids).clone()
    gated.reads[0].o_proj.weight.add_(torch.randn_like(gated.reads[0].o_proj.weight) * 1e-3)
    after = gated(ids)
    live = not torch.equal(before, after)
print(f'perturbing reads[0].o_proj changes output: {live}  (max delta {(before-after).abs().max().item():.3e})')

out = E / 'outputs/arm-quality'
out.mkdir(parents=True, exist_ok=True)
(out / f'GATED_VERIFY_split{SPLIT}.json').write_text(json.dumps({
    'schema': 1, 'evidence_kind': 'exploratory_architecture_verification_only',
    'variant': 'zero_gated_shared_memory_augmentation', 'split': SPLIT,
    'rows': len(batches), 'scored_tokens': ntok,
    'bitwise_identical_to_source': bool(exact), 'max_abs_logit_delta': max_delta,
    'ce_source_nats': ce_src / ntok, 'ce_gated_nats': ce_gat / ntok,
    'branch_params_with_nonzero_grad': len(nonzero), 'branch_params_checked': len(grads),
    'perturbation_changes_output': bool(live),
    'load_info': info,
    'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}, indent=2) + '\n', encoding='utf-8')
print('\nwrote', out / f'GATED_VERIFY_split{SPLIT}.json')
