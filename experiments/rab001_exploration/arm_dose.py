"""Same dose, same data, same order -- four arms. Which one actually gets smarter?

`retention_dose_gpu.py` measured one arm (plain B28) and reported one side of the result: at
lr 1e-5 over 5,233 supervised tokens it LOST 6 of the 47 reference answers, so the gate said
FAIL. The other side was sitting unread in the same file. The same run took the 150-row panel
from **47 correct to 73** -- it gained 32 rows while losing 6, a net +26.

That number is the bar. It is what ordinary finetuning of the existing model buys, at zero
architectural cost, and any CED-family arm has to beat it to justify 1.94x the parameters and
2.04x the KV cache.

    python experiments/rab001_exploration/arm_dose.py <arm> [lr] [budget] [seed]

  B28        the source, every parameter trainable (596,049,920). Reproduces the existing
             lr 1e-5 run, but records per-row predictions, which that run did not.
  B28last8   the source, last 8 transformer blocks trainable (125,847,552). The matched-budget
             control: 5.2% MORE trainable parameters than H56 gets, no added inference cost.
  H          H56, added memory+reads trainable (119,571,072). The architecture.
  P          P56, identical tensor inventory, read restricted to the same token's memory. The
             control that isolates *context* from *parameter count*.

Every arm consumes the identical rows in the identical order from one seed, so every probe row
is a matched pair and the comparison is paired per-row (McNemar exact), not a rate difference.
Thresholds are pre-registered in `evidence/retention-design-v1/DISTINGUISHABILITY_PREREG.md`
and were fixed before any arm was trained.

Placement: train on the GPU, probe on the CPU. The frozen reference is a CPU measurement and
CUDA/CPU FP32 differ by ~1e-4, enough to flip a greedy argmax and manufacture a retention
"loss" that is really a device artifact. Step 0 is a device-integrity control: the untrained
arm, moved GPU -> CPU, must reproduce the frozen reference on all 150 rows or the run stops.

Recipe, identical across arms and inherited from `retention_dose_gpu.py`: plain next-token CE,
micro-batch 1, no accumulation, global-norm clip 1.0, AdamW(0.9, 0.999, eps 1e-8) wd 0, linear
warmup 3% then cosine to 10%. Adam moments split GPU/disk by `OffloadedAdam`, imported from
that module rather than reimplemented.

Exploratory. Outside the hashed root: no receipt, no gate, no stage opened, no efficacy claim,
and no D28-001 checkpoint is released.
"""
from __future__ import annotations

import gc
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import hostbudget                                                       # noqa: E402
from hostbudget import LowMemoryAbort                                    # noqa: E402
from experiments.architecture_benchmark_001 import retention, run        # noqa: E402
from experiments.architecture_benchmark_001.gates import (implementation_digest,  # noqa: E402
                                                          protocol)
from experiments.architecture_benchmark_001.source import ROOT           # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else 'H'
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5
BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 5200
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 17
if ARM not in ('B28', 'B28last8', 'H', 'P'):
    raise SystemExit(__doc__)
TAG = f'{ARM}_lr{LR:g}_b{BUDGET}_seed{SEED}'

# OffloadedAdam and force_tf32_off are proven in retention_dose_gpu; import rather than fork.
# That module reads its own constants from argv at import time, so hand it the matching ones.
_saved_argv = sys.argv
sys.argv = ['retention_dose_gpu', str(LR), str(BUDGET), str(SEED)]
try:
    from experiments.rab001_exploration.retention_dose_gpu import (OffloadedAdam,   # noqa: E402
                                                                   force_tf32_off)
finally:
    sys.argv = _saved_argv

SRC = REPO / 'Supa'
if not (SRC / 'model.safetensors').is_file():
    SRC = Path('C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/'
               'Rineya-Supa-1.7.0-devfreeze')
SPL = ROOT / 'outputs' / 'splits_large'
OUT = ROOT / 'evidence' / 'retention-design-v1'
N_DEV = 40
WARMUP_FRAC, MIN_LR_FRAC = 0.03, 0.10


def lr_at(done: int) -> float:
    import math
    frac = min(1.0, done / BUDGET)
    if frac < WARMUP_FRAC:
        return LR * frac / WARMUP_FRAC
    p = (frac - WARMUP_FRAC) / (1 - WARMUP_FRAC)
    return LR * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * p)))


def build(device: str):
    """Returns (model, trainable, description). H/P peak at source+result, so build on the GPU."""
    if ARM in ('B28', 'B28last8'):
        model = run._load(SRC, device)[0]
        if ARM == 'B28':
            trainable = [p for p in model.parameters()]
            for p in trainable:
                p.requires_grad_(True)
            return model, trainable, 'B28 source, all parameters'
        index = lambda n: int(m.group(2)) if (m := re.match(r'^(.*?)\.(\d+)\.', n)) else -1
        top = max(index(n) for n, _ in model.named_parameters())
        for name, p in model.named_parameters():
            p.requires_grad_(index(name) >= top - 7)
        trainable = [p for p in model.parameters() if p.requires_grad]
        return model, trainable, f'B28 source, blocks {top-7}..{top} trainable'
    from experiments.dual28_ced_v1.model import DualStack
    src = run._load(SRC, device)[0]
    model = DualStack.from_source(src, ARM)
    del src
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    model.gradient_checkpointing = False
    return model, model.freeze_backbones(), f'DualStack.from_source(mode={ARM}), memory+reads'


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit('this harness trains on the GPU; CUDA is required')
    numerics = force_tf32_off()
    # Host peak is the probe: one fp32 copy of the arm on the CPU. H56 is 4.31 GiB, B28 2.22.
    hostbudget.require(6.0 if ARM in ('H', 'P') else 3.6, f'arm dose {TAG}')
    free_v, total_v = torch.cuda.mem_get_info()
    print(f'{hostbudget.describe()} | VRAM free {free_v/2**30:.2f}/{total_v/2**30:.2f} GiB')
    print(f'TF32 forced off: {numerics}', flush=True)

    policy = protocol()['G1_RETENTION']
    short = implementation_digest()[:8]
    receipt_path = ROOT / 'outputs' / f'retention-{short}' / 'B28' / 'retention.json'
    if not receipt_path.is_file():
        raise SystemExit(f'no G1_RETENTION_B28 artifact at digest {short}: {receipt_path}')
    frozen = json.loads(receipt_path.read_text(encoding='utf-8'))
    references = {k: set(v) for k, v in frozen['reference_ids_by_policy'].items()}
    source_raw = {r['id']: r['source'] for r in frozen['rows']}

    rows = run.load_probe_rows(ROOT / 'data' / 'monitor_probe.jsonl')
    audit = json.loads((OUT / 'PROBE_LABEL_AUDIT.json').read_text(encoding='utf-8'))
    overlays = {n: retention.build_overlay(audit, n) for n in retention.BARE_POLICIES}
    if set(source_raw) != {r['id'] for r in rows}:
        raise SystemExit('frozen receipt does not cover exactly the probe rows')
    source_scored = {}
    for name, overlay in overlays.items():
        scored = {i: {**retention.score_prediction(rec['raw'], rec['terminated'],
                                                   overlay['targets'][i]),
                      'reply_class': next(r['reply_class'] for r in rows if r['id'] == i)}
                  for i, rec in source_raw.items()}
        if {i for i, r in scored.items() if r['correct']} != references[name]:
            raise SystemExit(f're-scored source under {name} does not reproduce the reference')
        source_scored[name] = scored
    print(f'frozen reference reproduced: {({k: len(v) for k, v in references.items()})}',
          flush=True)

    torch.manual_seed(SEED)
    tok = run._load_tokenizer(SRC)
    eos = run._eos_ids(SRC, tok)
    jl = lambda p: [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines()
                    if l.strip()]
    train_rows, dev_rows = jl(SPL / 'train.jsonl'), jl(SPL / 'dev.jsonl')[:N_DEV]
    fam = lambda rs: {r['group_id'] for r in rs}
    assert not (fam(train_rows) & fam(rows)), 'train/probe family overlap'
    assert not (fam(train_rows) & fam(dev_rows)), 'train/dev family overlap'
    random.Random(SEED).shuffle(train_rows)
    enc = lambda t: tok(t, return_tensors='pt', add_special_tokens=False)['input_ids']
    train = [x for x in (enc(r['text']) for r in train_rows) if x.shape[1] >= 2]
    dev = [x for x in (enc(r['text']) for r in dev_rows) if x.shape[1] >= 2]
    ntok = sum(x.shape[1] - 1 for x in dev)

    model, trainable, described = build('cuda')
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in trainable)
    gc.collect()
    torch.cuda.empty_cache()
    print(f'{ARM}: {described} | {n_all:,} params, {n_train:,} trainable | '
          f'VRAM free {torch.cuda.mem_get_info()[0]/2**30:.2f} GiB', flush=True)

    params_bytes = sum(p.numel() * 4 for p in trainable)
    reserve = params_bytes + int(0.6 * 2 ** 30)
    gpu_budget = max(0, torch.cuda.mem_get_info()[0] - reserve)
    spill = ROOT / 'outputs' / 'adam-spill'
    opt = OffloadedAdam(trainable, LR, gpu_budget, spill)
    print(f'Adam moments {opt.bytes()/2**30:.2f} GiB: {opt.gpu_bytes/2**30:.2f} on GPU, '
          f'{opt.disk_bytes()/2**30:.2f} on disk | VRAM free '
          f'{torch.cuda.mem_get_info()[0]/2**30:.2f} GiB', flush=True)

    @torch.inference_mode()
    def dev_ce() -> float:
        model.eval()
        total = 0.0
        for ids in dev:
            lp = F.log_softmax(model(ids.to('cuda')).logits.float()[:, :-1], -1)[0]
            total += float(F.nll_loss(lp, ids[0, 1:].to('cuda'), reduction='sum'))
            del lp
        model.train()
        return total / ntok

    def probe() -> dict:
        """Model to CPU, moments stay on the GPU. Neither device ever holds both."""
        model.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        hostbudget.guard(1.2, f'arm dose {TAG} probe')
        model.eval()
        outputs = run._greedy_probe(model, tok, rows, eos, 'cpu')
        model.to('cuda')
        model.train()
        gc.collect()
        torch.cuda.empty_cache()
        record = {}
        for name, overlay in overlays.items():
            scored = run._score_probe(outputs, overlay['targets'])
            counts = run._retention_classes(rows, source_scored[name], scored, references[name])
            verdict = retention.monitor_verdict(counts, policy, policy['probe_classes'])
            record[name] = {
                'status': verdict['status'],
                'correct_total': sum(1 for r in scored.values() if r['correct']),
                'lost_total': sum(c['lost'] for c in counts.values()),
                'retained_total': sum(c['retained'] for c in counts.values()),
                'terminated_total': sum(c['terminated'] for c in counts.values()),
                'gained_total': sum(c['gained'] for c in counts.values()),
                'reference_n': len(references[name]), 'per_class': counts,
                'failed_classes': verdict['failed_classes']}
            if name == policy['bare_policy']:
                # Per-row correctness is what makes the arms comparable as matched pairs.
                record['per_row'] = {i: bool(r['correct']) for i, r in sorted(scored.items())}
        record['_outputs'] = outputs
        return record

    milestones = sorted({0, *(int(BUDGET * f) for f in (0.25, 0.5, 0.75, 1.0))})
    checkpoints, tokens, steps, cursor, consumed = [], 0, 0, 0, []
    started = time.perf_counter()
    aborted = None
    reg = policy['bare_policy']

    def checkpoint(done: int, milestone):
        ce = dev_ce()
        p = probe()
        outputs = p.pop('_outputs')
        changed = sum(1 for o in outputs if o['text'] != source_raw[o['id']]['raw'])
        r = p[reg]
        rec = {'tokens': done, 'milestone': milestone, 'steps': steps, 'lr': lr_at(done),
               'dev_ce_nats': ce, 'answers_changed_vs_source': changed, 'retention': p,
               'wall_seconds': time.perf_counter() - started}
        if milestone == milestones[-1] or milestone == 0:
            rec['outputs'] = {o['id']: {'text': o['text'], 'terminated': o['terminated']}
                              for o in outputs}
        checkpoints.append(rec)
        print(f"  {done:>7} tok | step {steps:>3} | devCE {ce:7.4f} | "
              f"kept {r['retained_total']:2d}/{r['reference_n']} lost {r['lost_total']:2d} "
              f"gain {r['gained_total']:2d} | correct {r['correct_total']:3d}/150 | "
              f"changed {changed:3d}/150 | {r['status']:12s} | "
              f"{time.perf_counter()-started:.0f}s", flush=True)
        return rec

    try:
        first = checkpoint(0, 0)
        if first['answers_changed_vs_source'] != 0:
            raise SystemExit(
                f"device-integrity control FAILED: {first['answers_changed_vs_source']} of 150 "
                'answers differ from the frozen CPU reference before any training. GPU->CPU '
                'transfer is not answer-preserving here and no retention number would mean '
                'anything.')
        print('  device-integrity control PASSED: 0/150 answers changed before training',
              flush=True)

        model.train()
        opt.zero_grad()
        done_ms = {0}
        while tokens < BUDGET:
            if cursor >= len(train):
                raise SystemExit('TRAIN exhausted before the budget')
            ids = train[cursor].to('cuda')
            cursor += 1
            logits = model(ids).logits
            tgt = ids[:, 1:]
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                   tgt.reshape(-1))
            loss.backward()
            tokens += tgt.numel()
            consumed.append(int(tgt.numel()))
            del logits
            torch.nn.utils.clip_grad_norm_(opt.params, 1.0)
            opt.step(lr_at(tokens))
            opt.zero_grad()
            steps += 1
            hostbudget.guard(1.0, f'arm dose {TAG}')
            for m in milestones:
                if m and tokens >= m and m not in done_ms:
                    done_ms.add(m)
                    checkpoint(tokens, m)
                    break
    except LowMemoryAbort as exc:
        aborted = str(exc)
        print(f'ABORTED: {exc}', flush=True)

    completed = aborted is None
    first, last = checkpoints[0], checkpoints[-1]
    fr, lr_ = first['retention'][reg], last['retention'][reg]
    body = {
        'schema': 1, 'evidence_kind': 'exploratory_arm_dose_response_only',
        'note': 'Matched-dose comparison across arms. Trains on the GPU, probes on the CPU (the '
                'frozen reference device). Per-row correctness is recorded so arms can be '
                'compared as matched pairs. Outside the hashed root: no receipt, no gate, no '
                'stage opened, no efficacy claim, no checkpoint released.',
        'arm': ARM, 'arm_description': described, 'completed': completed,
        'aborted_reason': aborted, 'lr': LR, 'token_budget': BUDGET, 'seed': SEED,
        'parameters': n_all, 'trainable_parameters': n_train,
        'train_device': 'cuda', 'probe_device': 'cpu', 'numerics': numerics,
        'optimizer': f'AdamW(offloaded) lr={LR:g} wd=0 clip=1.0 micro_batch=1',
        'schedule': f'linear warmup {WARMUP_FRAC:.0%} then cosine to {MIN_LR_FRAC:.0%}',
        'supervised_target_tokens': tokens, 'optimizer_steps': steps,
        'train_rows_consumed': cursor, 'per_step_target_tokens': consumed,
        'reference_receipt': str(receipt_path), 'reference_digest': implementation_digest(),
        'reference_sizes': {k: len(v) for k, v in references.items()},
        'registered_bare_policy': reg,
        'device_integrity_control_passed': first['answers_changed_vs_source'] == 0,
        'headline': {
            'dev_ce_start': first['dev_ce_nats'], 'dev_ce_end': last['dev_ce_nats'],
            'correct_start': fr['correct_total'], 'correct_end': lr_['correct_total'],
            'gained_end': lr_['gained_total'], 'lost_end': lr_['lost_total'],
            'net_end': lr_['gained_total'] - lr_['lost_total'],
            'retained_end': lr_['retained_total'],
            'answers_changed_end': last['answers_changed_vs_source'],
            'status_start': fr['status'], 'status_end': lr_['status']},
        'checkpoints': checkpoints, 'wall_seconds': time.perf_counter() - started,
        'licenses_registered_training': False, 'licenses_efficacy_claim': False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'ARM_DOSE_{TAG}.json'
    path.write_text(json.dumps(body, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    h = body['headline']
    print(f"\n{TAG}: {'COMPLETE' if completed else 'INTERRUPTED'}")
    print(f"  correct {h['correct_start']}/150 -> {h['correct_end']}/150 "
          f"(gained {h['gained_end']}, lost {h['lost_end']}, net {h['net_end']:+d})")
    print(f"  devCE   {h['dev_ce_start']:.4f} -> {h['dev_ce_end']:.4f} | "
          f"gate {h['status_start']} -> {h['status_end']}")
    print('  wrote', path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
