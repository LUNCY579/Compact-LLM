"""The lr 1e-4 dose test, split across both devices so it fits.

The question is the one the retention gate exists for and has never been asked: this programme
measured that **lr 1e-4 destroys the frozen source's arithmetic (6/6 -> 0/6) while IMPROVING
held-out cross-entropy**. Does the capability-side gate fire on that run?

Three CPU-only attempts were stood down by `hostbudget.guard`: AdamW fp32 on 596M parameters
needs ~9.5 GB (params 2.38 + grads 2.38 + two moments 4.77) and this 15.2 GiB host keeps ~7 GiB
for the OS and desktop applications. Dropping the moments for plain SGD saved 4.7 GB and still
aborted, so the optimizer was never the constraint.

**The GPU is the less contended resource here** -- ~6.9 GiB free against the host's ~5.8, while
another project's job holds ~5.5 GB of this machine -- so the work splits by what each device is
short of, and neither device is ever asked to hold the whole problem:

    training   params + gradients on the GPU; Adam moments split between both devices, filling
               the GPU up to whatever VRAM is actually free at startup and spilling the rest to
               the host. Host-resident moments stream through the GPU in bounded chunks.
    probing    model on the CPU; the GPU-resident moments spill to disk so the probe
               has the whole card, then come back afterwards.

Gradients have to be resident all at once because the recipe clips on the global gradient norm,
so they are budgeted for rather than fused away. A first attempt put the whole 4.44 GiB moment
set on the host and drove commit to 18.8 GiB over physical; the split is what makes it fit.

Why the probe is NOT on the GPU, which is the part that matters scientifically:

  * CUDA FP32 differs from CPU FP32 by up to 9.97e-05 per logit with TF32 disabled. That is
    inside the parity tolerance for a single forward, but greedy decoding turns it into a
    discrete choice 3,600 times per probe. A flipped argmax becomes a changed answer, and a
    changed answer becomes a fabricated retention "loss" against a reference that was measured
    on CPU. The comparison has to happen on the reference's device.
  * Sustained GPU work in this programme has crashed three times with
    CUBLAS_STATUS_EXECUTION_FAILED. One cause was found and fixed; the second was never
    root-caused, and it lives in exactly this shape of workload -- long runs of sequential
    cached-decode calls. That is the probe.

TF32 is forced off and asserted: cuDNN defaults it ON on this Ada card, and it costs 5.42e-02,
540x outside tolerance.

    python experiments/rab001_exploration/retention_dose_gpu.py <lr> [budget] [seed]

Step 0 carries a device-integrity control: the untrained model moved GPU -> CPU must reproduce
the frozen reference EXACTLY (47 permissive / 41 strict). If moving devices changed any answer,
nothing downstream would mean anything and the run aborts.

Exploratory. Outside the hashed root: no receipt, no gate, open splits only, no efficacy claim.
"""
from __future__ import annotations

import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import hostbudget                                                          # noqa: E402
from hostbudget import LowMemoryAbort                                      # noqa: E402
from experiments.architecture_benchmark_001 import retention, run          # noqa: E402
from experiments.architecture_benchmark_001.gates import (implementation_digest,   # noqa: E402
                                                          protocol)
from experiments.architecture_benchmark_001.source import ROOT             # noqa: E402

SRC = Path('C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/'
           'Rineya-Supa-1.7.0-devfreeze')
SPL = ROOT / 'outputs' / 'splits_large'
OUT = ROOT / 'evidence' / 'retention-design-v1'

LR = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-4
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 5200
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 17
N_DEV, ACCUM = 40, 1
WARMUP_FRAC, MIN_LR_FRAC = 0.03, 0.10
B1, B2, EPS = 0.9, 0.999, 1e-8
#: Elements per streamed optimizer chunk. Bounds the GPU-side temporaries of the offloaded Adam
#: update to about three chunks; 16M elements at fp32 is 64 MB apiece.
CHUNK = 16 << 20
TAG = f'gpu_lr{LR:g}_b{BUDGET}_seed{SEED}'


def lr_at(done: int) -> float:
    frac = min(1.0, done / BUDGET)
    if frac < WARMUP_FRAC:
        return LR * frac / WARMUP_FRAC
    p = (frac - WARMUP_FRAC) / max(1e-9, 1 - WARMUP_FRAC)
    return LR * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * p)))


def force_tf32_off() -> dict:
    """cuDNN defaults TF32 ON for this card and it costs 5.42e-02 -- 540x outside tolerance."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.fp32_precision = 'ieee' if hasattr(
        torch.backends.cuda.matmul, 'fp32_precision') else None
    state = {'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
             'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
             'float32_matmul_precision': torch.get_float32_matmul_precision()}
    assert not state['cuda_matmul_allow_tf32'] and not state['cudnn_allow_tf32'], state
    return state


class OffloadedAdam:
    """AdamW (weight_decay=0) whose moments live on the CPU while the parameters live on the GPU.

    The update streams each parameter through the GPU in chunks, so the GPU-side temporaries stay
    bounded and the host never allocates anything larger than a chunk. `release_gpu()` and
    `reacquire_gpu()` hand the card to the probe and take it back, so the model can sit on the
    host while it is scored there.
    """

    def __init__(self, params, lr: float, gpu_budget_bytes: int, spill_dir: Path):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.t = 0
        self.spill_dir = spill_dir
        # Two tiers, and the second is DISK rather than the host. The host is the contended
        # resource on this machine -- another project's job holds ~5.5 GB of 15.2 GiB -- while
        # 38 GB of disk sits idle, and file-backed pages are evictable so they never add to the
        # commit charge that caused three earlier aborts. Fill the GPU up to whatever VRAM is
        # actually free, spill the rest to memmap. The cost is I/O per optimizer step, which is
        # the trade this run is allowed to make: slower, but it completes.
        order = sorted(range(len(self.params)), key=lambda i: self.params[i].numel())
        self.home = ['disk'] * len(self.params)
        used = 0
        for i in order:                     # smallest first: many small tensors stay resident
            need = self.params[i].numel() * 4 * 2
            if used + need <= gpu_budget_bytes:
                self.home[i] = 'cuda'
                used += need
        self.gpu_bytes = used
        spill_dir.mkdir(parents=True, exist_ok=True)
        self.m, self.v = [], []
        for i, p in enumerate(self.params):
            if self.home[i] == 'cuda':
                self.m.append(torch.zeros(p.numel(), dtype=torch.float32, device='cuda'))
                self.v.append(torch.zeros(p.numel(), dtype=torch.float32, device='cuda'))
            else:
                mm = np.memmap(spill_dir / f'm{i}.f32', dtype='float32',
                               mode='w+', shape=(p.numel(),))
                vv = np.memmap(spill_dir / f'v{i}.f32', dtype='float32',
                               mode='w+', shape=(p.numel(),))
                # A freshly zeroed memmap is dirty pages until it is flushed, and dirty pages are
                # resident. Push them to the file now so the tier is genuinely off the host.
                mm.flush(); vv.flush()
                self.m.append(mm)
                self.v.append(vv)

    def bytes(self) -> int:
        return sum(p.numel() * 4 for p in self.params) * 2

    def disk_bytes(self) -> int:
        return sum(p.numel() * 4 for p, h in zip(self.params, self.home) if h == 'disk') * 2

    def release_gpu(self) -> None:
        """Spill the GPU-resident moments to disk so the probe can have the whole card."""
        for i, h in enumerate(self.home):
            if h != 'cuda':
                continue
            for store, name in ((self.m, 'm'), (self.v, 'v')):
                arr = np.memmap(self.spill_dir / f'park_{name}{i}.f32', dtype='float32',
                                mode='w+', shape=(store[i].numel(),))
                arr[:] = store[i].cpu().numpy()
                arr.flush()
                store[i] = arr
        gc.collect()
        torch.cuda.empty_cache()

    def reacquire_gpu(self) -> None:
        """Restore the startup placement after a probe."""
        for i, h in enumerate(self.home):
            if h != 'cuda':
                continue
            for store in (self.m, self.v):
                store[i] = torch.from_numpy(np.asarray(store[i])).to('cuda')
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def step(self, lr: float) -> None:
        self.t += 1
        bc1 = 1 - B1 ** self.t
        bc2 = 1 - B2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad.view(-1)
            flat = p.view(-1)
            resident = isinstance(m, torch.Tensor)
            for start in range(0, flat.numel(), CHUNK):
                stop = min(start + CHUNK, flat.numel())
                gg = g[start:stop]
                # Moments already on the GPU are updated in place; host-resident ones stream
                # through in bounded chunks so neither device holds a second full copy.
                if resident:
                    gm, gv = m[start:stop], v[start:stop]
                else:
                    gm = torch.from_numpy(np.asarray(m[start:stop])).to(p.device)
                    gv = torch.from_numpy(np.asarray(v[start:stop])).to(p.device)
                gm.mul_(B1).add_(gg, alpha=1 - B1)
                gv.mul_(B2).addcmul_(gg, gg, value=1 - B2)
                denom = (gv / bc2).sqrt_().add_(EPS)
                flat[start:stop].addcdiv_(gm, denom, value=-lr / bc1)
                if not resident:
                    m[start:stop] = gm.cpu().numpy()
                    v[start:stop] = gv.cpu().numpy()
                    del gm, gv
                del denom

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit('this variant requires CUDA; use retention_dose.py for CPU')
    numerics = force_tf32_off()
    # The host only ever holds the runtime plus, during a probe, one fp32 copy of the
    # model (2.22 GiB). The moments live on the GPU and on disk. 3.6 GiB is that peak
    # plus a margin -- not the 5.0 the abandoned host-resident design needed.
    hostbudget.require(3.6, f'retention dose {TAG}')
    free_v, total_v = torch.cuda.mem_get_info()
    print(f'{hostbudget.describe()} | VRAM free {free_v/2**30:.2f}/{total_v/2**30:.2f} GiB',
          flush=True)
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
    audit = json.loads((ROOT / 'evidence/retention-design-v1/PROBE_LABEL_AUDIT.json')
                       .read_text(encoding='utf-8'))
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
    model, tok, _ = run._load(SRC, 'cuda')
    eos = run._eos_ids(SRC, tok)
    gc.collect()

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

    # Reserve what training itself needs on the GPU before handing any VRAM to the optimizer:
    # a second copy of the parameters for gradients (global-norm clipping needs them all at
    # once), plus headroom for activations and the allocator.
    params_bytes = sum(p.numel() * 4 for p in model.parameters() if p.requires_grad)
    free_v = torch.cuda.mem_get_info()[0]
    reserve = params_bytes + int(0.6 * 2 ** 30)
    gpu_budget = max(0, free_v - reserve)
    spill = ROOT / 'outputs' / 'adam-spill'
    opt = OffloadedAdam(model.parameters(), LR, gpu_budget, spill)
    print(f'Adam moments {opt.bytes()/2**30:.2f} GiB total: '
          f'{opt.gpu_bytes/2**30:.2f} GiB on GPU, {opt.disk_bytes()/2**30:.2f} GiB on disk '
          f'(GPU budget {gpu_budget/2**30:.2f}, spill {spill})', flush=True)
    print(f'  after optimizer alloc | {hostbudget.describe()} | '
          f'VRAM free {torch.cuda.mem_get_info()[0]/2**30:.2f} GiB', flush=True)

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
        """Model to CPU, moments to GPU. Neither device ever holds both."""
        # The GPU-resident moments simply stay put: once the model leaves the card there is
        # room for both, and an earlier version spilled them to disk here, which cost a ~3.7 GiB
        # host spike (a full CPU copy plus dirty memmap pages) and tripped the guard at step 0.
        model.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        hostbudget.guard(1.2, f'retention dose {TAG} probe')
        model.eval()
        outputs = run._greedy_probe(model, tok, rows, eos, 'cpu')
        model.to('cuda')
        model.train()
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
                'reference_n': len(references[name]), 'per_class': counts,
                'failed_classes': verdict['failed_classes']}
        record['_outputs'] = outputs
        return record

    milestones = sorted({0, *(int(BUDGET * f) for f in (0.25, 0.5, 0.75, 1.0))})
    checkpoints, tokens, steps, cursor = [], 0, 0, 0
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
        checkpoints.append(rec)
        print(f"  {done:>7} tok | step {steps:>3} | devCE {ce:7.4f} | "
              f"kept {r['retained_total']:2d}/{r['reference_n']} lost {r['lost_total']:2d} | "
              f"changed {changed:3d}/150 | term {r['terminated_total']:3d}/150 | "
              f"{r['status']:12s} | {time.perf_counter()-started:.0f}s", flush=True)
        return rec

    try:
        first = checkpoint(0, 0)
        # Device-integrity control: untrained, moved GPU -> CPU, the model must reproduce the
        # frozen reference exactly. If crossing devices changed an answer, stop.
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
            del logits
            torch.nn.utils.clip_grad_norm_(opt.params, 1.0)
            opt.step(lr_at(tokens))
            opt.zero_grad()
            steps += 1
            hostbudget.guard(1.0, f'retention dose {TAG}')
            for m in milestones:
                if m and tokens >= m and m not in done_ms:
                    done_ms.add(m)
                    checkpoint(tokens, m)
                    break
    except LowMemoryAbort as exc:
        aborted = str(exc)
        print(f'ABORTED: {exc}', flush=True)
    except torch.cuda.OutOfMemoryError as exc:                       # pragma: no cover
        aborted = f'CUDA OOM: {exc}'
        print(f'ABORTED: {aborted}', flush=True)

    completed = aborted is None and tokens >= BUDGET
    if not checkpoints:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f'RETENTION_DOSE_{TAG}.json').write_text(json.dumps({
            'schema': 1, 'evidence_kind': 'exploratory_retention_dose_response_only',
            'completed': False, 'aborted_reason': aborted or 'no checkpoint completed',
            'completion_note': 'INTERRUPTED before the first checkpoint; nothing was measured',
            'lr': LR, 'token_budget': BUDGET, 'seed': SEED,
            'licenses_registered_training': False, 'licenses_efficacy_claim': False,
        }, indent=2) + '\n', encoding='utf-8')
        print(f'no checkpoint completed: {aborted}')
        return 1
    first, last = checkpoints[0], checkpoints[-1]
    body = {
        'schema': 1, 'evidence_kind': 'exploratory_retention_dose_response_only',
        'note': 'Trains the frozen source at one learning rate and reports held-out CE beside '
                'typed retention against the FROZEN G1_RETENTION_B28 reference. Training runs on '
                'the GPU with Adam moments offloaded to the host; every PROBE runs on the CPU, '
                'the reference device, because CUDA/CPU FP32 differ by ~1e-4 and greedy decoding '
                'turns that into changed answers. Outside the hashed root: no receipt, no gate, '
                'open splits only, no efficacy claim.',
        'completed': completed,
        'completion_note': ('budget fully spent' if completed else
                            'INTERRUPTED -- not a complete run and must not be reported as one'),
        'aborted_reason': aborted,
        'lr': LR, 'token_budget': BUDGET, 'seed': SEED,
        'train_device': 'cuda', 'probe_device': 'cpu', 'dev_ce_device': 'cuda',
        'numerics': numerics,
        'optimizer': f'AdamW(offloaded) lr={LR:g} wd=0 clip=1.0 accum={ACCUM} micro_batch=1',
        'adam_state_gib': opt.bytes() / 2 ** 30,
        'adam_state_gpu_gib': opt.gpu_bytes / 2 ** 30,
        'adam_state_disk_gib': opt.disk_bytes() / 2 ** 30,
        'schedule': f'linear warmup {WARMUP_FRAC:.0%} then cosine to {MIN_LR_FRAC:.0%}',
        'supervised_target_tokens': tokens, 'optimizer_steps': steps,
        'train_rows_consumed': cursor, 'dev_rows': len(dev), 'dev_target_tokens': ntok,
        'probe_rows': len(rows), 'registered_bare_policy': reg,
        'reference_receipt': str(receipt_path), 'reference_digest': implementation_digest(),
        'reference_sizes': {k: len(v) for k, v in references.items()},
        'reference_recomputed': False,
        'device_integrity_control_passed': first['answers_changed_vs_source'] == 0,
        'headline': {
            'dev_ce_start': first['dev_ce_nats'], 'dev_ce_end': last['dev_ce_nats'],
            'dev_ce_improved': last['dev_ce_nats'] < first['dev_ce_nats'],
            'retained_start': first['retention'][reg]['retained_total'],
            'retained_end': last['retention'][reg]['retained_total'],
            'lost_end': last['retention'][reg]['lost_total'],
            'answers_changed_end': last['answers_changed_vs_source'],
            'status_start': first['retention'][reg]['status'],
            'status_end': last['retention'][reg]['status'],
            'gate_fired_while_ce_improved': bool(
                last['dev_ce_nats'] < first['dev_ce_nats']
                and last['retention'][reg]['status'] != 'PASS')},
        'checkpoints': checkpoints,
        'wall_seconds': time.perf_counter() - started,
        'licenses_registered_training': False, 'licenses_efficacy_claim': False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'RETENTION_DOSE_{TAG}.json'
    path.write_text(json.dumps(body, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    h = body['headline']
    print(f"\n{TAG}: {'COMPLETE' if completed else 'INTERRUPTED'}")
    print(f"  dev CE   {h['dev_ce_start']:.4f} -> {h['dev_ce_end']:.4f} "
          f"({'improved' if h['dev_ce_improved'] else 'worsened'})")
    print(f"  kept     {h['retained_start']} -> {h['retained_end']} of "
          f"{body['reference_sizes'][reg]}   lost {h['lost_end']}")
    print(f"  answers changed vs source: {h['answers_changed_end']}/150")
    print(f"  gate     {h['status_start']} -> {h['status_end']}")
    print(f"  GATE FIRED WHILE CE IMPROVED: {h['gate_fired_while_ce_improved']}")
    print('  wrote', path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
