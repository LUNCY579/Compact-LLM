"""Does the retention gate fire on the run that cross-entropy says is improving?

This is the test the retention gate exists for, and the one it has not yet had. `G2_RETENTION_B28`
passes, but `arm_answers_changed` was 0 of 150: the registered engineering lane is 25 SGD steps
at lr 1e-5 and it changed nothing, so the gate is mechanically verified and behaviourally
untested.

This programme already measured the case that matters: **lr 1e-4 destroys the frozen source's
arithmetic (6/6 -> 0/6) while IMPROVING held-out cross-entropy**, and lr 1e-5 preserves it. That
is the decoupling no likelihood endpoint here can see. So train the source at a given learning
rate and, at each milestone, record both:

  * held-out CE, which is what a likelihood gate would look at, and
  * typed retention against the FROZEN reference from the G1_RETENTION_B28 receipt.

If CE falls while retention falls too, the capability-side criterion does what no likelihood
threshold in this programme can. If retention does not move while capability is destroyed, the
gate is not fit for purpose -- which is the more important result and must be reported as such.

    python experiments/rab001_exploration/retention_dose.py <lr> [budget] [seed] [device] [optim]

The reference is read, never recomputed: recomputing it from the trained model would let the run
redefine what it was supposed to have kept.

Memory, and why the first run of this script aborted. It deliberately does NOT build the
reference-logit cache `long_budget.py` uses (2.82 GB memmap), because KL is not needed -- but
model + grads + AdamW moments is still ~9.5 GB steady state on a 15.2 GB host, and AdamW
allocates both moment buffers at once on the first `step()`. With another project's job holding
5.3 GB, free physical fell to 0.22 GiB and `hostbudget.guard` stood the run down mid-flight with
`completed: false`, which is the designed behaviour. It now refuses at STARTUP instead: waiting
beats paging, measured 10x on this machine. Run `sgd` (6.0 GiB) if the host stays busy, but note
the established destructive result is AdamW -- a different optimizer is a different experiment
and must be labelled as one.

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

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import hostbudget                                                       # noqa: E402
from hostbudget import LowMemoryAbort                                   # noqa: E402
from experiments.architecture_benchmark_001 import retention, run       # noqa: E402
from experiments.architecture_benchmark_001.gates import (implementation_digest,       # noqa: E402
                                                          protocol)
from experiments.architecture_benchmark_001.source import ROOT          # noqa: E402

SRC = Path('C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/'
           'Rineya-Supa-1.7.0-devfreeze')
SPL = ROOT / 'outputs' / 'splits_large'
OUT = ROOT / 'evidence' / 'retention-design-v1'


def _reference_receipt() -> Path:
    """The G1_RETENTION_B28 artifact for the CURRENT implementation digest.

    This was pinned to a literal `retention-125d5481` path, which is the class of staleness this
    programme keeps finding in its own documents: the reference happened to be byte-identical
    after the next requalification, so a wrong result would not have announced itself. Resolve it
    from the digest instead, and refuse rather than silently reach for a superseded receipt.
    """
    short = implementation_digest()[:8]
    path = ROOT / 'outputs' / f'retention-{short}' / 'B28' / 'retention.json'
    if not path.is_file():
        raise SystemExit(
            f'no G1_RETENTION_B28 artifact at the current digest {short}: {path}\n'
            'Run the retention stage at this digest first -- the frozen reference must come '
            'from the same implementation that will score the candidate.')
    return path

LR = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-4
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 5200
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 17
DEVICE = sys.argv[4] if len(sys.argv) > 4 else 'cpu'
OPTIM = sys.argv[5] if len(sys.argv) > 5 else 'adamw'
N_DEV, ACCUM = 40, 1

#: Steady-state host memory, measured not guessed. Params and grads are 2.38 GB each at 596M
#: fp32; AdamW adds two moment buffers at 2.38 GB apiece. The first run of this script aborted
#: at free 0.22 GiB because AdamW allocates those moments all at once on the first step(), while
#: another project's job held 5.3 GB on the same 15.2 GB host. Refuse at startup instead: a
#: paging run is slower than waiting, measured 10x on this machine.
OPTIM_GIB = {'adamw': 11.0, 'sgd_momentum': 8.5, 'sgd': 6.0}
if OPTIM not in OPTIM_GIB:
    raise SystemExit(f'optimizer must be one of {sorted(OPTIM_GIB)}')
WARMUP_FRAC, MIN_LR_FRAC = 0.03, 0.10
TAG = f'{OPTIM}_lr{LR:g}_b{BUDGET}_seed{SEED}'


def lr_at(done: int) -> float:
    frac = min(1.0, done / BUDGET)
    if frac < WARMUP_FRAC:
        return LR * frac / WARMUP_FRAC
    p = (frac - WARMUP_FRAC) / max(1e-9, 1 - WARMUP_FRAC)
    return LR * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * p)))


def main() -> int:
    hostbudget.require(OPTIM_GIB[OPTIM], f'retention dose {TAG} ({OPTIM})')
    print(hostbudget.describe(), flush=True)
    policy = protocol()['G1_RETENTION']

    receipt = _reference_receipt()
    frozen = json.loads(receipt.read_text(encoding='utf-8'))
    references = {k: set(v) for k, v in frozen['reference_ids_by_policy'].items()}
    source_correct = set(frozen['source_correct_ids'])
    print(f'frozen reference: {len(references[policy["bare_policy"]])} items '
          f'({policy["bare_policy"]}), source_correct {len(source_correct)}', flush=True)

    rows = run.load_probe_rows(ROOT / 'data' / 'monitor_probe.jsonl')
    audit = json.loads((ROOT / 'evidence/retention-design-v1/PROBE_LABEL_AUDIT.json')
                       .read_text(encoding='utf-8'))
    overlays = {n: retention.build_overlay(audit, n) for n in retention.BARE_POLICIES}

    # The source side must come from the SOURCE, not from the candidate: `source_probe_informative`
    # is what stops a probe the teacher cannot do reading as a clean verdict, and feeding it the
    # candidate's own scores would turn a destroyed arm's FAIL into an INSUFFICIENT. The frozen
    # receipt stores the source's raw text per row, so re-score that -- no model needed, and it
    # is the same text the reference was built from.
    source_raw = {r['id']: r['source'] for r in frozen['rows']}
    if set(source_raw) != {r['id'] for r in rows}:
        raise SystemExit('frozen receipt does not cover exactly the probe rows')
    source_scored = {
        name: {i: {**retention.score_prediction(rec['raw'], rec['terminated'],
                                                overlays[name]['targets'][i]),
                   'reply_class': next(r['reply_class'] for r in rows if r['id'] == i)}
               for i, rec in source_raw.items()}
        for name in retention.BARE_POLICIES}
    for name in retention.BARE_POLICIES:
        got = {i for i, r in source_scored[name].items() if r['correct']}
        if got != references[name]:
            raise SystemExit(f're-scored source under {name} does not reproduce the frozen '
                             f'reference: {len(got)} vs {len(references[name])}')
    print('frozen source re-scored and reproduces both references exactly', flush=True)

    torch.manual_seed(SEED)
    model, tok, _ = run._load(SRC, DEVICE)
    eos = run._eos_ids(SRC, tok)
    # `_load` reads the full safetensors state dict (2.38 GB) and then builds the model from it.
    # The dict goes out of scope on return, but nothing forces the collection, and on a host with
    # ~8 GiB free that duplicate is the difference between running and standing down.
    gc.collect()
    print(f'after load + collect: {hostbudget.describe()}', flush=True)

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

    @torch.inference_mode()
    def dev_ce() -> float:
        model.eval()
        total = 0.0
        for ids in dev:
            lp = F.log_softmax(model(ids.to(DEVICE)).logits.float()[:, :-1], -1)[0]
            total += float(F.nll_loss(lp, ids[0, 1:].to(DEVICE), reduction='sum'))
            del lp
        model.train()
        return total / ntok

    def probe() -> dict:
        model.eval()
        outputs = run._greedy_probe(model, tok, rows, eos, DEVICE)
        model.train()
        record = {}
        for name, overlay in overlays.items():
            scored = run._score_probe(outputs, overlay['targets'])
            correct = {i for i, r in scored.items() if r['correct']}
            ref = references[name]
            counts = run._retention_classes(rows, source_scored[name], scored, ref)
            verdict = retention.monitor_verdict(counts, policy)
            record[name] = {'status': verdict['status'], 'correct_total': len(correct),
                            'lost_total': sum(c['lost'] for c in counts.values()),
                            'retained_total': sum(c['retained'] for c in counts.values()),
                            'reference_n': len(ref), 'per_class': counts,
                            'failed_classes': verdict['failed_classes']}
        record['samples'] = [{'id': o['id'], 'text': o['text'], 'terminated': o['terminated']}
                             for o in outputs[:8]]
        return record

    milestones = sorted({0, *(int(BUDGET * f) for f in (0.25, 0.5, 0.75, 1.0))})
    trainable = [p for p in model.parameters() if p.requires_grad]
    if OPTIM == 'adamw':
        opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
    else:
        opt = torch.optim.SGD(trainable, lr=LR,
                              momentum=0.9 if OPTIM == 'sgd_momentum' else 0.0)
    checkpoints, tokens, steps, micro, cursor = [], 0, 0, 0, 0
    started = time.perf_counter()
    aborted = None

    def checkpoint(done: int, milestone):
        ce = dev_ce()
        p = probe()
        reg = p[policy['bare_policy']]
        rec = {'tokens': done, 'milestone': milestone, 'steps': steps, 'lr': lr_at(done),
               'dev_ce_nats': ce, 'retention': p,
               'wall_seconds': time.perf_counter() - started}
        checkpoints.append(rec)
        print(f"  {done:>7} tok | step {steps:>4} | devCE {ce:7.4f} | "
              f"retained {reg['retained_total']:2d}/{reg['reference_n']} "
              f"lost {reg['lost_total']:2d} | {reg['status']:12s} | "
              f"{time.perf_counter() - started:.0f}s", flush=True)
        return rec

    try:
        checkpoint(0, 0)
        gc.collect()
        print(f'after step-0 probe: {hostbudget.describe()}', flush=True)
        model.train()
        opt.zero_grad(set_to_none=True)
        done_ms = {0}
        while tokens < BUDGET:
            if cursor >= len(train):
                raise SystemExit('TRAIN exhausted before the budget')
            ids = train[cursor].to(DEVICE)
            cursor += 1
            logits = model(ids).logits
            tgt = ids[:, 1:]
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                   tgt.reshape(-1))
            (loss / ACCUM).backward()
            tokens += tgt.numel()
            micro += 1
            if micro == ACCUM:
                for g in opt.param_groups:
                    g['lr'] = lr_at(tokens)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                steps += 1; micro = 0
                hostbudget.guard(1.5, f'retention dose {TAG}')
            for m in milestones:
                if m and tokens >= m and m not in done_ms:
                    done_ms.add(m)
                    checkpoint(tokens, m)
                    break
    except LowMemoryAbort as exc:
        aborted = str(exc)
        print(f'ABORTED: {exc}', flush=True)

    completed = aborted is None and tokens >= BUDGET
    first, last = checkpoints[0], checkpoints[-1]
    reg = policy['bare_policy']
    body = {
        'schema': 1, 'evidence_kind': 'exploratory_retention_dose_response_only',
        'note': 'Trains the frozen source at one learning rate and reports held-out CE beside '
                'typed retention against the FROZEN G1_RETENTION_B28 reference. Outside the '
                'hashed root: no receipt, no gate, open splits only, no efficacy claim.',
        'completed': completed,
        'completion_note': ('budget fully spent' if completed else
                            'INTERRUPTED -- not a complete run and must not be reported as one'),
        'aborted_reason': aborted,
        'lr': LR, 'token_budget': BUDGET, 'seed': SEED, 'device': DEVICE,
        'optimizer': f'{OPTIM} lr={LR:g} wd=0 clip=1.0 accum={ACCUM} micro_batch=1',
        'schedule': f'linear warmup {WARMUP_FRAC:.0%} then cosine to {MIN_LR_FRAC:.0%}, '
                    'driven by token count',
        'supervised_target_tokens': tokens, 'optimizer_steps': steps,
        'train_rows_consumed': cursor, 'dev_rows': len(dev), 'dev_target_tokens': ntok,
        'probe_rows': len(rows), 'registered_bare_policy': reg,
        'reference_receipt': str(receipt),
        'reference_digest': implementation_digest(),
        'reference_sizes': {k: len(v) for k, v in references.items()},
        'reference_recomputed': False,
        'headline': {
            'dev_ce_start': first['dev_ce_nats'], 'dev_ce_end': last['dev_ce_nats'],
            'dev_ce_improved': last['dev_ce_nats'] < first['dev_ce_nats'],
            'retained_start': first['retention'][reg]['retained_total'],
            'retained_end': last['retention'][reg]['retained_total'],
            'lost_end': last['retention'][reg]['lost_total'],
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
    print(f"  retained {h['retained_start']} -> {h['retained_end']} of "
          f"{body['reference_sizes'][reg]}   lost {h['lost_end']}")
    print(f"  gate     {h['status_start']} -> {h['status_end']}")
    print(f"  GATE FIRED WHILE CE IMPROVED: {h['gate_fired_while_ce_improved']}")
    print('  wrote', path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
