"""Does a displaced initialization recover if you actually pay for it? 200x the budget.

Every conclusion in this programme about CED trainability rests on ~5.2k supervised target
tokens. The registered ceiling is 1,000,000 per complete matched recipe. "Training does not
rescue a displaced initialization" is therefore established only at 0.5% of the affordable
budget, and this run is the test that could overturn it.

`rigorous_pilot.py` is the 40-step harness and cannot be scaled by raising its step count.
What changes here, and why:

  1. STOPS ON A TOKEN BUDGET, not a step count, so the registered ceiling is exact rather than
     estimated. The run also asserts it never revisits a TRAIN row, so a null result cannot be
     explained away as memorisation of a small set cycled many times.
  2. The frozen source's DEV log-probs are cached to a float32 memmap ON DISK instead of being
     retained in RAM. The old list is 2.82 GiB at 40 dev rows and 10.57 GiB at the full 150-row
     split -- larger than host RAM. The cache is keyed by the dev fingerprint and is shared by
     every arm and seed, so it is computed once and each later run skips it.
  3. LOG-SPACED checkpoints. The question is not only "does it recover" but "at what dose", so
     the run must produce a curve. Fixed cadence at this scale would also cost more in eval
     than in training.
  4. Warmup and cosine decay. A constant LR over 200x more steps makes the endpoint an artifact
     of where you stopped.
  5. Gradient ACCUMULATION at micro-batch 1, not padded batching. Padding introduces mask and
     loss-masking risk, and this programme has already had measurement bugs invert a published
     conclusion twice. Accumulation gives the same effective batch with per-row arithmetic
     identical to the 40-step runs.
  6. RESUMABLE. A multi-hour run that dies at hour three must not be lost, and a partial run
     must never be presentable as a complete one -- the artifact records `completed` explicitly
     and the budget is re-derived from the resumed token count, never re-spent.

The comparison is matched WITHIN this run: every arm sees the same recipe, the same splits, the
same seeds, the same device and dtype. It is NOT directly comparable to the 40-step CPU FP32
numbers if the device or dtype differs; the artifact records both so that can never be fudged.

Exploratory. Outside the hashed experiment root; no receipt, no gate, open splits only.

usage: long_budget.py <arm> <seed> <token_budget> [lr] [accum] [n_dev] [n_task] [device] [dtype]
"""
import gc, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification')
from experiments.architecture_benchmark_001.run import _load, _restore, _load_tokenizer
from experiments.rab001_exploration.scoring import answer_matches, self_test

E = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification'
         r'\experiments\architecture_benchmark_001')
SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble'
           r'\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')
LAD = E / 'outputs/ladder-fc44701e-v1'
SPL = E / 'outputs/splits_large'          # DEV/TASK are bit-identical to outputs/splits
CACHE = E / 'outputs/refcache'
OUT = E / 'outputs/longbudget'

ARM = sys.argv[1] if len(sys.argv) > 1 else 'C28'
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 17
BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
ACCUM = int(sys.argv[5]) if len(sys.argv) > 5 else 8
N_DEV = int(sys.argv[6]) if len(sys.argv) > 6 else 40
N_TASK = int(sys.argv[7]) if len(sys.argv) > 7 else 60
DEVICE = sys.argv[8] if len(sys.argv) > 8 else 'cpu'
DTYPE = sys.argv[9] if len(sys.argv) > 9 else 'float32'

CEILING = 1_000_000
if BUDGET > CEILING:
    raise SystemExit(f'budget {BUDGET} exceeds the registered ceiling {CEILING}')
EOS = {151643, 151645}
TAG = f'{ARM}_seed{SEED}_b{BUDGET}_lr{LR:g}'
WARMUP_FRAC, MIN_LR_FRAC = 0.03, 0.10
SAVE_EVERY_S = 1800   # the resume state is ~7.15 GB; this is crash recovery, not history
dtype = getattr(torch, DTYPE)

self_test()          # a drifted scorer already inverted one conclusion; fail loudly instead
torch.manual_seed(SEED)
tok = _load_tokenizer(SRC)

import random as _random
jl = lambda p: [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]
train_rows = jl(SPL / 'train.jsonl')
dev_rows = jl(SPL / 'dev.jsonl')[:N_DEV]
task_rows = jl(SPL / 'task_probe.jsonl')[:N_TASK]

fam = lambda rows: {r['group_id'] for r in rows}
assert not (fam(train_rows) & fam(dev_rows)), 'train/dev family overlap'
assert not (fam(train_rows) & fam(task_rows)), 'train/task family overlap'
assert not (fam(dev_rows) & fam(task_rows)), 'dev/task family overlap'
assert 'arithmetic_chain' not in {r['reply_class'] for r in train_rows}
assert len(fam(train_rows)) == len(train_rows), 'a TRAIN family appears twice'

_random.Random(SEED).shuffle(train_rows)
enc = lambda t: tok(t, return_tensors='pt', add_special_tokens=False)['input_ids']
train = [x for x in (enc(r['text']) for r in train_rows) if x.shape[1] >= 2]
dev = [x for x in (enc(r['text']) for r in dev_rows) if x.shape[1] >= 2]
dev_lens = [x.shape[1] - 1 for x in dev]
NTOK = sum(dev_lens)

# Which DEV target positions are the assistant's reply? Measured on TRAIN, only 10.0% of a
# row's tokens are the answer; the other 90% is system prompt, user turn and chat-template
# boilerplate. An overall CE is therefore dominated by text the model has seen a thousand
# times, which is a candidate mechanism for the CE-vs-capability decoupling already measured
# here. Both are reported; neither is dropped.
MARKER = tok('<|im_start|>assistant' + chr(10), add_special_tokens=False)['input_ids']
# The assistant turn opens with an empty reasoning scaffold. It is boilerplate too, so
# the mask starts after it when present; otherwise the 'reply' would still be ~44%
# scaffold and would understate the gap between arms.
THINK_END = tok('</think>' + chr(10) + chr(10), add_special_tokens=False)['input_ids']


def response_mask(ids):
    """Bool over the row's TARGET positions (ids[1:]): True inside the assistant reply.

    The marker is located in the row's own token sequence rather than by re-tokenising a text
    prefix, because re-tokenising a prefix can split differently at the boundary and would
    silently misalign the mask by a token or two.
    """
    seq = ids[0].tolist()
    m, cut = MARKER, -1
    for i in range(len(seq) - len(m), -1, -1):
        if seq[i:i + len(m)] == m:
            cut = i + len(m)
            break
    if cut >= 0:
        t = THINK_END
        for i in range(cut, len(seq) - len(t) + 1):
            if seq[i:i + len(t)] == t:
                cut = i + len(t)
                break
    mask = torch.zeros(len(seq) - 1, dtype=torch.bool)
    if cut >= 0:
        mask[max(0, cut - 1):] = True
    return mask


dev_resp = [response_mask(x) for x in dev]
NRESP = int(sum(int(m.sum()) for m in dev_resp))
if not NRESP:
    raise SystemExit('no assistant-reply positions found in DEV; the marker or template changed')
# The resume cursor is an index into this permuted list, so the resume is only valid if the
# split is byte-identical to the one the interrupted run used. Fingerprinted, checked below.
train_fingerprint = hashlib.sha256((SPL / 'train.jsonl').read_bytes()).hexdigest()
available = sum(x.shape[1] - 1 for x in train)
if available < BUDGET:
    raise SystemExit(f'TRAIN holds {available} target tokens, below the {BUDGET} budget: the run '
                     f'would have to revisit rows and would measure memorisation')

print(f'{TAG} | device {DEVICE} dtype {DTYPE} | train {len(train)} rows ({available} target '
      f'tokens available) | dev {len(dev)} rows ({NTOK} tokens, {NRESP} in the reply = '
      f'{NRESP / NTOK:.1%}) | task {len(task_rows)}', flush=True)
print(f'budget {BUDGET} target tokens = {BUDGET / available:.1%} of one epoch; accum {ACCUM}; '
      f'lr {LR:g} warmup {WARMUP_FRAC:.0%} cosine to {MIN_LR_FRAC:.0%}', flush=True)


# ---- frozen-source DEV reference, cached to disk ---------------------------
# Shared across arms and seeds. Keyed by the dev content so a changed split cannot silently
# reuse a stale cache.
digest = hashlib.sha1()
for x in dev:
    digest.update(x.numpy().tobytes())
fingerprint = f'{len(dev)}_{NTOK}_{digest.hexdigest()[:12]}'
CACHE.mkdir(parents=True, exist_ok=True)
ref_path = CACHE / f'devref_{fingerprint}.f32'
meta_path = CACHE / f'devref_{fingerprint}.json'

if not (ref_path.is_file() and meta_path.is_file()):
    t0 = time.perf_counter()
    src, _, _ = _load(SRC, 'cpu'); src.eval()
    V = src.config.vocab_size   # NOT tok.vocab_size; they differ on this family
    print(f'building source DEV reference cache -> {ref_path.name} '
          f'({NTOK * V * 4 / 2**30:.2f} GiB on disk, built once)', flush=True)
    mm = np.lib.format.open_memmap(ref_path, mode='w+', dtype=np.float32, shape=(NTOK, V))
    ce_src, at = 0.0, 0
    with torch.inference_mode():
        for ids in dev:
            lp = F.log_softmax(src(ids).logits.float()[:, :-1], -1)[0]
            n = lp.shape[0]
            mm[at:at + n] = lp.numpy()
            tgt = ids[0, 1:]
            ce_src += float(F.nll_loss(lp, tgt, reduction='sum'))
            at += n
            del lp
    mm.flush(); del mm
    ce_src /= NTOK
    meta_path.write_text(json.dumps({'dev_rows': len(dev), 'dev_target_tokens': NTOK,
                                     'vocab': V, 'source_dev_ce_nats': ce_src,
                                     'fingerprint': fingerprint}, indent=2), encoding='utf-8')
    del src; gc.collect()
    print(f'  cached in {time.perf_counter() - t0:.0f}s | source dev CE {ce_src:.4f}', flush=True)
else:
    print(f'reusing source DEV reference cache {ref_path.name}', flush=True)

meta = json.loads(meta_path.read_text(encoding='utf-8'))
ce_src, V = meta['source_dev_ce_nats'], meta['vocab']
assert meta['dev_target_tokens'] == NTOK, 'cache/dev token mismatch'
REF = np.load(ref_path, mmap_mode='r')


# ---- arm construction, identical to rigorous_pilot.py ----------------------
def build(arm):
    if arm in ('C28', 'C40'):
        return _restore(LAD / f'G1_CED_{arm}/attempt-0001', 'cpu')
    if arm == 'D40':
        return _restore(LAD / 'G1_WIDTH/attempt-0001', 'cpu')
    if arm.startswith('ced_last'):
        from dataclasses import replace as _replace
        from experiments.architecture_benchmark_001.transfer import to_ced
        n = int(arm[len('ced_last'):])
        s, _, _ = _load(SRC, 'cpu')
        e = s.config.total_layers - n
        m, _ = to_ced(s, _replace(s.config, architecture='ced', encoder_layers=e,
                                  decoder_layers=n), donor_layer=e)
        del s; gc.collect()
        return m
    if arm == 'source_control':
        return _load(SRC, 'cpu')[0]
    raise SystemExit(f'unknown arm {arm}')


model = build(ARM)
if DEVICE != 'cpu' or dtype != torch.float32:
    model = model.to(device=DEVICE, dtype=dtype)
model.gradient_checkpointing = True
trainable = [p for p in model.parameters() if p.requires_grad]
frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
assert not frozen, f'{len(frozen)} parameters are frozen; the comparison would not be matched'
opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)

OUT.mkdir(parents=True, exist_ok=True)
STATE = OUT / f'STATE_{TAG}.pt'
tokens = steps = row_cursor = 0
elapsed_prior = 0.0
checkpoints, traj = [], []
if STATE.is_file():
    blob = torch.load(STATE, map_location=DEVICE, weights_only=False)
    model.load_state_dict(blob['model']); opt.load_state_dict(blob['opt'])
    saved_fp = blob.get('train_fingerprint')
    if saved_fp and saved_fp != train_fingerprint:
        raise SystemExit(
            'REFUSING TO RESUME: outputs/splits_large/train.jsonl has changed since this run '
            'was interrupted (saved ' + saved_fp[:16] + ', now ' + train_fingerprint[:16] + '). '
            'The resume cursor indexes a seed-permuted view of that file, so continuing would '
            'revisit rows already trained and skip others while still reporting '
            'rows_revisited: 0. Rebuild the run from scratch or restore the original split.')
    tokens, steps, row_cursor = blob['tokens'], blob['steps'], blob['row_cursor']
    checkpoints, traj = blob['checkpoints'], blob['traj']
    elapsed_prior = blob.get('elapsed', 0.0)
    print(f'RESUMED at {tokens} tokens / {steps} steps / row {row_cursor}', flush=True)


def lr_at(tok_done):
    """Warmup then cosine decay, driven by TOKENS so a resume lands on the same schedule."""
    frac = min(1.0, tok_done / BUDGET)
    if frac < WARMUP_FRAC:
        return LR * frac / WARMUP_FRAC
    p = (frac - WARMUP_FRAC) / max(1e-9, 1 - WARMUP_FRAC)
    return LR * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.inference_mode()
def evaluate_dev():
    model.eval()
    ce = kl = rce = rkl = 0.0
    agree = ragree = 0
    at = 0
    for ids, rmask in zip(dev, dev_resp):
        lp = F.log_softmax(model(ids.to(DEVICE)).logits.float()[:, :-1], -1)[0].cpu()
        n = lp.shape[0]
        rf = torch.from_numpy(np.asarray(REF[at:at + n]))
        tgt = ids[0, 1:]
        per_tok = F.nll_loss(lp, tgt, reduction='none')
        per_kl = (rf.exp() * (rf - lp)).sum(-1)
        hit = (rf.argmax(-1) == lp.argmax(-1))
        ce += float(per_tok.sum()); kl += float(per_kl.sum()); agree += int(hit.sum())
        rce += float(per_tok[rmask].sum()); rkl += float(per_kl[rmask].sum())
        ragree += int(hit[rmask].sum())
        at += n
        del lp, rf, per_tok, per_kl, hit
    model.train()
    return {'dev_ce_nats': ce / NTOK, 'dev_kl_to_source_nats': kl / NTOK,
            'dev_top1_agreement': agree / NTOK,
            'dev_response_ce_nats': rce / NRESP, 'dev_response_kl_nats': rkl / NRESP,
            'dev_response_top1_agreement': ragree / NRESP}


@torch.inference_mode()
def evaluate_task():
    model.eval()
    numeric = strict = term = 0
    outs, distinct = [], set()
    for r in task_rows:
        cur = enc(tok.apply_chat_template(r['messages'], tokenize=False, enable_thinking=False,
                                          add_generation_prompt=True)).to(DEVICE)
        cache, got = None, []
        for _ in range(24):
            step = model(cur, past=cache, use_cache=True, last_position_only=True)
            cache = step.cache
            t = int(step.logits[0, -1].argmax())
            if t in EOS:
                term += 1; break
            got.append(t)
            cur = torch.tensor([[t]], dtype=torch.long, device=DEVICE)
        text = tok.decode(got, skip_special_tokens=True).strip()
        distinct.add(text)
        numeric += answer_matches(text, r['answer']); strict += (text == r['answer'])
        if len(outs) < 8:
            outs.append({'id': r['id'], 'expected': r['answer'], 'got': text[:50],
                         'numeric_match': bool(answer_matches(text, r['answer']))})
    model.train()
    n = len(task_rows)
    return {'task_numeric_hits': numeric, 'task_strict_hits': strict, 'task_n': n,
            'task_numeric_rate': numeric / n, 'task_terminated': term,
            'task_distinct': len(distinct), 'samples': outs}


# Log-spaced milestones: the question is at what DOSE recovery happens, not just whether.
MILESTONES = sorted({0, *(int(BUDGET * f) for f in
                          (0.005, 0.015, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0))})
t0 = time.perf_counter()


def checkpoint(done, milestone=None):
    d, t = evaluate_dev(), evaluate_task()
    rec = {'tokens': done, 'milestone': milestone, 'steps': steps, 'lr': lr_at(done), **d,
           **{k: v for k, v in t.items() if k != 'samples'}}
    checkpoints.append(rec)
    print(f"  {done:>8} tok | devCE {d['dev_ce_nats']:7.4f} (resp {d['dev_response_ce_nats']:7.4f}) "
          f"KL {d['dev_kl_to_source_nats']:6.3f} top1 {d['dev_top1_agreement']:.3f} | task {t['task_numeric_hits']:3d}/{t['task_n']} "
          f"({t['task_numeric_rate']:.2%}) strict {t['task_strict_hits']:3d} "
          f"term {t['task_terminated']:3d} dist {t['task_distinct']:3d} | "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    for s in t['samples'][:2]:
        print(f"        expected={s['expected']!r:8} got={s['got']!r:30} ok={s['numeric_match']}",
              flush=True)
    return t['samples']


# Seeded with the start time, not 0.0: otherwise the first periodic call fires
# immediately and writes 7.15 GB that is byte-identical to what is already on disk.
last_save = [time.perf_counter()]


def save_state(force=False):
    """Rolling crash-recovery point. ~7.15 GB per write (model 2.38 + Adam 4.77), so this is
    time-gated rather than fired at every milestone -- eight milestone saves would be ~57 GB of
    I/O per arm for a file that is only ever read after a crash."""
    now = time.perf_counter()
    if not force and now - last_save[0] < SAVE_EVERY_S:
        return
    last_save[0] = now
    tmp = STATE.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'tokens': tokens,
                'steps': steps, 'row_cursor': row_cursor, 'checkpoints': checkpoints,
                'train_fingerprint': train_fingerprint,
                'traj': traj, 'elapsed': elapsed_prior + time.perf_counter() - t0}, tmp)
    os.replace(tmp, STATE)          # atomic: a crash mid-save cannot corrupt the resume point


last_samples = checkpoint(0, 0) if not checkpoints else None
done_milestones = {c.get('milestone') for c in checkpoints}
model.train()
opt.zero_grad(set_to_none=True)
micro = 0

while tokens < BUDGET:
    if row_cursor >= len(train):
        raise SystemExit('TRAIN exhausted before the budget; this must not happen -- the '
                         'availability assertion above should have caught it')
    ids = train[row_cursor].to(DEVICE)
    row_cursor += 1
    logits = model(ids).logits
    tgt = ids[:, 1:]
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
    (loss / ACCUM).backward()
    tokens += tgt.numel()
    micro += 1
    if micro == ACCUM:
        for g in opt.param_groups:
            g['lr'] = lr_at(tokens)
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        steps += 1; micro = 0
        traj.append({'step': steps, 'tokens': tokens, 'train_ce': float(loss.detach()),
                     'grad_norm': float(gnorm), 'lr': lr_at(tokens)})
    # Periodic crash-recovery save, independent of milestones. Without this the time gate in
    # save_state() can never fire, because milestones are hours apart late in a run.
    save_state()
    for m in MILESTONES:
        if m and tokens >= m and m not in done_milestones:
            done_milestones.add(m)
            last_samples = checkpoint(tokens, m)
            save_state()
            break

if micro:
    # The budget rarely lands on an ACCUM boundary. Apply the partial window
    # rather than discarding tokens that were already counted as spent.
    for g in opt.param_groups:
        g['lr'] = lr_at(tokens)
    gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step(); opt.zero_grad(set_to_none=True); steps += 1
    traj.append({'step': steps, 'tokens': tokens, 'train_ce': float(loss.detach()),
                 'grad_norm': float(gnorm), 'lr': lr_at(tokens), 'partial_window': micro})
    micro = 0

completed = tokens >= BUDGET
if completed and (not checkpoints or checkpoints[-1]['tokens'] < tokens):
    last_samples = checkpoint(tokens)
save_state(force=True)

(OUT / f'LONGBUDGET_{TAG}.json').write_text(json.dumps({
    'schema': 1, 'evidence_kind': 'exploratory_large_budget_matched_comparison_only',
    'note': 'Large-budget matched comparison. Stops on a TOKEN budget so the registered '
            '1,000,000-token ceiling is exact. One epoch, no row revisited, so a null result '
            'cannot be attributed to cycling a small set. Open splits only; DEV and TASK are '
            'bit-identical to outputs/splits. No receipt, no gate, no sealed data, no efficacy '
            'claim.',
    'completed': completed,
    'completion_note': ('budget fully spent' if completed else
                        'INTERRUPTED -- this is NOT a complete run and must not be reported as '
                        'one; resume by re-invoking with identical arguments'),
    'arm': ARM, 'seed': SEED, 'device': DEVICE, 'dtype': DTYPE,
    'comparability': 'Matched WITHIN this run (same recipe, splits, seeds, device, dtype for '
                     'every arm). NOT directly comparable to the 40-step CPU FP32 runs unless '
                     'device and dtype match them.',
    'token_budget': BUDGET, 'registered_ceiling': CEILING,
    'supervised_target_tokens': tokens, 'optimizer_steps': steps,
    'optimizer': f'AdamW lr={LR:g} wd=0 clip=1.0 accum={ACCUM} micro_batch=1',
    'schedule': f'linear warmup {WARMUP_FRAC:.0%} then cosine to {MIN_LR_FRAC:.0%} of peak, '
                f'driven by token count so a resume lands on the same schedule',
    'train_rows_available': len(train), 'train_rows_consumed': row_cursor,
    'train_target_tokens_available': available,
    'epochs': tokens / available,
    # Justified, not asserted: the cursor is strictly monotonic within a run, and a resume
    # refuses unless train.jsonl is byte-identical to the interrupted run's.
    'rows_revisited': 0,
    'rows_revisited_basis': 'monotonic row cursor + train.jsonl sha256 verified on every resume',
    'train_fingerprint': train_fingerprint,
    'dev_rows': len(dev), 'dev_target_tokens': NTOK,
    'dev_response_target_tokens': NRESP,
    'dev_response_fraction': NRESP / NTOK,
    'metric_note': 'Only ~10% of a row is the assistant reply; the rest is system '
                   'prompt, user turn and chat-template boilerplate. dev_ce_nats is '
                   'over ALL target positions and is therefore dominated by '
                   'boilerplate. dev_response_ce_nats restricts to the reply. Both '
                   'are reported because neither alone has been shown to track '
                   'capability in this programme.',
    'task_rows': len(task_rows),
    'source_dev_ce_nats': ce_src,
    'trainable_parameters': sum(p.numel() for p in trainable),
    'wall_seconds': elapsed_prior + time.perf_counter() - t0,
    'milestones': MILESTONES, 'checkpoints': checkpoints, 'trajectory': traj,
    'final_samples': last_samples,
    'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}, indent=2) + '\n', encoding='utf-8')

first, last = checkpoints[0], checkpoints[-1]
print(f"\n{TAG}: {'COMPLETE' if completed else 'INTERRUPTED'} | "
      f"dev CE {first['dev_ce_nats']:.4f} -> {last['dev_ce_nats']:.4f} | "
      f"task {first['task_numeric_hits']}/{first['task_n']} -> "
      f"{last['task_numeric_hits']}/{last['task_n']} | {tokens} tokens")
print('wrote', OUT / f'LONGBUDGET_{TAG}.json')
