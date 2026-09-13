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
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
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

#: `H`/`P` train all 28 reads. `Hlast<N>`/`Plast<N>` train only the last N, freezing the rest at
#: their zero output projection so they stay exactly inert. H56's collapse at lr 1e-5 is
#: consistent with 28 branches waking at once into a backbone that is frozen and cannot
#: compensate; if that is the mechanism, N=4 should be far more stable than N=28 at the same lr,
#: and that is a cheap, decisive test of the mechanism rather than of the learning rate.
#: `B28last<N>` trains the last N of 28 transformer blocks and freezes the rest. The two
#: endpoints are measured -- 8 blocks gives 97.9% retention and 1 gain, all 28 gives 87.2% and
#: 32 -- and nothing between them has ever been run. That interval is the only place in this
#: programme where a real capability/retention trade-off has been observed, so N is a sweep
#: parameter rather than the hard-coded 8 it used to be.
_ARM_RE = re.compile(r'^(?P<mode>[HP])(?:last(?P<n>\d+))?$|^B28(?:last(?P<bn>\d+))?$')
ARM = sys.argv[1] if len(sys.argv) > 1 else 'H'
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5
BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 5200
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 17
#: Probe fractions of the budget. A CPU probe of a 1.156B arm costs ~379 s, so a 5-point curve
#: is most of the wall clock of an H56 run; pass "1.0" to buy a sweep point for two probes.
FRACTIONS = [float(x) for x in (sys.argv[5] if len(sys.argv) > 5 else '0.25,0.5,0.75,1.0').split(',')]
_M = _ARM_RE.match(ARM)
if not _M:
    raise SystemExit(__doc__)
MODE = _M.group('mode') or 'B28'
READ_TAIL = int(_M.group('n')) if _M.group('n') else None
#: For B28 arms this is the number of trainable transformer blocks, not read sites.
BLOCK_TAIL = int(_M.group('bn')) if _M.group('bn') else None
for tail, what in ((READ_TAIL, 'read tail'), (BLOCK_TAIL, 'block tail')):
    if tail is not None and not 1 <= tail <= 28:
        raise SystemExit(f'{what} must be between 1 and 28 blocks')
if not FRACTIONS or any(not 0 < f <= 1 for f in FRACTIONS) or 1.0 not in FRACTIONS:
    raise SystemExit('probe fractions must lie in (0,1] and include the endpoint 1.0')

#: Schedule spec: `cosine:<warmup>` or `wsd:<warmup>:<decay>`, fractions of the token budget.
#:
#: The inherited default is `cosine:0.03`, and at a 5,200-token budget that ends warmup at 156
#: tokens -- ONE optimizer step. That is very close to no warmup at all, and it is the worst
#: possible setting for a zero-initialised branch: Adam's first update is -lr*sign(g) for every
#: element regardless of gradient magnitude (at t=1, sqrt(v/bc2) = |g| and m/bc1 = g exactly),
#: so H56's 28 read projections each move by a full lr on step one, into a frozen backbone.
#:
#: WSD (Warmup-Stable-Decay) holds lr flat through the middle and decays late, which lets the
#: warmup be long without spending the budget climbing. The decay uses the 1-sqrt form, which
#: is the variant with the strongest published support and is flat early in the decay phase.
SCHEDULE = sys.argv[6] if len(sys.argv) > 6 else 'cosine:0.03'
#: beta2=0.999 has a ~1000-step horizon. A 41-step run never escapes bias correction, so the
#: second moment stays noisy and step sizes erratic. Short runs normally want ~0.95.
BETA2 = float(sys.argv[7]) if len(sys.argv) > 7 else 0.999
#: `final` | `none`. Every dose run before this one discarded its trained weights at process
#: exit -- nothing in this programme has ever saved a trained model, so the B28 run that reached
#: 73/150 no longer exists and neither does H56's. Weight averaging ("model soup") needs the
#: ingredients to exist, so the default is now to keep them.
#:
#: Only the TRAINABLE tensors are written. For H/P the backbone is frozen and bit-identical to
#: the source, so the trained delta is 119,571,072 parameters (0.48 GiB) rather than 4.31 GiB,
#: and the full model reconstructs exactly as source -> build arm -> overwrite these tensors.
SAVE = sys.argv[8] if len(sys.argv) > 8 else 'final'
if SAVE not in ('final', 'none'):
    raise SystemExit('save policy is `final` or `none`')
RUN_LABEL = sys.argv[9] if len(sys.argv) > 9 else None
if RUN_LABEL is not None and not re.fullmatch(r'[A-Za-z0-9-]+', RUN_LABEL):
    raise SystemExit('optional run label must contain only letters, digits and hyphens')
STORAGE_BACKEND = os.environ.get('RAB001_ADAM_STORAGE', 'memmap')
if STORAGE_BACKEND not in ('memmap', 'streamed'):
    raise SystemExit('RAB001_ADAM_STORAGE must be memmap or streamed')
GPU_WORKSPACE_GIB = float(os.environ.get('RAB001_GPU_WORKSPACE_GIB', '0.6'))
if not 0 < GPU_WORKSPACE_GIB < 8:
    raise SystemExit('RAB001_GPU_WORKSPACE_GIB must lie strictly between 0 and 8')
MIN_LR_FRAC = 0.10

_S = SCHEDULE.split(':')
SCHED_KIND = _S[0]
WARMUP_FRAC = float(_S[1]) if len(_S) > 1 else 0.03
DECAY_FRAC = float(_S[2]) if len(_S) > 2 else 0.20
if SCHED_KIND not in ('cosine', 'wsd'):
    raise SystemExit('schedule must be cosine:<warmup> or wsd:<warmup>:<decay>')
if not 0 <= WARMUP_FRAC < 1 or not 0 < DECAY_FRAC < 1 or WARMUP_FRAC + DECAY_FRAC >= 1:
    raise SystemExit('warmup and decay fractions must be in [0,1) and leave a stable phase')

TAG = (f'{ARM}_lr{LR:g}_b{BUDGET}_seed{SEED}'
       + (f'_{SCHED_KIND}{WARMUP_FRAC:g}' + (f'-{DECAY_FRAC:g}' if SCHED_KIND == 'wsd' else '')
          if SCHEDULE != 'cosine:0.03' else '')
       + (f'_b2{BETA2:g}' if BETA2 != 0.999 else ''))
CANONICAL_TAG = TAG
if RUN_LABEL:
    TAG += '_' + RUN_LABEL

# OffloadedAdam and force_tf32_off are proven in retention_dose_gpu; import rather than fork.
# That module reads its own constants from argv at import time, so hand it the matching ones.
_saved_argv = sys.argv
sys.argv = ['retention_dose_gpu', str(LR), str(BUDGET), str(SEED)]
try:
    from experiments.rab001_exploration import retention_dose_gpu as _rdg            # noqa: E402
    from experiments.rab001_exploration.retention_dose_gpu import (OffloadedAdam,   # noqa: E402
                                                                   force_tf32_off)
finally:
    sys.argv = _saved_argv
# OffloadedAdam reads its betas from that module's globals. Overriding here keeps the proven
# optimizer and makes beta2 a swept variable; the value used is recorded in every artifact.
_rdg.B2 = BETA2

SRC = REPO / 'Supa'
if not (SRC / 'model.safetensors').is_file():
    SRC = Path('C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/'
               'Rineya-Supa-1.7.0-devfreeze')
SPL = ROOT / 'outputs' / 'splits_large'
OUT = ROOT / 'evidence' / 'retention-design-v1'
N_DEV = 40
RUN_DIR = None


def file_sha256(path: Path) -> str:
    """Bound hashing memory even for a full 2.22 GiB trained checkpoint."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_record(path: Path, record: dict) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write('\n')


def lr_at(done: int) -> float:
    import math
    frac = min(1.0, done / BUDGET)
    if WARMUP_FRAC and frac < WARMUP_FRAC:
        return LR * frac / WARMUP_FRAC
    if SCHED_KIND == 'cosine':
        p = (frac - WARMUP_FRAC) / max(1e-9, 1 - WARMUP_FRAC)
        return LR * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * p)))
    start = 1 - DECAY_FRAC
    if frac < start:                                   # stable phase: hold lr flat
        return LR
    p = (frac - start) / DECAY_FRAC                    # decay phase: 1 - sqrt to the floor
    return LR * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * max(0.0, 1 - math.sqrt(p)))


def build(device: str):
    """Returns (model, trainable, description). H/P peak at source+result, so build on the GPU."""
    if MODE == 'B28':
        model = run._load(SRC, device)[0]
        if BLOCK_TAIL is None:
            trainable = [p for p in model.parameters()]
            for p in trainable:
                p.requires_grad_(True)
            return model, trainable, 'B28 source, all parameters'
        index = lambda n: int(m.group(2)) if (m := re.match(r'^(.*?)\.(\d+)\.', n)) else -1
        top = max(index(n) for n, _ in model.named_parameters())
        first = top - BLOCK_TAIL + 1
        for name, p in model.named_parameters():
            p.requires_grad_(index(name) >= first)
        trainable = [p for p in model.parameters() if p.requires_grad]
        return (model, trainable,
                f'B28 source, blocks {first}..{top} trainable ({BLOCK_TAIL} of {top+1}); '
                'embeddings, final norm and all earlier blocks frozen')
    from experiments.dual28_ced_v1.model import DualStack
    src = run._load(SRC, device)[0]
    model = DualStack.from_source(src, MODE)
    del src
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    model.gradient_checkpointing = False
    trainable = model.freeze_backbones()
    if READ_TAIL is None:
        return model, trainable, f'DualStack.from_source(mode={MODE}), all 28 reads + memory'
    # Freeze the early reads. Their o_proj stays exactly zero, so those blocks contribute
    # nothing for the whole run -- the arm is genuinely "reads on the last N blocks only",
    # not "all 28 reads with a small early contribution".
    depth = len(model.reads)
    keep = depth - READ_TAIL
    frozen_reads = 0
    for name, p in model.named_parameters():
        m = re.match(r'^reads\.(\d+)\.', name)
        if m and int(m.group(1)) < keep:
            p.requires_grad_(False)
            frozen_reads += 1
    trainable = [p for p in model.parameters() if p.requires_grad]
    for i, block in enumerate(model.reads):
        if i < keep and bool((block.o_proj.weight != 0).any()):
            raise SystemExit(f'frozen read {i} is not inert at build time')
    return (model, trainable,
            f'DualStack.from_source(mode={MODE}), reads on blocks {keep}..{depth-1} only '
            f'({READ_TAIL} of {depth}) + memory; {frozen_reads} early read tensors frozen inert')


def main() -> int:
    global RUN_DIR
    if not torch.cuda.is_available():
        raise SystemExit('this harness trains on the GPU; CUDA is required')
    numerics = force_tf32_off()
    # Host peak is the probe: one fp32 copy of the arm on the CPU. H56 is 4.31 GiB, B28 2.22.
    full_memmap = MODE == 'B28' and BLOCK_TAIL is None and STORAGE_BACKEND == 'memmap'
    host_start = hostbudget.require(6.0 if MODE in ('H', 'P') or full_memmap else 3.6,
                                    f'arm dose {TAG}')
    result_path = OUT / f'ARM_DOSE_{TAG}.json'
    destination = ROOT / 'outputs' / 'arm-checkpoints' / TAG
    if result_path.exists() or destination.exists():
        raise SystemExit(f'create-only result/checkpoint already exists for {TAG}')
    RUN_DIR = ROOT / 'outputs' / 'arm-runs' / (
        TAG + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    harness_digest = file_sha256(Path(__file__))
    optimizer_digest = file_sha256(Path(_rdg.__file__))
    storage_path = Path(__file__).with_name('streamed_adam.py') if STORAGE_BACKEND == 'streamed' else None
    storage_digest = file_sha256(storage_path) if storage_path else None
    serialization_path = Path(__file__).with_name('tensor_stream.py')
    serialization_digest = file_sha256(serialization_path)
    digest = implementation_digest()
    source_assets = json.loads((ROOT / 'source_lock.json').read_text(encoding='utf-8'))['assets']
    for name, expected in source_assets.items():
        if file_sha256(SRC / name) != expected:
            raise SystemExit(f'source asset mismatch: {name}')
    data_paths = {'train': SPL / 'train.jsonl', 'dev': SPL / 'dev.jsonl',
                  'monitor_probe': ROOT / 'data' / 'monitor_probe.jsonl',
                  'probe_label_audit': OUT / 'PROBE_LABEL_AUDIT.json'}
    data_hashes = {name: file_sha256(path) for name, path in data_paths.items()}

    def verify_run_inputs():
        if implementation_digest() != digest or file_sha256(Path(__file__)) != harness_digest:
            raise RuntimeError('implementation or active harness changed during run')
        if file_sha256(Path(_rdg.__file__)) != optimizer_digest:
            raise RuntimeError('optimizer harness changed during run')
        if storage_path and file_sha256(storage_path) != storage_digest:
            raise RuntimeError('optimizer storage backend changed during run')
        if file_sha256(serialization_path) != serialization_digest:
            raise RuntimeError('checkpoint serialization helper changed during run')
        if any(file_sha256(path) != data_hashes[name] for name, path in data_paths.items()):
            raise RuntimeError('training/probe data changed during run')

    write_record(RUN_DIR / 'START.json', {
        'pid': os.getpid(), 'argv': sys.argv, 'tag': TAG, 'completed': False,
        'canonical_tag': CANONICAL_TAG, 'run_label': RUN_LABEL,
        'optimizer_storage_backend': STORAGE_BACKEND,
        'optimizer_storage_harness_sha256': storage_digest,
        'gpu_workspace_reserve_gib': GPU_WORKSPACE_GIB,
        'checkpoint_serialization_sha256': serialization_digest,
        'implementation_digest': digest, 'exploration_harness_sha256': harness_digest,
        'optimizer_harness_sha256': optimizer_digest, 'data_sha256': data_hashes,
        'source_assets_sha256': source_assets, 'host_before': host_start})
    free_v, total_v = torch.cuda.mem_get_info()
    print(f'{hostbudget.describe()} | VRAM free {free_v/2**30:.2f}/{total_v/2**30:.2f} GiB')
    print(f'TF32 forced off: {numerics}', flush=True)

    policy = protocol()['G1_RETENTION']
    short = digest[:8]
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
    train, train_ids = [], []
    for row in train_rows:
        encoded = enc(row['text'])
        if encoded.shape[1] >= 2:
            train.append(encoded)
            train_ids.append(row['id'])
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
    reserve = params_bytes + int(GPU_WORKSPACE_GIB * 2 ** 30)
    gpu_budget = max(0, torch.cuda.mem_get_info()[0] - reserve)
    spill = RUN_DIR / 'adam-spill'
    optimizer_class = OffloadedAdam
    if STORAGE_BACKEND == 'streamed':
        from streamed_adam import StreamedAdam
        optimizer_class = StreamedAdam
    opt = optimizer_class(trainable, LR, gpu_budget, spill)
    print(f'Adam storage={STORAGE_BACKEND}, GPU workspace reserve={GPU_WORKSPACE_GIB:g} GiB', flush=True)
    print(f'Adam moments {opt.bytes()/2**30:.2f} GiB: {opt.gpu_bytes/2**30:.2f} on GPU, '
          f'{opt.disk_bytes()/2**30:.2f} on disk | VRAM free '
          f'{torch.cuda.mem_get_info()[0]/2**30:.2f} GiB', flush=True)

    resource_trace = []

    def record_resources(phase, step=0):
        record = {'phase': phase, 'step': step, 'host': hostbudget.snapshot(),
                  'cuda_allocated_gib': torch.cuda.memory_allocated() / 2**30,
                  'cuda_reserved_gib': torch.cuda.memory_reserved() / 2**30,
                  'cuda_peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
        resource_trace.append(record)
        with (RUN_DIR / 'resources.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')

    record_resources('optimizer_ready')

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

    # How loud is the added branch? o_proj starts at exactly zero, so its Frobenius norm is a
    # direct, probe-free readout of how far the read has woken up. A probe costs 379 s on a
    # 1.156B arm; this costs a GPU reduction, so an lr can be rejected in seconds rather than
    # in an hour of decoding. Adam's early update is ~lr*sign(g) regardless of gradient size,
    # so a zero-init tensor moves at a rate set by lr alone -- this is what that looks like.
    read_o_proj = [p for n, p in model.named_parameters()
                   if n.startswith('reads.') and n.endswith('o_proj.weight') and p.requires_grad]

    def branch_norm() -> float:
        if not read_o_proj:
            return 0.0
        with torch.inference_mode():
            return float(torch.sqrt(sum((p.detach().float() ** 2).sum() for p in read_o_proj)))

    milestones = sorted({0, *(int(BUDGET * f) for f in FRACTIONS)})
    checkpoints, tokens, steps, cursor, consumed = [], 0, 0, 0, []
    trace = []
    started = time.perf_counter()
    aborted = None
    reg = policy['bare_policy']

    def checkpoint(done: int, milestone):
        ce = dev_ce()
        p = probe()
        outputs = p.pop('_outputs')
        changed = sum(1 for o in outputs if o['text'] != source_raw[o['id']]['raw'])
        termination_changed = sum(1 for o in outputs
                                  if o['terminated'] != source_raw[o['id']]['terminated'])
        r = p[reg]
        rec = {'tokens': done, 'milestone': milestone, 'steps': steps, 'lr': lr_at(done),
               'dev_ce_nats': ce, 'answers_changed_vs_source': changed,
               'termination_changed_vs_source': termination_changed, 'retention': p,
               'wall_seconds': time.perf_counter() - started}
        if milestone == milestones[-1] or milestone == 0:
            rec['outputs'] = {o['id']: {'text': o['text'], 'terminated': o['terminated']}
                              for o in outputs}
        checkpoints.append(rec)
        write_record(RUN_DIR / f'probe_{milestone:07d}.json', rec)
        print(f"  {done:>7} tok | step {steps:>3} | devCE {ce:7.4f} | "
              f"kept {r['retained_total']:2d}/{r['reference_n']} lost {r['lost_total']:2d} "
              f"gain {r['gained_total']:2d} | correct {r['correct_total']:3d}/150 | "
              f"changed {changed:3d}/150 | {r['status']:12s} | "
              f"{time.perf_counter()-started:.0f}s", flush=True)
        return rec

    try:
        first = checkpoint(0, 0)
        if first['answers_changed_vs_source'] != 0 or first['termination_changed_vs_source'] != 0:
            raise SystemExit(
                f"device-integrity control FAILED: {first['answers_changed_vs_source']} of 150 "
                f"texts and {first['termination_changed_vs_source']} termination flags differ "
                'from the frozen CPU reference before any training. GPU->CPU '
                'transfer is not answer-preserving here and no retention number would mean '
                'anything.')
        print('  device-integrity control PASSED: 0/150 answers changed before training',
              flush=True)
        record_resources('initial_control_passed')

        model.train()
        opt.zero_grad()
        done_ms = {0}
        while tokens < BUDGET:
            if cursor >= len(train):
                raise SystemExit('TRAIN exhausted before the budget')
            ids = train[cursor].to('cuda')
            cursor += 1
            if steps < 3:
                record_resources('before_forward', steps + 1)
            logits = model(ids).logits
            tgt = ids[:, 1:]
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                   tgt.reshape(-1))
            loss.backward()
            if steps < 3:
                record_resources('after_backward', steps + 1)
            tokens += tgt.numel()
            consumed.append(int(tgt.numel()))
            del logits
            gnorm = torch.nn.utils.clip_grad_norm_(opt.params, 1.0)
            opt.step(lr_at(tokens))
            opt.zero_grad()
            steps += 1
            trace.append({'step': steps, 'tokens': tokens, 'lr': lr_at(tokens),
                          'train_ce': float(loss.detach()), 'grad_norm': float(gnorm),
                          'read_o_proj_norm': branch_norm()})
            del loss
            record_resources('after_optimizer_step', steps)
            if STORAGE_BACKEND == 'streamed':
                # Gradients and the graph are gone. Release unused CUDA blocks;
                # this changes allocation lifetime, not any tensor arithmetic.
                torch.cuda.empty_cache()
                record_resources('after_unused_cuda_cache_release', steps)
            hostbudget.guard(1.0, f'arm dose {TAG}')
            for m in milestones:
                if m and tokens >= m and m not in done_ms:
                    done_ms.add(m)
                    checkpoint(tokens, m)
                    break
    except LowMemoryAbort as exc:
        aborted = str(exc)
        print(f'ABORTED: {exc}', flush=True)

    completed = aborted is None and tokens >= BUDGET
    if not checkpoints:
        raise RuntimeError(aborted or 'no completed probe; see preserved run directory')
    if completed and (checkpoints[-1]['tokens'] != tokens
                      or checkpoints[-1]['milestone'] != milestones[-1]):
        checkpoint(tokens, milestones[-1])
    first, last = checkpoints[0], checkpoints[-1]
    # Optimization is over; moment storage and transfer buffers are unnecessary
    # during saving. Keep model parameters alive, release only optimizer state.
    del opt
    gc.collect()
    torch.cuda.empty_cache()
    record_resources('optimizer_released_before_save', steps)

    # Save trainables in bounded chunks. A prior full-B28 attempt completed its
    # final probe but could not allocate a second 2.22 GiB CPU parameter copy.
    # Parameter references below do not copy data; the writer transfers <=1 MiB.
    checkpoint_info = None
    if SAVE == 'final' and completed:
        from tensor_stream import write_parameters
        destination = ROOT / 'outputs' / 'arm-checkpoints' / TAG
        if destination.exists():
            raise SystemExit(f'checkpoint directory already exists (create-only): {destination}')
        hostbudget.guard(1.2, f'arm dose {TAG} streamed checkpoint save')
        tensors = {name: p.detach()
                   for name, p in model.named_parameters() if p.requires_grad}
        destination.mkdir(parents=True, exist_ok=False)
        weights = destination / 'trainable.safetensors'
        save_details = write_parameters(
            weights, tensors, guard=lambda: hostbudget.guard(1.2, f'arm dose {TAG} save chunk'))
        checkpoint_info = {
            'path': str(destination), 'tensors': len(tensors),
            'parameters': sum(t.numel() for t in tensors.values()),
            'bytes': weights.stat().st_size,
            'sha256': file_sha256(weights),
            'dtype': 'float32', 'saved_after_steps': steps, 'saved_after_tokens': tokens,
            'complete_model': MODE == 'B28' and BLOCK_TAIL is None,
            'source_weights_sha256': source_assets['model.safetensors'],
            'source_assets_sha256': source_assets,
            'implementation_digest': digest,
            'save_backend': 'bounded_fp32_parameter_chunks_v1',
            'save_max_chunk_elements': save_details['max_chunk_elements'],
            'checkpoint_serialization_sha256': serialization_digest,
            # Everything needed to rebuild without consulting this script.
            'reconstruct': {
                'source': str(SRC), 'mode': MODE, 'read_tail': READ_TAIL,
                'block_tail': BLOCK_TAIL,
                'builder': ('run._load(source)' if MODE.startswith('B28')
                            else f'DualStack.from_source(run._load(source)[0], {MODE!r})'),
                'then': 'load trainable.safetensors and copy_ each tensor by name'},
            # Two checkpoints may only be averaged if these agree.
            'soup_key': {'mode': MODE, 'read_tail': READ_TAIL, 'block_tail': BLOCK_TAIL,
                         'source_digest': frozen.get('source_state_sha256')
                         or digest,
                         'tensor_names_sha256': hashlib.sha256(
                             '\n'.join(sorted(tensors)).encode()).hexdigest()},
        }
        verify_run_inputs()
        (destination / 'MANIFEST.json').write_text(json.dumps({
            'schema': 1, 'evidence_kind': 'exploratory_trained_weights_only',
            'note': 'Trainable tensors only. Exploratory: this is NOT a released model, opens '
                    'no gate and carries no receipt. Averaging is only defined against another '
                    'checkpoint with an identical soup_key.',
            'tag': TAG, 'arm': ARM, 'lr': LR, 'schedule': SCHEDULE, 'beta2': BETA2,
            'canonical_tag': CANONICAL_TAG, 'run_label': RUN_LABEL,
            'optimizer_storage_backend': STORAGE_BACKEND,
            'optimizer_storage_harness_sha256': storage_digest,
            'gpu_workspace_reserve_gib': GPU_WORKSPACE_GIB,
            'seed': SEED, 'token_budget': BUDGET, 'completed': completed,
            'headline': last['retention'][reg] if checkpoints else None,
            **checkpoint_info,
            'tensor_shapes': {n: list(t.shape) for n, t in sorted(tensors.items())},
            'licenses_release': False,
        }, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        del tensors
        gc.collect()
        print(f"  saved {checkpoint_info['parameters']:,} trained parameters "
              f"({checkpoint_info['bytes']/2**30:.2f} GiB) -> {destination}", flush=True)

    fr, lr_ = first['retention'][reg], last['retention'][reg]
    body = {
        'schema': 1, 'evidence_kind': 'exploratory_arm_dose_response_only',
        'note': 'Matched-dose comparison across arms. Trains on the GPU, probes on the CPU (the '
                'frozen reference device). Per-row correctness is recorded so arms can be '
                'compared as matched pairs. Outside the hashed root: no receipt, no gate, no '
                'stage opened, no efficacy claim, no checkpoint released.',
        'arm': ARM, 'mode': MODE, 'read_tail': READ_TAIL, 'block_tail': BLOCK_TAIL,
        'canonical_tag': CANONICAL_TAG, 'run_label': RUN_LABEL,
        'arm_description': described, 'completed': completed,
        'aborted_reason': aborted, 'lr': LR, 'token_budget': BUDGET, 'seed': SEED,
        'probe_fractions': FRACTIONS, 'probe_milestones': milestones,
        'parameters': n_all, 'trainable_parameters': n_train,
        'train_device': 'cuda', 'probe_device': 'cpu', 'numerics': numerics,
        'optimizer': f'AdamW(offloaded) lr={LR:g} wd=0 clip=1.0 micro_batch=1 '
                     f'betas=(0.9,{BETA2:g}) eps=1e-8',
        'beta2': BETA2, 'schedule_spec': SCHEDULE, 'schedule_kind': SCHED_KIND,
        'warmup_fraction': WARMUP_FRAC, 'decay_fraction': DECAY_FRAC if SCHED_KIND == 'wsd' else None,
        'warmup_tokens': int(BUDGET * WARMUP_FRAC),
        'schedule': (f'linear warmup {WARMUP_FRAC:.0%} then cosine to {MIN_LR_FRAC:.0%}'
                     if SCHED_KIND == 'cosine' else
                     f'WSD: linear warmup {WARMUP_FRAC:.0%}, stable, '
                     f'1-sqrt decay over final {DECAY_FRAC:.0%} to {MIN_LR_FRAC:.0%}'),
        'supervised_target_tokens': tokens, 'optimizer_steps': steps,
        'train_rows_consumed': cursor, 'per_step_target_tokens': consumed,
        'train_row_ids_consumed': train_ids[:cursor],
        'train_row_ids_sha256': hashlib.sha256('\n'.join(train_ids[:cursor]).encode()).hexdigest(),
        'data_sha256': data_hashes, 'source_assets_sha256': source_assets,
        'source_weights_sha256': source_assets['model.safetensors'],
        'exploration_harness_sha256': harness_digest,
        'optimizer_harness_sha256': optimizer_digest, 'run_directory': str(RUN_DIR),
        'optimizer_storage_backend': STORAGE_BACKEND,
        'optimizer_storage_harness_sha256': storage_digest,
        'gpu_workspace_reserve_gib': GPU_WORKSPACE_GIB, 'resource_trace': resource_trace,
        'checkpoint_serialization_sha256': serialization_digest,
        'read_o_proj_tensors': len(read_o_proj), 'step_trace': trace,
        'checkpoint': checkpoint_info, 'save_policy': SAVE,
        'read_o_proj_norm_end': trace[-1]['read_o_proj_norm'] if trace else 0.0,
        'reference_receipt': str(receipt_path), 'reference_digest': digest,
        'reference_sizes': {k: len(v) for k, v in references.items()},
        'registered_bare_policy': reg,
        'device_integrity_control_passed': (first['answers_changed_vs_source'] == 0
                                          and first['termination_changed_vs_source'] == 0),
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
    verify_run_inputs()
    path = result_path
    write_record(path, body)
    h = body['headline']
    print(f"\n{TAG}: {'COMPLETE' if completed else 'INTERRUPTED'}")
    if not completed:
        print(f"  No completed endpoint: last measured probe at {last['tokens']} tokens; "
              f'training reached {tokens} tokens. Counts below describe that last probe.')
    print(f"  correct {h['correct_start']}/150 -> {h['correct_end']}/150 "
          f"(gained {h['gained_end']}, lost {h['lost_end']}, net {h['net_end']:+d})")
    print(f"  devCE   {h['dev_ce_start']:.4f} -> {h['dev_ce_end']:.4f} | "
          f"gate {h['status_start']} -> {h['status_end']}")
    print('  wrote', path)
    return 0 if completed else 1


if __name__ == '__main__':
    try:
        status = main()
    except BaseException as exc:
        if RUN_DIR is not None:
            write_record(RUN_DIR / 'failure.json', {
                'completed': False, 'type': type(exc).__name__, 'message': str(exc),
                'traceback': traceback.format_exc(), 'tag': TAG})
        raise
    raise SystemExit(status)
