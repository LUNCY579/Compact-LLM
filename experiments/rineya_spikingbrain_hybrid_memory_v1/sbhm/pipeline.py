"""Native staged runner: capture -> attention distill -> LM recovery -> evaluation.

All stages are individually authorized and receipt-bound. No implicit downloads,
protected evaluation, full-model training, checkpoint overwrite, or automatic
promotion. Sequential attention backend is for correctness, not speed claims.
"""
from pathlib import Path
import math
import time
import shutil
import torch
from safetensors.torch import save_file, load_file
from . import native
from .core import cache_bytes
from .data import load_corpus, ordered, encode
from .protocol import (Blocked, bind, authorize, require_qualification, resources,
                       read_json, write_json, sha, safe_file, digest, now)


def prepare(args):
    config = read_json(args.config)
    binding = bind(config, args.arm, args.delta)
    authorize(config, args.command, binding, args.allow_model_load, args.allow_training)
    if args.command != 'qualify':
        require_qualification(args.qualification, binding)
    if args.command == 'teacher' and (args.arm != 'A' or args.delta):
        raise Blocked('teacher capture must use the exact frozen baseline')
    if args.command in ('distill', 'recover') and args.arm == 'A':
        raise Blocked('arm A is never trained')
    if args.command != 'qualify':
        corpus(config)  # validate role/hash/disjointness BEFORE allocating a model
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'request.json', {'created_utc': now(), 'binding': binding, 'stage': args.command})
    model, tokenizer = native.load_host(config)
    if args.command != 'qualify':
        native.install(model, config, args.arm)
        native.load_delta(model, args.delta, binding)
    return config, binding, out, model, tokenizer


def corpus(config):
    return load_corpus(config.get('corpus_manifest'))


def tokens(config, tokenizer, row, training):
    ids, labels = encode(tokenizer, row, training)
    cap = config['limits']['max_train_tokens' if training else 'max_prompt_tokens']
    if ids.shape[1] > cap:
        raise Blocked(f'row {row["id"]} has {ids.shape[1]} tokens, exceeds {cap}; truncation forbidden')
    return ids.to(config['device']), labels.to(config['device'])


def capture(model, tokenizer, config, binding, out):
    data, data_hash = corpus(config)
    captured, entries, written_bytes = {}, [], 0
    consumed = list({r['id']: r for r in ordered(data['train'], config['seed'], config['limits']['steps'])}.values())
    def hook(_module, _args, kwargs, output):
        cos, sin = kwargs['position_embeddings']
        captured.update(hidden=kwargs['hidden_states'].detach().cpu().clone(),
                        cos=cos.detach().cpu().clone(), sin=sin.detach().cpu().clone(),
                        target=output[0].detach().cpu().clone())
    module = model.model.layers[config['layer']].self_attn
    handle = module.register_forward_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            for i, row in enumerate(consumed):
                ids, _ = tokens(config, tokenizer, row, True)
                model(input_ids=ids, use_cache=False, logits_to_keep=1)
                size = sum(t.numel()*t.element_size() for t in captured.values())
                if written_bytes + size > 512*2**20 or shutil.disk_usage(out).free < size + 2**30:
                    raise Blocked('teacher cache exceeds 512 MiB cap or 1 GiB disk reserve')
                written_bytes += size
                path = out/f'target-{i:06d}.safetensors'
                save_file({n: t.contiguous() for n, t in captured.items()}, str(path))
                entries.append({'id': row['id'], 'row_sha256': digest(row), 'path': path.name, 'sha256': sha(path)})
                captured.clear()
                resources(config['device'], runtime=True)
    finally:
        handle.remove()
    receipt = {'status': 'TEACHER_TARGETS_CAPTURED', 'binding': binding, 'corpus_sha256': data_hash,
               'layer': config['layer'], 'entries': entries, 'split': 'train', 'created_utc': now()}
    write_json(out/'teacher.json', receipt)
    return receipt


def training(model, tokenizer, config, binding, out, stage, teacher=None, parent_delta=None):
    data, data_hash = corpus(config)
    rows = ordered(data['train'], config['seed'], config['limits']['steps'])
    target_index = {}
    if stage == 'distill':
        if not teacher:
            raise Blocked('--teacher directory is required')
        manifest = read_json(Path(teacher)/'teacher.json')
        if manifest['corpus_sha256'] != data_hash or manifest['split'] != 'train':
            raise Blocked('teacher targets belong to another corpus/split')
        for key in ('source_sha256', 'code_sha256', 'config_sha256'):
            if manifest['binding'][key] != binding[key]:
                raise Blocked(f'teacher target {key} mismatch')
        if manifest['binding']['arm'] != 'A' or manifest['binding']['delta_sha256'] is not None:
            raise Blocked('teacher must be unadapted arm A')
        target_index = {item['id']: item for item in manifest['entries']}
        if len(target_index) != len(manifest['entries']) or set(target_index) != {r['id'] for r in rows}:
            raise Blocked('teacher inventory does not exactly cover consumed TRAIN rows')
    parameters = native.trainables(model)
    optimizer = torch.optim.AdamW(list(parameters.values()), lr=config['learning_rate'],
                                  betas=(0.9, 0.95), eps=1e-8, weight_decay=0, foreach=False)
    module = model.model.layers[config['layer']].self_attn
    records = []
    model.eval()  # deterministic, no dropout; autograd remains enabled
    for step, row in enumerate(rows):
        resources(config['device'], runtime=True)
        optimizer.zero_grad(set_to_none=True)
        if stage == 'distill':
            item = target_index[row['id']]
            path = safe_file(teacher, item['path'])
            if item['row_sha256'] != digest(row) or sha(path) != item['sha256']:
                raise Blocked('teacher row/activation hash changed')
            batch = {k: v.to(config['device']) for k, v in load_file(str(path)).items()}
            length = batch['hidden'].shape[1]
            if length > config['limits']['max_train_tokens']:
                raise Blocked('cached target exceeds registered training shape')
            prediction = module(hidden_states=batch['hidden'], position_embeddings=(batch['cos'], batch['sin']),
                                attention_mask=None, past_key_value=None)[0]
            target = batch['target'].float()
            loss = (prediction.float()-target).square().mean() / target.square().mean().clamp_min(1e-8)
            supervised = length  # positions, not vocabulary labels
        else:
            ids, labels = tokens(config, tokenizer, row, True)
            length = ids.shape[1]
            loss = model(input_ids=ids, labels=labels, use_cache=False).loss
            supervised = int((labels[:, 1:] != -100).sum())
        if not torch.isfinite(loss):
            raise Blocked('nonfinite loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters.values()):
            raise Blocked('nonfinite gradients')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise Blocked('frozen parameter received a gradient')
        gradient_norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters.values()):
            raise Blocked('nonfinite adaptation parameter')
        records.append({'step': step, 'row_id': row['id'], 'loss': loss.item(), 'input_tokens': length,
                        'supervised_positions': supervised, 'gradient_norm': float(gradient_norm)})
        resources(config['device'], runtime=True)
    receipt = {'status': 'TRAINING_COMPLETED_NOT_EFFICACY', 'stage': stage, 'created_utc': now(),
               'binding': binding, 'corpus_sha256': data_hash, 'records': records,
               'optimizer': 'AdamW(.9,.95), wd=0, eps=1e-8, clip=1', 'resume_supported': False,
               'parent_record': read_json(Path(parent_delta)/'delta.json')['record'] if parent_delta else None,
               'parameter_names': sorted(parameters), 'trainable_parameters': sum(p.numel() for p in parameters.values())}
    native.save_delta(model, out, binding, receipt)
    return receipt


def _sync():
    torch.cuda.synchronize()


@torch.no_grad()
def generate(model, ids, tokenizer, config, arm):
    """Instrumented loop timing, including guards; tokenization/loading excluded."""
    length = ids.shape[1]
    reset_at = config.get('reset_recurrent_at')
    if reset_at is not None and (arm != 'D' or not 0 < reset_at < length):
        raise Blocked('state-reset diagnostic requires arm D and an interior prompt boundary')
    cache, output, offset = None, None, 0
    torch.cuda.reset_peak_memory_stats()
    _sync()
    start = time.perf_counter()
    while offset < length:
        end = min(length, offset + config['limits']['prefill_chunk'])
        if reset_at is not None and offset < reset_at < end:
            end = reset_at
        output = native.model_forward(model, ids[:, offset:end], cache)
        cache = output.past_key_values
        if not torch.isfinite(output.logits).all():
            raise Blocked('nonfinite prefill logits')
        resources(config['device'], runtime=True)
        offset = end
        if offset == reset_at:
            cache._sbhm[config['layer']].reset_recurrent()
    first = int(output.logits[0, -1].argmax())
    _sync()
    first_time = time.perf_counter()
    generated, stop = [first], 'max_new_tokens'
    rss_max = resources(config['device'], runtime=True)['rss_bytes']
    for _ in range(1, config['limits']['max_new_tokens']):
        if generated[-1] == tokenizer.eos_token_id:
            stop = 'eos'
            break
        next_ids = ids.new_tensor([[generated[-1]]])
        output = native.model_forward(model, next_ids, cache)
        cache = output.past_key_values
        if not torch.isfinite(output.logits).all():
            raise Blocked('nonfinite decoding logits')
        generated.append(int(output.logits[0, -1].argmax()))
        rss_max = max(rss_max, resources(config['device'], runtime=True)['rss_bytes'])
    if generated[-1] == tokenizer.eos_token_id:
        stop = 'eos'
    _sync()
    end_time = time.perf_counter()
    native.check_cache(cache, config, arm, length + len(generated)-1)
    elapsed = end_time - first_time
    return {'prediction': tokenizer.decode(generated, skip_special_tokens=True), 'generated_ids': generated,
            'stop_reason': stop, 'prompt_tokens': length, 'generated_tokens': len(generated),
            'model_ttft_seconds': first_time-start, 'decode_seconds_excluding_first': elapsed,
            'decode_tokens_per_second': (len(generated)-1)/elapsed if len(generated)>1 and elapsed>0 else None,
            'cache_storage_bytes': cache_bytes(cache), 'peak_vram_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_vram_reserved_bytes': torch.cuda.max_memory_reserved(), 'rss_sample_max_bytes': rss_max,
            'reset_recurrent_at': reset_at, 'backend': 'native_sdpa' if arm in ('A','B') else 'mixed_sdpa_and_sequential_reference'}


def evaluate(model, tokenizer, config, binding, out, arm, delta=None):
    data, data_hash = corpus(config)
    # Fixed synthetic warmup; no evaluation examples used to tune warmup.
    with torch.no_grad():
        model(input_ids=torch.full((1, 8), 10, device=config['device'], dtype=torch.long),
              use_cache=False, logits_to_keep=1)
    predictions = []
    for row in data['dev']:
        ids, _ = tokens(config, tokenizer, row, False)
        result = generate(model, ids, tokenizer, config, arm)
        result.update(id=row['id'], family_id=row['family_id'], target=row['target'],
                      strict_exact_match=result['prediction'] == row['target'],
                      normalized_exact_match=result['prediction'].strip().casefold() == row['target'].strip().casefold())
        predictions.append(result)
        resources(config['device'], runtime=True)
    return {'status': 'DEV_MEASUREMENT_ONLY', 'binding': binding, 'created_utc': now(),
            'corpus_sha256': data_hash, 'predictions': predictions,
            'adaptation_record': read_json(Path(delta)/'delta.json')['record'] if delta else None,
            'strict_exact_match': sum(x['strict_exact_match'] for x in predictions)/len(predictions),
            'normalized_exact_match': sum(x['normalized_exact_match'] for x in predictions)/len(predictions),
            'semantic_accuracy': None, 'speedup_claim': False, 'energy_measured': False}


def compare(paths):
    """Paired repairs/damages relative to A; no automatic statistical promotion."""
    reports = {r['binding']['arm']: r for r in [read_json(p) for p in paths]}
    if len(paths) != 4 or set(reports) != set('ABCD'):
        raise Blocked('exactly one completed DEV report per arm A/B/C/D is required')
    base = reports['A']
    if base['binding']['delta_sha256'] is not None:
        raise Blocked('baseline A must remain unadapted')
    exposure = None
    result = {}
    for arm, report in reports.items():
        if report['status'] != 'DEV_MEASUREMENT_ONLY' or report['corpus_sha256'] != base['corpus_sha256']:
            raise Blocked('incompatible measurement/corpus')
        for key in ('source_sha256', 'code_sha256', 'config_sha256', 'environment'):
            if report['binding'][key] != base['binding'][key]:
                raise Blocked(f'unpaired {key}')
        if arm != 'A':
            recovery = report.get('adaptation_record')
            if not recovery or recovery.get('stage') != 'recover':
                raise Blocked('all trained controls require completed recovery receipts')
            distill = recovery.get('parent_record')
            if not distill or distill.get('stage') != 'distill':
                raise Blocked('all trained controls require the registered distillation stage')
            trace = [[(x['row_id'], x['input_tokens'], x['supervised_positions']) for x in stage['records']]
                     for stage in (distill, recovery)]
            if exposure is not None and trace != exposure:
                raise Blocked('training exposure/order/steps differ across controls')
            exposure = trace
        rows = report['predictions']
        if [r['id'] for r in rows] != [r['id'] for r in base['predictions']]:
            raise Blocked('row/order mismatch')
        if any(r['reset_recurrent_at'] is not None for r in rows):
            raise Blocked('state-reset diagnostic cannot be pooled with main comparison')
        result[arm] = {'repairs': sum(not b['strict_exact_match'] and r['strict_exact_match'] for b, r in zip(base['predictions'], rows)),
                       'damages': sum(b['strict_exact_match'] and not r['strict_exact_match'] for b, r in zip(base['predictions'], rows)),
                       'strict_exact_match': report['strict_exact_match']}
    return {'status': 'DESCRIPTIVE_PAIRED_DEV_ONLY', 'arms': result, 'promotion': False,
            'warning': 'Exposure is paired; trainable counts and runtime costs differ and must be reported.'}


def execute(args):
    config, binding, out, model, tokenizer = prepare(args)
    if args.command == 'qualify':
        receipt = native.qualify(model, config, args.arm, binding, args.delta)
    elif args.command == 'teacher':
        receipt = capture(model, tokenizer, config, binding, out)
    elif args.command in ('distill', 'recover'):
        receipt = training(model, tokenizer, config, binding, out, args.command, args.teacher, args.delta)
    else:
        receipt = evaluate(model, tokenizer, config, binding, out, args.arm, args.delta)
    write_json(out/'receipt.json', receipt)
    return receipt
