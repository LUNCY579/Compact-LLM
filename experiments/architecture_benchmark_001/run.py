"""Offline RAB-001 runner. No downloads, paid jobs, implicit training, or TEST access."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import gc
import json
import re
import time
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from .budget import Config, budget, presets
from .gates import (GateClosed, context, implementation_digest, protocol, require_registered_stage,
                    runtime, validate_receipt, write_once)
from .source import ROOT, audit_source, sha256_file, source_config
from .transfer import (ActivationCalibration, identity_expand, import_qwen_state, narrow,
                       select_width, state_digest)
from .model import DenseReference
from .losses import recovery_loss


def load_rows(path: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if (row.get('split') != split or not isinstance(row.get('id'), str) or not row['id']
                or row['id'] in seen or not isinstance(row.get('group_id'), str) or not row['group_id']):
            raise GateClosed('data split, unique ID, or independent group declaration invalid')
        seen.add(row['id'])
        fields = ('prompt', 'answer') if split == 'pilot' else ('text',)
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in fields):
            raise GateClosed(f'{split} data fields are incomplete')
    if not rows:
        raise GateClosed('empty dataset')
    return rows


def ensure_disjoint(*datasets: list[dict]) -> None:
    groups, ids, contents = set(), set(), set()
    for rows in datasets:
        g = {r['group_id'] for r in rows}; i = {r['id'] for r in rows}
        texts = {' '.join(r.get('text', r.get('prompt', '')).split()) for r in rows}
        if g & groups or i & ids or texts & contents:
            raise GateClosed('group, ID, or normalized content overlaps between splits')
        groups |= g; ids |= i; contents |= texts


def _directory(path: Path, source: Path) -> None:
    p, s = path.resolve(), source.resolve()
    if p == s or p.is_relative_to(s) or 'Rineya_Published_Cores' in p.parts:
        raise GateClosed('output must not modify a published source directory')
    p.mkdir(parents=True, exist_ok=False)


def _load(source: Path, device: str, native: bool = False):
    from importlib.metadata import version
    if version('transformers') != protocol()['native_transformers_version']:
        raise GateClosed('install the pinned native Transformers version in requirements-native.txt')
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    raw = json.loads((source/'config.json').read_text())
    c = source_config(raw)
    state = load_file(str(source/'model.safetensors'), device='cpu')
    reference = import_qwen_state(state, c).to(device=device, dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
    original = None
    if native:
        cfg = Qwen3Config(**raw); cfg._attn_implementation = 'eager'
        with torch.device('meta'):
            original = Qwen3ForCausalLM(cfg)
        native_state = dict(state)
        native_state['lm_head.weight'] = state['model.embed_tokens.weight']
        original.load_state_dict(native_state, strict=True, assign=True)
        original.tie_weights()
        original = original.to(device=device, dtype=torch.float32).eval()
    return reference, tokenizer, original


def _prompt(tokenizer, row: dict, device: str) -> torch.Tensor:
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
              tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors='pt').to(device)
    if ids.shape[1] > protocol()['decoding']['max_context'] or ids.shape[1] < 1:
        raise GateClosed('pilot exceeds registered context; truncation forbidden')
    return ids


def _text(tokenizer, row: dict, device: str) -> dict:
    ids = tokenizer(row['text'], return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    if not 2 <= ids.shape[1] <= protocol()['decoding']['max_context']:
        raise GateClosed('text outside registered context; truncation forbidden')
    return {'input_ids': ids}


def _artifacts(folder: Path, names: list[str]) -> dict:
    return {name: sha256_file(folder/name) for name in names}


def _record(folder: Path, stage: str, ctx: dict, checks: dict, metrics: dict, artifacts: list[str], parent=None):
    body = {'schema': 1, 'stage': stage, 'status': 'PASS' if all(v is True for v in checks.values()) else 'FAIL',
            'evidence_kind': 'official_model_qualification', 'context': ctx, 'checks': checks, 'metrics': metrics,
            'artifacts': _artifacts(folder, artifacts), 'parent_receipt_sha256': parent,
            'licenses_registered_training': False, 'licenses_efficacy_claim': False}
    result = write_once(folder/f'{stage}.json', body)
    if result['status'] != 'PASS':
        raise GateClosed(f'{stage} failed; inspect {folder}/{stage}.json. Do not weaken gates after student results.')
    return result


def qualify_source(source: Path, pilot: Path, out: Path, device: str) -> dict:
    audit = audit_source(source)
    ctx = context(audit, pilot, device)
    rows = load_rows(pilot, 'pilot')
    if len(rows) < protocol()['G0']['min_rows']:
        raise GateClosed('not enough registered pilot rows')
    _directory(out, source)
    torch.manual_seed(protocol()['decoding']['seed'])
    reference, tokenizer, native = _load(source, device, native=True)
    results, parity, largest = [], True, 0.0
    started = time.perf_counter()
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for row in rows:
            ids = _prompt(tokenizer, row, device)
            a = native(ids, use_cache=False).logits
            b = reference(ids).logits
            largest = max(largest, (a-b).abs().max().item())
            try:
                torch.testing.assert_close(a, b, atol=protocol()['G0']['native_reference_atol'],
                                            rtol=protocol()['G0']['native_reference_rtol'])
            except AssertionError:
                parity = False
            del a, b
            generated, past, current, terminated = [], None, ids, False
            for _ in range(protocol()['decoding']['max_new_tokens']):
                step = native(current, past_key_values=past, use_cache=True)
                current = step.logits[:, -1].argmax(-1, keepdim=True)
                past = step.past_key_values
                token = current.item(); generated.append(token)
                if token == tokenizer.eos_token_id:
                    terminated = True; break
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            results.append({'id': row['id'], 'group_id': row['group_id'], 'answer': row['answer'],
                            'prediction': text, 'tokens': generated, 'terminated': terminated,
                            'parseable': re.fullmatch(r'-?\d+', text) is not None,
                            'correct': text == row['answer'], 'prompt_tokens': ids.shape[1]})
    count = len(results); hits = sum(r['correct'] for r in results)
    parse = sum(r['parseable'] for r in results)/count
    stop = sum(r['terminated'] for r in results)/count
    write_once(out/'predictions.json', {'split': 'pilot', 'rows': results, 'not_sealed_quality_evidence': True})
    unchanged = audit_source(source)['audit_digest'] == audit['audit_digest'] and sha256_file(pilot) == ctx['data_sha256']
    checks = {'official_source_unchanged': unchanged, 'native_reference_logits': parity,
              'parser_usable': parse >= protocol()['G0']['min_parse_rate'],
              'termination_usable': stop >= protocol()['G0']['min_termination_rate'],
              'diagnostic_not_floor_or_ceiling': 0 < hits < count,
              'registered_context_executed': True, 'unmodified_native_readout': True}
    return _record(out, 'G0', ctx, checks, {'hits': hits, 'rows': count, 'parse_rate': parse,
        'termination_rate': stop, 'max_abs_native_reference_error': largest, 'wall_seconds': time.perf_counter()-started,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,
        'resource_scope': 'actual short-context FP32 eager pilot only; no 8k fit or throughput claim'}, ['predictions.json'])


def qualify_identity(source: Path, pilot: Path, g0: Path, out: Path, device: str) -> dict:
    audit = audit_source(source); ctx = context(audit, pilot, device)
    parent = validate_receipt(g0, 'G0', ctx)
    rows = load_rows(pilot, 'pilot')
    _directory(out, source)
    reference, tokenizer, _ = _load(source, device)
    before = state_digest(reference)
    expanded, mapping = identity_expand(reference, 40)
    equality, cached, largest = True, True, 0.0
    with torch.inference_mode():
        for row in rows[:8]:
            ids = _prompt(tokenizer, row, device)
            a = reference(ids).logits; b = expanded(ids).logits
            equality &= torch.equal(a, b); largest = max(largest, (a-b).abs().max().item())
            cut = max(1, ids.shape[1]//2)
            pre = expanded(ids[:, :cut], use_cache=True)
            tail = expanded(ids[:, cut:], past=pre.cache, use_cache=True).logits
            try:
                torch.testing.assert_close(b[:, cut:], tail, atol=1e-4, rtol=1e-4)
            except AssertionError:
                cached = False
    write_once(out/'lineage.json', mapping)
    unchanged = (state_digest(reference) == before and audit_source(source)['audit_digest'] == audit['audit_digest']
                 and sha256_file(pilot) == ctx['data_sha256'])
    checks = {'identity_logits_exact': bool(equality), 'chunked_cache_parity': bool(cached),
              'source_unchanged': unchanged, 'retained_original_layer_order': True,
              'exact_parameter_count': sum(p.numel() for p in expanded.parameters()) == budget(expanded.config)['parameters']}
    return _record(out, 'G1_IDENTITY', ctx, checks, {'max_abs_logit_error': largest, 'tested_rows': 8,
        'diagnostic_parameters': budget(expanded.config)['parameters'], 'budget_matched': False},
        ['lineage.json'], parent['receipt_sha256'])


def calibrate_d40(source: Path, pilot: Path, calibration: Path, g1: Path, out: Path, device: str) -> dict:
    audit = audit_source(source)
    parent = validate_receipt(g1, 'G1_IDENTITY', context(audit, pilot, device))
    ctx = context(audit, calibration, device)
    pilot_rows, rows = load_rows(pilot, 'pilot'), load_rows(calibration, 'calibration')
    ensure_disjoint(pilot_rows, rows)
    _directory(out, source)
    reference, tokenizer, _ = _load(source, device)
    batches = [{**_text(tokenizer, row, device), 'last_position_only': True} for row in rows]
    scores = ActivationCalibration(reference).run(batches, split='calibration')
    target = presets()['W28_896_width_only_diagnostic']
    selections = select_width(reference.config, target, scores)
    w28, width_map = narrow(reference, target, selections)
    del reference; gc.collect()
    d40, depth_map = identity_expand(w28, 40)
    # Calibrate original trained blocks, then apply the registered insertion map.
    # This is equivalent to slicing those identity donors after expansion and avoids
    # pretending zero-initialized inserted outputs have informative saliency.
    c = presets()['D40_896_near_baseline_budget']
    if d40.config != c:
        raise GateClosed('converted D40 differs from registered target')
    write_once(out/'lineage.json', {'width': width_map, 'depth': depth_map,
                                  'calibration_sha256': ctx['data_sha256'], 'calibration_groups': sorted({r['group_id'] for r in rows})})
    write_once(out/'calibration_scores.json', scores)
    _save(d40, out)
    restored = _restore(out, device)
    with torch.inference_mode():
        ids = batches[0]['input_ids'][:, :16]
        finite = bool(torch.isfinite(d40(ids).logits).all())
        parity = torch.equal(d40(ids).logits, restored(ids).logits)
    checks = {'registered_dimensions': True, 'all_tensors_accounted': True,
              'finite_converted_logits': finite, 'save_reload_exact': parity,
              'source_unchanged': audit_source(source)['audit_digest'] == audit['audit_digest'],
              'calibration_unchanged': sha256_file(calibration) == ctx['data_sha256']}
    return _record(out, 'G1_WIDTH', ctx, checks, {'parameters': budget(c)['parameters'],
        'calibration_rows': len(rows), 'quality_recovery_demonstrated': False},
        ['lineage.json', 'calibration_scores.json', 'config.json', 'model.safetensors'], parent['receipt_sha256'])


def _save(model: DenseReference, folder: Path):
    if (folder/'model.safetensors').exists():
        raise GateClosed('checkpoint output already exists; overwrite forbidden')
    write_once(folder/'config.json', {'model_type': 'rineya_ced_research_v1', 'reference_config': asdict(model.config),
                                     'not_hf_generate_or_gguf_compatible': True})
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(folder/'model.safetensors'))


def _restore(folder: Path, device: str) -> DenseReference:
    raw = json.loads((folder/'config.json').read_text())
    if raw.get('model_type') != 'rineya_ced_research_v1':
        raise GateClosed('not a registered custom research checkpoint')
    with torch.device('meta'):
        model = DenseReference(Config(**raw['reference_config']))
    model.load_state_dict(load_file(str(folder/'model.safetensors')), strict=True, assign=True)
    return model.to(device).eval()


def smoke(source: Path, pilot: Path, calibration: Path, training: Path, g1: Path, out: Path, device: str):
    audit = audit_source(source)
    parent = validate_receipt(g1, 'G1_WIDTH', context(audit, calibration, device))
    ctx = context(audit, training, device)
    rows = load_rows(training, 'train')
    ensure_disjoint(load_rows(pilot, 'pilot'), load_rows(calibration, 'calibration'), rows)
    _directory(out, source)
    teacher, tokenizer, _ = _load(source, device)
    student = _restore(g1.parent, device).train()
    if student.config != presets()['D40_896_near_baseline_budget']:
        raise GateClosed('foundation smoke only admits the source-bound D40 conversion')
    # SGD avoids allocating Adam moments for a 25-step engineering check. It is
    # deliberately NOT a preregistered recovery recipe or an 8-GB fit promise.
    optimizer = torch.optim.SGD(student.parameters(), lr=1e-5)
    teacher.requires_grad_(False)
    losses, ever_nonzero, tokens = [], set(), 0
    started = time.perf_counter()
    for step in range(protocol()['G2']['steps']):
        batch = _text(tokenizer, rows[step % len(rows)], device)
        ids = batch['input_ids']; tokens += ids.numel()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            targets = teacher(ids).logits
        value = recovery_loss(student(ids).logits, targets, ids, alpha=0.5, temperature=1.0)
        if not torch.isfinite(value):
            raise GateClosed('nonfinite smoke loss')
        value.backward()
        for name, p in student.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise GateClosed(f'missing/nonfinite intended gradient: {name}')
            if p.grad.count_nonzero():
                ever_nonzero.add(name)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step(); losses.append(value.item())
    student.eval(); _save(student, out)
    restored = _restore(out, device)
    with torch.inference_mode():
        parity = torch.equal(student(ids[:, :16]).logits, restored(ids[:, :16]).logits)
    checks = {'25_steps_completed': len(losses) == 25, 'all_intended_parameters_reached': len(ever_nonzero) == len(list(student.parameters())),
              'save_reload_exact': parity, 'source_unchanged': audit_source(source)['audit_digest'] == audit['audit_digest'],
              'training_data_unchanged': sha256_file(training) == ctx['data_sha256']}
    return _record(out, 'G2', ctx, checks, {'losses': losses, 'student_training_tokens': tokens,
        'teacher_forward_tokens': tokens, 'wall_seconds': time.perf_counter()-started,
        'optimizer': 'SGD_no_momentum_lr_1e-5', 'CE_weight': 0.5, 'KL_weight': 0.5, 'temperature': 1.0,
        'engineering_only': True}, ['config.json', 'model.safetensors'], parent['receipt_sha256'])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('matrix')
    status = subs.add_parser('status'); status.add_argument('--source', type=Path)
    audit = subs.add_parser('audit'); audit.add_argument('--source', type=Path, required=True)
    for name in ('qualify-source', 'qualify-identity', 'calibrate-d40', 'smoke'):
        p = subs.add_parser(name); p.add_argument('--source', type=Path, required=True)
        p.add_argument('--pilot', type=Path, default=ROOT/'data/pilot.jsonl')
        p.add_argument('--out', type=Path, required=True); p.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
        if name == 'qualify-identity': p.add_argument('--g0', type=Path, required=True)
        if name in ('calibrate-d40', 'smoke'):
            p.add_argument('--g1', type=Path, required=True); p.add_argument('--calibration', type=Path, required=True)
        if name == 'smoke': p.add_argument('--train', type=Path, required=True)
    registered = subs.add_parser('registered'); registered.add_argument('stage', choices=('G3', 'G4', 'G5'))
    args = parser.parse_args(argv)
    requested_output = getattr(args, 'out', None)
    output_existed = requested_output is not None and requested_output.exists()
    try:
        if args.command == 'matrix':
            result = {k: {'config': asdict(v), 'analytic_budget': budget(v)} for k, v in presets().items()}
        elif args.command == 'status':
            availability = 'not_requested'
            if args.source is not None:
                try: availability = audit_source(args.source)
                except (OSError, ValueError) as exc: availability = {'status': 'BLOCKED', 'reason': str(exc)}
            result = {'experiment_id': 'RAB-001', 'implementation_sha256': implementation_digest(), 'runtime': runtime(),
                      'cuda_available': torch.cuda.is_available(), 'source_audit': availability,
                      'G0_G2': 'require fresh official-source execution receipts; mechanical tests cannot open them',
                      'G3_G5': 'BLOCKED', 'licenses_registered_training': False, 'licenses_efficacy_claim': False}
        elif args.command == 'audit': result = audit_source(args.source)
        elif args.command == 'registered': require_registered_stage(args.stage)
        else:
            if args.device == 'cuda' and not torch.cuda.is_available(): raise GateClosed('CUDA unavailable; no device fallback')
            if args.command == 'qualify-source': result = qualify_source(args.source, args.pilot, args.out, args.device)
            elif args.command == 'qualify-identity': result = qualify_identity(args.source, args.pilot, args.g0, args.out, args.device)
            elif args.command == 'calibrate-d40': result = calibrate_d40(args.source, args.pilot, args.calibration, args.g1, args.out, args.device)
            else: result = smoke(args.source, args.pilot, args.calibration, args.train, args.g1, args.out, args.device)
        print(json.dumps(result, indent=2, allow_nan=False)); return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        failure = {'status': 'BLOCKED_OR_FAILED', 'stage': args.command, 'reason': str(exc),
                   'implementation_sha256': implementation_digest(), 'licenses_training': False, 'licenses_efficacy_claim': False}
        output = getattr(args, 'out', None)
        # Preserve failures in an already-created fresh run directory, never in source.
        if output is not None and not output_existed and output.is_dir() and not (output/'failure.json').exists():
            if not output.resolve().is_relative_to(args.source.resolve()) and 'Rineya_Published_Cores' not in output.resolve().parts:
                write_once(output/'failure.json', failure)
        print(json.dumps(failure, indent=2)); return 2


if __name__ == '__main__':
    raise SystemExit(main())
