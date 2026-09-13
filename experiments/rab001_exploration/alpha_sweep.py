"""Fixed-grid, descriptive source interpolation / P output-gate evaluation.

Run ONE tag per process, after other model jobs finish, using the GPU runtime::

    python experiments/rab001_exploration/alpha_sweep.py TAG \
        --kind b28-wise --out experiments/rab001_exploration/outputs/alpha-TAG-v1

For P/Plast<N>, require ``--kind p-output-gate``. This loads ALL saved trainables
and scales ONLY reads.*.o_proj.weight. It is not WiSE interpolation of all P
trainables: trained memory, query projections and norms remain fixed.

The grid is fixed at 0, .25, .5, .75, 1. Execution order is 0, 1, .25, .5, .75.
Both endpoints must reproduce every saved output text AND termination flag before
any intermediate is run. The panel reports the complete curve, selects no alpha,
and licenses no efficacy claim, gate, receipt, training or release.

Only one model is built, on CUDA, then moved to CPU for all probes. Weight updates
stream source and checkpoint tensors; no checkpoint-size float64 accumulator or
extra source model is retained. Imports are lazy so metadata/unit tests need no
model runtime. Failed/incomplete reports and completed alpha artifacts are kept.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
EXPLORATION = Path(__file__).resolve().parent
ROOT = REPO / 'experiments' / 'architecture_benchmark_001'
EVIDENCE = ROOT / 'evidence' / 'retention-design-v1'
CHECKPOINTS = ROOT / 'outputs' / 'arm-checkpoints'
OUTPUTS = EXPLORATION / 'outputs'
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
EXECUTION_ORDER = (0.0, 1.0, 0.25, 0.5, 0.75)
KINDS = ('b28-wise', 'p-output-gate')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path: Path, body: dict, *, exclusive: bool = False) -> None:
    encoded = json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        pending = path.with_suffix(path.suffix + '.pending')
        with pending.open('w', encoding='utf-8', newline='\n') as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        pending.replace(path)


def arm_identity(manifest: dict, kind: str) -> tuple[str, int | None]:
    """Handle legacy B28last8 mode and current B28 mode + arm tail explicitly."""
    arm = manifest['arm']
    match = re.fullmatch(r'(B28|P)(?:last([0-9]+))?', arm)
    if not match:
        raise ValueError(f'unsupported alpha arm: {arm!r}')
    mode, tail_text = match.groups()
    tail = int(tail_text) if tail_text is not None else None
    if tail is not None and not 1 <= tail <= 28:
        raise ValueError('arm tail must be within 1..28')
    expected_kind = 'b28-wise' if mode == 'B28' else 'p-output-gate'
    if kind != expected_kind:
        raise ValueError(f'{arm} requires --kind {expected_kind}')
    key = manifest['soup_key']
    if key['mode'] not in ({'B28', arm} if mode == 'B28' else {'P'}):
        raise ValueError('soup mode and arm disagree')
    if key.get('read_tail') != (tail if mode == 'P' else None):
        raise ValueError('soup read_tail and arm disagree')
    reconstruction = manifest['reconstruct']
    if reconstruction['mode'] != key['mode'] or reconstruction.get('read_tail') != key.get('read_tail'):
        raise ValueError('reconstruction and soup identities disagree')
    if 'block_tail' in reconstruction and reconstruction['block_tail'] != (tail if mode == 'B28' else None):
        raise ValueError('reconstruction block tail disagrees with arm')
    return mode, tail


def verify_source_assets(source: Path, lock_path: Path) -> dict:
    """Actual on-disk asset hashes, independent of the misleading legacy soup key."""
    lock = read_json(lock_path)
    actual = {}
    for name, expected in lock['assets'].items():
        asset = source / name
        if not asset.resolve().is_relative_to(source.resolve()) or not asset.is_file() or asset.is_symlink():
            raise ValueError(f'missing, escaping or symlinked source asset: {name}')
        actual[name] = sha256_file(asset)
        if actual[name] != expected:
            raise ValueError(f'source asset SHA-256 mismatch: {name}')
    if (source / 'model.safetensors').stat().st_size != lock['weight_bytes']:
        raise ValueError('source weight byte count mismatch')
    if (source / 'adapter_config.json').exists() or list(source.glob('*.py')):
        raise ValueError('custom source or adapter directory forbidden')
    return {'core_id': lock['core_id'], 'assets': actual,
            'source_lock_sha256': sha256_file(lock_path),
            'audit_digest': canonical_hash({'core_id': lock['core_id'], 'assets': actual})}


def source_key_kind(key: str, implementation: str, source_audit: dict, frozen: dict) -> str:
    if key == implementation:
        return 'legacy_implementation_digest_fallback_NOT_source_asset_digest'
    if key == source_audit['audit_digest']:
        return 'verified_source_asset_audit_digest'
    if key == source_audit['assets']['model.safetensors']:
        return 'verified_source_weight_file_sha256'
    # A state digest differs from the file hash; never relabel one as the other.
    if key and key == frozen.get('source_state_sha256'):
        return 'frozen_receipt_source_state_digest_actual_assets_verified_separately'
    raise ValueError('checkpoint source key matches neither current implementation nor verified source')


def endpoint_control(actual: dict, expected: dict, label: str) -> dict:
    """Do not infer generation identity from correctness or text alone."""
    if set(actual) != set(expected):
        raise ValueError(f'{label}: output row inventory mismatch')
    for group in (actual, expected):
        for row in group.values():
            if not isinstance(row.get('text'), str) or type(row.get('terminated')) is not bool:
                raise ValueError(f'{label}: missing/invalid text or termination flag')
    text_ids = sorted(i for i in actual if actual[i]['text'] != expected[i]['text'])
    termination_ids = sorted(i for i in actual
                             if actual[i]['terminated'] != expected[i]['terminated'])
    return {'label': label, 'passed': not (text_ids or termination_ids), 'rows': len(actual),
            'text_mismatch_ids': text_ids, 'termination_mismatch_ids': termination_ids}


def expected_trainable_names(model, mode: str, tail: int | None) -> set[str]:
    names = set(dict(model.named_parameters()))
    depth = len(model.encoder)
    if mode == 'B28':
        if tail is None:
            return names
        return {name for name in names
                if (m := re.match(r'^encoder\.(\d+)\.', name)) and int(m[1]) >= depth - tail}
    return {name for name in names if name.startswith('memory.') or
            ((m := re.match(r'^reads\.(\d+)\.', name)) and
             (tail is None or int(m[1]) >= depth - tail))}


def validate_tensor_inventory(model, checkpoint: Path, manifest: dict, mode: str,
                              tail: int | None) -> list[str]:
    from safetensors import safe_open
    by_name = dict(model.named_parameters())
    shapes = manifest['tensor_shapes']
    expected = expected_trainable_names(model, mode, tail)
    if set(shapes) != expected:
        raise ValueError('manifest tensor names do not exactly match the arm trainable inventory')
    names_hash = hashlib.sha256('\n'.join(sorted(expected)).encode()).hexdigest()
    if names_hash != manifest['soup_key']['tensor_names_sha256']:
        raise ValueError('tensor-name hash mismatch')
    parameters = 0
    with safe_open(str(checkpoint), framework='pt', device='cpu') as handle:
        if set(handle.keys()) != expected:
            raise ValueError('checkpoint tensor names do not exactly match the manifest')
        for name in sorted(expected):
            view = handle.get_slice(name)
            if view.get_shape() != shapes[name] or shapes[name] != list(by_name[name].shape):
                raise ValueError(f'tensor shape mismatch: {name}')
            if view.get_dtype() != 'F32':
                raise ValueError(f'checkpoint must contain float32 trainables: {name}')
            parameters += by_name[name].numel()
    if len(expected) != manifest['tensors'] or parameters != manifest['parameters']:
        raise ValueError('checkpoint tensor/parameter count mismatch')
    if mode == 'P' and not any(is_read_output(name) for name in expected):
        raise ValueError('P checkpoint has no trainable read output projections')
    return sorted(expected)


def is_read_output(name: str) -> bool:
    return re.fullmatch(r'reads\.\d+\.o_proj\.weight', name) is not None


def apply_alpha(model, checkpoint: Path, names: list[str], alpha: float, kind: str,
                *, source_weights: Path | None = None, source_names: dict | None = None,
                guard=None) -> None:
    """Reset from immutable files at every point, so evaluation order cannot compound alpha."""
    import torch
    from experiments.rab001_exploration.tensor_stream import MAX_ELEMENTS, TensorFile
    if alpha not in GRID or kind not in KINDS:
        raise ValueError('alpha/kind outside the fixed protocol')
    if kind == 'b28-wise' and (source_weights is None or source_names is None):
        raise ValueError('B28 interpolation requires source weights and native-name mapping')
    by_name = dict(model.named_parameters())
    with ExitStack() as stack, torch.no_grad():
        trained = stack.enter_context(TensorFile(checkpoint))
        source = (stack.enter_context(TensorFile(source_weights))
                  if kind == 'b28-wise' else None)
        for name in names:
            target = by_name[name]
            if target.device.type != 'cpu' or target.dtype != torch.float32 or not target.is_contiguous():
                raise ValueError('alpha updates require contiguous float32 CPU parameters')
            info = trained.header.get(name)
            if info is None or info['shape'] != list(target.shape) or info['dtype'] != 'F32':
                raise ValueError(f'invalid trained tensor: {name}')
            native_name = None
            if kind == 'b28-wise' and alpha != 1.0:
                native_name = source_names[name]
                info = source.header.get(native_name)
                if info is None or info['shape'] != list(target.shape):
                    raise ValueError(f'invalid source tensor: {name}')
            flat = target.view(-1)
            for start in range(0, target.numel(), MAX_ELEMENTS):
                if guard is not None:
                    guard()
                count = min(MAX_ELEMENTS, target.numel() - start)
                part = flat[start:start + count]
                # Copy endpoints directly; a + alpha*(b-a) can round even at alpha=1.
                if kind == 'p-output-gate' or alpha == 1.0:
                    value = trained.read(name, start, count)
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError(f'invalid trained tensor: {name}')
                    part.copy_(value)
                    if kind == 'p-output-gate' and is_read_output(name):
                        part.mul_(alpha)
                    del value
                else:
                    value = source.read(native_name, start, count)
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError(f'invalid source tensor: {name}')
                    part.copy_(value)  # BF16 -> FP32 as in the frozen source loader.
                    del value
                    if alpha:
                        value = trained.read(name, start, count)
                        if not bool(torch.isfinite(value).all()):
                            raise ValueError(f'invalid trained tensor: {name}')
                        # Same float32 operations as before; only IO granularity changed.
                        part.mul_(1.0 - alpha).add_(value, alpha=alpha)
                        del value
                if not bool(torch.isfinite(part).all()):
                    raise ValueError(f'nonfinite interpolated tensor: {name}')


def build_arm(source: Path, mode: str):
    from experiments.architecture_benchmark_001 import run
    model = run._load(source, 'cuda')[0]
    if mode == 'P':
        from experiments.dual28_ced_v1.model import DualStack
        rebuilt = DualStack.from_source(model, mode)
        del model
        model = rebuilt
        gc.collect()
    model.gradient_checkpointing = False
    model.eval()
    return model


def force_tf32_off() -> dict:
    # Avoid importing arm_dose/retention_dose_gpu: they parse argv at module import.
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
              'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
              'float32_matmul_precision': torch.get_float32_matmul_precision(),
              'weight_interpolation_dtype': 'torch.float32', 'model_dtype': 'torch.float32',
              'build_device': 'cuda', 'probe_device': 'cpu'}
    if result['cuda_matmul_allow_tf32'] or result['cudnn_allow_tf32']:
        raise ValueError('TF32 could not be disabled')
    return result


def summarize(outputs: list[dict], rows: list[dict], overlay: dict, references: set,
              source_scored: dict, strata: dict, policy: dict) -> dict:
    from experiments.architecture_benchmark_001 import retention, run
    scored = run._score_probe(outputs, overlay['targets'])
    counts = run._retention_classes(rows, source_scored, scored, references)
    verdict = retention.monitor_verdict(counts, policy, policy['probe_classes'])
    gained = sorted(i for i in scored if scored[i]['correct'] and i not in references)
    lost = sorted(i for i in references if not scored[i]['correct'])
    gain_ids = {name: [i for i in gained if strata[i] == name]
                for name in ('wrong_answer', 'format_only', 'runaway')}
    return {'correct_total': sum(bool(row['correct']) for row in scored.values()),
            'retained_total': len(references) - len(lost), 'lost_total': len(lost),
            'gained_total': len(gained), 'reference_n': len(references),
            'terminated_total': sum(bool(row['terminated']) for row in scored.values()),
            'gain_strata': {name: len(ids) for name, ids in gain_ids.items()},
            'gain_ids_by_stratum': gain_ids, 'lost_ids': lost,
            'status': verdict['status'], 'failed_classes': verdict['failed_classes'],
            'per_class': counts, 'per_row': {i: bool(row['correct']) for i, row in scored.items()},
            'per_row_scored': scored,
            'outputs': {row['id']: {key: row[key] for key in ('text', 'terminated', 'new_tokens')}
                        for row in outputs}}


def evaluate(args, report: dict, out: Path) -> None:
    for path in (REPO, EXPLORATION):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import hostbudget
    tag = args.tag
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', tag):
        raise ValueError('tag must be one checkpoint-directory name')
    directory = CHECKPOINTS / tag
    manifest_path = directory / 'MANIFEST.json'
    manifest = read_json(manifest_path)
    if manifest.get('completed') is not True or manifest.get('tag') != tag:
        raise ValueError('checkpoint is incomplete or tag does not match directory')
    mode, tail = arm_identity(manifest, args.kind)
    report.update(arm=manifest['arm'], mode=mode, tail=tail, phase='preflight')
    report['host_before'] = hostbudget.require(6.0 if mode == 'P' else 3.6, f'alpha {tag}')
    checkpoint = directory / 'trainable.safetensors'
    if checkpoint.stat().st_size != manifest['bytes'] or sha256_file(checkpoint) != manifest['sha256']:
        raise ValueError('checkpoint byte count or SHA-256 mismatch')
    source = Path(manifest['reconstruct']['source']).resolve()
    source_audit = verify_source_assets(source, ROOT / 'source_lock.json')

    import torch
    from experiments.architecture_benchmark_001 import retention, run
    from experiments.architecture_benchmark_001.gates import implementation_digest, protocol, runtime
    from experiments.architecture_benchmark_001.transfer import native_key_map
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required for the GPU-build / CPU-probe control')
    digest = implementation_digest()
    policy = protocol()['G1_RETENTION']
    registered = policy['bare_policy']
    frozen_path = ROOT / 'outputs' / f'retention-{digest[:8]}' / 'B28' / 'retention.json'
    frozen = read_json(frozen_path)
    unsigned = {k: v for k, v in frozen.items() if k != 'receipt_sha256'}
    if canonical_hash(unsigned) != frozen.get('receipt_sha256'):
        raise ValueError('frozen source receipt content digest mismatch')
    source_binding = source_key_kind(manifest['soup_key']['source_digest'], digest, source_audit, frozen)
    dose_path = EVIDENCE / f'ARM_DOSE_{tag}.json'
    dose = read_json(dose_path)
    if (dose.get('completed') is not True or dose.get('reference_digest') != digest or
            dose.get('arm') != manifest['arm'] or dose.get('seed') != manifest['seed'] or
            dose.get('checkpoint', {}).get('sha256') != manifest['sha256']):
        raise ValueError('ARM_DOSE endpoint is incomplete or does not bind the checkpoint/source/seed')
    if (dose['supervised_target_tokens'] != manifest['saved_after_tokens'] or
            dose['optimizer_steps'] != manifest['saved_after_steps']):
        raise ValueError('checkpoint save dose differs from ARM_DOSE endpoint')
    endpoint = dose['checkpoints'][-1]
    if endpoint['tokens'] != manifest['saved_after_tokens'] or endpoint['steps'] != manifest['saved_after_steps']:
        raise ValueError('last ARM_DOSE probe differs from checkpoint save dose')
    expected_one = endpoint['outputs']
    expected_zero = {row['id']: {'text': row['source']['raw'],
                                  'terminated': row['source']['terminated']} for row in frozen['rows']}
    panel_path = ROOT / 'data' / 'monitor_probe.jsonl'
    rows = run.load_probe_rows(panel_path)
    ids = {row['id'] for row in rows}
    if set(expected_one) != ids or set(expected_zero) != ids:
        raise ValueError('source/trained endpoints must cover every panel row')
    audit_path = EVIDENCE / 'PROBE_LABEL_AUDIT.json'
    strata_path = EVIDENCE / 'HEADROOM_AUDIT.json'
    prereg_path = EVIDENCE / 'DISTINGUISHABILITY_PREREG.md'
    overlay = retention.build_overlay(read_json(audit_path), registered)
    references = set(frozen['reference_ids_by_policy'][registered])
    source_scored = {i: retention.score_prediction(row['text'], row['terminated'], overlay['targets'][i])
                     for i, row in expected_zero.items()}
    if {i for i, row in source_scored.items() if row['correct']} != references:
        raise ValueError('re-scored source differs from frozen retention reference')
    strata = {}
    for name, group in read_json(strata_path)['policies'][registered]['ids'].items():
        if name not in ('wrong_answer', 'format_only', 'runaway'):
            raise ValueError(f'unknown source headroom stratum: {name}')
        for rid in group:
            if rid in strata:
                raise ValueError('duplicate row in headroom strata')
            strata[rid] = name
    if set(strata) != ids - references:
        raise ValueError('headroom strata do not exactly partition source failures')

    report['numerics'] = force_tf32_off()
    report['runtime'] = runtime()
    script_paths = [Path(__file__), EXPLORATION / 'hostbudget.py', EXPLORATION / 'tensor_stream.py',
                    REPO / 'experiments' / 'dual28_ced_v1' / 'model.py']
    report['provenance'] = {
        'implementation_digest': digest, 'actual_source': source_audit,
        'checkpoint_source_key': manifest['soup_key']['source_digest'],
        'checkpoint_source_key_kind': source_binding,
        'checkpoint_path': str(checkpoint), 'checkpoint_sha256': manifest['sha256'],
        'manifest_sha256': sha256_file(manifest_path),
        'frozen_source_path': str(frozen_path), 'frozen_source_sha256': sha256_file(frozen_path),
        'saved_endpoint_path': str(dose_path), 'saved_endpoint_sha256': sha256_file(dose_path),
        'data_hashes': {str(path.relative_to(REPO)): sha256_file(path)
                        for path in (panel_path, audit_path, strata_path, prereg_path)},
        'script_hashes': {str(path.relative_to(REPO)): sha256_file(path) for path in script_paths},
        'seed': manifest['seed'], 'saved_after_tokens': manifest['saved_after_tokens'],
        'saved_after_steps': manifest['saved_after_steps'],
    }
    report['registered_bare_policy'] = registered
    report['reference_n'] = len(references)
    report['panel_rows'] = len(rows)
    report['phase'] = 'build'
    write_json(out / 'REPORT.json', report)
    torch.manual_seed(manifest['seed'])
    model = build_arm(source, mode)
    try:
        names = validate_tensor_inventory(model, checkpoint, manifest, mode, tail)
        report['parameters'] = sum(p.numel() for p in model.parameters())
        report['trainable_parameters'] = manifest['parameters']
        report['interpolated_tensor_names'] = (names if mode == 'B28' else
                                                [name for name in names if is_read_output(name)])
        report['loaded_checkpoint_tensor_names'] = names
        source_names = ({dense: native for native, dense in native_key_map(model.config).items()}
                        if mode == 'B28' else None)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        gc.collect()
        torch.cuda.empty_cache()
        hostbudget.require(6.0 if mode == 'P' else 3.6, f'alpha {tag} CPU move')
        model.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        hostbudget.guard(1.2, f'alpha {tag} probe')
        tokenizer = run._load_tokenizer(source)
        eos = run._eos_ids(source, tokenizer)
        for alpha in EXECUTION_ORDER:
            if alpha not in (0.0, 1.0) and not report['endpoint_controls_passed']:
                raise ValueError('intermediate blocked: endpoint controls have not passed')
            report['phase'] = f'alpha_{alpha:g}'
            write_json(out / 'REPORT.json', report)
            apply_alpha(model, checkpoint, names, alpha, args.kind,
                        source_weights=source / 'model.safetensors', source_names=source_names,
                        guard=lambda: hostbudget.guard(1.2, f'alpha {tag} tensor update'))
            # Guard between panel rows; individual calls use the identical registered decoder.
            outputs = []
            for row in rows:
                hostbudget.guard(1.2, f'alpha {tag} {alpha:g} probe')
                outputs.extend(run._greedy_probe(model, tokenizer, [row], eos, 'cpu'))
            result = summarize(outputs, rows, overlay, references, source_scored, strata, policy)
            result.update(alpha=alpha, kind=args.kind, wall_seconds=time.monotonic() - report['_started'])
            if alpha in (0.0, 1.0):
                label = 'frozen_source' if alpha == 0.0 else 'saved_ARM_DOSE_endpoint'
                result['endpoint_control'] = endpoint_control(result['outputs'],
                    expected_zero if alpha == 0.0 else expected_one, label)
                report['endpoint_controls'][str(alpha)] = result['endpoint_control']
            artifact = out / f'alpha_{alpha:g}.json'
            write_json(artifact, result, exclusive=True)
            report['results'][str(alpha)] = result
            report['endpoint_controls_passed'] = (len(report['endpoint_controls']) == 2 and
                all(control['passed'] for control in report['endpoint_controls'].values()))
            write_json(out / 'REPORT.json', report)
            print(f"alpha {alpha:g}: correct {result['correct_total']}/{len(rows)}, "
                  f"kept {result['retained_total']}/{len(references)}, "
                  f"gains {result['gain_strata']}, {result['status']}", flush=True)
            if alpha in (0.0, 1.0) and not result['endpoint_control']['passed']:
                raise ValueError(f'{label} endpoint control FAILED; no intermediate may be interpreted')
        # Detect implementation/data edits during evaluation before calling the run complete.
        if implementation_digest() != digest:
            raise ValueError('implementation changed during the sweep')
        for relative, expected in {**report['provenance']['data_hashes'],
                                    **report['provenance']['script_hashes']}.items():
            if sha256_file(REPO / relative) != expected:
                raise ValueError(f'evaluation input changed during the sweep: {relative}')
        if (sha256_file(checkpoint) != manifest['sha256'] or
                sha256_file(manifest_path) != report['provenance']['manifest_sha256'] or
                sha256_file(dose_path) != report['provenance']['saved_endpoint_sha256'] or
                sha256_file(frozen_path) != report['provenance']['frozen_source_sha256']):
            raise ValueError('checkpoint, manifest or endpoint artifact changed during the sweep')
        if verify_source_assets(source, ROOT / 'source_lock.json') != source_audit:
            raise ValueError('actual source assets changed during the sweep')
        report.update(completed=True, phase='complete')
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tag', help='exact saved checkpoint tag; one tag per invocation')
    parser.add_argument('--kind', required=True, choices=KINDS)
    parser.add_argument('--out', required=True, type=Path,
                        help='new report directory beneath experiments/rab001_exploration/outputs')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    allowed = OUTPUTS.resolve()
    if out == allowed or not out.is_relative_to(allowed):
        parser.error(f'--out must be a new directory beneath {allowed}')
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 1, 'evidence_kind': 'exploratory_alpha_curve_only',
              'completed': False, 'phase': 'created', 'tag': args.tag, 'kind': args.kind,
              'grid': GRID, 'execution_order': EXECUTION_ORDER,
              'endpoint_controls_passed': False, 'endpoint_controls': {}, 'results': {},
              'selection_rule': 'none; the full fixed grid is descriptive; never select alpha on panel',
              'formula': ('theta(alpha)=(1-alpha)*source+alpha*trained on saved trainables'
                          if args.kind == 'b28-wise' else
                          'load all trained tensors; reads.*.o_proj.weight=alpha*trained o_proj; '
                          'other trainables fixed at trained values; NOT all-trainable WiSE interpolation'),
              'licenses_efficacy_claim': False, 'licenses_release': False,
              'pid': os.getpid(), 'python': sys.executable, 'platform': platform.platform(),
              '_started': time.monotonic()}
    write_json(out / 'REPORT.json', report, exclusive=True)
    code = 0
    try:
        evaluate(args, report, out)
    except BaseException as exc:
        report.update(completed=False, failed_phase=report['phase'], phase='failed',
                      error={'type': type(exc).__name__, 'message': str(exc),
                             'traceback': traceback.format_exc()})
        print(f'INCOMPLETE alpha sweep: {exc}', file=sys.stderr, flush=True)
        code = 1
    finally:
        report['wall_seconds'] = time.monotonic() - report.pop('_started')
        write_json(out / 'REPORT.json', report)
    print(f"{'COMPLETE' if report['completed'] else 'INCOMPLETE'}: {out / 'REPORT.json'}", flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
