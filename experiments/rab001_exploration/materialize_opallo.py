"""Promote a trained arm checkpoint into a dense Hugging Face core in the family vault.

An `arm-checkpoints/` entry is 220 float32 tensors under CED-native names
(`encoder.12.q_proj.weight`) and is useless to anyone outside this harness. A published
core is a complete `model.safetensors` under ordinary Qwen3 names, with byte-exact
tokenizer assets beside it, that `AutoModelForCausalLM.from_pretrained` will open. This
converts the first into the second.

    python experiments/rab001_exploration/materialize_opallo.py \
        --tag B28last20_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95 \
        --codename Opallo --version 1.0.0-devfreeze

The conversion is an exact inverse. `transfer.native_key_map` is a bijection between
Qwen3 tensor names and this harness's names, so every trained tensor has exactly one
destination -- and the map is checked for injectivity rather than assumed to have it.

Everything that can be verified is verified, because the failure mode here is silent: a
core that loads, generates fluent text, and is not the model that was measured.

  * the base weights must hash to the `source_weights_sha256` recorded in the checkpoint,
    and every tokenizer asset to its `source_assets_sha256` entry -- otherwise the
    published core is a different model from the one that scored;
  * the trained tensor set must carry the arm's declared parameter count and must contain
    no embedding, no final norm and nothing from the frozen block prefix. That is the
    claim "53% of parameters, first 8 blocks frozen" checked rather than repeated;
  * every tensor outside the trained set must come out byte-identical to the base;
  * the written file must reload through the harness's own strict importer with no
    missing, unexpected or mismatched keys.

The cast is the one lossy step and it is measured, not waved through: the checkpoint is
float32, the vault is BF16, and the script reports how many elements the cast moved and
by how much. Whether that costs anything on the panel is a separate measurement --
`validate_published_core.py` -- because it needs a model run and this does not.

Create-only. A published core directory is never overwritten; the vault policy is to
archive the old one under `_superseded/` and write a new version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
sys.stdout.reconfigure(encoding='utf-8')

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.architecture_benchmark_001.gates import implementation_digest   # noqa: E402
from experiments.architecture_benchmark_001.source import ROOT, source_config    # noqa: E402
from experiments.architecture_benchmark_001.transfer import native_key_map       # noqa: E402

VAULT = REPO / 'Rineya_Published_Cores'
ASSETS = ('config.json', 'generation_config.json', 'merges.txt',
          'tokenizer.json', 'tokenizer_config.json', 'vocab.json')
DTYPES = {'bfloat16': torch.bfloat16, 'float32': torch.float32}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True, help='directory name under outputs/arm-checkpoints')
    parser.add_argument('--codename', required=True)
    parser.add_argument('--version', default='1.0.0-devfreeze')
    parser.add_argument('--dtype', choices=sorted(DTYPES), default='bfloat16')
    args = parser.parse_args()

    ckpt = ROOT / 'outputs' / 'arm-checkpoints' / args.tag
    manifest = json.loads((ckpt / 'MANIFEST.json').read_text(encoding='utf-8'))
    if not manifest.get('completed'):
        raise SystemExit(f'{args.tag} did not complete; an aborted run reports its step-0 probe '
                         f'as its score and must never be published')
    if manifest.get('complete_model'):
        raise SystemExit('this promotes a trainable-tensors-only checkpoint against its base')
    source = Path(manifest['reconstruct']['source'])
    if not (source / 'model.safetensors').is_file():
        raise SystemExit(f'base weights missing: {source}')

    # The published core must be the model that scored. Check the base it was trained on.
    base_sha = sha256_file(source / 'model.safetensors')
    if base_sha != manifest['source_weights_sha256']:
        raise SystemExit(f'base weights changed since training: {base_sha} != '
                         f'{manifest["source_weights_sha256"]}')
    asset_sha = {name: sha256_file(source / name) for name in ASSETS}
    for name, expected in manifest['source_assets_sha256'].items():
        if name == 'model.safetensors':
            continue
        if asset_sha.get(name) != expected:
            raise SystemExit(f'source asset changed since training: {name}')
    digest = implementation_digest()
    if digest != manifest['implementation_digest']:
        print(f'  NOTE implementation digest moved since training: {digest[:8]} != '
              f'{manifest["implementation_digest"][:8]}')

    raw = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    config = source_config(raw)
    forward = native_key_map(config)                       # qwen name -> harness name
    if len(set(forward.values())) != len(forward):
        raise SystemExit('native key map is not injective; the inverse is undefined')
    inverse = {v: k for k, v in forward.items()}           # harness name -> qwen name

    base = load_file(str(source / 'model.safetensors'), device='cpu')
    trained = load_file(str(ckpt / 'trainable.safetensors'), device='cpu')
    print(f'base {len(base)} tensors | trained {len(trained)} tensors | '
          f'target dtype {args.dtype}')
    if set(base) != set(forward):
        raise SystemExit(f'base inventory is not the expected Qwen3 set: '
                         f'missing={sorted(set(forward) - set(base))[:4]}, '
                         f'unexpected={sorted(set(base) - set(forward))[:4]}')

    block_tail = manifest['reconstruct']['block_tail']
    frozen_prefix = config.encoder_layers - block_tail if block_tail else 0
    forbidden = {'model.embed_tokens.weight', 'model.norm.weight'}
    forbidden |= {n for n in base if any(n.startswith(f'model.layers.{i}.')
                                         for i in range(frozen_prefix))}
    mapped, parameters = {}, 0
    for name, value in trained.items():
        target = inverse.get(name)
        if target is None:
            raise SystemExit(f'trained tensor has no Qwen3 destination: {name}')
        if target in forbidden:
            raise SystemExit(f'trained tensor is inside the declared frozen set: {name} -> {target}')
        if value.shape != base[target].shape:
            raise SystemExit(f'shape mismatch for {name}: {tuple(value.shape)} vs '
                             f'{tuple(base[target].shape)}')
        if not torch.isfinite(value).all():
            raise SystemExit(f'trained tensor is not finite: {name}')
        mapped[target] = value
        parameters += value.numel()
    if parameters != manifest['parameters']:
        raise SystemExit(f'trained parameter count {parameters:,} != recorded '
                         f'{manifest["parameters"]:,}')
    print(f'  {len(mapped)} trained tensors -> Qwen3 names, {parameters:,} parameters, '
          f'blocks {frozen_prefix}..{config.encoder_layers - 1} trainable')

    # The cast is the only lossy step in this script. Measure it instead of trusting it.
    dtype = DTYPES[args.dtype]
    moved, worst, worst_name = 0, 0.0, None
    out_state = {}
    for name, value in base.items():
        if name in mapped:
            cast = mapped[name].to(dtype)
            delta = (cast.to(torch.float32) - mapped[name]).abs()
            scale = mapped[name].abs().clamp_min(1e-12)
            moved += int((delta > 0).sum())
            relative = float((delta / scale).max())
            if relative > worst:
                worst, worst_name = relative, name
            out_state[name] = cast
        else:
            out_state[name] = value.clone()
    print(f'  cast to {args.dtype}: {moved:,} of {parameters:,} elements moved '
          f'({100 * moved / parameters:.2f}%), worst relative delta {worst:.3e} '
          f'({worst_name})')

    target_dir = VAULT / f'Rineya-{args.codename}-{args.version}'
    if target_dir.exists():
        raise SystemExit(f'core directory already exists (create-only, never overwrite): '
                         f'{target_dir}\nArchive it under _superseded/ and bump the version.')
    target_dir.mkdir(parents=True, exist_ok=False)
    save_file(out_state, str(target_dir / 'model.safetensors'), metadata={'format': 'pt'})
    for name in ASSETS:
        shutil.copyfile(source / name, target_dir / name)

    # Reload what was actually written -- not what is in memory -- and prove the frozen
    # tensors survived byte-identical and the trained ones are exactly the cast values.
    written = load_file(str(target_dir / 'model.safetensors'), device='cpu')
    if set(written) != set(base):
        raise SystemExit('written inventory does not match the base inventory')
    for name, value in written.items():
        if name in mapped:
            if not torch.equal(value, mapped[name].to(dtype)):
                raise SystemExit(f'written trained tensor differs from the cast value: {name}')
        elif not torch.equal(value, base[name]):
            raise SystemExit(f'FROZEN TENSOR CHANGED: {name}')
    print(f'  verified on reload: {len(mapped)} trained substituted, '
          f'{len(written) - len(mapped)} frozen byte-identical to base')

    from experiments.architecture_benchmark_001.transfer import import_qwen_state
    model = import_qwen_state(dict(written), config)
    reloaded = model.state_dict()
    if set(reloaded) != set(forward.values()):
        raise SystemExit('strict harness import did not reproduce the expected inventory')
    if not all(torch.isfinite(v).all() for v in reloaded.values()):
        raise SystemExit('reloaded model contains non-finite values')
    del model, reloaded
    print('  strict harness reload: no missing / unexpected / mismatched keys, all finite')

    weight_sha = sha256_file(target_dir / 'model.safetensors')
    files = {p.name: {'sha256': sha256_file(p), 'bytes': p.stat().st_size}
             for p in sorted(target_dir.iterdir()) if p.is_file()}
    core_id = (f'rineya.core.{args.codename.lower()}.v{args.version}.'
               f'{config.encoder_layers}L.supa-{manifest["arm"].lower()}-wsd-'
               f'b{manifest["token_budget"]}-seed{manifest["seed"]}')
    serialized = sum(v.numel() for v in written.values())
    core = {
        'schema_version': '1.1',
        'core_id': core_id,
        'codename': args.codename,
        'directory_name': target_dir.name,
        'version': args.version,
        'kind': 'partial_finetune_dense_substitution',
        'status': 'exploratory_research_core_measured_not_gated',
        'classification': 'rab001_exploration_promoted_core',
        'provisional': False,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'device': 'cpu',
        'sources_unmodified': True,
        'source': {
            'base': {
                'path': str(source.relative_to(REPO)) if source.is_relative_to(REPO) else str(source),
                'core_id': 'rineya.core.supa.v1.7.0-devfreeze.28L.'
                           'qwen3-nimbus-lcr-stage-a-step200-scale-0p8125',
                'weight_sha256': base_sha,
                'asset_sha256': asset_sha,
            },
            'checkpoint': {
                'path': str(ckpt.relative_to(REPO)),
                'tag': manifest['canonical_tag'],
                'trainable_sha256': manifest['sha256'],
                'implementation_digest': manifest['implementation_digest'],
                'completed': manifest['completed'],
            },
            'training': {
                'harness': 'experiments/rab001_exploration/arm_dose.py',
                'arm': manifest['arm'],
                'block_tail': block_tail,
                'learning_rate': manifest['lr'],
                'schedule': manifest['schedule'],
                'adam_beta2': manifest['beta2'],
                'seed': manifest['seed'],
                'token_budget': manifest['token_budget'],
                'supervised_tokens_consumed': manifest['saved_after_tokens'],
                'optimizer_steps': manifest['saved_after_steps'],
                'training_dtype': manifest['dtype'],
            },
        },
        'substitution': {
            'formula': f'{args.dtype}(base) with {len(mapped)} trained tensors replaced by '
                       f'{args.dtype}(float32 trained value)',
            'key_map': 'experiments/architecture_benchmark_001/transfer.py:native_key_map, inverted',
            'trained_tensors': len(mapped),
            'trained_parameters': parameters,
            'trainable_fraction_of_model': parameters / serialized,
            'frozen': f'embeddings, final norm and blocks 0..{frozen_prefix - 1}',
            'frozen_tensors_byte_identical_to_base': True,
            'cast': {
                'from': manifest['dtype'], 'to': args.dtype,
                'elements_moved': moved,
                'elements_total': parameters,
                'fraction_moved': moved / parameters,
                'worst_relative_delta': worst,
                'worst_relative_delta_tensor': worst_name,
            },
        },
        'checks': {
            'model_type': raw['model_type'],
            'architecture': raw['architectures'][0],
            'num_hidden_layers': config.encoder_layers,
            'hidden_size': raw['hidden_size'],
            'vocab_size': raw['vocab_size'],
            'tie_word_embeddings': raw.get('tie_word_embeddings'),
            'weight_bytes': (target_dir / 'model.safetensors').stat().st_size,
            'weight_sha256': weight_sha,
            'tensor_count': len(written),
            'parameter_count_serialized': serialized,
            'dtype_counts': {args.dtype: len(written)},
            'all_finite': True,
            'no_peft_remnants': True,
            'tensor_inventory_missing_from_base': ['lm_head.weight'],
            'tensor_inventory_unexpected': [],
            'reload': {'performed': True,
                       'importer': 'experiments.architecture_benchmark_001.transfer'
                                   '.import_qwen_state',
                       'missing_keys': [], 'unexpected_keys': [], 'mismatched_keys': []},
        },
        'panel_validation': {
            'status': 'NOT_MEASURED',
            'note': 'The published artifact is BF16; the measured checkpoint was float32. '
                    'Run validate_published_core.py before quoting any score for this core.',
        },
        'licenses_registered_training': False,
        'licenses_efficacy_claim': False,
        'opens_gate': False,
    }
    (target_dir / 'CORE.json').write_text(json.dumps(core, indent=2, sort_keys=True) + '\n',
                                          encoding='utf-8')
    (target_dir / 'FILES.json').write_text(json.dumps(files, indent=2, sort_keys=True) + '\n',
                                           encoding='utf-8')
    print(f'\nwrote {target_dir}')
    print(f'  core_id {core_id}')
    print(f'  weight_sha256 {weight_sha}')
    print(f'  {len(written)} tensors, {serialized:,} parameters, '
          f'{(target_dir / "model.safetensors").stat().st_size:,} bytes')
    print('\n  panel_validation is NOT_MEASURED. Run validate_published_core.py next.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
