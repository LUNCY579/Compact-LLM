"""Materialise an MQ/RRP training checkpoint into a scoreable model directory.

The RRP engine stores a checkpoint as one safetensors file per trainable parameter, each
carrying `parameter`, `exp_avg` and `exp_avg_sq`, under a sealed `manifest.json`. That is the
right shape for resuming an optimizer and the wrong shape for every scorer in this repository,
which wants a directory holding `config.json` + `model.safetensors`.

This overlays the trained tensors onto the frozen source and writes that directory. It is a
copy, never arithmetic: each trained tensor is verified against the manifest sha256 and cast
only to the source dtype, and every parameter the checkpoint does not name is carried through
from the source unchanged. So the result is exactly "the source, with these 220 tensors
replaced", and the receipt records which ones moved.

    python experiments/rab001_exploration/materialize_checkpoint.py \
        --source Supa --checkpoint <...>/microfit/checkpoint-00110 --out <dir>

Writes MATERIALIZATION.json beside the weights with the source and checkpoint identities, the
per-tensor changed/identical split, and the count of source tensors left untouched. No model
is constructed and no model call is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.model_quality_gpu_v1.integrity import unseal                     # noqa: E402

# Carried verbatim so the directory is loadable by the same path as the source.
SIDECARS = ("config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_research_format(source: Path, state: dict, out: Path) -> None:
    """Write the DenseReference research format `retention_probe`/`run._restore` expect.

    The HF layout is what the source ships and what the RRP trainer writes. The RAB scorers
    restore a DenseReference through a hash-sealed `config.json` carrying
    `model_type: rineya_ced_research_v1`, so an HF directory fails their gate with
    "receipt content hash mismatch". `import_qwen_state` is the repository's own converter --
    the same one `run._load` uses for every baseline in this programme -- so scoring the
    converted model keeps the trained and frozen measurements on one path.
    """
    from experiments.architecture_benchmark_001 import run
    from experiments.architecture_benchmark_001.source import source_config
    from experiments.architecture_benchmark_001.transfer import import_qwen_state

    raw = json.loads((source / "config.json").read_text(encoding="utf-8"))
    model = import_qwen_state(state, source_config(raw)).to(dtype=torch.float32).eval()
    out.mkdir(parents=True, exist_ok=True)
    run._save(model, out)


def materialize(source: Path, checkpoint: Path, out: Path, *, research: bool = False) -> dict:
    if out.exists():
        raise SystemExit('create-only: {} exists'.format(out))
    body = unseal(json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8")))
    if body.get("status") != "COMPLETE":
        raise SystemExit('checkpoint is not COMPLETE and must not be materialised')

    state = load_file(str(source / "model.safetensors"), device="cpu")
    source_names = set(state)
    changed, identical, applied = [], [], 0
    for entry in body["entries"]:
        name, path = entry["name"], checkpoint / entry["file"]
        if name not in state:
            raise SystemExit('checkpoint names a parameter the source lacks: {}'.format(name))
        if sha256_file(path) != entry["sha256"]:
            raise SystemExit('tensor bytes differ from the manifest: {}'.format(entry["file"]))
        trained = load_file(str(path), device="cpu")["parameter"]
        if list(trained.shape) != entry["shape"] or trained.shape != state[name].shape:
            raise SystemExit('shape differs from the source: {}'.format(name))
        trained = trained.to(state[name].dtype)
        (identical if torch.equal(trained, state[name]) else changed).append(name)
        state[name] = trained
        applied += 1

    if research:
        write_research_format(source, state, out)
        copied = []
    else:
        out.mkdir(parents=True)
        save_file(state, str(out / "model.safetensors"))
        copied = []
        for name in SIDECARS:
            if (source / name).is_file():
                shutil.copy2(source / name, out / name)
                copied.append(name)

    receipt = {
        "record": "CHECKPOINT_MATERIALIZATION",
        "schema": 1,
        "what_this_is": ("The frozen source with the checkpoint's trainable tensors copied over "
                         "it. A copy, not arithmetic. No model was constructed, no model call "
                         "was made, and this receipt establishes no quality claim."),
        "source": str(source),
        "source_weight_sha256": sha256_file(source / "model.safetensors"),
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": sha256_file(checkpoint / "manifest.json"),
        "checkpoint_experiment": body.get("experiment"),
        "checkpoint_progress": body.get("progress"),
        "tensors_in_checkpoint": len(body["entries"]),
        "tensors_applied": applied,
        "tensors_changed_by_training": len(changed),
        "tensors_identical_to_source": len(identical),
        "identical_tensor_names": identical,
        "source_tensors_untouched": sorted(source_names - {e["name"] for e in body["entries"]}),
        "out_weight_sha256": sha256_file(out / "model.safetensors"),
        "layout": "rineya_ced_research_v1" if research else "huggingface",
        "sidecars_copied": copied,
        "quality_claim": False,
        "materialized_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "MATERIALIZATION.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--research-format', action='store_true',
                        help='write the DenseReference layout run._restore expects, '
                             'which is what retention_probe and the RAB scorers read')
    args = parser.parse_args(argv)

    source = args.source if args.source.is_absolute() else REPO / args.source
    receipt = materialize(source, args.checkpoint, args.out, research=args.research_format)
    print('materialised {}'.format(args.out))
    print('  experiment          : {}'.format(receipt['checkpoint_experiment']))
    print('  updates             : {}'.format((receipt['checkpoint_progress'] or {}).get('updates')))
    print('  tensors applied     : {}'.format(receipt['tensors_applied']))
    print('  changed by training : {}'.format(receipt['tensors_changed_by_training']))
    print('  identical to source : {}'.format(receipt['tensors_identical_to_source']))
    print('  source untouched    : {}'.format(len(receipt['source_tensors_untouched'])))
    print('  layout              : {}'.format(receipt['layout']))
    print('  out weight sha256   : {}'.format(receipt['out_weight_sha256'][:16]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
