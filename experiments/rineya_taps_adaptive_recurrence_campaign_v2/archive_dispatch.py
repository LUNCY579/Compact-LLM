"""Preserve the exact uncommitted reviewed dispatch bytes in an exclusive archive."""
import argparse
import json
from pathlib import Path
import shutil

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.run_apparatus import ROOT, PACKAGE, sha


def main(admission_path, name):
    if Path(name).name != name:
        raise ValueError("archive name only")
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    copied = {}
    for relative, expected in admission["dispatch_hashes"].items():
        source = (ROOT / relative).resolve(strict=True)
        if ROOT not in source.parents or sha(source) != expected:
            raise RuntimeError("dispatch bytes changed before archiving: " + relative)
        copied[relative] = expected
    out = PACKAGE / "versions" / name
    out.mkdir(parents=True, exist_ok=False)
    for relative in copied:
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        if sha(destination) != copied[relative]:
            raise RuntimeError("archive copy mismatch")
    (out / "admission.json").write_bytes(admission_path.read_bytes())
    source = Path(admission["source"])
    for asset in ("modeling_supa_a11.py", "config.json", "generation_config.json", "tokenizer_config.json", "CORE.json", "FILES.json"):
        expected = admission["checkpoint_files"][asset]
        if sha(source / asset) != expected:
            raise RuntimeError("asset metadata changed before archive")
        destination = out / "canonical_metadata" / asset
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source / asset, destination)
        copied["canonical_metadata/" + asset] = expected
    (out / "ARCHIVE.json").write_text(json.dumps({"admission_sha256": sha(admission_path),
        "files_sha256": copied, "weights_copied": False, "source_only": True,
        "warning": "Immutable evidence snapshot; do not execute its nested inherited entrypoints or restore over source-bound originals."}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(out), "files": len(copied), "admission_sha256": sha(admission_path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    main(args.admission.resolve(strict=True), args.name)
