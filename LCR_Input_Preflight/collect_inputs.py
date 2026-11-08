#!/usr/bin/env python3
"""Collect only explicitly selected LCR inputs; no network/model/training calls.

This is an input collector, NOT a corpus generator or acceptance certificate.
Requires Python 3.10+. It does not modify the selected source files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

EXPECTED = {
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
SOURCE_REF = {
    "repository": "LUNCY579/ChatBubble",
    "revision": "dbcc3d6e81898d7d637f181948e3971db511433a",
    "path": "experiments/model_quality_gpu_v1/source_lock.json",
    "subject": "Rineya-Supa-1.7.0-devfreeze",
}
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024


def read_checked(path: Path, limit: int = MAX_FILE_BYTES) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular non-symlink file: {path}")
    before = path.stat()
    if before.st_size > limit:
        raise ValueError(f"File exceeds explicit size limit: {path}")
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed during read: {path}")
    if len(data) != after.st_size:
        raise ValueError(f"Incomplete read: {path}")
    if data.startswith(b"version https://git-lfs.github.com/spec/"):
        raise ValueError(f"LFS pointer is not an actual asset: {path}")
    return data


def verify_assets(folder: Path, expected: dict[str, str]) -> dict[str, bytes]:
    """Expected hashes are fixed by the production CLI, not supplied by users."""
    result = {}
    for name, digest in expected.items():
        if Path(name).name != name:
            raise ValueError("Unsafe asset basename")
        data = read_checked(folder / name)
        observed = hashlib.sha256(data).hexdigest()
        if observed != digest:
            raise ValueError(f"Asset mismatch for {name}: expected {digest}; got {observed}")
        result[name] = data
    return result


def native_fixture_probe(folder: Path) -> dict:
    """Optional native tokenizer only; use local files and never load a model."""
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError("Native probe requires Transformers in the selected Python environment; no install was attempted") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True,
                                              trust_remote_code=False)
    fixtures = [
        ("The parcel label is Naviri. Nothing else is recorded.", "Naviri"),
        ("No current destination is recorded for the requested courier.", "UNKNOWN"),
        ("The verified quantities are 17 and 23; their exact sum is requested.", "40"),
        ("Note: re-enter; pre-approval; ID-A7.\nSecond line: 003 and 1.25.", "1.25"),
        ("Quoted names: ‘Félin’, \"O'Rana\". Non-ASCII check: 東京; α; →.", "Félin"),
        ("  Leading spaces\n\nTabs\tand punctuation: [x], {y}, (z)!", "x"),
    ]
    records = []
    for user, answer in fixtures:
        messages = [
            {"role":"system", "content":"Use only the record. Return the exact short answer or UNKNOWN."},
            {"role":"user", "content":user},
            {"role":"assistant", "content":answer},
        ]
        prefix = tokenizer.apply_chat_template(messages[:2], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=False, enable_thinking=False)
        ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                            add_generation_prompt=False, enable_thinking=False)
        prefix_ids = tokenizer.apply_chat_template(messages[:2], tokenize=True,
                                                   add_generation_prompt=True, enable_thinking=False)
        if not rendered.startswith(prefix) or ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError("Native fixture failed generation-prefix equality")
        if ids != tokenizer.encode(rendered, add_special_tokens=False):
            raise ValueError("Native template tokenization and text encoding disagree")
        records.append({"messages":messages, "rendered":rendered,
                        "input_ids":ids, "generation_prefix_ids":prefix_ids,
                        "raw_user_ids":tokenizer.encode(user, add_special_tokens=False)})
    return {"status":"NATIVE_TOKENIZER_FIXTURES_EXECUTED_NOT_CORPUS_MEASUREMENT",
            "transformers_version":transformers.__version__,
            "tokenizer_class":type(tokenizer).__name__,
            "eos_token":tokenizer.eos_token, "eos_token_id":tokenizer.eos_token_id,
            "fixtures":records, "model_calls":0}


def collect(folder: Path, banks: list[Path], output: Path,
            expected: dict[str, str] | None = None, native_probe: bool = False) -> dict:
    # Alternate hashes exist only to test collector mechanics with tiny fixtures.
    # main() always uses the locked production EXPECTED values.
    expected = EXPECTED if expected is None else expected
    if output.exists():
        raise ValueError("Output exists; choose a new path. No overwrite is allowed.")
    assets = verify_assets(folder, expected)
    if not banks:
        raise ValueError("Select at least one permitted exclusion file explicitly.")
    entries = {f"tokenizer/{name}": value for name, value in assets.items()}
    bank_inventory = []
    total_bytes = sum(map(len, entries.values()))
    for index, path in enumerate(banks, 1):
        if path.suffix.lower() not in {".json", ".jsonl", ".txt", ".py", ".md", ".gz", ".zip"}:
            raise ValueError(f"Unsupported exclusion-file extension: {path.name}")
        data = read_checked(path)
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_BYTES:
            raise ValueError("Combined source files exceed the collection size limit.")
        name = f"exclusion_inputs/{index:02d}_{path.name}"
        entries[name] = data
        bank_inventory.append({
            "archive_path": name, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "purpose": "exclusion audit only; not corpus generation or model evaluation",
        })
    probe_status = "NOT_RUN"
    if native_probe:
        probe = native_fixture_probe(folder)
        verify_assets(folder, expected)  # Reject an asset change during the probe.
        entries["NATIVE_TOKENIZER_FIXTURES.json"] = (json.dumps(probe, indent=2, ensure_ascii=False) + "\n").encode()
        probe_status = probe["status"]
    manifest = {
        "schema_version": 1,
        "source_ref": SOURCE_REF,
        "status": "COLLECTED_INPUTS_NOT_CORPUS_ACCEPTANCE",
        "tokenizer_asset_hashes_verified": True,
        "native_tokenization": probe_status,
        "corpus_rows_generated": 0,
        "model_calls": 0,
        "optimizer_steps": 0,
        "exclusion_scope": "only explicitly selected files; global completeness not established",
        "selected_exclusion_inputs": bank_inventory,
        "files": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(entries.items())
        },
    }
    entries["INPUT_MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build and validate before exposing the final archive. An exclusive create
    # also prevents replacing a file created after the first existence check.
    with tempfile.TemporaryDirectory(dir=output.parent) as td:
        draft = Path(td) / "input_bundle.zip"
        with zipfile.ZipFile(draft, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, data in sorted(entries.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 16, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data)
        with zipfile.ZipFile(draft) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive CRC check failed")
            for name, data in entries.items():
                if archive.read(name) != data:
                    raise ValueError(f"Archive byte mismatch: {name}")
        with output.open("xb") as target:
            target.write(draft.read_bytes())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--exclusion", type=Path, action="append", required=True,
                        help="Permitted historical prompt bank / renderer / entity manifest. Repeatable.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-probe", action="store_true",
                        help="Use the already installed local Transformers tokenizer; never load a model.")
    args = parser.parse_args()
    try:
        manifest = collect(args.tokenizer_dir, args.exclusion, args.output, native_probe=args.native_probe)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Collection refused: {exc}\n")
    print(json.dumps({"archive": str(args.output), "status": manifest["status"],
                      "asset_count": len(EXPECTED), "exclusion_files": len(args.exclusion)}, indent=2))


if __name__ == "__main__":
    main()
