from __future__ import annotations

from pathlib import Path

import pytest

from rineya_significance_lab.integrity import (
    artifact_hash_map,
    content_tree_sha256,
    hash_tree,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def test_hash_tree_is_deterministic_across_order_and_overlaps(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "src" / "z.py", "Z = 1\n")
    _write(root / "src" / "nested" / "a.py", "A = 1\n")
    _write(root / "configs" / "halo.yaml", "model: halo\n")

    first = hash_tree(root, ["src", "configs/halo.yaml"])
    second = hash_tree(root, ["configs", "src/nested/a.py", "src"])

    assert first == second
    assert list(first) == sorted(first)
    assert set(first) == {
        "configs/halo.yaml",
        "src/nested/a.py",
        "src/z.py",
    }
    assert content_tree_sha256(first) == content_tree_sha256(dict(reversed(first.items())))


def test_hash_tree_and_content_sha_detect_mutation(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    source = root / "src" / "halo.py"
    _write(source, "VALUE = 1\n")

    before = hash_tree(root, ["src"])
    before_tree = content_tree_sha256(before)
    _write(source, "VALUE = 2\n")
    after = hash_tree(root, ["src"])

    assert before["src/halo.py"] != after["src/halo.py"]
    assert before_tree != content_tree_sha256(after)


def test_artifact_hash_map_is_recursive_and_excludes_only_its_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write(run_dir / "content_manifest.json", '{"self": "changes"}\n')
    _write(run_dir / "summary.json", '{"accuracy": 1.0}\n')
    _write(run_dir / "traces" / "trace.jsonl", '{"case": 1}\n')
    _write(run_dir / "traces" / "content_manifest.json", "nested artifact\n")
    _write(run_dir / ".diagnostics", "kept\n")

    before = artifact_hash_map(run_dir)
    _write(run_dir / "content_manifest.json", '{"self": "changed again"}\n')
    after = artifact_hash_map(run_dir)

    assert before == after
    assert set(before) == {
        ".diagnostics",
        "summary.json",
        "traces/content_manifest.json",
        "traces/trace.jsonl",
    }


def test_windows_style_inputs_and_hash_keys_are_normalized(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "src" / "rineya" / "halo.py", "HALO = True\n")
    _write(root / "configs" / "halo.yaml", "family: rineya\n")

    hashes = hash_tree(root, [r"src\rineya", r"configs\halo.yaml"])
    digest = hashes["src/rineya/halo.py"]

    assert all("\\" not in path for path in hashes)
    assert content_tree_sha256({r"src\rineya\halo.py": digest}) == content_tree_sha256(
        {"src/rineya/halo.py": digest}
    )


def test_hash_tree_rejects_selections_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.py"
    _write(outside, "unsafe = True\n")

    with pytest.raises(ValueError, match="outside the hash root"):
        hash_tree(root, [outside])


def test_content_tree_rejects_normalization_collisions() -> None:
    digest = "0" * 64
    with pytest.raises(ValueError, match="same path"):
        content_tree_sha256({r"src\halo.py": digest, "src/halo.py": digest})
