from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from rineya_state_lab.data.prepare import UnsafeArchiveError, safe_extract_zip


def test_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "bad")
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(archive, tmp_path / "out")


def test_safe_extract_accepts_normal_file(tmp_path: Path) -> None:
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("root/file.txt", "hello")
    files = safe_extract_zip(archive, tmp_path / "out")
    assert len(files) == 1
    assert (tmp_path / "out/root/file.txt").read_text() == "hello"
