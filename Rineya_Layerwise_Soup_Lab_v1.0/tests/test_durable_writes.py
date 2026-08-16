from __future__ import annotations

from pathlib import Path

from rineya_soup_lab.evaluation.native_v45 import load_existing_rows
from rineya_soup_lab.utils import append_jsonl, atomic_write_text, is_file_lock_error, replace_file


def test_replace_file_retries_windows_share_lock(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "src.txt"
    destination = tmp_path / "dst.txt"
    source.write_text("new", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")
    attempts = {"count": 0}

    def flaky_replace(src, dst):
        attempts["count"] += 1
        if attempts["count"] < 3:
            error = PermissionError(13, "Access is denied")
            error.winerror = 5
            raise error
        Path(dst).write_text(Path(src).read_text(encoding="utf-8"), encoding="utf-8")
        Path(src).unlink()

    monkeypatch.setattr("rineya_soup_lab.utils.os.replace", flaky_replace)
    replace_file(source, destination)
    assert destination.read_text(encoding="utf-8") == "new"
    assert attempts["count"] == 3


def test_atomic_write_text_falls_back_to_in_place_copy(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("old", encoding="utf-8")

    def always_locked(src, dst):
        error = PermissionError(13, "Access is denied")
        error.winerror = 5
        raise error

    monkeypatch.setattr("rineya_soup_lab.utils.os.replace", always_locked)
    monkeypatch.setattr("rineya_soup_lab.utils._LOCK_ATTEMPTS", 2)
    monkeypatch.setattr("rineya_soup_lab.utils._LOCK_BASE_DELAY_SECONDS", 0.0)
    atomic_write_text(path, "replacement\n")
    assert path.read_text(encoding="utf-8") == "replacement\n"


def test_load_existing_rows_skips_truncated_trailing_line(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    append_jsonl(path, {"id": "a", "reply": "ok"})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "b", "reply":')
    rows = load_existing_rows(path)
    assert list(rows) == ["a"]


def test_permission_error_with_winerror_is_lock_error() -> None:
    error = PermissionError(13, "Access is denied")
    error.winerror = 5
    assert is_file_lock_error(error)
