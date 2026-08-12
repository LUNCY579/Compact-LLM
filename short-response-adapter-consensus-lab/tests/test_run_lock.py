from __future__ import annotations

from pathlib import Path

import pytest

from consensus_lab.run_lock import exclusive_run_lock


def test_exclusive_run_lock_rejects_concurrent_writer(tmp_path: Path) -> None:
    lock = tmp_path / "run.lock"
    with exclusive_run_lock(lock):
        assert lock.is_file()
        with pytest.raises(RuntimeError, match="already owned"):
            with exclusive_run_lock(lock):
                pass
    assert not lock.exists()
