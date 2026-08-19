from __future__ import annotations

import zipfile
from pathlib import Path

import pytest


@pytest.mark.integration
def test_bundled_archive_has_expected_dataset_files() -> None:
    root = Path(__file__).resolve().parents[1]
    archive = root / "assets/datasets/Rineya_Sequence_Independence_Dataset_v1.0_COMPLETE.zip"
    assert archive.stat().st_size > 50_000_000
    with zipfile.ZipFile(archive) as zf:
        assert zf.testzip() is None
        names = set(zf.namelist())
    for relative in [
        "rineya_sequence_independence_dataset_v1/data/full_context/train.jsonl",
        "rineya_sequence_independence_dataset_v1/data/full_context/dev.jsonl",
        "rineya_sequence_independence_dataset_v1/data/full_context/test.jsonl",
        "rineya_sequence_independence_dataset_v1/data/split_transitions/train.jsonl",
        "rineya_sequence_independence_dataset_v1/data/split_transitions/dev.jsonl",
        "rineya_sequence_independence_dataset_v1/data/split_transitions/test.jsonl",
    ]:
        assert relative in names
