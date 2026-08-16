from pathlib import Path

from rineya_soup_lab.evaluation.dataset import audit_splits
from conftest import write_dataset


def test_split_audit(tmp_path: Path) -> None:
    dev = write_dataset(tmp_path / "dev.jsonl", prefix="dev")
    test = write_dataset(tmp_path / "test.jsonl", prefix="test")
    report = audit_splits(dev, test, allow_id_overlap=False, allow_prompt_overlap=False)
    assert report["valid"] is True
    assert report["id_overlap_count"] == 0
