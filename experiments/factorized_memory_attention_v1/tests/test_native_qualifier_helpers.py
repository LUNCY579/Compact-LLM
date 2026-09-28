from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from experiments.factorized_memory_attention_v1.scripts.qualify_native_checkpoint import (
    critical_source_snapshot,
    sha256_file,
)


def test_streaming_sha256_matches_hashlib_without_read_bytes(tmp_path, monkeypatch):
    payload = (b"factorized-memory-attention-" * 10000) + b"tail"
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(payload)

    def forbidden_read_bytes(self):
        raise AssertionError("qualifier must stream large checkpoint hashes")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    assert sha256_file(path, chunk_bytes=1024) == hashlib.sha256(payload).hexdigest()


def test_critical_source_snapshot_binds_checkpoint_config_and_tokenizer(tmp_path):
    assets = {
        "model.safetensors": b"weights",
        "config.json": b'{"model_type":"qwen3"}',
        "tokenizer.json": b'{"version":"1"}',
    }
    for name, payload in assets.items():
        (tmp_path / name).write_bytes(payload)
    snapshot = critical_source_snapshot(tmp_path)
    assert set(snapshot) == set(assets)
    for name, payload in assets.items():
        assert snapshot[name]["bytes"] == len(payload)
        assert snapshot[name]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_critical_source_snapshot_fails_closed_on_missing_asset(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x")
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        critical_source_snapshot(tmp_path)
