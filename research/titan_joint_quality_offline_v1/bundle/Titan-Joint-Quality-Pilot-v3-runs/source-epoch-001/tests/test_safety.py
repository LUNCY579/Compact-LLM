from pathlib import Path
import subprocess
import sys
import pytest
from titan_joint_quality_v1.common import SOURCE_ROOT, safe_output, new_directory
from titan_joint_quality_v1.source import header

def test_outputs_never_enter_source_or_other_checkout(tmp_path):
    with pytest.raises(ValueError): safe_output(SOURCE_ROOT / "generated")
    repo = tmp_path / "repository"; repo.mkdir(); (repo / ".git").write_text("gitdir: fixture")
    with pytest.raises(ValueError): safe_output(repo / "data")

def test_output_creation_is_exclusive(tmp_path):
    path = tmp_path / "fresh"; new_directory(path)
    with pytest.raises(FileExistsError): new_directory(path)

def test_native_and_confirmation_cli_refuse_without_creating_output():
    for command in ("native-run", "confirmation"):
        result = subprocess.run([sys.executable, "-B", "-m", "titan_joint_quality_v1", command],
            cwd=SOURCE_ROOT, capture_output=True, text=True)
        assert result.returncode == 2 and "REFUSED" in result.stderr

def test_malformed_checkpoint_header_never_deserializes_tensor_payload(tmp_path):
    p = tmp_path / "invalid.safetensors"; p.write_bytes(b"1234")
    with pytest.raises(ValueError): header(p)
