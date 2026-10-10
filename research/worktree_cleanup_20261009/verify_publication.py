"""Verify the committed record set; no network or research runtime required."""
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "PUBLICATION_MANIFEST.json").read_text(encoding="utf-8"))
for item in manifest["files"]:
    path = ROOT / item["relative_path"]
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    assert path.stat().st_size == item["bytes"], item["relative_path"]
    assert h.hexdigest() == item["sha256"], item["relative_path"]
print("PASS:", len(manifest["files"]), "committed publication files verified")
