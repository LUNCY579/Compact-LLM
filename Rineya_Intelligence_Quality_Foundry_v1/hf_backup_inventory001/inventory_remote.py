"""Read-only, pinned Hugging Face dataset directory metadata inventory."""
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

OUT = Path(__file__).resolve().parent
REPO = "Luncy1978/rineya-crs-001-research"
api = HfApi()


def entry(value):
    result = {"type": type(value).__name__, "path": value.path}
    for field in ("size", "blob_id", "tree_id", "lfs", "xet_hash"):
        data = getattr(value, field, None)
        if data is not None:
            result[field] = data.__dict__ if hasattr(data, "__dict__") else data
    return result


info = api.repo_info(REPO, repo_type="dataset", files_metadata=False)
revision = info.sha


def shallow(path):
    return [entry(item) for item in api.list_repo_tree(
        REPO, repo_type="dataset", revision=revision, path_in_repo=path,
        recursive=False, expand=False,
    )]


root = shallow(None)
snapshots = shallow("snapshots")
paths = [item["path"] for item in snapshots if item["type"] == "RepoFolder"]
with ThreadPoolExecutor(max_workers=4) as pool:
    children = dict(zip(paths, pool.map(shallow, paths)))

result = {
    "observed_at_utc": datetime.now(timezone.utc).isoformat(),
    "repo_id": info.id,
    "repo_type": "dataset",
    "private": info.private,
    "gated": info.gated,
    "revision": revision,
    "last_modified_utc": str(info.last_modified),
    "root": root,
    "snapshots": snapshots,
    "snapshot_immediate_children": children,
    "method": "HfApi.repo_info and list_repo_tree(recursive=False); revision pinned",
    "payload_files_opened": 0,
    "payload_files_downloaded": 0,
    "protected_directories_recursed": 0,
}
(OUT / "remote_metadata.json").write_text(
    json.dumps(result, ensure_ascii=False, default=str, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps({"private": info.private, "revision": revision,
    "snapshot_count": len(paths), "snapshot_immediate_children": children},
    ensure_ascii=False, default=str, indent=2))
