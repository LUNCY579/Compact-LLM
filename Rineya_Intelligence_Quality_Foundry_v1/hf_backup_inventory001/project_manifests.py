"""Download metadata only into memory; record a bounded identity projection."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers, get_session

OUT = Path(__file__).resolve().parent
remote = json.loads((OUT / "remote_metadata.json").read_text(encoding="utf-8"))
REPO = remote["repo_id"]
REV = remote["revision"]
source_fields = ("content_sha256", "git_head_provenance_only", "included_bytes",
                 "included_files", "largest_file_bytes", "prefix", "root")


def metadata_bytes(path, maximum):
    response = get_session().get(
        hf_hub_url(REPO, path, repo_type="dataset", revision=REV),
        headers=build_hf_headers(), timeout=30,
    )
    response.raise_for_status()
    if len(response.content) > maximum:
        raise RuntimeError("Metadata exceeds permitted projection size")
    return response.content


readme = metadata_bytes("README.md", 262144)
(OUT / "remote_README_at_pin.md").write_bytes(readme)


def project(item):
    raw = metadata_bytes(item["path"], 2097152)
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != item["blob_id"] or len(raw) != item["size"]:
        raise RuntimeError("Pinned manifest identity mismatch")
    manifest = json.loads(raw)
    archive_item = next(value for value in remote["snapshot_immediate_children"]
                        [item["path"].rsplit("/", 1)[0]]
                        if value["path"].endswith("/archive.tar"))
    archive = {key: manifest["archive"].get(key) for key in ("bytes", "name", "sha256")}
    sources = [{key: source.get(key) for key in source_fields}
               for source in manifest.get("sources", [])]
    return {
        "manifest_path": item["path"], "manifest_bytes": len(raw),
        "manifest_git_blob_sha1": blob, "manifest_git_blob_verified": True,
        "content_sha256": manifest.get("content_sha256"),
        "coverage": manifest.get("coverage"),
        "schema_version": manifest.get("schema_version"),
        "repo_id": manifest.get("repo_id"), "repo_type": manifest.get("repo_type"),
        "visibility_required": manifest.get("visibility_required"),
        "training_eligible_recorded": manifest.get("training_eligible"),
        "archive": archive,
        "manifest_archive_hash_matches_hub_lfs": archive["sha256"] == archive_item["lfs"]["sha256"],
        "manifest_archive_bytes_matches_hub_lfs": archive["bytes"] == archive_item["lfs"]["size"],
        "sources": sources,
        "inventory_file_count": len(manifest.get("files", [])),
    }


manifest_items = [item for children in remote["snapshot_immediate_children"].values()
                  for item in children if item["path"].endswith("/manifest.json")]
with ThreadPoolExecutor(max_workers=4) as pool:
    projections = list(pool.map(project, manifest_items))
result = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
          "repo_id": REPO, "revision": REV,
          "metadata_files_read": ["README.md"] + [x["path"] for x in manifest_items],
          "payload_files_opened": 0, "payload_files_downloaded": 0,
          "manifest_projections": projections}
(OUT / "manifest_identity_projection.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"revision": REV, "manifest_count": len(projections),
    "all_manifest_git_blobs_verified": all(x["manifest_git_blob_verified"] for x in projections),
    "all_archive_metadata_matches": all(x["manifest_archive_hash_matches_hub_lfs"] and
        x["manifest_archive_bytes_matches_hub_lfs"] for x in projections),
    "distinct_source_roots": sorted({source["root"] for x in projections for source in x["sources"]}),
    "last_lexicographic_listing_projection_not_chronology": projections[-1]}, ensure_ascii=False, indent=2))
