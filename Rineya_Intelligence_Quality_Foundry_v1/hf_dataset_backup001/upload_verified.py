"""Upload one fully read-back byte backup to its fresh private HF prefix.

Uses the existing installed Hugging Face client and credentials. No model,
inference, cloud compute, deletion, root README change or repo creation.
"""
import hashlib
import json
from pathlib import Path
import re
import sys
import time

from huggingface_hub import CommitOperationAdd, HfApi
import backup

REPO = "Luncy1978/rineya-crs-001-research"


def digest(path):
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            data = stream.read(1024 * 1024)
            if not data:
                return sha.hexdigest(), size
            sha.update(data)
            size += len(data)


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def main():
    root = Path(__file__).resolve().parent
    captured = (root / "runs/capture001").resolve(strict=True)
    output = root / "upload001"
    output.mkdir(exist_ok=False)
    manifest_path = captured / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    outcome = json.loads((captured / "OUTCOME.json").read_bytes())
    archive = captured / "backup.tar.gz"
    if manifest.get("status") != "VERIFIED_BYTE_BACKUP" or outcome.get("status") != "SUCCESS":
        raise RuntimeError("capture/readback is not successful")
    if (manifest.get("destination") != REPO or manifest.get("claims") != backup.CLAIMS
            or outcome.get("claims") != backup.CLAIMS
            or any(item.get("training_eligible") is not False for item in manifest.get("files", []))):
        raise RuntimeError("destination or training boundary mismatch")
    content_id = manifest.get("content_sha256", "")
    if (not isinstance(content_id, str) or not re.fullmatch(r"[0-9a-f]{64}", content_id)
            or hashlib.sha256(backup.json_bytes(backup._content_descriptor(manifest))).hexdigest() != content_id):
        raise RuntimeError("content descriptor identity mismatch")
    prefix = manifest["namespace_path"]
    if prefix != "rineya-dataset-snapshots/" + manifest["content_sha256"]:
        raise RuntimeError("noncanonical backup namespace")
    if any(outcome.get(key) != manifest.get(key) for key in ("content_sha256", "archive_sha256", "namespace_path")):
        raise RuntimeError("success outcome belongs to a different manifest")
    if digest(archive) != (manifest["archive_sha256"], manifest["archive_size"]):
        raise RuntimeError("archive changed since full readback")
    api = HfApi()
    if api.whoami().get("name", "").casefold() != "luncy1978":
        raise RuntimeError("authenticated account mismatch")
    info = api.repo_info(REPO, repo_type="dataset")
    if info.private is not True:
        raise RuntimeError("target is not confirmed private")
    for entry in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo="", revision=info.sha, recursive=False):
        if entry.path == "rineya-dataset-snapshots":
            for child in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=entry.path, revision=info.sha, recursive=False):
                if child.path == prefix:
                    raise RuntimeError("backup prefix already exists; inspect exact bytes before retry")
    save(output / "START.json", {"repo": REPO, "private": True, "parent_commit": info.sha,
         "prefix": prefix, "archive_sha256": manifest["archive_sha256"], "started_unix": time.time()})
    commit = api.create_commit(REPO, repo_type="dataset", parent_commit=info.sha,
        commit_message="Back up Rineya-Dataset TRAIN declarations and source metadata",
        commit_description="Exact byte backup only. Original roles and false eligibility retained; no row, native, training or learner qualification.",
        num_threads=1, operations=[
            CommitOperationAdd(path_in_repo=prefix + "/backup.tar.gz", path_or_fileobj=str(archive)),
            CommitOperationAdd(path_in_repo=prefix + "/manifest.json", path_or_fileobj=str(manifest_path))])
    save(output / "COMMIT.json", {"repo": REPO, "commit": commit.oid, "url": commit.commit_url,
         "prefix": prefix, "archive_sha256": manifest["archive_sha256"], "finished_unix": time.time(),
         "remote_content_verification": "PENDING_SEPARATE_METADATA_CHECK"})
    print(json.dumps({"commit": commit.oid, "url": commit.commit_url, "prefix": prefix}))


if __name__ == "__main__":
    main()
