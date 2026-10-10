"""Read-only Git index versus raw publication file-byte check (stdlib only).

Run from the repository root. Prints a receipt; redirect outside the staged
inputs, then explicitly add that receipt. This does not validate research rows.
"""
import hashlib
import json
from pathlib import Path
import subprocess

PREFIX = "Rineya_Intelligence_Quality_Foundry_v1/"


def records(stream):
    pending = b""
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        batch = (pending + chunk).split(b"\0")
        pending = batch.pop()
        for record in batch:
            if len(record) > 65536:
                raise RuntimeError("oversized index record")
            yield record
        if len(pending) > 65536:
            raise RuntimeError("oversized index record")
    if pending:
        raise RuntimeError("unterminated index record")


def main():
    if subprocess.check_output(["git", "rev-parse", "--show-object-format"]).strip() != b"sha1":
        raise RuntimeError("this checker requires Git SHA-1 object format")
    changed = subprocess.check_output(["git", "diff", "--cached", "--no-renames", "--name-only", "-z"])
    if any(not item.decode("utf-8").startswith(PREFIX) for item in changed.split(b"\0") if item):
        raise RuntimeError("staged change outside publication scope")
    process = subprocess.Popen(["git", "ls-files", "--stage", "-z", "--", PREFIX], stdout=subprocess.PIPE)
    binding = hashlib.sha256()
    content_binding = hashlib.sha256()
    count = total = 0
    try:
        for record in records(process.stdout):
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_id, stage = metadata.split()
            path = Path(raw_path.decode("utf-8"))
            if stage != b"0" or mode not in (b"100644", b"100755"):
                raise RuntimeError("nonregular or conflicted staged input")
            if path.is_symlink():
                raise RuntimeError("symlink publication input")
            before = path.stat()
            digest = hashlib.sha1(b"blob %d\0" % before.st_size)
            raw_digest = hashlib.sha256()
            size = 0
            with path.open("rb") as stream:
                while True:
                    data = stream.read(1024 * 1024)
                    if not data:
                        break
                    digest.update(data)
                    raw_digest.update(data)
                    size += len(data)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise RuntimeError("publication input changed during byte check")
            if size != before.st_size or digest.hexdigest().encode() != object_id:
                raise RuntimeError("raw file bytes differ from staged Git blob: " + str(path))
            binding.update(record + b"\0")
            content_binding.update(raw_path + b"\0" + raw_digest.digest())
            count += 1
            total += size
    finally:
        process.stdout.close()
    if process.wait() != 0:
        raise RuntimeError("Git index enumeration failed")
    final_process = subprocess.Popen(["git", "ls-files", "--stage", "-z", "--", PREFIX], stdout=subprocess.PIPE)
    final_binding = hashlib.sha256()
    for record in records(final_process.stdout):
        final_binding.update(record + b"\0")
    final_process.stdout.close()
    if final_process.wait() != 0 or final_binding.digest() != binding.digest():
        raise RuntimeError("Git index changed during byte check")
    final_changes = subprocess.check_output(["git", "diff", "--cached", "--no-renames", "--name-only", "-z"])
    if final_changes != changed:
        raise RuntimeError("staged change scope changed during byte check")
    print(json.dumps({"verified": True, "scope": "Every currently staged publication file versus its raw bytes",
                      "files_checked": count, "bytes_checked": total, "index_binding_sha256": binding.hexdigest(),
                      "raw_content_binding_sha256": content_binding.hexdigest(), "outside_scope_changes": 0,
                      "limits": "Read-only byte binding, not dataset semantic, native, training or model qualification"}))


if __name__ == "__main__":
    main()
