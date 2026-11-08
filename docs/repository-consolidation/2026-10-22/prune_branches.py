#!/usr/bin/env python3
"""Prune exactly the reviewed ChatBubble 38-ref allowlist; default: read-only.

Requires Python 3.10+, git, and GitHub CLI with an authenticated github.com
account able to read the repository. Native git transport must independently
be authenticated for fetch (and push for --apply); gh login alone may not
configure git authentication. No checkout,
global Git configuration, training job, workflow, tag, or PR is modified.

    python prune_branches.py --receipt dry-run.jsonl
    python prune_branches.py --apply --receipt apply.jsonl
    python prune_branches.py --print-recovery

Receipts are append-only JSON lines in an exclusively created file. --apply
is the only execution switch. A changed audit requires a new reviewed script;
editing the manifest cannot broaden this fixed allowlist.

Git protocol references: https://git-scm.com/docs/git-push
GitHub CLI pagination: https://cli.github.com/manual/gh_api
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
import uuid

REPOSITORY = "LUNCY579/ChatBubble"
REMOTE = "https://github.com/LUNCY579/ChatBubble.git"
ALLOWED_REMOTES = {
    REMOTE, "https://github.com/LUNCY579/ChatBubble",
    "git@github.com:LUNCY579/ChatBubble.git",
    "ssh://git@github.com/LUNCY579/ChatBubble.git",
}
APPROVED_COUNT = 38
APPROVED_DIGEST = "9967a2a37227331e9089ac3726e6d8e942bf5913583f87177fb723309f4c839d"
SHA = re.compile(r"[0-9a-f]{40}\Z")
# Git hooks export these variables; inheriting them can redirect even `git -C`
# into the caller's repository. Keep authentication and transport variables.
REPOSITORY_ENV = {
    "GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_IMPLICIT_WORK_TREE",
    "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_PREFIX", "GIT_NAMESPACE", "GIT_SHALLOW_FILE", "GIT_GRAFT_FILE",
    "GIT_REPLACE_REF_BASE", "GIT_QUARANTINE_PATH", "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT",
}


class Refusal(RuntimeError):
    """A failed guard; no automatic fallback or relaxed retry."""


def stamp():
    return datetime.now(timezone.utc).isoformat()


def command(args, *, cwd=None):
    env = os.environ.copy()
    for key in REPOSITORY_ENV:
        env.pop(key, None)
    for key in list(env):
        if re.fullmatch(r"GIT_CONFIG_(KEY|VALUE)_\d+", key):
            env.pop(key)
    env.update(GIT_TERMINAL_PROMPT="0", GH_PROMPT_DISABLED="1")
    try:
        result = subprocess.run(args, cwd=cwd, env=env, text=True,
                                capture_output=True, check=False)
    except OSError as exc:
        raise Refusal(f"Cannot start {args[0]}: {exc}") from exc
    if result.returncode:
        # Command arguments never contain a token; do not request or print one.
        raise Refusal(f"{args[0]} failed ({result.returncode}): "
                      f"{result.stderr.strip()[-3000:]}")
    return result.stdout


def approved_digest(candidates):
    rows = sorted((c["branch"], c["expected_sha"], c["preservation"]["ref"],
                   c["preservation"]["ref_object_sha"],
                   c["preservation"]["peeled_commit_sha"],
                   c["preservation"]["proof"]) for c in candidates)
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def validate_manifest(data, remote):
    if remote not in ALLOWED_REMOTES:
        raise Refusal("Remote must be exactly the reviewed github.com/LUNCY579/ChatBubble repository.")
    if data.get("repository") != REPOSITORY or data.get("schema_version") != 1:
        raise Refusal("Wrong repository or unsupported manifest schema.")
    candidates = data.get("candidates", [])
    try:
        if len(candidates) != APPROVED_COUNT or approved_digest(candidates) != APPROVED_DIGEST:
            raise Refusal("Manifest does not match the fixed 38-branch reviewed allowlist.")
        refs = {"refs/heads/" + c["branch"] for c in candidates}
        if len(refs) != APPROVED_COUNT:
            raise Refusal("Duplicate branch entries in manifest.")
        for c in candidates:
            p = c["preservation"]
            if c.get("delete_ref") != "refs/heads/" + c["branch"]:
                raise Refusal("Deletion ref does not match branch.")
            if p["ref"] in refs:
                raise Refusal("A preservation ref is also slated for deletion.")
            for value in (c["expected_sha"], p["ref_object_sha"], p["peeled_commit_sha"]):
                if not SHA.fullmatch(value):
                    raise Refusal("Invalid full object SHA.")
    except (KeyError, TypeError, AttributeError) as exc:
        raise Refusal("Malformed manifest.") from exc
    return candidates


def github_snapshot():
    def get(path, paginate=False):
        args = ["gh", "api", "--hostname", "github.com", "--method", "GET"]
        if paginate:
            args += ["--paginate", "--slurp"]
        value = json.loads(command(args + ["repos/" + REPOSITORY + path]))
        return [item for page in value for item in page] if paginate else value
    return (get(""), get("/branches?per_page=100", True),
            get("/pulls?state=open&per_page=100", True))


def remote_refs(remote):
    refs = {}
    for line in command(["git", "ls-remote", "--heads", "--tags", remote]).splitlines():
        sha, ref = line.split("\t", 1)
        if ref in refs or not SHA.fullmatch(sha):
            raise Refusal("Invalid or duplicate remote ref response.")
        refs[ref] = sha
    return refs


def verify_guards(candidates, metadata, branches, pulls, refs):
    if metadata.get("full_name") != REPOSITORY:
        raise Refusal("Authenticated repository identity does not match.")
    if metadata.get("archived") or metadata.get("disabled"):
        raise Refusal("Repository is archived or disabled.")
    default = metadata.get("default_branch")
    if not default:
        raise Refusal("Cannot determine default branch.")
    branch_map = {b["name"]: b for b in branches}
    for c in candidates:
        name, expected, p = c["branch"], c["expected_sha"], c["preservation"]
        branch = branch_map.get(name)
        if not branch or branch.get("commit", {}).get("sha") != expected:
            raise Refusal(f"Branch missing or advanced: {name}")
        if name == default or branch.get("protected") is not False:
            raise Refusal(f"Default, protected, or protection unknown: {name}")
        if refs.get(c["delete_ref"]) != expected:
            raise Refusal(f"Git remote head differs from audit: {name}")
        for pr in pulls:
            if pr.get("state") != "open":
                continue
            head, base = pr.get("head", {}), pr.get("base", {})
            head_repo = (head.get("repo") or {}).get("full_name", REPOSITORY)
            if base.get("ref") == name or (head.get("ref") == name and head_repo == REPOSITORY):
                raise Refusal(f"Branch participates in open PR #{pr.get('number')}: {name}")
        if refs.get(p["ref"]) != p["ref_object_sha"]:
            raise Refusal(f"Preservation ref missing or changed: {p['ref']}")
        peeled = refs.get(p["ref"] + "^{}", refs.get(p["ref"]))
        if peeled != p["peeled_commit_sha"]:
            raise Refusal(f"Preservation target differs: {p['ref']}")


def fetch_and_verify(candidates, remote, directory):
    """Only a new temporary bare repository; full history, no file blobs requested."""
    empty_hooks = directory / "empty-hooks"
    empty_hooks.mkdir()
    bare = directory / "audit.git"
    command(["git", "init", "--bare", "--template=" + str(empty_hooks), str(bare)])
    command(["git", "-C", str(bare), "config", "core.hooksPath", str(empty_hooks)])
    command(["git", "-C", str(bare), "remote", "add", "origin", remote])
    command(["git", "-C", str(bare), "config", "remote.origin.promisor", "true"])
    command(["git", "-C", str(bare), "config", "remote.origin.partialclonefilter", "blob:none"])
    keepers = sorted({c["preservation"]["ref"] for c in candidates})
    keeper_locals = {ref: f"refs/audit/keeper/{i:03d}" for i, ref in enumerate(keepers)}
    specs = [c["delete_ref"] + f":refs/audit/source/{i:03d}" for i, c in enumerate(candidates)]
    specs += [ref + ":" + local for ref, local in keeper_locals.items()]
    command(["git", "-C", str(bare), "fetch", "--no-tags", "--filter=blob:none", "origin", *specs])
    for i, c in enumerate(candidates):
        p = c["preservation"]
        def rev(ref):
            return command(["git", "-C", str(bare), "rev-parse", "--verify", ref]).strip()
        if rev(f"refs/audit/source/{i:03d}") != c["expected_sha"]:
            raise Refusal(f"Source changed during fetch: {c['branch']}")
        local = keeper_locals[p["ref"]]
        if rev(local) != p["ref_object_sha"] or rev(local + "^{commit}") != p["peeled_commit_sha"]:
            raise Refusal(f"Keeper changed during fetch: {p['ref']}")
        command(["git", "-C", str(bare), "merge-base", "--is-ancestor",
                 c["expected_sha"], p["peeled_commit_sha"]])
    return bare


def deletion_command(candidates, bare):
    leases = ["--force-with-lease=" + c["delete_ref"] + ":" + c["expected_sha"] for c in candidates]
    return ["git", "-C", str(bare), "push", "--atomic", "--porcelain", *leases,
            "origin", *[":" + c["delete_ref"] for c in candidates]]


def verify_after(candidates, before, after):
    deleted = {c["delete_ref"] for c in candidates}
    if any(ref in after for ref in deleted):
        raise Refusal("Post-check: one or more requested branch refs still exist.")
    for ref, sha in before.items():
        if ref not in deleted and after.get(ref) != sha:
            raise Refusal(f"Post-check: kept ref changed concurrently: {ref}")


class Receipt:
    def __init__(self, path):
        self.path = path
        self.file = path.open("x", encoding="utf-8")

    def add(self, event, **data):
        self.file.write(json.dumps({"time_utc": stamp(), "event": event, **data}) + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("cleanup_manifest.json"))
    parser.add_argument("--remote", default=REMOTE)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--receipt", type=Path, help="New JSONL file; existing files are never overwritten.")
    parser.add_argument("--print-recovery", action="store_true", help="Print recovery commands; no network or writes.")
    args = parser.parse_args(argv)
    receipt = None
    directory = None
    applied_attempt = False
    verified = False
    try:
        data = json.loads(args.manifest.read_text(encoding="utf-8"))
        candidates = validate_manifest(data, args.remote)
        if args.print_recovery:
            if args.apply:
                raise Refusal("--print-recovery cannot be combined with --apply.")
            print("Only restore a missing branch; never overwrite an existing or advanced ref.")
            for c in candidates:
                # Derive from pinned fields, never execute unpinned manifest prose.
                print("git fetch --no-tags " + args.remote + " " + c["preservation"]["ref"])
                print("git push --force-with-lease=" + c["delete_ref"] + ": " +
                      args.remote + " " + c["expected_sha"] + ":" + c["delete_ref"])
            return 0
        path = args.receipt or Path(tempfile.gettempdir()) / ("chatbubble-prune-" + uuid.uuid4().hex + ".jsonl")
        receipt = Receipt(path)
        receipt.add("START", repository=REPOSITORY, mode="apply" if args.apply else "dry_run",
                    manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    candidate_digest=APPROVED_DIGEST, candidates=candidates)
        first_meta = github_snapshot()
        first_refs = remote_refs(args.remote)
        verify_guards(candidates, *first_meta, first_refs)
        directory = Path(tempfile.mkdtemp(prefix="chatbubble-prune-objects-"))
        bare = fetch_and_verify(candidates, args.remote, directory)
        # Re-read policies and exact refs after object transfer, immediately before push.
        final_meta = github_snapshot()
        before = remote_refs(args.remote)
        verify_guards(candidates, *final_meta, before)
        receipt.add("PREFLIGHT_PASS", count=len(candidates), preserved_refs=len({c['preservation']['ref'] for c in candidates}))
        if not args.apply:
            receipt.add("DRY_RUN_PASS", deletions=0, would_delete=len(candidates))
            print(f"Dry run passed: {len(candidates)} refs; no remote mutation. Use --apply to execute.")
            return 0
        # The leases guard deletion races. Atomic push fails as a whole if any head
        # changed or server policy rejects a deletion; no non-atomic fallback.
        receipt.add("APPLY_ATTEMPT", recovery_object_directory=str(directory), expected_heads={c['branch']:c['expected_sha'] for c in candidates})
        applied_attempt = True
        push_output = command(deletion_command(candidates, bare))
        receipt.add("PUSH_RETURNED_SUCCESS", output=push_output)
        after = remote_refs(args.remote)
        verify_after(candidates, before, after)
        verified = True
        receipt.add("APPLY_VERIFIED", removed=len(candidates),
                    remaining_branches=sum(ref.startswith('refs/heads/') for ref in after))
        print(f"Verified deletion of {len(candidates)} reviewed branches; all observed keeper refs unchanged.")
        return 0
    except (Refusal, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        if receipt:
            receipt.add("STOP", error=str(exc), remote_mutation_attempted=applied_attempt,
                        recovery_object_directory=str(directory) if applied_attempt and directory else None)
        return 2
    finally:
        if directory and (not applied_attempt or verified):
            shutil.rmtree(directory)
        elif directory:
            print(f"Retained recovery object repository: {directory}", file=sys.stderr)
        if receipt:
            receipt.file.close()
            print(f"Receipt: {receipt.path}")


if __name__ == "__main__":
    raise SystemExit(main())
