"""Offline guard tests and tiny local-remote atomic-push tests; no GitHub writes."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prune", HERE / "prune_branches.py")
prune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prune)
MANIFEST = json.loads((HERE / "cleanup_manifest.json").read_text())


def snapshot(candidates):
    metadata = {"full_name": prune.REPOSITORY, "default_branch": "main", "archived": False, "disabled": False}
    branches = [{"name": c["branch"], "commit": {"sha": c["expected_sha"]}, "protected": False} for c in candidates]
    refs = {c["delete_ref"]: c["expected_sha"] for c in candidates}
    for c in candidates:
        p = c["preservation"]
        refs[p["ref"]] = p["ref_object_sha"]
        if p["ref_object_sha"] != p["peeled_commit_sha"]:
            refs[p["ref"] + "^{}"] = p["peeled_commit_sha"]
    return metadata, branches, [], refs


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.data = copy.deepcopy(MANIFEST)
        self.candidates = self.data["candidates"]
        self.meta, self.branches, self.pulls, self.refs = snapshot(self.candidates)

    def verify(self):
        prune.verify_guards(self.candidates, self.meta, self.branches, self.pulls, self.refs)

    def test_reviewed_38_and_all_guards_pass(self):
        self.assertEqual(len(prune.validate_manifest(self.data, prune.REMOTE)), 38)
        self.verify()

    def test_inherited_git_repository_scope_cannot_redirect_temporary_repository(self):
        inherited = {key: "/unexpected/caller/repository" for key in prune.REPOSITORY_ENV}
        inherited.update(GIT_SSH_COMMAND="ssh -o BatchMode=yes", GH_TOKEN="offline-test-placeholder")
        inherited.update(GIT_CONFIG_KEY_0="core.hooksPath", GIT_CONFIG_VALUE_0="/unexpected/hooks")
        result = SimpleNamespace(returncode=0, stdout="ok", stderr="")
        with patch.dict(os.environ, inherited), patch.object(prune.subprocess, "run", return_value=result) as run:
            self.assertEqual(prune.command(["git", "-C", "/temporary/audit.git", "rev-parse", "--git-dir"]), "ok")
        passed = run.call_args.kwargs["env"]
        self.assertTrue(prune.REPOSITORY_ENV.isdisjoint(passed))
        self.assertNotIn("GIT_CONFIG_KEY_0", passed)
        self.assertNotIn("GIT_CONFIG_VALUE_0", passed)
        self.assertEqual(passed["GIT_SSH_COMMAND"], inherited["GIT_SSH_COMMAND"])
        self.assertEqual(passed["GH_TOKEN"], inherited["GH_TOKEN"])

    def test_wrong_remote_and_authenticated_repository_refused(self):
        for remote in ("https://github.com/other/ChatBubble.git", "file:///tmp/remote.git", "https://evil.test/LUNCY579/ChatBubble.git"):
            with self.subTest(remote=remote), self.assertRaises(prune.Refusal):
                prune.validate_manifest(self.data, remote)
        self.meta["full_name"] = "LUNCY579/Other"
        with self.assertRaises(prune.Refusal): self.verify()

    def test_manifest_cannot_broaden_or_replace_allowlist(self):
        for change in ("remove", "add", "rename", "sha", "keeper", "delete_ref"):
            data = copy.deepcopy(self.data)
            if change == "remove": data["candidates"].pop()
            elif change == "add": data["candidates"].append(copy.deepcopy(data["candidates"][0]))
            elif change == "rename": data["candidates"][0]["branch"] = "main"
            elif change == "sha": data["candidates"][0]["expected_sha"] = "a" * 40
            elif change == "keeper": data["candidates"][0]["preservation"]["ref"] = "refs/heads/main"
            else: data["candidates"][0]["delete_ref"] = "refs/heads/main"
            with self.subTest(change=change), self.assertRaises(prune.Refusal):
                prune.validate_manifest(data, prune.REMOTE)

    def test_missing_or_advanced_candidate_refused(self):
        ref = self.candidates[0]["delete_ref"]
        self.refs.pop(ref)
        with self.assertRaises(prune.Refusal): self.verify()
        self.refs[ref] = "a" * 40
        with self.assertRaises(prune.Refusal): self.verify()
        self.refs[ref] = self.candidates[0]["expected_sha"]
        self.branches[0]["commit"]["sha"] = "b" * 40
        with self.assertRaises(prune.Refusal): self.verify()

    def test_missing_changed_or_wrongly_peeled_keeper_refused(self):
        p = self.candidates[0]["preservation"]
        for key, value in ((p["ref"], None), (p["ref"], "b" * 40), (p["ref"] + "^{}", "c" * 40)):
            saved = self.refs.copy()
            if value is None: self.refs.pop(key)
            else: self.refs[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(prune.Refusal): self.verify()
            self.refs = saved

    def test_protected_unknown_and_default_branch_refused(self):
        for protection in (True, None):
            self.branches[0]["protected"] = protection
            with self.subTest(protection=protection), self.assertRaises(prune.Refusal): self.verify()
        self.branches[0]["protected"] = False
        self.meta["default_branch"] = self.candidates[0]["branch"]
        with self.assertRaises(prune.Refusal): self.verify()

    def test_open_pr_head_or_base_refused(self):
        name = self.candidates[0]["branch"]
        for side in ("head", "base"):
            self.pulls[:] = [{"state": "open", "number": 99, side: {"ref": name, "repo": {"full_name": prune.REPOSITORY}}}]
            with self.subTest(side=side), self.assertRaises(prune.Refusal): self.verify()

    def test_postcheck_detects_failed_delete_or_changed_keeper(self):
        after = {r: s for r, s in self.refs.items() if r not in {c['delete_ref'] for c in self.candidates}}
        prune.verify_after(self.candidates, self.refs, after)
        after[self.candidates[0]["delete_ref"]] = self.candidates[0]["expected_sha"]
        with self.assertRaises(prune.Refusal): prune.verify_after(self.candidates, self.refs, after)
        after.pop(self.candidates[0]["delete_ref"])
        after[self.candidates[0]["preservation"]["ref"]] = "f" * 40
        with self.assertRaises(prune.Refusal): prune.verify_after(self.candidates, self.refs, after)

    def test_default_is_dry_run_and_receipt_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "receipt.jsonl"
            with patch.object(prune, "github_snapshot", return_value=(self.meta, self.branches, self.pulls)), \
                 patch.object(prune, "remote_refs", return_value=self.refs), \
                 patch.object(prune, "fetch_and_verify", return_value=Path(tmp) / "audit.git"), \
                 patch.object(prune, "command") as commands:
                self.assertEqual(prune.main(["--receipt", str(receipt)]), 0)
                commands.assert_not_called()
                before = receipt.read_bytes()
                self.assertEqual(prune.main(["--receipt", str(receipt)]), 2)
                self.assertEqual(receipt.read_bytes(), before)
            events = [json.loads(line)["event"] for line in receipt.read_text().splitlines()]
            self.assertEqual(events, ["START", "PREFLIGHT_PASS", "DRY_RUN_PASS"])


@unittest.skipUnless(shutil.which("git"), "git is required for local bare-remote tests")
class LocalAtomicTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.remote = self.root / "remote.git"
        self.client = self.root / "client.git"
        self.git("init", "--bare", str(self.remote))
        self.git("init", "--bare", str(self.client))
        self.old = self.commit("first")
        for name in ("a", "b"):
            self.git("-C", str(self.remote), "update-ref", "refs/heads/" + name, self.old)
        self.git("-C", str(self.remote), "update-ref", "refs/tags/keep", self.old)
        self.git("-C", str(self.client), "remote", "add", "origin", str(self.remote))
        self.git("-C", str(self.client), "fetch", "origin", "refs/heads/a:refs/heads/a", "refs/heads/b:refs/heads/b")
        self.candidates = [{"delete_ref": "refs/heads/" + n, "expected_sha": self.old} for n in ("a", "b")]

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args, input=None):
        env = os.environ.copy()
        env.update(GIT_AUTHOR_NAME="Offline test", GIT_AUTHOR_EMAIL="offline@example.invalid",
                   GIT_COMMITTER_NAME="Offline test", GIT_COMMITTER_EMAIL="offline@example.invalid",
                   GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        r = subprocess.run(["git", *args], input=input, env=env, text=True, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.strip()

    def commit(self, message, parent=None):
        tree = self.git("-C", str(self.remote), "hash-object", "-t", "tree", "-w", "--stdin", input="")
        args = ["-C", str(self.remote), "commit-tree", tree, "-m", message]
        if parent: args += ["-p", parent]
        return self.git(*args)

    def test_moved_head_lease_rejects_whole_atomic_delete(self):
        moved = self.commit("concurrent update", self.old)
        self.git("-C", str(self.remote), "update-ref", "refs/heads/a", moved, self.old)
        with self.assertRaises(prune.Refusal):
            prune.command(prune.deletion_command(self.candidates, self.client))
        self.assertEqual(self.git("-C", str(self.remote), "rev-parse", "refs/heads/a"), moved)
        self.assertEqual(self.git("-C", str(self.remote), "rev-parse", "refs/heads/b"), self.old)
        self.assertEqual(self.git("-C", str(self.remote), "rev-parse", "refs/tags/keep"), self.old)

    def test_success_deletes_only_requested_heads_and_preserves_tag(self):
        prune.command(prune.deletion_command(self.candidates, self.client))
        refs = self.git("-C", str(self.remote), "for-each-ref", "--format=%(refname)")
        self.assertEqual(refs, "refs/tags/keep")


if __name__ == "__main__":
    unittest.main(verbosity=2)
