import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from experiments.rineya_compute_depth_elasticity_v1.core import digest
from experiments.rineya_compute_depth_elasticity_v1.relabel import Verdict, propose

MODULE = "experiments.rineya_compute_depth_elasticity_v1"


def kwargs():
    return dict(parent_id="train-1", split="train", prompt="2+2", old="3", candidate="4",
                protected_parents={"test-1"}, verifier=lambda p, a: Verdict(a == "4", float(a == "4")),
                verifier_sha256=digest("verifier"), generation_contract_sha256=digest("contract"),
                deployment_contract_sha256=digest("contract"), objective_id="exact-validity")


def test_relabel_replays_both_old_and_candidate_without_mutation():
    calls = []
    def verifier(p, a):
        calls.append((p, a))
        return Verdict(a == "4", float(a == "4"))
    args = kwargs()
    args["verifier"] = verifier
    out = propose(**args)
    assert out["accepted"] and not out["dataset_mutated"]
    assert calls == [("2+2", "3"), ("2+2", "4")]
    assert out["old_sha256"] != out["candidate_sha256"]
    assert args["old"] == "3"


@pytest.mark.parametrize("change", [dict(split="test"), dict(split="dev"), dict(parent_id="test-1"),
                                    dict(deployment_contract_sha256=digest("other")),
                                    dict(verifier_sha256="fake"), dict(candidate="")])
def test_relabel_guards(change):
    args = kwargs()
    args.update(change)
    with pytest.raises(ValueError):
        propose(**args)


def test_validity_required_and_ties_do_not_relabel():
    args = kwargs()
    args["candidate"] = "5"
    assert not propose(**args)["accepted"]
    args.update(old="4", candidate="04", verifier=lambda p, a: Verdict(True, 1.0))
    assert not propose(**args)["accepted"]


def test_higher_objective_accepted_only_for_valid_candidate():
    args = kwargs()
    args.update(old="long", candidate="short", verifier=lambda p, a: Verdict(True, float(a == "short")))
    assert propose(**args)["accepted"]
    args["verifier"] = lambda p, a: Verdict(a == "long", 100.0 if a == "short" else 0.0)
    assert not propose(**args)["accepted"]


def run(command, *args):
    return subprocess.run([sys.executable, "-m", MODULE, command, *map(str, args)],
                          capture_output=True, text=True, check=False)


def test_smoke_cli():
    proc = run("smoke")
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["evidence"] == "SCRIPTED_FIXTURE_ONLY" and out["native_forwards"] == 0
    assert len(out["cells"]) == 16


def test_native_always_blocked_and_output_not_overwritten(tmp_path):
    path = tmp_path / "receipt.json"
    proc = run("native-preflight", "--out", path)
    assert proc.returncode == 2
    assert json.loads(path.read_text())["status"] == "BLOCKED"
    original = path.read_bytes()
    proc = run("smoke", "--out", path)
    assert proc.returncode != 0 and path.read_bytes() == original


def test_cli_does_not_import_torch():
    code = ("import sys; from " + MODULE + ".__main__ import main; "
            "sys.argv=['cde','plan']; main(); assert 'torch' not in sys.modules")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
