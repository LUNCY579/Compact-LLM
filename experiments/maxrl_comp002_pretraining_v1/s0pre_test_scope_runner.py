"""Build the TEST scope receipt (amendment section 1) under an audit hook that logs every data-root open.

Spec-only: scope_receipt retains id/parent_id/tag/skills from canonical rows and id from policy rows.
Allowed sealed opens: canonical/test.jsonl and policy/test.jsonl only; anything else sealed is refused.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(r"C:\Users\clclc\BusinessProject\ChatBubble-comp-rs-amendment-1")
DATA = Path(r"C:\Users\clclc\BusinessProject\Rineya-Dataset\data").resolve()
OUT = REPO / "experiments/maxrl_comp002_pretraining_v1/evidence/S0PRE_TEST_SCOPE.json"
LOG = REPO / "experiments/maxrl_comp002_pretraining_v1/evidence/S0PRE_TEST_SCOPE_OPEN_LOG.json"
ALLOWED_SEALED = {"canonical/test.jsonl", "policy/test.jsonl"}
SEALED = {"test.jsonl", "challenge.jsonl", "guard_test.jsonl", "guard_challenge.jsonl"}
opened, refused = {}, []


def hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    try:
        p = Path(os.fsdecode(args[0])).resolve()
    except Exception:
        return
    try:
        rel = p.relative_to(DATA).as_posix()
    except ValueError:
        return
    if p.name.lower().strip(". ") in SEALED and rel not in ALLOWED_SEALED:
        refused.append(rel)
        raise PermissionError("sealed file refused: " + rel)
    opened[rel] = opened.get(rel, 0) + 1


sys.path.insert(0, str(REPO))
os.chdir(REPO)
sys.addaudithook(hook)
from experiments.maxrl_comp002_pretraining_v1 import scope_receipt  # noqa: E402

if OUT.exists() or LOG.exists():
    raise SystemExit("receipt exists (create-only)")
rc = scope_receipt.main(["--data", str(DATA), "--role", "test", "--spec-only-ack", "--out", str(OUT)])
receipt_sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
log = {"experiment": "COMP-RS-001", "stage": "S0PRE_TEST_SCOPE_OPEN_LOG",
       "amendment": "COMP_RS_001_AMENDMENT_1.md section 1: TEST scope receipt before any donor read",
       "data_root": str(DATA), "opened": opened, "refused": refused,
       "allowed_sealed": sorted(ALLOWED_SEALED),
       "fields_retained": ["id", "parent_id", "tag", "skills"], "model_outputs_read": False,
       "oracles_opened": any(k.startswith("oracles/") for k in opened),
       "receipt": OUT.name, "receipt_sha256": receipt_sha, "return_code": rc,
       "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with LOG.open("x", encoding="utf-8", newline="\n") as f:
    json.dump(log, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps({"rc": rc, "opened": opened, "refused": refused, "receipt_sha256": receipt_sha[:16]}))
