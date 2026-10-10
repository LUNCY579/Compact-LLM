"""Exclusive local dataset-cycle ownership. Never manages any model process."""
import argparse
from datetime import datetime, timezone
import json
import os
import secrets
from common import ROOT, strict_json, write_json

LOCK = ROOT / "state/cycle.lock.json"

def acquire(owner):
    token = secrets.token_hex(24)
    payload = {"owner": owner, "token": token, "pid": os.getpid(),
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "dataset cycle; pid is claimant helper, not persistent worker",
               "stale_policy": "never automatically delete; verify cycle owner ended"}
    try:
        write_json(LOCK, payload)
    except FileExistsError:
        raise SystemExit("Another dataset cycle owns the lock; exit without work.")
    return payload

def release(token):
    payload = strict_json(LOCK.read_text(encoding="utf-8"))
    if not secrets.compare_digest(payload["token"], token):
        raise ValueError("not the cycle lock owner")
    LOCK.unlink()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("acquire"); a.add_argument("--owner", required=True)
    r = sub.add_parser("release"); r.add_argument("--token", required=True)
    args = p.parse_args()
    if args.command == "acquire":
        print(json.dumps(acquire(args.owner)))
    else:
        release(args.token)
