"""Capture live provenance and a bounded grant from this task's explicit authorization.

No model execution. No operator-supplied numbers or human data review are claimed.
Run once; all records are create-only.
"""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PREPARED = ROOT.parent / "rineya-dataset-retention-records" / "a1-prepared-003"


def write(name, value):
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


api = HfApi()
repos = []
for repo in ("Luncy1978/rineya-published-cores", "Luncy1978/Rineya-Supa-A1.1"):
    info = api.model_info(repo, files_metadata=True)
    repos.append({"id": info.id, "revision": info.sha, "private": info.private,
                  "last_modified": str(info.last_modified),
                  "weights": [{"path": s.rfilename, "bytes": s.size,
                               "sha256": getattr(s.lfs, "sha256", None)}
                              for s in info.siblings if s.rfilename.endswith(".safetensors")]})
write("LIVE_INVENTORY.json", {
    "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "hub": repos, "account_models": [m.id for m in api.list_models(author="Luncy1978")],
    "remote_heads": subprocess.check_output(["git", "ls-remote", "--heads", "origin"], cwd=ROOT, text=True).splitlines(),
    "source_local_weight_sha256": sha(ROOT / "Supa" / "model.safetensors"),
    "source_expected_weight_sha256": "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b",
    "model_calls": 0,
})
for name in ("PLAN.json", "BUILD.json", "DATA_CONTRACT.json", "BUDGET_REQUEST.json"):
    if (OUT / name).exists():
        raise FileExistsError(name)
    shutil.copyfile(PREPARED / name, OUT / name)
authorization = {
    "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "authority": "current user message in this Codex task",
    "user_verbatim": "Fully arrange the experiments and wisely utilize the GPU and CPU to start enhancing the models. I will give you the full permissions to improve and enhance the codebase as long as the models' reasoning abilities can be proven to improve",
    "interpretation": "Authorizes bounded local CPU/GPU research, including fine-tuning and evaluation. It does not establish model efficacy or human review.",
    "numeric_provenance": "Agent-instantiated conservative ceiling under the user's full local-experiment authorization; these are NOT operator-quoted numbers. Separate from every historical grant and budget journal.",
    "scope": "One RRP-004 corrected Forge v2 control from exact Supa 1.7, with fresh in-run qualification and three hash-bound development/preservation evaluations. One bounded retry only after a diagnosed failure, subject to remaining budget. No automatic retries.",
    "exclusions": ["cloud spend", "teacher spend", "RL", "release", "production replacement", "independent-review claim"],
    "human_data_review": "NOT_PERFORMED; machine verification only",
    "quality_claim": False,
}
write("SESSION_AUTHORIZATION.json", authorization)
grant = json.loads((PREPARED / "BUDGET_REQUEST.json").read_text(encoding="utf-8"))
grant.update(decision="APPROVED", training_permission=True,
             approved_by="current-session user: broad local-experiment authority; numeric ceiling instantiated by agent",
             authorization_reference=str(OUT / "SESSION_AUTHORIZATION.json"),
             prepared=str(PREPARED), human_data_review="NOT_PERFORMED",
             quality_claim_permitted=False, release_permitted=False)
write("GRANT.json", grant)
print("Captured live inventory, frozen plan/build/contract, and scoped session grant; no model executed.")
