"""Scoped hash-only historical audit; never opens a protected evaluation bank.

Exact chat hashes are comparable with the supplied MaxRL inventory. Semantic
specifications across distinct generator schemas are NOT interchangeable hashes.
This receipt is partial historical coverage, never a universal clean-data grant.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
from zipfile import ZipFile

ARCHIVE_SHA = "79b8171e83388ff0bf406328f90b5d74ab4cb6d94f29d0ba66af559cbd9b75da"
MEMBER = "data/diagnostics/v1_exclusion_hashes.json"
MEMBER_SHA = "05668db3651b935e68733e8e6964f0a4e8a4e73443077016a8d27e05cac46681"
OPEN_ROLES = ("TRAIN", "TRAIN_SELECTION", "DEV")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def require(value,message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(",",":"),allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()


def inventory(raw):
    obj=json.loads(raw)
    require(set(obj)=={"problem_hashes","prompt_hashes","provenance"},"inventory schema")
    for field in ("problem_hashes","prompt_hashes"):
        values=obj[field]
        require(type(values) is list and len(values)==len(set(values)) and
                all(type(v) is str and re.fullmatch("[0-9a-f]{64}",v) for v in values),"invalid fingerprints")
    require(len(obj["problem_hashes"])==obj["provenance"]["canonical_instances"],"inventory count")
    return obj


def readable(root,path,registry):
    resolved=Path(path).resolve()
    relative=resolved.relative_to(root.resolve()).as_posix()
    for entry in registry["entries"]:
        prohibited=entry.get("path","").replace("\\","/")
        require(not ("ANY_READ" in entry["bars"] and relative==prohibited),"protected ANY_READ path")
    require("fresh_sealed" not in relative and "test2" not in Path(relative).name.lower(),"protected bank name")
    return resolved


def policy_messages(row):
    if "messages" in row:
        messages=row["messages"]
    else:
        require(type(row.get("prompt")) is str,"unbound prompt")
        messages=[{"role":"user","content":row["prompt"]}]
    require(type(messages) is list and messages and all(type(m) is dict and set(m)=={"role","content"}
            and m["role"] in ("system","user") and type(m["content"]) is str for m in messages),"policy boundary")
    return messages


def scan_rows(groups,registry,prior):
    role_by_parent={}; text_owners={}; counts=Counter(); errors=[]; comparisons=0
    barred_ids={identifier for e in registry["entries"] if any(x in e["bars"] for x in
                ("ANY_READ","TRAIN","CONFIRMATION")) for identifier in e.get("ids",[])}
    prior_prompts=set(prior["prompt_hashes"])
    for origin,role,rows in groups:
        require(role in OPEN_ROLES,"closed role")
        for row in rows:
            require(row["role"]==role and type(row["parent_id"]) is str,"row role/parent")
            parent=row["parent_id"]
            old=role_by_parent.setdefault(parent,role)
            if old!=role: errors.append({"type":"cross_role_parent","parent_id":parent})
            if parent in barred_ids: errors.append({"type":"known_barred_id","parent_id":parent})
            messages=policy_messages(row); fingerprint=digest(messages)
            if fingerprint in prior_prompts: errors.append({"type":"historical_exact_chat","parent_id":parent})
            normalized=[{**m,"content":" ".join(unicodedata.normalize("NFKC",m["content"]).split())} for m in messages]
            norm=digest(normalized)
            owner=text_owners.setdefault(norm,(origin,role,parent))
            if owner[1]!=role or owner[2]!=parent:
                errors.append({"type":"current_normalized_prompt_collision","parent_id":parent,"other_parent":owner[2]})
            comparisons+=1;counts[origin+":"+role]+=1
    return dict(row_counts=dict(counts),rows_compared=comparisons,known_barred_identifiers=len(barred_ids),
                historical_chat_fingerprints=len(prior_prompts),current_parent_units=len(role_by_parent),
                errors=errors,scoped_exact_checks_pass=not errors)


def audit(root,datasets,archive,output):
    started=time.perf_counter();require(not output.exists(),"write-once output exists")
    qpath=root/"research/reasoning_posttraining_v1/QUARANTINE.json"
    registry=json.loads(qpath.read_bytes())
    require(registry["no_quarantine_is_lifted"] and all(not e["lifted"] for e in registry["entries"]),"quarantine lifted")
    require(sha(archive)==ARCHIVE_SHA,"archive identity mismatch")
    with ZipFile(archive) as z:
        info=z.getinfo(MEMBER);require(info.file_size<8*1024**2,"oversize fingerprint inventory")
        raw=z.read(MEMBER)
    require(hashlib.sha256(raw).hexdigest()==MEMBER_SHA,"inventory identity mismatch")
    prior=inventory(raw);groups=[];bindings={}
    for directory in datasets:
        receipt_path=readable(root,directory/"DATA_RECEIPT.json",registry)
        receipt=json.loads(receipt_path.read_bytes())
        bindings[str(directory)]={"receipt_sha256":sha(receipt_path),"open_files":{}}
        for role in OPEN_ROLES:
            name=role+".jsonl"
            if name not in receipt["files"]:
                continue
            path=readable(root,directory/name,registry)
            ident={"sha256":sha(path),"bytes":path.stat().st_size}
            require(ident==receipt["files"][name],"open dataset drift")
            bindings[str(directory)]["open_files"][name]=ident
            groups.append((str(directory),role,[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]))
    result=scan_rows(groups,registry,prior)
    result.update(schema_version=1,status="PASS_SCOPED_EXACT_AUDIT_ONLY" if result["scoped_exact_checks_pass"] else "BLOCKED_COLLISION",
                  source_code_sha256=sha(__file__),dataset_bindings=bindings,
                  quarantine_file_sha256=sha(qpath),quarantine_entries=len(registry["entries"]),
                  protected_paths_preserved=[e["path"] for e in registry["entries"] if "ANY_READ" in e["bars"]],
                  historical_archive_sha256=ARCHIVE_SHA,historical_member=MEMBER,historical_member_sha256=MEMBER_SHA,
                  provenance=prior["provenance"],historical_problem_hashes_not_compared=len(prior["problem_hashes"]),
                  not_compared_reason="Distinct generator schemas have no qualified common semantic fingerprint",
                  read_scope=["quarantine metadata","one hash-only archive member","manifest-bound TRAIN/TRAIN_SELECTION/DEV"],
                  sealed_payloads_read=False,full_historical_exclusion_qualified=False,e1_launch_ready=False,
                  limitations=["Whole-chat equality does not test user-only equality when system messages differ",
                               "No historical normalized-text inventory was available",
                               "No claim of semantic, all-Titan-ancestry or pretraining exclusion"],
                  wall_seconds=time.perf_counter()-started,model_calls=0,teacher_calls=0)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8",newline="\n") as f:json.dump(result,f,indent=2);f.write("\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,action="append",required=True)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();r=audit(Path(__file__).resolve().parents[3],a.dataset,a.archive,a.output)
    print(json.dumps({k:r[k] for k in ("status","rows_compared","errors","full_historical_exclusion_qualified")}))
