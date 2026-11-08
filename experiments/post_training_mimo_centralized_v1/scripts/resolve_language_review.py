"""Apply a frozen source-only review resolution; no target rewriting or model calls."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from . import select_language_replay as source


def resolve(primary,backups,pool,review_rows,replacements,seed):
    pids={c.origin.row_id for c in primary}
    if len(pids)!=len(primary) or set(review_rows)!=pids:
        raise ValueError("primary/review coverage mismatch")
    flagged={key for key,value in review_rows.items() if value=="FLAG_REVIEW"}
    if set(replacements)!=flagged or any(x not in {"FLAG_REVIEW","ACCEPTABLE_REFERENCE"} for x in review_rows.values()):
        raise ValueError("replacements must resolve exactly all flags")
    alternative_ids={c.origin.row_id for c in backups};known={c.origin.row_id:c for c in pool}
    used={c.cluster for c in primary};output=[];decisions=[]
    for old in primary:
        oldid=old.origin.row_id
        if oldid not in replacements:
            output.append(old);continue
        rule=replacements[oldid];new=known.get(rule["new_id"])
        if new is None or new.origin.reply_class!=old.origin.reply_class:
            raise ValueError("unknown candidate or changed class")
        if rule["mode"]=="same_cluster_ranked_alternative":
            if new.cluster!=old.cluster or new.origin.row_id not in alternative_ids:
                raise ValueError("not a frozen same-cluster alternative")
        elif rule["mode"]=="next_unselected_cluster":
            remaining={c.cluster for c in pool if c.origin.reply_class==old.origin.reply_class and c.cluster not in used}
            next_cluster=min(remaining,key=lambda c:(source.cluster_order(seed,old.origin.reply_class,c),c))
            ordered=sorted((c for c in pool if c.cluster==next_cluster),key=lambda c:(source.row_order(seed,c),c.origin.row_id))
            if new!=ordered[0]:raise ValueError("not the next frozen cluster/row")
        else:raise ValueError("unregistered replacement mode")
        output.append(new);decisions.append({"rejected_id":oldid,"selected_id":new.origin.row_id,
           "old_cluster":old.cluster,"new_cluster":new.cluster,"mode":rule["mode"],"reason":rule["reason"]})
    if len({c.cluster for c in output})!=len(output):raise ValueError("duplicate selected cluster")
    if sum(r["mode"]=="next_unselected_cluster" for r in replacements.values())>1:
        raise ValueError("additional cluster budget exceeded")
    return output,decisions


def build(registration_path,output):
    started=time.perf_counter();root=Path(__file__).resolve().parents[3]
    if output.exists():raise FileExistsError("output exists")
    raw=registration_path.read_bytes();reg=source.parse_json(raw)
    if reg["status"]!="FROZEN_SOURCE_REVIEW_RESOLUTION_ONLY":raise ValueError("unfrozen resolution")
    directory=source.resolve_bound_path(root,reg["candidate_directory"])
    old=source.parse_json((directory/"DATA_RECEIPT.json").read_bytes())
    for name,expected in reg["input_hashes"].items():
        path=source.resolve_bound_path(root,name)
        if source.sha256(path.read_bytes())!=expected:raise ValueError("input drift: "+name)
    selection=source.parse_json((directory/"REGISTRATION.json").read_bytes())
    audit=source.parse_json(source.resolve_bound_path(root,selection["source_audit_path"]).read_bytes())
    source.validate_registration(selection,audit)
    originals=source.scan_original(source.resolve_bound_path(root,selection["original_v2_source"]["path"]),selection["original_v2_source"])
    pool,_=source.scan_candidates(source.resolve_bound_path(root,selection["source"]["path"]),selection["source"],originals,audit)
    primary,backups,_=source.select_candidates(pool,selection["class_quotas"],selection["selection_seed"],3)
    def lines(candidates):
        return b"".join(c.raw_line if c.raw_line.endswith(b"\n") else c.raw_line+b"\n" for c in candidates)
    if lines(primary)!=(directory/"TRAIN_CANDIDATES.jsonl").read_bytes():
        raise ValueError("primary regeneration differs")
    review={}
    for path in reg["review_receipts"]:
        for row in source.parse_json(source.resolve_bound_path(root,path).read_bytes())["rows"]:
            if row["id"] in review:raise ValueError("duplicate review")
            review[row["id"]]=row["status"]
    selected,decisions=resolve(primary,backups,pool,review,reg["replacements"],selection["selection_seed"])
    if len(selected)!=256:raise ValueError("quota count changed")
    output.mkdir(parents=True)
    path=output/"TRAIN_CANDIDATES.jsonl";path.write_bytes(lines(selected))
    (output/"REGISTRATION.json").write_bytes(raw)
    lineage=[{"row_id":c.origin.row_id,"cluster":c.cluster,"reply_class":c.origin.reply_class,
              "source_raw_line_sha256":source.sha256(c.raw_line),"source_line":c.line_number,
              "original_v2_canonical_sha256":c.origin.canonical_sha256,"source_record_modified":False,
              "training_eligibility":False} for c in selected]
    (output/"LINEAGE.json").write_bytes(source.json_bytes(lineage)+b"\n")
    files={f.name:{"sha256":source.sha256(f.read_bytes()),"bytes":f.stat().st_size} for f in output.iterdir()}
    receipt={"status":"MEASURED_TRAIN_LANGUAGE_CANDIDATES_REVIEW_PENDING","primary_rows":256,"files":files,
             "recorded_at_utc":datetime.now(timezone.utc).isoformat(),
             "code_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
             "dirty_status":subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True),
             "semantic_review":"ASSISTANT_SOURCE_REVIEW_COMPLETE_NOT_HUMAN_OR_CALIBRATED_GRADING",
             "resolution_registration_sha256":source.sha256(raw),"prior_candidate_receipt":old,
             "script_sha256":source.sha256(Path(__file__).read_bytes()),"selector_dependency_sha256":source.sha256(Path(source.__file__).read_bytes()),
             "decisions":decisions,"unchanged_accepted_primaries":250,"source_records_modified":0,
             "new_teacher_targets":0,"human_approval":False,"native_outcomes_used":False,
             "training_eligibility":False,"e1_launch_ready":False,
             "remaining_gates":["native E0","source-correct enrollment","fresh language DEV/grader calibration","long-context sentinel"],
             "cost":{"wall_seconds":time.perf_counter()-started,"model_calls":0,"teacher_calls":0,"training_tokens":0}}
    (output/"DATA_RECEIPT.json").write_bytes(source.json_bytes(receipt)+b"\n")
    print(json.dumps({"rows":256,"replaced":len(decisions),"changed_source_records":0,"training_eligibility":False}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--registration",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path);a=p.parse_args();build(a.registration,a.output)
