"""Read and bind frozen TRAIN token arrays. Stdlib only; no model admission.

This module can be imported by a future native worker without changing framework
environment variables. Tokenizer preparation modules must stay out of that worker.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re

EOS=151645
WEIGHT_SHA="537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7"
COUNTS={"acquisition":4096,"deterministic_replay":768,"language_replay":256}
TRAIN_FILES={"acquisition":"artifacts/mimo_central/bridge_tokenized_v1/TRAIN.tokenized.jsonl",
             "deterministic_replay":"artifacts/mimo_central/retention_tokenized_v1/TRAIN.tokenized.jsonl",
             "language_replay":"artifacts/mimo_central/language_train_arrays_v1/TRAIN.tokenized.jsonl"}
TOKENS={"acquisition":(414554,49272),"deterministic_replay":(243336,11814),"language_replay":(22642,2460)}
RECEIPTS={"acquisition":("receipts/BRIDGE_TOKENIZATION_RECEIPT.json","MEASURED_OPEN_ROLE_TOKENIZATION_ONLY"),
          "deterministic_replay":("receipts/RETENTION_TOKENIZATION_RECEIPT.json","MEASURED_OPEN_RETENTION_ROLE_TOKENIZATION_ONLY"),
          "language_replay":("receipts/LANGUAGE_TRAIN_ARRAYS_RECEIPT.json","MEASURED_LANGUAGE_TRAIN_ARRAYS_NOT_TRAINING_ADMISSION")}


def require(value,message):
    if not value:raise ValueError(message)


def unique(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,"duplicate JSON key")
        result[key]=value
    return result


def parse(raw):
    def bad(value):raise ValueError("nonfinite JSON: "+value)
    return json.loads(raw,object_pairs_hook=unique,parse_constant=bad)


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def identity(path):
    before=path.stat()
    with path.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
    after=path.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),"file changed during hash")
    return {"sha256":sha,"bytes":after.st_size}


def read_json(path):
    """Bind the exact bytes parsed, not a later reopening of the same path."""
    raw=path.read_bytes()
    return parse(raw),{"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)}


def safe_path(root,relative):
    require(type(relative) is str and relative,"missing relative path")
    path=Path(relative)
    require(not path.is_absolute() and ".." not in path.parts,"path escape")
    resolved=(root/path).resolve();require(resolved.is_relative_to(root.resolve()),"resolved path escape")
    parts=path.parts+resolved.relative_to(root.resolve()).parts
    require(not any(re.search(r"(^|[^a-z])(check|final|test2|fresh_sealed)([^a-z]|$)",part.lower()) for part in parts),"closed bank")
    return resolved


def validate_array(row,component):
    require(type(row) is dict and component in COUNTS,"array row schema")
    require(row.get("role")=="TRAIN" and row.get("training_role_eligible") is True,"TRAIN role required")
    for field in ("row_id","parent_id","family"):
        require(type(row.get(field)) is str and row[field].strip()==row[field] and row[field],"row identity")
    p,c,n=row.get("prompt_tokens"),row.get("completion_tokens"),row.get("nonpadding_tokens")
    require(all(type(x) is int for x in (p,c,n)) and p>0 and 0<c<=128 and p+c==n<=1024,"token counts")
    require(type(row.get("padding_tokens")) is int and row["padding_tokens"]==0 and
            type(row.get("completion_eos_id")) is int and row["completion_eos_id"]==EOS,"padding/EOS contract")
    ids=row.get("input_ids");labels=row.get("labels")
    require(type(ids) is list and len(ids)==n and all(type(x) is int and 0<=x<151936 for x in ids),"token IDs")
    require(type(labels) is list and all(type(x) is int for x in labels),"label dtype")
    require(labels==[-100]*p+ids[p:],"completion-only labels")
    require(row.get("attention_mask")==[1]*n and row.get("completion_mask")==[0]*p+[1]*c,"position masks")
    require(all(type(x) is int for x in row["attention_mask"]+row["completion_mask"]),"mask dtype")
    require(ids[-1]==EOS and not set(ids[p:-1]).intersection((151643,151645)),"completion termination")
    for field,value in (("input_token_ids_sha256",ids),("tokenized_prompt_sha256",ids[:p]),("completion_token_ids_sha256",ids[p:])):
        require(row.get(field)==digest(value),field+" drift")
    require(type(row.get("source_row_sha256")) is str and re.fullmatch("[0-9a-f]{64}",row["source_row_sha256"]),"source row hash")
    return {"key":component+":"+row["row_id"],"component":component,"role":"TRAIN",
            "parent_id":row["parent_id"],"nonpadding_tokens":n,"completion_tokens":c}


def validate_population(records,component):
    require(len(records)==COUNTS[component],"component quota")
    require(len({r["row_id"] for r in records})==len(records),"duplicate row")
    groups=defaultdict(list)
    for row in records:groups[row["parent_id"]].append(row)
    if component in ("acquisition","language_replay"):
        require(len(groups)==len(records),"duplicate parent/cluster")
    if component=="acquisition":
        require(Counter(r["family"] for r in records)==Counter(dict.fromkeys(
          ("bounded_integer_arithmetic","symbolic_variable_updates","finite_set_queries","constrained_text_transformations"),1024)),"acquisition family quota")
    elif component=="deterministic_replay":
        require(len(groups)==640,"replay parent count")
        require(Counter(r["family"] for r in records)==Counter(dict.fromkeys(
          ("retrieval","instruction_following","answerability_abstention"),256)),"replay family quota")
        for group in groups.values():
            families={r["family"] for r in group};require(len(families)==1,"parent family drift")
            expected={"answerable","missing_evidence"} if group[0]["family"]=="answerability_abstention" else {"standard"}
            require(len(group)==len(expected) and {r["variant_id"] for r in group}==expected,"replay twin coverage")
    require((sum(r["nonpadding_tokens"] for r in records),sum(r["completion_tokens"] for r in records))==TOKENS[component],"measured component dose changed")


def bind(package_root,config):
    root=package_root.parents[1]
    dose,dose_id=read_json(package_root/"manifests/E1_CANDIDATE_DOSE.json")
    require(dose["model_weight_sha256"]==WEIGHT_SHA and dose["training_sequences"]==5120,"dose identity")
    require(dose["nonpadding_tokens_per_sequence_epoch"]==680532 and dose["supervised_tokens_per_sequence_epoch"]==63546,"dose totals")
    primary,primary_id=read_json(package_root/"receipts/PRIMARY_STUDENT_RECEIPT.json")
    require(primary["weight_sha256"]==WEIGHT_SHA and primary["parameter_count"]==596049920,"primary identity")
    pin,pin_identity=read_json(package_root/"receipts/TOKENIZER_RECEIPT.json");pin_id=pin_identity["sha256"]
    require(dose_id["sha256"]==config["candidate_dose_sha256"],"frozen dose drift")
    require(primary_id["sha256"]==config["primary_receipt_sha256"] and pin_id==config["tokenizer_receipt_sha256"],"frozen source receipt drift")
    require(pin["tokenizer_sha256"]==primary["tokenizer_sha256"] and pin["chat_template_sha256"]==primary["chat_template_sha256"],"primary tokenizer mismatch")
    paths=config["train_token_files"];require(paths==TRAIN_FILES,"fixed TRAIN file inventory")
    rows=[];bindings={"dose":dose_id,"primary":primary_id,"tokenizer":pin_identity};locations={}
    all_ids=set()
    for component in COUNTS:
        receipt_rel,status=RECEIPTS[component];receipt,receipt_id=read_json(package_root/receipt_rel)
        require(receipt_id["sha256"]==config["component_receipt_sha256"][component],"frozen component receipt drift")
        require(receipt["status"]==status,"component receipt status")
        require(receipt["chat_template_sha256"]==primary["chat_template_sha256"],"template identity drift")
        assets=receipt.get("tokenizer_files",receipt.get("tokenizer_assets"));require(assets==pin["tokenizer_files"],"tokenizer assets drift")
        require(receipt.get("tokenizer_receipt_sha256",receipt.get("source_tokenizer_receipt_sha256"))==pin_id,"tokenizer receipt drift")
        if component!="language_replay":
            dose_key=component
            require(receipt_id["sha256"]==dose["components"][dose_key]["receipt_sha256"],"dose component binding drift")
        else:
            require(receipt["profile_sha256"]==dose["components"]["language_replay_candidates"]["receipt_sha256"],"language profile binding drift")
        path=safe_path(root,paths[component]);require(path.name=="TRAIN.tokenized.jsonl","only TRAIN token files")
        expected=receipt["files"][path.name];h=hashlib.sha256();position=0;records=[]
        with path.open("rb") as stream:
            for number,line in enumerate(stream,1):
                h.update(line);record=parse(line);normalized=validate_array(record,component)
                require(record["row_id"] not in all_ids,"row ID reused across components");all_ids.add(record["row_id"])
                rows.append(normalized);records.append({k:record[k] for k in ("row_id","parent_id","family","nonpadding_tokens","completion_tokens")}
                                                        | {"variant_id":record.get("variant_id")})
                locations[normalized["key"]]={"file":paths[component],"line":number,"byte_offset":position,
                    "raw_line_sha256":hashlib.sha256(line).hexdigest(),"source_row_sha256":record["source_row_sha256"],
                    "input_token_ids_sha256":record["input_token_ids_sha256"]}
                position+=len(line)
        observed={"sha256":h.hexdigest(),"bytes":position};require(observed==expected,"TRAIN file hash/size drift")
        validate_population(records,component)
        if component=="language_replay":
            require(Counter(r["family"] for r in records)==Counter(config["language_class_quotas"]),"language class quota drift")
        bindings[component]={"receipt":receipt_id,"file":paths[component],"token_file":observed}
    require(sum(r["nonpadding_tokens"] for r in rows)==680532 and sum(r["completion_tokens"] for r in rows)==63546,"full dose mismatch")
    data_sha=digest({"rows":rows,"bindings":bindings,"locations":locations})
    return {"status":"BOUND_TRAIN_INPUTS_NOT_TRAINING_ADMISSION","rows":rows,"bindings":bindings,
            "locations":locations,"data_sha256":data_sha,"parent_sha256":WEIGHT_SHA,
            "model_sha256":bindings["primary"]["sha256"],"native_qualified":False,"training_authorized":False}
