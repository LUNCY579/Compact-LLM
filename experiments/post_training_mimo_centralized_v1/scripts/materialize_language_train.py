"""Tokenizer-only materialization, checked against the previously measured hashes.

Never import this preparation module into a native model worker: its tokenizer
helper deliberately disables model frameworks. Native consumers read the export.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
from . import tokenize_language_candidates as measured
from . import tokenize_bridge as common


def encode_row(row, expected, tokenizer):
    observed=measured.tokenize_row(row,tokenizer)
    common.require(observed==expected,"previous per-row measurement differs")
    prompt=tokenizer.apply_chat_template(row["messages"][:-1],tokenize=True,**common.TEMPLATE_KWARGS)
    target=tokenizer.encode(row["messages"][-1]["content"],add_special_tokens=False)+[common.EOS_ID]
    encoded=common.assemble_example(prompt,target)
    common.require(common.bridge.sha256(common.bridge.json_bytes(prompt))==expected["prompt_token_ids_sha256"],"prompt token drift")
    common.require(common.bridge.sha256(common.bridge.json_bytes(target))==expected["completion_token_ids_sha256"],"completion token drift")
    return {**encoded,"row_id":row["id"],"parent_id":expected["parent_cluster"],"role":"TRAIN",
            "family":row["metadata"]["reply_class"],"component":"language_replay",
            "source_row_sha256":common.bridge.sha256(common.bridge.json_bytes(row)),
            "input_token_ids_sha256":common.bridge.sha256(common.bridge.json_bytes(encoded["input_ids"])),
            "tokenized_prompt_sha256":expected["prompt_token_ids_sha256"],
            "completion_token_ids_sha256":expected["completion_token_ids_sha256"],
            "completion_eos_id":common.EOS_ID,"training_role_eligible":True,"training_eligibility":False,
            "prior_assistant_turns":expected["prior_assistant_turns"],"prior_turns_supervised":False,
            "reference_review":"assistant-source-review; not human or calibrated grading"}


def build(data, profile_path, tokenizer_receipt_path, output):
    started=time.perf_counter();common.require(not output.exists(),"output exists")
    profile=common.read_json(profile_path)
    common.require(profile["status"]=="MEASURED_LANGUAGE_CANDIDATE_TOKENIZATION_ONLY" and
                   profile["row_count"]==256 and not profile["torch_imported"],"unbound prior profile")
    common.require(profile["script_sha256"]==common.file_identity(Path(measured.__file__))["sha256"],"measurement source drift")
    common.require(profile["helper_sha256"]==common.file_identity(Path(common.__file__))["sha256"],"measurement helper drift")
    source_path=data/"TRAIN_CANDIDATES.jsonl"
    common.require(common.file_identity(source_path)==profile["source_primary_file"],"source candidate drift")
    common.require(common.file_identity(data/"DATA_RECEIPT.json")["sha256"]==profile["candidate_receipt_sha256"],"candidate receipt drift")
    pin=common.read_json(tokenizer_receipt_path);model=Path(pin["checkpoint_path"])
    assets={n:common.file_identity(model/n) for n in common.TOKENIZER_FILES}
    common.require(assets==pin["tokenizer_files"]==profile["tokenizer_assets"],"tokenizer drift")
    common.require(common.file_identity(tokenizer_receipt_path)["sha256"]==profile["tokenizer_receipt_sha256"],"tokenizer receipt drift")
    tokenizer=common.load_tokenizer(model)
    common.require(common.bridge.sha256(tokenizer.chat_template.encode())==profile["chat_template_sha256"],"runtime template drift")
    rows=[json.loads(line,object_pairs_hook=common.bridge._unique_object,parse_constant=common.bridge._reject_constant)
          for line in source_path.read_text(encoding="utf-8").splitlines()]
    expected={r["row_id"]:r for r in profile["rows"]}
    common.require(len(rows)==len(expected)==256 and {r["id"] for r in rows}==set(expected),"row inventory drift")
    encoded=[encode_row(row,expected[row["id"]],tokenizer) for row in rows]
    common.require(sum(r["nonpadding_tokens"] for r in encoded)==profile["distributions"]["nonpadding_tokens"]["sum"],"total token drift")
    common.require(sum(r["completion_tokens"] for r in encoded)==profile["distributions"]["completion_tokens"]["sum"],"supervision drift")
    common.require({n:common.file_identity(model/n) for n in common.TOKENIZER_FILES}==assets,"assets changed")
    output.mkdir(parents=True)
    path=output/"TRAIN.tokenized.jsonl"
    with path.open("xb") as stream:
        for row in encoded:stream.write(common.bridge.json_bytes(row)+b"\n")
    receipt={"status":"MEASURED_LANGUAGE_TRAIN_ARRAYS_NOT_TRAINING_ADMISSION","rows":256,
             "files":{path.name:common.file_identity(path)},"profile_sha256":common.file_identity(profile_path)["sha256"],
             "source_primary_file":profile["source_primary_file"],"source_tokenizer_receipt_sha256":profile["tokenizer_receipt_sha256"],
             "chat_template_sha256":profile["chat_template_sha256"],"tokenizer_assets":assets,
             "script_sha256":common.file_identity(Path(__file__))["sha256"],
             "measurement_script_sha256":profile["script_sha256"],"helper_sha256":profile["helper_sha256"],
             "nonpadding_tokens":sum(r["nonpadding_tokens"] for r in encoded),
             "supervised_tokens":sum(r["completion_tokens"] for r in encoded),"torch_imported":"torch" in sys.modules,
             "model_calls":0,"training_eligibility":False,"e1_launch_ready":False,
             "wall_seconds":time.perf_counter()-started,"CHECK_FINAL_read":False}
    (output/"DATA_RECEIPT.json").write_bytes(common.bridge.json_bytes(receipt)+b"\n")
    print(json.dumps({k:receipt[k] for k in ("status","rows","nonpadding_tokens","supervised_tokens","torch_imported")}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--profile",type=Path,required=True)
    p.add_argument("--tokenizer-receipt",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();build(a.data,a.profile,a.tokenizer_receipt,a.output)
