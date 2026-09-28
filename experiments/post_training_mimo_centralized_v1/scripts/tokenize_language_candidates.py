"""Profile only fixed TRAIN language primaries; only last assistant is supervised."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
from . import tokenize_bridge as common


def tokenize_row(row,tokenizer):
    common.require(row.get("metadata",{}).get("split")=="train","TRAIN only")
    messages=row["messages"]
    common.require(type(messages) is list and len(messages)>=2,"conversation missing")
    common.require(all(type(m) is dict and set(m)=={"role","content"} and type(m["content"]) is str
                   and m["content"] for m in messages),"conversation schema")
    start=1 if messages[0]["role"]=="system" else 0
    common.require(all(m["role"]==("user" if i%2==0 else "assistant")
                       for i,m in enumerate(messages[start:])),"conversation roles")
    common.require(messages[-1]["role"]=="assistant" and messages[-2]["role"]=="user","final target boundary")
    common.require(tokenizer.eos_token_id==common.EOS_ID,"EOS drift")
    prompt=tokenizer.apply_chat_template(messages[:-1],tokenize=False,**common.TEMPLATE_KWARGS)
    prompt_ids=tokenizer.apply_chat_template(messages[:-1],tokenize=True,**common.TEMPLATE_KWARGS)
    common.require(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),"generation prefix drift")
    common.require(prompt_ids==tokenizer.encode(prompt,add_special_tokens=False),"prompt IDs disagree")
    target=messages[-1]["content"]
    target_ids=tokenizer.encode(target,add_special_tokens=False)
    common.require(common.EOS_ID not in target_ids,"embedded target EOS")
    completion=target_ids+[common.EOS_ID]
    common.require(tokenizer.decode(completion,skip_special_tokens=False)==target+tokenizer.eos_token,"target roundtrip")
    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
    common.require(rendered==prompt+target+tokenizer.eos_token+"\n","full template boundary")
    full=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,enable_thinking=False)
    common.require(full==tokenizer.encode(rendered,add_special_tokens=False),"full IDs disagree")
    common.require(full[:len(prompt_ids)]==prompt_ids and
                   full[len(prompt_ids):]==completion+tokenizer.encode("\n",add_special_tokens=False),"completion prefix drift")
    encoded=common.assemble_example(prompt_ids,completion)
    return {"row_id":row["id"],"parent_cluster":"v2.0:"+row["metadata"]["base_template_family"],
            "reply_class":row["metadata"]["reply_class"],"prior_assistant_turns":sum(m["role"]=="assistant" for m in messages[:-1]),
            "prompt_tokens":encoded["prompt_tokens"],"completion_tokens":encoded["completion_tokens"],
            "nonpadding_tokens":encoded["nonpadding_tokens"],"prompt_token_ids_sha256":common.bridge.sha256(common.bridge.json_bytes(prompt_ids)),
            "completion_token_ids_sha256":common.bridge.sha256(common.bridge.json_bytes(completion)),
            "all_prior_turns_masked":all(x==-100 for x in encoded["labels"][:len(prompt_ids)]),
            "supervised_positions":sum(x!=-100 for x in encoded["labels"]),"training_eligibility":False}


def profile(data,tokenizer_receipt,output):
    started=time.perf_counter();common.require(not output.exists(),"output exists")
    receipt=common.read_json(data/"DATA_RECEIPT.json")
    common.require(receipt["status"]=="MEASURED_TRAIN_LANGUAGE_CANDIDATES_REVIEW_PENDING","unbound candidate status")
    path=data/"TRAIN_CANDIDATES.jsonl"
    common.require(common.file_identity(path)==receipt["files"][path.name],"candidate drift")
    rows=[json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
    common.require(len(rows)==receipt["primary_rows"]==256,"candidate count drift")
    common.require(len({r["id"] for r in rows})==256,"duplicate candidate")
    pin=common.read_json(tokenizer_receipt);model_dir=Path(pin["checkpoint_path"])
    assets={name:common.file_identity(model_dir/name) for name in common.TOKENIZER_FILES}
    common.require(assets==pin["tokenizer_files"],"tokenizer asset drift")
    template=common.read_json(model_dir/"tokenizer_config.json")["chat_template"]
    common.require(common.bridge.sha256(template.encode())==pin["chat_template_sha256"],"template drift")
    tok=common.load_tokenizer(model_dir);common.require(tok.chat_template==template,"runtime template drift")
    measured=[tokenize_row(row,tok) for row in rows]
    common.require({name:common.file_identity(model_dir/name) for name in common.TOKENIZER_FILES}==assets,"tokenizer changed")
    result={"status":"MEASURED_LANGUAGE_CANDIDATE_TOKENIZATION_ONLY","rows":measured,
            "row_count":256,"candidate_receipt_sha256":common.file_identity(data/"DATA_RECEIPT.json")["sha256"],
            "source_primary_file":receipt["files"][path.name],"tokenizer_assets":assets,
            "tokenizer_receipt_sha256":common.file_identity(tokenizer_receipt)["sha256"],
            "chat_template_sha256":pin["chat_template_sha256"],"template_kwargs":common.TEMPLATE_KWARGS,
            "script_sha256":common.file_identity(Path(__file__))["sha256"],
            "helper_sha256":common.file_identity(Path(common.__file__))["sha256"],
            "distributions":{key:common.distribution([r[key] for r in measured]) for key in
                             ("prompt_tokens","completion_tokens","nonpadding_tokens")},
            "masked_prior_assistant_turns":sum(r["prior_assistant_turns"] for r in measured),
            "tokenizer_class":type(tok).__name__,"python_executable":sys.executable,"torch_imported":"torch" in sys.modules,
            "truncation":False,"oversize_rows":0,"semantic_review_qualified":False,"training_eligibility":False,
            "native_qualified":False,"e1_launch_ready":False,"CHECK_FINAL_read":False,
            "cost":{"wall_seconds":time.perf_counter()-started,"model_calls":0,"teacher_calls":0,"training_tokens":0}}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8",newline="\n") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps({"rows":256,"tokens":result["distributions"],"torch_imported":result["torch_imported"]}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--tokenizer-receipt",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args();profile(a.data,a.tokenizer_receipt,a.output)
