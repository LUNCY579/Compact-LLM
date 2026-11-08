"""Freeze an auditable E1 input/schedule plan; this command cannot launch training."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
import time
from . import e1_data,e1_schedule,e1_gates


def write(path,value):
    with path.open("xb") as f:f.write(e1_data.canonical(value)+b"\n")


def validate_config(cfg):
    r=e1_data.require
    fixed={"schema_version":1,"status":"FROZEN_E1_PLAN_ONLY","amendment":"AMENDMENT_005.md",
        "seed":101,"native_execution":False,
        "optimizer":{"name":"AdamW","learning_rate":0.0001,"betas":[0.9,0.999],
            "epsilon":1e-8,"weight_decay":0,"max_gradient_norm":1.0},
        "lora":{"rank":8,"alpha":16,"dropout":0,"targets":["q_proj","v_proj"],
            "base_dtype":"bfloat16","adapter_dtype":"float32","base_frozen":True},
        "sequence":{"max_total_tokens":1024,"max_completion_tokens":128,
            "microbatch":1,"gradient_accumulation":20,"acquisition_per_step":16,"replay_per_step":4},
        "schedule":{"steps":256,"warmup_steps":13,"formula":"u<=13:peak*u/13;else:peak*(256-u)/243",
            "checkpoint_steps":[0,64,128,192,256],"selection":"final_step_256"},
        "budget":{"training_token_ceiling":5242880,"generation_token_ceiling":262144,
            "teacher_call_ceiling":0,"wall_seconds":86400},
        "loss":"mean completion-token CE per sequence, then mean of 20 sequences",
        "resource_minima":{"host_free_gib":7.4,"worker_free_gib":6.0,"gpu_free_gib":4.0,"disk_free_gib":4.0},
        "gate_semantics":"diagnostics only; no native executor or launch authorization",
        "train_token_files":e1_data.TRAIN_FILES}
    variable={"candidate_dose_sha256","primary_receipt_sha256","tokenizer_receipt_sha256",
              "component_receipt_sha256","language_class_quotas"}
    r(type(cfg) is dict and set(cfg)==set(fixed)|variable,"config field inventory")
    for field,value in fixed.items():
        # JSON identity also distinguishes boolean/integer and integer/float substitutions.
        r(e1_data.canonical(cfg[field])==e1_data.canonical(value),field+" drift")
    for field in ("candidate_dose_sha256","primary_receipt_sha256","tokenizer_receipt_sha256"):
        r(type(cfg[field]) is str and re.fullmatch("[0-9a-f]{64}",cfg[field]),field+" digest")
    r(type(cfg["component_receipt_sha256"]) is dict and set(cfg["component_receipt_sha256"])==set(e1_data.COUNTS),"receipt inventory")
    for value in cfg["component_receipt_sha256"].values():
        r(type(value) is str and re.fullmatch("[0-9a-f]{64}",value),"component digest")
    quotas=cfg["language_class_quotas"]
    r(type(quotas) is dict and len(quotas)==19 and all(type(k) is str and k and type(v) is int and v>0 for k,v in quotas.items())
      and sum(quotas.values())==256,"language quotas")


def prepare(config_path,output):
    started=time.perf_counter();e1_data.require(not output.exists(),"write-once output exists")
    package=Path(__file__).resolve().parents[1];root=package.parents[1]
    code_commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    cfg,config_identity=e1_data.read_json(config_path);validate_config(cfg)
    files=[Path(__file__),Path(e1_data.__file__),Path(e1_schedule.__file__),Path(e1_gates.__file__),config_path]
    source_files={}
    for path in files:
        rel=path.resolve().relative_to(root).as_posix()
        subprocess.run(["git","-C",str(root),"ls-files","--error-unmatch","--",rel],check=True,capture_output=True)
        dirty=subprocess.check_output(["git","-C",str(root),"status","--porcelain","--",rel],text=True)
        e1_data.require(not dirty.strip(),"planning source/config uncommitted: "+rel)
        source_files[rel]=e1_data.identity(path)
    e1_data.require(source_files[config_path.resolve().relative_to(root).as_posix()]==config_identity,"config changed while parsing")
    binding=e1_data.bind(package,cfg)
    identities={"parent_sha256":binding["parent_sha256"],"model_sha256":binding["model_sha256"],
                "data_sha256":binding["data_sha256"],"source_sha256":e1_data.digest(source_files),
                "config_sha256":config_identity["sha256"]}
    plan=e1_schedule.build_plan(binding["rows"],identities=identities,seed=cfg["seed"])
    e1_schedule.validate_plan(plan,binding["rows"],identities=identities)
    gates=e1_gates.preflight(package)
    e1_data.require(gates["native_launch_authorized"] is False,"preparation cannot authorize native launch")
    for rel,expected in source_files.items():
        e1_data.require(e1_data.identity(root/rel)==expected,"planning source/config changed: "+rel)
        dirty=subprocess.check_output(["git","-C",str(root),"status","--porcelain","--",rel],text=True)
        e1_data.require(not dirty.strip(),"planning source/config became uncommitted: "+rel)
    e1_data.require(subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()==code_commit,"source commit changed while planning")
    # Gate diagnostics are separate from the deterministic schedule hash.
    output.mkdir(parents=True)
    write(output/"DATA_BINDING.json",binding);write(output/"PLAN.json",plan);write(output/"GATES.json",gates)
    write(output/"CURSOR_STEP_0.json",e1_schedule.make_resume_cursor(plan,0))
    write(output/"CONFIG.json",cfg)
    receipt={"status":"FROZEN_INPUT_AND_SCHEDULE_PLAN_NO_NATIVE_EXECUTION",
             "recorded_at_utc":datetime.now(timezone.utc).isoformat(),
             "code_commit":code_commit,
             "overall_dirty_status":subprocess.check_output(["git","-C",str(root),"status","--porcelain"],text=True),
             "source_files":source_files,"identities":identities,"plan_sha256":plan["plan_sha256"],
             "rows":len(binding["rows"]),"nonpadding_tokens":sum(r["nonpadding_tokens"] for r in binding["rows"]),
             "supervised_tokens":sum(r["completion_tokens"] for r in binding["rows"]),
             "seed":101,"steps":256,"files":{f.name:e1_data.identity(f) for f in output.iterdir()},
             "native_launch_authorized":False,"model_calls":0,"training_updates":0,
             "wall_seconds":time.perf_counter()-started,
             "remaining":"Native E0, actual response collection/enrollment, qualified language/long-context gates and native executor"}
    write(output/"PLAN_RECEIPT.json",receipt)
    print(json.dumps({k:receipt[k] for k in ("status","rows","nonpadding_tokens","supervised_tokens","steps","native_launch_authorized")}))
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args();prepare(a.config,a.output)
