"""Optional native-format exports. Never silently truncate or download assets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .corpus import read_jsonl, file_hash
from .specs import require

def parquet(data: Path, out: Path) -> dict:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet export BLOCKED: install pyarrow in the approved environment; canonical and verl-shaped JSONL remain available") from exc
    require(not out.exists(), "new output directory required")
    out.mkdir(parents=True)
    result={"engine":"pyarrow", "version":pa.__version__, "files":{}}
    for path in sorted((data/"verl_jsonl").glob("*.jsonl")):
        rows=list(read_jsonl(path)); target=out/(path.stem+".parquet")
        pq.write_table(pa.Table.from_pylist(rows),target,compression="zstd")
        require(pq.read_table(target).to_pylist() == rows, "Parquet roundtrip changed records")
        result["files"][target.name]={"rows":len(rows),"sha256":file_hash(target)}
    return result

def tokenizer_audit(data: Path, tokenizer_path: Path, out: Path, context: int, response_budget: int) -> dict:
    require(tokenizer_path.is_dir() and context > response_budget > 0, "invalid local tokenizer/budget")
    require(not out.exists(), "new receipt required")
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_path),local_files_only=True,trust_remote_code=False)
    require(tokenizer.eos_token_id is not None, "missing EOS")
    lengths=[]; over=[]
    for path in sorted((data/"canonical").glob("*.jsonl")):
        for row in read_jsonl(path):
            ids=tokenizer.apply_chat_template(row["prompt"],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            require(type(ids) is list and ids, "unexpected tokenizer output")
            lengths.append(len(ids))
            if len(ids)+response_budget > context: over.append(row["id"])
    assets={p.name:file_hash(p) for p in tokenizer_path.iterdir() if p.is_file() and p.suffix in (".json",".model",".txt",".jinja")}
    receipt={"status":"PASS_LENGTH_ONLY" if not over else "BLOCKED_OVERLENGTH", "tokenizer_assets":assets,
             "eos_token_id":tokenizer.eos_token_id,"context":context,"response_budget":response_budget,
             "rows":len(lengths),"max_prompt_tokens":max(lengths),"overlength_ids":over,"truncation":False,
             "scope":"tokenizer length only; NOT native EOS generation, loss-mask parity, training readiness, or SFT target token qualification"}
    out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("parquet"); q.add_argument("--data",type=Path,required=True); q.add_argument("--out",type=Path,required=True)
    t=sub.add_parser("tokenizer-audit"); t.add_argument("--data",type=Path,required=True); t.add_argument("--tokenizer",type=Path,required=True); t.add_argument("--out",type=Path,required=True); t.add_argument("--context",type=int,default=2048); t.add_argument("--response-budget",type=int,default=256)
    args=p.parse_args()
    try:
        result=parquet(args.data,args.out) if args.command == "parquet" else tokenizer_audit(args.data,args.tokenizer,args.out,args.context,args.response_budget)
    except (RuntimeError,ImportError) as exc: p.exit(2,str(exc)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    if result.get("status","").startswith("BLOCKED"): raise SystemExit(2)

if __name__ == "__main__": main()
