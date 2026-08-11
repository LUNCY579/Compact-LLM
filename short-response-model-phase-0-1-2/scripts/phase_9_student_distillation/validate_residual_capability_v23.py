from __future__ import annotations

"""Static validator for both v2.3 residual corpora."""
import argparse, hashlib, json
from pathlib import Path
from typing import Any

ABSTAIN="I don't know based on the available information."


def read_jsonl(p:Path)->list[dict[str,Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""):h.update(c)
    return h.hexdigest()
def check_hashes(root:Path,problems:list[str])->None:
    hp=root/"manifests/file_hashes.json"; expected=json.loads(hp.read_text(encoding="utf-8"))
    for rel,digest in expected.items():
        p=root/rel
        if not p.exists():problems.append(f"missing tracked file: {rel}")
        elif sha256(p)!=digest:problems.append(f"hash mismatch: {rel}")

def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument("--direct-root",type=Path,required=True);ap.add_argument("--rag-root",type=Path,required=True);ap.add_argument("--require-direct-rows",type=int,default=24000);ap.add_argument("--require-rag-rows",type=int,default=7000);a=ap.parse_args();problems=[]
    dm=json.loads((a.direct_root/"manifests/dataset_manifest.json").read_text());rm=json.loads((a.rag_root/"manifests/dataset_manifest.json").read_text())
    direct=read_jsonl(a.direct_root/"canonical.jsonl"); rag=read_jsonl(a.rag_root/"canonical.jsonl")
    if len(direct)<a.require_direct_rows:problems.append(f"direct rows {len(direct)} < {a.require_direct_rows}")
    if len(rag)<a.require_rag_rows:problems.append(f"rag rows {len(rag)} < {a.require_rag_rows}")
    ids=[r.get("id") for r in direct]
    if len(ids)!=len(set(ids)):problems.append("duplicate direct IDs")
    prompts=[]
    for r in direct:
        msgs=r.get("messages") or []
        if not msgs or msgs[-1].get("role")!="assistant":problems.append(f"{r.get('id')}: missing assistant target");continue
        if msgs[-1].get("content")!=r.get("approved_answer"):problems.append(f"{r.get('id')}: target mismatch")
        if any(k in r for k in ("rationale","chain_of_thought","cot")):problems.append(f"{r.get('id')}: visible reasoning field")
        user=next((m.get("content","") for m in msgs if m.get("role")=="user"),"");prompts.append(user)
        if r.get("decision")=="abstain" and r.get("approved_answer")!=ABSTAIN:problems.append(f"{r.get('id')}: noncanonical abstention")
    if len(prompts)!=len(set(prompts)):problems.append("direct user prompts are not exact-string unique")
    fmt={s:set() for s in ("discovery","dev","ood_dev","fresh_sealed")}
    for r in direct:fmt[str(r.get("intended_split"))].add(str(r.get("surface_format")))
    names=list(fmt)
    for i,x in enumerate(names):
        for y in names[i+1:]:
            if fmt[x]&fmt[y]:problems.append(f"surface format leakage {x}<->{y}: {sorted(fmt[x]&fmt[y])}")
    # Topic IDs in RAG may repeat by variant, but must never cross splits.
    topic_splits={}
    for r in rag:
        topic=str(r.get("topic_id"));split=str(r.get("intended_split"));topic_splits.setdefault(topic,set()).add(split)
        if r.get("decision")=="answer" and not r.get("support_document_ids"):problems.append(f"{r.get('id')}: supported RAG row has no support ID")
    leaking=[t for t,s in topic_splits.items() if len(s)>1]
    if leaking:problems.append(f"RAG topic split leakage: {len(leaking)} topics")
    if dm.get("canonical_rows")!=len(direct):problems.append("direct manifest count mismatch")
    if rm.get("canonical_rows")!=len(rag):problems.append("rag manifest count mismatch")
    check_hashes(a.direct_root,problems);check_hashes(a.rag_root,problems)
    out={"status":"PASS" if not problems else "FAIL","direct_rows":len(direct),"rag_rows":len(rag),"direct_domains":len({r.get('domain') for r in direct}),"direct_operations":len({r.get('reasoning_family') for r in direct}),"rag_families":len({r.get('procedure_family') for r in rag}),"problems":problems}
    print(json.dumps(out,indent=2,sort_keys=True))
    if problems:raise SystemExit(2)
if __name__=="__main__":main()
