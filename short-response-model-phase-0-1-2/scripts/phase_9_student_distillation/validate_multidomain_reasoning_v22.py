from __future__ import annotations

import argparse, hashlib, json
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_DOMAINS = 27
EXPECTED_ARCHETYPES = 15
EXPECTED_ROWS = 40500
EXPECTED_SPLITS = {"train":32400,"dev":4050,"test":4050}
ABSTAIN = "I don't know based on the available information."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--corpus-root',type=Path,required=True); p.add_argument('--require-rows',type=int,default=40000); a=p.parse_args()
    root=a.corpus_root
    problems=[]
    manifest=json.loads((root/'manifests/dataset_manifest.json').read_text(encoding='utf-8'))
    rows=read_jsonl(root/'canonical.jsonl')
    if len(rows) < a.require_rows: problems.append(f"rows {len(rows)} < required {a.require_rows}")
    if len(rows) != EXPECTED_ROWS: problems.append(f"canonical rows {len(rows)} != {EXPECTED_ROWS}")
    ids=[str(r.get('id')) for r in rows]
    if len(ids)!=len(set(ids)): problems.append('duplicate IDs')
    user_prompts=[]
    for r in rows:
        users=[m.get('content','') for m in (r.get('messages') or []) if m.get('role')=='user']
        user_prompts.append(str(users[-1] if users else ''))
    if len(user_prompts)!=len(set(user_prompts)): problems.append('duplicate canonical user prompts')
    domains={str(r.get('domain')) for r in rows}; families={str(r.get('reasoning_family')) for r in rows}
    if len(domains)!=EXPECTED_DOMAINS: problems.append(f"domains {len(domains)} != {EXPECTED_DOMAINS}")
    if len(families)!=EXPECTED_ARCHETYPES: problems.append(f"archetypes {len(families)} != {EXPECTED_ARCHETYPES}")
    split_ids={}
    for split,n in EXPECTED_SPLITS.items():
        srows=read_jsonl(root/f'reviewed_splits/{split}.reviewed.jsonl'); split_ids[split]={r['id'] for r in srows}
        if len(srows)!=n: problems.append(f"{split} rows {len(srows)} != {n}")
        if any(r.get('intended_split')!=split for r in srows): problems.append(f"{split} contains mismatched intended_split")
    if split_ids['train'] & split_ids['dev'] or split_ids['train'] & split_ids['test'] or split_ids['dev'] & split_ids['test']:
        problems.append('split overlap detected')
    for row in rows:
        msgs=row.get('messages') or []
        if len(msgs)!=3 or msgs[-1].get('role')!='assistant': problems.append(f"{row.get('id')}: invalid message shape"); break
        if str(msgs[-1].get('content') or '') != str(row.get('approved_answer') or ''): problems.append(f"{row.get('id')}: target mismatch"); break
        if len(str(row.get('approved_answer') or '').split()) > 12: problems.append(f"{row.get('id')}: target too long"); break
        if 'rationale' in row or 'chain_of_thought' in row: problems.append(f"{row.get('id')}: visible reasoning field present"); break
        if row.get('decision')=='abstain' and row.get('approved_answer')!=ABSTAIN: problems.append(f"{row.get('id')}: noncanonical abstention"); break
    hashes=json.loads((root/'manifests/file_hashes.json').read_text(encoding='utf-8'))
    for rel,digest in hashes.items():
        path=root/rel
        if not path.exists() or sha256(path)!=digest: problems.append(f"hash mismatch: {rel}")
    view_expected={"track_a_broad_general":15000,"track_b_anchored_mixed":15000,"specialist_general":10000,"specialist_grounding":5500,"specialist_procedure":5500,"specialist_abstention":4500,"specialist_social":4500}
    for name,n in view_expected.items():
        v=read_jsonl(root/f'curriculum_views/{name}.reviewed.jsonl')
        if len(v)!=n: problems.append(f"{name} rows {len(v)} != {n}")
        if any(not (r.get('messages') and r['messages'][-1].get('role')=='assistant') for r in v): problems.append(f"{name}: missing assistant target")
    social_source = root/'curriculum_sources/social_synthetic_train.jsonl'
    if not social_source.exists():
        problems.append('missing social synthetic train source')
    else:
        social_rows = read_jsonl(social_source)
        if len(social_rows) != 3000: problems.append(f"social synthetic rows {len(social_rows)} != 3000")
        if any(r.get('intended_split') != 'train' or r.get('decision') != 'social' for r in social_rows): problems.append('social synthetic source contains non-train/non-social row')
    report={"status":"PASS" if not problems else "FAIL","canonical_rows":len(rows),"domains":len(domains),"archetypes":len(families),"split_counts":dict(Counter(r['intended_split'] for r in rows)),"problems":problems,"manifest_schema":manifest.get('schema_version')}
    print(json.dumps(report,indent=2,sort_keys=True))
    if problems: raise SystemExit(1)

if __name__=='__main__': main()
