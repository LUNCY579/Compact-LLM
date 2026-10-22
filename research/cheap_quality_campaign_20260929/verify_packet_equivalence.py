"""Prove unchanged blinded population/rubric before reusing preserved reviews."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/cheap_quality_abstention_v1/runs'
OLD=BASE/'abstention-001/packet'
NEW=BASE/'abstention-002/packet'
files=['annotation.packet.jsonl','REVIEW_INSTRUCTIONS.md','review_manifest.json',
       'response.groups.jsonl','private.join.jsonl']
for directory in ('batches','grouped-batches'):
    old_files={p.name for p in (OLD/directory).iterdir() if p.is_file()}
    new_files={p.name for p in (NEW/directory).iterdir() if p.is_file()}
    assert old_files==new_files, 'review batch inventory changed'
    files.extend(directory+'/'+name for name in sorted(old_files))
comparison={}
for name in files:
    left,right=(OLD/name).read_bytes(),(NEW/name).read_bytes()
    assert left==right, 'prepared semantic/public data changed: '+name
    comparison[name]={'bytes':len(left),'sha256':hashlib.sha256(left).hexdigest(),'identical':True}
old_freeze=json.loads((OLD/'INPUT_FREEZE.json').read_bytes())
new_freeze=json.loads((NEW/'INPUT_FREEZE.json').read_bytes())
assert old_freeze['freeze_id']!=new_freeze['freeze_id'], 'new source was not rebound'
packet_hash=comparison['annotation.packet.jsonl']['sha256']
reviews={}
ids={json.loads(line)['id'] for line in (NEW/'annotation.packet.jsonl').read_text().splitlines()}
assert len(ids)==512
for name in ('review-a.json','review-b.json'):
    path=BASE/'abstention-001'/name
    data=json.loads(path.read_bytes())
    assert data['packet_sha256']==packet_hash
    assert len(data['labels'])==512 and {r['id'] for r in data['labels']}==ids
    reviews[name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'reviewer_id':data['reviewer_id'],'rows':512}
receipt={'status':'PASS_BYTE_IDENTICAL_REBOUND_PACKET','old_freeze_id':old_freeze['freeze_id'],
    'new_freeze_id':new_freeze['freeze_id'],'files':comparison,'preserved_reviews':reviews,
    'annotations_redone':False,'rubric_changed':False,'private_evaluator_join_changed':False,
    'source_change':'documented evaluation integrity hardening only; original freeze/code/packet preserved'}
with (BASE/'abstention-002/PACKET_EQUIVALENCE.json').open('x',encoding='utf-8') as f:
    json.dump(receipt,f,indent=2)
print(json.dumps({'status':receipt['status'],'equivalent_files':len(files),'rows':512,
    'old_freeze_id':receipt['old_freeze_id'],'new_freeze_id':receipt['new_freeze_id']},indent=2))
