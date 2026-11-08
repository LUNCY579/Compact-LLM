"""Model-free manifest admission. Never opens a historical protected bank."""
from __future__ import annotations
import hashlib
import re

PROTECTED=('RCI-001','LONG_CONTEXT_DEMONSTRATION_SPENT','UNKNOWN64_SPENT')


def fingerprints(text: str) -> tuple[str,str]:
    if not isinstance(text,str) or not text.strip():
        raise ValueError('nonempty prompt required')
    text=' '.join(text.casefold().split())
    return tuple(hashlib.sha256(s.encode()).hexdigest() for s in (text,re.sub(r'\d+','#',text)))


def admit(rows: list[dict], use: str) -> dict:
    if use not in ('TRAIN','SELECTION','CONFIRMATION') or not rows:
        raise ValueError('explicit use and nonempty bank required')
    ids=set(); parents=set(); exact=set(); blind=set()
    for row in rows:
        required={'id','parent_id','prompt','instrument','allowed_uses','verifier_id'}
        if not required<=row.keys() or not all(isinstance(row[k],str) and row[k] for k in ('id','parent_id','instrument','verifier_id')):
            raise ValueError('incomplete row custody')
        if not isinstance(row['allowed_uses'],list) or any(u not in ('TRAIN','SELECTION','CONFIRMATION','REPORT','SPENT') for u in row['allowed_uses']):
            raise ValueError('explicit allowed-use list required')
        if row['instrument'] in PROTECTED or 'SPENT' in row['allowed_uses'] or use not in row['allowed_uses']:
            raise ValueError('instrument not admitted for this use')
        if row['id'] in ids:
            raise ValueError('duplicate row ID')
        a,b=fingerprints(row['prompt'])
        ids.add(row['id']); parents.add(row['parent_id']); exact.add(a); blind.add(b)
    return {'ids':ids,'parents':parents,'exact':exact,'digit_blinded':blind}


def overlap(train: list[dict], development: list[dict]) -> dict:
    a,b=admit(train,'TRAIN'),admit(development,'SELECTION')
    counts={k:len(a[k]&b[k]) for k in a}
    return {'overlaps':counts,'clean':not any(counts.values()),
            'semantic_disjointness_established':False,'training_authorized':False}


def policy_input(row: dict) -> str:
    """Never serialize the entire verifier/gold/role-bearing row into a prompt."""
    fingerprints(row['prompt'])
    return row['prompt']
