"""Public problem schema, deterministic rendering, and bounded strict parsing."""
from __future__ import annotations
import hashlib
import json
import re
from itertools import combinations
from typing import Any

VERSION = '2.0.0'
SKILLS = ('R','U','S','P')
NAMES = {'R':'reference_binding','U':'state_updates','S':'constrained_selection','P':'precedence_ordering'}
FIELDS = {'R':'lookup','U':'updates','S':'selection','P':'precedence'}
SIGNATURES = tuple(''.join(c) for n in range(1,5) for c in combinations(SKILLS,n))
SEEN_PAIRS = ('RU','US','SP')
TRAIN_SIGNATURES = SKILLS + SEEN_PAIRS
ROLES = ('train','calibration','dev','test','challenge','warmstart')
SYSTEM = ('Solve the self-contained task using only its stated data and rules. Return exactly one JSON object '
          'with the single key "answer". The answer is a nonempty list of distinct item indices satisfying ALL '
          'requirements, or the string "UNKNOWN" only when no such list exists. No explanations, extra keys, '
          'or code fences. Any valid list is accepted, not just one preferred reference.')
INTRO = ('Complete this dispatch task.', 'Find one valid plan for these items.',
         'Return a certificate meeting every requirement below.', 'Resolve this finite planning instance.')
RULES = {
 'R': 'REFERENCE BINDING: Start with the items reached by resolving every requested alias through links. A link maps an alias to another alias or to an integer item index. Resolve until an integer is reached. Repeated resolved items count once. An unresolved alias or a reference cycle means no valid answer exists. Other aliases are distractors.',
 'U': 'STATE UPDATES: Replay operations on ALL initial values, in listed order. ["add",i,x] adds x to item i; ["set",i,x] overwrites it; ["copy",i,j] copies the CURRENT value of j to i; ["swap",i,j] exchanges both CURRENT values. Retain only candidate items whose final value is at least cutoff.',
 'S': 'CONSTRAINED SELECTION: Choose exactly k distinct eligible items whose values sum to target. Use final values when updates are present; otherwise use initial values. There is no preferred subset.',
 'P': 'PRECEDENCE: Order the returned items so u occurs before v for every listed pair [u,v] WHEN BOTH appear in the answer. Precedence pairs with an unselected endpoint impose no requirement. All listed pairs form an acyclic directed graph.'
}

def require(test: bool, message: str) -> None:
    if not test: raise ValueError(message)

def canonical(x: Any) -> str:
    return json.dumps(x,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)

def digest(x: Any) -> str:
    return hashlib.sha256(canonical(x).encode('ascii')).hexdigest()

def filehash(path) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def _pairs(pairs):
    out={}
    for k,v in pairs:
        require(k not in out,'duplicate key')
        out[k]=v
    return out

def _bad_const(x): raise ValueError('nonfinite JSON')

def strict_json(text: str, max_bytes: int=65536):
    require(type(text) is str and len(text)<=max_bytes,'wrong type or excessive characters')
    require(len(text.encode('utf-8',errors='strict'))<=max_bytes,'excessive bytes')
    return json.loads(text,object_pairs_hook=_pairs,parse_constant=_bad_const)

def isint(x, lo, hi): return type(x) is int and lo<=x<=hi

def intlist(x,lo,hi,n=None):
    return type(x) is list and (n is None or len(x)==n) and all(isint(v,lo,hi) for v in x)

def validate(s: dict) -> None:
    require(type(s) is dict and set(s).issuperset({'initial'}) and set(s)<=({'initial'}|set(FIELDS.values())),'invalid problem keys')
    require(intlist(s['initial'],-100,100) and 6<=len(s['initial'])<=15,'invalid initial values')
    n=len(s['initial'])
    require(any(k in s for k in FIELDS.values()),'no active skills')
    if 'lookup' in s:
        q=s['lookup']; require(type(q) is dict and set(q)=={'links','requested'},'lookup keys')
        require(type(q['links']) is dict and 1<=len(q['links'])<=80,'lookup map size')
        for k,v in q['links'].items():
            require(type(k) is str and re.fullmatch(r'[a-z][a-z0-9_]{0,19}',k) is not None,'alias key')
            require(isint(v,0,n-1) or (type(v) is str and re.fullmatch(r'[a-z][a-z0-9_]{0,19}',v) is not None),'alias target')
        require(type(q['requested']) is list and 1<=len(q['requested'])<=n and all(type(k) is str and re.fullmatch(r'[a-z][a-z0-9_]{0,19}',k) is not None for k in q['requested']),'requested aliases')
    if 'updates' in s:
        q=s['updates'];require(type(q) is dict and set(q)=={'operations','cutoff'} and isint(q['cutoff'],-100,100),'update keys/cutoff')
        require(type(q['operations']) is list and 1<=len(q['operations'])<=90,'operation limit')
        for op in q['operations']:
            require(type(op) is list and len(op)==3 and type(op[0]) is str and op[0] in ('add','set','copy','swap') and isint(op[1],0,n-1),'invalid operation')
            require(isint(op[2],-500,500) if op[0] in ('add','set') else isint(op[2],0,n-1),'invalid operation operand')
    if 'selection' in s:
        q=s['selection'];require(type(q) is dict and set(q)=={'k','target'} and isint(q['k'],1,min(5,n)) and isint(q['target'],-50000,50000),'selection keys/limits')
    if 'precedence' in s:
        q=s['precedence'];require(type(q) is list and 1<=len(q)<=n*(n-1)//2,'precedence size')
        require(all(intlist(e,0,n-1,2) and e[0]!=e[1] for e in q) and len({tuple(e) for e in q})==len(q),'precedence edges')
        # Domain contract requires a globally acyclic relation.
        done=set()
        for _ in range(n):
            nxt={i for i in range(n) if i not in done and all(u in done for u,v in q if v==i)}
            done |= nxt
        require(len(done)==n,'cyclic precedence metadata')

def signature(s): return ''.join(k for k in SKILLS if FIELDS[k] in s)

def normalized(s):
    s=json.loads(canonical(s))
    if 'lookup' in s: s['lookup']['requested']=sorted(set(s['lookup']['requested']))
    if 'precedence' in s: s['precedence'].sort()
    return s

def problem_hash(s): return digest(normalized(s))

def partition(parent_hash: str) -> str:
    b=int(parent_hash[:12],16)%1000
    return 'train' if b<650 else 'calibration' if b<700 else 'dev' if b<790 else 'test' if b<880 else 'challenge' if b<960 else 'warmstart'

def render(s: dict, template: int=0) -> list[dict]:
    validate(s);require(isint(template,0,3),'template index')
    sig=signature(s)
    defaults=[]
    if 'R' not in sig: defaults.append('All item indices are initial candidates; no reference lookup is needed.')
    if 'U' not in sig: defaults.append('There are no state updates or cutoff filtering; values are initial values.')
    if 'S' not in sig: defaults.append('Return ALL eligible items, each exactly once; an empty eligible set has no valid nonempty answer.')
    if 'P' not in sig: defaults.append('The order of returned indices does not matter.')
    rules='\n'.join(RULES[k] for k in sig)+'\n'+'\n'.join(defaults)
    # Two actual data renderings, not just different introductions. Both are fully invertible.
    if template%2==0:
        body='SPEC\n'+canonical(s)+'\nEND_SPEC'
    else:
        lines=['DATA_FIELDS']
        for key in ('initial','lookup','updates','selection','precedence'):
            if key in s: lines.append(key+' = '+canonical(s[key]))
        body='\n'.join(lines)+'\nEND_DATA_FIELDS'
    text=INTRO[template]+'\nItem indices are zero-based.\n'+rules+'\n'+body
    return [{'role':'system','content':SYSTEM},{'role':'user','content':text}]

def parse_visible(messages: list[dict]) -> dict:
    require(type(messages) is list and len(messages)==2 and messages[0]=={'role':'system','content':SYSTEM},'policy boundary')
    require(type(messages[1]) is dict and set(messages[1])=={'role','content'} and messages[1]['role']=='user','user boundary')
    text=messages[1]['content'];require(type(text) is str,'prompt type')
    if '\nSPEC\n' in text:
        spec=strict_json(text.split('\nSPEC\n',1)[1].removesuffix('\nEND_SPEC'))
    else:
        block=text.split('\nDATA_FIELDS\n',1)[1].removesuffix('\nEND_DATA_FIELDS')
        spec={}
        for line in block.splitlines():
            key,value=line.split(' = ',1);require(key not in spec,'duplicate field');spec[key]=strict_json(value)
    require(any(messages==render(spec,t) for t in range(4)),'rules or renderer drift')
    return spec
