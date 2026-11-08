"""Exact offset-table serialization beneath the original finalizer owner.

These helpers issue no authority. Raw verifiers still validate every source row;
receipts externalize their already-verified offset tables with exact identities.
"""
from __future__ import annotations

import hashlib
import json

OFFSET_SCHEMA = 'CE8-EXACT-REPLAY-OFFSETS-1'
OFFSET_RECORDS = 400008
OFFSET_LINE_BYTES = 126
OFFSET_BYTES = 64 * 1024 ** 2
JOURNAL_BYTES = 256 * 1024 ** 2
MAX_TABLES = 9
MAX_TABLE_PLANS = 64
FINAL_RECEIPT_BYTES = 1024 ** 2
FINALIZER_JOURNAL_BYTES = 120 * 1024 ** 2
FINALIZER_RECORDS = 80000
FINALIZER_LINE_BYTES = 256 * 1024
FINAL_SUFFIX_RECORDS = 20
FINAL_SUFFIX_JOURNAL_BYTES = 256 * 1024
FINAL_SEAL_BYTES = 64 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')


def hex64(value):
    return type(value) is str and len(value) == 64 and all(ch in '0123456789abcdef' for ch in value)


def offset_plan(rows, journal_ref):
    """Exact rows/bytes/hash before any companion paths or IO are touched."""
    require(type(journal_ref) is dict and set(journal_ref) == {'path','sha256','bytes'}
            and type(journal_ref['path']) is str and hex64(journal_ref['sha256'])
            and type(journal_ref['bytes']) is int and 0 < journal_ref['bytes'] <= JOURNAL_BYTES,
            'exact already-verified original journal reference')
    require(type(rows) is list and 0 < len(rows) <= OFFSET_RECORDS, 'bounded original offset row count')
    end = size = 0
    hashed = hashlib.sha256()
    for index, row in enumerate(rows):
        require(type(row) is dict and set(row) == {'seq','offset','bytes','sha256'}
                and type(row['seq']) is int and row['seq'] == index
                and type(row['offset']) is int and row['offset'] == end
                and type(row['bytes']) is int and 0 < row['bytes'] <= 4*1024**2
                and hex64(row['sha256']), 'exact original offset order/extent/hash')
        end += row['bytes']
        require(end <= journal_ref['bytes'], 'original offsets escape the verified journal extent')
        blob = canonical(row)+b'\n'
        require(len(blob) <= OFFSET_LINE_BYTES, 'fixed126-byte offset row bound')
        size += len(blob)
        require(size <= OFFSET_BYTES, 'bounded exact offset companion')
        hashed.update(blob)
    require(end == journal_ref['bytes'], 'original offsets do not cover the exact verified journal')
    return {'schema':OFFSET_SCHEMA,'records':len(rows),'bytes':size,'sha256':hashed.hexdigest(),
            'original_journal_ref':dict(journal_ref)}


def offset_blocks(rows):
    for row in rows:
        yield canonical(row)+b'\n'


def final_receipt_shape(value):
    """Validate the small final source receipt domain without truncation.

    The large prepared/numeric/offset evidence remains in exact referenced
    artifacts. This final source receipt contains counters, refs and fixed
    status/claim/clock groups, with at most256 scalar leaves and512 nodes.
    The conservative canonical bound is330752B, below the1MiB write cap.
    """
    counts={'leaves':0,'nodes':0}
    def walk(item,depth=0):
        require(depth<=32,'final receipt shape depth')
        counts['nodes']+=1;require(counts['nodes']<=512,'final receipt fixed node count')
        if type(item) is dict:
            for key,entry in item.items():
                require(type(key) is str and len(canonical(key))<=130,'final receipt fixed key extent')
                walk(entry,depth+1)
        elif type(item) is list:
            for entry in item:walk(entry,depth+1)
        else:
            require(item is None or type(item) in (str,int,float,bool),'final receipt builtin scalars')
            require(len(canonical(item))<=1024,'final receipt fixed scalar extent')
            counts['leaves']+=1;require(counts['leaves']<=256,'final receipt fixed scalar count')
    walk(value)
    counts['canonical_bytes_conservative_upper']=256*1024+512*(130+2)+512*2
    require(counts['canonical_bytes_conservative_upper']<FINAL_RECEIPT_BYTES,'fixed receipt domain below final write cap')
    return counts


def compact_replays(value, externalize):
    """Copy receipts while delegating only exact verifier-owned offset lists.

    The callback must be the original live FinalizerCostOwner. This helper is
    merely serialization: it never substitutes for raw replay or issues a pass.
    """
    counter={'bytes':0,'nodes':0}
    def charge(size):
        counter['bytes']+=size;counter['nodes']+=1
        require(counter['bytes']<=4*1024**2 and counter['nodes']<=500000,
                'bounded compact receipt graph before allocation')
    def walk(item,depth):
        require(depth<=64,'bounded finalizer receipt nesting')
        if type(item) is dict:
            charge(2);result={}
            offsets='original_offsets' in item
            if offsets:require('original_journal_ref' in item,'offsets missing their exact original journal reference')
            for key,entry in item.items():
                require(type(key) is str,'builtin receipt keys only')
                if key=='original_offsets':continue
                charge(len(canonical(key))+2);result[key]=walk(entry,depth+1)
            if offsets:
                key='original_offsets_ref';charge(len(canonical(key))+2)
                result[key]=walk(externalize(item['original_offsets'],item['original_journal_ref']),depth+1)
            return result
        if type(item) is list:
            charge(2);result=[]
            for entry in item:
                charge(1);result.append(walk(entry,depth+1))
            return result
        require(item is None or type(item) in (str,int,float,bool),'builtin finalizer receipt values only')
        require(type(item) is not str or len(item)<=4*1024**2,'bounded scalar receipt string')
        charge(len(canonical(item)));return item
    return walk(value,0)
