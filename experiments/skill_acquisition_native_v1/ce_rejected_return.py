"""Bounded prepaid rejection diagnostics; no execution authority or ML imports.

Typed diagnostics preserve invalid observations. They are never valid measured
wall/IO facts, successful operation returns, clock ends or physical cleanup.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct

from . import ce_operation_contracts as contracts

POLICY = 'CE8-PREPAID-REJECTED-RETURN-POLICY-1'
TICKET = 'CE8-PREPAID-REJECTED-RETURN-TICKET-1'
RECORD = 'CE8-REJECTED-RETURN-1'
DEFAULT = dict(schema=POLICY, record_bytes_upper=65536, encoding_nodes_upper=4096,
    encoding_depth_upper=32, string_bytes_upper=32768, fields_work_bytes_upper=1048576,
    encoding_work_bytes_upper=67108864, header_bytes_upper=4194304, profile_bytes_upper=4194304,
    hash_attempts_upper=8, hash_bytes_upper=33554432, publication_wall_s_upper=10)


def require(condition, message):
    if not condition: raise ValueError(message)
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def default_policy(): return dict(DEFAULT)
def validate_policy(policy):
    require(type(policy) is dict and set(policy) == set(DEFAULT)
            and all(type(policy[key]) is type(value) and policy[key] == value for key, value in DEFAULT.items()),
            'exact fixed prepaid rejected-return policy')
    return policy
def policy_from_spec(spec, owner):
    policy = spec.get('rejected_return_policy')
    if policy is None: return None
    validate_policy(policy)
    require(contracts.manifest_from_spec(spec) is not None, 'rejection policy needs frozen source contracts')
    if owner == 'qualification':
        require(spec.get('operation_profile_rule') == contracts.EXPLICIT_RESERVATION_RULE_V2,
                'qualification rejection policy needs reservation version 2')
    require(owner in ('qualification', 'bootstrap'), 'exact rejection ticket owner')
    return policy
def reservation_costs(policy):
    validate_policy(policy); size = policy['record_bytes_upper']
    return dict(operation_attempts_upper=1, io_read_bytes_upper=0, io_write_bytes_upper=size,
        metadata_bytes_upper=size, identity_hash_attempts_upper=policy['hash_attempts_upper'],
        identity_hash_bytes_upper=policy['hash_bytes_upper'],
        wall_seconds_upper=policy['publication_wall_s_upper'])
def reservation_payload(policy, ledger_schema, header):
    require(len(canonical(header)) <= validate_policy(policy)['header_bytes_upper'], 'bounded rejection owner header')
    return dict(schema=TICKET, policy_sha256=digest(validate_policy(policy)),
        owner=dict(schema=ledger_schema, binding_sha256=digest(header)), bounds=reservation_costs(policy))


class _Encoder:
    def __init__(self, policy):
        self.policy = validate_policy(policy); self.nodes = self.bytes = self.strings = self.work = 0
    def account(self, encoded=0, strings=0, work=0):
        self.bytes += encoded; self.strings += strings; self.work += work
        require(self.bytes <= self.policy['record_bytes_upper'] - 4096
                and self.strings <= self.policy['string_bytes_upper']
                and self.work <= self.policy['fields_work_bytes_upper'], 'bounded rejected-return encoder allowance')
    def string(self, value):
        # Reject before UTF8/JSON allocation. Each later pass stays bounded by
        # aggregate codepoint, string, encoded-byte and analytical work limits.
        require(len(value) <= self.policy['string_bytes_upper'] - self.strings, 'aggregate diagnostic string allowance')
        encoded_length = 2
        for char in value:
            code = ord(char)
            encoded_length += 2 if char in '\\"\b\f\n\r\t' else 6 if code < 32 or 127 <= code <= 65535 else 12 if code > 65535 else 1
            require(self.bytes + encoded_length <= self.policy['record_bytes_upper'] - 4096,
                    'aggregate diagnostic encoded byte allowance')
        size = len(value.encode('utf8'))
        self.account(encoded_length, size, 16*size)
        return value
    def encode(self, value, depth=0):
        self.nodes += 1
        require(self.nodes <= self.policy['encoding_nodes_upper'] and depth <= self.policy['encoding_depth_upper'],
                'bounded diagnostic nodes/depth')
        self.account(80, work=64)
        if value is None: return {'t':'none'}
        if type(value) is bool: return {'t':'bool','v':value}
        if type(value) is int:
            require(value.bit_length() <= 128, 'bounded diagnostic integer encoding')
            return {'t':'int','v':str(value)}
        if type(value) is float: return {'t':'float64','bits':struct.pack('>d',value).hex()}
        if type(value) is str or any(value is token for token in contracts._TOKENS):
            tag = 'source_contract_id' if any(value is token for token in contracts._TOKENS) else 'str'
            return {'t':tag,'v':self.string(str.__str__(value))}
        if type(value) in (list, tuple):
            require(len(value) <= self.policy['encoding_nodes_upper'] - self.nodes, 'bounded diagnostic list length')
            return {'t':'tuple' if type(value) is tuple else 'list', 'v':[self.encode(item,depth+1) for item in value]}
        if type(value) is dict:
            require(len(value) <= self.policy['encoding_nodes_upper'] - self.nodes
                    and all(type(key) is str for key in value), 'bounded string-keyed diagnostic dictionary')
            return {'t':'dict','v':[[self.string(key),self.encode(item,depth+1)] for key,item in value.items()]}
        raise ValueError('unsupported exact diagnostic value type')


def encode_fields(fields, policy):
    require(type(fields) is dict, 'exact observed return field dictionary')
    return _Encoder(policy).encode(fields)
def decode_fields(tree, policy):
    validate_policy(policy); count = [0]
    def decode(node, depth=0):
        count[0] += 1
        require(type(node) is dict and count[0] <= policy['encoding_nodes_upper']
                and depth <= policy['encoding_depth_upper'], 'bounded typed diagnostic tree')
        tag=node.get('t')
        if tag=='none': require(set(node)=={'t'}, 'typed null shape'); return None
        if tag=='bool': require(set(node)=={'t','v'} and type(node['v']) is bool, 'typed bool shape'); return node['v']
        if tag=='int':
            require(set(node)=={'t','v'} and type(node['v']) is str and len(node['v'])<=40, 'typed integer shape')
            value=int(node['v']); require(str(value)==node['v'] and value.bit_length()<=128, 'canonical typed integer'); return value
        if tag=='float64':
            require(set(node)=={'t','bits'} and type(node['bits']) is str and len(node['bits'])==16
                    and not set(node['bits'])-set('0123456789abcdef'), 'exact IEEE754 diagnostic encoding')
            return struct.unpack('>d',bytes.fromhex(node['bits']))[0]
        if tag in ('str','source_contract_id'):
            require(set(node)=={'t','v'} and type(node['v']) is str, 'typed string shape')
            if tag=='source_contract_id': require(node['v'] in contracts.EMITTED_ROLES, 'known source diagnostic identifier')
            return node['v']
        if tag in ('list','tuple'):
            require(set(node)=={'t','v'} and type(node['v']) is list, 'typed sequence shape')
            values=[decode(item,depth+1) for item in node['v']]; return tuple(values) if tag=='tuple' else values
        require(tag=='dict' and set(node)=={'t','v'} and type(node['v']) is list, 'typed dictionary shape')
        result={}
        for pair in node['v']:
            require(type(pair) is list and len(pair)==2 and type(pair[0]) is str and pair[0] not in result, 'unique diagnostic field names')
            result[pair[0]]=decode(pair[1],depth+1)
        return result
    result=decode(tree); require(type(result) is dict, 'typed observed dictionary')
    # Recheck aggregate allocations; do not recreate live source capabilities.
    _Encoder(policy).encode(result)
    require(len(canonical(tree))<=policy['record_bytes_upper']-4096, 'bounded raw diagnostic tree')
    return result


_ACTUAL = dict(bytes_read='read_bytes_upper',known_bytes_read='read_bytes_upper',
    bytes_written='write_bytes_upper',known_bytes_written='write_bytes_upper',
    identity_hash_attempts_actual='identity_hash_attempts_upper',identity_hash_bytes_actual='identity_hash_bytes_upper')
def rejection_codes(tree, profile, expected_role):
    fields=decode_fields(tree,DEFAULT); codes=[]; walls=[]
    for key in ('wall_s','wall_seconds'):
        if key not in fields: continue
        value=fields[key]; walls.append(value)
        if type(value) not in (int,float) or not math.isfinite(value) or value<0: codes.append('invalid:'+key)
        elif value>profile['wall_s_upper']: codes.append('exceeded:'+key)
    if len(walls)==2 and walls[0]!=walls[1]: codes.append('conflicting:wall_aliases')
    for key,upper in _ACTUAL.items():
        if key in fields:
            value=fields[key]
            if type(value) is not int or value<0: codes.append('invalid:'+key)
            elif value>profile[upper]: codes.append('exceeded:'+key)
    if fields.get('wall_contract')!=expected_role: codes.append('mismatched:wall_contract')
    elif expected_role is not None:
        role_node=next(node for key,node in tree['v'] if key=='wall_contract')
        if role_node['t']!='source_contract_id': codes.append('invalid:live_source_contract')
    return sorted(set(codes))


def make_terminal_payload(policy,ticket,start_row,profile,operation_binding,kind,operation_id,phase,fields,error):
    require(len(canonical(profile)) <= validate_policy(policy)['profile_bytes_upper'], 'bounded rejection original profile')
    expected=start_row['payload']['fields'].get('wall_contract')
    try:
        tree=encode_fields(fields,policy); codes=rejection_codes(tree,profile,expected)
        require(codes, 'diagnostic record requires an actually rejected observation')
        status='EXACT_TYPED'
    except (ValueError,TypeError,UnicodeError,OverflowError):
        tree=None; codes=['unknown:bounded_diagnostic_encoding']; status='UNREPRESENTABLE_UNKNOWN'
    payload=dict(schema=RECORD,ticket_sha256=digest(ticket),operation=kind,operation_id=operation_id,
        attempted_phase=phase,started_seq=start_row['seq'],started_sha256=start_row['sha256'],
        selected_profile_sha256=digest(profile),operation_binding=operation_binding,
        observation_status=status,observed_fields=tree,
        rejection=dict(type=type(error).__name__,message=str(error)[:1000],codes=codes))
    require(len(canonical(payload))<=policy['record_bytes_upper']-2048, 'bounded rejection payload with row framing allowance')
    return payload
def validate_terminal_payload(payload,policy,ticket,start_row,profile,operation_binding):
    require(len(canonical(profile)) <= validate_policy(policy)['profile_bytes_upper'], 'bounded rejection replay profile')
    require(type(payload) is dict and set(payload)=={'schema','ticket_sha256','operation','operation_id','attempted_phase',
        'started_seq','started_sha256','selected_profile_sha256','operation_binding','observation_status',
        'observed_fields','rejection'}, 'exact rejection record fields')
    original=start_row['payload']; kind=original.get('kind',original.get('operation')); identifier=original.get('id',original.get('operation_id'))
    require(payload['schema']==RECORD and payload['ticket_sha256']==digest(ticket)
            and payload['operation']==kind and payload['operation_id']==identifier
            and payload['attempted_phase'] in ('returned','failed') and payload['started_seq']==start_row['seq']
            and payload['started_sha256']==start_row['sha256'] and payload['selected_profile_sha256']==digest(profile)
            and payload['operation_binding']==operation_binding, 'rejection original start/profile/owner binding drift')
    rejection=payload['rejection']
    if payload['observation_status']=='EXACT_TYPED':
        fields=decode_fields(payload['observed_fields'],policy)
        codes=rejection_codes(payload['observed_fields'],profile,original['fields'].get('wall_contract'))
    else:
        require(payload['observation_status']=='UNREPRESENTABLE_UNKNOWN' and payload['observed_fields'] is None,
                'explicit unknown bounded diagnostic fallback')
        fields=None; codes=['unknown:bounded_diagnostic_encoding']
    require(type(rejection) is dict and set(rejection)=={'type','message','codes'}
            and type(rejection['type']) is str and type(rejection['message']) is str and len(rejection['message'])<=1000
            and codes and rejection['codes']==codes, 'independently derived rejected return diagnostics')
    require(len(canonical(payload))<=policy['record_bytes_upper']-2048, 'bounded retained rejection payload')
    return fields
