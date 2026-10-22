"""Isolated candidate source contracts; no native activation or authority issuer.

Live metadata needs source token identity before JSON cloning. Raw replay can
price serialized identifiers only through an explicit replay argument and the
exact frozen manifest. Inclusive measured wall intervals remain unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from types import MappingProxyType

CONTRACT_RULE = 'CE8-SOURCE-OPERATION-CONTRACTS-1'
EXPLICIT_RESERVATION_RULE_V2 = 'CE8-QUALIFICATION-EXPLICIT-RESERVATIONS-2'
_SEAL = object()


def require(condition, message):
    if not condition: raise ValueError(message)
def _integer(value, minimum=0): return type(value) is int and value >= minimum
def _finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0 and (not positive or value > 0)
def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


class _SourceContract(str):
    __slots__ = ()
    def __new__(cls, value, *, seal):
        require(seal is _SEAL, 'operation contract token constructor is closed')
        return str.__new__(cls, value)
    def __copy__(self): return self
    def __deepcopy__(self, memo): return self


FILE_HASH_LEAF = _SourceContract('file_hash_leaf', seal=_SEAL)
BACKEND_REF_FRAME = _SourceContract('backend_ref_frame', seal=_SEAL)
TEACHER_ROW = _SourceContract('teacher_row', seal=_SEAL)
PREPARED_ROW = _SourceContract('prepared_row', seal=_SEAL)
CP_FROZEN_HASH = _SourceContract('cp_frozen_hash', seal=_SEAL)
CP_FILE_HASH = _SourceContract('cp_file_hash', seal=_SEAL)
CP_DATA_HASH = _SourceContract('cp_data_hash', seal=_SEAL)
CP_PUBLICATION_HASH = _SourceContract('cp_publication_hash', seal=_SEAL)
CP_TENSOR_PAIR_HASH = _SourceContract('cp_tensor_pair_hash', seal=_SEAL)
CP_SAVE_FRESH = _SourceContract('cp_save_fresh', seal=_SEAL)
CP_RESTORE = _SourceContract('cp_restore', seal=_SEAL)
CP_SAVE_ADAPTER = _SourceContract('cp_save_adapter', seal=_SEAL)
CP_RESTORE_ADAPTER = _SourceContract('cp_restore_adapter', seal=_SEAL)
LEASE_PARENT_CHECK = _SourceContract('lease_parent_check', seal=_SEAL)
MODEL_FINGERPRINT = _SourceContract('model_fingerprint', seal=_SEAL)
OPAQUE_INPUT_FRAME = _SourceContract('opaque_input_frame', seal=_SEAL)
OPAQUE_TOKENIZER_FRAME = _SourceContract('opaque_tokenizer_frame', seal=_SEAL)
OPAQUE_CHECKPOINT_SAVE = _SourceContract('opaque_checkpoint_save', seal=_SEAL)
OPAQUE_CHECKPOINT_RESTORE = _SourceContract('opaque_checkpoint_restore', seal=_SEAL)
OPAQUE_CP_ADAPTER_FRAME = _SourceContract('opaque_cp_adapter_frame', seal=_SEAL)
OPAQUE_LEASE_FRAME = _SourceContract('opaque_lease_frame', seal=_SEAL)
_TOKENS = (FILE_HASH_LEAF, BACKEND_REF_FRAME, TEACHER_ROW, PREPARED_ROW, CP_FROZEN_HASH,
    CP_FILE_HASH, CP_DATA_HASH, CP_PUBLICATION_HASH, CP_TENSOR_PAIR_HASH, CP_SAVE_FRESH,
    CP_RESTORE, CP_SAVE_ADAPTER, CP_RESTORE_ADAPTER, LEASE_PARENT_CHECK, MODEL_FINGERPRINT, OPAQUE_INPUT_FRAME,
    OPAQUE_TOKENIZER_FRAME, OPAQUE_CHECKPOINT_SAVE, OPAQUE_CHECKPOINT_RESTORE, OPAQUE_CP_ADAPTER_FRAME, OPAQUE_LEASE_FRAME)
OPAQUE_ROLES = frozenset(str(token) for token in (OPAQUE_INPUT_FRAME, OPAQUE_TOKENIZER_FRAME,
                                               OPAQUE_CHECKPOINT_SAVE, OPAQUE_CHECKPOINT_RESTORE,
                                               OPAQUE_CP_ADAPTER_FRAME, OPAQUE_LEASE_FRAME))
CP_HASH_ROLES = frozenset(str(token) for token in (CP_FROZEN_HASH, CP_FILE_HASH, CP_DATA_HASH,
                                               CP_PUBLICATION_HASH, CP_TENSOR_PAIR_HASH))
_Q, _BOTH = ('qualification',), ('qualification', 'bootstrap')
_DEFINITIONS = MappingProxyType({
    'file_hash_leaf': ('input_io', _BOTH), 'backend_ref_frame': ('input_io', _Q),
    'teacher_row': ('input_io', _BOTH), 'prepared_row': ('tokenizer_render', _BOTH),
    'cp_frozen_hash': ('fingerprint', _Q), 'cp_file_hash': ('fingerprint', _Q),
    'cp_data_hash': ('fingerprint', _Q), 'cp_publication_hash': ('fingerprint', _Q),
    'cp_tensor_pair_hash': ('fingerprint', _Q), 'cp_save_fresh': ('checkpoint_save', _Q),
    'cp_restore': ('checkpoint_restore', _Q), 'lease_parent_check': ('supervisor_lease', ('bootstrap',)),
    'cp_save_adapter': ('checkpoint_adapter_io', _Q), 'cp_restore_adapter': ('checkpoint_adapter_io', _Q),
    'model_fingerprint': ('fingerprint', _Q), 'cp_hash_leaf': ('fingerprint', _BOTH),
    'guard': ('composed_source_guard', _BOTH), 'resource_probe_leaf': ('resource_probe', _BOTH),
    'cuda_probe_leaf': ('resource_cuda_probe', _BOTH), 'worker_os_leaf': ('worker_os_binding', _BOTH),
    'opaque_input_frame': ('input_io', _BOTH), 'opaque_tokenizer_frame': ('tokenizer_render', _BOTH),
    'opaque_checkpoint_save': ('checkpoint_save', _Q), 'opaque_checkpoint_restore': ('checkpoint_restore', _Q),
    'opaque_cp_adapter_frame': ('checkpoint_adapter_io', _Q),
    'opaque_lease_frame': ('supervisor_lease', ('bootstrap',))})
REQUIRED_ROLES = frozenset(_DEFINITIONS)
EMITTED_ROLES = frozenset(str(token) for token in _TOKENS)
MAPPED_KINDS = frozenset(_DEFINITIONS[role][0] for role in EMITTED_ROLES)
CP_REASONS = MappingProxyType({
    'cp_frozen_hash': 'checkpoint_frozen_state', 'cp_file_hash': 'checkpoint_payload_file',
    'cp_data_hash': 'checkpoint_payload_data', 'cp_publication_hash': 'checkpoint_streamed_publication',
    'cp_tensor_pair_hash': 'checkpoint_restored_saved_and_live_tensor'})


def validate_manifest(manifest, profiles=None, *, owner=None):
    require(type(manifest) is dict and set(manifest) == {'schema', 'publication_tail_s_upper', 'contracts'}
            and manifest['schema'] == CONTRACT_RULE and _finite(manifest['publication_tail_s_upper'], True),
            'exact finite source operation contract manifest')
    roles = manifest['contracts']; require(type(roles) is dict and set(roles) == REQUIRED_ROLES, 'exact source contract role catalog')
    for role, row in roles.items():
        kind, allowed = _DEFINITIONS[role]
        require(type(row) is dict and set(row) == {'kind', 'owners', 'wall_s_upper', 'local_s_upper', 'children'}
                and row['kind'] == kind, 'exact source contract kind/fields')
        require(type(row['owners']) is list and row['owners']
                and all(type(value) is str and value in allowed for value in row['owners'])
                and len(set(row['owners'])) == len(row['owners']), 'source contract owner scope')
        require(_finite(row['wall_s_upper'], True) and _finite(row['local_s_upper'], True)
                and type(row['children']) is dict and all(type(key) is str and key in roles and _integer(value, 1)
                    for key, value in row['children'].items()), 'finite source contract composition')
        if role in EMITTED_ROLES and profiles is not None and (owner is None or owner in row['owners']):
            require(kind in profiles and _finite(profiles[kind]['wall_s_upper'], True)
                    and row['wall_s_upper'] <= profiles[kind]['wall_s_upper'], 'source contract exceeds immutable kind wall maximum')
    visiting, done = set(), set()
    def visit(role):
        require(role not in visiting, 'source wall contract composition cycle')
        if role in done: return
        visiting.add(role); row = roles[role]; lower = row['local_s_upper']
        for child, count in row['children'].items():
            visit(child); lower += count * (roles[child]['wall_s_upper'] + manifest['publication_tail_s_upper'])
        require(row['wall_s_upper'] >= lower, 'undercomposed inclusive source wall contract: ' + role)
        visiting.remove(role); done.add(role)
    for role in roles: visit(role)
    leaf = roles['cp_hash_leaf']
    for role in CP_HASH_ROLES:
        row = roles[role]
        require(row['children'] == leaf['children'] and row['wall_s_upper'] == leaf['wall_s_upper']
                and row['local_s_upper'] == leaf['local_s_upper'],
                'exact source CP hash leaf composition alias')
    return manifest


def _freeze(value):
    if type(value) is dict: return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) is list: return tuple(_freeze(item) for item in value)
    return value
def frozen_manifest(manifest): return _freeze(copy.deepcopy(validate_manifest(manifest)))
def manifest_sha256(manifest): return hashlib.sha256(_canonical(validate_manifest(manifest))).hexdigest()


def manifest_from_spec(spec):
    require(type(spec) is dict, 'source operation contract specification')
    manifest = spec.get('operation_contracts')
    if manifest is not None: validate_manifest(manifest)
    return manifest


def _role(spec, kind, fields, owner, *, replay):
    manifest = manifest_from_spec(spec); token = fields.get('wall_contract')
    if manifest is None: return None
    if token is None:
        require(kind not in MAPPED_KINDS, 'mapped source operation requires exact wall contract')
        return None
    if replay:
        require((type(token) is str or any(token is item for item in _TOKENS)) and str(token) in EMITTED_ROLES,
                'explicit replay requires exact serialized source contract')
    else:
        require(any(token is item for item in _TOKENS), 'live immutable source contract token required before serialization')
    role = str(token); row = manifest['contracts'][role]
    require(row['kind'] == kind and owner in row['owners'], 'source contract kind/owner mismatch')
    if role == str(LEASE_PARENT_CHECK):
        require(fields.get('lease_action') == 'check_parent_use', 'source lease contract requires exact parent check action')
    return role


def validate_live_fields(spec, kind, phase, fields, *, owner, previous_role=None):
    require(type(fields) is dict and phase in ('started', 'returned', 'failed'), 'source live operation metadata')
    manifest = manifest_from_spec(spec); copied = dict(fields); token = fields.get('wall_contract')
    if token is not None:
        require(any(token is item for item in _TOKENS), 'live immutable source contract token required before serialization')
        expected_kind, allowed = _DEFINITIONS[str(token)]
        require(kind == expected_kind and owner in allowed, 'source token kind/owner mismatch')
    if manifest is None:
        copied.pop('wall_contract', None); return copied
    if phase == 'started': _role(spec, kind, fields, owner, replay=False)
    else:
        role = _role(spec, kind, fields, owner, replay=False) if token is not None else None
        require(role == previous_role, 'terminal source role differs from permanent start')
    return copied


def is_cp_extent(spec, kind, fields, *, owner, replay=False):
    role = _role(spec, kind, fields, owner, replay=replay)
    return role in CP_HASH_ROLES


def reservation_profile(spec, kind, fields, *, owner, maximum, replay=False):
    """Pure wall/CP-extent selection; this helper grants no execution capability."""
    role = _role(spec, kind, fields, owner, replay=replay)
    selected = copy.deepcopy(maximum)
    if role is None: return selected
    row = spec['operation_contracts']['contracts'][role]
    require(_finite(maximum['wall_s_upper'], True) and row['wall_s_upper'] <= maximum['wall_s_upper'],
            'source contract exceeds immutable kind wall maximum')
    # An existing caller profile may declare its maximum. It cannot select or
    # discount the source role's wall; finite fields remain bounded by maxima.
    if 'wall_s_upper' in fields:
        require(_finite(fields['wall_s_upper'], True) and fields['wall_s_upper'] <= maximum['wall_s_upper'],
                'declared wall exceeds immutable kind maximum')
    if role in OPAQUE_ROLES:
        require(row['wall_s_upper'] == maximum['wall_s_upper'], 'opaque source contract must retain full kind wall maximum')
        return selected
    selected['wall_s_upper'] = row['wall_s_upper']
    if role in CP_HASH_ROLES:
        require(spec.get('operation_profile_rule') == EXPLICIT_RESERVATION_RULE_V2,
                'source CP extent needs explicit reservation rule version 2')
        require(fields.get('checkpoint_hash_scope') is True and fields.get('reason') == CP_REASONS[role]
                and _integer(fields.get('state_bytes'), 1)
                and _integer(fields.get('identity_hash_attempts_upper'))
                and _integer(fields.get('identity_hash_bytes_upper')), 'exact source CP built-in plus extra extent')
        attempts = 1 + fields['identity_hash_attempts_upper']
        size = fields['state_bytes'] + fields['identity_hash_bytes_upper']
        require(attempts <= maximum['identity_hash_attempts_upper'] and size <= maximum['identity_hash_bytes_upper'],
                'source CP total hash extent exceeds immutable maximum')
        selected['identity_hash_attempts_upper'] = attempts; selected['identity_hash_bytes_upper'] = size
    return selected


def check_wall_return(fields, profile):
    values = [fields[key] for key in ('wall_s', 'wall_seconds') if key in fields]
    require(all(_finite(value) and value <= profile['wall_s_upper'] for value in values)
            and (len(values) < 2 or values[0] == values[1]), 'actual work exceeds selected reservation: wall aliases exceed or conflict')
