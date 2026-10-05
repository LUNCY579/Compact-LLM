"""Versioned producer/consumer bounds; no work or execution authority."""
from . import ce_operation_contracts as contracts
from . import ce_rejected_return as rejected

SCHEMA = 'CE8-BOUNDED-JOURNAL-PRODUCER-1'
MIB = 1024**2

def require(condition, message):
    if not condition: raise ValueError(message)

def default_limits(owner):
    require(owner in ('bootstrap', 'qualification'), 'exact bounded journal owner')
    return {'schema': SCHEMA, 'records_upper': 400008 if owner == 'bootstrap' else 200000,
            'bytes_upper': 256*MIB, 'line_bytes_upper': MIB if owner == 'bootstrap' else 4*MIB}

def limits_from_spec(spec, owner):
    value = spec.get('journal_limits')
    if value is None: return None
    expected = default_limits(owner)
    require(type(value) is dict and set(value) == set(expected)
            and all(type(value[key]) is type(item) and value[key] == item for key,item in expected.items()),
            'exact source-bound journal producer limits')
    require(contracts.manifest_from_spec(spec) is not None
            and rejected.policy_from_spec(spec, owner) is not None,
            'bounded journal requires frozen contracts and its prepaid fatal-record ticket')
    if owner == 'bootstrap':
        require(type(spec['caps']['operation_attempts']) is int
                and spec['caps']['operation_attempts'] <= 100000,
                '400008 service rows cover four rows per registered start plus eight fixed rows')
    return value

def check_append(limits, *, rows, size, line_bytes, terminal=False, tail_bytes=65536):
    if limits is None: return
    # Normal rows retain one whole prepaid failure slot. Terminal publication
    # may consume it; it cannot extend the registered producer/reader envelope.
    row_tail, byte_tail = (0, 0) if terminal else (1, tail_bytes)
    require(type(rows) is int and rows >= 0 and type(size) is int and size >= 0
            and type(line_bytes) is int and 0 < line_bytes <= limits['line_bytes_upper'],
            'finite bounded journal append shape')
    require(rows + 1 + row_tail <= limits['records_upper']
            and size + line_bytes + byte_tail <= limits['bytes_upper'],
            'journal producer record/byte ceiling with reserved failure tail')
