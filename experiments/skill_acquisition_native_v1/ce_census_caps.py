"""Compiled prospective owner bounds. Not approved and not a grant."""
import copy

BASIS = {'schema': 'CE8-CENSUS-DERIVED-PROSPECTIVE-CAPS-1', 'matrix_sha256': '49cc9b462b1592bd2062f0fd35083cf1e16ab334c3d474e9a473ed64361a2e60', 'stage_bounds': {'runtime_anchor': {'operation_attempts_upper': 64, 'io_read_bytes_upper': 34359738368, 'io_write_bytes_upper': 1073741824, 'identity_hash_attempts_upper': 262144, 'identity_hash_bytes_upper': 68719476736, 'metadata_bytes_upper': 536870912, 'wall_seconds_upper': 65536}, 'longest_masks': {'operation_attempts_upper': 1024, 'io_read_bytes_upper': 536870912, 'io_write_bytes_upper': 1073741824, 'identity_hash_attempts_upper': 262144, 'identity_hash_bytes_upper': 2147483648, 'metadata_bytes_upper': 536870912, 'wall_seconds_upper': 131072}, 'likelihood_gradient_parity': {'operation_attempts_upper': 1024, 'io_read_bytes_upper': 8589934592, 'io_write_bytes_upper': 4294967296, 'identity_hash_attempts_upper': 262144, 'identity_hash_bytes_upper': 34359738368, 'metadata_bytes_upper': 536870912, 'wall_seconds_upper': 131072}, 'adam_clipping': {'operation_attempts_upper': 128, 'io_read_bytes_upper': 4294967296, 'io_write_bytes_upper': 1073741824, 'identity_hash_attempts_upper': 262144, 'identity_hash_bytes_upper': 68719476736, 'metadata_bytes_upper': 536870912, 'wall_seconds_upper': 32768}, 'checkpoint_resume': {'operation_attempts_upper': 8192, 'io_read_bytes_upper': 274877906944, 'io_write_bytes_upper': 17179869184, 'identity_hash_attempts_upper': 1048576, 'identity_hash_bytes_upper': 1099511627776, 'metadata_bytes_upper': 536870912, 'wall_seconds_upper': 8388608}, 'delta_roundtrip': {'operation_attempts_upper': 16384, 'io_read_bytes_upper': 68719476736, 'io_write_bytes_upper': 4294967296, 'identity_hash_attempts_upper': 262144, 'identity_hash_bytes_upper': 137438953472, 'metadata_bytes_upper': 536870912, 'wall_seconds_upper': 1048576}, 'resource_io_measurement': {'operation_attempts_upper': 16, 'io_read_bytes_upper': 4294967296, 'io_write_bytes_upper': 1073741824, 'identity_hash_attempts_upper': 262144, 'identity_hash_bytes_upper': 4294967296, 'metadata_bytes_upper': 536870912, 'wall_seconds_upper': 32768}}, 'global_bounds': {'operation_attempts_upper': 32768, 'io_read_bytes_upper': 549755813888, 'io_write_bytes_upper': 34359738368, 'identity_hash_attempts_upper': 4194304, 'identity_hash_bytes_upper': 2199023255552, 'metadata_bytes_upper': 4294967296, 'wall_seconds_upper': 16777216}, 'service_bounds': {'operation_attempts_upper': 65536, 'io_read_bytes_upper': 68719476736, 'io_write_bytes_upper': 8589934592, 'identity_hash_attempts_upper': 2097152, 'identity_hash_bytes_upper': 137438953472, 'metadata_bytes_upper': 536870912, 'wall_seconds_upper': 4194304}, 'source_cardinality_review': 'PENDING_INDEPENDENT_FINAL_SOURCE_REVIEW', 'physical_deadline_seconds': 19200, 'approved': False, 'dispatch_authorized': False, 'native_qualified': False}

def apply_qualification(stage_caps, global_caps):
    for stage, row in BASIS['stage_bounds'].items():
        for field, value in row.items(): stage_caps[stage][field]=max(stage_caps[stage][field],value)
    for field, value in BASIS['global_bounds'].items(): global_caps[field]=max(global_caps[field],value)
    # Include an unchanged Q stage not present in the native owner matrix.
    for field in global_caps:
        global_caps[field]=max(global_caps[field],sum(row[field] for row in stage_caps.values())+128*1024**2
            if field in ('io_read_bytes_upper','io_write_bytes_upper','identity_hash_bytes_upper','metadata_bytes_upper')
            else sum(row[field] for row in stage_caps.values())+19200 if field=='wall_seconds_upper'
            else sum(row[field] for row in stage_caps.values())+10000 if field=='identity_hash_attempts_upper'
            else sum(row[field] for row in stage_caps.values())+100)

def apply_service(caps):
    aliases={'operation_attempts_upper':'operation_attempts','io_read_bytes_upper':'read_bytes',
      'io_write_bytes_upper':'write_bytes','identity_hash_attempts_upper':'hash_attempts',
      'identity_hash_bytes_upper':'hash_bytes','metadata_bytes_upper':'metadata_bytes','wall_seconds_upper':'wall_seconds'}
    for field,value in BASIS['service_bounds'].items(): caps[aliases[field]]=max(caps[aliases[field]],value)
    return copy.deepcopy(BASIS)
