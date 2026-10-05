"""Exact one-site fixed05/06 adapter; preservation only, no probe or success audit."""
from pathlib import Path
import ast, hashlib, types

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
ORIGINAL = PACKAGE / 'artifacts/phase3_launch_registration_work/preserve_terminal.py'
PRESERVED = HERE / 'PRESERVED_PRESERVER_29790d.py'
PIN = '29790d005ee98073125f35748b1878eeafbbc4f209e93d0083ab385446a0eac1'
HEAD = '9cabf4891221592be7946bb6ded7315bd2962185'
SIGNAL = 'ROOT_TERMINAL_FROZEN_BYTES_AND_COPY_AUTHORIZED'
OLD = "scope['batch_selector']=={'source_kind':'atomic_RS_DEV','ordinal':4}"
NEW = "scope['batch_selector']=={'source_kind':'atomic_RS_DEV','ordinal':EXPECTED_ORDINAL}"
ANCHORS = {
    5: ('b8e2efe1e832ca865bc8a5de8f48b9d601246cd14335a2e6e09ecc4a1362ba00',
        'add9d169f88a6d62a8f77c521d0b7da57083264238f5b0f8a3ded290bc10624f'),
    6: ('06677a414195afb137daba2f434edd8f4ba0a4f82b029c52e17582d8fb6e095c',
        'a66a23c69209bc43eab58075e705ee1a2bd4793252e8f205f1926b603337da96')}

def require(value, message):
    if not value:
        raise ValueError(message)
def hashpin(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

def adapted(ordinal):
    require(type(ordinal) is int and ordinal in (5,6), 'explicit fixed remaining ordinal5 or6 required before reads')
    raw = PRESERVED.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PIN and ORIGINAL.read_bytes() == raw, 'exact preserved/current original29790d bytes')
    source = raw.decode('utf-8')
    require(source.count(OLD) == 1, 'exact one selector adaptation site')
    source = source.replace(OLD, NEW, 1)
    ast.parse(source)
    module = types.ModuleType('fixed_remaining_preserver_%02d'%ordinal)
    module.__file__ = str(ORIGINAL)
    module.EXPECTED_ORDINAL = ordinal
    exec(compile(source, str(PRESERVED), 'exec'), module.__dict__)
    return module, hashlib.sha256(source.encode('utf-8')).hexdigest()

def preserve_fixed(ordinal, config, leaf, saved, inventory, *, config_sha, inventory_sha,
        commit, sup_sha, terminal_signal=None, current_death_path=None, current_death_sha=None):
    require(type(ordinal) is int and ordinal in (5,6), 'explicit fixed remaining ordinal5 or6 required before reads')
    require(terminal_signal == SIGNAL, 'explicit Root terminal frozen copy signal required')
    require(commit == HEAD, 'explicit exact committed05/06 launch HEAD')
    require(config_sha == ANCHORS[ordinal][0] and inventory_sha == ANCHORS[ordinal][1], 'exact external fixed config/inventory byte anchors')
    require(hashpin(sup_sha), 'explicit externally frozen supervisor SHA256')
    require((current_death_path is None and current_death_sha is None) or
        (current_death_path is not None and hashpin(current_death_sha)), 'paired external current-death anchor when worker existed')
    require(Path(config).absolute() == PACKAGE / ('configs/SYNTH_SHORT_M0_ATOMIC_BATCH%02d_PHASE3_A1.json'%ordinal), 'exact fixed ordinal config path')
    require(Path(inventory).absolute() == PACKAGE / ('artifacts/short_m0_audit_registrations/SYNTH_SHORT_PHASE3_BATCH%02d_A1_AUDIT_INVENTORY_001.json'%ordinal), 'exact fixed ordinal inventory path')
    module, source_sha = adapted(ordinal)
    # The unchanged terminal gate requires death proof only if a process was
    # actually created. Admission-denied evidence remains a no-worker failure.
    result = module.preserve(config, leaf, saved, inventory, config_sha=config_sha, inventory_sha=inventory_sha,
        sup_sha=sup_sha, commit=commit, terminal_signal=terminal_signal,
        current_death_path=current_death_path, current_death_sha=current_death_sha)
    return result
