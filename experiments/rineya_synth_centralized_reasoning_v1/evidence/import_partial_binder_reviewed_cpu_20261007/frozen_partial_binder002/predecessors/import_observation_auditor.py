"""Selected import-effect replay; never exact-effect PASS or native permission."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
REPLAY_DESCRIPTOR={'bytes': 12369, 'sha256': 'ce8ea36cd113310d35bd09ee223f279bdbf5703b8c7d93ae143325bd9f95124f', 'role': 'python_source'}
def replay(raw,*,policy,grammar,identity,files,sources):
    from types import ModuleType
    path=HERE/'measurement_replay.py';source=path.read_bytes()
    if REPLAY_DESCRIPTOR!={'bytes':len(source),'sha256':hashlib.sha256(source).hexdigest(),'role':'python_source'}:raise ValueError('externally selected independent measurement reader')
    module=ModuleType('selected_import_measurement_replay');module.__file__=str(path);exec(compile(source,str(path),'exec'),module.__dict__)
    return module.validate_measurement(raw,policy=policy,grammar=grammar,expected_identity=identity,selected_file_map=files,selected_source_map=sources,expected_file_sha256=hashlib.sha256(raw).hexdigest())
def native_entry(*args,**kwargs):raise ValueError('import observation independent replay has no native grant')
