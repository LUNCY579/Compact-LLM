"""Fixed roots for the separately registered integration experiment."""
from pathlib import Path
import hashlib
import json

BASE = Path('C:/Users/clclc/BusinessProject')
HERE = Path(__file__).resolve().parent
OUT = BASE / 'rineya-dataset-retention-records/binding-integration001-20260920'
PILOT = BASE / 'ChatBubble-binding-integration-pilot-20260920'
MQPY = BASE / '.venv-mq001/Scripts/python.exe'
RABPY = BASE / '.venv-rab001-cu126/Scripts/python.exe'
MIXTURE = OUT.parent / 'binding-mixture001-20260920'
CONTROL = OUT.parent / 'highdose-lora001-resume-20260920'
DIAGNOSTIC = OUT.parent / 'binding-diagnostic001-20260920'
BASELINE = OUT.parent / 'binding-dev-baseline001-20260920'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
