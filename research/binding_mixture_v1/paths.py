"""Fixed locations and create-only metadata I/O for BINDING-MIXTURE-001."""
from pathlib import Path
import hashlib
import json

BASE = Path('C:/Users/clclc/BusinessProject')
OUT = BASE/'rineya-dataset-retention-records/binding-mixture001-20260920'
HERE = Path(__file__).resolve().parent
PILOT = BASE/'ChatBubble-binding-mixture-pilot-20260920'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
