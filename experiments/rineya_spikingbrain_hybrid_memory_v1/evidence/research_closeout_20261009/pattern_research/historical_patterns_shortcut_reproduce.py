"""Reproduce the root's offline shortcut audit with only its output name changed."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
source = root / 'shortcut_opportunity_audit.py'
copy = root / 'historical_patterns_shortcut_reproduction.py'
code = source.read_text(encoding='utf-8')
old = "OUT / 'shortcut-opportunities.json'"
new = "OUT / 'historical_patterns_shortcut_reproduced.json'"
assert code.count(old) == 1
modified = code.replace(old, new)
assert modified.replace(new, old) == code
with copy.open('x', encoding='utf-8') as stream:
    stream.write(modified)
raise SystemExit(subprocess.run([sys.executable, '-B', str(copy)], check=False).returncode)
