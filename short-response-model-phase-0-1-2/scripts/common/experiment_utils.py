from __future__ import annotations
import hashlib, json, platform, subprocess, sys
from pathlib import Path
from typing import Any

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open(encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if line.strip():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError as e: raise ValueError(f"{path}:{n}: {e}") from e
    return rows
def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def run(cmd: list[str]) -> None:
    print('$',' '.join(cmd), flush=True); subprocess.run(cmd, check=True)
def environment() -> dict[str, Any]:
    return {'python':sys.version,'platform':platform.platform()}
