"""Freeze hashes for a reviewed artifact tree. This is not a digital signature."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from forge_rl.common import sha_file, write_json, dumps


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();target=root/'RELEASE_MANIFEST.json'
    if not root.is_dir():raise ValueError('Root must exist')
    if target.exists():raise FileExistsError('Release manifest already exists; never silently refresh integrity after a change')
    entries={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('Symlinks are not permitted in a frozen bundle')
        if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.pyc','.pyo'):
            continue
        rel=path.relative_to(root).as_posix()
        entries[rel]={'sha256':sha_file(path),'bytes':path.stat().st_size}
    if not entries:raise ValueError('Refusing to freeze an empty tree')
    write_json(target,{'release':'Rineya Reasoning Forge RL v2.0.0','hash_algorithm':'sha256',
                       'note':'The manifest excludes itself. Compare the separately supplied archive digest for an external integrity anchor.',
                       'files':entries})
    print(dumps({'status':'FROZEN','files':len(entries),'bytes':sum(v['bytes'] for v in entries.values())}))


if __name__=='__main__':main()
