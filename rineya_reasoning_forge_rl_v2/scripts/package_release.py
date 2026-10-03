"""Create a deterministic ZIP of exactly the frozen manifest and covered files."""
from __future__ import annotations
import argparse
import sys
import zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from forge_rl.common import loads, sha_file, dumps
from forge_rl.validate import verify_manifest


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if args.output.exists():raise FileExistsError('Archive already exists')
    verify_manifest(root,'RELEASE_MANIFEST.json')
    manifest=loads((root/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with zipfile.ZipFile(args.output,'x',allowZip64=True) as archive:
            for rel in sorted([*manifest['files'],'RELEASE_MANIFEST.json']):
                path=root/rel
                info=zipfile.ZipInfo('rineya_reasoning_forge_rl_v2/'+rel,date_time=(2026,9,14,0,0,0))
                info.compress_type=zipfile.ZIP_STORED if path.suffix=='.gz' else zipfile.ZIP_DEFLATED
                info.external_attr=0o100644<<16
                archive.writestr(info,path.read_bytes(),compresslevel=6)
        with zipfile.ZipFile(args.output) as archive:
            bad=archive.testzip()
            if bad:raise ValueError('ZIP CRC failed: '+bad)
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    print(dumps({'status':'PASS','path':str(args.output),'bytes':args.output.stat().st_size,
                 'sha256':sha_file(args.output),'files':len(manifest['files'])+1,'zip_crc':'PASS'}))


if __name__=='__main__':main()
