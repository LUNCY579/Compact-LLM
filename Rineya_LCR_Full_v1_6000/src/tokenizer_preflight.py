"""Read-only, hash-locked native-tokenizer input qualification. NEVER emits training data.

Exit 0 means only asset and tokenizer-smoke checks passed. Exit 2 means blocked.
Only four named tokenizer files are read; no weights or evaluation banks are opened.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, stat, sys, tempfile
from pathlib import Path

ASSETS = {
 'merges.txt':'8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
 'tokenizer.json':'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
 'tokenizer_config.json':'d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101',
 'vocab.json':'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
}
MAX_FILE_BYTES=32*1024*1024
LFS_PREFIX=b'version https://git-lfs.github.com/spec/v1'
EXPECTED_TRANSFORMERS='4.57.1'


def read_regular_file(path: Path) -> bytes:
    """Reject symlinks (including parents), LFS pointers, oversized/changing files."""
    path=path.absolute()
    for part in (path,*path.parents):
        if part.is_symlink():raise ValueError(f'Symlink is not accepted: {part}')
    before=path.stat()
    if not stat.S_ISREG(before.st_mode):raise ValueError(f'Not a regular file: {path}')
    if before.st_size<=0 or before.st_size>MAX_FILE_BYTES:
        raise ValueError(f'Invalid asset byte size: {path.name}: {before.st_size}')
    flags=os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)
    fd=os.open(path,flags)
    try:
        opened=os.fstat(fd)
        if (opened.st_dev,opened.st_ino)!=(before.st_dev,before.st_ino):
            raise ValueError(f'File identity changed: {path.name}')
        with os.fdopen(fd,'rb',closefd=False) as f:
            data=f.read(MAX_FILE_BYTES+1)
        after_fd=os.fstat(fd)
    finally:os.close(fd)
    after_path=path.stat()
    attrs=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
    fingerprint=lambda s:tuple(getattr(s,k) for k in attrs)
    if not (fingerprint(before)==fingerprint(after_fd)==fingerprint(after_path)):
        raise ValueError(f'File changed while reading: {path.name}')
    if len(data)!=before.st_size:raise ValueError(f'Incomplete read: {path.name}')
    if data.startswith(LFS_PREFIX):raise ValueError(f'Unresolved Git LFS pointer: {path.name}')
    return data


def inspect_assets(root: Path, expected: dict[str,str] | None=None) -> tuple[list[dict],list[str]]:
    """The optional map supports unit fixtures only; CLI always passes ASSETS."""
    expected=ASSETS if expected is None else expected
    entries=[];errors=[]
    for name,wanted in expected.items():
        try:
            if Path(name).name!=name or name in ('.','..'):
                raise ValueError('Only fixed basenames are allowed')
            data=read_regular_file(root/name)
            actual=hashlib.sha256(data).hexdigest()
            match=actual==wanted
            entries.append({'file':name,'size_bytes':len(data),'expected_sha256':wanted,
                'actual_sha256':actual,'hash_match':match})
            if not match:errors.append(f'Hash mismatch: {name}')
        except (OSError,ValueError) as exc:
            entries.append({'file':name,'expected_sha256':wanted,'actual_sha256':None,
                'hash_match':False,'error':str(exc)})
            errors.append(f'{name}: {exc}')
    return entries,errors


def measure_smoke(tokenizer) -> list[dict]:
    """Native callers pass the real tokenizer. Unit-test doubles are not native evidence."""
    measurements=[]
    for answer in ('Willow','17','UNKNOWN'):
        messages=[{'role':'system','content':'Answer only from the record. Give the exact answer or UNKNOWN.'},
            {'role':'user','content':'This is a tokenizer-only mechanical fixture, not a training example.'}]
        prompt=tokenizer.apply_chat_template(messages,tokenize=True,
            add_generation_prompt=True,enable_thinking=False)
        full=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':answer}],
            tokenize=True,add_generation_prompt=False,enable_thinking=False)
        if not isinstance(prompt,list) or not isinstance(full,list):
            raise ValueError('Expected token ID lists')
        if not prompt or not full or not all(type(x) is int for x in prompt+full):
            raise ValueError('Invalid token ID sequence')
        if len(full)<=len(prompt) or full[:len(prompt)]!=prompt:
            raise ValueError('Full-chat/prompt token prefix is not exact')
        completion=full[len(prompt):]
        eos=tokenizer.eos_token_id
        if type(eos) is not int or completion.count(eos)!=1:
            raise ValueError('Completion must contain exactly one EOS token')
        if len(prompt)>1920 or len(full)>2048:
            raise ValueError('Mechanical smoke exceeds the requested window')
        measurements.append({'fixture_answer':answer,'rendered_prompt_tokens':len(prompt),
            'rendered_total_tokens':len(full),'completion_tokens':len(completion),
            'completion_eos_count':1,'prompt_prefix_matches':True,
            'unused_prompt_capacity':1920-len(prompt),
            'scope':'TOKENIZER_MECHANICS_ONLY_NOT_CORPUS'})
    return measurements


def qualify(root: Path) -> dict:
    entries,errors=inspect_assets(root,ASSETS)
    report={'status':'BLOCKED','asset_checks':entries,'errors':errors,
        'native_smoke':None,'native_corpus_rows_measured':0,
        'corpus_generated':False,'corpus_accepted':False,'training_authorized':False,
        'model_weight_files_read':0,'evaluation_banks_read':0,
        'only_read_asset_basenames':list(ASSETS)}
    if errors:return report
    try:
        actual=importlib.metadata.version('transformers')
        report['transformers_version']=actual
        if actual!=EXPECTED_TRANSFORMERS:
            raise ValueError(f'Transformers {EXPECTED_TRANSFORMERS} required; found {actual}')
        report['tokenizers_version']=importlib.metadata.version('tokenizers')
        from transformers import Qwen2TokenizerFast
        # Isolate exactly the four locked files. Additional local tokenizer config,
        # custom Python files and special-token maps cannot influence this load.
        with tempfile.TemporaryDirectory(prefix='lcr-tokenizer-input-') as td:
            for name,wanted in ASSETS.items():
                data=read_regular_file(root/name)
                if hashlib.sha256(data).hexdigest()!=wanted:
                    raise ValueError(f'Asset changed after initial inspection: {name}')
                (Path(td)/name).write_bytes(data)
            tokenizer=Qwen2TokenizerFast.from_pretrained(td,local_files_only=True,
                trust_remote_code=False)
            if tokenizer.__class__.__name__!='Qwen2TokenizerFast':
                raise ValueError('Unexpected tokenizer runtime class')
            report['runtime_class']=tokenizer.__class__.__name__
            report['native_smoke']=measure_smoke(tokenizer)
        report['status']='ASSETS_AND_TOKENIZER_SMOKE_PASS_ONLY'
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    return report


def write_create_only(path: Path, report: dict) -> None:
    if path.is_symlink():raise FileExistsError('Refusing symlink output')
    if not path.parent.is_dir():raise FileNotFoundError('Output parent must already exist')
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        print('Refusing existing output.',file=sys.stderr);return 2
    report=qualify(args.tokenizer_dir)
    try:write_create_only(args.output,report)
    except (OSError,ValueError) as exc:
        print(f'Cannot write receipt: {exc}',file=sys.stderr);return 2
    print(report['status'])
    return 0 if report['status']=='ASSETS_AND_TOKENIZER_SMOKE_PASS_ONLY' else 2

if __name__=='__main__':raise SystemExit(main())
