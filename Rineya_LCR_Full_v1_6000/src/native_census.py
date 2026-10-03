"""Read-only census under the FOUR exact Supa/Qwen tokenizer assets.

No model weights, no training, no truncation or row filtering. Counts all rows.
This command never upgrades external quarantine/structural gates to PASS.
"""
from __future__ import annotations
import argparse,hashlib,json,tempfile,sys
from collections import defaultdict
from pathlib import Path
from tokenizer_preflight import qualify,read_regular_file,ASSETS,measure_smoke

BANDS={'short':(300,600),'medium':(600,1100),'long':(1100,1600)}

def profile(values):
    values=sorted(values)
    if not values:return None
    def pct(q):
        x=(len(values)-1)*q;lo=int(x);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(x-lo)
    return {'min':values[0],'p50':pct(.50),'p95':pct(.95),'max':values[-1]}

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--tokenizer-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(argv)
    if a.output.exists() or a.output.is_symlink():raise SystemExit('Refusing existing output directory')
    a.output.mkdir(parents=True,exist_ok=False)
    pre=qualify(a.tokenizer_dir)
    (a.output/'native_preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
    if pre['status']!='ASSETS_AND_TOKENIZER_SMOKE_PASS_ONLY':
        (a.output/'token_profile.json').write_text(json.dumps({'status':'BLOCKED','rows_measured':0,'token_counts':None,'errors':pre['errors']},indent=2)+'\n')
        print('BLOCKED: no corpus token counts produced.');return 2
    from transformers import Qwen2TokenizerFast
    with tempfile.TemporaryDirectory(prefix='lcr-native-') as td:
        for name,expected in ASSETS.items():
            data=read_regular_file(a.tokenizer_dir/name)
            if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Asset changed after preflight')
            (Path(td)/name).write_bytes(data)
        tok=Qwen2TokenizerFast.from_pretrained(td,local_files_only=True,trust_remote_code=False)
        measure_smoke(tok)
        measurements=[];inputs={};by_band=defaultdict(list);by_task=defaultdict(list)
        for split in ('train','dev','holdout'):
            path=a.root/'data'/(split+'.jsonl');inputs[str(path.relative_to(a.root))]=hashlib.sha256(path.read_bytes()).hexdigest()
            for line in path.read_text().splitlines():
                row=json.loads(line);msgs=row['messages'];start,end=row['metadata']['context_char_span'];context=msgs[1]['content'][start:end]
                ctxt=tok.encode(context,add_special_tokens=False)
                P=tok.apply_chat_template(msgs[:2],tokenize=True,add_generation_prompt=True,enable_thinking=False)
                S=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=False,enable_thinking=False)
                if S[:len(P)]!=P:raise ValueError('Token prefix mismatch')
                comp=S[len(P):]
                if comp.count(tok.eos_token_id)!=1:raise ValueError('Expected exactly one completion EOS')
                band=row['metadata']['length_band'];lo,hi=BANDS[band]
                # This candidate resolves shared endpoints by assigning them to the higher band;
                # long retains 1600. The choice is explicit and does not alter the uploaded text.
                band_ok=(lo<=len(ctxt)<hi) if band!='long' else lo<=len(ctxt)<=hi
                record={'id':row['id'],'split':split,'task_type':row['metadata']['task_type'],'intended_length_band':band,
                   'context_tokens':len(ctxt),'rendered_total_tokens':len(S),'rendered_prompt_tokens':len(P),
                   'completion_tokens':len(comp),'unused_capacity':1920-len(P),
                   'total_le_2048':len(S)<=2048,'prompt_le_1920':len(P)<=1920,'context_band_fit':band_ok}
                measurements.append(record);by_band[band].append(len(S));by_task[row['metadata']['task_type']].append(len(S))
        with (a.output/'native_rows.jsonl').open('w',encoding='utf-8',newline='\n') as f:
            for item in measurements:f.write(json.dumps(item,separators=(',',':'))+'\n')
        failures=[m for m in measurements if not all(m[k] for k in ('total_le_2048','prompt_le_1920','context_band_fit'))]
        report={'status':'NATIVE_CENSUS_COMPLETED_NOT_CORPUS_ACCEPTANCE','rows_total':len(measurements),
          'input_sha256':inputs,'rows_total_le_2048':sum(m['total_le_2048'] for m in measurements),
          'rows_prompt_le_1920':sum(m['prompt_le_1920'] for m in measurements),'rows_context_band_fit':sum(m['context_band_fit'] for m in measurements),
          'rendered_total_tokens':profile([m['rendered_total_tokens'] for m in measurements]),
          'rendered_prompt_tokens':profile([m['rendered_prompt_tokens'] for m in measurements]),
          'total_tokens_by_intended_band':{k:profile(v) for k,v in by_band.items()},
          'total_tokens_by_task':{k:profile(v) for k,v in by_task.items()},
          'native_failures':len(failures),'rows_dropped':0,'rows_truncated':0,'training_authorized':False,
          'external_quarantine_status':'UNCHANGED_UNVERIFIED','structural_status':'UNCHANGED_SCOPED_LIMITATIONS',
          'band_endpoint_convention':'[300,600), [600,1100), [1100,1600] context tokens; explicit candidate convention'}
        (a.output/'token_profile.json').write_text(json.dumps(report,indent=2)+'\n')
        (a.output/'native_failures.json').write_text(json.dumps(failures,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
