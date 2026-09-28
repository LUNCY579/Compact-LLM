# Identity reproduction of typed-scorer correct_ids (Supa B28, Opallo BF16 PANEL, Opallo FP32 dose file)
import hashlib, importlib.util, json, glob, os, collections
ROOT='C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification'
AB=ROOT+'/experiments/architecture_benchmark_001'
OUT=os.path.dirname(os.path.abspath(__file__))
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
RP=AB+'/retention.py'
h=sha(RP); assert h.startswith('c52b6a48'), h
spec=importlib.util.spec_from_file_location('_r2_bdcr_retention', RP)
ret=importlib.util.module_from_spec(spec); spec.loader.exec_module(ret)
AUD=AB+'/evidence/retention-design-v1/PROBE_LABEL_AUDIT.json'
audit=json.load(open(AUD,encoding='utf-8'))
ov={n: ret.build_overlay(audit,n)['targets'] for n in ret.BARE_POLICIES}
PR=AB+'/data/monitor_probe.jsonl'
raw=open(PR,'rb').read()
probe=[json.loads(l) for l in raw.decode('utf-8').splitlines() if l.strip()]
print('retention.py',h[:8],'| audit',sha(AUD)[:8],'| probe',sha(PR)[:8],'rows',len(probe),'CRLF lines',raw.count(b'\r\n'))
cls=collections.Counter(r.get('reply_class') or r['id'].split('__')[1] for r in probe); print('probe classes',dict(cls))
ids={r['id'] for r in probe}; assert set(ov['strict'])==ids==set(ov['permissive'])
def score(preds):
    return {n:sorted(i for i in ids if ret.score_prediction(preds[i]['text'],preds[i]['terminated'],t[i])['correct']) for n,t in ov.items()}
res={}
B28=AB+'/outputs/retention-a329eb7b/B28/retention.json'
b=json.load(open(B28,encoding='utf-8'))
sp={r['id']:{'text':r['source']['raw'],'terminated':r['source']['terminated']} for r in b['rows']}
s=score(sp); res['supa_B28']=s
for n in ret.BARE_POLICIES:
    ref=set(b['reference_ids_by_policy'][n]); print('Supa B28',sha(B28)[:8],n,len(s[n]),'IDENTITY',set(s[n])==ref,'recorded',len(ref))
P=ROOT+'/Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze/PANEL.json'
pd=json.load(open(P,encoding='utf-8'))
o=score(pd['predictions']); res['opallo_bf16']=o
for n in ret.BARE_POLICIES:
    ref=set(pd['by_policy'][n]['correct_ids']); print('Opallo BF16 PANEL',sha(P)[:8],n,len(o[n]),'IDENTITY',set(o[n])==ref,'recorded',len(ref))
# NEG control: rotate targets by one row within sorted ids
idl=sorted(ids); rot=dict(zip(idl, idl[1:]+idl[:1]))
for n,t in ov.items():
    c=sum(ret.score_prediction(pd['predictions'][i]['text'],pd['predictions'][i]['terminated'],t[rot[i]])['correct'] for i in idl)
    c2=sum(ret.score_prediction(sp[i]['text'],sp[i]['terminated'],t[rot[i]])['correct'] for i in idl)
    print('NEG rotated targets',n,'Opallo',c,'Supa',c2)
for c in sorted(glob.glob(AB+'/**/ARM_DOSE_B28last20*b20.95*.json',recursive=True)):
    d=json.load(open(c,encoding='utf-8'))
    if not d.get('checkpoints') or 'outputs' not in d['checkpoints'][-1]: print('no-outputs',os.path.basename(c)); continue
    outs=d['checkpoints'][-1]['outputs']
    if isinstance(outs,list): outs={x['id']:x for x in outs}
    if set(outs)!=ids: print('skip',os.path.basename(c)); continue
    f=score(outs); print('FP32cand',os.path.basename(c),sha(c)[:8],{n:len(v) for n,v in f.items()})
    if len(f['permissive'])==74 and len(f['strict'])==59: res.setdefault('opallo_fp32_candidates',{})[os.path.basename(c)]=f
# per-class breakdown for Opallo BF16 / Supa
for nm in ('supa_B28','opallo_bf16'):
    for n in ret.BARE_POLICIES:
        print(nm,n,dict(collections.Counter(i.split('__')[1] for i in res[nm][n])))
json.dump(res,open(OUT+'/identity_ids.json','w'),indent=1)
