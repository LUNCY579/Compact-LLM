import json,sys
sys.dont_write_bytecode=True
AB='C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/experiments/architecture_benchmark_001/'
def sk(d,depth=0,pre='',maxd=2):
    for k,v in d.items():
        if isinstance(v,dict) and depth<maxd and len(v)<40:
            print(pre+k+':'); sk(v,depth+1,pre+'  ',maxd)
        elif isinstance(v,list): print(pre+k, 'list', len(v), repr(v[:2])[:300])
        elif isinstance(v,dict): print(pre+k,'dict',len(v))
        else: print(pre+k, repr(v)[:300])
s=json.load(open(AB+'outputs/retention-a329eb7b/B28/retention.json',encoding='utf-8'))
sk(s)
print('=====FP32')
f=json.load(open(AB+'evidence/retention-design-v1/ARM_DOSE_B28last20_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95.json',encoding='utf-8'))
sk({k:v for k,v in f.items() if k!='checkpoints'})
print('n checkpoints',len(f['checkpoints']))
c=f['checkpoints'][-1]
sk({k:v for k,v in c.items() if k!='outputs'},maxd=3)
k=next(iter(c['outputs'])); print(k,c['outputs'][k])
