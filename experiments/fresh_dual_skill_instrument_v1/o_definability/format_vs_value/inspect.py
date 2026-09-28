import json,sys
sys.dont_write_bytecode=True
ROOT='C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/'
p=json.load(open(ROOT+'Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze/PANEL.json',encoding='utf-8'))
def sk(d,depth=0,pre=''):
    for k,v in d.items():
        if isinstance(v,dict) and depth<3 and k not in('predictions',):
            print(pre+k+':'); sk(v,depth+1,pre+'  ')
        elif isinstance(v,list): print(pre+k, 'list', len(v), (v[:3] if v and not isinstance(v[0],dict) else v[:1]))
        elif isinstance(v,dict): print(pre+k,'dict',len(v))
        else: print(pre+k, repr(v)[:300])
sk(p)
k=next(iter(p['predictions'])); print(k, p['predictions'][k])
