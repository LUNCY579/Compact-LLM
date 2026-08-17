from __future__ import annotations
import argparse, json, re, math, statistics
from pathlib import Path
from collections import Counter, defaultdict

ap=argparse.ArgumentParser(description='Deterministic semantic and textual audit for the Rineya v2 corpus.')
ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == 'scripts' else Path('/mnt/data/Rineya_General_Reasoning_and_Natural_Communication_Corpus_v2.0'))
a=ap.parse_args()
ROOT=a.root.resolve()
SFT=ROOT/'data/sft/rineya_v2_new_all_train.jsonl'
DPO=ROOT/'data/dpo/rineya_v2_preferences_train_12000.jsonl'

def read(path):
    return [json.loads(x) for x in path.open(encoding='utf-8') if x.strip()]

def fmt(v):
    if abs(v-round(v))<1e-9: return str(int(round(v)))
    return f'{v:.2f}'.rstrip('0').rstrip('.')

def money(v):
    return f'${int(round(v))}' if abs(v-round(v))<1e-9 else f'${v:.2f}'

def count_noun(v, singular, plural=None):
    plural=plural or singular+'s'
    return f"{fmt(v)} {singular if abs(float(v)-1.0)<1e-9 else plural}"

def latest(row): return [m['content'] for m in row['messages'] if m['role']=='user'][-1]
def users(row): return [m['content'] for m in row['messages'] if m['role']=='user']
def ans(row): return row['messages'][-1]['content']
def family(row): return row['metadata']['base_template_family']
def tnum(row): return int(family(row).rsplit('_',1)[1])
def num_target(a):
    m=re.search(r'-?\d+(?:\.\d+)?',a.replace(',',''))
    return float(m.group()) if m else None

def parse_time(s):
    m=re.fullmatch(r'(\d{1,2}):(\d{2}) (AM|PM)',s)
    h=int(m.group(1))%12 + (12 if m.group(3)=='PM' else 0)
    return h*60+int(m.group(2))
def show_time(total):
    total%=1440; h=total//60; mi=total%60; suf='AM' if h<12 else 'PM'; h12=h%12 or 12
    return f'{h12}:{mi:02d} {suf}'

errors=[]; checked=Counter(); class_errors=Counter()
def fail(row,msg):
    errors.append({'id':row['id'],'class':row['metadata']['reply_class'],'family':family(row),'message':msg,'latest_user':latest(row),'target':ans(row)})
    class_errors[row['metadata']['reply_class']]+=1

def expect(row, expected):
    checked[row['metadata']['reply_class']]+=1
    if ans(row)!=expected: fail(row,f'expected {expected!r}')

def check_row(row):
    c=row['metadata']['reply_class']; t=tnum(row); q=latest(row); a=ans(row)
    try:
      if c=='arithmetic_chain':
        if t==0:
          m=re.search(r'count starts at (\d+), gains (\d+), loses (\d+), then gains (\d+)',q,re.I); e=int(m[1])+int(m[2])-int(m[3])+int(m[4])
        elif t==1:
          m=re.search(r'score starts at (\d+), increases by (\d+), then doubles',q,re.I); e=(int(m[1])+int(m[2]))*2
        elif t==2:
          m=re.search(r'There are (\d+) .*?\. (\d+) are removed and (\d+) are added',q,re.I); e=int(m[1])-int(m[2])+int(m[3])
        elif t==3:
          m=re.search(r'(\d+) packs with (\d+) .*? each\. After using (\d+)',q,re.I); e=int(m[1])*int(m[2])-int(m[3])
        elif t==4:
          m=re.search(r'awards (\d+) points in each of (\d+) rounds, then subtracts a (\d+)-point',q,re.I); e=int(m[1])*int(m[2])-int(m[3])
        elif t==5:
          m=re.search(r'bin has (\d+) items\. Half are moved out, then (\d+) are returned',q,re.I); e=int(m[1])//2+int(m[2])
        elif t==6:
          m=re.search(r'total starts at (\d+), then (\d+) and (\d+) are added.*?(\d+) groups',q,re.I); e=(int(m[1])+int(m[2])+int(m[3]))//int(m[4])
        else:
          m=re.search(r'balance is (\d+) units\. It uses (\d+) units, then (\d+) more',q,re.I); e=int(m[1])-int(m[2])-int(m[3])
        expect(row,f'{e}.')
      elif c=='percentage_reasoning':
        if t==0:
          m=re.search(r'What is (\d+)% of (\d+)',q); e=int(m[1])*int(m[2])/100; exp=f'{fmt(e)}.'
        elif t==1:
          m=re.search(r'costs \$(\d+) and is (\d+)% off',q); e=int(m[1])*(1-int(m[2])/100); exp=f'{money(e)}.'
        elif t==2:
          m=re.search(r'value of (\d+) increases by (\d+)%',q); e=int(m[1])*(1+int(m[2])/100); exp=f'{fmt(e)}.'
        elif t==3:
          m=re.search(r'(\d+) of (\d+) tasks',q); e=int(m[1])/int(m[2])*100; exp=f'{fmt(e)}%.'
        elif t==4:
          m=re.search(r'rises from (\d+) to (\d+(?:\.\d+)?)',q); e=(float(m[2])/int(m[1])-1)*100; exp=f'{fmt(e)}%.'
        elif t==5:
          m=re.search(r'(\d+)% tip on a \$(\d+)',q); e=int(m[1])*int(m[2])/100; exp=f'{money(e)}.'
        elif t==6:
          m=re.search(r'(\d+)-unit capacity is (\d+)% full',q); e=int(m[1])*int(m[2])/100; exp=f'{fmt(e)} units.'
        else:
          m=re.search(r'quantity of (\d+) decreases by (\d+)%',q); e=int(m[1])*(1-int(m[2])/100); exp=f'{fmt(e)}.'
        expect(row,exp)
      elif c=='ratio_proportion':
        if t==0:
          m=re.search(r'(\d+) notebooks cost \$(\d+).*what do (\d+) notebooks cost',q,re.I); e=int(m[2])/int(m[1])*int(m[3]); exp=f'{money(e)}.'
        elif t==1:
          m=re.search(r'ratio .*? (\d+):(\d+), with (\d+) tokens',q); e=int(m[3])/(int(m[1])+int(m[2]))*int(m[2]); exp=f'{fmt(e)}.'
        elif t==2:
          m=re.search(r'uses (\d+) cups? for (\d+) servings.*for (\d+) servings',q); e=int(m[1])*int(m[3])/int(m[2]); exp=f'{count_noun(e, "cup")}.'
        elif t==3:
          m=re.search(r'(\d+) miles per hour for (\d+) hours',q); exp=f'{int(m[1])*int(m[2])} miles.'
        elif t==4:
          m=re.search(r'1 cm represents (\d+) km.*does (\d+) cm',q); exp=f'{int(m[1])*int(m[2])} km.'
        elif t==5:
          m=re.search(r'each pack (\d+) boxes.*how many boxes do (\d+) workers',q,re.I); exp=f'{int(m[1])*int(m[2])} boxes.'
        elif t==6:
          m=re.search(r'in a (\d+):(\d+) ratio.*total is (\d+) cups?',q); e=int(m[3])*int(m[1])/(int(m[1])+int(m[2])); exp=f'{count_noun(e, "cup")}.'
        else:
          m=re.search(r'makes (\d+) pages in (\d+) minutes.*in (\d+) minutes',q); e=int(m[1])*int(m[3])/int(m[2]); exp=f'{fmt(e)} pages.'
        expect(row,exp)
      elif c=='unit_conversion':
        if t==0:
          m=re.search(r'Convert (\d+) hours? and (\d+) minutes?',q); exp=f'{count_noun(int(m[1])*60+int(m[2]), "minute")}.'
        elif t==1:
          m=re.search(r'Convert (\d+) minutes',q); mins=int(m[1]); h=mins//60; mi=mins%60; exp=f"{h} {'hour' if h==1 else 'hours'}{f' {mi} minutes' if mi else ''}."
        elif t==2:
          m=re.search(r'Convert (\d+) centimeters?',q); exp=f'{count_noun(int(m[1])/100, "meter")}.'
        elif t==3:
          m=re.search(r'Convert (\d+) grams?',q); exp=f'{count_noun(int(m[1])/1000, "kilogram")}.'
        elif t==4:
          m=re.search(r'Convert (\d+) milliliters?',q); exp=f'{count_noun(int(m[1])/1000, "liter")}.'
        elif t==5:
          m=re.search(r'(\d+) days',q); exp=f'{int(m[1])*24} hours.'
        elif t==6:
          m=re.search(r'Convert (\d+) kilometers?',q); exp=f'{count_noun(int(m[1])*1000, "meter")}.'
        else:
          m=re.search(r'Convert (\d+) seconds?',q); exp=f'{count_noun(int(m[1])/60, "minute")}.'
        expect(row,exp)
      elif c=='time_duration':
        times=re.findall(r'(\d{1,2}:\d{2} (?:AM|PM))',q)
        if t<=3:
          dur=int(re.search(r'lasts (\d+) minutes|takes (\d+) minutes|runs for (\d+) minutes',q).group(1) or re.search(r'lasts (\d+) minutes|takes (\d+) minutes|runs for (\d+) minutes',q).group(2) or re.search(r'lasts (\d+) minutes|takes (\d+) minutes|runs for (\d+) minutes',q).group(3)); exp=f'{show_time(parse_time(times[0])+dur)}.'
        else:
          dur=(parse_time(times[1])-parse_time(times[0]))%1440; exp=f'{dur} minutes.'
        expect(row,exp)
      elif c=='schedule_ordering':
        task=r'(?:draft|review|test|approve|deploy|backup|pack|leave|check in|board)'
        if t==0:
          m=re.search(rf'\b({task}) must happen before ({task}), and \2 before ({task})\.',q,re.I)
          exp=f'{m[1].strip().title()} → {m[2].strip().title()} → {m[3].strip().title()}.'
        elif t==1:
          m=re.search(rf'\b({task}) is at 9:00 AM, ({task}) at 10:30 AM, and ({task}) at 1:00 PM.*Which happens (first|last)',q,re.I)
          exp=f'{(m[1] if m[4].lower()=="first" else m[3]).strip().title()}.'
        elif t==2:
          m=re.search(rf'Do ({task}) after ({task}) but before ({task})\.',q,re.I)
          exp=f'{m[1].strip().title()}.'
        else:
          m=re.search(rf'\b({task}) and ({task}) must finish before ({task});',q,re.I)
          exp=f'{m[3].strip().title()}.'
        expect(row,exp)
      elif c=='comparison_ranking':
        if t==0:
          m=re.search(r'A scores (\d+), B scores (\d+), and C scores (\d+).*?(highest|lowest)',q); vals=list(map(int,m.groups()[:3])); idx=(max if m[4]=='highest' else min)(range(3),key=lambda i:vals[i]); exp=f'Option {"ABC"[idx]}.'
        elif t==1:
          m=re.search(r'A costs \$(\d+), B costs \$(\d+), and C costs \$(\d+).*?(cheapest|most expensive)',q); vals=list(map(int,m.groups()[:3])); idx=(min if m[4]=='cheapest' else max)(range(3),key=lambda i:vals[i]); exp=f'Option {"ABC"[idx]}.'
        elif t==2:
          m=re.search(r'A gives (\d+) hours for \$(\d+), B gives (\d+) hours for \$(\d+), and C gives (\d+) hours for \$(\d+).*?(best|worst)',q); g=list(map(int,m.groups()[:6])); vals=[g[0]/g[1],g[2]/g[3],g[4]/g[5]]; idx=(max if m[7]=='best' else min)(range(3),key=lambda i:vals[i]); exp=f'Option {"ABC"[idx]}.'
        else:
          m=re.search(r'A=(\d+) ms, B=(\d+) ms, C=(\d+) ms.*?(fastest|slowest)',q); vals=list(map(int,m.groups()[:3])); idx=(min if m[4]=='fastest' else max)(range(3),key=lambda i:vals[i]); exp=f'Option {"ABC"[idx]}.'
        expect(row,exp)
      elif c=='logical_implication':
        if t==0: exp='No—not from that information.' if 'is not a' in q else 'Yes.'
        elif t==1: exp='No—not necessarily.' if 'switch is off' in q else 'Yes.'
        elif t==2: exp='It cannot be determined.' if 'is not a' in q else 'No.'
        elif t==3: exp='No—the reverse does not follow.' if 'definitely signed' in q else 'Yes.'
        else: exp='No—not necessarily.' if 'This request has an ID.' in q else 'Yes.'
        expect(row,exp)
      elif c=='conditional_rules':
        if t==0:
          level=int(re.search(r'battery is (\d+)%',q)[1]); exp='Charge it.' if level<20 else 'No charge is required.'
        elif t==1: exp='A delivery fee applies.' if 'not a member' in q else 'Free delivery.'
        elif t==2: exp='No.' if 'invalid' in q else 'Yes.'
        elif t==3:
          temp=int(re.search(r'room is (\d+)°C',q)[1]); exp='Turn on the heat.' if temp<5 else 'Leave the heat off.'
        else: exp='No.' if 'failed' in q else 'Yes.'
        expect(row,exp)
      elif c=='contradiction_detection':
        if t==0:
          m=re.search(r'folder is (red|blue|green).*folder is (red|blue|green)',q); conflict=m[1]!=m[2]
        elif t==1:
          m=re.search(r'group has (\d+).*group has (\d+)',q); conflict=m[1]!=m[2]
        elif t==2:
          m=re.search(r'deadline is (Monday|Tuesday|Wednesday).*it is (Monday|Tuesday|Wednesday)',q); conflict=m[1]!=m[2]
        elif t==3: conflict='offline' in q and 'online at the same moment' in q
        else: conflict='has not arrived yet' in q
        exp='Yes, they conflict.' if conflict else 'No, they can both be true.'; expect(row,exp)
      elif c=='sequence_pattern':
        seq=list(map(int,re.search(r'What comes next: ([-0-9, ]+)\?',q)[1].split(', ')))
        s=seq[0]
        if t==0: e=seq[-1]+(seq[1]-seq[0])
        elif t==1: e=seq[-1]*(seq[1]//seq[0])
        elif t==2: e=seq[0]
        elif t==3: e=seq[-1]-(seq[0]-seq[1])
        elif t==4: e=s+10
        elif t==5: e=s+4
        elif t==6: e=seq[-1]+1
        else: e=s+6
        expect(row,f'{e}.')
      elif c=='counting_sets':
        if t==0:
          m=re.search(r'(\d+) people chose tea, (\d+) chose coffee, and (\d+) chose both',q); e=int(m[1])+int(m[2])-int(m[3])
        elif t==1:
          m=re.search(r'group of (\d+), (\d+) use app A, (\d+) use app B, and (\d+) use both',q); e=int(m[1])-(int(m[2])+int(m[3])-int(m[4]))
        elif t==2:
          m=re.search(r'(\d+) colors with (\d+) cards',q); e=int(m[1])*int(m[2])
        elif t==3:
          n=int(re.search(r'from (\d+) people',q)[1]); e=n*(n-1)//2
        elif t==4:
          m=re.search(r'(\d+) rows and (\d+) columns, with (\d+) cells? blocked',q); e=int(m[1])*int(m[2])-int(m[3])
        elif t==5:
          m=re.search(r'list has (\d+) entries\. (\d+) are removed and (\d+) new',q); e=int(m[1])-int(m[2])+int(m[3])
        elif t==6:
          m=re.search(r'(\d+) teams, each with (\d+) players and (\d+) substitute',q); e=int(m[1])*(int(m[2])+int(m[3]))
        else:
          m=re.search(r'(\d+) pages with (\d+) items',q); e=int(m[1])*int(m[2])
        expect(row,f'{e}.')
      elif c=='constraint_selection':
        labels='ABC'; chosen=a.replace('Option ','').replace('.','')
        if t==0:
          m=re.search(r'at most \$(\d+) and battery at least (\d+) hours\. A: \$(\d+), (\d+)h\. B: \$(\d+), (\d+)h\. C: \$(\d+), (\d+)h',q); budget,minh=map(int,m.groups()[:2]); vals=list(map(int,m.groups()[2:])); valid=[i for i in range(3) if vals[2*i]<=budget and vals[2*i+1]>=minh]
        elif t==1:
          m=re.search(r'no more than (\d+) kg and capacity at least (\d+) L\. A: (\d+) kg/(\d+) L\. B: (\d+) kg/(\d+) L\. C: (\d+) kg/(\d+) L',q); maxw,mins=map(int,m.groups()[:2]); vals=list(map(int,m.groups()[2:])); valid=[i for i in range(3) if vals[2*i]<=maxw and vals[2*i+1]>=mins]
        elif t==2:
          m=re.search(r'at most (\d+) minutes with at most (\d+) stops?\. A: (\d+) min/(\d+) stops?\. B: (\d+) min/(\d+) stops?\. C: (\d+) min/(\d+) stops?',q); maxt,maxs=map(int,m.groups()[:2]); vals=list(map(int,m.groups()[2:])); valid=[i for i in range(3) if vals[2*i]<=maxt and vals[2*i+1]<=maxs]
        else:
          m=re.search(r'quality at least (\d+) and price at most \$(\d+)\. A: (\d+)/\$(\d+)\. B: (\d+)/\$(\d+)\. C: (\d+)/\$(\d+)',q); mins,maxp=map(int,m.groups()[:2]); vals=list(map(int,m.groups()[2:])); valid=[i for i in range(3) if vals[2*i]>=mins and vals[2*i+1]<=maxp]
        checked[c]+=1
        if labels.index(chosen) not in valid: fail(row,f'chosen option not valid; valid={valid}')
      elif c=='spatial_relations':
        exps=['North.','Left.','Below.','Northeast.',None,'Under the table.','Southwest.','To the left.']
        if t==4:
          m=re.search(r'between (\w+) and (\w+), with \w+ on the left',q); exp=f'{m[2]}.'
        else: exp=exps[t]
        expect(row,exp)
      elif c=='estimation_sanity':
        if t==0:
          m=re.search(r'(\d+) items at roughly \$(\d+)',q); e=round(int(m[1])*int(m[2]),-1); exp=f'About {money(e)}.'
        elif t==1:
          w=int(re.search(r'weigh (\d+) kg',q)[1]); exp='Yes.' if w<=3 else 'No.'
        elif t==2:
          m=re.search(r'Walking (\d+) km in (\d+) minutes',q); exp='Yes, broadly.' if int(m[2])>=int(m[1])*8 else 'Probably not.'
        elif t==3:
          m=re.search(r'estimate (\d+) × (\d+)',q); e=round(int(m[1])*int(m[2]),-1); exp=f'About {e}.'
        elif t==4:
          m=re.search(r'pizzas serve (\d+) people',q); e=math.ceil(int(m[1])*2/8); exp=f'About {e} pizzas.'
        elif t==5:
          m=re.search(r'About (\d+)% of (\d+)',q); e=math.floor(int(m[1])*int(m[2])/100+0.5); exp=f'About {e}.'
        elif t==6:
          d=int(re.search(r'takes (\d+) minutes',q)[1]); exp='Yes.' if d<=15 else 'No.'
        else:
          m=re.search(r'about (\d+) units per day for (\d+) days',q); e=round(int(m[1])*int(m[2]),-1); exp=f'About {e}.'
        expect(row,exp)
      elif c=='evidence_inference':
        if t==0:
          m=re.search(r'Manual: (\w+) means (\w+).*device (\w+)',q); exp='Yes.' if m[2]==m[3] else 'No.'
        elif t==1:
          if 'Who left it' in q: exp="I don't know based on the available information."
          else:
            pl=re.search(r'left at the ([a-z ]+)\.',q)[1]; exp=f'At the {pl}.'
        elif t==2:
          if 'When was it created' in q: exp="I don't know based on the available information."
          else:
            owner=re.search(r'Record: (\w+) owns',q)[1]; exp=f'{owner}.'
        else:
          temp=int(re.search(r'Reading: (\d+)°C',q)[1]); exp='Yes.' if temp<5 else 'No.'
        expect(row,exp)
      elif c=='insufficient_information':
        unknown="I don't know based on the available information."
        if t==0:
          if 'Some apples' in q: exp=unknown
          else:
            m=re.search(r'starts with (\d+) apples and (\d+) are removed',q); exp=f'{int(m[1])-int(m[2])}.'
        elif t==1:
          if "price is missing" in q: exp=unknown
          else:
            m=re.search(r'A costs \$(\d+) and B costs \$(\d+)',q); exp=f'Option {"A" if int(m[1])<int(m[2]) else "B"}.'
        elif t==2:
          if 'speed is not given' in q: exp=unknown
          else:
            m=re.search(r'travels (\d+) mph for (\d+) hours',q); exp=f'{int(m[1])*int(m[2])} miles.'
        elif t==3:
          if 'percentage is missing' in q: exp=unknown
          else:
            m=re.search(r'What is (\d+)% of (\d+)',q); exp=f'{fmt(int(m[1])*int(m[2])/100)}.'
        else:
          if 'gives no name' in q: exp=unknown
          else: exp=f'{re.search(r"says (\w+) booked",q)[1]}.'
        expect(row,exp)
      elif c=='multi_turn_context_update':
        us=users(row); correction=us[-2]
        if t==0: exp=f'{re.search(r"it is ([A-Za-z]+)",correction)[1]}.'
        elif t==1: exp=f'The {re.search(r"use the ([a-z]+) route",correction,re.I)[1]} route.'
        elif t==2: exp=f'{re.search(r"moved to ([0-9:]+ [AP]M)",correction)[1]}.'
        elif t==3: exp=f'The {re.search(r"to the ([a-z]+) folder",correction)[1]} folder.'
        elif t==4: exp=f'${re.search(r"to \$(\d+)",correction)[1]}.'
        elif t==5: exp=f'{re.search(r"Actually, (\w+) is",correction)[1]}.'
        elif t==6: exp=f'By {re.search(r"Use (.+) instead",correction)[1].rstrip(".")}.'
        elif t==7: exp=f'The {re.search(r"Switch to the ([a-z]+) model",correction)[1]} model.'
        elif t==8: exp=f'At the {re.search(r"changed to the ([a-z]+)",correction)[1]}.'
        elif t==9: exp=f'{re.search(r"moved to (\w+)",correction)[1]}.'
        elif t==10: exp=f'{re.search(r"quantity to (\d+)",correction)[1]}.'
        elif t==11: exp=f'{re.search(r"Switch to (.+)",correction)[1].rstrip(".")}.'
        elif t==12: exp=f'{re.search(r"now ([a-z]+)",correction)[1].title()}.'
        elif t==13: exp=f'{re.search(r"renamed it to ([\w.]+)",correction)[1].rstrip(".")}.'
        elif t==14: exp=f'After {re.search(r"after ([a-z]+)",correction)[1]}.'
        else: exp=f'{re.search(r"Use ([A-Za-z]+) instead",correction)[1]}.'
        expect(row,exp)
      elif c=='evidence_grounded_dialogue':
        if t==0: exp=f'At the {re.search(r"is at the ([a-z]+)"," ".join(users(row)))[1]}.'
        elif t in {1,7,15}: exp="I don't know based on the available information."
        elif t in {2,13}: exp='The provided information conflicts.'
        elif t==3: exp=f'{re.search(r"Record: (\w+) owns", " ".join(users(row)))[1]}.'
        elif t==4: exp='Close the editor.'
        elif t==5: exp='No.'
        elif t==6: exp='4 PM.'
        elif t==8:
          m=re.search(r'(\d+) checks ran and (\d+) failed'," ".join(users(row))); exp=f'{int(m[1])-int(m[2])}.'
        elif t==9: exp='No.'
        elif t==10: exp=f'In the {re.search(r"drawer is in the ([a-z]+)"," ".join(users(row)))[1]}.'
        elif t==11: exp='No—staff are not visitors.'
        elif t==12: exp='Encrypt the export.'
        else: exp='Approved.'
        expect(row,exp)
      else:
        return
    except Exception as exc:
      fail(row,f'checker exception: {type(exc).__name__}: {exc}')

rows=read(SFT)
for row in rows: check_row(row)

# Corpus-level textual quality checks.
control=[]; bad_phrases=[]
literal_bad=['Context tag:', 'project update update', '\ufffd']
regex_bad=[(r'\b1 substitutes\b','1 substitutes'),(r'\b1 hours\b','1 hours')]
for r in rows:
    for m in r['messages']:
        if any(ord(ch)<32 and ch not in '\n\t' for ch in m['content']): control.append((r['id'],m['role']))
    alltext=' '.join(m['content'] for m in r['messages'])
    for phrase in literal_bad:
        if phrase in alltext: bad_phrases.append((r['id'],phrase))
    for pattern,label in regex_bad:
        if re.search(pattern,alltext): bad_phrases.append((r['id'],label))

# DPO contract and hardness diagnostics.
dpo=read(DPO); dpo_same=[]; dpo_source_mismatch=[]; dpo_rejected_equals_other_chosen=[]
source={r['id']:r for r in rows}
for r in dpo:
    if r['chosen'].strip()==r['rejected'].strip(): dpo_same.append(r['id'])
    src=source.get(r['metadata']['source_record_id'])
    if not src or r['chosen']!=ans(src): dpo_source_mismatch.append(r['id'])

report={
  'status':'PASS' if not errors and not control and not bad_phrases and not dpo_same and not dpo_source_mismatch else 'FAIL',
  'sft_rows':len(rows),
  'semantic_rows_checked':sum(checked.values()),
  'checks_by_class':dict(sorted(checked.items())),
  'semantic_error_count':len(errors),
  'semantic_error_counts_by_class':dict(sorted(class_errors.items())),
  'semantic_errors':errors[:100],
  'control_character_issues':control[:100],
  'bad_phrase_issues':bad_phrases[:100],
  'dpo_pairs':len(dpo),
  'dpo_same_choice_count':len(dpo_same),
  'dpo_source_mismatch_count':len(dpo_source_mismatch),
  'note':'Deterministic checks validate recomputable/template-grounded classes; human review remains required for open-ended communication, safety, empathy, and preference quality.',
}
print(json.dumps(report,indent=2,ensure_ascii=False))
(ROOT/'reports/semantic_audit_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
if report['status']!='PASS': raise SystemExit(1)
