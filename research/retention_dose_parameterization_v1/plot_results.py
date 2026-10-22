"""Plot only a completed, comparable result; never invent or fill missing arms."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--name',default='FRONTIER')
    a=ap.parse_args()
    if not a.name or Path(a.name).name!=a.name or a.name in ('.','..'):
        raise ValueError('figure name must be one filename component')
    source=a.root/'RESULTS.json'
    report=json.loads(source.read_text(encoding='utf-8'))
    if report.get('status')!='COMPLETE_COMPARABLE':
        raise ValueError('plotting requires completed comparability gates')
    destinations=[a.root/(a.name+'.png'),a.root/(a.name+'.svg'),a.root/(a.name+'_FIGURE.json')]
    if any(p.exists() for p in destinations):
        raise FileExistsError('figure outputs are create-only')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    colors={'a025':'#2879b9','a050':'#238a67','a075':'#bf7932','a100':'#414955'}
    order=list(colors)
    endpoints=report['endpoints']
    xvalues=[endpoints[arm][bank]['correct'] for bank in ('dose','select') for arm in order]
    yvalues=[endpoints[arm]['retention']['policies']['permissive']['references_retained'] for arm in order]
    xlo=max(0,min(xvalues+[284])-12); xhi=min(384,max(xvalues+[284])+12)
    ylo=max(0,min(yvalues+[40])-3)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.4),sharex=True,sharey=True)
    for ax,bank,title in zip(axes,('dose','select'),('Dose bank · 384 held-out parents','SELECT bank · 384 held-out parents')):
        ax.set_title(title,loc='left',fontweight='bold',pad=14)
        ax.set_xlim(xlo,xhi);ax.set_ylim(ylo,49)
        ax.set_yticks([tick for tick in (0,5,10,15,20,25,30,35,40,45,47) if tick>=ylo])
        ax.grid(axis='y',color='#e8ecf0',linewidth=.8)
        ax.set_axisbelow(True)
        if bank=='dose':
            ax.add_patch(Rectangle((284,40),384-284,7,facecolor='#e4f1e9',edgecolor='none',zorder=0))
            ax.axvline(284,color='#87aa94',linestyle='--',linewidth=1)
            ax.axhline(40,color='#87aa94',linestyle='--',linewidth=1)
        points=[(endpoints[arm][bank]['correct'],endpoints[arm]['retention']['policies']['permissive']['references_retained']) for arm in order]
        ax.plot([p[0] for p in points],[p[1] for p in points],color='#bac4cf',linewidth=1.2,zorder=1)
        grouped={}
        for arm,(x,y) in zip(order,points):
            ax.scatter(x,y,s=72,color=colors[arm],edgecolors='white',linewidth=.9,zorder=3)
            grouped.setdefault((x,y),[]).append(f"{endpoints[arm]['alpha']:.2f}")
        for (x,y),alphas in grouped.items():
            offset=-17 if alphas==['1.00'] else 10
            ax.annotate('α='+', '.join(alphas),(x,y),xytext=(0,offset),textcoords='offset points',ha='center',fontsize=9)
        ax.set_xlabel('Acquisition answers correct / 384',labelpad=9)
    axes[0].set_ylabel('Frozen source answers retained / 47',labelpad=9)
    fig.suptitle('High dose + interpolation: one trained checkpoint, four scalar arms',x=.065,ha='left',fontsize=15,fontweight='bold')
    fig.text(.065,.085,'Shading: fixed dose-bank target (acquisition ≥284, retention ≥40); arithmetic ≥62 is an additional requirement.',fontsize=9,color='#485363')
    fig.text(.065,.045,'Banks remain separate. Lines connect α values, not independent training draws. Exact paired statistics are in RESULTS.md.',fontsize=9,color='#485363')
    fig.subplots_adjust(top=.82,bottom=.23,left=.065,right=.98,wspace=.14)
    fig.savefig(destinations[0],dpi=180,facecolor='white')
    fig.savefig(destinations[1],facecolor='white')
    plt.close(fig)
    metadata={'source_results_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'matplotlib':matplotlib.__version__,
              'plotter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destinations[:2]},'quality_claim':False}
    with destinations[2].open('x',encoding='utf-8',newline='\n') as f:
        json.dump(metadata,f,indent=2);f.write('\n')
    print(destinations[0])

if __name__=='__main__':
    main()
