"""Standalone scientific figure for exact toy identities; native evidence excluded."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import exact_math as m


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if not (args.out/'RECEIPT.json').is_file():
        raise SystemExit('exact audit receipt required')
    for ext in ('png','pdf'):
        if (args.out/f'method_mechanics.{ext}').exists():
            raise SystemExit('figure exists (create-only)')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    logp=np.geomspace(1e-4,.9999,200)
    for method,label,color in [('reinforce','REINFORCE / RLOO','#0072B2'),
                               ('grpo_sample','GRPO (sample SD)','#009E73'),
                               ('maxrl_practical','Practical MaxRL (T=7)','#D55E00'),
                               ('maxrl_success','Success-average MaxRL (T=8)','#CC79A7')]:
        values=[m.gradient_statistics(method,float(p),8)['expected_logit_gradient'] for p in logp]
        axs[0,0].plot(logp,values,label=label,color=color)
    axs[0,0].set(xscale='log',xlabel='Success probability p',ylabel='Expected logit update',title='A. Fixed N=8, epsilon=0; scales differ')
    axs[0,0].legend(fontsize=8)
    linear=np.linspace(0,1,401)
    for n,color in [(2,'#56B4E9'),(4,'#009E73'),(8,'#D55E00'),(16,'#CC79A7')]:
        axs[0,1].plot(linear,[m.mixed_probability(float(p),n) for p in linear],label=f'N={n}',color=color)
    axs[0,1].axhline(.1,color='#555555',ls='--',lw=1,label='m_min=0.10')
    axs[0,1].set(xlabel='Success probability p',ylabel='Probability of mixed group',title='B. Centered signal vanishes at both extremes',ylim=(0,1.05))
    axs[0,1].legend(fontsize=8)
    for gamma,color in [(1,'#D55E00'),(.5,'#009E73')]:
        values=[m.bounded_failure_continuation(float(p),8,4,gamma)['expected_draws'] for p in logp]
        axs[1,0].plot(logp,values,color=color,label=f'Continue all-fail with gamma={gamma}')
    axs[1,0].axhline(8,color='#555555',ls=':',label='Initial group: 8 draws')
    axs[1,0].axhline(32,color='#555555',ls='--',label='Hard cap: 32 draws')
    axs[1,0].set(xscale='log',xlabel='Success probability p',ylabel='Expected attempted draws',title='C. Bounded continuation spends more on hard inputs')
    axs[1,0].legend(fontsize=8)
    for label,values,color in [
        ('Fixed N=4',[m.maxrl_closed_gradient(float(p),4,practical=True) for p in logp],'#0072B2'),
        ('Fixed N=16',[m.maxrl_closed_gradient(float(p),16,practical=True) for p in logp],'#CC79A7'),
        ('Stop after success; 4x4 cap',[m.bounded_failure_continuation(float(p),4,4,1)['expected_pooled_practical_gradient'] for p in logp],'#D55E00')]:
        axs[1,1].plot(logp,values,label=label,color=color)
    axs[1,1].set(xscale='log',xlabel='Success probability p',ylabel='Expected logit update',title='D. Pooled stopped samples change the estimator')
    axs[1,1].legend(fontsize=8)
    for ax in axs.flat:
        ax.grid(alpha=.2)
    fig.suptitle('Exact Bernoulli mechanisms: acquisition signal, cost and estimator identity\nThese curves do not measure language-model acquisition or retention.',fontsize=13)
    files=[]
    for ext in ('png','pdf'):
        path=args.out/f'method_mechanics.{ext}'
        fig.savefig(path,dpi=180)
        files.append(path)
    plt.close(fig)
    receipt={'kind':'exact-toy scientific figure','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'matplotlib':matplotlib.__version__,'numpy':np.__version__,
             'outputs_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (args.out/'FIGURE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(str(files[0].resolve()))


if __name__=='__main__':
    main()
