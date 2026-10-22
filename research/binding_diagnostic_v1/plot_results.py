"""Create a scientific figure from complete comparable diagnostic results only."""
import json
from pathlib import Path
from paths import ROOT
from common import read, sha, write_new, require


def main():
    source = ROOT / 'RESULTS.json'
    source_sha = sha(source)
    data = read(source)
    require(data['status'] == 'COMPLETE_COMPARABLE', 'completed comparable results required')
    outputs = [ROOT / ('DIAGNOSTIC_COMPARISON' + extension) for extension in ('.png', '.svg', '_FIGURE.json')]
    require(not any(path.exists() for path in outputs), 'figure outputs are create-only')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    roles = ('source', 'lora', 'mixed')
    labels = {'source': 'Source', 'lora': 'LoRA', 'mixed': 'Mixture'}
    colors = {'source': '#a8b2bc', 'lora': '#617589', 'mixed': '#287d70'}
    cohort = data['source_conditioned_retention_descriptive']
    panels = [
        ('Binding selectivity', 32, roles,
         [data['endpoints'][role]['parent_counts']['binding']['canonical_complete_exact/all16'] for role in roles],
         'All 16 related responses correct'),
        ('Arithmetic anchors', 96, roles,
         [data['endpoints'][role]['parent_counts']['arithmetic']['strict_anchor'] for role in roles],
         'Unselected, strict explicit-answer anchors'),
        ('Source-known arithmetic', cohort['n'], ('lora', 'mixed'),
         [cohort['models'][role]['retained'] for role in ('lora', 'mixed')],
         'Same source-successful anchors; descriptive rates'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.7))
    fig.subplots_adjust(left=.055, right=.98, bottom=.34, top=.76, wspace=.25)
    plotted = []
    for ax, (title, denominator, selected, values, subtitle) in zip(axes, panels):
        bars = ax.bar(range(len(selected)), values, color=[colors[role] for role in selected], width=.58, zorder=3)
        ax.bar_label(bars, labels=[f'{value}/{denominator}' for value in values], padding=5, fontsize=11)
        ax.set_xticks(range(len(selected)), [labels[role] for role in selected])
        ax.set_ylim(0, max(denominator, 1) * 1.18)
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=14)
        ax.grid(axis='y', color='#e4e9ec', zorder=0)
        ax.set_ylabel('Correct parent problems')
        ax.text(0, -.19, subtitle, transform=ax.transAxes, fontsize=8.7, color='#44515f')
        plotted.append({'title': title, 'n': denominator, 'roles': selected, 'counts': values})
    fig.suptitle('Fresh diagnostic: selectivity and preservation', x=.055, y=.956, ha='left', fontsize=17, fontweight='bold')
    fig.text(.055, .90, 'Three frozen 596M models · identical 896-view bank · 32 binding parents and 96 arithmetic parents', color='#44515f')
    fig.text(.055, .845, 'The earlier failed mixture criteria remain failed. This diagnostic cannot promote a model.', color='#8b3d34', fontsize=10)
    binding = data['primary']['binding']; arithmetic = data['primary']['arithmetic']
    def describe(value):
        if not value['observed_test']:
            return 'No source reference cohort; arithmetic inference unavailable.'
        return f"net={value['net']:+d}, nd={value['nd']}, exact p={value['exact_p']:.4g}, Holm p={value['holm_adjusted_p']:.4g}"
    fig.text(.055, .197, 'Primary mixture − LoRA binding contrast: ' + describe(binding), fontsize=9)
    fig.text(.055, .155, 'Primary mixture − LoRA conditional arithmetic contrast: ' + describe(arithmetic), fontsize=9)
    fig.text(.055, .112, f"Reference cohort: N={cohort['n']}; {arithmetic['reference_cohort_status']}. Eligibility uses source output only.", fontsize=9)
    fig.text(.055, .068, 'Views within each parent are correlated. Relative retention and nonsignificance do not establish preservation or equivalence.', fontsize=9)
    require(sha(source) == source_sha, 'results changed during figure construction')
    for path, fmt in zip(outputs, ('png', 'svg')):
        with path.open('xb') as stream:
            fig.savefig(stream, format=fmt, dpi=175, facecolor='white')
    plt.close(fig)
    require(sha(source) == source_sha, 'results changed before figure receipt')
    write_new(outputs[2], {'schema': 1, 'status': 'COMPLETE', 'results_sha256': source_sha,
        'plotter_sha256': sha(Path(__file__).resolve()), 'plots': plotted,
        'outputs': {path.name: sha(path) for path in outputs[:2]}, 'quality_claim': False})
    print(json.dumps({'status': 'COMPLETE', 'outputs': [str(path) for path in outputs]}))


if __name__ == '__main__':
    main()
