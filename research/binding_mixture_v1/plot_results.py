"""Plot only complete comparable mixture results; offline, create-only artifacts."""
import hashlib
import json
from pathlib import Path
from paths import OUT, sha, read, write


def main():
    source = OUT / 'RESULTS.json'
    source_sha = sha(source)
    report = read(source)
    assert report['status'] == 'COMPLETE_COMPARABLE'
    assert report['experiment'] == 'BINDING-MIXTURE-001'
    outputs = [OUT / ('MIXTURE_COMPARISON' + suffix) for suffix in ('.png', '.svg', '_FIGURE.json')]
    assert not any(path.exists() for path in outputs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 7.5))
    fig.subplots_adjust(left=.07, right=.98, top=.76, bottom=.32, wspace=.22)
    colors = {'lora': '#617589', 'mixed': '#287d70'}
    names = {'lora': 'Original LoRA', 'mixed': 'Binding mixture'}
    measured = {}
    for role, offset in (('lora', -.19), ('mixed', .19)):
        endpoint = report['endpoints'][role]
        retention = endpoint['retention']['policies']['permissive']
        counts = [endpoint['dose']['correct'], endpoint['select']['correct'], retention['references_retained'], retention['panel_correct']]
        denominators = [384, 384, 47, 150]
        binding = report['binding']['endpoints'][role]
        categories = ('person_missing', 'person_vacant', 'person_answerable', 'role_answerable')
        binding_counts = [binding['categories'][category]['canonical_complete_exact'] for category in categories]
        measured[role] = {'standard_counts': counts, 'standard_denominators': denominators, 'binding_counts': binding_counts}
        for ax, values, denoms in ((axes[0], counts, denominators), (axes[1], binding_counts, [96] * 4)):
            heights = [100 * count / denom for count, denom in zip(values, denoms)]
            bars = ax.bar(np.arange(4) + offset, heights, width=.35, color=colors[role], label=names[role], zorder=3)
            ax.bar_label(bars, labels=[str(value) for value in values], fontsize=9, padding=4)
    axes[0].set_xticks(range(4), ['Dose\n/384', 'SELECT\n/384', 'Retained\n/47', 'Arithmetic\n/150'])
    axes[1].set_xticks(range(4), ['Missing\n/96', 'Vacant\n/96', 'Person\n/96', 'Role\n/96'])
    axes[0].set_title('Earlier skills', loc='left', fontweight='bold', pad=15)
    axes[1].set_title('Fresh binding DEV', loc='left', fontweight='bold', pad=15)
    for ax in axes:
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
        ax.grid(axis='y', color='#e3e8eb', zorder=0)
    axes[0].set_ylabel('Answers correct (count labels above bars)')
    fig.suptitle('One mixed-data run: skill gains and their costs', x=.07, y=.962, ha='left', fontsize=17, fontweight='bold')
    fig.text(.07, .907, '596M parameters · same starting model and LoRA settings · 2,697 original + 768 binding training views', color='#44515f')
    verdict = report['predictions']['primary_engineering_conjunction']
    efficacy = report['predictions']['binding_efficacy']
    fig.text(.07, .861, f'Registered joint criterion: {verdict}   |   Binding efficacy: {efficacy}', fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.53, .22), ncol=2, frameon=False)
    contrasts = report['acquisition_contrasts_mixed_minus_control']
    text = 'Acquisition contrasts (mixture − original): ' + ' · '.join(
        f"{bank} (net={v['net']:+d}, nd={v['nd']}, exact p={v['exact_p']:.4g})" for bank, v in contrasts.items())
    fig.text(.07, .185, text, fontsize=9)
    fig.text(.07, .14, 'Each acquisition bank is separate; cross-draw claims also require |net| ≥20. Retention uses the fixed 47-source cohort.', fontsize=9)
    fig.text(.07, .101, 'Binding counts are descriptive. Inference uses 32 worlds, all three category views correct, with Holm(2) for UNKNOWN.', fontsize=9)
    fig.text(.07, .06, 'Exploratory measurements; extra training data and compute change together. Strict scoring and full paired tests are in RESULTS.md.', fontsize=9)
    assert sha(source) == source_sha
    for path, fmt in zip(outputs, ('png', 'svg')):
        with path.open('xb') as stream:
            fig.savefig(stream, format=fmt, dpi=170, facecolor='white')
    plt.close(fig)
    assert sha(source) == source_sha
    write(outputs[2], {'schema': 1, 'status': 'COMPLETE', 'source_results_sha256': source_sha,
        'plotter_sha256': sha(Path(__file__)), 'plotted_values': measured,
        'outputs': {path.name: sha(path) for path in outputs[:2]}, 'quality_claim': False})
    print(json.dumps({'status': 'COMPLETE', 'outputs': [str(path) for path in outputs]}))


if __name__ == '__main__':
    main()
