"""Export descriptive figures from the immutable complete generation report."""
import argparse
import hashlib
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPORT_SHA = '67c7d5d9fc128b148afc6d056cb26f9ecef4e05dde78a7405358125250cb9d21'


def main(out):
    out = Path(out).resolve()
    if PACKAGE not in out.parents:
        raise ValueError('campaign output only')
    out.mkdir(exist_ok=False, parents=True)
    report_path = PACKAGE/'analyses/bulk-restart-manifest-reviewed-001/REPORT.json'
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == REPORT_SHA
    report = json.loads(report_path.read_text())
    assert report['complete'] and not report['errors']
    os.environ['MPLCONFIGDIR'] = str(out/'mpl_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    artifacts = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), sharey=True)
    for ax, split, title in zip(axes, ('DEVELOPMENT', 'OPEN_VALIDATION'),
                              ('Development: 192 cases / 96 parents', 'Exploratory validation: 96 cases / 48 parents')):
        b = report['reports'][split]
        curve = b['fixed_curve']
        score = {row['policy']: row['accuracy'] for row in b['tables']}
        ax.plot([v['alpha'] for v in curve], [100*v['accuracy'] for v in curve],
                marker='o', color='#246c9e', linewidth=2, label='Fixed scale')
        ax.axhline(100*score['A3'], color='#be6425', linewidth=2, linestyle='--', label='TAPS = random control')
        ax.axhline(100*score['A0'], color='#777777', linewidth=1.6, linestyle=':', label='Intact model')
        ax.axhline(100*b['constant_oracle']['accuracy'], color='#6a8b41', linewidth=1.5,
                   linestyle='-.', label='Retrospective finite-grid oracle')
        floor = b['own_bank_blind_floors']['overall']['query_blind_last_entity_solver']['parent_category_accuracy']
        ax.axhline(100*floor, color='#9556a0', linewidth=1.5, label='Own-bank query-blind floor')
        ax.set(title=title, xlabel='Extra-block residual scale', xlim=(.78, 1.22), ylim=(28, 46))
        ax.set_xticks([.8, .9, 1, 1.1, 1.2])
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Strict answer correctness (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(.53, .18), ncol=3, fontsize=10)
    fig.suptitle('Scaling adds no observed adaptive value beyond the fixed control', fontsize=16, y=.97)
    fig.subplots_adjust(left=.08, right=.98, top=.83, bottom=.32, wspace=.12)
    fig.text(.08, .045, 'Same frozen checkpoint, one extra L12–L13 block; six shared authored generators.\n'
             'Fixed alpha 1.0 selected on DEV. Random seeds averaged within cases; oracle is diagnostic only.\n'
             'Previously exposed validation; no confidence intervals, equivalence or source-confirmation claim.', fontsize=10)
    for ext in ('png', 'svg'):
        name='fixed_scale_curves.'+ext
        fig.savefig(out/name, dpi=160)
        artifacts.append(name)
    plt.close(fig)
    labels = ['Deduction', 'Signed\nledgers', 'Program\nstate', 'Dialogue\nmemory', 'Document\nretrieval', 'Source\nconflict']
    categories = ['deduction_composition', 'signed_ledgers', 'program_state', 'dialogue_memory',
                  'document_retrieval_reconciliation', 'answerability_source_conflict']
    b = report['reports']['OPEN_VALIDATION']
    fig, ax = plt.subplots(figsize=(13, 5.8))
    for offset, policy, color, label in ((-.25, 'A0', '#777777', 'Intact'), (0, 'A1', '#246c9e', 'Unit / frozen fixed'),
                                       (.25, 'A3', '#be6425', 'TAPS / matched random')):
        values = [100*b['family'][c]['policy_endpoints'][policy]['score'] for c in categories]
        bars = ax.bar([i+offset for i in range(6)], values, width=.24, color=color, label=label)
        ax.bar_label(bars, labels=[str(round(v*.16))+'/16' for v in values], fontsize=10, padding=3)
    ax.set_xticks(range(6), labels)
    ax.set(ylabel='Strict answer correctness (%)', ylim=(0, 78))
    fig.suptitle('Validation gains and harms vary by task family', fontsize=16, y=.97)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.54, .90), ncol=3, fontsize=11)
    fig.subplots_adjust(left=.08, right=.98, top=.77, bottom=.27)
    fig.text(.08, .075, '16 cases / 8 related parents per category; 48 validation parents share six generators.\n'
             'TAPS ties fixed overall: one dialogue gain offsets one document loss.\n'
             'Counts are descriptive; arithmetic/program failures and abstention harms prevent promotion.', fontsize=10)
    for ext in ('png', 'svg'):
        name='validation_families.'+ext
        fig.savefig(out/name, dpi=160)
        artifacts.append(name)
    plt.close(fig)
    manifest={'source_report':str(report_path), 'source_report_sha256':REPORT_SHA,
              'plot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'artifacts':{n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in artifacts},
              'native_model_calls':0, 'scope':'Descriptive export of complete report; no new estimator or endpoint'}
    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
