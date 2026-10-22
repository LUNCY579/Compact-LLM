"""Plot the completed native screen, with no acquisition claim."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    base = Path(__file__).resolve().parent / 'evidence'
    source = base / 'NATIVE_WARM_COMPETENCE_20261001.json'
    value = json.loads(source.read_text())
    assert value['completed'] and value['outcome'] == 'STOP_NO_SIGNAL'
    skills = ['R', 'S', 'P']
    summary = value['summary']
    mixed = [100 * summary[s]['groups']['mixed'] / summary[s]['prompts'] for s in skills]
    success = [100 * summary[s]['successes'] / summary[s]['samples'] for s in skills]
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6))
    colors = ['#b54c4c', '#3673a8', '#888888']
    for ax, rates, title, ylabel, labels in [
        (axes[0], mixed, 'Mixed reward groups', 'Percent of 150 groups',
         ['5/150', '72/150', '50/150']),
        (axes[1], success, 'Strict sampled successes', 'Percent of 1,200 samples',
         ['6/1,200', '179/1,200', '285/1,200'])]:
        ax.bar(skills, rates, color=colors, width=.58)
        ax.set(title=title, ylabel=ylabel, ylim=(0, max(rates) * 1.24))
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        for i, (rate, label) in enumerate(zip(rates, labels)):
            ax.text(i, rate + max(rates) * .025, f'{rate:.2f}%\n{label}', ha='center', va='bottom', fontsize=10)
    axes[0].axhline(10, color='#b54c4c', linestyle='--', linewidth=1.2)
    axes[0].text(.02, 11.3, 'R/S gate: 10%', color='#973b3b', fontsize=10)
    fig.suptitle('Complete native warm-anchor screen: STOP_NO_SIGNAL', fontsize=14)
    fig.text(.5, .035, 'G = 8; 450 prompts. R fails the mixed-group gate. P is descriptive.\n'
             'Sample successes are baseline diagnostics, not acquired-skill or retention evidence.',
             ha='center', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.085, right=.97, top=.84, bottom=.22, wspace=.34)
    outputs = {}
    for suffix in ('png', 'pdf'):
        path = base / ('native_warm_signal_20261001.' + suffix)
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180)
        outputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    receipt = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'plot_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'outputs_sha256': outputs, 'model_loads': 0, 'optimizer_steps': 0,
               'classification': 'measured completed baseline diagnostic; no acquisition claim'}
    with (base / 'NATIVE_WARM_SIGNAL_FIGURE_20261001.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)


if __name__ == '__main__':
    main()
