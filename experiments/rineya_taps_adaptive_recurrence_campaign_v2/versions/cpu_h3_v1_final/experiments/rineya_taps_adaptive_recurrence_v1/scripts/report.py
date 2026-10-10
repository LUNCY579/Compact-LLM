"""Derive human-readable tables and compact records from an immutable fixture run."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def report(run, output):
    run=Path(run); output=Path(output)
    summary=json.loads((run/'summary.json').read_text())
    ledger=json.loads((run/'compute_ledger.json').read_text())
    def table(filename, columns):
        rows=list(csv.DictReader((run/filename).open()))
        lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
        lines += ['| '+' | '.join(str(r[k]) if r[k] else '—' for k in columns)+' |' for r in rows]
        return '\n'.join(lines)
    text=f'''# TAPS-RINEYA-001 research report

## Decision: BLOCKED for native scientific qualification

The inference/diagnostic apparatus is implemented. No Rineya checkpoint was loaded,
no native benchmark was run, and no training/co-design was attempted. The tables
below describe five public mathematical fixtures per split, not reasoning tasks.
Two RNG seeds are technical repetitions, not ten independent problems.

## Question and adaptation

Scale the live residual of a repeatedly applied frozen middle block according to
bias-corrected progress/fluctuation moments. Preserve the ordinary first pass and
seed history from its whole-block update. This differs from changing whether a
block runs, adding loop iterations, or learning a recurrence gate. For native
ARMR the initial lane has one extra application only; the old failed three-total-
application arm is NOT licensed. The controller formula follows the paper; this
full-residual ARMR host is not its eta/3 Qwen host or co-designed TRM checkpoint.

## Controlled outcomes (generated from headroom.csv)

The fixed comparator here is the retrospective best GLOBAL fixed candidate on
that split. TEST selection by this comparator is diagnostic only; the deployable
fixed comparator is frozen from DEV, separately present in the CSV files. A4 is
label-informed and never deployable. Missing step-oracle entries mean the frozen
search cap prevented enumeration, not zero headroom. Capture is undefined when
headroom is zero; ratios are never clipped or padded with invented zeros.

'''+table('headroom.csv',['split','extra_loops','oracle','retrospective_global_fixed_accuracy','taps_accuracy','oracle_accuracy','adaptive_headroom','scheduler_capture_ratio'])+f'''

## Computation actually executed

Physical apparatus rollouts: {ledger['physical_rollouts']}. Total core calls,
including oracle search: {ledger['core_calls_all_including_oracle_search']}.
Oracle search accounts for {ledger['oracle_search_rollouts']} physical rollouts
and {ledger['oracle_search_core_calls']} core calls. No oracle work is attributed
to a deployable controller. All ordinary arms at one loop budget have identical
core-call counts. Full timing totals, including controller and logging overhead,
are generated in compute_ledger.json. Tiny tensor timings are NOT GPU or native
Rineya speed estimates. The controller dominates these intentionally trivial
fixture updates; that is not evidence about Transformer-block overhead.

## Retrospective accuracy-target comparison

This is a sweep over fixed budgets, not an online policy that observes labels to
stop. Full accuracy/latency curves and controller timing are in
accuracy_loops_latency.csv. Native loop savings remain unmeasured.

'''+table('accuracy_target.csv',['split','arm','accuracy_target','extra_loops','instrumented_wall_seconds'])+'''

## Required interpretation

1. **Is scale useful in Rineya?** Unknown. Different update-scale preferences in
   mathematical systems establish instrument sensitivity only.
2. **Adaptive headroom beyond a global fixed scale?** Present in one DEV fixture
   contrast above, not established for native Rineya; the fixture TEST contrast
   does not preserve that headroom. This null result is retained.
3. **Do trajectory statistics predict useful scale?** Aligned, oscillating,
   coherent-but-wrong and zero-gated systems are present by construction. TAPS
   does not capture the nonzero DEV accuracy headroom. No native predictive
   calibration, causal accuracy model, or reliable proxy has been established.
4. **Fewer loops?** No native evidence. Retrospective fixture budget tables are
   only a harness exercise and cannot license a dynamic stopping policy.
5. **Accuracy at fixed loops?** No native evidence; the fixture comparison to the
   strongest global fixed-alpha baseline does not show a TAPS advantage.
6. **Overhead worthwhile?** Unknown on a real block. Separate measured controller
   and core timings are available, but both include diagnostic-path conventions.
7. **Cross-task generalization?** Not established. These are five analytic systems,
   not math, language, logical reasoning or state-tracking benchmarks.
8. **Interaction with existing gates?** The native bridge targets ungated ARMR.
   The zero-update fixture confirms alpha cannot revive a zero update. A real
   recurrent-adapter/gate factorial still requires a separately qualified host.

## Existing research and claim boundary

ARMR provides the audited full-prefix no-cache middle-block location and frozen
MERGEDEV-fast scorer. CDE concerns depth/width allocation; LoopSpec concerns
scheduling/cache parity; CED paired-host and RRP have independent qualification,
training and data boundaries. None of their closed gates is opened by this work.
See REPOSITORY_AUDIT.md for exactly what was reviewed and the audit limits.

## Recommendation and next experiment

**BLOCKED**, not PROMOTE or REJECT. On the original qualified CPU environment,
restore the exact canonical Supa source and inherited checkout, run preflight,
then fresh native unit identity and all-arm smoke gates. Inspect the complete
DEV alpha grid and trajectory variation at one extra L12–L13 application before
requesting any fresh protected evaluation or larger loop budget. First ask if
per-example oracle scale exceeds the best global fixed scale. If not, do not
add a more complex controller or begin co-design training.

The branch is suitable for apparatus/code review, not a production default,
verified native experiment, exact empirical paper reproduction, or automatic
merge. Native loader execution and the full inherited suite remain outstanding.
'''
    with output.open('x') as f:f.write(text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();report(a.run,a.out)
