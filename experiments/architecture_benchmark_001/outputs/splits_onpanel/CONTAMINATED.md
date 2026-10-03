# QUARANTINED — this split leaks the evaluation panel. Do not train on it. Do not select on it.

Measured 2026-09-14 against `experiments/architecture_benchmark_001/data/monitor_probe.jsonl`
(the 150-row panel), extracting each panel row's arithmetic stem as the text after the final
`': '` of its last user turn:

    split                   rows   verbatim stems   digit-blinded stems
    splits_onpanel/train.jsonl   845      47/150            102/150
    splits_onpanel/dev.jsonl     120      22/150             64/150
    splits_large/train.jsonl   10000       0/150              0/150   <- clean control

Training on this split would train directly on 47 panel questions together with their reference
answers. Selecting model-soup ingredients on its dev slice would select on 22 panel questions.
Either would make a panel score meaningless while looking rigorous.

## Why the builder's checks passed anyway

`build_splits_onpanel.py` enforced disjointness on `metadata.template_family` (as `group_id`) and
on row `id`, and all eight checks returned zero. They were the wrong checks: **different template
families generate identical question stems**, so family-disjointness does not imply
question-disjointness. The builder never compared the rendered text.

`splits_large` is clean because its builder draws TRAIN from the 36 non-probe classes, which have
no textual overlap with the three panel classes at all. That is a property of the class partition,
not of the disjointness test — the same test would have missed this leak there too.

## Status

Nothing has been trained on this split. It was built and verified but never consumed: no
ARM_DOSE artifact references `splits_onpanel`, and every completed arm used `splits_large`. The
73/150 transfer result is unaffected.

`model_soup.py` refuses to run while this file exists.

## To fix

The builder needs a text-level guard that fails closed: reject any candidate row whose rendered
stem, digit-blinded, matches any panel stem. Expect that to remove most of the 845 rows —
102/150 blinded overlap suggests the in-domain corpus is largely template-shared with the panel,
and a genuinely disjoint in-domain split may not be constructible from this corpus at all. If it
is not, say so rather than shipping a smaller contaminated split.
