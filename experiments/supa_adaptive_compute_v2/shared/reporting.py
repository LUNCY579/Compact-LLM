"""Quality/cost tradeoffs without choosing an unregistered scalar utility."""
import math


def quality_cost_frontier(arms):
    """An arm dominates only with no lower accuracy and no greater measured cost.

    Point-estimate frontier is descriptive; it does not establish statistical
    dominance. Use paired uncertainty alongside this output.
    """
    cost_keys = ('mean_wall_seconds','flops','added_parameters','peak_memory_bytes')
    if not arms:
        raise ValueError('no evaluated arms')
    for arm, values in arms.items():
        if any(key not in values or not isinstance(values[key],(int,float)) or not math.isfinite(values[key]) for key in ('accuracy',*cost_keys)):
            raise ValueError(f'incomplete/nonfinite frontier inputs for {arm}')
        if not 0 <= values['accuracy'] <= 1 or any(values[key] < 0 for key in cost_keys):
            raise ValueError('invalid frontier value')
    dominated = {}
    for name, point in arms.items():
        by = []
        for other, candidate in arms.items():
            if other == name:
                continue
            no_worse = candidate['accuracy'] >= point['accuracy'] and all(candidate[k] <= point[k] for k in cost_keys)
            better = candidate['accuracy'] > point['accuracy'] or any(candidate[k] < point[k] for k in cost_keys)
            if no_worse and better:
                by.append(other)
        if by:
            dominated[name] = sorted(by)
    return {'frontier':sorted(set(arms)-set(dominated)), 'dominated_by':dominated,
            'arms':arms, 'interpretation':'descriptive point estimates; paired uncertainty required; no single utility score'}
