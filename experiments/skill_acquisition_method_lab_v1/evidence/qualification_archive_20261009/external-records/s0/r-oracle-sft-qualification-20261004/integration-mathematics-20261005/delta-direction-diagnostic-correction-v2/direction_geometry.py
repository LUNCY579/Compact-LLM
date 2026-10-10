"""Shared scalar geometry policy; no tensor or checkpoint dependencies."""
from __future__ import annotations
import math

def direction_metrics(dot: float, x_norm: float, y_norm: float) -> dict:
    if not all(math.isfinite(v) for v in (dot,x_norm,y_norm)) or min(x_norm,y_norm)<0:
        raise ValueError('Nonfinite or negative direction summary')
    if x_norm==0 or y_norm==0:
        if dot!=0:raise ValueError('Nonzero dot product with a zero norm')
        return {'cosine':None,'angular_degrees':None,'raw_cosine_ratio':None,'direction_status':'undefined_zero_norm'}
    ratio=(dot/x_norm)/y_norm
    if not math.isfinite(ratio):raise ValueError('Nonfinite direction ratio')
    if abs(ratio)>1:
        return {'cosine':None,'angular_degrees':None,'raw_cosine_ratio':ratio,'direction_status':'outside_unit_interval'}
    return {'cosine':ratio,'angular_degrees':math.degrees(math.acos(ratio)),'raw_cosine_ratio':ratio,'direction_status':'finite'}

def optional_number(value: float | None,digits: int=6) -> str:
    return 'undefined' if value is None else f'{value:.{digits}f}'
