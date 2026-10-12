"""Warmup-stable-decay (WSD) learning-rate schedule (spec `training.pretraining.schedule: wsd`)."""
from __future__ import annotations

import math


def wsd_lr(step: int, total_steps: int, peak_lr: float, warmup_steps: int, decay_steps: int, final_frac: float = 0.1,
           shape: str = "1-sqrt") -> float:
    """Linear warmup to `peak_lr`, constant, then decay over the last `decay_steps` to `final_frac * peak_lr`.
    `shape` is the decay curve: '1-sqrt' (lr = peak * (1 - (1 - f) * sqrt(p))) or 'linear'; p in [0, 1] is decay progress."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    decay_start = total_steps - decay_steps
    if step < decay_start or decay_steps <= 0:
        return peak_lr
    p = min(1.0, (step - decay_start + 1) / decay_steps)
    k = math.sqrt(p) if shape == "1-sqrt" else p
    return peak_lr * (1.0 - (1.0 - final_frac) * k)
