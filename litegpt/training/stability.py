"""Training-stability checks over a run's logged metrics (EXP-M2-LADDER protocol correction, 2026-10-13)."""
from __future__ import annotations

import statistics


def grad_norm_divergence(metrics: list[dict], total_steps: int, warmup_steps: int = 0, factor: float = 10.0,
                         min_frac: float = 0.01) -> dict:
    """A run has diverged if its logged gradient norm exceeds `factor` x the median of all earlier post-warmup logged values for
    logged points that together cover more than `min_frac` of the run's steps. Each logged point covers the steps since the
    previous one. Points during warmup are ignored (gradient norms are naturally high there)."""
    pts = [m for m in metrics if m.get("grad_norm") is not None and m["step"] > warmup_steps]
    seen, flagged_steps, first, worst, prev_step = [], 0, None, 0.0, warmup_steps
    for m in pts:
        span = m["step"] - prev_step
        prev_step = m["step"]
        if len(seen) >= 5:
            ref = statistics.median(seen)
            ratio = m["grad_norm"] / ref if ref > 0 else 0.0
            worst = max(worst, ratio)
            if ratio > factor:
                flagged_steps += span
                first = first if first is not None else m["step"]
        seen.append(m["grad_norm"])
    frac = flagged_steps / max(total_steps, 1)
    return {"diverged": frac > min_frac, "flagged_steps": flagged_steps, "flagged_fraction": round(frac, 4),
            "first_flagged_step": first, "max_ratio_to_running_median": round(worst, 2), "factor": factor, "min_frac": min_frac}
