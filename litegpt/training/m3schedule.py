"""The M3 pretraining schedule (spec training.pretraining, ADR-001) and the pre-registered extension gate.

Phases, in optimizer steps: warmup + stable on the stable index until the gate; then the decay on the decay index, whose last
tokens run at the extended context. The learning rate stays at its peak until the decay, so the gate's answer only moves the
decay's start. `decide_gate` is a pure function of the run's records (evaluations, metrics) and writes nothing.
"""
from __future__ import annotations

import math

from litegpt.training.schedule import wsd_lr
from litegpt.training.stability import grad_norm_divergence


class Schedule:
    def __init__(self, s: dict, tokens_per_step: int):
        self.s, self.tps = s, tokens_per_step
        self.warmup_steps = self.steps(s["warmup_tokens"])
        self.decay_steps = self.steps(s["decay_tokens"])
        self.extension_steps = self.steps(s["extension"]["tokens"])
        self.gate_step = self.steps(s["stable_end_tokens"])
        self.stable_end_step = self.gate_step
        self.max_stable_step = None
        self.decided = False
        if self.extension_steps > self.decay_steps:
            raise ValueError("the context extension must fit inside the decay")
        if s.get("budget") and "rented_at_unix" not in s["budget"]:      # fail now, not at the gate about a day into the run
            raise ValueError("the schedule has a budget but no rented_at_unix: pass scripts/train.py --rented-at")

    def steps(self, tokens: float) -> int:
        """First step count whose tokens reach `tokens`, so the evaluation at exactly `tokens` still belongs to the phase."""
        return math.ceil(tokens / self.tps)

    @property
    def total_steps(self) -> int:
        return self.stable_end_step + self.decay_steps

    @property
    def extension_start_step(self) -> int:
        return self.total_steps - self.extension_steps

    def phase(self, step: int) -> str:
        """The phase of the step about to run. Before the gate is decided every step up to it is stable."""
        if step < self.stable_end_step:
            return "stable"
        return "decay" if step < self.extension_start_step else "extension"

    def lr(self, step: int, peak: float, final_frac: float) -> float:
        return wsd_lr(step, self.total_steps, peak, self.warmup_steps, self.decay_steps, final_frac)

    def limit_stable(self, max_steps: int) -> None:
        """The last whole step the stable index can feed: its 14.0B tokens end a fraction of a step short of steps(14.0B)."""
        self.max_stable_step = max_steps
        self.stable_end_step = min(self.stable_end_step, max_steps)

    def apply_gate(self, record: dict) -> None:
        end = self.steps(record["stable_end_tokens"])
        self.stable_end_step = min(end, self.max_stable_step) if self.max_stable_step else end
        self.decided = True


def _at(evals: list[dict], target: float, tps: int) -> float | None:
    """val BPB of the evaluation at `target` tokens (the first step reaching it). A repeated evaluation after a resume: the last."""
    hits = [e for e in evals if target <= e["tokens"] < target + tps]
    return hits[-1]["val"]["bpb"] if hits else None


def _tokens_per_second(metrics: list[dict], end_tokens: float, span_tokens: float) -> float | None:
    """Wall-clock speed over the last `span_tokens`, evaluations and checkpoints included."""
    pts = sorted({m["tokens"]: m for m in metrics if end_tokens - span_tokens <= m["tokens"] <= end_tokens}.values(), key=lambda m: m["tokens"])
    if len(pts) < 2 or pts[-1]["t"] <= pts[0]["t"]:
        return None
    return (pts[-1]["tokens"] - pts[0]["tokens"]) / (pts[-1]["t"] - pts[0]["t"])


def decide_gate(sched: Schedule, evals: list[dict], metrics: list[dict], now: float) -> dict:
    """EXP-G1-EXTENSION-GATE: delta = mean BPB of window A - mean BPB of window B; delta >= threshold extends, subject to the
    budget (largest affordable stable end in fixed steps); a missing evaluation or a divergence means stop."""
    g, tps = sched.s["gate"], sched.tps
    a, b = [[_at(evals, t, tps) for t in g[w]] for w in ("window_a", "window_b")]
    gate_tokens = sched.s["stable_end_tokens"]
    out = {"inputs": {"window_a": dict(zip(map(str, g["window_a"]), a)), "window_b": dict(zip(map(str, g["window_b"]), b))},
           "threshold_bpb": g["threshold_bpb"], "gate_tokens": gate_tokens, "t": now}
    stable = [m for m in metrics if m["tokens"] <= gate_tokens + tps]
    div = grad_norm_divergence(stable, sched.gate_step, warmup_steps=sched.warmup_steps)
    out["divergence"] = div
    stop = {"decision": "stop", "stable_end_tokens": gate_tokens}
    if None in a or None in b:
        return {**out, **stop, "delta": None, "reason": "invalid: an evaluation inside a window is missing"}
    delta = sum(a) / len(a) - sum(b) / len(b)
    out["delta"] = delta
    if div["diverged"]:
        return {**out, **stop, "reason": "invalid: divergence"}
    if delta < g["threshold_bpb"]:
        return {**out, **stop, "reason": "delta below the threshold"}
    top = g["max_stable_end_tokens"]
    bud = sched.s.get("budget")
    if not bud:
        return {**out, "decision": "extend", "stable_end_tokens": top, "reason": "delta at or above the threshold"}
    speed = _tokens_per_second(metrics, gate_tokens + tps, g["speed_window_tokens"])
    if speed is None:
        return {**out, **stop, "reason": "invalid: no speed measurement for the budget projection"}
    rate = bud["price_usd_h"] * bud["inr_per_usd"] / 3600
    spent = (now - bud["rented_at_unix"]) * rate

    def projected(end: float) -> float:
        return spent + (end - gate_tokens + sched.s["decay_tokens"]) / speed * rate

    # candidates count down from the top in fixed steps, so the largest affordable stable end is the first that fits
    n = round((top - gate_tokens) / g["step_tokens"])
    cands = [gate_tokens + i * g["step_tokens"] for i in range(n, 0, -1)]
    table = [{"stable_end_tokens": c, "projected_inr": round(projected(c), 1)} for c in cands]
    out["budget"] = {**bud, "spent_inr": round(spent, 1), "measured_tok_per_s": round(speed, 1), "projections": table}
    for row in table:
        if row["projected_inr"] <= bud["gate_budget_inr"]:
            trimmed = row["stable_end_tokens"] < top
            return {**out, "decision": "extend", "stable_end_tokens": row["stable_end_tokens"],
                    "reason": "delta at or above the threshold" + (", trimmed to the budget" if trimmed else "")}
    return {**out, **stop, "reason": "delta at or above the threshold, but no extension fits the budget"}
