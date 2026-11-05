import math

import pytest

from benchmarks import power_sweep as ps

TH = dict(ps.THRESHOLDS)


def _samples(t0, n, power=150.0, temp=75, clock=1990, reasons=0x4, limit=155.0):
    return [{"t": t0 + i, "power_w": power, "temp_c": temp, "sm_clock_mhz": clock, "util_pct": 99, "mem_used_mib": 12000.0,
             "throttle_reasons": reasons, "power_limit_w": limit} for i in range(n)]


def _workload(windows, start=1000.0, win_s=30.0, losses=None, final=7.0):
    ends = [start + win_s * (i + 1) for i in range(len(windows))]
    return {"status": "ok", "measure_start_t": start, "final_loss": final,
            "sustained": {"window_tok_per_s": windows, "window_end_t": ends,
                          "window_loss": losses if losses is not None else [8.0 - 0.1 * i for i in range(len(windows))]}}


def _entry(spec="observe", control="observe", value=None, **metric_overrides):
    w = _workload([30000.0] * 10)
    m = ps.summarize(w, _samples(1000, 300), settle_windows=2)
    m.update(metric_overrides)
    return {"spec": spec, "label": spec, "control": control, "value": value, "status": "ok", "metrics": m}


# --- point parsing / safety bounds ---------------------------------------------------------------------------------
def test_parse_points_and_bounds():
    assert ps.parse_point("observe").control == "observe"
    assert ps.parse_point("manual:performance").label == "manual: performance"
    p = ps.parse_point("lgc:1800", max_clock_mhz=3105)
    assert (p.control, p.value) == ("lgc", 1800.0)
    for bad in ("lgc:800", "lgc:4000"):
        with pytest.raises(ValueError, match="outside the allowed range"):
            ps.parse_point(bad, max_clock_mhz=3105)
    for bad in ("volt:-100", "lgc:abc", "pl:200", ""):
        with pytest.raises(ValueError):
            ps.parse_point(bad, max_clock_mhz=3105)


def test_tool_never_exposes_voltage_offsets_or_power_raises():
    src = open(ps.__file__, encoding="utf-8").read()
    for forbidden in ("--power-limit", "SetGpcClkVfOffset", "VfOffset", "--lock-memory-clocks", "-pl "):
        assert forbidden not in src.replace("never raises power limits", "")


# --- summarize -----------------------------------------------------------------------------------------------------
def test_summarize_uses_post_settle_windows_and_matching_telemetry():
    windows = [20000.0, 25000.0] + [30000.0] * 8           # first two windows are warm-up / settling
    w = _workload(windows)
    samples = _samples(1000, 61, power=100.0, temp=60) + _samples(1061, 240, power=150.0, temp=76)
    m = ps.summarize(w, samples, settle_windows=2)
    assert m["tok_per_s"] == 30000.0 and m["windows_measured"] == 8 and m["cv"] == 0.0
    assert m["power_w_mean"] == 150.0 and m["temp_c_max"] == 76          # settle-period samples excluded
    assert m["tok_per_s_per_w"] == 200.0
    assert m["frac_power_capped"] == 1.0 and m["frac_thermal_throttled"] == 0.0
    assert m["power_limit_w_max"] == 155.0 and m["losses_finite"] and m["loss_max_rel_rise"] == 0.0


def test_summarize_detects_loss_spike_and_nonfinite():
    m = ps.summarize(_workload([30000.0] * 6, losses=[8.0, 7.0, 6.0, 7.5, 5.0, 4.0]), _samples(1000, 200), 1)
    assert m["loss_max_rel_rise"] == pytest.approx(0.25)
    m = ps.summarize(_workload([30000.0] * 4, losses=[8.0, math.nan, 7.0, 6.0]), _samples(1000, 200), 1)
    assert not m["losses_finite"]


# --- validity / eligibility ----------------------------------------------------------------------------------------
def test_judge_valid_and_eligible():
    e = ps.judge(_entry(), TH)
    assert e["valid"] and e["eligible"] and e["invalid_reasons"] == [] and e["ineligible_reasons"] == []


@pytest.mark.parametrize("override,reason", [
    ({"loss_max_rel_rise": 0.2}, "loss spike"),
    ({"losses_finite": False}, "non-finite"),
    ({"cv": 0.12}, "unstable"),
    ({"temp_c_max": 91}, "abort"),
    ({"hw_slowdown_seen": True}, "hardware slowdown"),
])
def test_judge_invalid(override, reason):
    e = ps.judge(_entry(**override), TH)
    assert not e["valid"] and not e["eligible"] and any(reason in r for r in e["invalid_reasons"])


@pytest.mark.parametrize("override,reason", [
    ({"frac_thermal_throttled": 0.05}, "thermal throttling"),
    ({"temp_c_max": 84}, "target - margin"),
    ({"cv": 0.05}, "variation"),
    ({"last_over_first": 0.9}, "decayed"),
])
def test_judge_valid_but_not_a_candidate(override, reason):
    e = ps.judge(_entry(**override), TH)
    assert e["valid"] and not e["eligible"] and any(reason in r for r in e["ineligible_reasons"])


def test_judge_status_and_clock_cap():
    e = ps.judge({**_entry(), "status": "thermal_abort"}, TH)
    assert not e["valid"] and "status=thermal_abort" in e["invalid_reasons"]
    e = ps.judge(_entry("lgc:1800", "lgc", 1800.0, sm_clock_mhz_max=1995, sm_clock_mhz_mean=1990), TH)
    assert not e["valid"] and any("not applied" in r for r in e["invalid_reasons"])
    e = ps.judge(_entry("lgc:1800", "lgc", 1800.0, sm_clock_mhz_max=1800, sm_clock_mhz_mean=1795), TH)
    assert e["valid"] and e["cap_binding"]


# --- candidate selection and Pareto front ---------------------------------------------------------------------------
def test_select_candidate_highest_throughput_then_lowest_power_on_ties():
    a = ps.judge(_entry("observe", tok_per_s=30000.0, power_w_mean=150.0), TH)
    b = ps.judge(_entry("lgc:1950", "lgc", 1950.0, tok_per_s=29800.0, power_w_mean=130.0, sm_clock_mhz_max=1950,
                        sm_clock_mhz_mean=1940), TH)                       # within 1%: lower power wins
    c = ps.judge(_entry("lgc:1800", "lgc", 1800.0, tok_per_s=27000.0, power_w_mean=110.0, sm_clock_mhz_max=1800,
                        sm_clock_mhz_mean=1790), TH)
    hot = ps.judge(_entry("manual:turbo", "manual", tok_per_s=33000.0, power_w_mean=175.0, temp_c_max=86), TH)
    assert ps.select_candidate([a, b, c, hot], TH)["spec"] == "lgc:1950"   # hottest point is fastest but not eligible
    assert ps.select_candidate([hot], TH) is None


def test_pareto_front():
    a = ps.judge(_entry("a", tok_per_s=30000.0, temp_c_max=80), TH)
    b = ps.judge(_entry("b", tok_per_s=28000.0, temp_c_max=72), TH)
    c = ps.judge(_entry("c", tok_per_s=27000.0, temp_c_max=78), TH)       # dominated by b
    d = ps.judge(_entry("d", tok_per_s=31000.0, temp_c_max=95), TH)       # invalid (abort temperature)
    assert ps.pareto([a, b, c, d]) == ["a", "b"]


# --- control safety --------------------------------------------------------------------------------------------------
def test_clock_cap_skipped_without_elevation():
    class NoAdmin(ps.GpuControl):
        def elevated(self):
            return False

        def lock(self, mhz):  # pragma: no cover - must never be reached
            raise AssertionError("must not try to lock clocks without elevation")

    e = ps.measure_point(ps.parse_point("lgc:1800", 3105), minutes=0.1, window_seconds=5, settle_windows=0, th=TH,
                         rec=None, control=NoAdmin(), workload=ps.WORKLOAD, cooldown_seconds=0)
    assert e["status"] == "skipped" and not e["valid"] and "elevated" in e["notes"][0]
    assert e["invalid_reasons"] == ["status=skipped"]


# --- auto protocol helpers -----------------------------------------------------------------------------------------
def test_derive_caps_below_natural_clock_on_grid_and_bounded():
    assert ps.derive_caps(2060, (100, 200, 300), 3105) == [1950, 1860, 1755]
    assert all(c % ps.CLOCK_GRID_MHZ == 0 for c in ps.derive_caps(2071.4, (100, 250), 3105))
    assert ps.derive_caps(1000, (50, 200), 3105) == [945]                    # 800 would be below the 900 MHz floor
    assert ps.derive_caps(2060, (0, 0, 100), 3105) == [2055, 1950]           # de-duplicated


def test_segment_metrics_excludes_settle_and_other_segments():
    t0, t1, settle, ws = 1000.0, 1240.0, 60.0, 30.0
    windows = [{"t_end": t0 + 30 * k, "seconds": 30.0, "tok_per_s": 20000.0 if k <= 2 else 30000.0, "loss": 8.0 - 0.01 * k}
               for k in range(1, 12)]                                       # last ones run past t1 -> next segment
    samples = _samples(t0, 60, power=100.0, temp=60) + _samples(t0 + 60, 200, power=150.0, temp=77)
    m = ps.segment_metrics(windows, samples, t0, t1, settle, ws)
    assert m["windows_measured"] == 6 and m["tok_per_s"] == 30000.0          # windows ending 1090..1240
    assert m["power_w_mean"] == 150.0 and m["temp_c_max"] == 77 and m["segment_seconds"] == 240.0
