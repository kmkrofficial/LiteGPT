"""
GPU operating-point sweep for the LiteGPT V2 training workload (EXP-M0-POWER).

  python -m benchmarks.power_sweep screen  [--points observe,lgc:2100,lgc:1950,lgc:1800] [--minutes 5]
  python -m benchmarks.power_sweep compare --from runs/<screen-run-id> [--top 3] [--minutes 10]
  python -m benchmarks.power_sweep soak    (--point lgc:1950 | --from runs/<compare-run-id>) [--minutes 90]
  python -m benchmarks.power_sweep auto    [--preheat-minutes 4] [--segment-minutes 4] [--cap-steps 100,200,300] [--soak-minutes 60]

`auto` (recommended, ~80 min): ONE continuous workload process (compiled once, never cooled down). The controller pre-heats,
measures the current setting at thermal steady state, derives clock caps below the measured natural clock, measures each cap,
selects the candidate with the same pre-registered rule, and soaks it - all while the workload keeps running. Per-window
results are streamed to runs/<run-id>/windows.jsonl, so nothing is lost if a segment is aborted.

Every point runs the same fixed workload (28L, micro-batch 4x2048, bf16 autocast, torch.compile, SDPA by default) in its own
process while the controller samples NVML telemetry once per second into runs/<run-id>/telemetry/<point>.jsonl. Each stage
writes one run record with bench/power_sweep.json (updated after every point, so partial sweeps are preserved).

Operating points (applied in the order given; start from the current stable setting and move down):
  observe          change nothing; measure the current state (e.g. the Lenovo power mode you selected)
  manual:<label>   pause until the operator applies a setting by hand (e.g. a Lenovo power mode) and presses Enter
  lgc:<MHz>        cap the GPU graphics clock to [210, MHz] with `nvidia-smi --lock-gpu-clocks`, reset with `--reset-gpu-clocks`
                   afterwards. Needs an elevated Windows session (from WSL: start WSL from an Administrator terminal).
Safety: this tool never applies voltage offsets, never changes memory clocks and never raises power limits. A clock cap only
moves the GPU down its factory voltage/frequency curve (lower clock -> the lower factory voltage for that clock); caps are
bounded to [900 MHz, device max]. CPU settings are never touched. Points are aborted on CUDA errors/crashes, non-finite or
spiking loss, temperature >= abort threshold, or any hardware slowdown; clocks are reset even on abort.

Voltage, GPU hotspot and VRAM temperatures are not exposed by NVML/nvidia-smi on this GPU and are recorded as unavailable.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.telemetry import _HW_SLOWDOWN, _HW_THERMAL, _POWER_BITS, _THERMAL_BITS, TelemetrySampler

WORKLOAD = {"config": "litegpt-v2-28l", "micro_batch": 4, "seq_len": 2048, "backend": "sdpa", "compile": True,
            "dtype": "bfloat16", "data": "fresh random tokens every step (unlearnable: loss stays near ln(vocab) ~ 10.4)"}
DEFAULT_POINTS = "observe,lgc:2100,lgc:1950,lgc:1800,lgc:1650,lgc:1500"
LOCK_MIN_MHZ = 210        # floor of the locked range (lowest supported graphics clock)
LOCK_CAP_MIN_MHZ = 900    # never cap below this
CLOCK_GRID_MHZ = 15       # supported graphics clocks are spaced 15 MHz apart
DEFAULT_CAP_STEPS = (100, 200, 300)   # auto: caps at natural clock minus these (caps above the natural clock never bind)
THRESHOLDS = {            # pre-registered (EXP-M0-POWER); recorded in every sweep record
    "temp_target_c": 85.0,         # GPU target temperature reported by nvidia-smi
    "thermal_margin_c": 3.0,       # candidate must stay <= target - margin
    "abort_temp_c": 90.0,          # point aborted (invalid) at or above this
    "max_cv": 0.03,                # candidate: coefficient of variation of post-settle tok/s windows
    "min_last_over_first": 0.97,   # candidate: last / first post-settle window
    "invalid_cv": 0.08,            # invalid (unstable) above this
    "loss_spike_rel": 0.10,        # invalid if a window loss exceeds the running minimum by more than this fraction
    "lock_tolerance_mhz": 30.0,    # lgc point invalid if the observed max SM clock exceeds the cap by more than this
    "tie_rel": 0.01,               # among candidates within 1% of the best tok/s, prefer the lowest mean power
}


# ------------------------------------------------------------------------------------------------- operating points
@dataclass(frozen=True)
class Point:
    spec: str
    control: str          # observe | manual | lgc
    value: float | None
    label: str


def parse_point(spec: str, max_clock_mhz: int | None = None) -> Point:
    spec = spec.strip()
    if spec == "observe":
        return Point(spec, "observe", None, "observe (current setting)")
    m = re.fullmatch(r"manual:([A-Za-z0-9_.+-]+)", spec)
    if m:
        return Point(spec, "manual", None, f"manual: {m.group(1)}")
    m = re.fullmatch(r"lgc:(\d+)", spec)
    if m:
        mhz = int(m.group(1))
        hi = max_clock_mhz or 10 ** 6
        if not (LOCK_CAP_MIN_MHZ <= mhz <= hi):
            raise ValueError(f"clock cap {mhz} MHz outside the allowed range [{LOCK_CAP_MIN_MHZ}, {hi}]")
        return Point(spec, "lgc", float(mhz), f"clock cap {mhz} MHz")
    raise ValueError(f"unknown point spec {spec!r} (use observe, manual:<label> or lgc:<MHz>)")


def slug(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", spec)


class GpuControl:
    """Applies/resets clock caps through the Windows nvidia-smi (also from WSL via interop)."""

    def __init__(self):
        wsl_exe = Path("/mnt/c/Windows/System32/nvidia-smi.exe")
        self.exe = str(wsl_exe) if wsl_exe.exists() else (shutil.which("nvidia-smi") or "nvidia-smi")
        self.locked = False

    def elevated(self) -> bool | None:
        if sys.platform == "win32":
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        out = powershell("([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent())"
                         ".IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)")
        if out in ("True", "False"):
            return out == "True"
        return os.geteuid() == 0 if hasattr(os, "geteuid") else None

    def _smi(self, *args) -> tuple[int, str]:
        p = subprocess.run([self.exe, "-i", "0", *args], capture_output=True, text=True, timeout=60)
        return p.returncode, (p.stdout + p.stderr).strip()

    def lock(self, mhz: int) -> tuple[bool, str]:
        rc, out = self._smi("--lock-gpu-clocks", f"{LOCK_MIN_MHZ},{mhz}")
        self.locked = rc == 0 or self.locked
        return rc == 0, out

    def reset(self) -> tuple[bool, str]:
        rc, out = self._smi("--reset-gpu-clocks")
        if rc == 0:
            self.locked = False
        return rc == 0, out


def powershell(command: str) -> str | None:
    exe = shutil.which("powershell.exe") or ("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
                                             if Path("/mnt/c/Windows").exists() else None)
    if not exe:
        return None
    try:
        p = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, text=True,
                           timeout=30)
        return p.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def host_info(control: GpuControl) -> dict:
    info = {"platform": platform.platform(), "nvidia_smi": control.exe, "windows_elevated": control.elevated(),
            "ac_power_online": powershell("(Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus | "
                                          "Select-Object -First 1).PowerOnline"),
            "lenovo_smart_fan_mode": powershell("try { $g = Get-CimInstance -Namespace root\\WMI -ClassName "
                                                "LENOVO_GAMEZONE_DATA -ErrorAction Stop | Select-Object -First 1; "
                                                "(Invoke-CimMethod -InputObject $g -MethodName GetSmartFanMode).Data } "
                                                "catch { 'unavailable: ' + $_.Exception.Message }"),
            "cpu_undervolt": "disabled / not touched by this tool",
            "voltage_readout": "unavailable (not exposed by NVML / nvidia-smi on this GPU)",
            "hotspot_temp": "unavailable", "vram_temp": "unavailable"}
    try:
        import pynvml as n
        n.nvmlInit()
        h = n.nvmlDeviceGetHandleByIndex(0)
        info.update(gpu_power_limit_constraints_w=[x / 1000 for x in n.nvmlDeviceGetPowerManagementLimitConstraints(h)],
                    gpu_power_limit_default_w=n.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000,
                    gpu_slowdown_threshold_c=n.nvmlDeviceGetTemperatureThreshold(h, n.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN),
                    gpu_max_sm_clock_mhz=n.nvmlDeviceGetMaxClockInfo(h, n.NVML_CLOCK_SM))
    except Exception as e:  # pragma: no cover - depends on the machine
        info["nvml_error"] = repr(e)
    return info


# ------------------------------------------------------------------------------------------------- analysis (pure)
def summarize(workload: dict, samples: list[dict], settle_windows: int) -> dict:
    """Metrics for one point from the workload result and the raw telemetry samples (restricted to the measured span)."""
    sus = workload.get("sustained") or {}
    win, ends, losses = sus.get("window_tok_per_s", []), sus.get("window_end_t", []), sus.get("window_loss", [])
    k = settle_windows if len(win) > settle_windows else 0
    measured = win[k:]
    t0 = ends[k - 1] if k > 0 else workload.get("measure_start_t", -math.inf)
    t1 = ends[-1] if ends else math.inf
    span = [s for s in samples if t0 < s["t"] <= t1] or samples   # measured span starts after the settle windows
    m: dict = {"windows_total": len(win), "windows_measured": len(measured), "settle_windows": k,
               "measured_seconds": round(t1 - t0, 1) if ends and math.isfinite(t0) else None}
    if measured:
        mean = statistics.fmean(measured)
        m.update(tok_per_s=round(mean, 1), tok_per_s_min=min(measured), tok_per_s_max=max(measured),
                 cv=round(statistics.pstdev(measured) / mean, 4) if mean else None,
                 last_over_first=round(measured[-1] / measured[0], 4))
    if span:
        n = len(span)
        pw = [s["power_w"] for s in span]
        m.update(power_w_mean=round(statistics.fmean(pw), 2), power_w_peak=round(max(pw), 2),
                 temp_c_mean=round(statistics.fmean(s["temp_c"] for s in span), 2), temp_c_max=max(s["temp_c"] for s in span),
                 sm_clock_mhz_mean=round(statistics.fmean(s["sm_clock_mhz"] for s in span), 1),
                 sm_clock_mhz_min=min(s["sm_clock_mhz"] for s in span), sm_clock_mhz_max=max(s["sm_clock_mhz"] for s in span),
                 util_pct_mean=round(statistics.fmean(s["util_pct"] for s in span), 1),
                 vram_used_mib_max=round(max(s["mem_used_mib"] for s in span), 1),
                 frac_thermal_throttled=round(sum(bool(s["throttle_reasons"] & _THERMAL_BITS) for s in span) / n, 4),
                 frac_power_capped=round(sum(bool(s["throttle_reasons"] & _POWER_BITS) for s in span) / n, 4),
                 hw_slowdown_seen=any(s["throttle_reasons"] & (_HW_SLOWDOWN | _HW_THERMAL) for s in span),
                 power_limit_w_min=min(s.get("power_limit_w", 0) for s in span),
                 power_limit_w_max=max(s.get("power_limit_w", 0) for s in span), telemetry_samples=n)
        if m.get("tok_per_s") and m["power_w_mean"]:
            m["tok_per_s_per_w"] = round(m["tok_per_s"] / m["power_w_mean"], 2)
    finite = [x for x in losses if isinstance(x, (int, float)) and math.isfinite(x)]
    m["losses_finite"] = bool(losses) and len(finite) == len(losses) and math.isfinite(workload.get("final_loss", math.nan))
    rise, run_min = 0.0, math.inf
    for x in finite:
        run_min = min(run_min, x)
        rise = max(rise, (x - run_min) / run_min if run_min > 0 else 0.0)
    m["loss_max_rel_rise"] = round(rise, 4)
    return m


def judge(entry: dict, th: dict) -> dict:
    """Fill valid / invalid_reasons and eligible / ineligible_reasons (the pre-registered candidate rule)."""
    m, inv = entry.get("metrics", {}), []
    if entry["status"] in ("skipped", "apply_failed"):        # nothing was measured: the status is the only reason
        entry.update(valid=False, invalid_reasons=[f"status={entry['status']}"], eligible=False,
                     ineligible_reasons=[f"status={entry['status']}"])
        return entry
    if entry["status"] != "ok":
        inv.append(f"status={entry['status']}")
    if not m.get("losses_finite"):
        inv.append("non-finite or missing loss")
    if m.get("loss_max_rel_rise", 0) > th["loss_spike_rel"]:
        inv.append(f"loss spike {m['loss_max_rel_rise']:.1%}")
    if m.get("cv") is not None and m["cv"] > th["invalid_cv"]:
        inv.append(f"unstable throughput (cv {m['cv']:.3f})")
    if not m.get("tok_per_s") or not m.get("telemetry_samples"):
        inv.append("no measurement")
    if m.get("temp_c_max", 0) >= th["abort_temp_c"]:
        inv.append(f"temperature {m['temp_c_max']} C >= abort {th['abort_temp_c']} C")
    if m.get("hw_slowdown_seen"):
        inv.append("hardware slowdown")
    if entry["control"] == "lgc" and m.get("sm_clock_mhz_max", 0) > entry["value"] + th["lock_tolerance_mhz"]:
        inv.append(f"clock cap not applied (max {m['sm_clock_mhz_max']} MHz)")
    entry["valid"], entry["invalid_reasons"] = not inv, inv
    inel = list(inv)
    if not inv:
        if m["frac_thermal_throttled"] > 0:
            inel.append(f"thermal throttling {m['frac_thermal_throttled']:.1%}")
        limit = th["temp_target_c"] - th["thermal_margin_c"]
        if m["temp_c_max"] > limit:
            inel.append(f"max temperature {m['temp_c_max']} C > {limit} C (target - margin)")
        if m.get("cv") is not None and m["cv"] > th["max_cv"]:
            inel.append(f"throughput variation cv {m['cv']:.3f} > {th['max_cv']}")
        if m.get("last_over_first", 1) < th["min_last_over_first"]:
            inel.append(f"throughput decayed (last/first {m['last_over_first']})")
    entry["eligible"], entry["ineligible_reasons"] = not inel, inel
    if entry["control"] == "lgc" and m.get("sm_clock_mhz_mean") is not None:
        entry["cap_binding"] = m["sm_clock_mhz_mean"] >= entry["value"] - 50
    return entry


def select_candidate(entries: list[dict], th: dict) -> dict | None:
    """Highest sustained tok/s among eligible points; within tie_rel of the best, prefer the lowest mean power."""
    el = [e for e in entries if e.get("eligible")]
    if not el:
        return None
    best = max(e["metrics"]["tok_per_s"] for e in el)
    near = [e for e in el if e["metrics"]["tok_per_s"] >= best * (1 - th["tie_rel"])]
    return min(near, key=lambda e: (e["metrics"]["power_w_mean"], -e["metrics"]["tok_per_s"]))


def pareto(entries: list[dict]) -> list[str]:
    """Valid points not dominated in (higher tok/s, lower max temperature)."""
    v = [e for e in entries if e.get("valid")]
    front = []
    for e in v:
        a, ta = e["metrics"]["tok_per_s"], e["metrics"]["temp_c_max"]
        dominated = any(o is not e and o["metrics"]["tok_per_s"] >= a and o["metrics"]["temp_c_max"] <= ta
                        and (o["metrics"]["tok_per_s"] > a or o["metrics"]["temp_c_max"] < ta) for o in v)
        if not dominated:
            front.append(e["spec"])
    return front


def derive_caps(natural_mhz: float, steps, max_clock_mhz: int | None = None) -> list[int]:
    """Clock caps below the measured natural (power-limited) clock, snapped down to the 15 MHz grid and bounded."""
    caps = []
    for d in steps:
        c = int((natural_mhz - d) // CLOCK_GRID_MHZ * CLOCK_GRID_MHZ)
        if LOCK_CAP_MIN_MHZ <= c <= (max_clock_mhz or 10 ** 6) and c not in caps:
            caps.append(c)
    return caps


def segment_metrics(windows: list[dict], samples: list[dict], t0: float, t1: float, settle_s: float,
                    window_s: float) -> dict:
    """Metrics for one segment of a continuous run: windows that start after the settle period and end inside it."""
    ws = [w for w in windows if w["t_end"] - w.get("seconds", window_s) >= t0 + settle_s and w["t_end"] <= t1]
    pseudo = {"measure_start_t": t0 + settle_s, "final_loss": ws[-1]["loss"] if ws else math.nan,
              "sustained": {"window_tok_per_s": [w["tok_per_s"] for w in ws], "window_end_t": [w["t_end"] for w in ws],
                            "window_loss": [w["loss"] for w in ws]}}
    m = summarize(pseudo, [x for x in samples if t0 + settle_s < x["t"] <= t1], settle_windows=0)
    m["segment_seconds"] = round(t1 - t0, 1)
    return m


# ------------------------------------------------------------------------------------------------- measurement
def measure_point(point: Point, *, minutes: float, window_seconds: float, settle_windows: int, th: dict, rec,
                  control: GpuControl, workload: dict, cooldown_seconds: float) -> dict:
    entry = {"spec": point.spec, "label": point.label, "control": point.control, "value": point.value,
             "status": "ok", "applied": {}, "notes": []}
    if point.control == "manual":
        if not sys.stdin.isatty():
            entry["status"] = "skipped"
            entry["notes"].append("manual point needs an interactive terminal")
            return judge(entry, th)
        input(f"\n>>> Apply '{point.label}' now (e.g. switch the Lenovo power mode), then press Enter to measure... ")
        entry["applied"]["operator_confirmed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if point.control == "lgc":
        if not control.elevated():
            entry["status"] = "skipped"
            entry["notes"].append("clock caps need an elevated Windows session (start WSL from an Administrator terminal)")
            return judge(entry, th)
        ok, out = control.lock(int(point.value))
        entry["applied"] = {"lock_gpu_clocks_mhz": [LOCK_MIN_MHZ, int(point.value)], "ok": ok, "nvidia_smi": out[-300:]}
        if not ok:
            entry["status"] = "apply_failed"
            return judge(entry, th)
    tel_dir = rec.path / "telemetry"
    tel_dir.mkdir(exist_ok=True)
    tel = TelemetrySampler(interval=1.0, record=rec, stream=f"telemetry/{slug(point.spec)}")
    if cooldown_seconds:
        time.sleep(cooldown_seconds)
    entry["start_temp_c"] = tel.sample()["temp_c"] if tel.available else None
    with tempfile.TemporaryDirectory() as td:
        out_json, err_log = Path(td) / "w.json", Path(td) / "stderr.log"
        cmd = [sys.executable, "-m", "benchmarks.train_mem", "single", "--config", workload["config"],
               "--micro-batch", str(workload["micro_batch"]), "--seq-len", str(workload["seq_len"]),
               "--backend", workload["backend"], "--compile", str(int(workload["compile"])), "--steps", "1",
               "--warmup", "10", "--sustained-seconds", str(minutes * 60), "--window-seconds", str(window_seconds),
               "--fresh-batches", "--json-out", str(out_json)]
        tel.start()
        try:
            with open(err_log, "w") as ef:
                p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=ef)
                while p.poll() is None:
                    time.sleep(1.0)
                    last = tel.samples[-1] if tel.samples else None
                    if last and (last["temp_c"] >= th["abort_temp_c"] or last["throttle_reasons"] & (_HW_SLOWDOWN | _HW_THERMAL)):
                        entry["status"] = "thermal_abort"
                        entry["notes"].append(f"aborted at {last['temp_c']} C, reasons=0x{last['throttle_reasons']:x}")
                        p.terminate()
                        try:
                            p.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            p.kill()
                        break
        finally:
            tel.stop()
            if point.control == "lgc" and control.locked:
                ok, out = control.reset()
                entry["applied"]["reset_ok"], entry["applied"]["reset_output"] = ok, out[-300:]
        w = json.loads(out_json.read_text()) if out_json.exists() else {}
        if entry["status"] == "ok":
            if not w:
                entry["status"] = "crash"
                entry["notes"].append(err_log.read_text(errors="replace")[-600:])
            elif w.get("status") != "ok":
                entry["status"] = w.get("status", "error")
                entry["notes"].append(str(w.get("error", ""))[:600])
    entry["workload"] = {k: w.get(k) for k in ("status", "warmup_s", "final_loss", "peak_alloc_gib", "vram_pressure",
                                               "steps_measured", "tok_per_s")}
    entry["workload"]["sustained"] = w.get("sustained")
    entry["metrics"] = summarize(w, tel.samples, settle_windows)
    entry["telemetry_file"] = f"telemetry/{slug(point.spec)}.jsonl"
    return judge(entry, th)


def run_stage(stage: str, points: list[Point], *, minutes: float, window_seconds: float, settle_windows: int,
              cooldown_seconds: float, workload: dict, source: str | None, notes: str) -> Path:
    from litegpt.records import create_run
    control = GpuControl()
    atexit.register(lambda: control.locked and control.reset())   # never leave a clock cap behind
    th = dict(THRESHOLDS)
    rec = create_run("bench", f"power-sweep-{stage}", attention_backend=workload["backend"], compile_enabled=workload["compile"],
                     config={"stage": stage, "points": [p.spec for p in points], "minutes_per_point": minutes,
                             "window_seconds": window_seconds, "settle_windows": settle_windows,
                             "cooldown_seconds": cooldown_seconds, "workload": workload, "source": source, "notes": notes},
                     extra={"host": host_info(control)})
    out = {"stage": stage, "workload": workload, "thresholds": th, "source": source, "notes": notes, "points": []}
    for pt in points:
        print(f"[{stage}] {pt.label}: measuring for {minutes} min ...", flush=True)
        e = measure_point(pt, minutes=minutes, window_seconds=window_seconds, settle_windows=settle_windows, th=th,
                          rec=rec, control=control, workload=workload, cooldown_seconds=cooldown_seconds)
        m = e.get("metrics", {})
        print(f"  -> {e['status']} valid={e['valid']} eligible={e['eligible']} tok/s={m.get('tok_per_s')} "
              f"power={m.get('power_w_mean')}W temp_max={m.get('temp_c_max')}C reasons={e['ineligible_reasons']}", flush=True)
        out["points"].append(e)
        cand = select_candidate(out["points"], th)
        out["recommended"] = cand["spec"] if cand else None
        out["pareto_front"] = pareto(out["points"])
        if stage == "soak":
            out["soak_passed"] = bool(out["points"] and out["points"][0].get("eligible"))
        rec.write_json("bench/power_sweep.json", out)
    print("recommended:", out.get("recommended"), "| pareto:", out.get("pareto_front"), "| run record:", rec.path)
    return rec.path


def run_auto(*, preheat_min: float, segment_min: float, settle_s: float, cap_steps, soak_min: float, window_s: float,
             workload: dict, notes: str) -> Path:
    from litegpt.records import create_run
    control = GpuControl()
    atexit.register(lambda: control.locked and control.reset())   # never leave a clock cap behind
    th = dict(THRESHOLDS)
    host = host_info(control)
    elevated = bool(host.get("windows_elevated"))
    steps = list(cap_steps) if elevated else []
    total_s = 60 * (preheat_min + segment_min * (1 + len(steps)) + soak_min) + 180
    protocol = {"preheat_minutes": preheat_min, "segment_minutes": segment_min, "settle_seconds": settle_s,
                "cap_steps_mhz": list(cap_steps), "soak_minutes": soak_min, "window_seconds": window_s,
                "caps_enabled": elevated}
    rec = create_run("bench", "power-sweep-auto", attention_backend=workload["backend"], compile_enabled=workload["compile"],
                     config={"stage": "auto", "workload": workload, "protocol": protocol, "notes": notes},
                     extra={"host": host})
    out = {"stage": "auto", "workload": workload, "thresholds": th, "protocol": protocol, "notes": notes,
           "points": [], "segments": [], "recommended": None, "pareto_front": [], "soak_passed": None}
    if not elevated:
        out["caps_skipped_reason"] = "not elevated: start WSL from an Administrator terminal to enable clock caps"
    (rec.path / "telemetry").mkdir(exist_ok=True)
    (rec.path / "raw").mkdir(exist_ok=True)
    wj = rec.path / "windows.jsonl"
    tel = TelemetrySampler(interval=1.0, record=rec, stream="telemetry/auto")
    cmd = [sys.executable, "-m", "benchmarks.train_mem", "single", "--config", workload["config"],
           "--micro-batch", str(workload["micro_batch"]), "--seq-len", str(workload["seq_len"]),
           "--backend", workload["backend"], "--compile", str(int(workload["compile"])), "--steps", "1", "--warmup", "10",
           "--sustained-seconds", str(total_s), "--window-seconds", str(window_s), "--windows-jsonl", str(wj),
           "--fresh-batches",
           "--json-out", str(rec.path / "raw" / "workload_result.json")]
    err = open(rec.path / "raw" / "workload_stderr.log", "w")

    def read_windows():
        if not wj.exists():
            return []
        rows = [json.loads(x) for x in wj.read_text(encoding="utf-8").splitlines() if x.strip()]
        return [r for r in rows if r.get("event") == "window"]

    def wait_until(t_end, proc):
        while time.time() < t_end:
            if proc.poll() is not None:
                return "workload_exited"
            last = tel.samples[-1] if tel.samples else None
            if last and (last["temp_c"] >= th["abort_temp_c"] or last["throttle_reasons"] & (_HW_SLOWDOWN | _HW_THERMAL)):
                return "thermal_abort"
            time.sleep(1.0)
        return None

    def segment(proc, label, spec, kind, value, minutes, role):
        applied = {}
        if kind == "lgc":
            ok, o = control.lock(int(value))
            applied = {"lock_gpu_clocks_mhz": [LOCK_MIN_MHZ, int(value)], "ok": ok, "nvidia_smi": o[-300:]}
        elif control.locked:
            ok, o = control.reset()
            applied = {"reset_ok": ok, "nvidia_smi": o[-300:]}
        print(f"[auto] {label}: {minutes} min ...", flush=True)
        t0 = time.time()
        res = "apply_failed" if applied.get("ok") is False else wait_until(t0 + minutes * 60, proc)
        t1 = time.time()
        e = {"spec": spec, "label": label, "control": kind, "value": value, "role": role, "status": res or "ok",
             "applied": applied, "notes": [], "segment": {"t0": t0, "t1": t1, "settle_s": 0 if role == "preheat" else settle_s},
             "metrics": segment_metrics(read_windows(), list(tel.samples), t0, t1,
                                        0 if role == "preheat" else settle_s, window_s)}
        judge(e, th)
        m = e["metrics"]
        print(f"  -> {e['status']} valid={e['valid']} eligible={e['eligible']} tok/s={m.get('tok_per_s')} "
              f"power={m.get('power_w_mean')}W clock={m.get('sm_clock_mhz_mean')}MHz temp_max={m.get('temp_c_max')}C "
              f"reasons={e['ineligible_reasons']}", flush=True)
        out["segments"].append(e)
        if role == "point":
            out["points"].append(e)
        rec.write_json("bench/power_sweep.json", out)
        return e

    tel.start()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err)
    try:
        deadline = time.time() + 900                      # compile + warm-up must finish within 15 minutes
        while time.time() < deadline and proc.poll() is None and not (
                wj.exists() and '"start"' in wj.read_text(encoding="utf-8")):
            time.sleep(1.0)
        if proc.poll() is not None or not wj.exists():
            out["error"] = "workload did not start: " + (rec.path / "raw" / "workload_stderr.log").read_text(errors="replace")[-600:]
            return rec.path
        pre = segment(proc, "pre-heat (discarded)", "preheat", "observe", None, preheat_min, "preheat")
        if pre["status"] == "ok":
            obs = segment(proc, "observe (current setting)", "observe", "observe", None, segment_min, "point")
            natural = obs["metrics"].get("sm_clock_mhz_mean")
            if obs["status"] == "ok" and steps and natural:
                out["natural_clock_mhz"] = natural
                for c in derive_caps(natural, steps, host.get("gpu_max_sm_clock_mhz")):
                    if segment(proc, f"clock cap {c} MHz", f"lgc:{c}", "lgc", float(c), segment_min, "point")["status"] != "ok":
                        break
            cand = select_candidate(out["points"], th)
            out["recommended"] = cand["spec"] if cand else None
            out["pareto_front"] = pareto(out["points"])
            if cand and soak_min > 0 and proc.poll() is None:
                soak = segment(proc, f"soak at {cand['spec']}", f"soak@{cand['spec']}", cand["control"], cand["value"],
                               soak_min, "soak")
                out["soak_passed"] = bool(soak.get("eligible"))
    finally:
        if control.locked:
            ok, o = control.reset()
            out["final_reset"] = {"ok": ok, "nvidia_smi": o[-300:]}
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
        tel.stop()
        err.close()
        rec.write_json("bench/power_sweep.json", out)
    print("recommended:", out["recommended"], "| soak passed:", out["soak_passed"], "| run record:", rec.path)
    return rec.path


def _load_stage(run_dir: str) -> dict:
    return json.loads((Path(run_dir) / "bench" / "power_sweep.json").read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name, minutes in (("screen", 5.0), ("compare", 10.0), ("soak", 90.0)):
        s = sub.add_parser(name)
        s.add_argument("--minutes", type=float, default=minutes)
        s.add_argument("--window-seconds", type=float, default=30.0 if name == "screen" else 60.0)
        s.add_argument("--settle-windows", type=int, default=2 if name == "screen" else 1)
        s.add_argument("--cooldown-seconds", type=float, default=60.0)
        s.add_argument("--backend", default=WORKLOAD["backend"], choices=["sdpa", "flash"])
        s.add_argument("--notes", default="", help="e.g. cooling setup, ambient temperature, Lenovo mode")
        if name == "screen":
            s.add_argument("--points", default=DEFAULT_POINTS)
        elif name == "compare":
            s.add_argument("--from", dest="source", required=True, help="screen run directory")
            s.add_argument("--top", type=int, default=3)
        else:
            g = s.add_mutually_exclusive_group(required=True)
            g.add_argument("--point")
            g.add_argument("--from", dest="source", help="compare run directory (uses its recommended point)")
    x = sub.add_parser("auto")
    x.add_argument("--preheat-minutes", type=float, default=4.0)
    x.add_argument("--segment-minutes", type=float, default=4.0)
    x.add_argument("--settle-seconds", type=float, default=60.0)
    x.add_argument("--cap-steps", default=",".join(str(c) for c in DEFAULT_CAP_STEPS),
                   help="MHz below the measured natural clock; empty string = no caps")
    x.add_argument("--soak-minutes", type=float, default=60.0)
    x.add_argument("--window-seconds", type=float, default=30.0)
    x.add_argument("--backend", default=WORKLOAD["backend"], choices=["sdpa", "flash"])
    x.add_argument("--notes", default="")
    a = ap.parse_args(argv)
    if a.stage == "auto":
        run_auto(preheat_min=a.preheat_minutes, segment_min=a.segment_minutes, settle_s=a.settle_seconds,
                 cap_steps=[int(c) for c in a.cap_steps.split(",") if c.strip()], soak_min=a.soak_minutes,
                 window_s=a.window_seconds, workload=dict(WORKLOAD, backend=a.backend), notes=a.notes)
        return 0
    host_max = host_info(GpuControl()).get("gpu_max_sm_clock_mhz")
    workload = dict(WORKLOAD, backend=a.backend)
    source = getattr(a, "source", None)
    if a.stage == "screen":
        specs = a.points.split(",")
    elif a.stage == "compare":
        prev = _load_stage(source)
        ranked = sorted([e for e in prev["points"] if e.get("valid")], key=lambda e: (not e.get("eligible"),
                        -e["metrics"]["tok_per_s"]))
        specs = [e["spec"] for e in ranked[:a.top]]
        if "observe" not in specs and any(e["spec"] == "observe" for e in prev["points"]):
            specs.append("observe")       # always keep the current setting as the reference
    else:
        specs = [a.point] if a.point else [_load_stage(source)["recommended"]]
        if specs == [None]:
            print("the compare run has no recommended point; nothing to soak")
            return 1
    points = [parse_point(s, host_max) for s in specs]
    run_stage(a.stage, points, minutes=a.minutes, window_seconds=a.window_seconds, settle_windows=a.settle_windows,
              cooldown_seconds=a.cooldown_seconds, workload=workload, source=source, notes=a.notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
