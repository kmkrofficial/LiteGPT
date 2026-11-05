"""GPU telemetry sampler (NVML): power, temperature, clocks, utilisation, memory, throttle reasons.

Laptop GPUs throttle; this records it so reports can show sustained throughput honestly.
Falls back gracefully (available=False) if NVML is not usable.
"""

from __future__ import annotations

import threading
import time

try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None

# NVML clocks-event-reason bits that mean "the GPU is being held back"
_SW_POWER_CAP, _HW_SLOWDOWN, _SW_THERMAL, _HW_THERMAL, _HW_POWER_BRAKE = 0x4, 0x8, 0x20, 0x40, 0x80
_THERMAL_BITS = _SW_THERMAL | _HW_THERMAL | _HW_SLOWDOWN
_POWER_BITS = _SW_POWER_CAP | _HW_POWER_BRAKE


class TelemetrySampler:
    def __init__(self, interval: float = 1.0, record=None, index: int = 0, stream: str = "telemetry"):
        self.interval, self.record, self.index, self.stream = interval, record, index, stream
        self.samples: list[dict] = []
        self.available = False
        self._stop = threading.Event()
        self._thread = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
                self.available = True
            except Exception:
                self.available = False

    def sample(self) -> dict:
        h = self._h
        reasons_fn = (getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None)
                      or getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None))
        s = {
            "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
            "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "mem_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
            "util_pct": pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
            "mem_used_mib": pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20,
            "throttle_reasons": int(reasons_fn(h)) if reasons_fn else 0,
            # the enforced limit is dynamic on laptops (OEM power modes / Dynamic Boost) and changes throughput a lot
            "power_limit_w": pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
        }
        return s

    def _loop(self):
        while not self._stop.is_set():
            try:
                s = {"t": time.time(), **self.sample()}
                self.samples.append(s)
                if self.record is not None:
                    self.record.log(self.stream, **{k: v for k, v in s.items() if k != "t"})
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        if self.available and self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> dict:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._thread = None
        return self.summary()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def summary(self) -> dict:
        s = self.samples
        if not s:
            return {"available": self.available, "n_samples": 0}
        n = len(s)
        dur = s[-1]["t"] - s[0]["t"]
        energy_wh = sum(0.5 * (a["power_w"] + b["power_w"]) * (b["t"] - a["t"]) for a, b in zip(s, s[1:])) / 3600.0
        busy = [x for x in s if x["util_pct"] >= 50] or s  # ignore idle samples for clock statistics
        return {
            "available": True, "n_samples": n, "duration_s": round(dur, 2), "energy_wh": round(energy_wh, 4),
            "power_w_mean": round(sum(x["power_w"] for x in s) / n, 2), "power_w_max": max(x["power_w"] for x in s),
            "temp_c_max": max(x["temp_c"] for x in s),
            "sm_clock_mhz_mean_busy": round(sum(x["sm_clock_mhz"] for x in busy) / len(busy), 1),
            "sm_clock_mhz_min_busy": min(x["sm_clock_mhz"] for x in busy),
            "frac_power_capped": round(sum(bool(x["throttle_reasons"] & _POWER_BITS) for x in busy) / len(busy), 3),
            "frac_thermal_throttled": round(sum(bool(x["throttle_reasons"] & _THERMAL_BITS) for x in busy) / len(busy), 3),
            "mem_used_mib_max": round(max(x["mem_used_mib"] for x in s), 1),
            "power_limit_w_min": min(x.get("power_limit_w", 0) for x in s),
            "power_limit_w_max": max(x.get("power_limit_w", 0) for x in s),
        }
