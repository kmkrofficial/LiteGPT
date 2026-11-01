"""M3 server guard and laptop monitor: every decision path, without a server."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "cloud" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


G = load("m3_guard")
M = load("m3_monitor")


def state(**kw):
    return G.State(trainer_alive=True, metrics_age_s=60, **kw)


def test_healthy_run_does_nothing():
    assert G.decide(state()) == ("none", "")


def test_stops_only_after_the_final_checkpoint_is_verified():
    assert G.decide(state(completed=True))[0] == "backup_final"
    assert G.decide(state(completed=True, final_attempts=2))[0] == "backup_final"
    assert G.decide(state(completed=True, final_verified=True))[0] == "stop"


def test_unverifiable_final_backup_still_stops_the_bill():
    action, why = G.decide(state(completed=True, final_attempts=G.FINAL_BACKUP_ATTEMPTS))
    assert action == "stop" and "could not be verified" in why


def test_dead_trainer_resumes_until_the_restart_limit():
    assert G.decide(G.State(trainer_alive=False, recent_restarts=2))[0] == "resume"
    action, why = G.decide(G.State(trainer_alive=False, recent_restarts=G.RESTART_LIMIT))
    assert action == "backup_final" and "restarts" in why
    assert G.decide(G.State(trainer_alive=False, halting=why, final_verified=True)) == ("stop", why)


def test_stall_interrupts_waits_then_kills():
    stalled = dict(metrics_age_s=G.STALL_SECONDS + 1)
    assert G.decide(G.State(trainer_alive=True, **stalled))[0] == "interrupt"
    assert G.decide(G.State(trainer_alive=True, interrupted_age_s=10, **stalled))[0] == "wait"
    assert G.decide(G.State(trainer_alive=True, interrupted_age_s=G.KILL_AFTER_SECONDS, **stalled))[0] == "kill"


def test_spending_cap_saves_backs_up_and_stops():
    near = 2500 - G.CAP_MARGIN_INR
    assert G.decide(state(spent_inr=near - 1))[0] == "none"
    assert G.decide(state(spent_inr=near))[0] == "interrupt"
    assert G.decide(G.State(trainer_alive=False, spent_inr=near))[0] == "backup_final"
    assert G.decide(G.State(trainer_alive=False, spent_inr=near, final_verified=True))[0] == "stop"


def test_periodic_backup():
    assert G.decide(state(backup_age_s=3 * 3600))[0] == "backup"


def test_run_completed_needs_the_summary_and_the_final_checkpoint(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    assert not G.run_completed(tmp_path)
    (tmp_path / "train_summary.json").write_text(json.dumps({"status": "completed", "step": 5, "total_steps": 5}))
    assert not G.run_completed(tmp_path)
    (tmp_path / "checkpoints" / "milestone-final.pt").write_bytes(b"x")
    assert G.run_completed(tmp_path)
    (tmp_path / "train_summary.json").write_text(json.dumps({"status": "completed", "step": 3, "total_steps": 5}))   # max_steps stop
    assert not G.run_completed(tmp_path)


def test_newest_run_ignores_other_names(tmp_path):
    for rid, name in (("20261101T100000Z-train-m3-aaa", "m3"), ("20261101T110000Z-train-m3-smoke-tiny-bbb", "m3-smoke-tiny")):
        (tmp_path / "runs" / rid).mkdir(parents=True)
        (tmp_path / "runs" / rid / "run.json").write_text(json.dumps({"config": {"name": name}}))
    assert G.newest_run(tmp_path, "m3").name == "20261101T100000Z-train-m3-aaa"
    assert G.newest_run(tmp_path, "m4") is None


def test_trainer_pattern_matches_the_trainer_not_the_guard():
    import re
    pat = re.compile(G.TRAINER)
    assert pat.search("/data/LiteGPT/.venv/bin/python scripts/train.py --config benchmarks/m3/m3_run.yaml --rented-at 1")
    assert pat.search(".venv/bin/python scripts/train.py --resume runs/x")
    guard = "/data/LiteGPT/.venv/bin/python scripts/cloud/m3_guard.py --name m3 --resume-cmd /data/LiteGPT/.venv/bin/python scripts/train.py --resume {run}"
    assert not pat.search(guard)
    assert not pat.search("bash -c cd /data/LiteGPT && python scripts/train.py --config x")


def test_stop_without_the_instance_key_reports_failure(monkeypatch):
    monkeypatch.delenv("CONTAINER_ID", raising=False)
    monkeypatch.delenv("CONTAINER_API_KEY", raising=False)
    assert G.stop_instance()[0] is False


def test_monitor_reports_each_event_once():
    seen = {}
    snap = {"status": {"temp_c": 80}, "evals": [{"tokens": 250, "val": {"bpb": 1.5}}],
            "guard_events": [{"t": 1, "event": "guard_start"}, {"t": 2, "event": "resume", "why": "trainer not running"}]}
    found = M.events(seen, snap)
    assert len(found) == 2 and any("resume" in v for v in found) and any("tokens=250" in v for v in found)
    assert not any("guard_start" in v for v in found)
    assert M.events(seen, snap) == []


def test_monitor_temperature_alert_fires_once_per_episode():
    seen = {}
    hot = {"status": {"temp_c": M.HOT_C}, "evals": [], "guard_events": []}
    cool = {"status": {"temp_c": 70}, "evals": [], "guard_events": []}
    assert len(M.events(seen, hot)) == 1 and M.events(seen, hot) == [] and M.events(seen, cool) == [] and len(M.events(seen, hot)) == 1
