"""The M4 job queue and its hand-offs, with small real processes instead of training jobs."""
import importlib.util
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "cloud" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


Q = load("m4_queue")
B = load("m4_backup")
S = load("start_m4")
PY = sys.executable


def job(i):
    return {"id": i, "cmd": f"{PY} -c pass"}


def test_all_jobs_run_in_order_then_the_done_file_appears(tmp_path):
    jobs = [job("one"), job("two")]
    assert Q.run_jobs(jobs, tmp_path, "a", poll=0.01) == 0
    q = tmp_path / Q.QUEUE_DIR
    assert (q / "session_a.done").exists()
    assert list(json.loads((q / "session_a_finished.json").read_text())) == ["one", "two"]


def test_a_failed_job_stops_the_queue_without_a_done_file_and_a_restart_resumes_there(tmp_path):
    jobs = [job("one"), {"id": "two", "cmd": f"{PY} {write_script(tmp_path, tmp_path / 'flag')}"}, job("three")]
    assert Q.run_jobs(jobs, tmp_path, "a", poll=0.01) == 1
    q = tmp_path / Q.QUEUE_DIR
    assert not (q / "session_a.done").exists()
    assert list(json.loads((q / "session_a_finished.json").read_text())) == ["one"]
    events = [json.loads(x) for x in (q / "session_a_events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "job_end" and e["job"] == "two" and e["code"] == 3 for e in events)
    assert Q.run_jobs(jobs, tmp_path, "a", poll=0.01) == 0
    ends = [json.loads(x) for x in (q / "session_a_events.jsonl").read_text().splitlines()]
    assert sum(e["event"] == "job_start" and e["job"] == "one" for e in ends) == 1
    assert sum(e["event"] == "job_start" and e["job"] == "two" for e in ends) == 2
    assert (q / "session_a.done").exists()


def write_script(tmp_path, flag):
    s = tmp_path / "fail_once.py"
    s.write_text(f"import sys, pathlib\np = pathlib.Path(r'{flag}')\nok = p.exists()\np.touch()\nsys.exit(0 if ok else 3)\n", encoding="utf-8")
    return s


def test_a_killed_queue_restarts_without_repeating_finished_jobs(tmp_path):
    """The real failure the guard handles: the whole process dies mid-job and a new one takes over."""
    runner = tmp_path / "run_queue.py"
    slow = tmp_path / "slow.py"
    marks = tmp_path / "marks.txt"
    slow.write_text(f"import time, sys\nopen(r'{marks}', 'a').write(sys.argv[1] + ' start\\n')\ntime.sleep(float(sys.argv[2]))\nopen(r'{marks}', 'a').write(sys.argv[1] + ' end\\n')\n", encoding="utf-8")
    runner.write_text(f"""
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("m4_queue", r"{ROOT / 'scripts' / 'cloud' / 'm4_queue.py'}")
Q = importlib.util.module_from_spec(spec); spec.loader.exec_module(Q)
jobs = [{{"id": "a", "cmd": r"{PY} {slow} a 0"}}, {{"id": "b", "cmd": r"{PY} {slow} b " + sys.argv[1]}}, {{"id": "c", "cmd": r"{PY} {slow} c 0"}}]
sys.exit(Q.run_jobs(jobs, Path(r"{tmp_path}"), "a", poll=0.05))
""", encoding="utf-8")
    p = subprocess.Popen([PY, str(runner), "30"], start_new_session=True)
    deadline = time.time() + 20
    while time.time() < deadline and not (marks.exists() and "b start" in marks.read_text()):
        time.sleep(0.05)
    assert "b start" in marks.read_text()
    import os
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()
    assert subprocess.run([PY, str(runner), "0"]).returncode == 0
    lines = marks.read_text().splitlines()
    assert lines.count("a start") == 1 and lines.count("b start") == 2 and "c end" in lines
    assert (tmp_path / Q.QUEUE_DIR / "session_a.done").exists()


def test_a_stopped_job_is_not_counted_as_finished(tmp_path):
    runner = tmp_path / "run_queue.py"
    slow = tmp_path / "slow.py"
    started = tmp_path / "started"
    slow.write_text(f"import signal, sys, time, pathlib\nsignal.signal(signal.SIGINT, lambda *a: sys.exit(1))\npathlib.Path(r'{started}').touch()\ntime.sleep(30)\n", encoding="utf-8")
    runner.write_text(f"""
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("m4_queue", r"{ROOT / 'scripts' / 'cloud' / 'm4_queue.py'}")
Q = importlib.util.module_from_spec(spec); spec.loader.exec_module(Q)
sys.exit(Q.run_jobs([{{"id": "b", "cmd": r"{PY} {slow}"}}, {{"id": "c", "cmd": r"{PY} -c pass"}}], Path(r"{tmp_path}"), "a", poll=0.05))
""", encoding="utf-8")
    p = subprocess.Popen([PY, str(runner)], start_new_session=True)
    deadline = time.time() + 20
    while time.time() < deadline and not started.exists():
        time.sleep(0.05)
    import os
    os.killpg(p.pid, signal.SIGINT)                 # as the guard does: both get the signal
    assert p.wait(timeout=20) == 1
    q = tmp_path / Q.QUEUE_DIR
    assert not (q / "session_a.done").exists() and not (q / "session_a_finished.json").exists()


def test_session_a_order_and_session_b_cut_order():
    a = [j["id"] for j in Q.session_a_jobs("py")]
    assert a == ["sft-sweep-a", "sft-sweep-b", "sft-sweep-c", "sft-choose-lr", "sft", "backup"]
    arms = ["qwen3-base", "v2-base", "gemma-base"]
    b = Q.session_b_jobs("py", arms, [1e-5, 3e-5], [1, 2, 3], ["v2", "gemma", "qwen3"], "cloud")
    ids = [j["id"] for j in b]
    assert ids.index("routes") > max(i for i, v in enumerate(ids) if v.startswith("fit-"))
    assert ids.index("test") > ids.index("routes") and ids.index("backup-weights") < ids.index("test")
    assert min(i for i, v in enumerate(ids) if "qwen3" in v) > max(i for i, v in enumerate(ids) if "fit-gemma" in v)
    assert ids.index("billsum-v2") > ids.index("test") and ids[-1] == "backup"
    assert len([v for v in ids if v.startswith("fit-")]) == 3 * (2 + 2)
    assert "test" not in [j["id"] for j in Q.session_b_jobs("py", arms, [1e-5], [1], ["v2"], "laptop")]


def test_grid_learning_rates_are_the_registered_sft_points():
    assert sorted(Q.SFT_LRS.values()) == [6e-5, 2e-4, 6e-4] and Q.SWEEP_KEEP == 0.03


def test_stop_is_the_registered_figure_or_the_balance_minus_one():
    assert S.stop_usd(None, 0.0) == 9.00
    assert S.stop_usd(10.37, 0.0) == 9.00
    assert S.stop_usd(8.00, 0.0) == 7.00
    assert S.stop_usd(5.0, 4.0) == pytest.approx(8.0)


class Info:
    def __init__(self, lfs=None, blob_id=None):
        self.lfs, self.blob_id = lfs, blob_id


class Lfs:
    def __init__(self, sha256):
        self.sha256 = sha256


def test_backup_trusts_only_the_hubs_own_hash(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    assert B.hub_matches(Info(lfs=Lfs(B.sha256(f))), f)
    assert not B.hub_matches(Info(lfs=Lfs("0" * 64)), f)
    assert B.hub_matches(Info(blob_id=B.git_blob_sha1(f)), f)
    assert B.git_blob_sha1(f) == "b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0"       # `git hash-object` of the same bytes
    assert not B.hub_matches(Info(blob_id="0" * 40), f)
