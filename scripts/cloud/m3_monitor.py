"""Laptop-side monitor of the rented M3 run. Exits with an EVENT line when something needs the assistant; run it again to continue.

    python scripts/cloud/m3_monitor.py            # ~/.litegpt_m3_ssh holds the instance's "<port> <host>"

Read-only on the server: it only reads the guard's status, the guard's events and the run's evaluations over SSH. What it has
already reported is kept in ~/.litegpt_m3_monitor.json, so a restart never repeats an event and a missed one is still reported.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

POLL_SECONDS = 300
UNREACHABLE_POLLS = 3          # ~15 minutes: a stopped or lost server looks the same from here, the guard's last events tell which
HOT_C = 88
STATE = Path.home() / ".litegpt_m3_monitor.json"
KEY = Path.home() / ".ssh" / "litegpt_vast"
REMOTE = ("cat ~/guard_status.json 2>/dev/null || echo '{}'; echo; echo ---; cat ~/guard_events.jsonl 2>/dev/null; echo ---; "
          "tail -q -n 400 $(ls -d ~/LiteGPT/runs/*-train-m3-* 2>/dev/null | tail -1)/eval.jsonl 2>/dev/null")
QUIET_EVENTS = {"guard_start", "backup", "backup_done", "wait"}


def fetch(port: str, host: str) -> dict | None:
    cmd = ["ssh", "-i", str(KEY), "-p", port, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new",
           f"root@{host}", REMOTE]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    if p.returncode != 0:
        return None
    parts = p.stdout.split("\n---\n")
    if len(parts) < 3:
        return None
    lines = lambda text: [json.loads(v) for v in text.splitlines() if v.strip()]  # noqa: E731
    return {"status": json.loads(parts[0] or "{}"), "guard_events": lines(parts[1]), "evals": lines(parts[2])}


def events(seen: dict, snap: dict) -> list[str]:
    """What is new in this snapshot. Updates `seen`."""
    out = []
    new_guard = snap["guard_events"][seen.get("guard_events", 0):]
    seen["guard_events"] = len(snap["guard_events"])
    for e in new_guard:
        if e["event"] not in QUIET_EVENTS and not (e["event"] == "backup_done" and e.get("verified")):
            out.append("guard " + json.dumps({k: v for k, v in e.items() if k != "t"}))
    new_evals = snap["evals"][seen.get("evals", 0):]
    seen["evals"] = len(snap["evals"])
    for e in new_evals:
        out.append(f"eval tokens={e.get('tokens')} val_bpb={e.get('val', {}).get('bpb')}")
    temp = snap["status"].get("temp_c")
    if temp is not None and temp >= HOT_C and not seen.get("hot"):
        out.append(f"gpu temperature {temp} C")
    seen["hot"] = bool(temp is not None and temp >= HOT_C)
    return out


def main() -> int:
    port, host = (Path.home() / ".litegpt_m3_ssh").read_text(encoding="utf-8").split()
    seen = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    misses = 0
    while True:
        snap = fetch(port, host)
        if snap is None:
            misses += 1
            if misses >= UNREACHABLE_POLLS:
                print(f"EVENT server unreachable for {misses} polls; last guard events reported: {seen.get('guard_events', 0)}", flush=True)
                return 0
        else:
            misses = 0
            found = events(seen, snap)
            STATE.write_text(json.dumps(seen), encoding="utf-8")
            if found:
                print("\n".join(f"EVENT {v}" for v in found), flush=True)
                return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
