"""Start one M4 session on the rented server, each part in its own tmux session.

    python scripts/cloud/start_m4.py --session a --part queue
    python scripts/cloud/start_m4.py --session a --part guard --rented-at <unix time> --price-usd-h 0.45 --ntfy-topic <topic> [--dry-run]

Start the queue first and the guard once it is healthy, so a start-up bug cannot trip the crash-loop stop. For session B, pass what session A
cost with --carried-usd. The guard stops the server at $9.00 in total, or at the balance minus $1 if that is lower (--balance-usd).
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "cloud"))
GUARD_STOP_USD = 9.00   # spec m4.budget


def stop_usd(balance_usd: float | None, carried_usd: float) -> float:
    """The registered stop, or the balance minus $1 if lower. The balance read at this session's start is on top of what was already spent."""
    if balance_usd is None:
        return GUARD_STOP_USD
    return min(GUARD_STOP_USD, carried_usd + balance_usd - 1.00)


def main() -> int:
    from m4_queue import WATCH_PATTERN
    from start_m3 import pid1_env
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", choices=["a", "b"], required=True)
    ap.add_argument("--part", choices=["queue", "guard"], required=True)
    ap.add_argument("--rented-at", type=float)
    ap.add_argument("--price-usd-h", type=float)
    ap.add_argument("--carried-usd", type=float, default=0.0)
    ap.add_argument("--balance-usd", type=float, help="credit shown in the console at this session's start")
    ap.add_argument("--test-on", choices=["cloud", "laptop"], default="cloud")
    ap.add_argument("--without-arm", action="append", default=[])
    ap.add_argument("--backup-repo", default="kmkrworks/litegpt-v2-m3-runs")
    ap.add_argument("--ntfy-topic", default=None)
    ap.add_argument("--backup-hours", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true", help="the guard logs its stop instead of stopping")
    a = ap.parse_args()
    home = Path.home()
    py = str(ROOT / ".venv" / "bin" / "python")
    queue_args = [py, "scripts/cloud/m4_queue.py", "--session", a.session, "--test-on", a.test_on] + [v for arm in a.without_arm for v in ("--without-arm", arm)]
    if a.part == "queue":
        if subprocess.run(["pgrep", "-f", WATCH_PATTERN], capture_output=True).stdout:
            raise SystemExit("a job is already running")
        cmd = f"cd {ROOT} && . {home}/.litegpt_env && PYTHONPATH=. PYTHONUNBUFFERED=1 {shlex.join(queue_args)} 2>&1 | tee -a {home}/trainer.log"
        subprocess.run(["tmux", "new-session", "-d", "-s", f"m4{a.session}", cmd], check=True)
        print("started queue, session", a.session)
        return 0
    if a.rented_at is None or a.price_usd_h is None:
        raise SystemExit("the guard needs --rented-at and --price-usd-h")
    if subprocess.run(["pgrep", "-f", "scripts/cloud/m3_guard[.]py"], capture_output=True).stdout:
        raise SystemExit("the guard is already running")
    env = pid1_env()
    if not a.dry_run and (not env.get("CONTAINER_ID") or not env.get("CONTAINER_API_KEY")):
        raise SystemExit("the instance key is missing: the automatic stop would not work")
    guard_args = [py, "scripts/cloud/m3_guard.py", "--repo-dir", str(ROOT), "--rented-at", str(a.rented_at), "--price-usd-h", str(a.price_usd_h), "--backup-repo", a.backup_repo,
                  "--stop-usd", f"{stop_usd(a.balance_usd, a.carried_usd):.2f}", "--carried-usd", str(a.carried_usd), "--backup-hours", str(a.backup_hours), "--alert-title", f"LiteGPT M4 {a.session.upper()}",
                  "--trainer-pattern", WATCH_PATTERN, "--done-file", f"runs/m4_queue/session_{a.session}.done", "--heartbeat", f"runs/m4_queue/session_{a.session}.heartbeat",
                  "--backup-cmd", f"{py} scripts/cloud/m4_backup.py --session {a.session}" + (" --weights" if a.session == "b" else ""),
                  "--resume-cmd", f"PYTHONPATH=. PYTHONUNBUFFERED=1 {shlex.join(queue_args)}"]
    if a.ntfy_topic:
        guard_args += ["--ntfy-topic", a.ntfy_topic]
    if a.dry_run:
        guard_args.append("--dry-run")
    keyfile = home / ".litegpt_guard_env"          # the key reaches the guard through its own file, readable by root only; never on a command line here
    keyfile.touch(mode=0o600)
    keyfile.write_text("".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items()), encoding="utf-8")
    cmd = f". {keyfile} && cd {ROOT} && . {home}/.litegpt_env && export PATH=/opt/instance-tools/bin:$PATH && PYTHONUNBUFFERED=1 {shlex.join(guard_args)} 2>&1 | tee -a {home}/guard.log"
    subprocess.run(["tmux", "new-session", "-d", "-s", "guard", cmd], check=True)
    print("started guard, stop at $", f"{stop_usd(a.balance_usd, a.carried_usd):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
