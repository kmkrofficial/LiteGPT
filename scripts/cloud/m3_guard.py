"""Server-side guard for the rented M3 run: works with the laptop off and no assistant attached.

    python scripts/cloud/m3_guard.py --name m3 --resume-cmd "uv run python scripts/train.py --resume {run}" --rented-at <unix time>

Every minute: restart a dead or stalled trainer from its checkpoint, push the records and the newest checkpoint to a private
Hugging Face repo, and STOP the instance when the run has completed and its final checkpoint is verified on the Hub (sha256), or
when the spending cap or the restart limit is reached. It only ever stops: a stopped instance keeps its disk and ends GPU billing.
Destroying the instance is left to the owner. Status goes to ~/guard_status.json and ~/guard_events.jsonl (read by m3_monitor.py).
With --ntfy-topic the owner's phone also gets short alerts (https://ntfy.sh): they outlive the laptop, any assistant and the stop itself.
Alerts carry status text only, never tokens or data, and a failed alert never affects the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

CHECK_SECONDS = 60
STALL_SECONDS = 30 * 60        # metrics are logged about once a minute; an evaluation takes minutes
KILL_AFTER_SECONDS = 5 * 60    # after SIGINT (which saves a checkpoint), before SIGKILL
RESTART_LIMIT = 3              # within RESTART_WINDOW_SECONDS, then stop: a crash loop must not keep billing
RESTART_WINDOW_SECONDS = 2 * 3600
FINAL_BACKUP_ATTEMPTS = 5      # then stop anyway: the disk survives a stop, the bill does not grow
CAP_MARGIN_INR = 50            # stop this far below the cap: bandwidth and the final backup are billed too
ALERTS = {"guard_start", "resume", "interrupt", "kill", "guard_error", "final_backup", "stop_requested"}
PROGRESS_SECONDS = 6 * 3600    # a quiet channel must mean "healthy", so the guard also says it is alive


@dataclass
class State:
    completed: bool = False
    final_verified: bool = False
    final_attempts: int = 0
    trainer_alive: bool = False
    metrics_age_s: float = 0.0
    recent_restarts: int = 0
    spent_inr: float = 0.0
    cap_inr: float = 2500.0
    backup_age_s: float = 0.0
    backup_every_s: float = 3 * 3600
    halting: str = ""             # reason, once a stop has been decided
    interrupted_age_s: float = -1.0


def decide(s: State) -> tuple[str, str]:
    """The next action and why. Pure, so every path is tested without a server."""
    if s.completed:
        if s.final_verified:
            return "stop", "completed: final checkpoint verified on the Hub"
        if s.final_attempts >= FINAL_BACKUP_ATTEMPTS:
            return "stop", "completed, but the final backup could not be verified: the checkpoint is still on the stopped disk"
        return "backup_final", "completed"
    reason = s.halting
    if not reason and s.spent_inr >= s.cap_inr - CAP_MARGIN_INR:
        reason = f"spending cap: INR {s.spent_inr:.0f} of {s.cap_inr:.0f}"
    if not reason and not s.trainer_alive and s.recent_restarts >= RESTART_LIMIT:
        reason = f"{s.recent_restarts} restarts within {RESTART_WINDOW_SECONDS // 3600} h"
    if reason:
        if s.trainer_alive:
            return ("kill", reason) if s.interrupted_age_s >= KILL_AFTER_SECONDS else ("wait" if s.interrupted_age_s >= 0 else "interrupt", reason)
        if s.final_verified or s.final_attempts >= FINAL_BACKUP_ATTEMPTS:
            return "stop", reason
        return "backup_final", reason
    if not s.trainer_alive:
        return "resume", "trainer not running"
    if s.metrics_age_s > STALL_SECONDS:
        return ("kill", "stalled") if s.interrupted_age_s >= KILL_AFTER_SECONDS else ("wait" if s.interrupted_age_s >= 0 else "interrupt", "stalled")
    if s.backup_age_s >= s.backup_every_s:
        return "backup", "periodic"
    return "none", ""


def alert_text(kind: str, fields: dict) -> str | None:
    """The phone message for a guard event, or None if the event is routine."""
    if kind == "backup_done" and not fields.get("verified"):
        return f"Backup NOT verified: {fields.get('checkpoint')}"
    if kind not in ALERTS:
        return None
    keep = ("why", "ok", "detail", "verified", "checkpoint", "attempt", "error", "spent_inr")
    return " ".join([kind.replace("_", " ")] + [f"{k}={v}" for k, v in fields.items() if k in keep])[:400]


def send_alert(topic: str | None, text: str, title: str = "LiteGPT M3", urgent: bool = False) -> bool:
    if not topic:
        return False
    req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=text.encode("utf-8"), headers={"Title": title, "Priority": "high" if urgent else "default"})
    try:
        urllib.request.urlopen(req, timeout=15).read()
        return True
    except Exception:  # noqa: BLE001 - an alert must never affect the run
        return False


def newest_run(repo: Path, name: str) -> Path | None:
    """The newest run whose configured name is exactly `name` (m3-smoke-tiny is not m3)."""
    def named(p: Path) -> bool:
        try:
            return json.loads((p / "run.json").read_text(encoding="utf-8"))["config"]["name"] == name
        except (OSError, ValueError, KeyError):
            return False
    runs = sorted((p for p in (repo / "runs").glob(f"*-train-{name}-*") if named(p)), key=lambda p: p.name)
    return runs[-1] if runs else None


def run_completed(run: Path) -> bool:
    try:
        s = json.loads((run / "train_summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return s.get("status") == "completed" and s.get("step") == s.get("total_steps") and (run / "checkpoints" / "milestone-final.pt").exists()


def newest_checkpoint(run: Path) -> Path | None:
    regular = sorted((run / "checkpoints").glob("ckpt-*.pt"))
    return regular[-1] if regular else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


class Backup:
    """Records and checkpoints to a private repo. New paths only: earlier backups are never overwritten or deleted."""

    def __init__(self, repo_id: str):
        from huggingface_hub import HfApi
        self.api, self.repo_id = HfApi(), repo_id
        self.api.create_repo(repo_id, private=True, exist_ok=True)

    def records(self, run: Path) -> None:
        self.api.upload_folder(repo_id=self.repo_id, folder_path=str(run), path_in_repo=run.name, ignore_patterns=["checkpoints/*"],
                               commit_message="records")

    def checkpoint(self, run: Path, ckpt: Path) -> bool:
        """Upload, then compare the Hub's sha256 with the local file."""
        dest = f"{run.name}/{ckpt.name}"
        self.api.upload_file(path_or_fileobj=str(ckpt), path_in_repo=dest, repo_id=self.repo_id, commit_message=ckpt.name)
        info = self.api.get_paths_info(self.repo_id, [dest])
        return bool(info and info[0].lfs and info[0].lfs.sha256 == sha256(ckpt))


def stop_instance() -> tuple[bool, str]:
    """`vastai stop instance` with the restricted per-instance key Vast provides; the key is never printed or logged."""
    cid = os.environ.get("CONTAINER_ID")
    if not cid or not os.environ.get("CONTAINER_API_KEY"):
        return False, "CONTAINER_ID / CONTAINER_API_KEY not in the environment"
    p = subprocess.run(["vastai", "stop", "instance", cid, "--api-key", os.environ["CONTAINER_API_KEY"]], capture_output=True, text=True)
    return p.returncode == 0, f"exit {p.returncode}"


TRAINER = r"^[^ ]*python[^ ]* scripts/train[.]py"       # anchored: the guard's own --resume-cmd argument contains this path too


def trainer_pids() -> list[int]:
    out = subprocess.run(["pgrep", "-f", TRAINER], capture_output=True, text=True).stdout.split()
    return [int(v) for v in out]


def gpu_temp() -> int | None:
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.split()[0])
    except (OSError, ValueError, IndexError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-dir", default=str(Path.home() / "LiteGPT"))
    ap.add_argument("--name", default="m3", help="run name: the guard watches the newest runs/*-train-<name>-*")
    ap.add_argument("--resume-cmd", required=True, help="{run} = the run directory")
    ap.add_argument("--backup-repo", default="kmkrworks/litegpt-v2-m3-runs")
    ap.add_argument("--rented-at", type=float, required=True, help="unix time the instance started billing")
    ap.add_argument("--price-usd-h", type=float, default=0.60)
    ap.add_argument("--inr-per-usd", type=float, default=98.40)
    ap.add_argument("--cap-inr", type=float, default=2500.0)
    ap.add_argument("--backup-hours", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true", help="log the stop instead of stopping (setup check)")
    ap.add_argument("--ntfy-topic", default=None, help="private ntfy.sh topic for phone alerts")
    a = ap.parse_args()
    repo, home = Path(a.repo_dir), Path.home()
    backup = Backup(a.backup_repo)
    log = home / "guard_events.jsonl"

    def event(kind: str, **kw) -> None:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "event": kind, **kw}) + "\n")
        text = alert_text(kind, kw)
        if text:
            send_alert(a.ntfy_topic, text, urgent=kind != "guard_start")

    s = State(cap_inr=a.cap_inr, backup_every_s=a.backup_hours * 3600)
    last_backup, restarts, interrupted_at = time.time(), [], None
    last_progress, gate_reported = time.time(), False
    event("guard_start", dry_run=a.dry_run)
    while True:
        now = time.time()
        run = newest_run(repo, a.name)
        pids = trainer_pids()
        metrics = run / "metrics.jsonl" if run else None
        s.completed = bool(run and run_completed(run))
        s.trainer_alive = bool(pids)
        s.metrics_age_s = now - metrics.stat().st_mtime if metrics and metrics.exists() else 0.0
        restarts = [t for t in restarts if now - t < RESTART_WINDOW_SECONDS]
        s.recent_restarts = len(restarts)
        s.spent_inr = (now - a.rented_at) / 3600 * a.price_usd_h * a.inr_per_usd
        s.backup_age_s = now - last_backup
        s.interrupted_age_s = now - interrupted_at if interrupted_at is not None and pids else -1.0
        if not pids:
            interrupted_at = None
        if run and not gate_reported and (run / "gate" / "extension_gate.json").exists():
            gate_reported = True
            g = json.loads((run / "gate" / "extension_gate.json").read_text(encoding="utf-8"))
            send_alert(a.ntfy_topic, f"Gate: {g.get('decision')}, delta {g.get('delta')}, stable end {g.get('stable_end_tokens')}, {g.get('reason')}", title="LiteGPT M3 gate")
        if metrics and metrics.exists() and now - last_progress >= PROGRESS_SECONDS:
            last_progress = now
            with open(metrics, "rb") as f:
                f.seek(max(0, metrics.stat().st_size - 400))
                last = json.loads(f.read().decode("utf-8").strip().splitlines()[-1])
            send_alert(a.ntfy_topic, f"OK: {last['tokens'] / 1e9:.2f}B tokens, loss {last['loss']:.3f}, {last['tok_per_s'] / 1e3:.0f}k tok/s, spent about INR {s.spent_inr:.0f}, GPU {gpu_temp()} C")
        action, why = decide(s)
        if action == "resume" and not run:          # started before the first run exists: nothing to resume yet
            action, why = "none", ""
        if action in ("interrupt", "wait", "kill", "backup_final", "stop") and why != "stalled" and not s.completed:
            s.halting = why
        if action != "none" and action != "wait":
            event(action, why=why, spent_inr=round(s.spent_inr))
        try:
            if action == "resume" and run:
                restarts.append(now)
                with open(home / "trainer.log", "a", encoding="utf-8") as f:
                    subprocess.Popen(a.resume_cmd.format(run=run.relative_to(repo)), shell=True, cwd=repo, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            elif action == "interrupt":
                interrupted_at = now
                for p in pids:
                    os.kill(p, signal.SIGINT)
            elif action == "kill":
                for p in pids:
                    os.kill(p, signal.SIGKILL)
            elif action == "backup" and run:
                backup.records(run)
                ck = newest_checkpoint(run)
                ok = backup.checkpoint(run, ck) if ck else False
                last_backup = now
                event("backup_done", checkpoint=ck.name if ck else None, verified=ok)
            elif action == "backup_final" and run:
                s.final_attempts += 1
                backup.records(run)
                final = run / "checkpoints" / "milestone-final.pt"
                ck = final if final.exists() else newest_checkpoint(run)
                s.final_verified = backup.checkpoint(run, ck) if ck else False
                event("final_backup", checkpoint=ck.name if ck else None, verified=s.final_verified, attempt=s.final_attempts)
            elif action == "stop":
                backup.api.upload_file(path_or_fileobj=str(log), path_in_repo=f"{run.name}/guard_events.jsonl", repo_id=a.backup_repo, commit_message="guard events") if run else None
                send_alert(a.ntfy_topic, f"Stopping the server now: {why}. Spent about INR {s.spent_inr:.0f}. Check the Vast console, then destroy the instance and delete the volume.", urgent=True)
                ok, detail = (True, "dry run") if a.dry_run else stop_instance()
                event("stop_requested", why=why, ok=ok, detail=detail)
                if a.dry_run:
                    return 0
        except Exception as e:  # noqa: BLE001 - the guard must outlive a network error; the next pass retries
            event("guard_error", action=action, error=f"{type(e).__name__}: {str(e)[:300]}")
        status = {"t": now, "run": run.name if run else None, "action": action, "why": why, "temp_c": gpu_temp(), **asdict(s)}
        (home / "guard_status.json").write_text(json.dumps(status), encoding="utf-8")
        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
