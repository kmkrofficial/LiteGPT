# Rented-server tools

How M3 and M4 ran unattended on one rented GPU. They are written for this project's Vast.ai setup and its private Hugging Face repos, whose names are the defaults in these scripts. To reuse them, point those at your own. No keys or tokens live in this repository: they are entered on the server.

| Tool | What it does |
|---|---|
| `bootstrap.sh` | First steps on a new server: environment, checks and data |
| `fetch_m4.sh`, `m4_fetch_models.py` | Fetch M4's data and pinned models, and check every hash |
| `upload_m4_data.py` | Put prepared data on the private repo |
| `start_m3.py`, `start_m4.py` | Start the job and its guard, each in its own tmux session |
| `m4_queue.py` | Run a job list in order. Finished jobs are skipped on restart |
| `m3_guard.py` | Restart a dead job, back up, and stop the server when done or at the spending limit |
| `m4_backup.py` | Back up records and chosen weights, each proved by hash |
| `m3_monitor.py` | Watch progress from another machine |

## How a run goes

1. Rent a server and enter a Hugging Face token on it yourself.
2. Run `bootstrap.sh`, then fetch the data and check the hashes.
3. Start the queue. Start the guard once the queue is healthy, so a start-up bug can't trip its crash-loop stop.
4. The guard stops the server by itself. It only ever stops: **destroying a server stays a person's job.**
5. Download the records, check them, then destroy the server and revoke its tokens.

## Lessons paid for

- **Check the backup list against what's actually written.** M3's guard skipped milestones, and M4's backup skipped an evaluation run.
- **Stopped servers can lose their GPU** to another renter, and a local volume can't be attached elsewhere. `vastai copy` from the volume to any cheap server worked.
- **A stopped server still charges for storage.**
