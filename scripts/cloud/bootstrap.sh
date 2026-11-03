#!/usr/bin/env bash
# First steps on the rented M3 server (docs/private/M3_CLOUD_RUNBOOK.md). Safe to run again.
#
#   bash bootstrap.sh <big disk mount> <litegpt-v2.bundle>
#
# Everything large goes on the big disk: the container disk is small and a full one would break the run. flash-attn comes only from
# the prebuilt wheel pinned in uv.lock; never build it from source. Before this, the owner runs `hf auth login` with a read token.
#
# M4: LITEGPT_MILESTONE=m4 bash bootstrap.sh <disk> <bundle> needs 70 GB free, skips the M3 shards and the flash-attn group (M4 trains with SDPA);
# then run scripts/cloud/fetch_m4.sh.
set -euo pipefail
BIG="${1:?big disk mount}"
BUNDLE="${2:?git bundle of the repo}"
MILESTONE="${LITEGPT_MILESTONE:-m3}"
MIN_FREE_GB=90
UV_GROUP=(--group flash)
if [ "$MILESTONE" = "m4" ]; then MIN_FREE_GB=70; UV_GROUP=(); fi
DATA_REPO="kmkrworks/litegpt-v2-data"
DATA_RUN="runs/20261011T173054Z-data-prepare-1f005bc"
BRANCH="v2/m0-foundations"

echo "=== host"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$driver" -ge 580 ] || { echo "STOP: driver $driver is below 580 (CUDA 13)"; exit 1; }
free_gb=$(df -BG --output=avail "$BIG" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge "$MIN_FREE_GB" ] || { echo "STOP: only ${free_gb} GB free on $BIG"; exit 1; }
df -h "$BIG" /
nproc

echo "=== environment on $BIG"
ENV_FILE="$HOME/.litegpt_env"
cat > "$ENV_FILE" <<EOF
export UV_CACHE_DIR="$BIG/uv-cache"
export UV_PYTHON_INSTALL_DIR="$BIG/uv-python"
export HF_HOME="$BIG/hf"
export TMPDIR="$BIG/tmp"
export TRITON_CACHE_DIR="$BIG/triton"
export PATH="\$HOME/.local/bin:\$PATH"
EOF
grep -q litegpt_env "$HOME/.bashrc" 2>/dev/null || echo ". \"$ENV_FILE\"" >> "$HOME/.bashrc"
. "$ENV_FILE"
mkdir -p "$BIG/uv-cache" "$BIG/uv-python" "$BIG/hf" "$BIG/tmp" "$BIG/triton"

command -v gcc > /dev/null || { apt-get update -qq; apt-get install -y -qq build-essential; }   # torch.compile needs a C compiler
command -v tmux > /dev/null || apt-get install -y -qq tmux
UV_VERSION=0.12.23                 # pyproject.toml pins it; base images ship other versions
[ "$(uv --version 2>/dev/null | cut -d' ' -f2)" = "$UV_VERSION" ] || curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh
hash -r
uv --version
command -v vastai > /dev/null || uv tool install --quiet vastai || true

echo "=== code"
if [ ! -d "$BIG/LiteGPT/.git" ]; then git clone -q -b "$BRANCH" "$BUNDLE" "$BIG/LiteGPT"; fi
ln -sfn "$BIG/LiteGPT" "$HOME/LiteGPT"
cd "$BIG/LiteGPT"
git log --oneline -1

echo "=== data + packages (at the same time)"
uv sync --locked "${UV_GROUP[@]}" > "$BIG/uv-sync.log" 2>&1 &
sync_pid=$!
wait "$sync_pid" || { for i in 1 2 3; do sleep 30; uv sync --locked "${UV_GROUP[@]}" >> "$BIG/uv-sync.log" 2>&1 && break; done; }
.venv/bin/hf auth whoami > /dev/null || { echo "STOP: run hf auth login first"; exit 1; }
if [ "$MILESTONE" = "m3" ]; then
  .venv/bin/hf download "$DATA_REPO" --repo-type dataset --local-dir "$DATA_RUN"
  echo "=== sha256 of every data file"
  (cd "$DATA_RUN" && sha256sum --quiet -c SHA256SUMS) || { echo "STOP: a data file does not match"; exit 1; }
  echo "all data files match"
fi

echo "=== checks the guard needs"
vastai --version 2>/dev/null | head -1 || echo "WARNING: vastai CLI missing"
[ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ] && echo "instance key present" || echo "WARNING: CONTAINER_ID / CONTAINER_API_KEY not in this shell (the automatic stop needs them)"
if [ "$MILESTONE" = "m3" ]; then .venv/bin/python -c "import flash_attn"; fi
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "=== bootstrap done"
