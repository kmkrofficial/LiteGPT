#!/usr/bin/env bash
# Fetch M4's data and the M3 final checkpoint on the rented server, into the big disk, and check every hash. Run from the repo after bootstrap.sh. Safe to run again.
#
#   bash scripts/cloud/fetch_m4.sh <big disk mount>
set -euo pipefail
BIG="${1:?big disk mount}"
DATA_REPO="kmkrworks/litegpt-v2-data"
RUNS_REPO="kmkrworks/litegpt-v2-m3-runs"
M3_RUN="20261101T142254Z-train-m3-673cd85"
M3_FINAL_SHA256="ba1103d8b6166f3c8dedf0e2e6d71bdd622ffa33015bf150e4cf06404a2c87a0"   # verified against the Hub and the laptop, 2026-11-03
HF=.venv/bin/hf

"$HF" download "$DATA_REPO" --repo-type dataset --include "m4/*" --local-dir "$BIG/m4_data"
(cd "$BIG/m4_data/m4" && sha256sum --quiet -c SHA256SUMS) || { echo "STOP: an M4 data file does not match"; exit 1; }
mkdir -p "$HOME/m4_t2d" "$HOME/m3_checkpoints/$M3_RUN"
ln -sfn "$BIG/m4_data/m4/sft" "$HOME/m4_sft"
ln -sfn "$BIG/m4_data/m4/t2d" "$HOME/m4_t2d/final"

"$HF" download "$RUNS_REPO" "$M3_RUN/milestone-final.pt" --local-dir "$BIG/m3"
ln -sfn "$BIG/m3/$M3_RUN/milestone-final.pt" "$HOME/m3_checkpoints/$M3_RUN/milestone-final.pt"
[ "$(sha256sum "$BIG/m3/$M3_RUN/milestone-final.pt" | cut -d' ' -f1)" = "$M3_FINAL_SHA256" ] || { echo "STOP: the M3 final checkpoint does not match"; exit 1; }
echo "all M4 files match"
