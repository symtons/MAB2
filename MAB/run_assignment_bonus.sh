#!/usr/bin/env bash
# Assignment Bonus runs (non-stationary bandits): compare epsilon-greedy
# sample-average vs constant step-size updates under random-walk means.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUT_DIR="${ROOT_DIR}/MAB2/MAB/plots_assignment_bonus"
mkdir -p "$OUT_DIR"

echo "== Activating venv in ${SCRIPT_DIR} (if present) =="
if [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.venv/bin/activate"
elif [[ -f "${SCRIPT_DIR}/.venv/Scripts/activate" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.venv/Scripts/activate"
fi

export MPLBACKEND=Agg

run() {
  echo "\n>> $*"
  uv run mab-student "$@"
}

# Defaults 
K=10
T=1000
RUNS=200
SEED=0
SIGMA=0.05   # drift stddev
EPS=0.1      # epsilon for comparison

echo "== Bonus: Non-stationary bandits (sigma=${SIGMA}) =="
echo "-- Epsilon-greedy (sample-average updates) --"
run --algo epsilon --epsilon "$EPS" --k "$K" --T "$T" --runs "$RUNS" --seed "$SEED" \
    --nonstationary --sigma "$SIGMA" --outdir "$OUT_DIR" --save-csv

echo "-- Epsilon-greedy (constant step-size: 0.1) --"
run --algo epsilon --epsilon "$EPS" --step-size 0.1 --k "$K" --T "$T" --runs "$RUNS" --seed "$SEED" \
    --nonstationary --sigma "$SIGMA" --outdir "$OUT_DIR" --save-csv

echo "-- Epsilon-greedy (constant step-size: 0.05) --"
run --algo epsilon --epsilon "$EPS" --step-size 0.05 --k "$K" --T "$T" --runs "$RUNS" --seed "$SEED" \
    --nonstationary --sigma "$SIGMA" --outdir "$OUT_DIR" --save-csv

echo "\nBonus runs completed. Artifacts in: $OUT_DIR"

