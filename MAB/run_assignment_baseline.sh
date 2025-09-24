#!/usr/bin/env bash
# Assignment Part 1 + Part 2 runs for student scaffold (Bernoulli testbed)
# Based on main.tex (PS1) — baseline and sensitivity experiments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Outputs at repo root
OUT_DIR="${ROOT_DIR}/MAB2/MAB/plots_assignment"
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

# Defaults from assignment text
K=10
T=1000
RUNS=200
SEED=0

echo "== Experiment 1: Baseline Comparison (epsilon in {0.0, 0.1}, UCB c=2.0) =="
run --algo epsilon --epsilon 0.0 --k "$K" --T "$T" --runs "$RUNS" --seed "$SEED" \
    --outdir "$OUT_DIR" --save-csv
run --algo epsilon --epsilon 0.1 --k "$K" --T "$T" --runs "$RUNS" --seed "$SEED" \
    --outdir "$OUT_DIR" --save-csv
run --algo ucb --c 2.0 --k "$K" --T "$T" --runs "$RUNS" --seed "$SEED" \
    --outdir "$OUT_DIR" --save-csv

echo "\n== Experiment 2: Hyperparameter Sensitivity (epsilon, c grids) =="
for EPS in 0.0 0.01 0.05 0.1 0.2; do
  run --algo epsilon --epsilon "$EPS" --k "$K" --T "$T" --runs "$RUNS" --seed "$SEED" \
      --outdir "$OUT_DIR" --save-csv
done

for C in 0.5 1.0 2.0; do
  run --algo ucb --c "$C" --k "$K" --T "$T" --runs "$RUNS" --seed "$SEED" \
      --outdir "$OUT_DIR" --save-csv
done

echo "\nAssignment runs completed. Artifacts in: $OUT_DIR"

