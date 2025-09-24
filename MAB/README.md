# Multi-Armed Bandits

Student scaffold for implementing and analyzing multi-armed bandit algorithms in a Gymnasium environment.

## Prerequisites

- **Python 3.10+**
- **uv** package manager ([installation guide](https://astral.sh/uv))

### Installing uv

```bash
# macOS (Homebrew)
brew install uv

# Linux/macOS (installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | more"
```

## Setup

```bash
# Create Python 3.10 virtual environment
uv python install 3.10
uv venv --python 3.10 

# Activate environment
source .venv/bin/activate  # Unix/macOS
# or
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install package
uv sync 

# Verify installation
python -c "import sys; print(sys.version)"
```

## Implementation Tasks

1. **Agents**: Complete TODOs in `MAB/agents.py`
2. **Environment**: Implement `KArmedBanditEnv` in `MAB/bandit_env.py`
3. **Optional**: Extend plotting/CLI in `MAB/experiments.py`

## Running Experiments

Use the `mab-student` command with appropriate parameters:

### Basic Examples

```bash
# Epsilon-greedy
uv run mab-student --algo epsilon --epsilon 0.1 \
  --k 10 --T 1000 --runs 200 --seed 0 --outdir plots --save-csv

# UCB
uv run mab-student --algo ucb --c 2.0 \
  --k 10 --T 1000 --runs 200 --seed 0 --outdir plots --save-csv

# Thompson Sampling
uv run mab-student --algo thompson \
  --k 10 --T 1000 --runs 200 --seed 0 --outdir plots --save-csv
```

### Parameters

| Parameter | Description | Options/Range |
|-----------|-------------|---------------|
| `--algo` | Algorithm selection | `epsilon`, `ucb`, `thompson` |
| `--k` | Number of arms | Integer |
| `--T` | Time steps | Integer |
| `--runs` | Number of runs | Integer |
| `--seed` | Random seed | Integer |
| `--epsilon` | Exploration rate (ε-greedy) | Float [0,1] |
| `--c` | Exploration constant (UCB) | Float |
| `--step-size` | Learning rate (optional) | Float |
| `--nonstationary` | Enable non-stationary environment | Flag |
| `--sigma` | Noise parameter | Float |
| `--outdir` | Output directory | Path (default: `plots/`) |
| `--save-csv` | Export results to CSV | Flag |

## Outputs

- **Plots**: Mean reward, optimal action percentage, and cumulative regret with 95% confidence intervals
- **CSV**: Summary statistics appended to `{outdir}/summary.csv` when `--save-csv` is enabled

## Batch Experiments

```bash
# Part 2: Baselines and sensitivity analysis
bash run_assignment_baseline.sh  # → plots_assignment/

# Bonus: Non-stationary comparison
bash run_assignment_bonus.sh  # → plots_assignment_bonus/
```

**Windows users**: Run `.sh` scripts via Git Bash

## Troubleshooting

- **Headless systems**: Set `export MPLBACKEND=Agg` (Unix) or `$env:MPLBACKEND='Agg'` (PowerShell)
- **uv not found**: Ensure proper installation from https://astral.sh/uv

## Deliverables

- Completed implementations (`MAB/agents.py`, `MAB/bandit_env.py`)
- Generated plots and `summary.csv`
- Analysis per `MAB/CHECKLIST.md` requirements
