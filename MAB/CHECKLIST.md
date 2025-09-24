# Homework 1: Multi-Armed Bandits

## Implementation Requirements

### Agents (`MAB/agents.py`)

**Epsilon-Greedy**
- `select_action(t)`: ε-probability random exploration, greedy exploitation with uniform tie-breaking
- `update(a, r)`: Sample-average (1/N[a]) or constant step-size when specified

**UCB1**
- `select_action(t)`: Cover unpulled arms first, then argmax[Q + c√(ln t / N)]
- `update(a, r)`: Sample-average update (1/N[a])

### Environment (`MAB/bandit_env.py`)

**KArmedBanditEnv Specifications**
- Action space: Discrete(k)
- Observation: Constant (0)
- Rewards: Bernoulli with p ~ Uniform(0,1) at reset
- Task type: Continuing
- Required info: `info['p']` (true probabilities), `info['optimal']` (best arm)
- Features: Seed support in reset, optional non-stationary random walk (σ parameter)

### Implementation Conventions
- 1-based time indexing
- Uniform tie-breaking among maxima
- Bernoulli testbed (stationary) unless specified

## Experimental Protocol

### Configuration
- **Default parameters**: k=10, T=1000, runs=200, seed=0
- **Environment**: Bernoulli testbed (stationary)

### Experiment 1: Baseline Comparison
**Algorithms**
- Epsilon-greedy: ε ∈ {0.0, 0.1}
- UCB: c = 2.0

**Required outputs**
- Three plots with 95% confidence intervals:
  - Mean reward over time
  - Percentage of optimal actions
  - Cumulative regret
- Brief analysis (2-3 sentences)

### Experiment 2: Sensitivity Analysis
**Parameter ranges**
- Epsilon-greedy: ε ∈ {0.0, 0.01, 0.05, 0.1, 0.2}
- UCB: c ∈ {0.5, 1.0, 2.0}

**Required outputs**
- Table of final mean rewards at t=T
- Brief discussion (2-3 sentences)

## Execution Instructions

### Environment Setup
```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate  # Unix/macOS


# Install package
uv sync
```

### Running Experiments
```bash
# Epsilon-greedy example
uv run mab-student --algo epsilon --epsilon 0.1 \
  --k 10 --T 1000 --runs 200 --seed 0 --outdir plots --save-csv

# UCB example
uv run mab-student --algo ucb --c 2.0 \
  --k 10 --T 1000 --runs 200 --seed 0 --outdir plots --save-csv
```

### Batch Execution
```bash
bash run_assignment_baseline.sh     # → plots_assignment/
bash run_assignment_bonus.sh        # → plots_assignment_bonus/ (optional)
```

## Output Requirements

- **Plots**: Save to `plots/` or specified `--outdir`
- **Data**: Use `--save-csv` for summary statistics in `outdir/summary.csv`
- **Format**: Ensure proper axis labels and readable legends
- **Headless systems**: Set `MPLBACKEND=Agg` environment variable

## Optional Extension: Non-Stationary Bandits

- Add flags: `--nonstationary --sigma 0.05`
- Compare sample-average vs. constant step-size for epsilon-greedy
- Document adaptation behavior

## Submission Guidelines

### Required Deliverables
1. Updated source files (`MAB/agents.py`, `MAB/bandit_env.py`)
2. Generated plots from experiments
3. Summary CSV file
4. Written analysis (PDF or Markdown) including:
   - Experiment plots
   - Brief interpretations as specified

### Reproducibility Requirements
- Maintain provided seeds for consistency
- Preserve API signatures
- Limit modifications to `MAB/` directory
