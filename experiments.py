from __future__ import annotations

"""Experiment runner for the Multi-Armed Bandits assignment.

This CLI runs N independent runs of horizon T on a k-armed Bernoulli bandit
environment using a specified agent and hyperparameters. Students should extend
plotting.

"""

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import os
import matplotlib.pyplot as plt

try:
    from .agents import Agent, EpsilonGreedyAgent, UCBAgent, ThompsonSamplingAgent
    from .bandit_env import KArmedBanditEnv
except Exception as e:  # pragma: no cover - allow editing without deps installed
    Agent = object  # type: ignore
    EpsilonGreedyAgent = object  # type: ignore
    UCBAgent = object  # type: ignore
    ThompsonSamplingAgent = object  # type: ignore
    KArmedBanditEnv = object  # type: ignore


@dataclass
class Config:
    """Configuration for a batch of experiments.

    Attributes mirror CLI flags (see ``parse_args``) and control the agent,
    environment, horizon, reproducibility settings and output
    behavior.
    """
    algo: str
    k: int
    T: int
    runs: int
    seed: int
    epsilon: float | None = None
    step_size: float | None = None
    c: float | None = None
    nonstationary: bool = False
    sigma: float = 0.1
    outdir: str = "plots"
    save_csv: bool = False


def make_agent(cfg: Config, rng: np.random.Generator) -> Agent:
    """Instantiate an agent according to config.

    Args:
        cfg: The experiment configuration (algorithm and hyperparameters).
        rng: Random generator to pass into the agent for reproducibility.

    Returns:
        A concrete ``Agent`` instance.

    Raises:
        ValueError: If ``cfg.algo`` is not recognized or required hyperparameters
        are missing (e.g., ``--epsilon`` for epsilon-greedy).
    """
    algo = cfg.algo.lower()
    if algo in {"epsilon", "eps", "epsilon-greedy", "egreedy"}:
        if cfg.epsilon is None:
            raise ValueError("--epsilon is required for epsilon-greedy")
        return EpsilonGreedyAgent(epsilon=cfg.epsilon, step_size=cfg.step_size, rng=rng)  # type: ignore[return-value]
    if algo == "ucb":
        c = cfg.c if cfg.c is not None else 2.0
        return UCBAgent(c=c, rng=rng)  # type: ignore[return-value]
    if algo in {"ts", "thompson", "thompson-sampling"}:
        return ThompsonSamplingAgent(rng=rng)  # type: ignore[return-value]
    raise ValueError(f"Unknown algo: {cfg.algo}")


def run_single(env: KArmedBanditEnv, agent: Agent, T: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one episode of length ``T``.

    Args:
        env: Bandit environment.
        agent: Agent implementing the bandit API.
        T: Horizon (number of steps).
        seed: Seed used to reset the environment for this run.

    Returns:
        Tuple of arrays ``(rewards, optimal, pstar_t)`` of length ``T``:
        - rewards: observed rewards per time step
        - optimal: 1 if the chosen action was optimal at the step, else 0
        - pstar_t: the instantaneous best arm probability (for regret)
    """
    # Reset environment and agent
    obs, _info = env.reset(seed=seed)
    k = int(getattr(env.action_space, "n", 0))  # type: ignore[attr-defined]
    agent.reset(k=k)

    rewards = np.zeros(T, dtype=float)
    optimal = np.zeros(T, dtype=int)
    pstar_t = np.zeros(T, dtype=float)

    for t in range(1, T + 1):
        a = agent.select_action(t)
        obs, r, term, trunc, info = env.step(a)
        agent.update(a, r)
        rewards[t - 1] = float(r)
        optimal[t - 1] = int(info.get("optimal", 0))
        p = info.get("p", None)
        if p is not None:
            pstar_t[t - 1] = float(np.max(p))
        else:
            pstar_t[t - 1] = np.nan
        # continuing bandit: ignore term/trunc as per spec

    return rewards, optimal, pstar_t


def aggregate_runs(
    all_rewards: List[np.ndarray],
    all_optimal: List[np.ndarray],
    all_pstar: List[np.ndarray],
    nonstationary: bool,
) -> Dict[str, np.ndarray]:
    """Aggregate per-time means and CIs across runs; returns metrics including cumulative regret.

    - Mean reward and % optimal at each time step
    - CIs (95%) for those metrics
    - Cumulative regret (stationary: t*p* - sum r; nonstationary: sum_t p*_t - sum r)
    """
    R = np.vstack(all_rewards)           # shape: (runs, T)
    O = np.vstack(all_optimal)           # shape: (runs, T)
    P = np.vstack(all_pstar)             # shape: (runs, T)

    runs, T = R.shape
    # Means
    mean_reward = R.mean(axis=0)
    pct_optimal = O.mean(axis=0) * 100.0

    # Regret per run
    cumulative_rewards = np.cumsum(R, axis=1)
    if nonstationary:
        # Dynamic oracle: cumulative sum of p*_t per run
        cumulative_pstar = np.cumsum(P, axis=1)
        cumulative_regret = cumulative_pstar - cumulative_rewards
    else:
        # Stationary: p* from first step for each run
        pstar0 = P[:, :1]  # shape (runs,1)
        t_vec = np.arange(1, T + 1, dtype=float)[None, :]  # shape (1,T)
        cumulative_regret = pstar0 * t_vec - cumulative_rewards

    mean_regret = cumulative_regret.mean(axis=0)

    # 95% CIs using normal approximation: mean ± 1.96 * SE
    se_reward = R.std(axis=0, ddof=1) / np.sqrt(runs)
    se_pct = O.std(axis=0, ddof=1) / np.sqrt(runs) * 100.0
    se_regret = cumulative_regret.std(axis=0, ddof=1) / np.sqrt(runs)

    ci = 1.96
    return {
        "mean_reward": mean_reward,
        "reward_lo": mean_reward - ci * se_reward,
        "reward_hi": mean_reward + ci * se_reward,
        "%_optimal": pct_optimal,
        "%_optimal_lo": pct_optimal - ci * se_pct,
        "%_optimal_hi": pct_optimal + ci * se_pct,
        "cumulative_regret": mean_regret,
        "cumulative_regret_lo": mean_regret - ci * se_regret,
        "cumulative_regret_hi": mean_regret + ci * se_regret,
    }


def save_plots(metrics: Dict[str, np.ndarray], cfg: Config) -> None:
    """Save time-series plots for mean reward, % optimal, and cumulative regret.

    Files are saved under ``cfg.outdir`` with filenames encoding the config.
    """


    os.makedirs(cfg.outdir, exist_ok=True)

    def tag() -> str:
        parts = [
            f"algo={cfg.algo}",
            f"k={cfg.k}", f"T={cfg.T}", f"runs={cfg.runs}", f"seed={cfg.seed}",
        ]
        if cfg.algo.lower() in {"epsilon", "eps", "epsilon-greedy", "egreedy"} and cfg.epsilon is not None:
            parts.append(f"epsilon={cfg.epsilon}")
        if cfg.algo.lower() == "ucb" and cfg.c is not None:
            parts.append(f"c={cfg.c}")
        if cfg.nonstationary:
            parts.append(f"nonstat_sigma={cfg.sigma}")
        return "_".join(parts)

    x = np.arange(1, metrics["mean_reward"].shape[0] + 1)

    # Mean reward with CI
    plt.figure(figsize=(6, 4))
    plt.plot(x, metrics["mean_reward"], label="mean reward")
    plt.fill_between(x, metrics["reward_lo"], metrics["reward_hi"], alpha=0.2, label="95% CI")
    plt.xlabel("t")
    plt.ylabel("Mean reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.outdir, f"reward_{tag()}.png"))
    plt.close()

    # % optimal with CI
    plt.figure(figsize=(6, 4))
    plt.plot(x, metrics["%_optimal"], label="% optimal")
    plt.fill_between(x, metrics["%_optimal_lo"], metrics["%_optimal_hi"], alpha=0.2, label="95% CI")
    plt.xlabel("t")
    plt.ylabel("% optimal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.outdir, f"pct_optimal_{tag()}.png"))
    plt.close()

    # Cumulative regret with CI
    plt.figure(figsize=(6, 4))
    plt.plot(x, metrics["cumulative_regret"], label="cumulative regret")
    plt.fill_between(x, metrics["cumulative_regret_lo"], metrics["cumulative_regret_hi"], alpha=0.2, label="95% CI")
    plt.xlabel("t")
    plt.ylabel("Cumulative regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.outdir, f"regret_{tag()}.png"))
    plt.close()


def parse_args() -> Config:
    """Parse CLI arguments into a ``Config`` dataclass."""
    p = argparse.ArgumentParser(description="MAB experiments skeleton")
    p.add_argument("--algo", type=str, required=True, help="epsilon|ucb|thompson")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    # Hyperparameters
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--step-size", type=float, default=None)
    p.add_argument("--c", type=float, default=None)
    # Environment
    p.add_argument("--nonstationary", action="store_true")
    p.add_argument("--sigma", type=float, default=0.1)
    # Output
    p.add_argument("--outdir", type=str, default="plots")
    p.add_argument("--save-csv", action="store_true")
    args = p.parse_args()
    return Config(
        algo=args.algo,
        k=args.k,
        T=args.T,
        runs=args.runs,
        seed=args.seed,
        epsilon=args.epsilon,
        step_size=args.step_size,
        c=args.c,
        nonstationary=args.nonstationary,
        sigma=args.sigma,
        outdir=args.outdir,
        save_csv=bool(args.save_csv),
    )


def main() -> None:
    """Entry point: run ``cfg.runs`` independent runs and save metrics/plots."""
    cfg = parse_args()
    rng = np.random.default_rng(cfg.seed)

    # Environment (students must implement KArmedBanditEnv in MAB/bandit_env.py)
    env = KArmedBanditEnv(k=cfg.k, nonstationary=cfg.nonstationary, sigma=cfg.sigma)  # type: ignore[call-arg]

    all_rewards: List[np.ndarray] = []
    all_optimal: List[np.ndarray] = []
    all_pstar: List[np.ndarray] = []

    for i in range(cfg.runs):
        agent_rng = np.random.default_rng(cfg.seed + i)
        agent = make_agent(cfg, rng=agent_rng)
        try:
            rewards, optimal, pstar_t = run_single(env, agent, cfg.T, seed=cfg.seed + i)
        except NotImplementedError as e:
            print(
                "Agent methods are not implemented yet. Complete TODOs in MAB/agents.py.",
                file=sys.stderr,
            )
            raise
        all_rewards.append(rewards)
        all_optimal.append(optimal)
        all_pstar.append(pstar_t)

    metrics = aggregate_runs(all_rewards, all_optimal, all_pstar, cfg.nonstationary)

    # Save plots
    try:
        save_plots(metrics, cfg)
    except Exception as e:
        print(f"Warning: plotting failed: {e}", file=sys.stderr)

    # Optionally save CSV of final metrics at t=T
    if cfg.save_csv:
        import csv, os
        os.makedirs(cfg.outdir, exist_ok=True)
        csv_path = os.path.join(cfg.outdir, "summary.csv")
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            # header if empty
            if f.tell() == 0:
                w.writerow(["algo","k","T","runs","seed","epsilon","step_size","c","nonstationary","sigma","final_mean_reward","final_%_optimal","final_cumulative_regret"])
            w.writerow([
                cfg.algo, cfg.k, cfg.T, cfg.runs, cfg.seed, cfg.epsilon, cfg.step_size, cfg.c, cfg.nonstationary, cfg.sigma,
                float(metrics["mean_reward"][-1]), float(metrics["%_optimal"][-1]), float(metrics["cumulative_regret"][-1])
            ])

    # Print simple summaries to stdout
    print(f"Mean reward (first 5 timesteps): {np.round(metrics['mean_reward'][:5], 3)}")
    print(f"% optimal (first 5 timesteps): {np.round(metrics['%_optimal'][:5], 1)}")


if __name__ == "__main__":
    main()
