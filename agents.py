from __future__ import annotations

"""Student skeleton for Assignment 1: Multi‑Armed Bandits.

Implement the TODOs in each agent. The public API must remain:
    - reset(k): initialize internal state for k arms
    - select_action(t): choose action index [0, k-1] at time step t (t starts at 1)
    - update(a, r): update internal state given action a and reward r

Conventions
- Time step t is 1-based in select_action(t).
- Tie-breaking among maximizing actions must be uniform at random (use random_argmax).
- Rewards are Bernoulli (0/1) in the baseline Bernoulli testbed.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


def random_argmax(x: np.ndarray, rng: np.random.Generator) -> int:
    """Return a uniformly tie‑broken argmax of a 1‑D array.

    Args:
        x: 1‑D array of scores.
        rng: NumPy RNG used for uniform tie‑breaking among argmax indices.

    Returns:
        An integer index corresponding to one of the maxima of x, chosen uniformly.
    """
    x = np.asarray(x)
    max_val = np.max(x)
    candidates = np.flatnonzero(x == max_val)
    return int(rng.choice(candidates))


class Agent(ABC):
    """Abstract Bandit Agent interface.

    Required API:
    - ``reset(k)``: initialize internal state for ``k`` arms.
    - ``select_action(t)``: choose action index in ``[0, k-1]`` at time step ``t`` (1-based).
    - ``update(a, r)``: update internal state with action ``a`` and reward ``r``.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        """Initialize the base agent.

        Args:
            rng: Optional numpy random generator; if ``None``, a default_rng() is used.
        """
        self.rng: np.random.Generator = rng or np.random.default_rng()
        self.k: Optional[int] = None

    @abstractmethod
    def reset(self, k: int) -> None:
        """Prepare internal state for ``k`` actions (arms)."""
        raise NotImplementedError

    @abstractmethod
    def select_action(self, t: int) -> int:
        """Return an action index at time step ``t`` (1-based)."""
        raise NotImplementedError

    @abstractmethod
    def update(self, a: int, r: float) -> None:
        """Update internal estimates based on observed reward ``r`` from action ``a``."""
        raise NotImplementedError


class EpsilonGreedyAgent(Agent):
    """Epsilon‑greedy policy with sample‑average or constant step‑size updates.

    With probability epsilon, selects a uniform random action; otherwise selects
    a greedy action according to the current value estimates Q. Values can be updated
    by sample average (1/N[a]) or by a constant step‑size.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        step_size: Optional[float] = None,
        use_sample_average: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Construct an epsilon‑greedy agent.

        Args:
            epsilon: Exploration probability in [0,1].
            step_size: Optional constant step‑size alpha in (0,1]. If None and
                use_sample_average is True, use sample‑average updates.
            use_sample_average: If True and step_size is None, update with 1/N[a].
            rng: Optional NumPy RNG for reproducibility.
        """
        super().__init__(rng=rng)
        if not (0.0 <= float(epsilon) <= 1.0):
            raise ValueError("epsilon must be in [0,1]")
        if step_size is not None and not (0.0 < float(step_size) <= 1.0):
            raise ValueError("step_size must be in (0,1] when provided")
        self.epsilon = float(epsilon)
        self.step_size = step_size
        self.use_sample_average = bool(use_sample_average)
        # Internal state
        self.Q: Optional[np.ndarray] = None  # action-value estimates
        self.N: Optional[np.ndarray] = None  # visit counts

    def reset(self, k: int) -> None:
        """Initialize value estimates and counts for ``k`` arms.

        Args:
            k: Number of arms.
        """
        self.k = int(k)
        self.Q = np.zeros(self.k, dtype=float)
        self.N = np.zeros(self.k, dtype=int)

    def select_action(self, t: int) -> int:
        """Select an action using the epsilon‑greedy rule.

        Args:
            t: 1‑based time step (unused for this agent).

        Returns:
            An integer action index in [0, k‑1].

        TODO:
        - With probability epsilon: return a uniform‑random action in [0, k‑1].
        - Else: return an argmax of Q with uniform tie‑breaking (use random_argmax).
        """
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.k)
    
        # Otherwise: return greedy action with uniform tie-breaking
        return random_argmax(self.Q, self.rng)

    def update(self, a: int, r: float) -> None:
        """Update value estimate for the chosen action.

        Args:
            a: Index of the selected action.
            r: Observed reward in {0,1} for the Bernoulli testbed.

        TODO:
        - Increment N[a].
        - If use_sample_average and step_size is None: Q[a] += (1/N[a]) * (r - Q[a]).
        - Else: Q[a] += step_size * (r - Q[a]).
        """
        # Increment the visit count for this action
        self.N[a] += 1
    
    # Update Q-value based on the update rule
        if self.use_sample_average and self.step_size is None:
        # Sample-average update: Q[a] += (1/N[a]) * (r - Q[a])
            self.Q[a] += (1.0 / self.N[a]) * (r - self.Q[a])
        else:
        # Constant step-size update: Q[a] += step_size * (r - Q[a])
            self.Q[a] += self.step_size * (r - self.Q[a])


class UCBAgent(Agent):
    """Upper Confidence Bound (UCB1) agent.

    Uses the UCB1 score Q[a] + c * sqrt(ln t / N[a]) after each arm has been
    pulled once; prior to that, selects among unpulled arms to ensure coverage.
    """

    def __init__(self, c: float = 2.0, rng: Optional[np.random.Generator] = None) -> None:
        """Construct a UCB agent.

        Args:
            c: Confidence parameter controlling the exploration bonus (``>= 0``).
            rng: Optional numpy random generator.
        """
        super().__init__(rng=rng)
        if float(c) < 0.0:
            raise ValueError("c must be non-negative")
        self.c = float(c)
        self.Q: Optional[np.ndarray] = None
        self.N: Optional[np.ndarray] = None

    def reset(self, k: int) -> None:
        """Initialize value estimates and counts for ``k`` arms."""
        self.k = int(k)
        self.Q = np.zeros(self.k, dtype=float)
        self.N = np.zeros(self.k, dtype=int)

    def select_action(self, t: int) -> int:
       
    # First check if any arm has never been pulled (N[a] == 0)
        unpulled = np.where(self.N == 0)[0]
        if len(unpulled) > 0:
            # Select the first unpulled arm (or could randomize among them)
            return unpulled[0]
        
        # All arms have been pulled at least once - compute UCB scores
        # UCB formula: Q[a] + c * sqrt(ln(t) / N[a])
        ucb_scores = self.Q + self.c * np.sqrt(np.log(t) / self.N)
        
        # Return action with highest UCB score (with tie-breaking)
        return random_argmax(ucb_scores, self.rng)

    def update(self, a: int, r: float) -> None:
        """Update sample-average value estimate for the chosen action.

        Args:
            a: Action index taken.
            r: Observed reward.

        TODO:
        - Increment ``N[a]``.
        - Update ``Q[a] += (1/N[a]) * (r - Q[a])``.
        """
        # Increment the visit count for this action
        self.N[a] += 1
    
        # Update Q-value using sample-average rule: Q[a] += (1/N[a]) * (r - Q[a])
        self.Q[a] += (1.0 / self.N[a]) * (r - self.Q[a])


class ThompsonSamplingAgent(Agent):
    """Optional (bonus): Thompson Sampling for Bernoulli bandits.

    Maintains independent Beta posteriors per arm; samples thetas each round
    and selects a tie‑broken argmax.
    """

    def __init__(
        self,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Construct a Thompson Sampling agent.

        Args:
            alpha0: Prior ``alpha`` hyperparameter (> 0).
            beta0: Prior ``beta`` hyperparameter (> 0).
            rng: Optional numpy random generator.
        """
        super().__init__(rng=rng)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.alpha: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None

    def reset(self, k: int) -> None:
        """Initialize Beta prior parameters for ``k`` arms."""
        self.k = int(k)
        self.alpha = np.full(self.k, self.alpha0, dtype=float)
        self.beta = np.full(self.k, self.beta0, dtype=float)

    def select_action(self, t: int) -> int:
        """Select an action by sampling from Beta posteriors.

        Args:
            t: Time step (unused; included for interface parity).

        Returns:
            An integer action index in ``[0, k-1]``.

        TODO:
        - For each arm ``a``, sample ``theta[a] ~ Beta(alpha[a], beta[a])`` and return a tie-broken argmax.
        """
        # Sample theta from Beta distribution for each arm
        # Beta(alpha[a], beta[a]) represents our belief about arm a's success probability
        sampled_thetas = self.rng.beta(self.alpha, self.beta)
        
        # Select the arm with the highest sampled theta value (with tie-breaking)
        return random_argmax(sampled_thetas, self.rng)

    def update(self, a: int, r: float) -> None:
        """Update Beta posterior with an observed Bernoulli reward.

        Args:
            a: Action index taken.
            r: Observed reward in ``{0,1}``.

        TODO:
        - ``alpha[a] += r`` and ``beta[a] += (1 - r)``.
        """
        if r == 1:
        # Success: increment alpha (number of successes)
            self.alpha[a] += 1
        else:
        # Failure: increment beta (number of failures)
            self.beta[a] += 1
