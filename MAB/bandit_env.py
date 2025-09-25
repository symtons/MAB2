from __future__ import annotations

"""Skeleton: Minimal Gymnasium environment for a k-armed bandit (student TODO).

Implement a minimal Gymnasium-compatible bandit environment that standardizes
evaluation for the assignment. The design targets are:

- Action space: ``gymnasium.spaces.Discrete(k)``
- Observation space: return a constant dummy observation (e.g., ``Discrete(1)`` with value ``0``)
- Rewards (stationary baseline): Bernoulli per arm with fixed probabilities ``p_a`` in ``[0,1]``,
  sampled at ``reset`` via ``Uniform(0,1)`` for each arm
- Rewards (non-stationary, bonus): optional Gaussian random walk on probabilities
  with std ``sigma`` each step, clipped to ``[0,1]``
- Continuing task: ``terminated=False`` and ``truncated=False`` at every step
- Info dict from ``step`` should include:
  - ``"p"``: current per-arm probabilities (vector of shape ``[k]``)
  - ``"optimal"``: ``1`` if the chosen action was optimal under current ``p``, else ``0``

Notes
- Support ``seed`` in ``reset`` for reproducibility (reseed the RNG for that run).
- Keep the environment minimal; no observations beyond the constant placeholder are required.
"""

from typing import Optional, Tuple, Dict, Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - import guard for skeleton
    gym = None  # type: ignore
    spaces = None  # type: ignore


BaseEnv = gym.Env if gym is not None else object  # type: ignore


class KArmedBanditEnv(BaseEnv):
    """Student TODO: implement a minimal k-armed Bernoulli bandit environment.

    Requirements (must match assignment and runner expectations):
    - Action space: ``spaces.Discrete(k)``
    - Observation space: constant ``0`` (e.g., ``spaces.Discrete(1)``)
    - Stationary case: at ``reset``, sample probabilities ``p ~ Uniform(0,1)^k``
    - Non-stationary (bonus): on each ``step``, apply Gaussian noise with std ``sigma`` to ``p`` and clip to ``[0,1]``
    - ``step`` returns ``(obs=0, reward in {0,1}, terminated=False, truncated=False, info)``
    - ``info`` must contain ``{"p": p.copy(), "optimal": 0/1}`` so the runner can compute metrics
    - Support ``seed`` in ``reset`` to reseed the RNG for the run
    """

    metadata = {"render_modes": []}

    def __init__(self, k: int = 10, nonstationary: bool = False, sigma: float = 0.1,
                 rng: Optional[np.random.Generator] = None) -> None:
        if spaces is None:
            raise ImportError(
                "Gymnasium is required for the environment. Install via `pip install gymnasium`."
            )
        # Store configuration
        self.k = int(k)
        self.nonstationary = bool(nonstationary)
        self.sigma = float(sigma)
        self._rng: np.random.Generator = rng or np.random.default_rng()

        # TODO (student): define action and observation spaces
        self.action_space = spaces.Discrete(self.k)
        self.observation_space = spaces.Discrete(1)

        
        self.p: Optional[np.ndarray] = None

    

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        """Reset the environment and (re)sample per-arm probabilities ``p``.

        Returns a dummy observation ``0`` and an (optionally empty) ``info`` dict.

        Student TODO:
        - If ``seed`` is provided, reseed ``self._rng`` for reproducibility.
        - Sample ``self.p ~ Uniform(0,1)^k`` (NumPy: ``self._rng.uniform``) and store it.
        - Return ``(0, {})``.
        """
    
        if seed is not None:
            self._rng = np.random.default_rng(seed)
    
    
        self.p = self._rng.uniform(0.0, 1.0, size=self.k)
    
   
        return 0, {}
    
    


    def step(self, action: int) -> Tuple[int, int, bool, bool, Dict[str, Any]]:
        """Advance one step by sampling a Bernoulli reward for ``action``.

         Student TODO:
        - If ``self.nonstationary``: apply Gaussian noise (std ``self.sigma``) to ``self.p`` and clip to ``[0,1]``.
        - Sample reward ``r in {0,1}`` as ``Bernoulli(p[action])`` using ``self._rng.random()``.
        - Compute ``optimal = 1`` if ``action`` equals ``argmax(self.p)``, else ``0``.
        - Return ``(0, r, False, False, {"p": self.p.copy(), "optimal": optimal})``.
        """
    
        if self.nonstationary:
           
            noise = self._rng.normal(0.0, self.sigma, size=self.k)
            self.p = np.clip(self.p + noise, 0.0, 1.0)
        
     
        reward = 1 if self._rng.random() < self.p[action] else 0
        
        # highest probability
        optimal_action = np.argmax(self.p)
        optimal = 1 if action == optimal_action else 0
        
        
        info = {
            "p": self.p.copy(),  
            "optimal": optimal   
        }
        
        
        return 0, reward, False, False, info
    def render(self) -> None: 
        return None
