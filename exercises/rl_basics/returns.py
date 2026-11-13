"""Exercise 1: the return/advantage maths every policy-gradient method is built on.

Pure numpy, no torch. Tests live in tests/exercises/test_returns.py and are skipped
until you replace each `raise NotImplementedError`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def discounted_returns(rewards: Array, dones: Array, gamma: float) -> Array:
    """Monte Carlo return G_t = r_t + gamma * G_{t+1}, resetting to 0 after a terminal step.

    rewards, dones: shape (T,). dones[t] == 1 means the episode ended AFTER step t.
    Returns shape (T,). The final step bootstraps from 0 (assume the rollout ends at a boundary).
    """
    raise NotImplementedError


def gae(
    rewards: Array,
    values: Array,
    dones: Array,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[Array, Array]:
    """Generalised Advantage Estimation.

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}

    values[t] = V(s_t); last_value = V(s_T) for the state after the rollout.
    Returns (advantages, returns) where returns = advantages + values (the value-function target).

    Sanity checks to reason through before coding:
      lam = 1 -> advantages equal discounted_returns(...) - values (when last_value = 0)
      lam = 0 -> advantages equal the one-step TD error delta_t
    """
    raise NotImplementedError


def normalize(x: Array, eps: float = 1e-8) -> Array:
    """Zero mean, unit std. Used on advantages per minibatch in PPO."""
    raise NotImplementedError
