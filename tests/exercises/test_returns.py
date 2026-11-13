"""Tests for Exercise 1. Each test SKIPS until you implement the function, then must pass."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from exercises.rl_basics import returns as R

pytestmark = pytest.mark.exercise


def call_or_skip(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


def test_discounted_returns_single_episode() -> None:
    r = np.array([1.0, 1.0, 1.0])
    d = np.array([0.0, 0.0, 1.0])
    out = call_or_skip(R.discounted_returns, r, d, 0.5)
    np.testing.assert_allclose(out, [1.75, 1.5, 1.0])


def test_discounted_returns_resets_at_episode_boundary() -> None:
    r = np.array([1.0, 2.0, 3.0, 4.0])
    d = np.array([0.0, 1.0, 0.0, 1.0])
    out = call_or_skip(R.discounted_returns, r, d, 0.9)
    np.testing.assert_allclose(out, [1.0 + 0.9 * 2.0, 2.0, 3.0 + 0.9 * 4.0, 4.0])


def test_gae_lambda_zero_is_td_error() -> None:
    r = np.array([1.0, 0.0, 2.0])
    v = np.array([0.5, 0.2, 0.1])
    d = np.zeros(3)
    adv, ret = call_or_skip(R.gae, r, v, d, last_value=0.3, gamma=0.9, lam=0.0)
    next_v = np.array([0.2, 0.1, 0.3])
    np.testing.assert_allclose(adv, r + 0.9 * next_v - v)
    np.testing.assert_allclose(ret, adv + v)


def test_gae_lambda_one_matches_monte_carlo() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(size=10)
    v = rng.normal(size=10)
    d = np.zeros(10)
    d[4] = d[9] = 1.0
    adv, _ = call_or_skip(R.gae, r, v, d, last_value=0.0, gamma=0.99, lam=1.0)
    mc = call_or_skip(R.discounted_returns, r, d, 0.99)
    np.testing.assert_allclose(adv, mc - v, atol=1e-10)


def test_gae_does_not_bootstrap_across_done() -> None:
    r = np.array([0.0, 0.0])
    v = np.array([0.0, 0.0])
    d = np.array([1.0, 0.0])
    adv, _ = call_or_skip(R.gae, r, v, d, last_value=100.0, gamma=1.0, lam=1.0)
    assert adv[0] == 0.0  # the huge value after step 1 must not leak into episode 1
    assert adv[1] == 100.0


def test_normalize() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    out = call_or_skip(R.normalize, x)
    assert abs(out.mean()) < 1e-9
    assert abs(out.std() - 1.0) < 1e-6
