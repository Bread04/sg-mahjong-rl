# Merjong-singapore-rl-learning-


A reinforcement-learning agent for **Singapore Mahjong**, built from first principles:
a fast multi-agent environment, a self-play PPO agent, and a statistically rigorous evaluation harness.

> Status: **Phase 0: RL fundamentals.** See [`docs/roadmap.md`](docs/roadmap.md).

## Why this is hard

- **4 players, imperfect information**: you can't see 3 hands or the wall.
- **Huge, sparse action space**: discard, chow, pong, kong and win, most of which are illegal at any moment.
- **High variance**: luck of the deal dominates single hands, so naive evaluation lies.

## Layout

```
src/sgmj/
  env/       rules engine + PettingZoo environment      (Phase 1)
  agents/    random / greedy / rule-based baselines      (Phase 2)
  eval/      duplicate-play evaluation harness           (Phase 2)
  rl/        model, behaviour cloning, PPO, league       (Phases 3-4)
exercises/   Phase 0 RL-from-scratch exercises (CartPole -> LunarLander)
tests/       pytest + hypothesis property tests
docs/        roadmap, decisions, lab journal
```

## Setup

```bash
uv sync                 # core + dev tools (fast, no torch)
uv sync --extra rl      # + CUDA torch, gymnasium, pettingzoo, tensorboard
uv run pre-commit install
uv run pytest
```

Check the GPU is visible:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Results

_Nothing yet, by design. No result goes here without a confidence interval from `sgmj.eval`._
