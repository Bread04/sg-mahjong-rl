# Phase 0: RL fundamentals (write every line yourself)

Mahjong is a bad first RL problem because bugs are silent: the agent just learns slowly.
Do these first, on environments where you know what "working" looks like.

| # | File | What you build | Done when |
|---|------|----------------|-----------|
| 1 | `rl_basics/returns.py` | discounted returns, GAE, advantage normalisation | `uv run pytest tests/exercises` passes (no skips) |
| 2 | `rl_basics/reinforce.py` | REINFORCE with a value baseline | CartPole-v1 mean return ≥ 475 |
| 3 | `rl_basics/dqn.py` | DQN with replay buffer and target network | CartPole-v1 mean return ≥ 475 |
| 4 | `rl_basics/ppo.py` | PPO-clip with GAE, minibatches, vectorised envs | LunarLander-v3 mean return ≥ 200 |
| 5 | (your choice) | PPO with **action masking** on a PettingZoo game | beats a random opponent > 90% |

Rules:
- Don't copy from CleanRL. Write it, get it working, and only **then** diff yours against
  CleanRL's single-file version and note every difference in `docs/journal.md`.
- Use AI to explain concepts and review your code, not to write the training loop.
- Log every run to TensorBoard (`runs/`). Plot return vs. steps for 3 seeds, not 1.

Reading, in order:
1. Sutton & Barto, *Reinforcement Learning: An Introduction*, ch. 1–6 and 13
2. OpenAI Spinning Up: "Key Concepts" → "Intro to Policy Optimization"
3. Schulman et al. 2016, *High-Dimensional Continuous Control Using GAE*
4. Schulman et al. 2017, *Proximal Policy Optimization Algorithms*
5. Huang et al. 2022, *The 37 Implementation Details of PPO* (ICLR blog track)
