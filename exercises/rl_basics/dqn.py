"""Exercise 3: DQN on CartPole-v1.

Run:  uv run python -m exercises.rl_basics.dqn

Steps:
  1. Q-network: obs -> Q-value per action.
  2. Replay buffer (fixed-size ring buffer of (s, a, r, s', done)).
  3. Epsilon-greedy exploration with a linear decay schedule.
  4. TD target: r + gamma * max_a' Q_target(s', a') * (1 - done)
  5. Target network, synced every N steps.
  6. Ablate: no target net; no replay buffer. Plot all three.

Journal questions:
  - Why does DQN need a target network when policy-gradient methods don't?
  - What is the "deadly triad"? Which two parts of it does this setup contain?
  - Why will DQN be the WRONG choice for mahjong? (hint: 4 players, hidden info, non-stationarity)
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Exercise 3: implement DQN")


if __name__ == "__main__":
    main()
