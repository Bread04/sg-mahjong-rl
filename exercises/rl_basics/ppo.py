"""Exercise 4: PPO-clip on LunarLander-v3. This is the algorithm you'll use for mahjong.

Run:  uv run python -m exercises.rl_basics.ppo

Steps:
  1. gymnasium.vector.SyncVectorEnv with 8 envs.
  2. Shared-trunk actor-critic (orthogonal init, small final-layer gain on the policy head).
  3. Rollout of num_steps per env -> buffer of obs, actions, logprobs, rewards, dones, values.
  4. Advantages via your `gae`; returns = advantages + values.
  5. For K epochs, over shuffled minibatches:
       ratio = exp(new_logprob - old_logprob)
       policy_loss = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A).mean()
       + value loss, - entropy bonus; clip grad norm.
  6. Log approx_kl, clip_fraction, explained_variance, entropy. These are your debugging dashboard.

Then add ACTION MASKING (needed for mahjong): set illegal-action logits to -inf
(use a large negative finite number) BEFORE building the distribution.

Journal questions:
  - What does clip_fraction tell you? What value is healthy?
  - What does explained_variance near 0 mean about your critic?
  - Why must the mask also be applied when recomputing logprobs during the update?
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Exercise 4: implement PPO")


if __name__ == "__main__":
    main()
