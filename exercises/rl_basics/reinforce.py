"""Exercise 2: REINFORCE with a learned value baseline on CartPole-v1.

Run:  uv run python -m exercises.rl_basics.reinforce
Needs the rl extra: uv sync --extra rl

Steps to implement (in order, test each before moving on):
  1. Policy net: obs -> logits over actions. Value net: obs -> scalar.
  2. Collect ONE full episode by sampling from Categorical(logits=...).
  3. Compute returns with your `discounted_returns` from returns.py.
  4. advantage = returns - value.detach()
  5. policy_loss = -(log_prob * advantage).mean();  value_loss = MSE(value, returns)
  6. Log episode return to TensorBoard (torch.utils.tensorboard.SummaryWriter("runs/reinforce")).
  7. Run 3 seeds. Then remove the baseline and compare: what happens to variance?

Questions to answer in docs/journal.md:
  - Why does subtracting a baseline not bias the gradient?
  - Why .detach() the value in the advantage?
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Exercise 2: implement REINFORCE")


if __name__ == "__main__":
    main()
