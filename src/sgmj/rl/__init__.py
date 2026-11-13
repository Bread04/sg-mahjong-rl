"""The learning stack (Phases 3-4). Write this yourself.

Planned modules:
    model.py     1D ResNet over the 34-tile axis, policy + value heads, masked logits
    bc.py        behaviour cloning from the rule-based agent (warm start)
    rollout.py   CPU actor processes -> trajectory buffer
    ppo.py       GAE + clipped PPO loss + entropy bonus
    league.py    frozen-snapshot opponent pool
    oracle.py    oracle guiding: hidden info visible early, annealed to zero
    train.py     Hydra entry point, TensorBoard logging, checkpointing
"""
