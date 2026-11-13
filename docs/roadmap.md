# Roadmap

Each phase has an exit criterion. Don't start the next phase until it's met.

| Phase | Goal | Exit criterion | Est. |
|---|---|---|---|
| 0 | RL fundamentals: REINFORCE, DQN, PPO from scratch | All `exercises/` targets hit, 3 seeds each; PPO diffed against CleanRL in the journal | 2–3 wk |
| 1 | Rules engine + PettingZoo AEC env with action masks | Differential test vs. old engine agrees on 10k seeded games; property tests pass; throughput ≥ 100k hands/hr on 12 threads | 3–4 wk |
| 2 | Baselines + duplicate-play evaluation harness | Greedy beats random with a 95% CI that excludes zero | 1 wk |
| 3 | Behaviour cloning from the rule bot | BC agent within CI of the rule bot | 1–2 wk |
| 4 | PPO self-play, league, oracle guiding | Beats the rule bot significantly; ablation table for ≥ 3 design choices | 4–8 wk |
| 5 | ONNX export, browser demo, write-up | Public demo link + blog post with learning curves and eval table | 2 wk |

## Lessons carried over from v1 (the old repo)

These are the failures that motivated the rewrite. Each one needs a test or check that prevents it coming back.

1. **The observation must include the table.** v1's inference features had zero dims for
   discards, melds or bonus tiles, so the agent was playing blind.
2. **No train/inference mismatch.** v1 trained on 100% oracle features and inferred on 0%.
   Oracle guiding must anneal, and the deployed network must never see oracle inputs.
3. **Learn every decision.** v1 learned discards only; claims were hardcoded rules.
4. **Measure placement over full sessions, not single hands.**
5. **Measure throughput correctly.** v1 printed `*1000` instead of `*3600`.
6. **Docs describe what the code does, verified by running it.**

## References

- Li et al. 2020, *Suphx: Mastering Mahjong with Deep Reinforcement Learning*
- Mortal (open-source Riichi Mahjong AI): architecture and training notes
- Vinyals et al. 2019, *AlphaStar*: league training
- Lanctot et al., *OpenSpiel*: imperfect-information game baselines
