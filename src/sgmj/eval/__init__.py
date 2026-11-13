"""Evaluation harness (Phase 2). Build this BEFORE any training.

- Duplicate play: each wall seed is replayed 4x with seats rotated, cancelling deal luck.
- Metrics: placement distribution, mean points/hand, win rate, deal-in rate.
- Always report confidence intervals; never claim "better" from a point estimate.
"""
