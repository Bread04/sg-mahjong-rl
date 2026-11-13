"""Tile encoding shared by every part of the project.

Regular tiles are indices 0-33 (4 copies each, 136 total):
    0-8   bamboo 1-9
    9-17  circles 1-9
    18-26 characters 1-9
    27-30 winds  E S W N
    31-33 dragons R G W
Bonus tiles are 34-45 (1 copy each):
    34-41 flowers / seasons (seat-owned)
    42-45 animals (cat, mouse, rooster, centipede)

A hand is a length-34 count vector: hand[t] = number of copies of tile t held.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

NUM_TILE_TYPES = 34
COPIES_PER_TILE = 4
NUM_REGULAR_TILES = NUM_TILE_TYPES * COPIES_PER_TILE  # 136

FLOWERS = range(34, 42)
ANIMALS = range(42, 46)
NUM_BONUS_TILES = len(FLOWERS) + len(ANIMALS)  # 12

SUITS = ("bamboo", "circles", "characters")
WINDS = ("E", "S", "W", "N")
DRAGONS = ("R", "G", "Wh")

HandCounts = npt.NDArray[np.int8]


def is_suited(tile: int) -> bool:
    return 0 <= tile < 27


def is_honor(tile: int) -> bool:
    return 27 <= tile < NUM_TILE_TYPES


def is_bonus(tile: int) -> bool:
    return NUM_TILE_TYPES <= tile < NUM_TILE_TYPES + NUM_BONUS_TILES


def suit_of(tile: int) -> int:
    """Suit index 0-2 for a suited tile."""
    if not is_suited(tile):
        raise ValueError(f"tile {tile} is not suited")
    return tile // 9


def rank_of(tile: int) -> int:
    """Rank 1-9 for a suited tile."""
    if not is_suited(tile):
        raise ValueError(f"tile {tile} is not suited")
    return tile % 9 + 1


def tile_name(tile: int) -> str:
    """Short human-readable name, e.g. '3b', '7c', '1k', 'E', 'R', 'F1', 'A2'."""
    if is_suited(tile):
        return f"{rank_of(tile)}{'bck'[suit_of(tile)]}"
    if 27 <= tile <= 30:
        return WINDS[tile - 27]
    if 31 <= tile <= 33:
        return DRAGONS[tile - 31]
    if tile in FLOWERS:
        return f"F{tile - 33}"
    if tile in ANIMALS:
        return f"A{tile - 41}"
    raise ValueError(f"invalid tile index {tile}")


def tiles_to_counts(tiles: list[int]) -> HandCounts:
    """Convert a list of regular tile indices into a length-34 count vector."""
    counts = np.zeros(NUM_TILE_TYPES, dtype=np.int8)
    for t in tiles:
        if not 0 <= t < NUM_TILE_TYPES:
            raise ValueError(f"not a regular tile: {t}")
        counts[t] += 1
    if counts.max(initial=0) > COPIES_PER_TILE:
        raise ValueError("more than 4 copies of a tile")
    return counts


def counts_to_tiles(counts: HandCounts) -> list[int]:
    """Inverse of tiles_to_counts, sorted ascending."""
    return [t for t in range(NUM_TILE_TYPES) for _ in range(int(counts[t]))]
