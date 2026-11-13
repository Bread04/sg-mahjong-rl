import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sgmj.env import tiles as T


def test_tile_counts() -> None:
    assert T.NUM_REGULAR_TILES == 136
    assert T.NUM_BONUS_TILES == 12


@pytest.mark.parametrize(
    ("tile", "name"),
    [
        (0, "1b"),
        (8, "9b"),
        (9, "1c"),
        (26, "9k"),
        (27, "E"),
        (30, "N"),
        (31, "R"),
        (34, "F1"),
        (42, "A1"),
        (45, "A4"),
    ],
)
def test_tile_names(tile: int, name: str) -> None:
    assert T.tile_name(tile) == name


def test_categories_partition_all_tiles() -> None:
    for t in range(46):
        assert T.is_suited(t) + T.is_honor(t) + T.is_bonus(t) == 1


def test_rank_rejects_honors() -> None:
    with pytest.raises(ValueError):
        T.rank_of(27)


def test_too_many_copies_rejected() -> None:
    with pytest.raises(ValueError):
        T.tiles_to_counts([5] * 5)


# Property-based test: the round trip holds for ANY legal hand, not just the ones we thought of.
legal_hands = st.lists(st.integers(0, 33), max_size=18).filter(
    lambda ts: all(ts.count(t) <= 4 for t in set(ts))
)


@given(legal_hands)
def test_counts_round_trip(tiles: list[int]) -> None:
    counts = T.tiles_to_counts(tiles)
    assert counts.sum() == len(tiles)
    assert T.counts_to_tiles(counts) == sorted(tiles)
    assert np.array_equal(T.tiles_to_counts(T.counts_to_tiles(counts)), counts)
