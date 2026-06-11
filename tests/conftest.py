"""Shared pytest fixtures and opening-route constants for tests."""

from __future__ import annotations

import pytest

from game.data.loaders import load_game_definitions

OPENING_DUNGEON_TO_WORKS_CACHE = (
    "old_road",
    "hunters_trail",
    "dry_creek_bed",
    "black_stone_sinkhole",
    "shallow_cave_entrance",
    "shallow_cave_room_1",
    "cave_fork",
    "fungus_chamber",
)

OPENING_POST_COMBAT = (
    "action:recover_gate_key",
    "fungus_chamber",
    "stone_gate",
    "action:unlock_black_gate",
    "maze_touched_lair",
)

_DEFINITIONS = None


def get_definitions():
    global _DEFINITIONS
    if _DEFINITIONS is None:
        _DEFINITIONS = load_game_definitions()
    return _DEFINITIONS


@pytest.fixture(scope="session")
def definitions():
    return get_definitions()
