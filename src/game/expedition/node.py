"""Expedition node definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ExpeditionNodeType(StrEnum):
    NARRATIVE = "narrative"
    CHECK = "check"
    ROAD = "road"
    WILDERNESS = "wilderness"
    CURIO = "curio"
    HAZARD = "hazard"
    LANDMARK = "landmark"
    CAVE_ENTRANCE = "cave_entrance"
    SHORTCUT = "shortcut"
    OBSTACLE = "obstacle"
    COMBAT = "combat"
    BOSS = "boss"
    BREACH = "breach"
    MAZE = "maze"
    TOWN = "town"


@dataclass(frozen=True)
class ExpeditionNode:
    id: str
    name: str
    node_type: ExpeditionNodeType
    text: str
