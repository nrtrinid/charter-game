"""Small deterministic recruitment helpers."""

from __future__ import annotations

from dataclasses import dataclass

from game.content.definitions import GameDefinitions
from game.core.rng import GameRng


@dataclass(frozen=True)
class RecruitChoice:
    name: str
    class_id: str
    background: str = ""
    motive: str = ""


def generate_recruit_choices(
    definitions: GameDefinitions,
    rng: GameRng,
    count: int = 2,
) -> list[RecruitChoice]:
    pool = definitions.recruits.recruitment_pool
    if not pool:
        return []
    choices: list[RecruitChoice] = []
    available = list(pool)
    for _ in range(count):
        if not available:
            available = list(pool)
        entry_index = rng.randint(0, len(available) - 1)
        entry = available.pop(entry_index)
        choices.append(
            RecruitChoice(
                name=entry.name,
                class_id=entry.class_id,
                background=entry.background,
                motive=entry.motive,
            )
        )
    return choices
