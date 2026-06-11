"""Single randomness gateway for deterministic game systems."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")


@dataclass
class GameRng:
    """Seedable wrapper around Python randomness.

    All game systems should receive this object instead of importing ``random``.
    """

    seed: int | None = None
    _random: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)

    def randint(self, low: int, high: int) -> int:
        return self._random.randint(low, high)

    def chance(self, percent: int) -> bool:
        clamped = max(0, min(100, percent))
        return self.randint(1, 100) <= clamped

    def choice(self, values: Sequence[T]) -> T:
        if not values:
            raise ValueError("Cannot choose from an empty sequence.")
        return values[self._random.randrange(len(values))]
