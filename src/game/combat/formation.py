"""2x2 combat formation and protection rules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class ProtectingCombatant(Protocol):
    def can_protect(self) -> bool: ...


class FormationSlot(StrEnum):
    BACK_LEFT = "BACK_LEFT"
    BACK_RIGHT = "BACK_RIGHT"
    FRONT_LEFT = "FRONT_LEFT"
    FRONT_RIGHT = "FRONT_RIGHT"


_COORDS: dict[FormationSlot, tuple[int, int]] = {
    FormationSlot.BACK_LEFT: (0, 0),
    FormationSlot.BACK_RIGHT: (0, 1),
    FormationSlot.FRONT_LEFT: (1, 0),
    FormationSlot.FRONT_RIGHT: (1, 1),
}


def is_front(slot: FormationSlot) -> bool:
    return slot in {FormationSlot.FRONT_LEFT, FormationSlot.FRONT_RIGHT}


def is_back(slot: FormationSlot) -> bool:
    return slot in {FormationSlot.BACK_LEFT, FormationSlot.BACK_RIGHT}


def lane_of(slot: FormationSlot) -> int:
    return _COORDS[slot][1]


def front_slot_for(slot: FormationSlot) -> FormationSlot:
    if slot == FormationSlot.BACK_LEFT:
        return FormationSlot.FRONT_LEFT
    if slot == FormationSlot.BACK_RIGHT:
        return FormationSlot.FRONT_RIGHT
    return slot


def back_slot_for(slot: FormationSlot) -> FormationSlot:
    if slot == FormationSlot.FRONT_LEFT:
        return FormationSlot.BACK_LEFT
    if slot == FormationSlot.FRONT_RIGHT:
        return FormationSlot.BACK_RIGHT
    return slot


def are_adjacent(first: FormationSlot, second: FormationSlot) -> bool:
    first_row, first_col = _COORDS[first]
    second_row, second_col = _COORDS[second]
    return abs(first_row - second_row) + abs(first_col - second_col) == 1


@dataclass
class Formation:
    slots: dict[FormationSlot, str | None] = field(
        default_factory=lambda: dict.fromkeys(FormationSlot, None)
    )

    @classmethod
    def empty(cls) -> Formation:
        return cls()

    @classmethod
    def from_mapping(cls, slots: Mapping[FormationSlot, str | None]) -> Formation:
        formation = cls.empty()
        for slot, actor_id in slots.items():
            formation.slots[slot] = actor_id
        return formation

    def actor_at(self, slot: FormationSlot) -> str | None:
        return self.slots.get(slot)

    def slot_of(self, actor_id: str) -> FormationSlot | None:
        for slot, occupant in self.slots.items():
            if occupant == actor_id:
                return slot
        return None

    def place(self, actor_id: str, slot: FormationSlot) -> None:
        old_slot = self.slot_of(actor_id)
        if old_slot is not None:
            self.slots[old_slot] = None
        self.slots[slot] = actor_id

    def remove(self, actor_id: str) -> None:
        slot = self.slot_of(actor_id)
        if slot is not None:
            self.slots[slot] = None

    def swap_slots(
        self,
        first: FormationSlot,
        second: FormationSlot,
        *,
        require_adjacent: bool = True,
    ) -> bool:
        if require_adjacent and not are_adjacent(first, second):
            return False
        self.slots[first], self.slots[second] = self.slots[second], self.slots[first]
        return True

    def protector_for(
        self,
        back_slot: FormationSlot,
        combatants: Mapping[str, ProtectingCombatant],
    ) -> str | None:
        if not is_back(back_slot):
            return None
        front_actor_id = self.actor_at(front_slot_for(back_slot))
        if front_actor_id is None:
            return None
        front_actor = combatants.get(front_actor_id)
        if front_actor is None or not front_actor.can_protect():
            return None
        return front_actor_id

    def is_protected(
        self,
        actor_id: str,
        combatants: Mapping[str, ProtectingCombatant],
    ) -> bool:
        slot = self.slot_of(actor_id)
        if slot is None or not is_back(slot):
            return False
        return self.protector_for(slot, combatants) is not None

    def is_exposed(
        self,
        actor_id: str,
        combatants: Mapping[str, ProtectingCombatant],
    ) -> bool:
        slot = self.slot_of(actor_id)
        if slot is None:
            return True
        if is_front(slot):
            return True
        return not self.is_protected(actor_id, combatants)
