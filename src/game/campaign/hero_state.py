"""Hero roster state for campaign saves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from game.campaign.hero_memory import (
    EarnedQuirkSlotState,
    FreshMemoryState,
    flat_quirks_from_slots,
    synthesize_earned_slots,
)
from game.campaign.serde_helpers import (
    life_state_from_raw,
    strain_marks_from_conditions,
    strain_marks_from_raw,
    strain_name_from_raw,
    tags_from_raw,
)
from game.combat.combat_state import (
    ActorStatus,
    FatigueState,
    LifeState,
    MoraleState,
    StatusSetProxy,
    StrainMark,
    StrainTier,
    Tag,
    fatigue_from_strain,
    life_state_from_statuses,
    strain_from_fatigue,
    tags_from_legacy_statuses,
)
from game.combat.formation import FormationSlot


@dataclass
class HeroState:
    hero_id: str
    name: str
    class_id: str
    max_hp: int
    hp: int
    speed: int
    accuracy: int
    defense: int
    damage: int
    max_effort: int
    effort: int
    skills: list[str]
    formation_slot: FormationSlot
    background: str = ""
    motive: str = ""
    life_state: LifeState = LifeState.ALIVE
    morale: MoraleState = MoraleState.STEADY
    strain: StrainTier = StrainTier.STEADY
    tags: set[Tag] = field(default_factory=set)
    quirks: list[str] = field(default_factory=list)
    strain_marks: set[StrainMark] = field(default_factory=set)
    personal_quirk: str | None = None
    mortal_wounds: int = 0
    in_surgery: bool = False
    equipped_gear_id: str | None = None
    career_signals: dict[str, int] = field(default_factory=dict)
    fresh_memories: list[FreshMemoryState] = field(default_factory=list)
    earned_quirk_slots: list[EarnedQuirkSlotState] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.earned_quirk_slots = synthesize_earned_slots(
            self.quirks,
            self.earned_quirk_slots,
        )
        self.quirks = flat_quirks_from_slots(self.earned_quirk_slots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hero_id": self.hero_id,
            "name": self.name,
            "class_id": self.class_id,
            "background": self.background,
            "motive": self.motive,
            "max_hp": self.max_hp,
            "hp": self.hp,
            "speed": self.speed,
            "accuracy": self.accuracy,
            "defense": self.defense,
            "damage": self.damage,
            "max_effort": self.max_effort,
            "effort": self.effort,
            "skills": list(self.skills),
            "formation_slot": self.formation_slot.value,
            "life_state": self.life_state.value,
            "morale": self.morale.name,
            "strain": self.strain.name,
            "tags": sorted(tag.name for tag in self.tags),
            "quirks": flat_quirks_from_slots(self.earned_quirk_slots)
            if self.earned_quirk_slots
            else list(self.quirks),
            "strain_marks": sorted(mark.value for mark in self.strain_marks),
            "personal_quirk": self.personal_quirk,
            "mortal_wounds": self.mortal_wounds,
            "in_surgery": self.in_surgery,
            "equipped_gear_id": self.equipped_gear_id,
            "career_signals": dict(self.career_signals),
            "fresh_memories": [memory.to_dict() for memory in self.fresh_memories],
            "earned_quirk_slots": [
                slot.to_dict() for slot in self.earned_quirk_slots
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeroState:
        return cls(
            hero_id=str(data["hero_id"]),
            name=str(data["name"]),
            class_id=str(data["class_id"]),
            background=str(data.get("background", "")),
            motive=str(data.get("motive", "")),
            max_hp=int(data["max_hp"]),
            hp=int(data["hp"]),
            speed=int(data["speed"]),
            accuracy=int(data["accuracy"]),
            defense=int(data["defense"]),
            damage=int(data["damage"]),
            max_effort=int(data["max_effort"]),
            effort=int(data["effort"]),
            skills=[str(skill) for skill in data["skills"]],
            formation_slot=FormationSlot(str(data["formation_slot"])),
            life_state=life_state_from_raw(data),
            morale=MoraleState[str(data.get("morale", MoraleState.STEADY.name))],
            strain=StrainTier[strain_name_from_raw(data)],
            tags=tags_from_raw(data),
            quirks=[str(quirk) for quirk in data.get("quirks", [])],
            strain_marks=strain_marks_from_raw(data),
            personal_quirk=(
                None
                if data.get("personal_quirk") is None
                else str(data["personal_quirk"])
            ),
            mortal_wounds=int(data.get("mortal_wounds", 0)),
            in_surgery=bool(data.get("in_surgery", False)),
            equipped_gear_id=(
                None
                if data.get("equipped_gear_id") is None
                else str(data["equipped_gear_id"])
            ),
            career_signals={
                str(key): int(value)
                for key, value in data.get("career_signals", {}).items()
            },
            fresh_memories=[
                FreshMemoryState.from_dict(memory)
                for memory in data.get("fresh_memories", [])
                if isinstance(memory, dict)
            ],
            earned_quirk_slots=[
                EarnedQuirkSlotState.from_dict(slot)
                for slot in data.get("earned_quirk_slots", [])
                if isinstance(slot, dict)
            ],
        )

    @property
    def statuses(self) -> set[ActorStatus]:
        return StatusSetProxy(self)

    @statuses.setter
    def statuses(self, value: set[ActorStatus]) -> None:
        self.life_state = life_state_from_statuses(value)
        self.tags.update(tags_from_legacy_statuses(value))

    @property
    def fatigue(self) -> FatigueState:
        return fatigue_from_strain(self.strain)

    @fatigue.setter
    def fatigue(self, value: FatigueState) -> None:
        self.strain = strain_from_fatigue(value)

    @property
    def conditions(self) -> list[str]:
        if self.strain == StrainTier.SPENT:
            return ["spent"]
        return sorted(mark.value for mark in self.strain_marks)

    @conditions.setter
    def conditions(self, value: list[str]) -> None:
        strain = self.strain
        marks = strain_marks_from_conditions(value)
        if any(condition in {"spent", "exhausted"} for condition in value):
            strain = StrainTier.SPENT
        self.strain = strain
        self.strain_marks = marks
