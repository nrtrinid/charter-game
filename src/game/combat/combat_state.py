"""Runtime combat state dataclasses."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import InitVar, dataclass, field
from enum import Enum, StrEnum, auto
from typing import Protocol

from game.combat.formation import Formation, FormationSlot


class Team(StrEnum):
    HERO = "hero"
    ENEMY = "enemy"


class ActorStatus(StrEnum):
    DOWNED = "downed"
    DEAD = "dead"
    STUNNED = "stunned"
    KNOCKED_DOWN = "knocked_down"


class LifeState(StrEnum):
    ALIVE = "alive"
    DOWNED = "downed"
    DEAD = "dead"


class MoraleState(Enum):
    BROKEN = 0
    SHAKEN = 1
    STEADY = 2
    INSPIRED = 3


class FatigueState(Enum):
    EXHAUSTED = 0
    TIRED = 1
    STEADY = 2
    ENERGIZED = 3


class StrainTier(Enum):
    FRESH = 0
    STEADY = 1
    WORN = 2
    STRAINED = 3
    SPENT = 4


class StrainMark(StrEnum):
    WINDED = "winded"
    DRAINED = "drained"
    BATTERED = "battered"
    FRAYED = "frayed"


class CohesionState(Enum):
    FRACTURED = auto()
    UNSTEADY = auto()
    STRONG = auto()


class Tag(Enum):
    WET = auto()
    BURNING = auto()
    FROZEN = auto()
    SHOCKED = auto()
    MARKED = auto()
    GUARDED = auto()
    STUNNED = auto()
    KNOCKED_DOWN = auto()


class StatusOwner(Protocol):
    life_state: LifeState
    tags: set[Tag]


DEFAULT_MARKED_TURNS = 2

SPENT_EFFECTIVE_MARKS: frozenset[StrainMark] = frozenset(
    {StrainMark.WINDED, StrainMark.DRAINED, StrainMark.FRAYED}
)


def life_state_from_statuses(statuses: set[ActorStatus]) -> LifeState:
    if ActorStatus.DEAD in statuses:
        return LifeState.DEAD
    if ActorStatus.DOWNED in statuses:
        return LifeState.DOWNED
    return LifeState.ALIVE


def statuses_from_life_state(life_state: LifeState) -> set[ActorStatus]:
    if life_state == LifeState.DEAD:
        return {ActorStatus.DEAD}
    if life_state == LifeState.DOWNED:
        return {ActorStatus.DOWNED}
    return set()


def tags_from_legacy_statuses(statuses: set[ActorStatus]) -> set[Tag]:
    tags: set[Tag] = set()
    if ActorStatus.STUNNED in statuses:
        tags.add(Tag.STUNNED)
    if ActorStatus.KNOCKED_DOWN in statuses:
        tags.add(Tag.KNOCKED_DOWN)
    return tags


def statuses_from_legacy_tags(tags: set[Tag]) -> set[ActorStatus]:
    statuses: set[ActorStatus] = set()
    if Tag.STUNNED in tags:
        statuses.add(ActorStatus.STUNNED)
    if Tag.KNOCKED_DOWN in tags:
        statuses.add(ActorStatus.KNOCKED_DOWN)
    return statuses


class StatusSetProxy(set[ActorStatus]):
    def __init__(self, owner: StatusOwner):
        self._owner = owner
        super().__init__(
            statuses_from_life_state(owner.life_state)
            | statuses_from_legacy_tags(owner.tags)
        )

    def add(self, element: ActorStatus) -> None:
        super().add(element)
        self._sync_owner()

    def discard(self, element: object) -> None:
        super().discard(element)
        self._sync_owner()

    def remove(self, element: ActorStatus) -> None:
        super().remove(element)
        self._sync_owner()

    def clear(self) -> None:
        super().clear()
        self._sync_owner()

    def update(self, *others: Iterable[ActorStatus]) -> None:
        super().update(*others)
        self._sync_owner()

    def _sync_owner(self) -> None:
        self._owner.life_state = life_state_from_statuses(set(self))
        legacy_tags = {Tag.STUNNED, Tag.KNOCKED_DOWN}
        self._owner.tags.difference_update(legacy_tags)
        self._owner.tags.update(tags_from_legacy_statuses(set(self)))


def strain_from_fatigue(fatigue: FatigueState) -> StrainTier:
    return {
        FatigueState.ENERGIZED: StrainTier.FRESH,
        FatigueState.STEADY: StrainTier.STEADY,
        FatigueState.TIRED: StrainTier.WORN,
        FatigueState.EXHAUSTED: StrainTier.STRAINED,
    }[fatigue]


def fatigue_from_strain(strain: StrainTier) -> FatigueState:
    return {
        StrainTier.FRESH: FatigueState.ENERGIZED,
        StrainTier.STEADY: FatigueState.STEADY,
        StrainTier.WORN: FatigueState.TIRED,
        StrainTier.STRAINED: FatigueState.EXHAUSTED,
        StrainTier.SPENT: FatigueState.EXHAUSTED,
    }[strain]


def has_strain_mark(
    strain: StrainTier,
    strain_marks: set[StrainMark],
    mark: StrainMark,
) -> bool:
    return mark in strain_marks or (strain == StrainTier.SPENT and mark in SPENT_EFFECTIVE_MARKS)


def effective_strain_marks(strain: StrainTier, strain_marks: set[StrainMark]) -> set[StrainMark]:
    marks = set(strain_marks)
    if strain == StrainTier.SPENT:
        marks.update(SPENT_EFFECTIVE_MARKS)
    return marks


@dataclass
class Character:
    id: str
    name: str
    max_hp: int
    hp: int
    max_effort: int
    effort: int
    morale: MoraleState = MoraleState.STEADY
    strain: StrainTier = StrainTier.STEADY
    life_state: LifeState = LifeState.ALIVE
    position: FormationSlot | None = None
    tags: set[Tag] = field(default_factory=set)
    quirks: list[str] = field(default_factory=list)
    strain_marks: set[StrainMark] = field(default_factory=set)
    personal_quirk: str | None = None


@dataclass
class Party:
    active_character_ids: list[str]
    formation: dict[FormationSlot, str]

    def derive_cohesion(self, characters: dict[str, Character]) -> CohesionState:
        active = [characters[character_id] for character_id in self.active_character_ids]
        return derive_cohesion_from_morale(active)


def derive_cohesion_from_morale(characters: Sequence[Character | Combatant]) -> CohesionState:
    active = [
        character
        for character in characters
        if not isinstance(character, Combatant) or character.is_alive()
    ]
    if not active:
        return CohesionState.FRACTURED
    unstable = sum(
        1
        for character in active
        if character.morale in {MoraleState.SHAKEN, MoraleState.BROKEN}
    )
    broken = sum(1 for character in active if character.morale == MoraleState.BROKEN)

    if broken > 0 or unstable * 2 >= len(active):
        return CohesionState.FRACTURED
    if unstable > 0:
        return CohesionState.UNSTEADY
    return CohesionState.STRONG


@dataclass
class Combatant:
    actor_id: str
    name: str
    team: Team
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
    life_state: LifeState = LifeState.ALIVE
    morale: MoraleState = MoraleState.STEADY
    strain: StrainTier = StrainTier.STEADY
    tags: set[Tag] = field(default_factory=set)
    tag_turns: dict[Tag, int] = field(default_factory=dict)
    quirks: list[str] = field(default_factory=list)
    conditions: InitVar[list[str] | None] = None
    strain_marks: set[StrainMark] = field(default_factory=set)
    personal_quirk: str | None = None
    mortal_wounds: int = 0
    class_id: str = ""

    def __post_init__(self, conditions: list[str] | None) -> None:
        if conditions is not None:
            if any(condition in {"spent", "exhausted"} for condition in conditions):
                self.strain = StrainTier.SPENT
            self.strain_marks = {
                StrainMark(condition)
                for condition in conditions
                if condition in {mark.value for mark in StrainMark}
            }
            if self.strain == StrainTier.SPENT:
                self.strain_marks.update(SPENT_EFFECTIVE_MARKS)

    def is_alive(self) -> bool:
        return self.life_state != LifeState.DEAD

    def is_downed(self) -> bool:
        return self.life_state == LifeState.DOWNED

    def can_act(self) -> bool:
        blocking = {Tag.FROZEN, Tag.STUNNED, Tag.KNOCKED_DOWN}
        return self.life_state == LifeState.ALIVE and self.tags.isdisjoint(blocking)

    def can_protect(self) -> bool:
        blocking = {Tag.STUNNED, Tag.KNOCKED_DOWN}
        return self.life_state == LifeState.ALIVE and self.tags.isdisjoint(blocking) and self.hp > 0

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



def apply_marked(combatant: Combatant, turns: int = DEFAULT_MARKED_TURNS) -> None:
    combatant.tags.add(Tag.MARKED)
    combatant.tag_turns[Tag.MARKED] = max(1, turns)


def clear_marked(combatant: Combatant) -> None:
    combatant.tags.discard(Tag.MARKED)
    combatant.tag_turns.pop(Tag.MARKED, None)


def tick_marked_turn(combatant: Combatant) -> bool:
    if Tag.MARKED not in combatant.tags:
        combatant.tag_turns.pop(Tag.MARKED, None)
        return False

    remaining = combatant.tag_turns.get(Tag.MARKED, DEFAULT_MARKED_TURNS) - 1
    if remaining > 0:
        combatant.tag_turns[Tag.MARKED] = remaining
        return False

    clear_marked(combatant)
    return True


@dataclass
class CombatState:
    heroes: dict[str, Combatant]
    enemies: dict[str, Combatant]
    party_formation: Formation
    enemy_formation: Formation
    round_number: int = 1
    quirk_once_per_combat: set[str] = field(default_factory=set)

    def all_combatants(self) -> dict[str, Combatant]:
        return {**self.heroes, **self.enemies}

    def actor(self, actor_id: str) -> Combatant:
        try:
            return self.heroes[actor_id]
        except KeyError:
            return self.enemies[actor_id]

    def formation_for(self, team: Team) -> Formation:
        return self.party_formation if team == Team.HERO else self.enemy_formation

    def enemy_formation_for(self, team: Team) -> Formation:
        return self.enemy_formation if team == Team.HERO else self.party_formation

    def side_for(self, team: Team) -> dict[str, Combatant]:
        return self.heroes if team == Team.HERO else self.enemies

    def opposing_side_for(self, team: Team) -> dict[str, Combatant]:
        return self.enemies if team == Team.HERO else self.heroes

    def active_combatants(self) -> list[Combatant]:
        return [combatant for combatant in self.all_combatants().values() if combatant.can_act()]

    def derive_cohesion(self) -> CohesionState:
        return derive_cohesion_from_morale(list(self.heroes.values()))

    def living_enemies(self) -> list[Combatant]:
        return [enemy for enemy in self.enemies.values() if enemy.is_alive()]

    def living_heroes(self) -> list[Combatant]:
        return [hero for hero in self.heroes.values() if hero.is_alive()]

    def hero_party_can_act(self) -> bool:
        return any(hero.can_act() for hero in self.heroes.values())

    def is_victory(self) -> bool:
        return not any(enemy.is_alive() for enemy in self.enemies.values())

    def is_defeat(self) -> bool:
        return not self.hero_party_can_act()
