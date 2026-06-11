"""Small quirk and condition hooks for combat."""

from __future__ import annotations

from game.combat.combat_state import (
    CohesionState,
    Combatant,
    CombatState,
    MoraleState,
    StrainMark,
    StrainTier,
    Tag,
)
from game.combat.combat_state import (
    has_strain_mark as _has_strain_mark,
)
from game.combat.formation import FormationSlot
from game.combat.morale import lower_morale

PERSONAL_HOLD_THE_LINE = "hold_the_line"
PERSONAL_OPPORTUNIST = "opportunist"
PERSONAL_GENTLE_HANDS = "gentle_hands"
PERSONAL_GRAVE_CALM = "grave_calm"

QUIRK_BLOOD_HOT = "blood_hot"
QUIRK_GRIM_FINISH = "grim_finish"
QUIRK_BATTLE_RHYTHM = "battle_rhythm"
QUIRK_CLOSER = "closer"
QUIRK_NO_WASTE = "no_waste"
QUIRK_STEADY_HAND = "steady_hand"
QUIRK_RED_WORK = "red_work"
QUIRK_HARD_LESSON = "hard_lesson"
QUIRK_MAZE_SIGHTED = "maze_sighted"
QUIRK_GOLD_FEVER = "gold_fever"
QUIRK_ICE_NERVES = "ice_nerves"
QUIRK_CLEAN_KILL = "clean_kill"
QUIRK_PREDATOR = "predator"
QUIRK_FIELD_MEDIC = "field_medic"
QUIRK_STEADY_VOICE = "steady_voice"
QUIRK_DESPERATE_FOCUS = "desperate_focus"
QUIRK_LAST_ANCHOR = "last_anchor"
QUIRK_KEEPS_COUNT = "keeps_count"

CONDITION_WINDED = "winded"
CONDITION_DRAINED = "drained"
CONDITION_BATTERED = "battered"
CONDITION_FRAYED = "frayed"
CONDITION_SPENT = "spent"
LEGACY_CONDITION_EXHAUSTED = "exhausted"

EARNED_QUIRK_LIMIT = 3


def has_trait(combatant: Combatant, trait_id: str) -> bool:
    return combatant.personal_quirk == trait_id or trait_id in combatant.quirks


def normalize_condition(condition_id: str) -> str:
    return CONDITION_SPENT if condition_id == LEGACY_CONDITION_EXHAUSTED else condition_id


def normalize_conditions(condition_ids: list[str]) -> list[str]:
    return [normalize_condition(condition_id) for condition_id in condition_ids]


def has_condition(combatant: Combatant, condition_id: str) -> bool:
    normalized = normalize_condition(condition_id)
    if normalized == CONDITION_SPENT:
        return combatant.strain == StrainTier.SPENT
    try:
        mark = StrainMark(normalized)
    except ValueError:
        return False
    return _has_strain_mark(combatant.strain, combatant.strain_marks, mark)


def has_strain_mark(combatant: Combatant, mark: StrainMark) -> bool:
    return _has_strain_mark(combatant.strain, combatant.strain_marks, mark)


def front_row(slot: FormationSlot) -> bool:
    return slot in {FormationSlot.FRONT_LEFT, FormationSlot.FRONT_RIGHT}


def apply_combat_start_traits(state: CombatState) -> None:
    for hero in state.heroes.values():
        if has_strain_mark(hero, StrainMark.DRAINED):
            hero.effort = max(0, hero.effort - 1)
        if has_strain_mark(hero, StrainMark.BATTERED):
            hero.defense -= 1
        if QUIRK_DESPERATE_FOCUS in hero.quirks and hero.morale == MoraleState.SHAKEN:
            hero.defense -= 1
        if has_strain_mark(hero, StrainMark.FRAYED):
            lower_morale(hero, min_state=MoraleState.SHAKEN)

    cohesion = state.derive_cohesion()
    if cohesion == CohesionState.FRACTURED:
        for hero in state.heroes.values():
            if (
                QUIRK_LAST_ANCHOR in hero.quirks
                and hero.morale in {MoraleState.STEADY, MoraleState.INSPIRED}
                and hero.can_protect()
            ):
                hero.tags.add(Tag.GUARDED)
    if cohesion.name != "STRONG":
        return
    for hero in state.heroes.values():
        if hero.personal_quirk != PERSONAL_HOLD_THE_LINE:
            continue
        if front_row(hero.formation_slot):
            hero.tags.add(Tag.GUARDED)


def can_gain_earned_quirk(combatant: Combatant, quirk_id: str) -> bool:
    return quirk_id not in combatant.quirks and len(combatant.quirks) < EARNED_QUIRK_LIMIT
