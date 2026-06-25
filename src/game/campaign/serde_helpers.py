"""Shared save/load parsing helpers for hero and report payloads."""

from __future__ import annotations

from typing import Any

from game.combat.combat_state import (
    ActorStatus,
    FatigueState,
    LifeState,
    StrainMark,
    StrainTier,
    Tag,
    life_state_from_statuses,
    strain_from_fatigue,
    tags_from_legacy_statuses,
)

_CONDITION_TO_STRAIN_MARK: dict[str, StrainMark] = {
    "winded": StrainMark.WINDED,
    "drained": StrainMark.DRAINED,
    "battered": StrainMark.BATTERED,
    "frayed": StrainMark.FRAYED,
}


def legacy_statuses_from_raw(data: dict[str, Any]) -> set[ActorStatus]:
    statuses: set[ActorStatus] = set()
    for raw in data.get("statuses", []):
        value = str(raw)
        try:
            statuses.add(ActorStatus(value))
            continue
        except ValueError:
            pass
        try:
            statuses.add(ActorStatus[value])
        except KeyError:
            continue
    return statuses


def life_state_from_raw(data: dict[str, Any], *, legacy: bool = True) -> LifeState:
    value = data.get("life_state")
    if value is not None:
        try:
            return LifeState(str(value))
        except ValueError:
            return LifeState.ALIVE
    if not legacy:
        return LifeState.ALIVE
    return life_state_from_statuses(legacy_statuses_from_raw(data))


def tags_from_raw(data: dict[str, Any], *, legacy: bool = True) -> set[Tag]:
    tags: set[Tag] = set()
    for raw in data.get("tags", []):
        value = str(raw)
        try:
            tags.add(Tag[value])
            continue
        except KeyError:
            pass
        normalized = value.upper()
        try:
            tags.add(Tag[normalized])
        except KeyError:
            continue
    if legacy:
        tags.update(tags_from_legacy_statuses(legacy_statuses_from_raw(data)))
    return tags


def strain_name_from_raw(data: dict[str, Any], *, legacy: bool = True) -> str:
    if data.get("strain") is not None:
        return str(data["strain"])
    if not legacy:
        return StrainTier.STEADY.name
    fatigue_raw = data.get("fatigue", FatigueState.STEADY.name)
    try:
        fatigue = FatigueState[str(fatigue_raw)]
    except KeyError:
        fatigue = FatigueState.STEADY
    strain = strain_from_fatigue(fatigue)
    if any(condition in {"spent", "exhausted"} for condition in data.get("conditions", [])):
        strain = StrainTier.SPENT
    return strain.name


def strain_marks_from_conditions(raw_values: object) -> set[StrainMark]:
    if not isinstance(raw_values, list | tuple | set):
        return set()
    marks: set[StrainMark] = set()
    for raw in raw_values:
        value = str(raw)
        if value in _CONDITION_TO_STRAIN_MARK:
            marks.add(_CONDITION_TO_STRAIN_MARK[value])
        elif value in {"spent", "exhausted"}:
            marks.update({StrainMark.WINDED, StrainMark.DRAINED, StrainMark.FRAYED})
    return marks


def strain_marks_from_raw(data: dict[str, Any], *, legacy: bool = True) -> set[StrainMark]:
    marks = strain_marks_from_conditions(data.get("strain_marks", ()))
    if legacy:
        marks.update(strain_marks_from_conditions(data.get("conditions", ())))
    return marks


def strain_mark_values_from_raw(data: dict[str, Any], *, legacy: bool = True) -> list[str]:
    if legacy:
        return sorted(mark.value for mark in strain_marks_from_raw(data, legacy=True))
    raw = data.get("strain_marks", ())
    if not isinstance(raw, (list, tuple, set)):
        return []
    return sorted(str(value) for value in raw)
