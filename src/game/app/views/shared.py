"""Shared helpers used across view builder modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from game.campaign.company import HeroState
from game.combat.combat_state import LifeState
from game.combat.formation import FormationSlot


def _life_state_labels(life_state: str) -> tuple[str, ...]:
    if life_state == LifeState.ALIVE.value:
        return ("ready",)
    return (life_state,)


def _skill_description(
    intent: str,
    attack_type: str,
    amount_label: str,
    target_count: int,
) -> str:
    if intent == "heal":
        return f"heals up to {amount_label} HP, {target_count} allies"
    return f"{attack_type}, {target_count} targets"


def _skill_intent(tags: Sequence[str]) -> str:
    tag_set = set(tags)
    if "treatment" in tag_set or "heal" in tag_set or "support" in tag_set:
        return "heal"
    if tag_set & {"debuff", "control", "horror", "status"}:
        return "debuff"
    return "attack"


def _slot_display(slot: str) -> str:
    return slot.replace("_", " ").title()


def _formation_slot_summaries(
    slots: Mapping[FormationSlot, str | None],
    roster_by_id: Mapping[str, HeroState],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            slot.value,
            roster_by_id[hero_id].name
            if (hero_id := slots.get(slot)) is not None and hero_id in roster_by_id
            else "empty",
        )
        for slot in FormationSlot
    )
