"""App-level HCI snapshot helpers."""

from __future__ import annotations

from typing import Any

from game.campaign.company import CompanyState, HeroState
from game.campaign.town import IN_SURGERY_LABEL
from game.core.events import GameEvent
from game.core.hci import (
    HciResultAnalysis,
    HeroStateSnapshot,
    StateSnapshot,
    build_hci_result_analysis,
)


def capture_hci_state(
    company: CompanyState | None,
    manual_combat: Any | None = None,
) -> StateSnapshot:
    if company is None:
        return StateSnapshot()

    expedition = company.active_expedition
    current_actor = manual_combat.pending_hero() if manual_combat is not None else None
    selected_skill_id = (
        str(manual_combat.selected_skill_id or "") if manual_combat is not None else ""
    )
    combat_state = manual_combat.state if manual_combat is not None else None
    return StateSnapshot(
        has_company=True,
        company_name=company.name,
        reputation=company.reputation,
        coin=company.coin,
        supplies=tuple(sorted((str(key), int(value)) for key, value in company.supplies.items())),
        inventory=tuple(
            sorted((str(key), int(value)) for key, value in company.inventory.items())
        ),
        location_id=str(company.town_state.get("location_id", "")),
        location_name=str(company.town_state.get("location", "")),
        active_expedition_id=expedition.expedition_id if expedition is not None else "",
        dungeon_node_id=expedition.current_node_id if expedition is not None else "",
        dungeon_pending_combat_node_id=(
            expedition.pending_combat_node_id if expedition is not None else ""
        )
        or "",
        combat_encounter_id=manual_combat.encounter_id if manual_combat is not None else "",
        combat_round=combat_state.round_number if combat_state is not None else None,
        combat_actor_id=current_actor.actor_id if current_actor is not None else "",
        combat_selected_skill_id=selected_skill_id,
        heroes=tuple(
            HeroStateSnapshot(
                hero_id=hero.hero_id,
                name=hero.name,
                hp=hero.hp,
                max_hp=hero.max_hp,
                effort=hero.effort,
                max_effort=hero.max_effort,
                mortal_wounds=hero.mortal_wounds,
                statuses=_hero_snapshot_statuses(hero),
            )
            for hero in sorted(company.roster, key=lambda roster_hero: roster_hero.hero_id)
        ),
    )


def _hero_snapshot_statuses(hero: HeroState) -> tuple[str, ...]:
    statuses: list[str] = []
    if hero.life_state.value != "alive":
        statuses.append(hero.life_state.value)
    if hero.in_surgery:
        statuses.append(IN_SURGERY_LABEL)
    return tuple(statuses)


def analyze_hci_result(
    before: StateSnapshot,
    after: StateSnapshot,
    events: list[GameEvent],
    *,
    error: str | None = None,
) -> HciResultAnalysis:
    return build_hci_result_analysis(before, after, events, error=error)
