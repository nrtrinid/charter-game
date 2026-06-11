"""Roster helpers and combat conversion."""

from __future__ import annotations

from game.campaign.company import CompanyState, HeroState
from game.campaign.gear import clear_dead_hero_gear, effective_hero_stats
from game.combat.combat_state import Combatant, LifeState, Tag, Team
from game.combat.formation import Formation, FormationSlot
from game.content.definitions import GameDefinitions


def living_roster(company: CompanyState) -> list[HeroState]:
    return [hero for hero in company.roster if hero.life_state != LifeState.DEAD]


def active_roster(company: CompanyState) -> list[HeroState]:
    living = {hero.hero_id: hero for hero in living_roster(company)}
    return [
        living[hero_id]
        for hero_id in company.active_party_slots.values()
        if hero_id is not None and hero_id in living
    ]


def reserve_roster(company: CompanyState) -> list[HeroState]:
    active_ids = {hero_id for hero_id in company.active_party_slots.values() if hero_id is not None}
    return [hero for hero in living_roster(company) if hero.hero_id not in active_ids]


def hero_to_combatant(
    hero: HeroState,
    formation_slot: FormationSlot | None = None,
    definitions: GameDefinitions | None = None,
) -> Combatant:
    stats = effective_hero_stats(hero, definitions)
    return Combatant(
        actor_id=hero.hero_id,
        name=hero.name,
        team=Team.HERO,
        max_hp=stats.max_hp,
        hp=min(hero.hp, stats.max_hp),
        speed=hero.speed,
        accuracy=stats.accuracy,
        defense=hero.defense,
        damage=stats.damage,
        max_effort=stats.max_effort,
        effort=min(hero.effort, stats.max_effort),
        skills=list(hero.skills),
        formation_slot=formation_slot or hero.formation_slot,
        life_state=hero.life_state,
        morale=hero.morale,
        strain=hero.strain,
        tags=_combat_start_tags(hero.tags),
        quirks=list(hero.quirks),
        strain_marks=set(hero.strain_marks),
        personal_quirk=hero.personal_quirk,
        mortal_wounds=hero.mortal_wounds,
        class_id=hero.class_id,
    )


def party_combatants(
    company: CompanyState,
    definitions: GameDefinitions | None = None,
) -> tuple[dict[str, Combatant], Formation]:
    living = {hero.hero_id: hero for hero in living_roster(company)}
    heroes: dict[str, Combatant] = {}
    formation = Formation.empty()
    for slot, hero_id in company.active_party_slots.items():
        if hero_id is None or hero_id not in living:
            continue
        hero = hero_to_combatant(living[hero_id], slot, definitions)
        heroes[hero.actor_id] = hero
        formation.place(hero.actor_id, slot)
    return heroes, formation


def sync_company_from_combat(
    company: CompanyState,
    combatants: dict[str, Combatant],
    party_formation: Formation | None = None,
) -> None:
    dead_ids = {hero.hero_id for hero in company.deceased_heroes}
    survivors: list[HeroState] = []
    for hero in company.roster:
        combatant = combatants.get(hero.hero_id)
        if combatant is None:
            survivors.append(hero)
            continue
        hero.hp = combatant.hp
        hero.effort = combatant.effort
        hero.life_state = combatant.life_state
        hero.morale = combatant.morale
        hero.strain = combatant.strain
        hero.tags = _persistent_tags(combatant.tags)
        hero.quirks = list(combatant.quirks)
        hero.strain_marks = set(combatant.strain_marks)
        hero.personal_quirk = combatant.personal_quirk
        hero.mortal_wounds = combatant.mortal_wounds
        if hero.life_state == LifeState.DEAD:
            clear_dead_hero_gear(hero)
            if hero.hero_id not in dead_ids:
                company.deceased_heroes.append(hero)
                dead_ids.add(hero.hero_id)
        else:
            survivors.append(hero)
    company.roster = survivors
    living_ids = {hero.hero_id for hero in survivors}
    if party_formation is not None:
        hero_by_id = {hero.hero_id: hero for hero in survivors}
        for slot in FormationSlot:
            hero_id = party_formation.actor_at(slot)
            if hero_id is not None and hero_id in living_ids:
                company.active_party_slots[slot] = hero_id
                hero_by_id[hero_id].formation_slot = slot
            else:
                company.active_party_slots[slot] = None
    for slot, hero_id in list(company.active_party_slots.items()):
        if hero_id is not None and hero_id not in living_ids:
            company.active_party_slots[slot] = None


def _combat_start_tags(tags: set[Tag]) -> set[Tag]:
    return _persistent_tags(tags)


def _persistent_tags(tags: set[Tag]) -> set[Tag]:
    return {tag for tag in tags if tag is not Tag.MARKED}
