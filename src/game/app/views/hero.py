"""App-facing view models for terminal rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from game.app.actions import (
    ActionProvider,
    ScreenAction,
    ScreenActionKind,
)
from game.app.views.shared import (
    _formation_slot_summaries,
    _life_state_labels,
    _slot_display,
)
from game.campaign.company import (
    CompanyState,
    HeroMemoryEntry,
    HeroState,
)
from game.campaign.gear import (
    available_gear_count,
    effective_hero_stats,
    gear_effect_summary,
    gear_unavailable_reason,
)
from game.campaign.recruitment import RecruitChoice
from game.campaign.roster import active_roster, living_roster, reserve_roster
from game.campaign.town import (
    IN_SURGERY_LABEL,
    effective_roster_cap,
)
from game.combat.combat_state import LifeState
from game.combat.formation import (
    Formation,
    FormationSlot,
    back_slot_for,
    is_back,
)
from game.content.definitions import GameDefinitions
from game.ui.wounds import mortal_wound_badge


@dataclass(frozen=True)
class HeroListEntry:
    hero_id: str
    name: str
    class_id: str
    slot: str
    hp: int
    max_hp: int
    effort: int
    max_effort: int
    mortal_wounds: int
    morale: str
    strain: str
    life_state: str
    personal_quirk: str = ""
    quirks: tuple[str, ...] = ()
    strain_marks: tuple[str, ...] = ()
    memory_count: int = 0
    latest_memory: str = ""
    equipped_gear: str = ""
    stat_bonus: str = ""

    @property
    def statuses(self) -> tuple[str, ...]:
        return _life_state_labels(self.life_state)

    @property
    def fatigue(self) -> str:
        return self.strain

    @property
    def conditions(self) -> tuple[str, ...]:
        return self.strain_marks

@dataclass(frozen=True)
class HeroSheetTraitView:
    trait_id: str
    name: str
    kind: str
    description: str = ""
    positive_text: str = ""
    negative_text: str = ""
    stability: str = ""

@dataclass(frozen=True)
class HeroSheetFreshMemoryView:
    family_id: str
    name: str
    intensity: int
    tags: tuple[str, ...] = ()
    source_summary: str = ""
    pending_manifestation: bool = False

@dataclass(frozen=True)
class HeroSheetMemoryEntryView:
    summary: str
    kind: str
    expedition_id: str
    dungeon_id: str
    node_id: str = ""
    encounter_id: str = ""

@dataclass(frozen=True)
class HeroSheetSignalView:
    label: str
    score: int

@dataclass(frozen=True)
class HeroSheetView:
    hero_id: str
    name: str
    class_id: str
    class_name: str
    roster_state: str
    slot: str
    background: str = ""
    motive: str = ""
    hp: int = 0
    max_hp: int = 0
    effort: int = 0
    max_effort: int = 0
    speed: int = 0
    accuracy: int = 0
    defense: int = 0
    damage: int = 0
    morale: str = ""
    strain: str = ""
    life_state: str = ""
    statuses: tuple[str, ...] = ()
    mortal_wounds: int = 0
    equipped_gear: str = ""
    equipped_gear_description: str = ""
    stat_bonus: str = ""
    personal_quirk: HeroSheetTraitView | None = None
    earned_quirks: tuple[HeroSheetTraitView, ...] = ()
    strain_marks: tuple[HeroSheetTraitView, ...] = ()
    fresh_memories: tuple[HeroSheetFreshMemoryView, ...] = ()
    permanent_memories: tuple[HeroSheetMemoryEntryView, ...] = ()
    career_signals: tuple[HeroSheetSignalView, ...] = ()
    available_kits: tuple[GearItemView, ...] = ()
    can_manage_gear: bool = True
    gear_manage_reason: str = ""

    @property
    def latest_memory(self) -> str:
        return self.permanent_memories[0].summary if self.permanent_memories else ""

@dataclass(frozen=True)
class RosterSectionView:
    title: str
    heroes: tuple[HeroListEntry, ...]

@dataclass(frozen=True)
class MemorialEntryView:
    hero_id: str
    name: str
    class_id: str
    mortal_wounds: int
    final_memory: str = ""

@dataclass(frozen=True)
class FormationSlotView:
    slot: FormationSlot
    slot_label: str
    hero_id: str | None
    hero_name: str
    condition: str
    class_name: str = ""
    vitals_line: str = ""
    protection_line: str = ""
    abnormal_status: str = ""
    mortal_wounds: int = 0

@dataclass(frozen=True)
class FormationView:
    slots: tuple[FormationSlotView, ...]
    assignable_heroes: tuple[HeroListEntry, ...]
    actions: tuple[ScreenAction, ...]

@dataclass(frozen=True)
class GearItemView:
    gear_id: str
    name: str
    description: str
    owned_count: int
    equipped_count: int
    available_count: int
    cost: int | None
    state: str
    effect_summary: str
    unavailable_reason: str = ""

@dataclass(frozen=True)
class GearHeroView:
    hero_id: str
    name: str
    class_id: str
    equipped_gear_id: str = ""
    equipped_gear_name: str = ""
    condition: str = ""

@dataclass(frozen=True)
class GearInventoryView:
    reputation: int
    coin: int
    can_manage: bool
    can_purchase: bool
    manage_reason: str = ""
    purchase_reason: str = ""
    items: tuple[GearItemView, ...] = ()
    heroes: tuple[GearHeroView, ...] = ()
    actions: tuple[ScreenAction, ...] = ()

@dataclass(frozen=True)
class RecruitOfferView:
    name: str
    class_id: str
    class_name: str
    background: str
    motive: str
    cost: int

@dataclass(frozen=True)
class RecruitOffersView:
    reputation: int
    coin: int
    roster_count: int
    roster_cap: int
    offers: tuple[RecruitOfferView, ...]
    actions: tuple[ScreenAction, ...]

def build_roster_sections(
    company: CompanyState,
    definitions: GameDefinitions | None = None,
) -> tuple[RosterSectionView, ...]:
    return (
        RosterSectionView(
            "Active Party",
            tuple(
                _hero_entry(hero, _hero_memories(company, hero.hero_id), definitions)
                for hero in active_roster(company)
            ),
        ),
        RosterSectionView(
            "Reserves",
            tuple(
                _hero_entry(hero, _hero_memories(company, hero.hero_id), definitions)
                for hero in reserve_roster(company)
            ),
        ),
        RosterSectionView(
            "Memorial",
            tuple(
                _hero_entry(hero, _hero_memories(company, hero.hero_id), definitions)
                for hero in company.deceased_heroes
            ),
        ),
    )

def build_memorial_entries(company: CompanyState) -> tuple[MemorialEntryView, ...]:
    entries: list[MemorialEntryView] = []
    for hero in company.deceased_heroes:
        memories = _hero_memories(company, hero.hero_id)
        final_memory = memories[-1].summary if memories else ""
        entries.append(
            MemorialEntryView(
                hero_id=hero.hero_id,
                name=hero.name,
                class_id=hero.class_id,
                mortal_wounds=hero.mortal_wounds,
                final_memory=final_memory,
            )
        )
    return tuple(entries)

def build_formation_view(
    company: CompanyState,
    definitions: GameDefinitions | None = None,
) -> FormationView:
    roster_by_id = {hero.hero_id: hero for hero in company.roster}
    formation = Formation.from_mapping(company.active_party_slots)
    protectors = _town_protectors(roster_by_id)
    slots: list[FormationSlotView] = []
    for slot in FormationSlot:
        hero = roster_by_id.get(company.active_party_slots.get(slot) or "")
        if hero is not None:
            stats = effective_hero_stats(hero, definitions)
            hero_class = (
                definitions.hero_classes.get(hero.class_id) if definitions is not None else None
            )
            class_name = (
                hero_class.name if hero_class is not None else _trait_label(hero.class_id)
            )
            vitals_line = (
                f"{hero.hp}/{stats.max_hp} HP, {hero.effort}/{stats.max_effort} Effort"
            )
            slots.append(
                FormationSlotView(
                    slot=slot,
                    slot_label=slot.value,
                    hero_id=hero.hero_id,
                    hero_name=hero.name,
                    condition=_hero_condition(hero, definitions),
                    class_name=class_name,
                    vitals_line=vitals_line,
                    protection_line=_formation_protection_line(
                        slot,
                        formation,
                        protectors,
                        roster_by_id,
                    ),
                    abnormal_status=_hero_abnormal_status(hero, definitions),
                    mortal_wounds=hero.mortal_wounds,
                )
            )
        else:
            slots.append(
                FormationSlotView(
                    slot=slot,
                    slot_label=slot.value,
                    hero_id=None,
                    hero_name="empty",
                    condition="",
                    mortal_wounds=0,
                )
            )
    actions: list[ScreenAction] = []
    for index, slot in enumerate(FormationSlot, start=1):
        slot_actor_id = company.active_party_slots.get(slot)
        slot_name = _slot_display(slot.value)
        label = (
            f"{slot_name}: {roster_by_id[slot_actor_id].name}"
            if slot_actor_id is not None and slot_actor_id in roster_by_id
            else f"{slot_name}: [empty]"
        )
        actions.append(
            ScreenAction(
                str(index),
                label,
                slot.value,
                (slot.value.lower(), slot.value.replace("_", " ").lower()),
                kind=ScreenActionKind.TOWN,
                preview=f"Choose who holds {slot_name}.",
                result_hint="Formation changes protection lanes before the next fight.",
            )
        )
    actions.append(
        ScreenAction("5", "Back", "back", ("back", "b"), kind=ScreenActionKind.NAVIGATE)
    )
    return FormationView(
        slots=tuple(slots),
        assignable_heroes=tuple(
            _hero_entry(hero, _hero_memories(company, hero.hero_id), definitions)
            for hero in living_roster(company)
            if not hero.in_surgery
        ),
        actions=tuple(actions),
    )

def build_gear_inventory_view(
    company: CompanyState,
    definitions: GameDefinitions,
    *,
    can_manage: bool,
    can_purchase: bool,
    manage_reason: str = "",
    purchase_reason: str = "",
) -> GearInventoryView:
    return GearInventoryView(
        reputation=company.reputation,
        coin=company.coin,
        can_manage=can_manage,
        can_purchase=can_purchase,
        manage_reason=manage_reason,
        purchase_reason=purchase_reason,
        items=_gear_item_views(company, definitions),
        heroes=_gear_hero_views(company, definitions),
        actions=ActionProvider.gear_actions(
            company,
            definitions,
            can_manage=can_manage,
            can_purchase=can_purchase,
            manage_reason=manage_reason,
            purchase_reason=purchase_reason,
        ),
    )

def build_hero_sheet_view(
    company: CompanyState,
    definitions: GameDefinitions,
    hero_id: str,
    *,
    can_manage_gear: bool = True,
    gear_manage_reason: str = "",
) -> HeroSheetView | None:
    hero = next(
        (candidate for candidate in (*company.roster, *company.deceased_heroes)
         if candidate.hero_id == hero_id),
        None,
    )
    if hero is None:
        return None
    stats = effective_hero_stats(hero, definitions)
    hero_class = definitions.hero_classes.get(hero.class_id)
    gear = (
        definitions.gear.get(hero.equipped_gear_id)
        if hero.equipped_gear_id is not None
        else None
    )
    active_ids = {active.hero_id for active in active_roster(company)}
    reserve_ids = {reserve.hero_id for reserve in reserve_roster(company)}
    memories = tuple(reversed(_hero_memories(company, hero.hero_id)))
    if hero.in_surgery:
        roster_state = IN_SURGERY_LABEL
    else:
        roster_state = _hero_roster_state(hero.hero_id, active_ids, reserve_ids)
    return HeroSheetView(
        hero_id=hero.hero_id,
        name=hero.name,
        class_id=hero.class_id,
        class_name=hero_class.name if hero_class is not None else _trait_label(hero.class_id),
        roster_state=roster_state,
        slot=hero.formation_slot.value,
        background=hero.background,
        motive=hero.motive,
        hp=hero.hp,
        max_hp=stats.max_hp,
        effort=hero.effort,
        max_effort=stats.max_effort,
        speed=hero.speed,
        accuracy=stats.accuracy,
        defense=hero.defense,
        damage=stats.damage,
        morale=hero.morale.name.title(),
        strain=hero.strain.name.title(),
        life_state=hero.life_state.value,
        statuses=_life_state_labels(hero.life_state.value),
        mortal_wounds=hero.mortal_wounds,
        equipped_gear=gear.name if gear is not None else "",
        equipped_gear_description=gear.description if gear is not None else "",
        stat_bonus=_stat_bonus_summary(hero, stats),
        personal_quirk=_sheet_trait(hero.personal_quirk, definitions, kind="personal"),
        earned_quirks=tuple(
            trait
            for slot in hero.earned_quirk_slots
            if (
                trait := _sheet_trait(
                    slot.quirk_id,
                    definitions,
                    kind="earned",
                    stability=slot.stability,
                )
            )
            is not None
        ),
        strain_marks=tuple(
            trait
            for mark in sorted(hero.strain_marks, key=lambda mark: mark.value)
            if (
                trait := _sheet_trait(mark.value, definitions, kind="strain", stability="")
            )
            is not None
        ),
        fresh_memories=tuple(
            HeroSheetFreshMemoryView(
                family_id=memory.family_id,
                name=memory.display_name,
                intensity=memory.intensity,
                tags=tuple(memory.tags),
                source_summary=_player_memory_summary(memory.source_summary),
                pending_manifestation=memory.pending_manifestation,
            )
            for memory in sorted(
                hero.fresh_memories,
                key=lambda memory: (memory.refreshed_order, memory.created_order),
                reverse=True,
            )
        ),
        permanent_memories=tuple(
            HeroSheetMemoryEntryView(
                summary=memory.summary,
                kind=memory.kind,
                expedition_id=memory.expedition_id,
                dungeon_id=memory.dungeon_id,
                node_id=memory.node_id or "",
                encounter_id=memory.encounter_id or "",
            )
            for memory in memories
        ),
        career_signals=tuple(
            HeroSheetSignalView(_signal_label(signal_id), score)
            for signal_id, score in sorted(
                hero.career_signals.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if score
        ),
        available_kits=tuple(
            item for item in _gear_item_views(company, definitions) if item.owned_count > 0
        ),
        can_manage_gear=can_manage_gear,
        gear_manage_reason=gear_manage_reason,
    )

def _gear_item_views(
    company: CompanyState,
    definitions: GameDefinitions,
) -> tuple[GearItemView, ...]:
    entries: list[GearItemView] = []
    for gear in definitions.gear.values():
        owned = company.gear_inventory.get(gear.id, 0)
        available = available_gear_count(company, gear.id)
        equipped = max(0, owned - available)
        unavailable = gear_unavailable_reason(company, definitions, gear.id)
        if owned:
            state = "owned"
        elif gear.cost is None:
            state = "reward"
        elif not unavailable:
            state = "available"
        elif unavailable.startswith(("Complete ", "Find ")):
            state = "locked"
        else:
            state = "unavailable"
        entries.append(
            GearItemView(
                gear_id=gear.id,
                name=gear.name,
                description=gear.description,
                owned_count=owned,
                equipped_count=equipped,
                available_count=available,
                cost=gear.cost,
                state=state,
                effect_summary=gear_effect_summary(gear),
                unavailable_reason=unavailable,
            )
        )
    return tuple(entries)

def _gear_hero_views(
    company: CompanyState,
    definitions: GameDefinitions,
) -> tuple[GearHeroView, ...]:
    entries: list[GearHeroView] = []
    for hero in living_roster(company):
        gear_id = hero.equipped_gear_id or ""
        gear = definitions.gear.get(gear_id) if gear_id else None
        entries.append(
            GearHeroView(
                hero_id=hero.hero_id,
                name=hero.name,
                class_id=hero.class_id,
                equipped_gear_id=gear_id,
                equipped_gear_name=gear.name if gear is not None else "",
                condition=_hero_condition(hero, definitions),
            )
        )
    return tuple(entries)

def build_recruit_offers_view(
    company: CompanyState,
    definitions: GameDefinitions,
    offers: Sequence[RecruitChoice],
) -> RecruitOffersView:
    cost = definitions.town.recruit_cost
    offer_views: list[RecruitOfferView] = []
    for offer in offers:
        name = offer.name
        class_id = offer.class_id
        hero_class = definitions.hero_classes.get(class_id)
        offer_views.append(
            RecruitOfferView(
                name=name,
                class_id=class_id,
                class_name=hero_class.name if hero_class is not None else _trait_label(class_id),
                background=offer.background,
                motive=offer.motive,
                cost=cost,
            )
        )
    return RecruitOffersView(
        reputation=company.reputation,
        coin=company.coin,
        roster_count=len(company.roster),
        roster_cap=effective_roster_cap(company, definitions),
        offers=tuple(offer_views),
        actions=ActionProvider.recruit_offer_actions(company, definitions, offers),
    )

def preview_assign_hero(
    slots: Mapping[FormationSlot, str | None],
    roster_by_id: Mapping[str, HeroState],
    hero_id: str,
    target_slot: FormationSlot,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """Simulate assign_active_hero swap/displace for before/after previews."""
    before = _formation_slot_summaries(slots, roster_by_id)
    after_slots = dict(slots)
    old_slot: FormationSlot | None = None
    for slot, occupant_id in after_slots.items():
        if occupant_id == hero_id:
            old_slot = slot
            after_slots[slot] = None
            break
    displaced_id = after_slots.get(target_slot)
    after_slots[target_slot] = hero_id
    if displaced_id is not None and old_slot is not None:
        after_slots[old_slot] = displaced_id
    elif displaced_id is not None:
        after_slots[target_slot] = hero_id
    after = _formation_slot_summaries(after_slots, roster_by_id)
    return before, after

@dataclass(frozen=True)
class _TownProtector:
    hero_id: str

    def can_protect(self) -> bool:
        return True

def _town_protectors(roster_by_id: Mapping[str, HeroState]) -> Mapping[str, _TownProtector]:
    protectors: dict[str, _TownProtector] = {}
    for hero_id, hero in roster_by_id.items():
        if hero.life_state == LifeState.ALIVE and hero.hp > 0:
            protectors[hero_id] = _TownProtector(hero_id=hero_id)
    return protectors

def _formation_protection_line(
    slot: FormationSlot,
    formation: Formation,
    protectors: Mapping[str, _TownProtector],
    roster_by_id: Mapping[str, HeroState],
) -> str:
    if is_back(slot):
        protector_id = formation.protector_for(slot, protectors)
        if protector_id is not None and protector_id in roster_by_id:
            return f"protected by {roster_by_id[protector_id].name}"
        return ""
    front_actor_id = formation.actor_at(slot)
    if front_actor_id is None:
        return ""
    back_slot = back_slot_for(slot)
    back_actor_id = formation.actor_at(back_slot)
    if back_actor_id is None:
        return ""
    if formation.protector_for(back_slot, protectors) != front_actor_id:
        return ""
    if back_actor_id in roster_by_id:
        return f"protects {roster_by_id[back_actor_id].name}"
    return ""

def _hero_abnormal_status(
    hero: HeroState,
    definitions: GameDefinitions | None = None,
) -> str:
    pieces: list[str] = []
    statuses = ", ".join(
        status
        for status in _life_state_labels(hero.life_state.value)
        if status != "ready"
    )
    if statuses:
        pieces.append(statuses)
    if hero.in_surgery:
        pieces.append(IN_SURGERY_LABEL)
    if hero.mortal_wounds:
        pieces.append(mortal_wound_badge(hero.mortal_wounds))
    if hero.strain != hero.strain.STEADY:
        pieces.append(f"Strain {hero.strain.name.title()}")
    if hero.strain_marks:
        mark_text = ", ".join(
            _trait_label(mark.value)
            for mark in sorted(hero.strain_marks, key=lambda item: item.value)
        )
        pieces.append(f"Marks {mark_text}")
    return ", ".join(pieces)

def hero_protection_line(
    company: CompanyState,
    hero_id: str,
) -> str:
    roster_by_id = {hero.hero_id: hero for hero in company.roster}
    hero = roster_by_id.get(hero_id)
    if hero is None:
        return ""
    formation = Formation.from_mapping(company.active_party_slots)
    protectors = _town_protectors(roster_by_id)
    slot = hero.formation_slot
    return _formation_protection_line(slot, formation, protectors, roster_by_id)

def _hero_entry(
    hero: HeroState,
    memories: Sequence[HeroMemoryEntry] = (),
    definitions: GameDefinitions | None = None,
) -> HeroListEntry:
    stats = effective_hero_stats(hero, definitions)
    gear = (
        definitions.gear.get(hero.equipped_gear_id)
        if (definitions is not None and hero.equipped_gear_id is not None)
        else None
    )
    return HeroListEntry(
        hero_id=hero.hero_id,
        name=hero.name,
        class_id=hero.class_id,
        slot=hero.formation_slot.value,
        hp=hero.hp,
        max_hp=stats.max_hp,
        effort=hero.effort,
        max_effort=stats.max_effort,
        mortal_wounds=hero.mortal_wounds,
        morale=hero.morale.name.title(),
        strain=hero.strain.name.title(),
        life_state=hero.life_state.value,
        personal_quirk=_trait_label(hero.personal_quirk),
        quirks=tuple(_trait_label(quirk) for quirk in hero.quirks),
        strain_marks=tuple(
            _trait_label(mark.value)
            for mark in sorted(hero.strain_marks, key=lambda mark: mark.value)
        ),
        memory_count=len(memories),
        latest_memory=memories[-1].summary if memories else "",
        equipped_gear=gear.name if gear is not None else "",
        stat_bonus=_stat_bonus_summary(hero, stats),
    )

def _hero_memories(
    company: CompanyState,
    hero_id: str,
) -> tuple[HeroMemoryEntry, ...]:
    return tuple(memory for memory in company.hero_memories if memory.hero_id == hero_id)

def _sheet_trait(
    trait_id: str | None,
    definitions: GameDefinitions,
    *,
    kind: str,
    stability: str = "",
) -> HeroSheetTraitView | None:
    if not trait_id:
        return None
    trait = definitions.traits.get(trait_id)
    return HeroSheetTraitView(
        trait_id=trait_id,
        name=trait.name if trait is not None else _trait_label(trait_id),
        kind=kind,
        description=trait.description if trait is not None else "",
        positive_text=trait.positive_text if trait is not None else "",
        negative_text=trait.negative_text if trait is not None else "",
        stability=stability,
    )

def _hero_roster_state(
    hero_id: str,
    active_ids: set[str],
    reserve_ids: set[str],
) -> str:
    if hero_id in active_ids:
        return "Active"
    if hero_id in reserve_ids:
        return "Reserve"
    return "Memorial"

def _signal_label(signal_id: str) -> str:
    label = {
        "killing_blow": "Killing Blows",
        "marked_execution": "Marked Executions",
        "relic_greed": "Relic Greed",
        "maze_thread": "Maze Thread",
        "breach_witness": "Breach Witness",
        "field_treatment": "Field Treatment",
        "morale_rally": "Morale Rally",
        "shaken_survival": "Shaken Survival",
        "downed_survival": "Downed Survival",
        "broken_survival": "Broken Survival",
        "ally_downed_witnessed": "Ally Downed Witnesses",
        "frost_shock": "Frost Shock",
        "tag:combat": "combat",
        "tag:kill": "killing",
        "tag:maze": "Maze exposure",
        "tag:greed": "greed",
        "tag:loot": "loot",
        "tag:route": "route pressure",
        "tag:breach": "breach exposure",
        "tag:marked": "marked targets",
        "tag:support": "support",
        "tag:morale": "morale pressure",
        "tag:healing": "healing",
        "tag:downed": "downed allies",
        "tag:ally": "ally pressure",
        "tag:survival": "survival",
        "tag:frozen": "freezing shock",
        "tag:shock": "shock",
        "tag:final_kill": "fight-ending kills",
        "tag:basic": "basic finishes",
        "tag:steady": "calm killing",
        "tag:shaken": "desperate killing",
        "tag:fractured": "fractured-line kills",
        "tag:wounded": "wounded killing",
        "tag:low_hp": "finisher targets",
        "tag:effort_kill": "costly kills",
        "tag:boss": "boss kills",
    }.get(signal_id)
    if label is not None:
        return label
    if signal_id.startswith("tag:"):
        return _trait_label(signal_id.removeprefix("tag:")).lower()
    return _trait_label(signal_id)

def _player_memory_summary(summary: str) -> str:
    if not summary:
        return ""
    replacements = {
        "shallow_cave_breach": "Shallow Cave Breach",
    }
    for raw, label in replacements.items():
        summary = summary.replace(raw, label)
    return summary

def _hero_condition(
    hero: HeroState,
    definitions: GameDefinitions | None = None,
) -> str:
    stats = effective_hero_stats(hero, definitions)
    statuses = ", ".join(_life_state_labels(hero.life_state.value))
    pieces = [
        f"{hero.hp}/{stats.max_hp} HP",
        f"{hero.effort}/{stats.max_effort} Effort",
        mortal_wound_badge(hero.mortal_wounds),
    ]
    if statuses != "ready":
        pieces.append(statuses)
    if hero.strain != hero.strain.STEADY:
        pieces.append(f"Strain {hero.strain.name.title()}")
    if hero.strain_marks:
        mark_text = ", ".join(
            _trait_label(mark.value)
            for mark in sorted(hero.strain_marks, key=lambda item: item.value)
        )
        pieces.append(f"Marks {mark_text}")
    return ", ".join(pieces)

def _stat_bonus_summary(hero: HeroState, stats: Any) -> str:
    pieces: list[str] = []
    if stats.max_hp != hero.max_hp:
        pieces.append(f"Max HP +{stats.max_hp - hero.max_hp}")
    if stats.max_effort != hero.max_effort:
        pieces.append(f"Max Effort +{stats.max_effort - hero.max_effort}")
    if stats.accuracy != hero.accuracy:
        pieces.append(f"Accuracy +{stats.accuracy - hero.accuracy}")
    if stats.damage != hero.damage:
        pieces.append(f"Damage +{stats.damage - hero.damage}")
    return ", ".join(pieces)

def _trait_label(trait_id: str | None) -> str:
    if not trait_id:
        return ""
    return trait_id.replace("_", " ").title()
