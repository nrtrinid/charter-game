"""Canonical app actions and HCI metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from game.campaign.company import CompanyState
from game.campaign.gear import (
    available_gear_count,
    gear_effect_summary,
    gear_unavailable_reason,
)
from game.campaign.recruitment import RecruitChoice
from game.campaign.roster import active_roster, living_roster
from game.campaign.town import (
    deep_surgery_candidates,
    effective_recovery_cost,
    effective_roster_cap,
    effective_supply_cost,
    effective_surgery_cost,
    upgrade_unavailable_reason,
)
from game.content.definitions import GameDefinitions
from game.ui.wounds import mortal_wound_badge


class ScreenActionKind(StrEnum):
    GENERAL = "general"
    NAVIGATE = "navigate"
    INSPECT = "inspect"
    TRAVEL = "travel"
    TOWN = "town"
    COMBAT = "combat"
    DUNGEON = "dungeon"
    SYSTEM = "system"
    CONFIRM = "confirm"
    CANCEL = "cancel"


class ScreenActionRisk(StrEnum):
    SAFE = "safe"
    LOW = "low"
    COSTLY = "costly"
    RISKY = "risky"
    IRREVERSIBLE = "irreversible"


@dataclass(frozen=True)
class ScreenAction:
    number: str
    label: str
    value: str
    aliases: tuple[str, ...] = ()
    enabled: bool = True
    default: bool = False
    description: str = ""
    kind: ScreenActionKind | str = ScreenActionKind.GENERAL
    risk: ScreenActionRisk | str = ScreenActionRisk.SAFE
    cost: str = ""
    unavailable_reason: str = ""
    preview: str = ""
    result_hint: str = ""
    confirm: str = ""
    route_warning: bool = False


class ActionProvider:
    """Single source for player-facing action metadata."""

    @staticmethod
    def combat_commands(
        *,
        has_skills: bool,
        has_usable_skills: bool | None = None,
        has_moves: bool,
        can_delay: bool,
        can_act: bool,
        retreat_available: bool,
        actor_name: str = "",
    ) -> tuple[ScreenAction, ...]:
        actor = actor_name or "The acting hero"
        usable_skills = has_skills if has_usable_skills is None else has_usable_skills
        return (
            ScreenAction(
                "1",
                "Skills",
                "skill",
                ("s", "skills"),
                enabled=has_skills,
                default=usable_skills,
                description="Choose a skill and target.",
                kind=ScreenActionKind.COMBAT,
                risk=ScreenActionRisk.LOW,
                unavailable_reason="No legal skill is available for this actor.",
                preview=f"Open skill choices for {actor}.",
                result_hint="A selected skill still needs a target.",
            ),
            ScreenAction(
                "2",
                "Move",
                "move",
                ("m",),
                enabled=has_moves,
                default=not usable_skills and has_moves,
                description="Move to an adjacent party slot.",
                kind=ScreenActionKind.COMBAT,
                risk=ScreenActionRisk.LOW,
                unavailable_reason="No adjacent movement is available.",
                preview=f"Reposition {actor} in the 2x2 formation.",
                result_hint="Movement spends this hero turn and may change protection lanes.",
            ),
            ScreenAction(
                "3",
                "Delay",
                "delay",
                ("d", "wait"),
                enabled=can_delay,
                default=not usable_skills and not has_moves and can_delay,
                description="Act later in the current round.",
                kind=ScreenActionKind.COMBAT,
                risk=ScreenActionRisk.LOW,
                unavailable_reason="No later turn slot is available this round.",
                preview=f"{actor} waits for a later opening without spending this action.",
                result_hint="Other combatants may act before this hero's delayed turn.",
            ),
            ScreenAction(
                "4",
                "Pass",
                "pass",
                ("p",),
                enabled=can_act,
                default=not usable_skills and not has_moves and not can_delay and can_act,
                description="End this hero turn without changing state.",
                kind=ScreenActionKind.COMBAT,
                risk=ScreenActionRisk.RISKY,
                unavailable_reason="No hero can pass right now.",
                preview=f"{actor} gives up this action and lets the turn order continue.",
                result_hint="Enemies may act before this hero gets another turn.",
            ),
            ScreenAction(
                "5",
                "Retreat",
                "retreat",
                ("r",),
                enabled=retreat_available,
                description="Begin a party withdrawal that resolves at round end.",
                kind=ScreenActionKind.COMBAT,
                risk=ScreenActionRisk.RISKY,
                unavailable_reason=(
                    "Retreat is only available on a hero command during dungeon combat."
                ),
                preview=f"{actor} begins a fighting withdrawal from the current combat.",
                result_hint="Enemies with remaining turns may still act before escape.",
                confirm="Begin retreat from this combat?",
            ),
        )

    @staticmethod
    def town_services(
        company: CompanyState,
        definitions: GameDefinitions,
    ) -> tuple[ScreenAction, ...]:
        active = active_roster(company)
        roster_cap = effective_roster_cap(company, definitions)
        recovery_cost = effective_recovery_cost(company, definitions)
        surgery_cost = effective_surgery_cost(company, definitions)
        surgery_candidates = deep_surgery_candidates(company)
        roster_full = len(company.roster) >= roster_cap
        budget = budget_detail(company.coin)
        return (
            ScreenAction(
                "1",
                "Expedition",
                "expedition",
                ("expedition", "x", "begin"),
                default=True,
                description="Begin or resume the opening route",
                kind=ScreenActionKind.TRAVEL,
                risk=ScreenActionRisk.LOW,
                preview=join_detail(
                    "Leave Haven and continue the current charter route.",
                    f"Active party: {len(active)}/4",
                    budget,
                ),
                result_hint="Opens the expedition staging flow before committing to rooms.",
            ),
            ScreenAction(
                "2",
                "Formation",
                "formation",
                ("formation", "party"),
                description=f"{len(active)} active party members",
                kind=ScreenActionKind.TOWN,
                preview=f"Reassign the four active expedition slots; {len(active)} are filled.",
                result_hint="Changes who enters danger first and who can react in combat.",
            ),
            ScreenAction(
                "3",
                "Recruiting",
                "recruit",
                ("recruit", "hire"),
                enabled=not roster_full and company.coin >= definitions.town.recruit_cost,
                description=f"Cost {definitions.town.recruit_cost}",
                kind=ScreenActionKind.TOWN,
                risk=ScreenActionRisk.COSTLY,
                cost=f"{definitions.town.recruit_cost} Coin",
                unavailable_reason=join_detail(
                    "Roster is full" if roster_full else "",
                    coin_gap(company.coin, definitions.town.recruit_cost),
                ),
                preview=join_detail(
                    f"Cost {definitions.town.recruit_cost} Coin.",
                    budget,
                    "Review candidates before hiring.",
                ),
                result_hint=(
                    "Hiring adds one living roster member and lowers Coin to "
                    f"{projected_coin(company.coin, definitions.town.recruit_cost)}."
                ),
            ),
            ScreenAction(
                "4",
                "Quartermaster",
                "buy",
                ("buy", "supplies"),
                kind=ScreenActionKind.TOWN,
                risk=ScreenActionRisk.COSTLY,
                preview=join_detail(
                    "Buy route supplies with Coin.",
                    budget,
                ),
                result_hint="Supplies can unlock room actions and soften expedition risk.",
            ),
            ScreenAction(
                "5",
                "Contract Board",
                "contracts",
                ("contracts", "board"),
                kind=ScreenActionKind.TOWN,
                preview="Review the visible Act 1 contract sequence and postings.",
                result_hint="Accepting a contract changes generated breach routes.",
            ),
            ScreenAction(
                "6",
                "Upgrades",
                "upgrades",
                ("upgrades", "u"),
                kind=ScreenActionKind.TOWN,
                risk=ScreenActionRisk.COSTLY,
                preview="Spend Coin on permanent company infrastructure.",
                result_hint="Installed upgrades change town costs or company capacity.",
            ),
            ScreenAction(
                "7",
                "Gear",
                "gear",
                ("gear", "inventory", "i"),
                kind=ScreenActionKind.TOWN,
                preview="Review company kits and assign one kit per living hero.",
                result_hint="Gear changes effective combat stats when equipped.",
            ),
            ScreenAction(
                "8",
                "Roster",
                "roster",
                ("roster", "r"),
                kind=ScreenActionKind.INSPECT,
                preview="Review active, reserve, and memorialized heroes.",
            ),
            ScreenAction(
                "9",
                "Back to Main",
                "back",
                ("back", "b"),
                kind=ScreenActionKind.NAVIGATE,
            ),
            ScreenAction(
                "10",
                "Recovery Ward",
                "recover",
                ("recover", "recovery"),
                enabled=company.coin >= recovery_cost,
                description=f"Cost {recovery_cost}",
                kind=ScreenActionKind.TOWN,
                risk=ScreenActionRisk.COSTLY,
                cost=f"{recovery_cost} Coin",
                unavailable_reason=coin_gap(company.coin, recovery_cost),
                preview=join_detail(
                    f"Cost {recovery_cost} Coin.",
                    budget,
                    "Restore HP and Effort, and clear Downed.",
                ),
                result_hint=("Living heroes recover for the next route; Mortal Wounds remain."),
            ),
            ScreenAction(
                "11",
                "Deep Surgery",
                "deep_surgery",
                ("deep_surgery", "surgery"),
                enabled=bool(surgery_candidates) and company.coin >= surgery_cost,
                description=f"Cost {surgery_cost}",
                kind=ScreenActionKind.TOWN,
                risk=ScreenActionRisk.COSTLY,
                cost=f"{surgery_cost} Coin",
                unavailable_reason=(
                    coin_gap(company.coin, surgery_cost)
                    if surgery_candidates
                    else "No wounded heroes need deep surgery."
                ),
                preview=join_detail(
                    f"Cost {surgery_cost} Coin.",
                    budget,
                    "Remove one Mortal Wound from a living hero.",
                ),
                result_hint=(
                    "Removes 1 Mortal Wound. Hero sits out the next expedition. "
                    "Use Recovery Ward for HP."
                ),
            ),
            ScreenAction(
                "12",
                "Ledger",
                "ledger",
                ("ledger", "l"),
                kind=ScreenActionKind.INSPECT,
                preview="Review company state, contracts, and resources.",
            ),
            ScreenAction(
                "13",
                "Memorial",
                "memorial",
                ("memorial",),
                kind=ScreenActionKind.INSPECT,
                preview="Read the names and last records of the dead.",
            ),
        )

    @staticmethod
    def deep_surgery_hero_actions(
        company: CompanyState,
        definitions: GameDefinitions,
    ) -> tuple[ScreenAction, ...]:
        cost = effective_surgery_cost(company, definitions)
        actions: list[ScreenAction] = []
        for index, hero in enumerate(deep_surgery_candidates(company), start=1):
            enabled = company.coin >= cost
            actions.append(
                ScreenAction(
                    str(index),
                    f"{hero.name} ({mortal_wound_badge(hero.mortal_wounds)})",
                    f"surgery:{hero.hero_id}",
                    (hero.name.lower(), hero.hero_id),
                    enabled=enabled,
                    description=hero.class_id.replace("_", " ").title(),
                    kind=ScreenActionKind.TOWN,
                    risk=ScreenActionRisk.COSTLY,
                    cost=f"{cost} Coin",
                    unavailable_reason=coin_gap(company.coin, cost),
                    preview=join_detail(
                        f"Treat {hero.name} for {cost} Coin.",
                        f"Current wounds: {mortal_wound_badge(hero.mortal_wounds)}.",
                    ),
                    result_hint=(
                        "Removes 1 Mortal Wound. Hero is In Surgery until the next "
                        "expedition returns."
                    ),
                )
            )
        actions.append(
            ScreenAction(
                str(len(actions) + 1),
                "Back",
                "back",
                ("back", "b"),
                kind=ScreenActionKind.NAVIGATE,
            )
        )
        return tuple(actions)

    @staticmethod
    def contract_board_actions(entries: Sequence[Any]) -> tuple[ScreenAction, ...]:
        actions = [
            ScreenAction(
                str(index),
                _contract_action_label(entry),
                f"accept:{entry.contract_id}",
                (entry.contract_id,),
                enabled=entry.state == "available",
                description=_contract_action_description(entry),
                kind=ScreenActionKind.TOWN,
                risk=ScreenActionRisk.LOW,
                unavailable_reason=entry.unavailable_reason,
                preview=entry.summary,
                result_hint=_contract_action_result_hint(entry),
            )
            for index, entry in enumerate(entries, start=1)
        ]
        actions.append(
            ScreenAction(
                str(len(actions) + 1),
                "Back to Haven",
                "back",
                ("back", "b"),
                kind=ScreenActionKind.NAVIGATE,
            )
        )
        return tuple(actions)

    @staticmethod
    def relic_broker_actions(
        company: CompanyState,
        definitions: GameDefinitions,
    ) -> tuple[ScreenAction, ...]:
        actions: list[ScreenAction] = []
        for index, loot in enumerate(definitions.loot.values(), start=1):
            owned = company.inventory.get(loot.id, 0)
            if loot.sell_price is not None and loot.sell_price > 0:
                actions.append(
                    ScreenAction(
                        str(index),
                        f"Sell {loot.name}",
                        f"sell_loot:{loot.id}",
                        (loot.id, "sell"),
                        enabled=owned > 0,
                        description=join_detail(
                            f"Owned {owned}",
                            f"Pays {loot.sell_price} Coin",
                        ),
                        kind=ScreenActionKind.TOWN,
                        risk=ScreenActionRisk.LOW,
                        unavailable_reason=(
                            f"No {loot.name.lower()} in company inventory."
                            if owned <= 0
                            else ""
                        ),
                        preview=join_detail(
                            loot.description,
                            f"Pays {loot.sell_price} Coin.",
                            budget_detail(company.coin),
                        ),
                        result_hint=(
                            f"Coin after sale: {projected_coin(company.coin, -loot.sell_price)}"
                        ),
                    )
                )
            elif loot.turn_in_flag:
                filed = company.flags.get(loot.turn_in_flag, False)
                actions.append(
                    ScreenAction(
                        str(index),
                        f"File {loot.name}",
                        f"turn_in_loot:{loot.id}",
                        (loot.id, "file"),
                        enabled=owned > 0 and not filed,
                        description=join_detail(
                            f"Owned {owned}",
                            "Already filed" if filed else "Posts new charter work",
                        ),
                        kind=ScreenActionKind.TOWN,
                        unavailable_reason=join_detail(
                            "That proof has already been filed." if filed else "",
                            (
                                f"No {loot.name.lower()} in company inventory."
                                if owned <= 0
                                else ""
                            ),
                        ),
                        preview=join_detail(
                            loot.description,
                            "File proof with Haven's relic clerk.",
                        ),
                        result_hint="Filing consumes the relic and may post new road contracts.",
                    )
                )
        return tuple(actions) + (
            ScreenAction(
                str(len(actions) + 1),
                "Back",
                "back",
                ("back", "b"),
                kind=ScreenActionKind.NAVIGATE,
            ),
        )

    @staticmethod
    def supply_shop_actions(
        company: CompanyState,
        definitions: GameDefinitions,
    ) -> tuple[ScreenAction, ...]:
        actions: list[ScreenAction] = []
        for index, supply in enumerate(definitions.supplies.catalog.values(), start=1):
            cost = effective_supply_cost(company, definitions, supply.id)
            actions.append(
                ScreenAction(
                    str(index),
                    supply.name,
                    supply.id,
                    (supply.id,),
                    enabled=company.coin >= cost,
                    description=join_detail(
                        f"Owned {company.supplies.get(supply.id, 0)}",
                        f"Cost {cost}",
                    ),
                    kind=ScreenActionKind.TOWN,
                    risk=ScreenActionRisk.COSTLY,
                    cost=f"{cost} Coin",
                    unavailable_reason=coin_gap(company.coin, cost),
                    preview=join_detail(
                        supply.description or f"Buy one {supply.name.lower()}.",
                        f"Cost {cost} Coin.",
                        budget_detail(company.coin),
                    ),
                    result_hint=join_detail(
                        f"Owned after purchase: {company.supplies.get(supply.id, 0) + 1}",
                        f"Coin after purchase: {projected_coin(company.coin, cost)}",
                    ),
                )
            )
        return tuple(actions) + (
            ScreenAction(
                str(len(definitions.supplies.catalog) + 1),
                "Back",
                "back",
                ("back", "b"),
                kind=ScreenActionKind.NAVIGATE,
            ),
        )

    @staticmethod
    def upgrade_actions(
        company: CompanyState,
        definitions: GameDefinitions,
    ) -> tuple[ScreenAction, ...]:
        actions = [
            ScreenAction(
                str(index),
                upgrade.name,
                f"upgrade:{upgrade.id}",
                (upgrade.id,),
                enabled=not upgrade_unavailable_reason(company, definitions, upgrade.id),
                description=join_detail(
                    "Installed" if upgrade.id in company.purchased_upgrade_ids else "",
                    f"Cost {upgrade.cost}",
                ),
                kind=ScreenActionKind.TOWN,
                risk=ScreenActionRisk.COSTLY,
                cost=f"{upgrade.cost} Coin",
                unavailable_reason=upgrade_unavailable_reason(
                    company,
                    definitions,
                    upgrade.id,
                ),
                preview=join_detail(
                    upgrade.description,
                    _upgrade_effect_description(upgrade.effects),
                    budget_detail(company.coin),
                ),
                result_hint=join_detail(
                    "Permanent company upgrade.",
                    f"Coin after purchase: {projected_coin(company.coin, upgrade.cost)}",
                ),
            )
            for index, upgrade in enumerate(definitions.town.upgrades.values(), start=1)
        ]
        actions.append(
            ScreenAction(
                str(len(actions) + 1),
                "Back",
                "back",
                ("back", "b"),
                kind=ScreenActionKind.NAVIGATE,
            )
        )
        return tuple(actions)

    @staticmethod
    def gear_actions(
        company: CompanyState,
        definitions: GameDefinitions,
        *,
        can_manage: bool,
        can_purchase: bool,
        manage_reason: str = "",
        purchase_reason: str = "",
    ) -> tuple[ScreenAction, ...]:
        actions: list[ScreenAction] = []
        if can_purchase:
            for gear in definitions.gear.values():
                gear_unavailable = gear_unavailable_reason(company, definitions, gear.id)
                purchase_enabled = gear.cost is not None and not gear_unavailable
                actions.append(
                    ScreenAction(
                        str(len(actions) + 1),
                        f"Buy {gear.name}",
                        f"gear:buy:{gear.id}",
                        (f"buy {gear.id}",),
                        enabled=purchase_enabled,
                        description=(
                            f"Cost {gear.cost}" if gear.cost is not None else "Reward only"
                        ),
                        kind=ScreenActionKind.TOWN,
                        risk=ScreenActionRisk.COSTLY,
                        cost=f"{gear.cost} Coin" if gear.cost is not None else "",
                        unavailable_reason=gear_unavailable,
                        preview=join_detail(
                            gear.description,
                            gear_effect_summary(gear),
                            f"Owned {company.gear_inventory.get(gear.id, 0)}",
                            budget_detail(company.coin) if gear.cost is not None else "",
                        ),
                        result_hint=f"Adds one {gear.name} to company gear inventory.",
                    )
                )
        actions.append(
            ScreenAction(
                str(len(actions) + 1),
                "Back",
                "back",
                ("back", "b"),
                kind=ScreenActionKind.NAVIGATE,
            )
        )
        return tuple(actions)

    @staticmethod
    def hero_gear_actions(
        company: CompanyState,
        definitions: GameDefinitions,
        hero_id: str,
        *,
        can_manage: bool,
        manage_reason: str = "",
    ) -> tuple[ScreenAction, ...]:
        hero = next(
            (candidate for candidate in living_roster(company) if candidate.hero_id == hero_id),
            None,
        )
        if hero is None:
            return (
                ScreenAction(
                    "1",
                    "Back",
                    "back",
                    ("back", "b"),
                    kind=ScreenActionKind.NAVIGATE,
                ),
            )

        actions: list[ScreenAction] = []
        if hero.equipped_gear_id is not None:
            equipped_gear = definitions.gear.get(hero.equipped_gear_id)
            gear_name = equipped_gear.name if equipped_gear is not None else hero.equipped_gear_id
            actions.append(
                ScreenAction(
                    str(len(actions) + 1),
                    f"Unequip {gear_name}",
                    f"gear:unequip:{hero.hero_id}",
                    (f"unequip {hero.hero_id}",),
                    enabled=can_manage,
                    description=gear_name,
                    kind=ScreenActionKind.TOWN,
                    unavailable_reason=manage_reason,
                    preview=f"Remove {gear_name} from {hero.name}.",
                    result_hint="Returns the kit to available company gear.",
                )
            )
        for gear in definitions.gear.values():
            owned = company.gear_inventory.get(gear.id, 0)
            already_equipped = hero.equipped_gear_id == gear.id
            if owned <= 0 and not already_equipped:
                continue
            available = available_gear_count(company, gear.id)
            unavailable = manage_reason
            if not unavailable and available <= 0 and not already_equipped:
                unavailable = "No available copy."
            actions.append(
                ScreenAction(
                    str(len(actions) + 1),
                    f"Equip {gear.name}",
                    f"gear:equip:{hero.hero_id}:{gear.id}",
                    (f"equip {gear.id}", f"equip {hero.hero_id} {gear.id}"),
                    enabled=can_manage and (available > 0 or already_equipped),
                    description=gear_effect_summary(gear),
                    kind=ScreenActionKind.TOWN,
                    unavailable_reason=unavailable,
                    preview=join_detail(
                        f"Assign {gear.name} to {hero.name}.",
                        f"Available copies: {available}.",
                    ),
                    result_hint="Equipped gear changes effective combat stats.",
                )
            )
        actions.append(
            ScreenAction(
                str(len(actions) + 1),
                "Back",
                "back",
                ("back", "b"),
                kind=ScreenActionKind.NAVIGATE,
            )
        )
        return tuple(actions)

    @staticmethod
    def recruit_offer_actions(
        company: CompanyState,
        definitions: GameDefinitions,
        offers: Sequence[RecruitChoice],
    ) -> tuple[ScreenAction, ...]:
        roster_cap = effective_roster_cap(company, definitions)
        roster_full = len(company.roster) >= roster_cap
        cost = definitions.town.recruit_cost
        actions = [
            ScreenAction(
                str(index),
                offer.name,
                str(index - 1),
                (offer.name.lower().replace(" ", "_"),),
                enabled=not roster_full and company.coin >= cost,
                description=join_detail(offer.class_id, f"Cost {cost}"),
                kind=ScreenActionKind.TOWN,
                risk=ScreenActionRisk.COSTLY,
                cost=f"{cost} Coin",
                unavailable_reason=join_detail(
                    "Roster is full" if roster_full else "",
                    coin_gap(company.coin, cost),
                ),
                preview=join_detail(
                    f"Cost {cost} Coin.",
                    budget_detail(company.coin),
                    offer.background,
                    offer.motive,
                ),
                result_hint=join_detail(
                    f"Hire {offer.name} as {offer.class_id}.",
                    f"Coin after hire: {projected_coin(company.coin, cost)}",
                    f"Roster size: {len(company.roster) + 1}/{roster_cap}",
                ),
            )
            for index, offer in enumerate(offers, start=1)
        ]
        actions.append(
            ScreenAction(
                str(len(actions) + 1),
                "Back",
                "back",
                ("back", "b"),
                kind=ScreenActionKind.NAVIGATE,
            )
        )
        return tuple(actions)

    @staticmethod
    def dungeon_actions(
        room_actions: Sequence[Any],
        exits: Sequence[Any],
        *,
        movement_enabled: bool,
        pending_combat: bool,
        current_map_id: str,
        current_safe_return: bool,
        can_enter_generated_maze: bool = False,
        can_retrace_generated_maze: bool = False,
        can_withdraw_generated_maze: bool = False,
        unstable_frontier_exit_ids: tuple[str, ...] = (),
        previous_node_id: str = "",
        current_node_id: str = "",
        opening_breach_at_breach: bool = False,
    ) -> tuple[ScreenAction, ...]:
        actions: list[ScreenAction] = []
        for action_view in room_actions:
            actions.append(
                ScreenAction(
                    str(len(actions) + 1),
                    str(action_view.label),
                    f"action:{action_view.action_id}",
                    (str(action_view.action_id),),
                    enabled=action_view.state == "available",
                    description=room_action_description(action_view),
                    kind=ScreenActionKind.DUNGEON,
                    risk=ScreenActionRisk.COSTLY if action_view.cost else ScreenActionRisk.LOW,
                    cost=quantity_detail(action_view.cost),
                    unavailable_reason=room_action_state_reason(action_view),
                    preview=room_action_preview(action_view),
                    result_hint=room_action_result_hint(action_view),
                )
            )
        if opening_breach_at_breach:
            actions.append(
                ScreenAction(
                    str(len(actions) + 1),
                    "Return to Haven",
                    "return_to_haven",
                    ("r",),
                    default=True,
                    kind=ScreenActionKind.TRAVEL,
                    risk=ScreenActionRisk.LOW,
                    preview="Report the breach and bring the company back to Haven.",
                    result_hint="Ends the expedition with proof and survivors.",
                )
            )
            actions.append(
                ScreenAction(
                    str(len(actions) + 1),
                    "Descend to Maze Depth 1",
                    "descend_maze_depth_1",
                    ("d",),
                    kind=ScreenActionKind.TRAVEL,
                    risk=ScreenActionRisk.RISKY,
                    preview="Push beyond the cave into the first impossible rooms.",
                    result_hint="Requires confirmation before the company descends.",
                    confirm="Descend into Maze Depth 1?",
                )
            )
            actions.append(
                ScreenAction(
                    str(len(actions) + 1),
                    "Back",
                    "back",
                    ("b",),
                    enabled=False,
                    description="Resolve Return or Descend before leaving this breach.",
                    kind=ScreenActionKind.NAVIGATE,
                    unavailable_reason="Choose Return to Haven or Descend to Maze Depth 1.",
                    preview="The breach decision must be resolved from this room.",
                )
            )
            return tuple(actions)
        for exit_view in exits:
            unstable = str(exit_view.node_id) in unstable_frontier_exit_ids
            detail = route_action_description(exit_view, current_map_id=current_map_id)
            if unstable:
                detail = join_detail(detail, "unstable passage")
            actions.append(
                ScreenAction(
                    str(len(actions) + 1),
                    route_action_label(exit_view, previous_node_id=previous_node_id),
                    str(exit_view.node_id),
                    (str(exit_view.node_id),),
                    enabled=movement_enabled,
                    default=not any(action.enabled for action in actions) and movement_enabled,
                    description=detail,
                    kind=ScreenActionKind.TRAVEL,
                    risk=ScreenActionRisk.RISKY if unstable else dungeon_route_risk(exit_view),
                    unavailable_reason=(
                        "Clear the pending room combat first."
                        if pending_combat
                        else "Clear or inspect this room before moving."
                    ),
                    preview=(
                        "Unstable passage — the room is not mapped yet."
                        if unstable
                        else route_action_preview(
                            exit_view,
                            detail=detail,
                            pending_combat=pending_combat,
                            current_safe_return=current_safe_return,
                            previous_node_id=previous_node_id,
                        )
                    ),
                    result_hint=(
                        "Open an unmapped room beyond the frontier."
                        if unstable
                        else route_action_result_hint(
                            exit_view,
                            previous_node_id=previous_node_id,
                        )
                    ),
                    route_warning=route_action_warns_player(exit_view),
                )
            )
        if can_enter_generated_maze:
            actions.append(
                ScreenAction(
                    str(len(actions) + 1),
                    "Enter The Breach",
                    "enter_generated_maze",
                    ("enter", "breach"),
                    enabled=not pending_combat,
                    description="Step through this breach into a temporary Maze route.",
                    kind=ScreenActionKind.DUNGEON,
                    risk=ScreenActionRisk.RISKY,
                    unavailable_reason="Resolve the pending room combat before entering.",
                    preview="Open a breach route that extends as the company travels deeper.",
                    result_hint="Unstable exits appear at the frontier. Retrace to withdraw.",
                )
            )
        if can_retrace_generated_maze:
            actions.append(
                ScreenAction(
                    str(len(actions) + 1),
                    "Retrace Steps",
                    "retrace_generated_maze",
                    ("retrace", "steps", "marks"),
                    enabled=movement_enabled,
                    description="Follow marks back to the Maze threshold. Keeps the route active.",
                    kind=ScreenActionKind.TRAVEL,
                    risk=ScreenActionRisk.LOW,
                    unavailable_reason="Resolve combat and clear the room before retracing.",
                    preview="Follow marks back to the Maze threshold. Keeps the route active.",
                    result_hint="Returns to the generated route entry without ending the route.",
                )
            )
        if can_withdraw_generated_maze:
            actions.append(
                ScreenAction(
                    str(len(actions) + 1),
                    "Withdraw to Shallow Cave",
                    "withdraw_generated_maze",
                    ("withdraw", "cave", "shallow"),
                    enabled=movement_enabled,
                    description=(
                        "Company decision: leave the generated route for the Shallow Cave "
                        "breach and collapse this run."
                    ),
                    kind=ScreenActionKind.TRAVEL,
                    risk=ScreenActionRisk.LOW,
                    unavailable_reason="Resolve combat and clear the threshold before withdrawing.",
                    preview=(
                        "Withdraw with whatever proof and survivors the company still has. "
                        "The breach remains for later exploitation."
                    ),
                    result_hint="Keeps the expedition active at the breach staging room.",
                )
            )
        if current_safe_return:
            return_label, return_preview, return_hint = _dungeon_return_action_copy(
                current_node_id=current_node_id,
                current_map_id=current_map_id,
            )
            actions.append(
                ScreenAction(
                    str(len(actions) + 1),
                    return_label,
                    "return",
                    ("r",),
                    enabled=not pending_combat,
                    description="Leave the dungeon from this safe return room.",
                    kind=ScreenActionKind.TRAVEL,
                    risk=ScreenActionRisk.LOW,
                    unavailable_reason="Resolve the pending room combat before returning.",
                    preview=return_preview,
                    result_hint=return_hint,
                )
            )
        actions.append(
            ScreenAction(
                str(len(actions) + 1),
                "Back",
                "back",
                ("b",),
                enabled=False,
                description="Expeditions can only be left through safe return rooms.",
                kind=ScreenActionKind.NAVIGATE,
                unavailable_reason="Use Return from a safe room to leave the dungeon.",
                preview=("Back is locked during expeditions; use Return from a safe room."),
            )
        )
        return tuple(actions)

    @staticmethod
    def report_actions(
        company: CompanyState,
        definitions: GameDefinitions,
    ) -> tuple[ScreenAction, ...]:
        recovery_cost = effective_recovery_cost(company, definitions)
        return (
            ScreenAction(
                "1",
                "Back to Haven",
                "town",
                ("t",),
                default=True,
                kind=ScreenActionKind.NAVIGATE,
                preview="Return to the current Haven place view.",
                result_hint="The filed record remains available from Haven Records.",
            ),
            ScreenAction(
                "2",
                "View Roster",
                "roster",
                ("r",),
                kind=ScreenActionKind.INSPECT,
                preview="Review who survived and who needs care.",
            ),
            ScreenAction(
                "3",
                "Recovery Ward",
                "recover",
                ("w",),
                enabled=company.coin >= recovery_cost,
                description=f"Cost {recovery_cost}",
                kind=ScreenActionKind.TOWN,
                risk=ScreenActionRisk.COSTLY,
                cost=f"{recovery_cost} Coin",
                unavailable_reason=coin_gap(company.coin, recovery_cost),
                preview=join_detail(
                    f"Cost {recovery_cost} Coin.",
                    budget_detail(company.coin),
                    "Treat the living before the next route.",
                ),
                result_hint="Restores HP and Effort; Mortal Wounds remain.",
            ),
            ScreenAction(
                "4",
                "Save Company",
                "save",
                ("s",),
                kind=ScreenActionKind.SYSTEM,
                risk=ScreenActionRisk.LOW,
                preview="Write the current company state to the save slot.",
                result_hint="May ask before overwriting an existing save.",
            ),
        )

    @staticmethod
    def _regional_walk_place_actions(
        view: Any,
        *,
        start_index: int = 1,
        exits: tuple[Any, ...] | None = None,
    ) -> tuple[ScreenAction, ...]:
        exit_views = view.exits if exits is None else exits
        return tuple(
            ScreenAction(
                str(start_index + index),
                f"Walk to {exit_view.name}",
                exit_view.node_id,
                kind=ScreenActionKind.TRAVEL,
                risk=ScreenActionRisk.LOW,
                preview=join_detail(
                    f"Walk to {exit_view.name}.",
                    "No ration cost.",
                ),
                result_hint="Arrives at the linked wilderness place.",
                default=index == 0 and start_index == 1,
                route_warning=route_action_warns_player(exit_view),
            )
            for index, exit_view in enumerate(exit_views)
        )

    @staticmethod
    def _regional_charted_hop_action(
        view: Any,
        *,
        index: int,
        default: bool,
    ) -> ScreenAction | None:
        if view.anchor_kind not in {"east_gate", "shallow_cave"}:
            return None
        if not view.route_charted or not view.travel_available:
            return None
        hop_label = (
            f"Take Charted Road to {view.other_node_name}"
            if view.anchor_kind == "east_gate"
            else "Take Charted Road to East Gate"
        )
        destination_name = (
            view.other_node_name if view.anchor_kind == "east_gate" else "East Gate"
        )
        return ScreenAction(
            str(index),
            hop_label,
            view.other_node_id,
            ("c", "cave") if view.anchor_kind == "east_gate" else ("h", "haven"),
            default=default,
            kind=ScreenActionKind.TRAVEL,
            risk=ScreenActionRisk.LOW,
            cost=view.travel_cost,
            preview=join_detail(
                f"Destination: {destination_name}.",
                "Route: charted road.",
                f"Cost: {view.travel_cost}.",
            ),
            result_hint=join_detail(
                (
                    "Arrive at Shallow Cave."
                    if view.anchor_kind == "east_gate"
                    else "Arrive at East Gate."
                ),
                "Skips cleared Old Road beats.",
                "No new discoveries on this route.",
            ),
        )

    @staticmethod
    def regional_place_actions(view: Any) -> tuple[ScreenAction, ...]:
        actions: list[ScreenAction] = []
        if view.anchor_kind == "east_gate":
            charted_hop = ActionProvider._regional_charted_hop_action(
                view,
                index=1,
                default=True,
            )
            if charted_hop is not None:
                actions.append(charted_hop)
            actions.append(
                ScreenAction(
                    str(len(actions) + 1),
                    "Leave by Old Road",
                    "old_road",
                    ("r", "road"),
                    default=charted_hop is None,
                    kind=ScreenActionKind.TRAVEL,
                    risk=ScreenActionRisk.LOW,
                    preview=(
                        join_detail("Walk the Old Road.", "No ration cost.")
                        if view.route_charted
                        else "Walk the uncharted road toward Shallow Cave."
                    ),
                    result_hint=(
                        "Arrives at the linked wilderness place."
                        if view.route_charted
                        else "Starts the opening expedition from East Gate."
                    ),
                )
            )
            other_exits = tuple(
                exit_view for exit_view in view.exits if exit_view.node_id != "old_road"
            )
            if other_exits:
                actions.extend(
                    ActionProvider._regional_walk_place_actions(
                        view,
                        start_index=len(actions) + 1,
                        exits=other_exits,
                    )
                )
        elif view.anchor_kind == "shallow_cave":
            charted_hop = ActionProvider._regional_charted_hop_action(
                view,
                index=1,
                default=False,
            )
            if charted_hop is not None:
                actions.append(charted_hop)
            actions.extend(
                ActionProvider._regional_walk_place_actions(
                    view,
                    start_index=len(actions) + 1,
                )
            )
        else:
            actions.extend(ActionProvider._regional_walk_place_actions(view))
        if view.anchor_kind == "east_gate":
            actions.append(
                ScreenAction(
                    str(len(actions) + 1),
                    "Open Roadbook",
                    "survey_route",
                    ("m",),
                    kind=ScreenActionKind.INSPECT,
                    preview="Unroll the company roadbook and zoom out to charted routes.",
                    result_hint=(
                        "Opens charted fast travel between East Gate and Shallow Cave."
                        if view.route_charted
                        else "Opens charted fast travel when the cave route is known."
                    ),
                )
            )
            actions.append(
                ScreenAction(
                    str(len(actions) + 1),
                    "Enter Haven",
                    "enter_haven",
                    ("h", "town"),
                    kind=ScreenActionKind.TOWN,
                    preview="Return to Haven services and contracts.",
                    result_hint="Opens the Haven town hub.",
                )
            )
        elif view.anchor_kind == "shallow_cave":
            actions.append(
                ScreenAction(
                    str(len(actions) + 1),
                    "Enter Cave",
                    "enter_cave",
                    ("e",),
                    default=not view.exits,
                    description="Enter or resume the Shallow Cave dungeon.",
                    kind=ScreenActionKind.TRAVEL,
                    risk=ScreenActionRisk.LOW,
                    preview=join_detail(
                        "Step from the cave mouth into the active dungeon route.",
                        "Safe return exists at the entrance.",
                    ),
                    result_hint="Opens the room-by-room dungeon stage.",
                )
            )
            if view.route_charted:
                actions.append(
                    ScreenAction(
                        str(len(actions) + 1),
                        "Open Roadbook",
                        "survey_route",
                        ("m",),
                        kind=ScreenActionKind.INSPECT,
                        preview="Unroll the company roadbook and zoom out to charted routes.",
                        result_hint="Shows charted fast travel back to East Gate.",
                    )
                )
        else:
            available_room_actions = [
                room_action
                for room_action in view.room_actions
                if room_action.state == "available"
            ]
            if view.room_actions:
                actions.append(
                    ScreenAction(
                        str(len(actions) + 1),
                        "Interact",
                        "interact",
                        ("i",),
                        enabled=bool(available_room_actions),
                        kind=ScreenActionKind.DUNGEON,
                        risk=ScreenActionRisk.COSTLY
                        if any(room_action.cost for room_action in available_room_actions)
                        else ScreenActionRisk.LOW,
                        unavailable_reason="Room actions are blocked.",
                        preview="Open wilderness-specific actions and blocked requirements.",
                        result_hint=(
                            "Room actions can reveal routes, spend supplies, or claim loot."
                        ),
                    )
                )
        if view.arrival_context is not None:
            renumbered: list[ScreenAction] = [
                ScreenAction(
                    "1",
                    "Read Filed Record",
                    "latest_record",
                    ("record", "report"),
                    default=True,
                    kind=ScreenActionKind.INSPECT,
                    preview="Open the latest filed company record.",
                    result_hint="Review what changed during the expedition.",
                )
            ]
            for index, action in enumerate(actions, start=2):
                renumbered.append(
                    replace(action, number=str(index), default=False)
                )
            return tuple(renumbered)
        return tuple(actions)

    @staticmethod
    def regional_map_travel_actions(view: Any) -> tuple[ScreenAction, ...]:
        if view.anchor_kind not in {"east_gate", "shallow_cave"}:
            return ActionProvider._regional_walk_place_actions(view)
        hop = ActionProvider._regional_charted_hop_action(view, index=1, default=True)
        if hop is None:
            return ()
        return (hop,)

    @staticmethod
    def regional_map_actions(view: Any) -> tuple[ScreenAction, ...]:
        return ActionProvider.regional_place_actions(view)

    @staticmethod
    def travel_actions(view: Any) -> tuple[ScreenAction, ...]:
        return ActionProvider.regional_map_actions(view)

    @staticmethod
    def confirmation_actions(
        confirm_label: str,
        cancel_label: str,
        *,
        confirm_value: str = "confirm",
        cancel_value: str = "cancel",
        consequence: str = "",
        irreversible: bool = False,
    ) -> tuple[ScreenAction, ...]:
        return (
            ScreenAction(
                "1",
                cancel_label,
                cancel_value,
                ("b", "n"),
                default=True,
                kind=ScreenActionKind.CANCEL,
                risk=ScreenActionRisk.SAFE,
                preview=f"Avoid: {consequence}" if consequence else "Return safely.",
                result_hint="Safe default: no state changes.",
            ),
            ScreenAction(
                "2",
                confirm_label,
                confirm_value,
                ("y",),
                kind=ScreenActionKind.CONFIRM,
                risk=ScreenActionRisk.IRREVERSIBLE if irreversible else ScreenActionRisk.RISKY,
                preview=consequence,
                result_hint=consequence or confirm_label,
                confirm=consequence or confirm_label,
            ),
        )


def route_action_label(exit_view: Any, *, previous_node_id: str = "") -> str:
    return exit_view.name


def _travel_destination_label(view: Any, destination: Any) -> str:
    if getattr(destination, "route_kind", "") == "charted_approach":
        return f"Take {destination.label}"
    if getattr(destination, "route_kind", "") == "return_road":
        return "Return Along Road to Haven"
    if getattr(destination, "route_kind", "") == "road":
        return f"Travel the {destination.label}"
    if destination.location_id == "haven" and view.current_location_id != "haven":
        return "Enter Haven"
    return f"Travel to {destination.label}"


def route_action_description(exit_view: Any, *, current_map_id: str) -> str:
    cleared = bool(getattr(exit_view, "cleared", False))
    warnings = {
        "combat": "danger likely",
        "hazard": "hazard likely",
        "boss": "serious danger",
        "breach": "ominous threshold",
        "maze": "expedition risk",
    }
    direction = getattr(exit_view, "direction", "")
    pieces = [f"direction {direction.lower()}" if direction else ""]
    pieces.append(str(exit_view.node_type).replace("_", " "))
    if exit_view.map_id != current_map_id:
        pieces.append(f"enter {str(exit_view.map_id).replace('_', ' ')}")
    warning = warnings.get(str(exit_view.node_type))
    if warning is not None and not cleared:
        pieces.append(warning)
    if exit_view.safe_return:
        pieces.append("safe return")
    return join_detail(*pieces)


def route_action_preview(
    exit_view: Any,
    *,
    detail: str,
    pending_combat: bool,
    current_safe_return: bool,
    previous_node_id: str = "",
) -> str:
    if pending_combat:
        return "Blocked: clear the pending room combat before taking any route."
    return join_detail(
        detail,
        f"Destination: {exit_view.name}.",
        "Safe return is available here."
        if current_safe_return
        else "Retreat requires a safe room.",
    )


def route_action_result_hint(exit_view: Any, *, previous_node_id: str = "") -> str:
    hints = [f"Enter {exit_view.name}."]
    if exit_view.safe_return:
        hints.append("This room can return the company to Haven.")
    return join_detail(*hints)


def dungeon_route_risk(exit_view: Any) -> ScreenActionRisk:
    if bool(getattr(exit_view, "cleared", False)):
        return ScreenActionRisk.LOW
    if exit_view.node_type in {"boss", "maze", "breach"}:
        return ScreenActionRisk.RISKY
    if exit_view.node_type in {"combat", "hazard"}:
        return ScreenActionRisk.COSTLY
    return ScreenActionRisk.LOW


def route_action_warns_player(exit_view: Any) -> bool:
    if bool(getattr(exit_view, "cleared", False)):
        return False
    return str(getattr(exit_view, "node_type", "")) in {
        "boss",
        "combat",
        "hazard",
        "breach",
        "maze",
    }


def room_action_description(action: Any) -> str:
    pieces: list[str] = []
    if action.description:
        pieces.append(str(action.description))
    if action.cost:
        pieces.append(f"Cost: {quantity_detail(action.cost)}")
    if action.reward:
        pieces.append(f"Reward: {', '.join(action.reward)}")
    return join_detail(*pieces)


def room_action_preview(action: Any) -> str:
    return join_detail(
        str(action.description),
        f"Requires: {quantity_detail(action.requirements)}" if action.requirements else "",
        f"Costs: {quantity_detail(action.cost)}" if action.cost else "",
        f"Rewards: {', '.join(action.reward)}" if action.reward else "",
        room_action_state_reason(action),
    )


def room_action_result_hint(action: Any) -> str:
    if action.state != "available":
        return room_action_state_reason(action)
    rewards = ", ".join(action.reward)
    if rewards:
        return f"Resolve this room action for: {rewards}."
    if action.cost:
        return f"Spend {quantity_detail(action.cost)} to resolve this room action."
    return "Resolve this room action and update the expedition report."


def room_action_state_reason(action: Any) -> str:
    if action.state == "available":
        return ""
    if action.state == "completed":
        return "Already completed."
    if action.state == "requires cleared room":
        return "Clear this room first."
    if action.state == "missing item":
        return "Needs " + quantity_detail(action.requirements) + "."
    if action.state == "missing supplies":
        return "Needs " + quantity_detail(action.cost) + "."
    return str(action.state).replace("_", " ").capitalize() + "."


def quantity_detail(values: Mapping[str, int] | Sequence[tuple[str, int]]) -> str:
    if isinstance(values, Mapping):
        items: Sequence[tuple[str, int]] = tuple(values.items())
    else:
        items = values
    return ", ".join(
        f"{quantity} {str(item_id).replace('_', ' ')}" for item_id, quantity in sorted(items)
    )


def coin_gap(current: int, cost: int) -> str:
    if current >= cost:
        return ""
    return f"Need {cost - current} more Coin."


def budget_detail(coin: int) -> str:
    return f"Budget {coin} Coin."


def projected_coin(current: int, cost: int) -> int:
    return max(0, current - cost)


def join_detail(*parts: str) -> str:
    return " | ".join(part for part in parts if part)


def _dungeon_return_action_copy(
    *,
    current_node_id: str,
    current_map_id: str,
) -> tuple[str, str, str]:
    if current_node_id == "town_gate":
        return (
            "Return to East Gate",
            "Withdraw to East Gate and file the latest company record.",
            "Arrives at East Gate with an actionable record brief.",
        )
    if current_node_id == "shallow_cave_entrance":
        return (
            "Mark Location",
            "Note the cave mouth and regroup at the entrance staging ground.",
            "Returns to the cave entrance with the latest company record filed.",
        )
    if current_map_id == "shallow_cave" or current_node_id == "shallow_cave_room_1":
        return (
            "Withdraw to Cave Mouth",
            "Leave the dungeon for the Shallow Cave staging ground.",
            "Arrives at the cave mouth with the latest company record filed.",
        )
    if current_map_id == "old_road_wilderness":
        return (
            "Return to East Gate",
            "Follow the road back toward Haven and file the latest company record.",
            "Arrives at East Gate with an actionable record brief.",
        )
    return (
        "Return to Staging Ground",
        "Leave the dungeon from this safe return room.",
        "Arrives at the regional staging ground with the latest record filed.",
    )


def _contract_action_label(entry: Any) -> str:
    return str(entry.name)


def contract_reward_detail(entry: Any) -> str:
    pieces: list[str] = []
    if getattr(entry, "reward_reputation", 0):
        pieces.append(f"+{entry.reward_reputation} reputation")
    if getattr(entry, "coin_reward", 0):
        pieces.append(f"+{entry.coin_reward} Coin")
    return ", ".join(pieces) or "no payout"


def _contract_action_description(entry: Any) -> str:
    return join_detail(
        str(entry.state).title(),
        f"Difficulty {entry.difficulty}",
        contract_reward_detail(entry),
    )


def _contract_action_result_hint(entry: Any) -> str:
    if entry.state == "available":
        return "Accepting adds this posting to the active company charter."
    if entry.state == "active":
        return "Already active; complete the listed objective to claim payment."
    if entry.state == "completed":
        return "Already completed; no further action is needed."
    return entry.unavailable_reason or "Unlock this contract before accepting it."


def _upgrade_effect_description(effects: Any) -> str:
    pieces: list[str] = []
    if effects.roster_cap_bonus:
        pieces.append(f"Roster cap +{effects.roster_cap_bonus}")
    if effects.recovery_cost_delta:
        pieces.append(f"Recovery cost {effects.recovery_cost_delta:+d}")
    for supply_id, delta in sorted(effects.supply_cost_deltas.items()):
        pieces.append(f"{supply_id.replace('_', ' ').title()} cost {delta:+d}")
    return join_detail(*pieces)
