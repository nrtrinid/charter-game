from __future__ import annotations

from pathlib import Path

from game.app.actions import ActionProvider
from game.app.commands import (
    AcceptContract,
    AssignActiveHero,
    BuySupply,
    EquipGear,
    GenerateRecruitOffers,
    HireRecruit,
    PerformDeepSurgery,
    PurchaseGear,
    PurchaseUpgrade,
    RecoverCompany,
    SellLoot,
    TurnInLoot,
    UnequipGear,
    ViewGear,
    ViewTown,
)
from game.app.contracts import contract_board_state, contract_is_posted
from game.app.controller import AppController
from game.app.views import build_relic_broker_view
from game.campaign.company import (
    SAVE_VERSION,
    STARTING_COIN,
    CompanyState,
    ContractRecordState,
    ExpeditionReportState,
    ExpeditionSessionState,
    GeneratedDungeonState,
    MazeRecipe,
    RecruitmentOfferState,
    create_new_company,
)
from game.campaign.memory import finalize_report_memory
from game.campaign.roster import party_combatants, reserve_roster, sync_company_from_combat
from game.campaign.save_load import load_company, save_company
from game.campaign.town import clear_surgery_recovery, deep_surgery
from game.combat.combat_state import ActorStatus, Tag, apply_marked
from game.combat.formation import FormationSlot
from tests.conftest import get_definitions


def test_coin_is_spent_for_hires_recovery_and_supplies() -> None:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)
    controller.company.coin = 11
    controller.company.reputation = 5

    offers = controller.handle(GenerateRecruitOffers())
    assert offers.success
    hire = controller.handle(HireRecruit(0))
    assert hire.success
    assert controller.company.coin == 5
    assert controller.company.reputation == 5
    assert len(controller.company.roster) == 5

    recover = controller.handle(RecoverCompany())
    assert recover.success
    assert controller.company.coin == 1
    assert controller.company.reputation == 5

    buy = controller.handle(BuySupply("rations", quantity=1))
    assert buy.success
    assert controller.company.coin == 0
    assert controller.company.reputation == 5


def test_town_services_cannot_overspend_or_exceed_roster_cap() -> None:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)
    controller.company.coin = 0
    controller.company.reputation = 99

    controller.handle(GenerateRecruitOffers())
    assert not controller.handle(HireRecruit(0)).success
    assert not controller.handle(RecoverCompany()).success
    assert not controller.handle(BuySupply("rope", quantity=1)).success
    assert controller.company.coin == 0
    assert controller.company.reputation == 99

    controller.company.coin = 12
    controller.handle(GenerateRecruitOffers())
    assert controller.handle(HireRecruit(0)).success
    controller.handle(GenerateRecruitOffers())
    assert controller.handle(HireRecruit(0)).success
    controller.handle(GenerateRecruitOffers())
    assert not controller.handle(HireRecruit(0)).success
    assert len(controller.company.roster) == controller.definitions.town.roster_cap


def test_recruitment_offers_persist_and_hiring_removes_only_selected_offer(
    tmp_path: Path,
) -> None:
    definitions = get_definitions()
    controller = AppController(definitions=definitions)
    controller.company = create_new_company(definitions)
    controller.company.coin = 10

    result = controller.handle(GenerateRecruitOffers())

    assert result.success, result.error
    assert controller.company.recruitment_state.refresh_count == 1
    original_offers = [
        (offer.name, offer.class_id, offer.background, offer.motive)
        for offer in controller.company.recruitment_state.current_offers
    ]
    assert len(original_offers) == definitions.town.recruit_offer_count
    assert len({offer[0] for offer in original_offers}) == len(original_offers)
    save_path = tmp_path / "company.json"
    save_company(controller.company, save_path)
    loaded, _event = load_company(save_path)
    loaded_controller = AppController(definitions=definitions, company=loaded)

    result = loaded_controller.handle(HireRecruit(0))

    assert result.success, result.error
    remaining_offers = [
        (offer.name, offer.class_id, offer.background, offer.motive)
        for offer in loaded.recruitment_state.current_offers
    ]
    assert remaining_offers == original_offers[1:]

    loaded.recruitment_state.current_offers.append(
        RecruitmentOfferState(
            name="Placeholder Recruit",
            class_id="watchman",
            background="temporary",
            motive="test replacement",
        )
    )
    result = loaded_controller.handle(GenerateRecruitOffers())

    assert result.success, result.error
    assert loaded.recruitment_state.refresh_count == 2
    assert len(loaded.recruitment_state.current_offers) == (
        definitions.town.recruit_offer_count
    )
    assert all(
        offer.name != "Placeholder Recruit"
        for offer in loaded.recruitment_state.current_offers
    )


def test_contract_board_locks_maze_work_until_shallow_cave_is_cleared() -> None:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)

    view_result = controller.handle(ViewTown())

    assert view_result.success
    entries = {
        entry.contract_id: entry for entry in view_result.value.contract_board
    }
    assert tuple(entries) == (
        "blackwood_road_charter",
    )
    assert entries["blackwood_road_charter"].state == "active"
    assert view_result.value.objective.title == "Complete Blackwood Road Charter"
    assert not controller.handle(AcceptContract("shallow_cave_breach_scout")).success


def test_campaign_objective_tracks_act_one_steps() -> None:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)

    assert controller.handle(ViewTown()).value.objective.title == (
        "Complete Blackwood Road Charter"
    )

    controller.company.completed_contract_ids.add("blackwood_road_charter")
    controller.company.active_contract_ids.discard("blackwood_road_charter")
    controller.company.known_breaches.add("shallow_cave_breach")
    assert controller.handle(ViewTown()).value.objective.title == (
        "Accept Shallow Cave Breach Scout"
    )

    controller.company.active_contract_ids.add("shallow_cave_breach_scout")
    assert controller.handle(ViewTown()).value.objective.title == (
        "Scout Shallow Cave Breach"
    )

    controller.company.active_contract_ids.discard("shallow_cave_breach_scout")
    controller.company.completed_contract_ids.add("shallow_cave_breach_scout")
    assert controller.handle(ViewTown()).value.objective.title == (
        "Accept Breach Stalker Hunt"
    )

    controller.company.active_contract_ids.add("shallow_cave_breach_hunt")
    assert controller.handle(ViewTown()).value.objective.title == (
        "Complete Breach Stalker Hunt"
    )

    controller.company.active_contract_ids.discard("shallow_cave_breach_hunt")
    controller.company.completed_contract_ids.add("shallow_cave_breach_hunt")
    objective = controller.handle(ViewTown()).value.objective
    assert objective.title == "Charter Review Complete"
    assert objective.status == "complete"


def test_campaign_objective_shows_active_scout_progress() -> None:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)
    controller.company.completed_contract_ids.add("blackwood_road_charter")
    controller.company.active_contract_ids.discard("blackwood_road_charter")
    controller.company.known_breaches.add("shallow_cave_breach")
    controller.company.active_contract_ids.add("shallow_cave_breach_scout")
    controller.company.active_expedition = ExpeditionSessionState(
        expedition_id="expedition_test",
        dungeon_id="generated_maze_breach",
        current_node_id="maze_run_0001_room_2",
        generated_dungeon=GeneratedDungeonState(
            run_id="maze_run_0001",
            seed=1,
            source_node_id="maze_breach",
            return_node_id="maze_breach",
            dungeon_id="generated_maze_breach",
            entry_node_id="maze_run_0001_entry",
            nodes=[],
            recipe=MazeRecipe(
                pressure_id="breach_probe",
                route_length=3,
                combat_budget=0,
                hazard_budget=0,
                reward_lure=False,
                include_hunt=False,
                enemy_policy_id="none",
            ),
            visited_node_ids=[
                "maze_run_0001_entry",
                "maze_run_0001_room_1",
                "maze_run_0001_room_2",
            ],
        ),
    )

    objective = controller.handle(ViewTown()).value.objective

    assert objective.title == "Scout Shallow Cave Breach"
    assert objective.progress == "2 / 4 Maze rooms charted."


def test_campaign_objective_shows_recorded_scout_progress_after_route() -> None:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)
    controller.company.completed_contract_ids.add("blackwood_road_charter")
    controller.company.active_contract_ids.discard("blackwood_road_charter")
    controller.company.known_breaches.add("shallow_cave_breach")
    controller.company.active_contract_ids.add("shallow_cave_breach_scout")
    controller.company.contract_records["shallow_cave_breach_scout"] = (
        ContractRecordState(
            contract_id="shallow_cave_breach_scout",
            state="active",
            rooms_scouted=4,
        )
    )

    objective = controller.handle(ViewTown()).value.objective

    assert objective.progress == "4 / 4 Maze rooms charted; complete one survey action."


def test_campaign_objective_shows_hunt_progress() -> None:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)
    controller.company.completed_contract_ids.update(
        {
            "blackwood_road_charter",
            "shallow_cave_breach_scout",
        }
    )
    controller.company.active_contract_ids.discard("blackwood_road_charter")
    controller.company.active_contract_ids.add("shallow_cave_breach_hunt")
    controller.company.known_breaches.add("shallow_cave_breach")
    controller.company.active_expedition = ExpeditionSessionState(
        expedition_id="expedition_test",
        dungeon_id="generated_maze_breach",
        current_node_id="maze_run_0002_hunt_lair",
        generated_dungeon=GeneratedDungeonState(
            run_id="maze_run_0002",
            seed=2,
            source_node_id="maze_breach",
            return_node_id="maze_breach",
            dungeon_id="generated_maze_breach",
            entry_node_id="maze_run_0002_entry",
            nodes=[],
            recipe=MazeRecipe(
                pressure_id="marked_hunt",
                route_length=3,
                combat_budget=1,
                hazard_budget=0,
                reward_lure=False,
                include_hunt=True,
                enemy_policy_id="none",
            ),
            visited_node_ids=[
                "maze_run_0002_entry",
                "maze_run_0002_room_1",
                "maze_run_0002_hunt_lair",
            ],
            cleared_node_ids=["maze_run_0002_hunt_lair"],
        ),
    )

    objective = controller.handle(ViewTown()).value.objective

    assert objective.title == "Complete Breach Stalker Hunt"
    assert objective.progress == "Marked lair cleared; return with proof."


def test_contract_board_accepts_scout_then_unlocks_hunt_after_scout_completion() -> None:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)
    controller.company.completed_contract_ids.add("blackwood_road_charter")
    controller.company.active_contract_ids.discard("blackwood_road_charter")
    controller.company.known_breaches.add("shallow_cave_breach")

    view_result = controller.handle(ViewTown())

    assert view_result.success
    scout = next(
        entry
        for entry in view_result.value.contract_board
        if entry.contract_id == "shallow_cave_breach_scout"
    )
    contract_ids = {
        entry.contract_id
        for entry in view_result.value.contract_board
    }
    assert contract_ids == {
        "shallow_cave_breach_scout",
    }
    assert scout.state == "available"

    accepted = controller.handle(AcceptContract("shallow_cave_breach_scout"))

    assert accepted.success, accepted.error
    assert "shallow_cave_breach_scout" in controller.company.active_contract_ids
    record = controller.company.contract_records["shallow_cave_breach_scout"]
    assert record.state == "active"
    assert record.accepted_count == 1

    controller.company.active_contract_ids.discard("shallow_cave_breach_scout")
    controller.company.completed_contract_ids.add("shallow_cave_breach_scout")
    view_result = controller.handle(ViewTown())
    contract_ids = {
        entry.contract_id
        for entry in view_result.value.contract_board
    }
    assert contract_ids == {
        "shallow_cave_breach_hunt",
        "shallow_cave_breach_scout_posting",
    }
    hunt = next(
        entry
        for entry in view_result.value.contract_board
        if entry.contract_id == "shallow_cave_breach_hunt"
    )
    assert hunt.state == "available"

    accepted = controller.handle(AcceptContract("shallow_cave_breach_hunt"))

    assert accepted.success, accepted.error
    assert "shallow_cave_breach_hunt" in controller.company.active_contract_ids
    record = controller.company.contract_records["shallow_cave_breach_hunt"]
    assert record.state == "active"
    assert record.accepted_count == 1


def test_contract_board_repeatable_postings_unlock_after_first_completions() -> None:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)
    controller.company.completed_contract_ids.update(
        {
            "blackwood_road_charter",
            "shallow_cave_breach_scout",
        }
    )
    controller.company.active_contract_ids.discard("blackwood_road_charter")
    controller.company.known_breaches.add("shallow_cave_breach")

    view_result = controller.handle(ViewTown())

    assert view_result.success
    repeatable_scout = next(
        entry
        for entry in view_result.value.contract_board
        if entry.contract_id == "shallow_cave_breach_scout_posting"
    )
    contract_ids = {
        entry.contract_id
        for entry in view_result.value.contract_board
    }
    assert repeatable_scout.state == "available"
    assert "shallow_cave_breach_hunt_posting" not in contract_ids

    accepted = controller.handle(AcceptContract("shallow_cave_breach_scout_posting"))

    assert accepted.success, accepted.error
    assert "shallow_cave_breach_scout_posting" in controller.company.active_contract_ids
    record = controller.company.contract_records["shallow_cave_breach_scout_posting"]
    assert record.state == "active"
    assert record.accepted_count == 1

    controller.company.active_contract_ids.discard("shallow_cave_breach_scout_posting")
    record.state = "repeatable_completed"
    record.completed_count = 2
    accepted = controller.handle(AcceptContract("shallow_cave_breach_scout_posting"))
    assert accepted.success, accepted.error
    assert record.state == "active"
    assert record.accepted_count == 2
    assert record.completed_count == 2

    controller.company.active_contract_ids.discard("shallow_cave_breach_scout_posting")
    controller.company.completed_contract_ids.add("shallow_cave_breach_hunt")
    view_result = controller.handle(ViewTown())
    repeatable_hunt = next(
        entry
        for entry in view_result.value.contract_board
        if entry.contract_id == "shallow_cave_breach_hunt_posting"
    )
    assert repeatable_hunt.state == "available"


def test_contract_board_allows_only_one_generated_maze_contract_at_a_time() -> None:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)
    controller.company.completed_contract_ids.update(
        {
            "blackwood_road_charter",
            "shallow_cave_breach_scout",
            "shallow_cave_breach_hunt",
        }
    )
    controller.company.active_contract_ids.discard("blackwood_road_charter")
    controller.company.known_breaches.add("shallow_cave_breach")

    accepted = controller.handle(AcceptContract("shallow_cave_breach_scout_posting"))

    assert accepted.success, accepted.error
    assert "shallow_cave_breach_scout_posting" in controller.company.active_contract_ids

    view_result = controller.handle(ViewTown())
    contract_ids = {
        entry.contract_id
        for entry in view_result.value.contract_board
    }
    assert "shallow_cave_breach_hunt_posting" not in contract_ids

    rejected = controller.handle(AcceptContract("shallow_cave_breach_hunt_posting"))

    assert not rejected.success
    assert rejected.error == "Finish Breach Scout Posting first."
    assert "shallow_cave_breach_hunt_posting" not in controller.company.active_contract_ids


def test_company_upgrades_unlock_purchase_persist_and_change_town_values(
    tmp_path: Path,
) -> None:
    definitions = get_definitions()
    controller = AppController(definitions=definitions)
    controller.company = create_new_company(definitions)

    view = controller.handle(ViewTown()).value
    shelf = next(entry for entry in view.upgrades if entry.upgrade_id == "quartermaster_shelf")
    assert shelf.state == "locked"
    assert not controller.handle(PurchaseUpgrade("quartermaster_shelf")).success

    controller.company.completed_contract_ids.add("blackwood_road_charter")
    controller.company.active_contract_ids.discard("blackwood_road_charter")
    controller.company.coin = 8
    purchased = controller.handle(PurchaseUpgrade("quartermaster_shelf"))
    assert purchased.success, purchased.error
    assert controller.company.coin == 0
    assert "quartermaster_shelf" in controller.company.purchased_upgrade_ids
    assert not controller.handle(PurchaseUpgrade("quartermaster_shelf")).success

    controller.company.coin = 1
    bought = controller.handle(BuySupply("rations", quantity=1))
    assert bought.success, bought.error
    assert controller.company.coin == 0

    controller.company.coin = 10
    assert controller.handle(PurchaseUpgrade("recovery_cot")).success
    hero = controller.company.roster[0]
    hero.hp = 1
    controller.company.coin = 3
    recovered = controller.handle(RecoverCompany())
    assert recovered.success, recovered.error
    assert hero.hp == hero.max_hp

    controller.company.completed_contract_ids.add("shallow_cave_breach_scout")
    controller.company.coin = 12
    assert controller.handle(PurchaseUpgrade("charter_desk")).success
    view = controller.handle(ViewTown()).value
    assert view.roster_cap == definitions.town.roster_cap + 1

    save_path = tmp_path / "company.json"
    save_company(controller.company, save_path)
    loaded, _event = load_company(save_path)
    assert loaded.purchased_upgrade_ids == {
        "quartermaster_shelf",
        "recovery_cot",
        "charter_desk",
    }


def test_company_gear_purchase_equip_duplicate_counts_and_stats() -> None:
    definitions = get_definitions()
    controller = AppController(definitions=definitions)
    controller.company = create_new_company(definitions)
    hero = controller.company.roster[0]
    second_hero = controller.company.roster[1]

    view = controller.handle(ViewGear()).value
    vest = next(item for item in view.items if item.gear_id == "reinforced_vest")
    assert vest.state == "locked"
    assert not controller.handle(PurchaseGear("reinforced_vest")).success

    controller.company.completed_contract_ids.add("blackwood_road_charter")
    controller.company.active_contract_ids.discard("blackwood_road_charter")
    controller.company.coin = 12
    assert controller.handle(PurchaseGear("reinforced_vest")).success
    assert controller.handle(PurchaseGear("reinforced_vest")).success
    assert controller.company.coin == 0
    assert controller.company.gear_inventory["reinforced_vest"] == 2

    assert controller.handle(EquipGear(hero.hero_id, "reinforced_vest")).success
    assert controller.handle(EquipGear(second_hero.hero_id, "reinforced_vest")).success
    view = controller.handle(ViewGear()).value
    vest = next(item for item in view.items if item.gear_id == "reinforced_vest")
    assert vest.owned_count == 2
    assert vest.equipped_count == 2
    assert vest.available_count == 0

    combatants, _formation = party_combatants(controller.company, definitions)
    assert combatants[hero.hero_id].max_hp == hero.max_hp + 1

    assert controller.handle(UnequipGear(hero.hero_id)).success
    view = controller.handle(ViewGear()).value
    vest = next(item for item in view.items if item.gear_id == "reinforced_vest")
    assert vest.available_count == 1


def test_gear_locker_actions_are_compact_and_hero_equipment_actions_are_scoped() -> None:
    definitions = get_definitions()
    controller = AppController(definitions=definitions)
    controller.company = create_new_company(definitions)
    assert controller.company is not None
    hero = controller.company.roster[0]
    other_hero = controller.company.roster[1]
    controller.company.completed_contract_ids.add("blackwood_road_charter")
    controller.company.active_contract_ids.discard("blackwood_road_charter")
    controller.company.gear_inventory["reinforced_vest"] = 1

    view = controller.handle(ViewGear()).value
    locker_values = tuple(action.value for action in view.actions)
    assert all(
        value.startswith("gear:buy:") or value == "back"
        for value in locker_values
    )
    assert not any(value.startswith("gear:equip:") for value in locker_values)
    assert not any(value.startswith("gear:unequip:") for value in locker_values)

    hero_actions = ActionProvider.hero_gear_actions(
        controller.company,
        definitions,
        hero.hero_id,
        can_manage=True,
    )
    hero_values = tuple(action.value for action in hero_actions)
    assert f"gear:equip:{hero.hero_id}:reinforced_vest" in hero_values
    assert not any(other_hero.hero_id in value for value in hero_values)
    assert not any("field_satchel" in value for value in hero_values)

    controller.company.town_state["location_id"] = "shallow_cave"
    view = controller.handle(ViewGear()).value
    assert not any(action.value.startswith("gear:buy:") for action in view.actions)
    assert not controller.handle(PurchaseGear("reinforced_vest")).success


def test_gear_unequip_clamps_current_hp_and_recovery_uses_effective_max() -> None:
    definitions = get_definitions()
    controller = AppController(definitions=definitions)
    controller.company = create_new_company(definitions)
    controller.company.completed_contract_ids.add("blackwood_road_charter")
    controller.company.active_contract_ids.discard("blackwood_road_charter")
    controller.company.gear_inventory["reinforced_vest"] = 1
    controller.company.coin = 4
    hero = controller.company.roster[0]

    assert controller.handle(EquipGear(hero.hero_id, "reinforced_vest")).success
    assert controller.handle(RecoverCompany()).success
    assert hero.hp == hero.max_hp + 1

    assert controller.handle(UnequipGear(hero.hero_id)).success
    assert hero.hp == hero.max_hp


def test_recovery_restores_living_heroes_without_removing_mortal_wounds() -> None:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)
    controller.company.coin = 4
    hero = controller.company.roster[0]
    hero.hp = 1
    hero.effort = 0
    hero.statuses.add(ActorStatus.DOWNED)
    hero.mortal_wounds = 1

    result = controller.handle(RecoverCompany())

    assert result.success
    assert hero.hp == hero.max_hp
    assert hero.effort == hero.max_effort
    assert ActorStatus.DOWNED not in hero.statuses
    assert hero.mortal_wounds == 1


def test_deep_surgery_removes_wound_and_marks_in_surgery() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    controller = AppController(definitions=definitions, company=company)
    hero = company.roster[0]
    hero.mortal_wounds = 2
    company.coin = 9

    result = controller.handle(PerformDeepSurgery(hero.hero_id))

    assert result.success
    assert hero.mortal_wounds == 1
    assert hero.in_surgery is True
    assert company.coin == 0


def test_deep_surgery_removes_hero_from_active_party() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    controller = AppController(definitions=definitions, company=company)
    hero_id = company.active_party_slots[FormationSlot.FRONT_LEFT]
    assert hero_id is not None
    hero = next(entry for entry in company.roster if entry.hero_id == hero_id)
    hero.mortal_wounds = 1
    company.coin = 9

    result = controller.handle(PerformDeepSurgery(hero_id))

    assert result.success
    assert company.active_party_slots[FormationSlot.FRONT_LEFT] is None
    assert hero.in_surgery is True


def test_assign_active_hero_blocks_in_surgery_hero() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    controller = AppController(definitions=definitions, company=company)
    hero = company.roster[0]
    hero.mortal_wounds = 1
    hero.in_surgery = True

    result = controller.handle(AssignActiveHero(hero.hero_id, FormationSlot.BACK_LEFT))

    assert not result.success
    assert "In Surgery" in (result.error or "")


def test_deep_surgery_requires_mortal_wound_and_coin() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    hero = company.roster[0]
    company.coin = 9

    no_wound = deep_surgery(company, definitions, hero.hero_id)
    assert not no_wound.success

    hero.mortal_wounds = 1
    company.coin = 0
    no_coin = deep_surgery(company, definitions, hero.hero_id)
    assert not no_coin.success


def test_clear_surgery_recovery_clears_flag_and_returns_moments() -> None:
    company = create_new_company(get_definitions())
    hero = company.roster[0]
    hero.in_surgery = True

    moments = clear_surgery_recovery(company)

    assert hero.in_surgery is False
    assert any(hero.name in moment for moment in moments)


def test_in_surgery_clears_after_finish_report() -> None:
    company = create_new_company(get_definitions())
    hero = company.roster[0]
    hero.in_surgery = True
    report = ExpeditionReportState(
        expedition_id="test_route",
        dungeon_id="test_dungeon",
        participant_ids=[],
    )

    finalize_report_memory(company, report, "returned_to_haven")

    assert hero.in_surgery is False
    assert any("surgery ward" in moment for moment in report.notable_moments)


def test_recover_company_still_restores_in_surgery_hero_hp() -> None:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)
    controller.company.coin = 4
    hero = controller.company.roster[0]
    hero.hp = 1
    hero.effort = 0
    hero.mortal_wounds = 1
    hero.in_surgery = True

    result = controller.handle(RecoverCompany())

    assert result.success
    assert hero.hp == hero.max_hp
    assert hero.effort == hero.max_effort
    assert hero.in_surgery is True
    assert hero.mortal_wounds == 1


def test_active_party_assignment_controls_combat_party_and_reserves() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    controller = AppController(definitions=definitions, company=company)

    result = controller.handle(AssignActiveHero("hero_cutpurse", FormationSlot.BACK_LEFT))

    assert result.success
    assert company.active_party_slots[FormationSlot.BACK_LEFT] == "hero_cutpurse"
    assert company.active_party_slots[FormationSlot.FRONT_RIGHT] == "hero_field_surgeon"
    combatants, formation = party_combatants(company)
    assert formation.actor_at(FormationSlot.BACK_LEFT) == "hero_cutpurse"
    assert formation.actor_at(FormationSlot.FRONT_RIGHT) == "hero_field_surgeon"
    assert "hero_cutpurse" in combatants
    assert "hero_field_surgeon" in combatants
    assert not any(hero.hero_id == "hero_field_surgeon" for hero in reserve_roster(company))


def test_active_party_assignment_swaps_between_occupied_active_slots() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    controller = AppController(definitions=definitions, company=company)
    left_id = company.active_party_slots[FormationSlot.BACK_LEFT]
    right_id = company.active_party_slots[FormationSlot.BACK_RIGHT]
    assert left_id is not None
    assert right_id is not None

    result = controller.handle(AssignActiveHero(right_id, FormationSlot.BACK_LEFT))

    assert result.success
    assert company.active_party_slots[FormationSlot.BACK_LEFT] == right_id
    assert company.active_party_slots[FormationSlot.BACK_RIGHT] == left_id


def test_marked_does_not_persist_across_combat_boundaries() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    company.roster[0].tags.add(Tag.MARKED)

    combatants, formation = party_combatants(company)

    assert Tag.MARKED not in combatants["hero_watchman"].tags

    apply_marked(combatants["hero_watchman"])
    sync_company_from_combat(company, combatants, formation)

    assert Tag.MARKED not in company.roster[0].tags


def test_save_load_preserves_active_party_and_older_saves_are_derived(tmp_path: Path) -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    company.active_party_slots[FormationSlot.BACK_LEFT] = "hero_cutpurse"
    company.deceased_heroes.append(company.roster[0])
    save_path = tmp_path / "company.json"

    save_company(company, save_path)
    loaded, _event = load_company(save_path)

    assert loaded.active_party_slots[FormationSlot.BACK_LEFT] == "hero_cutpurse"
    assert loaded.deceased_heroes[0].hero_id == "hero_watchman"
    assert loaded.save_version == SAVE_VERSION

    old_payload = company.to_dict()
    old_payload.pop("active_party_slots")
    old_payload.pop("coin")
    old_payload.pop("purchased_upgrade_ids")
    old_payload.pop("gear_inventory")
    old_payload["roster"][0].pop("equipped_gear_id")
    old_payload["save_version"] = 1
    older = CompanyState.from_dict(old_payload)

    assert older.active_party_slots[FormationSlot.FRONT_LEFT] == "hero_watchman"
    assert older.coin == STARTING_COIN
    assert older.purchased_upgrade_ids == set()
    assert older.gear_inventory == {}
    assert older.roster[0].equipped_gear_id is None
    assert older.save_version == SAVE_VERSION


def test_sell_maze_glass_adds_coin_and_consumes_inventory() -> None:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)
    controller.company.inventory["maze_glass"] = 2
    coin_before = controller.company.coin

    result = controller.handle(SellLoot("maze_glass"))

    assert result.success, result.error
    assert controller.company.inventory["maze_glass"] == 1
    assert controller.company.coin == coin_before + 6


def test_turn_in_cave_relic_posts_missing_carters_contract() -> None:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)
    controller.company.inventory["cave_relic"] = 1
    controller.company.completed_contract_ids.add("blackwood_road_charter")

    result = controller.handle(TurnInLoot("cave_relic"))

    assert result.success, result.error
    assert "cave_relic" not in controller.company.inventory
    assert controller.company.flags["cave_relic_filed"] is True
    assert contract_is_posted(
        controller.company,
        controller.definitions,
        controller.definitions.contracts["missing_carters"],
    )
    state, reason = contract_board_state(
        controller.company,
        controller.definitions,
        "missing_carters",
    )
    assert state == "available", reason


def test_missing_carters_can_be_accepted_after_cave_relic_turn_in() -> None:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)
    controller.company.inventory["cave_relic"] = 1
    controller.company.completed_contract_ids.add("blackwood_road_charter")

    state, reason = contract_board_state(
        controller.company,
        controller.definitions,
        "missing_carters",
    )
    assert state == "locked", reason

    assert controller.handle(TurnInLoot("cave_relic")).success

    accept = controller.handle(AcceptContract("missing_carters"))
    assert accept.success, accept.error
    assert "missing_carters" in controller.company.active_contract_ids


def test_relic_broker_view_lists_owned_loot_actions() -> None:
    controller = AppController(definitions=get_definitions())
    controller.company = create_new_company(controller.definitions)
    controller.company.inventory["maze_glass"] = 1
    controller.company.inventory["cave_relic"] = 1
    view = build_relic_broker_view(controller.company, controller.definitions)
    enabled = {
        action.value: action.enabled
        for action in view.actions
        if action.value != "back"
    }
    assert enabled == {
        "sell_loot:maze_glass": True,
        "turn_in_loot:cave_relic": True,
    }
