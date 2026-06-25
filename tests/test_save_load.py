from pathlib import Path

from game.app.commands import LoadGame
from game.app.controller import AppController
from game.campaign.company import (
    STARTING_COIN,
    BreachMemoryState,
    CompanyState,
    CompanyTimelineEntry,
    ContractRecordState,
    DungeonMemoryState,
    ExpeditionReportState,
    HeroMemoryEntry,
    RecruitmentOfferState,
    WorldLocationMemoryState,
    create_new_company,
)
from game.campaign.hero_memory import (
    STABILITY_LOOSE,
    STABILITY_SETTLED,
    EarnedQuirkSlotState,
    FreshMemoryState,
    RecentSignal,
)
from game.campaign.migrations import upgrade_raw_save
from game.campaign.save_load import load_company, save_company
from game.combat.combat_state import LifeState, StrainMark, StrainTier, Tag
from tests.conftest import get_definitions


def test_company_can_be_saved_and_loaded(tmp_path: Path) -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    assert all(hero.strain == StrainTier.STEADY for hero in company.roster)
    company.known_breaches.add("shallow_cave_breach")
    company.reputation = 4
    company.coin = 17
    company.inventory["cave_relic"] = 1
    company.gear_inventory["reinforced_vest"] = 1
    company.roster[1].equipped_gear_id = "reinforced_vest"
    company.roster[1].strain = StrainTier.WORN
    company.roster[1].strain_marks = {StrainMark.DRAINED}
    company.roster[1].career_signals = {"killing_blow": 2}
    company.roster[1].fresh_memories = [
        FreshMemoryState(
            family_id="killing_blow",
            display_name="Killing Blow",
            intensity=2,
            source_summary="Enemy fell.",
            created_order=1,
            refreshed_order=2,
        )
    ]
    company.roster[1].earned_quirk_slots = [
        EarnedQuirkSlotState("blood_hot", STABILITY_LOOSE, 1)
    ]
    company.roster[1].quirks = ["blood_hot"]
    company.supplies["rations"] = 3
    company.expedition_history.append("shallow_cave_boss_defeated")
    company.dungeon_memory["shallow_cave"] = DungeonMemoryState(
        dungeon_id="shallow_cave",
        visited_node_ids=["shallow_cave_room_1", "old_works_cache"],
        cleared_node_ids=["shallow_cave_room_1", "old_works_cache"],
        completed_action_ids=["old_works_cache:recover_gate_key"],
        revealed_exit_ids=["stone_gate->maze_touched_lair"],
    )
    report = ExpeditionReportState(
        expedition_id="opening",
        dungeon_id="shallow_cave",
        outcome="returned_to_haven",
        rooms_entered=["town_gate", "old_road"],
        coin_gained=3,
        start_coin=8,
        end_coin=11,
        memory_signals=[
            RecentSignal(
                hero_id=company.roster[1].hero_id,
                family_id="killing_blow",
                source_summary="Enemy fell.",
                order=1,
            )
        ],
    )
    company.last_expedition_report = report
    company.expedition_reports.append(report)
    company.world_memory["shallow_cave"] = WorldLocationMemoryState(
        location_id="shallow_cave",
        visited=True,
        visit_count=2,
        discovered_node_ids=["shallow_cave_room_1", "old_works_cache"],
        cleared_threat_node_ids=["old_works_cache"],
        consumed_rumor_ids=["rumor_black_stones"],
        unlocked_shortcut_ids=["stone_gate->maze_touched_lair"],
    )
    company.recruitment_state.current_offers = [
        RecruitmentOfferState(
            name="Sera",
            class_id="watchman",
            background="road watch",
            motive="steady work",
        )
    ]
    company.recruitment_state.refresh_count = 3
    company.contract_records["shallow_cave_breach_scout"] = ContractRecordState(
        contract_id="shallow_cave_breach_scout",
        state="completed",
        accepted_count=1,
        completed_count=1,
        last_run_id="maze_run_0001",
        rooms_scouted=3,
    )
    company.breach_memory["maze_breach"] = BreachMemoryState(
        source_node_id="maze_breach",
        run_count=2,
        collapsed_run_ids=["maze_run_0001"],
        scouted_run_ids=["maze_run_0001"],
        hunt_run_ids=["maze_run_0002"],
        last_pressure_id="marked_hunt",
        pressure_counts={"breach_probe": 1, "marked_hunt": 1},
        last_seed=11,
    )
    company.deceased_heroes.append(company.roster[0])
    company.hero_memories.append(
        HeroMemoryEntry(
            entry_id="hero_memory_0001",
            hero_id="hero_watchman",
            hero_name="Mara Vell",
            kind="death",
            summary="Mara Vell fell during Shallow Cave.",
            expedition_id="opening",
            dungeon_id="shallow_cave",
            encounter_id="shallow_cave",
        )
    )
    company.company_timeline.append(
        CompanyTimelineEntry(
            entry_id="company_timeline_0001",
            kind="breach_discovered",
            summary="Breach discovered: shallow_cave_breach.",
            expedition_id="opening",
            dungeon_id="shallow_cave",
            node_id="maze_breach",
        )
    )
    save_path = tmp_path / "company.json"

    save_company(company, save_path)
    loaded, _event = load_company(save_path)

    assert loaded.name == company.name
    assert [hero.name for hero in loaded.roster] == [hero.name for hero in company.roster]
    assert "shallow_cave_breach" in loaded.known_breaches
    assert loaded.reputation == 4
    assert loaded.coin == 17
    assert loaded.deceased_heroes[0].name == company.roster[0].name
    assert loaded.supplies["rations"] == 3
    assert loaded.inventory["cave_relic"] == 1
    assert loaded.gear_inventory["reinforced_vest"] == 1
    assert loaded.roster[1].equipped_gear_id == "reinforced_vest"
    assert loaded.roster[1].strain == StrainTier.WORN
    assert loaded.roster[1].strain_marks == {StrainMark.DRAINED}
    assert loaded.roster[1].career_signals == {"killing_blow": 2}
    assert loaded.roster[1].fresh_memories[0].family_id == "killing_blow"
    assert loaded.roster[1].fresh_memories[0].intensity == 2
    assert loaded.roster[1].earned_quirk_slots == [
        EarnedQuirkSlotState("blood_hot", STABILITY_LOOSE, 1)
    ]
    assert loaded.roster[1].quirks == ["blood_hot"]
    assert loaded.expedition_history == ["shallow_cave_boss_defeated"]
    assert loaded.dungeon_memory["shallow_cave"].visited_node_ids == [
        "shallow_cave_room_1",
        "old_works_cache",
    ]
    assert "old_works_cache" in loaded.dungeon_memory["shallow_cave"].cleared_node_ids
    assert (
        "old_works_cache:recover_gate_key"
        in loaded.dungeon_memory["shallow_cave"].completed_action_ids
    )
    assert (
        "stone_gate->maze_touched_lair"
        in loaded.dungeon_memory["shallow_cave"].revealed_exit_ids
    )
    assert loaded.hero_memories[0].summary == "Mara Vell fell during Shallow Cave."
    assert loaded.hero_memories[0].encounter_id == "shallow_cave"
    assert loaded.company_timeline[0].kind == "breach_discovered"
    assert loaded.company_timeline[0].node_id == "maze_breach"
    assert loaded.last_expedition_report is not None
    assert loaded.last_expedition_report.outcome == "returned_to_haven"
    assert loaded.last_expedition_report.coin_gained == 3
    assert loaded.last_expedition_report.start_coin == 8
    assert loaded.last_expedition_report.end_coin == 11
    assert loaded.last_expedition_report.memory_signals[0].family_id == "killing_blow"
    assert len(loaded.expedition_reports) == 1
    assert loaded.expedition_reports[0].rooms_entered == ["town_gate", "old_road"]
    assert loaded.world_memory["shallow_cave"].visit_count == 2
    assert "old_works_cache" in loaded.world_memory["shallow_cave"].discovered_node_ids
    assert "old_works_cache" in loaded.world_memory["shallow_cave"].cleared_threat_node_ids
    assert "rumor_black_stones" in loaded.world_memory["shallow_cave"].consumed_rumor_ids
    assert (
        "stone_gate->maze_touched_lair"
        in loaded.world_memory["shallow_cave"].unlocked_shortcut_ids
    )
    assert loaded.recruitment_state.refresh_count == 3
    assert loaded.recruitment_state.current_offers[0].name == "Sera"
    scout_record = loaded.contract_records["shallow_cave_breach_scout"]
    assert scout_record.state == "completed"
    assert scout_record.completed_count == 1
    assert scout_record.last_run_id == "maze_run_0001"
    assert scout_record.rooms_scouted == 3
    maze_memory = loaded.breach_memory["maze_breach"]
    assert maze_memory.run_count == 2
    assert maze_memory.scouted_run_ids == ["maze_run_0001"]
    assert maze_memory.hunt_run_ids == ["maze_run_0002"]
    assert maze_memory.pressure_counts == {"breach_probe": 1, "marked_hunt": 1}

    old_payload = company.to_dict()
    old_payload.pop("coin")
    old_payload["last_expedition_report"].pop("coin_gained")
    old_payload["last_expedition_report"].pop("start_coin")
    old_payload["last_expedition_report"].pop("end_coin")
    old_payload["expedition_reports"][0].pop("coin_gained")
    old_payload["expedition_reports"][0].pop("start_coin")
    old_payload["expedition_reports"][0].pop("end_coin")
    old_company = CompanyState.from_dict(old_payload)

    assert old_company.coin == STARTING_COIN
    assert old_company.last_expedition_report is not None
    assert old_company.last_expedition_report.coin_gained == 0
    assert old_company.last_expedition_report.start_coin == 0
    assert old_company.last_expedition_report.end_coin == 0
    assert old_company.expedition_reports[0].coin_gained == 0


def test_kill_quirk_slots_and_context_tags_round_trip(tmp_path: Path) -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    hero = company.roster[1]
    hero.career_signals = {"killing_blow": 4, "tag:final_kill": 1, "tag:basic": 2}
    hero.fresh_memories = [
        FreshMemoryState(
            family_id="killing_blow",
            display_name="Killing Blow",
            tags=("kill", "combat", "final_kill", "basic"),
            intensity=2,
            source_summary="Enemy fell.",
            created_order=1,
            refreshed_order=2,
        )
    ]
    hero.earned_quirk_slots = [
        EarnedQuirkSlotState("closer", STABILITY_SETTLED, 1),
        EarnedQuirkSlotState("no_waste", STABILITY_LOOSE, 2),
    ]
    hero.quirks = ["closer", "no_waste"]

    save_path = tmp_path / "kill_quirks.json"
    save_company(company, save_path)
    loaded, _event = load_company(save_path)

    loaded_hero = loaded.roster[1]
    assert loaded_hero.career_signals == hero.career_signals
    assert loaded_hero.fresh_memories[0].tags == hero.fresh_memories[0].tags
    assert loaded_hero.earned_quirk_slots == hero.earned_quirk_slots
    assert loaded_hero.quirks == ["closer", "no_waste"]


def test_old_hero_state_fields_migrate_to_life_tags_and_strain() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    raw = company.to_dict()
    hero = raw["roster"][0]
    hero.pop("life_state", None)
    hero.pop("strain", None)
    hero.pop("strain_marks", None)
    hero.pop("career_signals", None)
    hero.pop("fresh_memories", None)
    hero.pop("earned_quirk_slots", None)
    hero["quirks"] = ["blood_hot"]
    hero["statuses"] = ["downed", "stunned", "knocked_down"]
    hero["fatigue"] = "TIRED"
    hero["conditions"] = ["spent"]

    loaded = CompanyState.from_dict(raw)
    migrated = loaded.roster[0]

    assert migrated.life_state == LifeState.DOWNED
    assert Tag.STUNNED in migrated.tags
    assert Tag.KNOCKED_DOWN in migrated.tags
    assert migrated.strain == StrainTier.SPENT
    assert migrated.strain_marks == {
        StrainMark.WINDED,
        StrainMark.DRAINED,
        StrainMark.FRAYED,
    }
    assert migrated.career_signals == {}
    assert migrated.fresh_memories == []
    assert migrated.earned_quirk_slots == [
        EarnedQuirkSlotState("blood_hot", STABILITY_SETTLED, 1)
    ]


def test_upgrade_raw_save_fills_missing_coin_fields() -> None:
    upgraded = upgrade_raw_save(
        {
            "company_id": "company_001",
            "name": "Haven Charter",
            "roster": [],
            "supplies": {},
            "expedition_reports": [
                {
                    "expedition_id": "opening",
                    "dungeon_id": "shallow_cave",
                }
            ],
        }
    )

    assert upgraded["coin"] == STARTING_COIN
    assert upgraded["expedition_reports"][0]["coin_gained"] == 0
    assert upgraded["expedition_reports"][0]["start_coin"] == 0
    assert upgraded["expedition_reports"][0]["end_coin"] == 0


def test_app_load_missing_save_returns_failure(tmp_path: Path) -> None:
    controller = AppController(definitions=get_definitions())

    result = controller.handle(LoadGame(tmp_path / "missing.json"))

    assert not result.success
    assert result.error is not None
