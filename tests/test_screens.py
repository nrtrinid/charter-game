from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from game.app.commands import ChooseCombatSkill, StartManualCombat, StartNewCompany, ViewCombat
from game.app.controller import AppController
from game.app.manual_combat import EnemyIntent, auto_advance_to_hero
from game.app.views import (
    CombatView,
    ScreenAction,
    build_combat_view,
    build_expedition_report_view,
    build_gear_inventory_view,
    build_hero_sheet_view,
    build_roster_sections,
    build_town_dashboard,
)
from game.campaign.company import (
    CompanyTimelineEntry,
    ExpeditionReportState,
    HeroMemoryEntry,
    ReportEventSignal,
    create_new_company,
)
from game.campaign.hero_memory import EarnedQuirkSlotState, FreshMemoryState
from game.campaign.town import ledger
from game.combat.preview import preview_attack
from game.combat.turn_order import InitiativeEntry
from game.core.rng import GameRng
from game.ui.screens import (
    build_event_beats,
    render_action_list,
    render_combat_view,
    render_command_dock,
    render_expedition_report,
    render_gear_inventory,
    render_ledger,
    render_main_menu,
    render_recent_log,
    render_roster,
    render_roster_sections,
    render_save_slot,
    render_screen,
    render_supplies,
    render_town,
)
from tests.conftest import get_definitions


def render_to_text(renderable: object) -> str:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=120)
    console.print(renderable)
    return output.getvalue()


def test_main_menu_lists_actions_and_aliases() -> None:
    text = render_to_text(render_main_menu())

    assert "Haven Town" in text
    assert "town, t" in text
    assert "quit, q" in text


def test_action_list_renders_as_command_dock() -> None:
    dock = render_action_list(
        (ScreenAction("1", "Continue", "continue", ("c",), default=True),)
    )
    text = render_to_text(dock)

    assert "Command Dock" in text
    assert "Hotkey" in text
    assert "default" in text
    assert "Continue" in text
    assert "c" in text
    assert str(dock.border_style) == "bright_yellow"


def test_screen_title_appears_once_for_menu_frame(tmp_path: Path) -> None:
    text = render_to_text(
        render_screen(None, tmp_path / "company.json", "Main Menu", render_main_menu())
    )

    assert text.count("Main Menu") == 1


def test_screen_frame_orders_body_log_and_command_dock(tmp_path: Path) -> None:
    dock = render_command_dock(
        (ScreenAction("1", "Continue", "continue", ("c",), default=True),),
        prompt="Continue",
    )
    text = render_to_text(
        render_screen(
            None,
            tmp_path / "company.json",
            "Main Menu",
            "Body region",
            log=render_recent_log([], title="Recent Log"),
            command_dock=dock,
        )
    )

    assert text.index("The Charter") < text.index("Main Menu")
    assert text.index("Main Menu") < text.index("Recent Log")
    assert text.index("Recent Log") < text.index("Command Dock")


def test_soft_spacer_is_terminal_only(tmp_path: Path) -> None:
    actions = (ScreenAction("1", "Continue", "continue", ("c",), default=True),)
    save_path = tmp_path / "company.json"

    terminal_output = StringIO()
    terminal = Console(
        file=terminal_output,
        force_terminal=True,
        no_color=True,
        width=100,
        height=36,
    )
    terminal.print(
        render_screen(
            None,
            save_path,
            "Main Menu",
            "Body region",
            command_dock=render_command_dock(actions, prompt="Continue"),
            console=terminal,
            enable_spacer=True,
        )
    )

    fake_output = StringIO()
    fake = Console(file=fake_output, force_terminal=False, width=100, height=36)
    fake.print(
        render_screen(
            None,
            save_path,
            "Main Menu",
            "Body region",
            command_dock=render_command_dock(actions, prompt="Continue"),
            console=fake,
            enable_spacer=True,
        )
    )

    assert _blank_lines_before("Command Dock", terminal_output.getvalue()) > 0
    assert _blank_lines_before("Command Dock", fake_output.getvalue()) == 0


def test_fixed_viewport_compacts_overflow_to_terminal_height(tmp_path: Path) -> None:
    terminal_output = StringIO()
    terminal = Console(
        file=terminal_output,
        force_terminal=True,
        no_color=True,
        width=100,
        height=24,
    )
    actions = tuple(
        ScreenAction(str(index), f"Action {index}", f"action_{index}", (str(index),))
        for index in range(1, 7)
    )
    body = "\n".join(f"Body line {index}" for index in range(30))

    terminal.print(
        render_screen(
            None,
            tmp_path / "company.json",
            "Long Screen",
            body,
            log=render_recent_log([], title="Recent Log"),
            command_dock=render_command_dock(actions, prompt="Main"),
            console=terminal,
            enable_spacer=True,
        )
    )

    text = terminal_output.getvalue()
    assert len(text.splitlines()) <= terminal.size.height - 1
    assert "Long Screen" in text
    assert "Command Dock" in text
    assert "..." in text


def test_roster_supplies_and_ledger_render_key_state(tmp_path: Path) -> None:
    company = create_new_company(get_definitions(), name="Lantern Road")
    company.hero_memories.append(
        HeroMemoryEntry(
            entry_id="hero_memory_0001",
            hero_id="hero_watchman",
            hero_name="Mara Vell",
            kind="first_expedition",
            summary="Mara Vell first marched with the company.",
            expedition_id="opening",
            dungeon_id="shallow_cave",
        )
    )
    company.company_timeline.append(
        CompanyTimelineEntry(
            entry_id="company_timeline_0001",
            kind="expedition_returned",
            summary="Opening expedition returned to Haven.",
            expedition_id="opening",
            dungeon_id="shallow_cave",
        )
    )
    company.roster[0].mortal_wounds = 1

    roster_text = render_to_text(render_roster(company.roster))
    roster_sections_text = render_to_text(render_roster_sections(build_roster_sections(company)))
    supplies_text = render_to_text(render_supplies(company.supplies))
    ledger_text = render_to_text(render_ledger(ledger(company, get_definitions())))
    welcome_text = render_to_text(
        render_screen(company, tmp_path / "company.json", "Main", render_main_menu())
    )

    assert "Mara Vell" in roster_text
    assert "Mortal Wounds" in roster_text
    assert "1/3" in roster_text
    assert "first marched" in roster_sections_text
    assert "1/3" in roster_sections_text
    assert "FRONT_LEFT" in roster_text
    assert "rations" in supplies_text
    assert "Lantern Road" in ledger_text
    assert "current_objective" in ledger_text
    assert "report_count" in ledger_text
    assert "Opening expedition returned to Haven" in ledger_text
    assert "Save slot" in welcome_text
    assert "Active" in welcome_text
    assert "Reserves" in welcome_text


def test_hero_sheet_view_collects_traits_gear_and_memory() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions, name="Lantern Road")
    hero = company.roster[0]
    hero.quirks = ["ice_nerves"]
    hero.earned_quirk_slots = [
        EarnedQuirkSlotState("ice_nerves", stability="locked", unlocked_order=1)
    ]
    hero.fresh_memories = [
        FreshMemoryState(
            family_id="killing_blow",
            display_name="Killing Blow",
            intensity=2,
            source_summary="Breach discovered: shallow_cave_breach.",
        )
    ]
    hero.career_signals = {"killing_blow": 2, "tag:kill": 2}
    company.gear_inventory["reinforced_vest"] = 1
    hero.equipped_gear_id = "reinforced_vest"
    company.hero_memories.append(
        HeroMemoryEntry(
            entry_id="hero_memory_0001",
            hero_id=hero.hero_id,
            hero_name=hero.name,
            kind="first_expedition",
            summary="Mara Vell first marched with the company.",
            expedition_id="opening",
            dungeon_id="shallow_cave",
        )
    )

    view = build_hero_sheet_view(company, definitions, hero.hero_id)

    assert view is not None
    assert view.personal_quirk is not None
    assert view.personal_quirk.name == "Hold the Line"
    assert "Guarded" in view.personal_quirk.positive_text
    assert view.earned_quirks[0].name == "Ice Nerves"
    assert view.earned_quirks[0].stability == "locked"
    assert view.fresh_memories[0].name == "Killing Blow"
    assert view.fresh_memories[0].intensity == 2
    assert (
        view.fresh_memories[0].source_summary == "Breach discovered: Shallow Cave Breach."
    )
    assert view.latest_memory == "Mara Vell first marched with the company."
    assert view.equipped_gear == "Reinforced Vest"
    assert view.stat_bonus == "Max HP +1"
    assert view.career_signals[0].label in {"Killing Blows", "killing"}
    assert {signal.label for signal in view.career_signals} == {"Killing Blows", "killing"}
    assert all(not signal.label.startswith("Tag:") for signal in view.career_signals)


def test_save_slot_reports_present_or_empty(tmp_path: Path) -> None:
    save_path = tmp_path / "company.json"

    assert "empty" in render_to_text(render_save_slot(save_path))
    save_path.write_text("{}", encoding="utf-8")
    assert "present" in render_to_text(render_save_slot(save_path))


def test_event_beats_bundle_combat_actions() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    from game.core.rng import GameRng
    from game.expedition.expedition import run_opening_route

    events = run_opening_route(company, definitions, GameRng(7), stop_at_breach=True)
    beats = build_event_beats(events)

    assert any(beat.combat for beat in beats)
    assert len(beats) < len(events)


def test_combat_screen_renders_operational_turn_surface(tmp_path: Path) -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    view_result = controller.handle(ViewCombat())
    assert isinstance(view_result.value, CombatView)
    skill = next(skill for skill in view_result.value.skills if skill.action.enabled)
    skill_result = controller.handle(ChooseCombatSkill(skill.skill_id))
    assert isinstance(skill_result.value, CombatView)

    text = render_to_text(render_combat_view(skill_result.value))

    assert "Party" in text
    assert "Enemies" in text
    assert "Focus" in text
    assert "Art" in text
    assert "Skills" in text
    assert "Targets" in text
    assert "Recent Log" not in text
    assert "Hit" in text
    assert "Cohesion" in text
    assert "Order" not in text

    framed_text = render_to_text(
        render_screen(
            controller.company,
            tmp_path / "company.json",
            "Combat",
            render_combat_view(skill_result.value),
            log=render_recent_log(skill_result.value.recent_events),
            command_dock=render_command_dock(
                tuple(option.action for option in skill_result.value.targets),
                prompt="Target",
            ),
        )
    )

    assert "Recent Log" in framed_text
    assert "Command Dock" in framed_text


def test_combat_screen_renders_turn_order_after_special_enemy_action() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("cave_mini_boss"))
    assert controller.manual_combat is not None
    controller.manual_combat.initiative = [
        InitiativeEntry("cave_maw_brute_1", 99),
        InitiativeEntry("hero_watchman", 98),
    ]
    controller.manual_combat.turn_index = 0
    auto_advance_to_hero(controller.manual_combat, controller.definitions, GameRng(1))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    text = render_to_text(render_combat_view(view_result.value))
    assert "Turns" in text
    assert "Maw" in text
    assert "Mara" in text
    assert "Enemy Intent" not in text
    assert "Class Reactions" not in text
    assert "Skip Reaction" not in text
    assert "Debug hit" not in text
    assert "Debug damage" not in text


def test_combat_screen_renders_debug_damage_range_for_pending_intent() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("cave_mini_boss"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    skill = controller.definitions.skills["maw_slam"]
    enemy = session.state.actor("cave_maw_brute_1")
    target = session.state.actor("hero_watchman")
    preview = preview_attack(session.state, enemy.actor_id, skill, target.actor_id)
    session.pending_enemy_intent = EnemyIntent(
        enemy_id=enemy.actor_id,
        enemy_name=enemy.name,
        skill_id=skill.id,
        skill_name=skill.name,
        label=skill.intent_label or skill.name,
        target_id=target.actor_id,
        target_name=target.name,
        threat_level=skill.threat_level,
        obvious_effect=skill.obvious_effect,
        hit_chance=preview.hit_chance,
        damage_estimate=preview.damage,
        damage_label=preview.damage_label,
    )

    view = build_combat_view(session, controller.definitions, debug_combat_preview=True)
    text = render_to_text(render_combat_view(view))

    assert "Debug damage: 5-6" in text


def test_town_dashboard_renders_active_reserve_and_budget_state() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions, name="Lantern Road")
    view = build_town_dashboard(company, definitions)

    text = render_to_text(render_town(view))

    assert "Haven Town" in text
    assert "Active Party" in text
    assert "Reserves" in text
    assert "Reputation" in text
    assert "Coin" in text
    assert "Current Objective" in text
    assert "Complete Blackwood Road Charter" in text
    assert "Progress" in text
    assert "Company Upgrades" in text
    assert "Quartermaster Shelf" in text


def test_gear_inventory_renders_items_and_equipped_kits() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions, name="Lantern Road")
    company.gear_inventory["reinforced_vest"] = 1
    company.roster[0].equipped_gear_id = "reinforced_vest"
    view = build_gear_inventory_view(
        company,
        definitions,
        can_manage=True,
        can_purchase=True,
    )

    text = render_to_text(render_gear_inventory(view))

    assert "Armory" in text
    assert "Reinforced Vest" in text
    assert "Mara Vell" in text
    assert "Max HP +1" in text


def test_expedition_report_renders_changes_and_next_objective() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    report = ExpeditionReportState(
        expedition_id="opening",
        dungeon_id="shallow_cave",
        outcome="returned_to_haven",
        rooms_entered=["old_road"],
        start_reputation=0,
        end_reputation=2,
        breaches_discovered=["shallow_cave_breach"],
        event_signals=[
            ReportEventSignal(
                kind="contract_completed",
                message="Contract completed: Blackwood Road Charter.",
            )
        ],
    )
    company.last_expedition_report = report

    text = render_to_text(
        render_expedition_report(build_expedition_report_view(company, definitions))
    )

    assert "Filed Company Record" in text
    assert "What Changed" in text
    assert "Contract completed" in text
    assert "Next Objective" in text
    assert "Complete Blackwood Road Charter" in text


def _blank_lines_before(marker: str, text: str) -> int:
    lines = text.splitlines()
    index = next(index for index, line in enumerate(lines) if marker in line)
    count = 0
    for line in reversed(lines[:index]):
        if line.strip():
            break
        count += 1
    return count
