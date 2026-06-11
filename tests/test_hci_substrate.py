from __future__ import annotations

from pathlib import Path

from game.app.actions import ActionProvider, ScreenActionRisk
from game.app.commands import (
    BuySupply,
    ChooseCombatSkill,
    SaveGame,
    StartExpedition,
    StartManualCombat,
    StartNewCompany,
    ViewCombat,
    ViewDungeon,
)
from game.app.controller import AppController
from game.app.views import (
    CombatView,
    DungeonView,
    ScreenAction,
    build_formation_view,
    build_town_dashboard,
)
from game.campaign.company import create_new_company
from game.combat.formation import FormationSlot
from game.core.events import BreachDiscoveredEvent, DamageEvent, DownedEvent, RoundStartedEvent
from game.core.hci import (
    EventImportance,
    HeroStateSnapshot,
    StateSnapshot,
    build_event_beats,
    build_hci_result_analysis,
    event_importance,
    tactical_brief_lines,
)
from game.ui.hci_text import (
    PARTY_WATCH_COLUMN_WIDTH,
    format_party_watch,
    format_party_watch_slot,
    generic_action_detail,
    unavailable_message,
)
from game.ui.screens import hci_summary_lines, render_hci_summary, render_resolution_card
from game.ui.tui_widgets import CommandDock
from tests.conftest import get_definitions


def test_controller_result_hci_tracks_resource_deltas() -> None:
    controller = AppController()
    start = controller.handle(StartNewCompany("Delta Charter"))

    assert start.hci is not None
    assert any(delta.key == "company" for delta in start.hci.deltas)

    assert controller.company is not None
    supply = next(iter(controller.definitions.supplies.catalog.values()))
    controller.company.coin = supply.cost
    result = controller.handle(BuySupply(supply.id))

    assert result.hci is not None
    summaries = hci_summary_lines(result.hci)
    assert "Action" in summaries
    assert "Changed" in summaries
    assert "Danger / Condition" in summaries
    assert "Next" in summaries
    assert any("Coin" in summary for summary in summaries)
    assert any(supply.id.replace("_", " ") in summary for summary in summaries)
    assert sum("Coin" in summary for summary in summaries) == 1


def test_action_provider_defaults_are_safe_and_disabled_actions_explain() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    company.coin = 0

    town = build_town_dashboard(company, definitions)
    focused = next(action for action in town.services if action.default)

    assert focused.risk not in {
        ScreenActionRisk.COSTLY,
        ScreenActionRisk.RISKY,
        ScreenActionRisk.IRREVERSIBLE,
    }
    assert all(
        action.unavailable_reason
        for action in town.services
        if not action.enabled and action.value != "back"
    )

    shop_actions = ActionProvider.supply_shop_actions(company, definitions)
    disabled_shop = [action for action in shop_actions if not action.enabled]
    assert disabled_shop
    assert all(action.unavailable_reason for action in disabled_shop)
    assert all("Budget" in action.preview for action in shop_actions if action.value != "back")


def test_shared_hci_text_helpers_explain_disabled_and_unsafe_defaults() -> None:
    action = ScreenAction(
        "1",
        "Recruit",
        "recruit",
        ("r",),
        enabled=False,
        default=True,
        description="Hire a recruit.",
        risk=ScreenActionRisk.COSTLY,
        unavailable_reason="Need 2 Coin.",
        confirm="Spend Coin?",
    )

    assert unavailable_message(action) == "Recruit is unavailable. Need 2 Coin."

    detail = generic_action_detail(action, safe_default=False)
    assert "Risk: Costly" in detail
    assert "Marked default, but Enter will avoid it because it is not safe." in detail
    assert "Need 2 Coin." in detail
    assert "Spend Coin?" in detail


def test_event_priority_classifies_noise_normal_and_critical() -> None:
    noise = RoundStartedEvent(
        message="Round 1 begins.",
        encounter_id="test",
        round_number=1,
        actor_ids=[],
    )
    normal = DamageEvent(
        message="Hero hits for 2.",
        source_id="hero_a",
        target_id="enemy_a",
        amount=2,
    )
    critical = BreachDiscoveredEvent(
        message="The breach opens.",
        node_id="maze_breach",
        breach_id="maze",
    )

    assert event_importance(noise) == EventImportance.NOISE
    assert event_importance(normal) == EventImportance.NORMAL
    assert event_importance(critical) == EventImportance.CRITICAL
    assert build_event_beats([noise, normal, critical])[-1].importance == EventImportance.CRITICAL


def test_tactical_brief_prioritizes_combat_condition() -> None:
    before = StateSnapshot(
        has_company=True,
        company_name="Brief Charter",
        combat_encounter_id="test_fight",
        heroes=(
            HeroStateSnapshot(
                "hero_watchman",
                "Mara Vell",
                hp=4,
                max_hp=8,
                effort=2,
                max_effort=3,
                mortal_wounds=0,
            ),
        ),
    )
    after = StateSnapshot(
        has_company=True,
        company_name="Brief Charter",
        combat_encounter_id="test_fight",
        heroes=(
            HeroStateSnapshot(
                "hero_watchman",
                "Mara Vell",
                hp=0,
                max_hp=8,
                effort=2,
                max_effort=3,
                mortal_wounds=1,
                statuses=("downed",),
            ),
        ),
    )
    hci = build_hci_result_analysis(
        before,
        after,
        [DownedEvent(message="Mara Vell is downed.", actor_id="hero_watchman")],
    )

    lines = tactical_brief_lines(hci)

    assert lines.index("Changed") < lines.index("Danger / Condition")
    assert any("Mara Vell HP: 4->0" in line for line in lines)
    assert any("Mara Vell is downed" in line for line in lines)
    assert any("Resolve the active combat turn" in line for line in lines)


def test_dungeon_actions_use_provider_metadata() -> None:
    controller = AppController()
    controller.handle(StartNewCompany("Provider Charter"))
    result = controller.handle(StartExpedition(interactive_dungeon=True))

    assert result.hci is not None
    assert controller.company is not None
    view_result = controller.handle(ViewDungeon())
    assert isinstance(view_result.value, DungeonView)
    view = view_result.value

    assert view.actions
    assert all(action.preview or action.unavailable_reason for action in view.actions)
    disabled = [action for action in view.actions if not action.enabled]
    assert disabled
    assert all(action.unavailable_reason for action in disabled)
    route_actions = [action for action in view.actions if str(action.kind) == "travel"]
    assert route_actions
    assert any("Destination:" in action.preview for action in route_actions)
    assert any("Unexplored room" in action.result_hint for action in route_actions)


def test_combat_actions_preview_actor_target_and_commitment() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany("Combat Brief"))
    controller.handle(StartManualCombat("shallow_cave"))
    result = controller.handle(ViewCombat())
    assert isinstance(result.value, CombatView)
    view = result.value
    assert view.current_actor is not None
    skill = next(skill for skill in view.skills if skill.action.enabled)

    assert view.current_actor.name in skill.action.preview
    assert "Choose a legal target" in skill.action.result_hint

    controller.handle(ChooseCombatSkill(skill.skill_id))
    target_result = controller.handle(ViewCombat())
    assert isinstance(target_result.value, CombatView)
    target_view = target_result.value
    target = target_view.targets[0]

    assert "HP" in target.action.preview
    assert "Projected:" in target.action.preview
    assert "Enter commits" in target.action.result_hint


def test_confirmation_defaults_cancel_and_dock_stays_compact() -> None:
    actions = ActionProvider.confirmation_actions(
        "Overwrite",
        "Cancel",
        consequence="Overwrite the save slot.",
        irreversible=True,
    )

    assert actions[0].default
    assert not actions[1].default
    assert actions[0].result_hint == "Safe default: no state changes."

    dock_text = CommandDock.render_text(actions, 0, width=0)
    assert "[Overwrite the save slot.]" not in dock_text
    assert "!Overwrite" in dock_text


def test_save_result_hci_feeds_resolution_summary(tmp_path: Path) -> None:
    controller = AppController()
    controller.handle(StartNewCompany("Save Charter"))
    result = controller.handle(SaveGame(tmp_path / "company.json"))

    assert result.hci is not None
    assert result.hci.beats
    panel = render_resolution_card(result.events, result.hci)
    text = str(panel.renderable)
    assert "Action" in text
    assert "Changed" in text
    assert "Next" in text
    assert "Saved Save Charter" in text

    cli_panel = render_hci_summary(result.hci, title="Result")
    assert "Changed" in str(cli_panel.renderable)


def test_format_party_watch_renders_active_party_grid() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    view = build_formation_view(company, definitions)

    body = format_party_watch(view)
    lines = body.splitlines()

    assert lines[0] == "Party Watch"
    assert "Ilyra" in lines[1]
    assert "Mara" in lines[1]
    assert "Orren" in lines[2]
    assert "Senn" in lines[2]
    assert len(lines[1]) >= PARTY_WATCH_COLUMN_WIDTH
    assert len(lines[2]) >= PARTY_WATCH_COLUMN_WIDTH


def test_format_party_watch_shows_wound_badge_for_mortal_wounds() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    hero = next(item for item in company.roster if item.hero_id == "hero_field_surgeon")
    hero.mortal_wounds = 1
    hero.hp = hero.max_hp
    view = build_formation_view(company, definitions)

    slot = next(item for item in view.slots if item.hero_id == hero.hero_id)
    assert format_party_watch_slot(slot).endswith("[xoo]")

    body = format_party_watch(view)
    assert "Ilyra" in body
    assert "[xoo]" in body
    assert "Mara" in body
    assert body.count("[xoo]") == 1


def test_format_party_watch_renders_empty_slots() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    company.active_party_slots[FormationSlot.BACK_LEFT] = None
    view = build_formation_view(company, definitions)

    slot = next(item for item in view.slots if item.slot_label == "BACK_LEFT")
    assert format_party_watch_slot(slot) == "—"

    body = format_party_watch(view)
    assert "—" in body


def test_format_party_watch_returns_empty_when_no_active_heroes() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    for slot in FormationSlot:
        company.active_party_slots[slot] = None

    assert format_party_watch(build_formation_view(company, definitions)) == ""
