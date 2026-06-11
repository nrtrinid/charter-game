from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.text import Text
from textual.widgets import Input

from game.app.commands import (
    BuySupply,
    ChooseCombatSkill,
    MoveDungeon,
    PassCombatTurn,
    ResolveCombatAction,
    ResolveCombatReaction,
    StartExpedition,
    StartManualCombat,
    StartNewCompany,
    UseDungeonAction,
    ViewCombat,
    ViewDungeon,
    ViewRegionalMap,
    ViewWorld,
)
from game.app.controller import AppController
from game.app.manual_combat import auto_advance_to_hero, legal_skill_ids, legal_target_ids
from game.app.views import (
    CombatView,
    DungeonMapNodeView,
    DungeonRoomView,
    DungeonView,
    GearInventoryView,
    HeroListEntry,
    ScreenAction,
    SupplyShopView,
    WorldView,
    build_formation_view,
    build_hero_sheet_view,
)
from game.campaign.company import (
    CompanyState,
    DungeonMemoryState,
    HeroMemoryEntry,
    create_new_company,
)
from game.campaign.hero_memory import FreshMemoryState
from game.combat.combat_state import ActorStatus
from game.combat.turn_order import InitiativeEntry
from game.core.events import (
    CombatEffectEvent,
    CombatEndedEvent,
    DamageEvent,
    DeathEvent,
    DownedEvent,
    EnemyIntentEvent,
    ExpeditionEvent,
    HealingEvent,
    MissEvent,
    MoveEvent,
    RoundEndedEvent,
    SkillUsedEvent,
    StatusChangedEvent,
)
from game.core.rng import GameRng
from game.ui.hci_text import format_compact_roster_row, format_formation_slot
from game.ui.tui import CharterApp
from game.ui.tui_widgets import (
    BodyPane,
    CombatPanel,
    CommandDock,
    DetailPane,
    DungeonMapPanel,
    DungeonRoomPanel,
    LogPane,
    StatusHeader,
    _mini_slot_nudge,
    _normalize_mini_lines,
)
from game.ui.wounds import mortal_wound_badge
from tests.conftest import OPENING_DUNGEON_TO_WORKS_CACHE, get_definitions

UNSAFE_FOCUS_RISKS = {"costly", "risky", "irreversible"}


def assert_focused_action_is_safe(app: CharterApp) -> None:
    assert app.focused_action is not None
    assert str(app.focused_action.risk) not in UNSAFE_FOCUS_RISKS


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_combat_panel_does_not_truncate_rich_markup_in_grid_cells() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    result = controller.handle(ViewCombat())

    assert isinstance(result.value, CombatView)
    body = CombatPanel.render_text(
        result.value,
        phase="command",
        focused_action=result.value.commands[0],
    )

    assert "[bold cyan]" in body
    assert "[bold bright[/]" not in body


def test_mini_art_normalization_centers_authored_block_not_each_row() -> None:
    head, body, legs = _normalize_mini_lines((" o ", "/|", "/_\\"))

    assert head.index("o") == body.index("|")
    assert body.index("/") == legs.index("/")


def test_mini_slot_nudge_is_stable_for_tactical_slot() -> None:
    first = SimpleNamespace(actor_id="hero_one", team="hero", slot="BACK_LEFT")
    second = SimpleNamespace(actor_id="hero_two", team="hero", slot="BACK_LEFT")
    moved = SimpleNamespace(actor_id="hero_one", team="hero", slot="BACK_RIGHT")

    assert _mini_slot_nudge(first) == _mini_slot_nudge(second)
    assert _mini_slot_nudge(first) != _mini_slot_nudge(moved)


def test_combat_panel_renders_open_mini_formation_with_compact_hp() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    result = controller.handle(ViewCombat())

    assert isinstance(result.value, CombatView)
    body = CombatPanel.render_text(
        result.value,
        phase="command",
        focused_action=result.value.commands[0],
        idle_frame=0,
    )
    plain = Text.from_markup(body).plain

    assert "PARTY" in plain
    assert "ENEMIES" in plain
    assert "VS" in plain
    assert "BACK LEFT" not in plain
    assert "FRONT RIGHT" not in plain
    assert "empty" not in plain.lower()
    assert any("Ilyra Penn" in line and "Mara Vell" in line for line in plain.splitlines())
    assert any("Orren Vale" in line and "Senn Crowe" in line for line in plain.splitlines())
    assert any(
        "Bone Soldier" in line and "Loop-Touched Skulk" in line for line in plain.splitlines()
    )
    assert any("Bone Soldier" in line and "Maze Leech" in line for line in plain.splitlines())
    assert "[#]" in plain
    assert "[#\\" in plain
    assert "[TARGET]" not in plain
    assert "[LOW]" not in plain
    assert "HP " in plain
    assert "HP 14/14" in plain
    assert "HP 18/18" in plain
    assert "EF " not in plain


def test_combat_panel_shows_mortal_wounds_as_badge() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    result = controller.handle(ViewCombat())

    assert isinstance(result.value, CombatView)
    wounded = replace(result.value.party[0], mortal_wounds=1)
    wounded_view = replace(
        result.value,
        party=(wounded, *result.value.party[1:]),
    )
    body = CombatPanel.render_text(
        wounded_view,
        phase="command",
        focused_action=wounded_view.commands[0],
    )
    plain = Text.from_markup(body).plain

    assert "[xoo]" in plain


def test_mortal_wound_badge_renders_zero_wounds_through_markup() -> None:
    badge = mortal_wound_badge(0, markup_safe=True)
    plain = Text.from_markup(f"- Wounds: {badge}").plain
    assert plain == "- Wounds: [ooo]"


def test_hero_sheet_shows_zero_wound_marker() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    company = controller.company
    assert company is not None
    hero = company.roster[0]
    hero.mortal_wounds = 0

    view = build_hero_sheet_view(company, controller.definitions, hero.hero_id)
    assert view is not None

    app = CharterApp()
    text = app._hero_sheet_text(view)
    plain = Text.from_markup(text).plain
    assert "- Wounds: [ooo]" in plain


@pytest.mark.anyio
async def test_tui_enemy_first_start_shows_enemy_beat_before_command() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    enemy_id = next(iter(session.state.enemies))
    hero_id = next(iter(session.state.heroes))
    session.initiative = [
        InitiativeEntry(enemy_id, 99),
        InitiativeEntry(hero_id, 98),
    ]
    session.turn_index = 0
    enemy_events = auto_advance_to_hero(session, controller.definitions, GameRng(7))
    assert any(isinstance(event, SkillUsedEvent) for event in enemy_events)
    session.event_log.extend(enemy_events)
    session.recent_events = enemy_events[-8:]
    app = CharterApp(controller=controller)
    playback_events = app._meaningful_playback_events(list(session.event_log))

    assert any(isinstance(event, SkillUsedEvent) for event in session.event_log)
    assert not any(
        isinstance(event, SkillUsedEvent | EnemyIntentEvent) for event in playback_events
    )
    assert not any("uses" in event.message for event in playback_events)

    async with app.run_test() as pilot:
        app._show_current_place()

        assert app.screen_state == "enemy_turn"
        assert "Enemy Response" in app.body_text
        assert "Combat Command" not in app.body_text
        assert "VS" in app.body_text

        for _ in range(4):
            await pilot.press("enter")
            await pilot.pause()
            if app.screen_state != "enemy_turn":
                break

        assert app.screen_state == "combat"
        assert "Combat Command" in app._live_body_text()


@pytest.mark.anyio
async def test_tui_opening_enemy_move_shows_before_command() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    enemy_id = next(iter(session.state.enemies))
    enemy = session.state.enemies[enemy_id]
    hero_id = next(iter(session.state.heroes))
    session.initiative = [
        InitiativeEntry(enemy_id, 99),
        InitiativeEntry(hero_id, 98),
    ]
    session.turn_index = 0
    move_event = MoveEvent(
        message=f"{enemy.name} shifts: BACK_LEFT -> FRONT_LEFT.",
        actor_id=enemy_id,
        from_slot="BACK_LEFT",
        to_slot="FRONT_LEFT",
    )
    session.event_log = [move_event]
    session.recent_events = [move_event]
    app = CharterApp(controller=controller)

    async with app.run_test() as pilot:
        app._show_current_place()

        assert app.screen_state == "enemy_turn"
        assert "Enemy Response" in app.body_text
        assert "Formation" in app.body_text
        assert enemy.name in app.body_text
        assert "Combat Command" not in app.body_text

        await pilot.press("enter")
        await pilot.pause()

        assert app.screen_state == "combat"
        assert "Combat Command" in app._live_body_text()


def test_split_turn_events_classifies_enemy_move_as_enemy_response() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    app = CharterApp(controller=controller)
    view_result = controller.handle(ViewCombat())
    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    hero = view.party[0]
    enemy = view.enemies[0]
    events = [
        SkillUsedEvent(
            message=f"{hero.name} uses Test Strike on {enemy.name}.",
            actor_id=hero.actor_id,
            skill_id="test_strike",
            target_id=enemy.actor_id,
        ),
        DamageEvent(
            message=f"{enemy.name} takes 3 damage.",
            source_id=hero.actor_id,
            target_id=enemy.actor_id,
            amount=3,
        ),
        MoveEvent(
            message=f"{enemy.name} shifts: FRONT_LEFT -> BACK_LEFT.",
            actor_id=enemy.actor_id,
            from_slot="FRONT_LEFT",
            to_slot="BACK_LEFT",
        ),
    ]

    hero_events, enemy_events = app._split_turn_events(events)

    assert [type(event).__name__ for event in hero_events] == ["SkillUsedEvent", "DamageEvent"]
    assert len(enemy_events) == 1
    assert isinstance(enemy_events[0], MoveEvent)
    beats = app._combat_event_beats(enemy_events)
    assert len(beats) == 1
    assert isinstance(beats[0][0], MoveEvent)

    text = app._resolution_text(events)
    hero_section, enemy_section = text.split("Enemy Response", 1)
    assert "shifts" not in hero_section
    assert "shifts" in enemy_section


def test_tui_result_log_uses_tactical_brief_sections() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany("TUI Brief"))
    assert controller.company is not None
    supply = next(iter(controller.definitions.supplies.catalog.values()))
    controller.company.coin = supply.cost
    result = controller.handle(BuySupply(supply.id))
    app = CharterApp(controller=controller)

    text = app._result_log_text(result.events, result.hci)

    assert "Action" in text
    assert "Changed" in text
    assert "Danger / Condition" in text
    assert "Next" in text
    assert "Coin" in text


def test_combat_panel_uses_focused_healing_highlight() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    session.initiative = [
        InitiativeEntry("hero_field_surgeon", 99),
        InitiativeEntry("hero_watchman", 98),
    ]
    session.turn_index = 0
    session.state.heroes["hero_watchman"].hp = 0
    session.state.heroes["hero_watchman"].statuses.add(ActorStatus.DOWNED)

    result = controller.handle(ChooseCombatSkill("emergency_stitch"))

    assert result.success
    assert isinstance(result.value, CombatView)
    view = result.value
    guard_target = next(target for target in view.targets if target.target_id == "hero_watchman")
    body = CombatPanel.render_text(
        view,
        phase="target",
        focused_action=guard_target.action,
    )
    detail = CombatPanel.detail_text(view, "target", guard_target.action)

    assert "PARTY" in body
    assert "ENEMIES" in body
    assert "Turn" in body
    assert "Ilyra Penn" in body
    assert "Target Focus" in detail
    assert "Targets for Emergency Dressing" not in body
    assert "Mara Vell" in detail
    assert "FOCUS v" in body
    assert "HEAL" not in body
    assert "[bold bright_green]" in body
    assert "[bold white on green]" not in body
    assert "[bold white on red]" not in body
    assert "BACK LEFT" not in body
    assert "FRONT RIGHT" not in body
    assert "Portrait" in detail
    assert "Portraits" not in detail
    assert "Ilyra Penn" not in detail
    assert "Mara Vell" in detail
    assert "[|#|\\" in Text.from_markup(detail).plain
    assert "DOWNED" in detail
    assert "magenta" in detail


def test_combat_panel_mini_idle_frames_animate_in_open_field() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    result = controller.handle(ViewCombat())

    assert isinstance(result.value, CombatView)
    frames = {
        Text.from_markup(
            CombatPanel.render_text(
                result.value,
                phase="command",
                focused_action=result.value.commands[0],
                idle_frame=frame,
            )
        ).plain
        for frame in range(8)
    }

    assert len(frames) > 1
    assert any("[#\\" in frame for frame in frames)
    assert any("[#|" in frame for frame in frames)


def test_combat_panel_caps_mini_status_tags_at_two_visible_tags() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    result = controller.handle(ViewCombat())

    assert isinstance(result.value, CombatView)
    view = result.value
    actor = view.party[0]
    marked_actor = replace(
        actor,
        statuses=("guarded", "warded", "bleeding"),
        tags=("marked",),
        hp=max(1, actor.max_hp // 4),
    )
    marked_view = replace(
        view,
        party=tuple(
            marked_actor if combatant.actor_id == actor.actor_id else combatant
            for combatant in view.party
        ),
    )
    body = CombatPanel.render_text(
        marked_view,
        phase="command",
        focused_action=marked_view.commands[0],
    )
    plain = Text.from_markup(body).plain

    assert "[GUARD] [WARD]" in plain
    assert "[BLEED]" not in plain
    assert "[LOW]" not in plain


def test_combat_board_keeps_state_dense_and_alert_only() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    result = controller.handle(ViewCombat())

    assert isinstance(result.value, CombatView)
    view = result.value
    actor = view.current_actor
    assert actor is not None
    stressed_actor = replace(
        actor,
        morale="Shaken",
        strain="Worn",
        strain_marks=("Winded", "Drained", "Battered", "Frayed"),
    )
    stressed_view = replace(
        view,
        current_actor=stressed_actor,
        party=tuple(
            stressed_actor if combatant.actor_id == actor.actor_id else combatant
            for combatant in view.party
        ),
    )

    body = Text.from_markup(
        CombatPanel.render_text(
            stressed_view,
            phase="command",
            focused_action=stressed_view.commands[0],
        )
    ).plain

    assert "Cohesion" in body
    assert "Morale Shaken" not in body
    assert "Strain Worn" not in body
    assert "Winded" not in body
    assert "Drained" not in body
    assert "Battered" not in body
    assert "Frayed" not in body


def test_combat_board_shows_spent_as_alert_badge() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    result = controller.handle(ViewCombat())

    assert isinstance(result.value, CombatView)
    view = result.value
    actor = view.current_actor
    assert actor is not None
    spent_actor = replace(actor, strain="Spent")
    spent_view = replace(
        view,
        current_actor=spent_actor,
        party=tuple(
            spent_actor if combatant.actor_id == actor.actor_id else combatant
            for combatant in view.party
        ),
    )

    body = Text.from_markup(
        CombatPanel.render_text(
            spent_view,
            phase="command",
            focused_action=spent_view.commands[0],
        )
    ).plain

    assert "[SPENT]" in body
    assert "Strain Spent" not in body


def test_combat_focus_detail_explains_strain_effects() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    result = controller.handle(ViewCombat())

    assert isinstance(result.value, CombatView)
    view = result.value
    actor = view.current_actor
    assert actor is not None
    strained_actor = replace(
        actor,
        morale="Shaken",
        strain="Worn",
        strain_marks=("Winded", "Drained", "Battered", "Frayed"),
    )
    strained_view = replace(view, current_actor=strained_actor)

    detail = CombatPanel.detail_text(strained_view, "command", strained_view.commands[0])

    assert "Morale Shaken" in detail
    assert "Strain Worn" in detail
    assert "Effects:" in detail
    assert "Winded: cannot move" in detail
    assert "Drained: -1 Effort at combat start" in detail
    assert "Battered: -1 Defense" in detail
    assert "Frayed: morale pressure" in detail


def test_combat_focus_detail_explains_spent_without_duplicate_marks() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    result = controller.handle(ViewCombat())

    assert isinstance(result.value, CombatView)
    view = result.value
    actor = view.current_actor
    assert actor is not None
    spent_actor = replace(
        actor,
        strain="Spent",
        strain_marks=("Winded", "Drained", "Battered", "Frayed"),
    )
    spent_view = replace(view, current_actor=spent_actor)

    detail = CombatPanel.detail_text(spent_view, "command", spent_view.commands[0])

    assert "Strain Spent" in detail
    assert detail.count("Spent: counts as Winded, Drained, and Frayed") == 1
    assert "Battered: -1 Defense" in detail
    assert "Winded: cannot move" not in detail
    assert "Drained: -1 Effort at combat start" not in detail
    assert "Frayed: morale pressure" not in detail


def test_combat_focus_detail_explains_status_tags_and_combat_quirks() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    result = controller.handle(ViewCombat())

    assert isinstance(result.value, CombatView)
    view = result.value
    actor = view.current_actor
    assert actor is not None
    tagged_actor = replace(
        actor,
        statuses=("downed",),
        tags=("Stunned", "Knocked_Down", "Frozen", "Guarded", "Marked"),
        mortal_wounds=2,
        personal_quirk="Hold the Line",
        quirks=("blood_hot", "Ice Nerves"),
    )
    tagged_view = replace(view, current_actor=tagged_actor)

    detail = CombatPanel.detail_text(tagged_view, "command", tagged_view.commands[0])

    assert "Downed: cannot act or protect" in detail
    assert "Mortal Wounds 2: death at 3" in detail
    assert "Stunned: cannot act or protect" in detail
    assert "Knocked Down: cannot act or protect" in detail
    assert "Frozen: cannot act or move" in detail
    assert "Guarded: protecting an ally" in detail
    assert "Marked: easier to target" in detail
    assert "Hold the Line: +1 Defense in front row" in detail
    assert "Blood Hot: attack pressure" in detail
    assert "Ice Nerves: resists frozen/shocked pressure" in detail
    assert "Memory" not in detail
    assert "Bond" not in detail


def test_combat_panel_renders_turn_order_after_special_enemy_action() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("cave_mini_boss"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    session.initiative = [
        InitiativeEntry("cave_maw_brute_1", 99),
        InitiativeEntry("hero_watchman", 98),
    ]
    session.turn_index = 0
    auto_advance_to_hero(session, controller.definitions, GameRng(1))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    assert view.pending_enemy_intent is None
    assert view.reaction_options == ()
    assert view.turn_order
    body = CombatPanel.render_text(
        view,
        phase="command",
        focused_action=view.commands[0],
    )

    assert "Turns  " in body
    assert "Maw" in body
    assert "Mara" in body
    assert "REACTION WINDOW" not in body
    assert "Enemy Intent" not in body
    assert "Skip Reaction" not in body


def test_tui_resolution_groups_enemy_drag_under_enemy_response() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("cave_mini_boss"))
    assert controller.manual_combat is not None
    session = controller.manual_combat
    session.initiative = [
        InitiativeEntry("cave_maw_brute_1", 99),
        InitiativeEntry("hero_watchman", 98),
    ]
    session.turn_index = 0
    events = auto_advance_to_hero(session, controller.definitions, GameRng(1))
    app = CharterApp(controller=controller)

    text = app._resolution_text(events)

    hero_section, enemy_section = text.split("Enemy Response", 1)
    assert "is dragged" not in hero_section
    assert "is dragged" in enemy_section


def test_combat_move_menu_uses_destination_first_labels_and_preview() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    assert controller.manual_combat is not None
    controller.manual_combat.initiative = [
        InitiativeEntry("hero_watchman", 99),
        InitiativeEntry("hero_field_surgeon", 98),
    ]
    controller.manual_combat.turn_index = 0

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    move = next(option for option in view.moves if option.to_slot == "BACK_LEFT")
    body = CombatPanel.render_text(view, phase="move", focused_action=move.action)
    detail = CombatPanel.detail_text(view, "move", move.action)
    dock_help = CombatPanel.command_help_text(view, "move", move.action)

    assert move.action.label == "Back Left — swap with Ilyra Penn"
    assert "Swap Mara Vell with Ilyra Penn" not in body
    assert "Movement" not in body
    assert "[1]" in body
    assert "BACK_LEFT (Ilyra Penn)" not in body
    assert "Move Preview" in detail
    assert "Actor: Mara Vell" in detail
    assert "Destination: Back Left" in detail
    assert "Formation Preview" in detail
    assert "[MV  ] [SC  ]  ->  [IP  ] [SC  ]" in detail
    assert "[IP  ] [OV  ]  ->  [MV  ] [OV  ]" in detail
    assert "Swap target: Ilyra Penn" in detail
    assert "Result: Turn ends and protection changes immediately." in detail
    assert "Formation Preview" in dock_help


def test_formation_dock_uses_player_facing_slot_labels() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    view = build_formation_view(company, definitions)

    assert format_formation_slot("BACK_LEFT") == "Back Left"
    assert any(action.label.startswith("Back Left:") for action in view.actions)
    assert not any(action.label.startswith("Edit ") for action in view.actions)


def test_formation_board_renders_mini_portraits() -> None:
    from game.app.views import build_hero_portrait_view
    from game.ui.tui_widgets import FormationBoard

    definitions = get_definitions()
    company = create_new_company(definitions)
    view = build_formation_view(company, definitions)
    roster_by_id = {hero.hero_id: hero for hero in company.roster}
    actors = {
        slot.slot_label: build_hero_portrait_view(
            roster_by_id[slot.hero_id],
            definitions,
            slot=slot.slot_label,
        )
        for slot in view.slots
        if slot.hero_id is not None and slot.hero_id in roster_by_id
    }

    body = FormationBoard.render_mini_text(actors, idle_frame=0)

    assert "Party Formation" in body
    assert "Back Left" in body
    assert "Front Left" in body
    assert company.roster[0].name in body


def test_formation_board_inward_facing_keeps_authored_art() -> None:
    from game.app.views import build_hero_portrait_view
    from game.ui.tui_widgets import FormationBoard, formation_slot_faces_inward

    definitions = get_definitions()
    company = create_new_company(definitions)
    view = build_formation_view(company, definitions)
    roster_by_id = {hero.hero_id: hero for hero in company.roster}
    actors = {
        slot.slot_label: build_hero_portrait_view(
            roster_by_id[slot.hero_id],
            definitions,
            slot=slot.slot_label,
        )
        for slot in view.slots
        if slot.hero_id is not None and slot.hero_id in roster_by_id
    }

    assert formation_slot_faces_inward("FRONT_LEFT")
    assert formation_slot_faces_inward("FRONT_RIGHT")
    assert not formation_slot_faces_inward("BACK_LEFT")
    assert not formation_slot_faces_inward("BACK_RIGHT")

    outward = FormationBoard.render_mini_text(actors, idle_frame=0, inward_facing=False)
    inward = FormationBoard.render_mini_text(actors, idle_frame=0, inward_facing=True)
    assert outward == inward


def test_enemy_mini_art_renders_authored_lines() -> None:
    from types import SimpleNamespace

    from game.ui.tui_widgets import _mini_art_lines

    lines = ("_0_", "/I\\", "/ \\")

    enemy = SimpleNamespace(team="enemy", mini_lines=lines, mini_frames={})
    rendered = _mini_art_lines(enemy, 0)

    assert any("/I\\" in line for line in rendered)


def test_enemy_mini_art_preserves_left_facing_wolf() -> None:
    from types import SimpleNamespace

    from game.ui.tui_widgets import _mini_art_lines

    lines = ("   /^\\   ", "  <o o\\  ", "  /ww/   ")

    wolf = SimpleNamespace(team="enemy", mini_lines=lines, mini_frames={})
    rendered = _mini_art_lines(wolf, 0)

    assert any("<o o" in line for line in rendered)
    assert all(">o o" not in line for line in rendered)


def test_formation_board_keeps_centered_mini_art_without_mirror() -> None:
    from game.app.views import build_hero_portrait_view
    from game.ui.tui_widgets import FormationBoard

    definitions = get_definitions()
    company = create_new_company(definitions)
    view = build_formation_view(company, definitions)
    roster_by_id = {hero.hero_id: hero for hero in company.roster}
    actors = {
        slot.slot_label: build_hero_portrait_view(
            roster_by_id[slot.hero_id],
            definitions,
            slot=slot.slot_label,
        )
        for slot in view.slots
        if slot.hero_id is not None and slot.hero_id in roster_by_id
    }

    body = FormationBoard.render_mini_text(actors, idle_frame=0, inward_facing=False)

    assert "  o   " in body
    assert "<+\\" not in body


def test_roster_focus_detail_includes_portrait() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    app = CharterApp(controller=controller)
    hero_id = controller.company.roster[0].hero_id

    detail = app._hero_sheet_preview_detail(hero_id)

    assert "Portrait" in detail
    assert controller.company.roster[0].name in detail


def test_roster_compact_row_omits_memory_text() -> None:
    hero = HeroListEntry(
        hero_id="hero_one",
        name="Mara Vell",
        class_id="guard",
        slot="FRONT_LEFT",
        hp=20,
        max_hp=24,
        effort=4,
        max_effort=4,
        mortal_wounds=0,
        morale="Steady",
        strain="Steady",
        life_state="alive",
        latest_memory="survived Opening with wounds recorded.",
        memory_count=3,
    )

    row = format_compact_roster_row(hero)

    assert "Front Left" in row
    assert "survived Opening" not in row
    assert "Memory:" not in row


def test_combat_move_beat_labels_formation_change() -> None:
    event = MoveEvent(
        message="Mara Vell shifts: FRONT_LEFT -> BACK_LEFT.",
        actor_id="hero_watchman",
        from_slot="FRONT_LEFT",
        to_slot="BACK_LEFT",
    )

    text = CombatPanel.render_combat_beat(
        None,
        [event],
        title="Hero Action",
        source_actor_ids={"hero_watchman"},
        target_intents={},
    )

    assert "Formation" in text
    assert "Mara Vell shifts" in text
    assert "FRONT_LEFT -> BACK_LEFT" in text


def test_combat_command_dock_focus_help_by_phase() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    assert controller.manual_combat is not None
    controller.manual_combat.initiative = [
        InitiativeEntry("hero_scribe", 99),
        InitiativeEntry("hero_watchman", 98),
    ]
    controller.manual_combat.turn_index = 0

    command_result = controller.handle(ViewCombat())

    assert isinstance(command_result.value, CombatView)
    command_view = command_result.value
    command_help = CombatPanel.command_help_text(
        command_view,
        "command",
        command_view.commands[0],
    )
    assert "available" in command_help

    skill = next(skill for skill in command_view.skills if skill.action.enabled)
    skill_help = CombatPanel.command_help_text(command_view, "skill", skill.action)
    assert "EF:" in skill_help
    assert "legal target" in skill_help

    target_result = controller.handle(ChooseCombatSkill(skill.skill_id))
    assert isinstance(target_result.value, CombatView)
    target_view = target_result.value
    target = target_view.targets[0]
    target_actor = next(
        combatant
        for combatant in (*target_view.party, *target_view.enemies)
        if combatant.actor_id == target.target_id
    )
    target_details = "\n".join(
        CombatPanel.detail_text(target_view, "target", target.action, idle_frame=frame)
        for frame in range(4)
    )
    breathing_idle_line = target_actor.art_frames["idle"][1][1].rstrip()
    target_help = CombatPanel.command_help_text(target_view, "target", target.action)
    target_dock = CommandDock.render_text(
        (target.action,),
        0,
        help_text=target_help,
    )
    wide_target_dock = CommandDock.render_text(
        (target.action,),
        0,
        help_text=target_help,
        width=110,
    )
    assert "Focus" in target_dock
    assert "hit" in target_dock
    assert target.legality_reason in target_dock
    assert target_actor.art_frames["idle"][1][0].strip()
    assert target_actor.art_lines[0].strip() in target_details
    assert breathing_idle_line != target_actor.art_lines[1].rstrip()
    assert breathing_idle_line in target_details
    assert "Hint" in wide_target_dock
    assert "Focus" not in wide_target_dock
    assert "  |  " in wide_target_dock
    assert target.legality_reason in wide_target_dock


def test_combat_skill_focus_help_shows_flavor_and_effect_copy() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    assert controller.manual_combat is not None
    controller.manual_combat.initiative = [
        InitiativeEntry("hero_cutpurse", 99),
        InitiativeEntry("hero_field_surgeon", 98),
    ]
    controller.manual_combat.turn_index = 0

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    cutpurse_skill = next(skill for skill in view.skills if skill.skill_id == "exposed_cut")
    cutpurse_help = CombatPanel.command_help_text(view, "skill", cutpurse_skill.action)
    assert "On the Mark" in cutpurse_help
    assert "EF: 1" in cutpurse_help
    assert "ranged" in cutpurse_help
    assert "Any" in cutpurse_help
    assert "Wherever they point, a dagger soon follows." in cutpurse_help
    assert "Effect: Throw a knife at one enemy." in cutpurse_help
    assert "extra damage to Marked, wounded, or exposed backline targets" in cutpurse_help

    controller.manual_combat.initiative = [
        InitiativeEntry("hero_field_surgeon", 99),
        InitiativeEntry("hero_cutpurse", 98),
    ]
    controller.manual_combat.turn_index = 0
    view_result = controller.handle(ViewCombat())
    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    dressing = next(skill for skill in view.skills if skill.skill_id == "emergency_stitch")
    dressing_help = CombatPanel.command_help_text(view, "skill", dressing.action)
    assert "Emergency Dressing" in dressing_help
    assert "Fast cloth, hard pressure" in dressing_help
    assert "Effect: Heal one living ally." in dressing_help
    assert "Heals more if the ally is Downed or at half HP or lower." in dressing_help


def test_command_dock_keeps_skill_labels_clean_and_marks_only_risky_actions() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    costly_skill = next(skill for skill in view.skills if skill.action.cost)
    skill_dock = CommandDock.render_text((costly_skill.action,), 0)
    risky_action = ScreenAction("1", "Descend", "descend", risk="risky")
    risky_dock = CommandDock.render_text((risky_action,), 0)

    assert costly_skill.action.label in skill_dock
    assert costly_skill.action.cost not in skill_dock
    assert f"!{costly_skill.action.label}" not in skill_dock
    assert "!Descend" in risky_dock


def test_pack_panel_uses_compact_tables_without_purpose_blocks() -> None:
    from game.ui.tui_widgets import PackPanel

    gear = GearInventoryView(
        reputation=1,
        coin=5,
        can_manage=False,
        can_purchase=False,
        manage_reason="Gear purchases are only available in Haven.",
    )
    body = PackPanel.render_text({"ration": 2}, {"cave_relic": 1}, gear)

    assert "Pack" in body
    assert "Supplies" in body
    assert "Ration" in body
    assert "x2" in body
    assert "Purpose" not in body
    assert "Gear purchases are only available in Haven." in body


def test_supply_shop_panel_keeps_stock_in_table_form() -> None:
    from game.ui.tui_widgets import SupplyShopPanel

    view = SupplyShopView(
        reputation=0,
        coin=10,
        actions=(
            ScreenAction(
                "1",
                "Rations",
                "rations",
                enabled=True,
                cost="2 Coin",
                description="Owned 0",
            ),
            ScreenAction("2", "Back", "back"),
        ),
    )
    body = SupplyShopPanel.render_text(view)

    assert "Quartermaster" in body
    assert "Coin 10" in body
    assert "Supply" in body
    assert "Rations" in body
    assert "Purpose" not in body


def test_command_dock_keeps_locked_reasons_out_of_command_rows() -> None:
    action = ScreenAction(
        "1",
        "Breach Stalker Hunt",
        "accept:shallow_cave_breach_hunt",
        enabled=False,
        unavailable_reason="Complete the breach scout first.",
    )

    dock = CommandDock.render_text((action,), 0)

    assert "x Breach Stalker Hunt" in dock
    assert "Locked: Breach Stalker Hunt" not in dock
    assert "Complete the breach scout first." not in dock


def test_command_dock_renders_persistent_shortcut_line_separately() -> None:
    action = ScreenAction("1", "Numbered Hall", "numbered_hall")

    dock = CommandDock.render_text(
        (action,),
        0,
        shortcut_text="Shortcuts: [M] Map  [P] Pack  [C] Company  [?] Help",
    )
    wide_dock = CommandDock.render_text(
        (action,),
        0,
        help_text="Take the focused route.",
        shortcut_text="Shortcuts: [M] Map  [P] Pack  [C] Company  [?] Help",
        width=110,
    )

    assert "Commands" in dock
    assert "Numbered Hall" in dock
    assert "\n\nShortcuts: \\[M] Map  \\[P] Pack  \\[C] Company  \\[?] Help" in dock
    assert wide_dock.splitlines()[-1] == (
        "Shortcuts: \\[M] Map  \\[P] Pack  \\[C] Company  \\[?] Help"
    )


def test_combat_command_dock_does_not_alarm_routine_combat_actions() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    retreat = next(action for action in view.commands if action.value == "retreat")
    retreat_dock = CommandDock.render_text((retreat,), 0)
    assert "Retreat" in retreat_dock
    assert "!Retreat" not in retreat_dock

    move = next(
        option for option in view.moves if "swap with" in option.action.label.lower()
    )
    move_dock = CommandDock.render_text((move.action,), 0)
    assert "swap with" in move_dock.lower()
    assert "!Back Left" not in move_dock


def test_combat_beat_renderer_uses_compact_result_not_raw_log() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    actor = view.current_actor
    assert actor is not None
    target = view.enemies[0]
    text = CombatPanel.render_combat_beat(
        view,
        (
            SkillUsedEvent(
                message=f"{actor.name} uses Test Strike on {target.name}.",
                actor_id=actor.actor_id,
                skill_id="test_strike",
                target_id=target.actor_id,
            ),
            DamageEvent(
                message=f"{target.name} takes 4 damage.",
                source_id=actor.actor_id,
                target_id=target.actor_id,
                amount=4,
            ),
        ),
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "attack"},
    )

    assert "Actor" not in text
    assert "Target" not in text
    portrait_block = Text.from_markup(text).plain.split("\n\n")[1]
    assert " EF " not in portrait_block
    assert actor.name in text
    assert target.name in text
    assert "[bold red]" in text
    assert "takes 4 damage" in text
    assert "Summary" not in text
    assert "Pressure" not in text
    assert "Log" not in text


def test_combat_beat_renderer_surfaces_effort_drain_at_impact() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    actor = view.enemies[0]
    target = view.current_actor
    assert target is not None
    text = CombatPanel.render_combat_beat(
        view,
        (
            SkillUsedEvent(
                message=f"{actor.name} uses Effort Drain on {target.name}.",
                actor_id=actor.actor_id,
                skill_id="effort_drain",
                target_id=target.actor_id,
            ),
            DamageEvent(
                message=f"{actor.name} deals 0 damage to {target.name}.",
                source_id=actor.actor_id,
                target_id=target.actor_id,
                amount=0,
            ),
            CombatEffectEvent(
                message=f"{target.name} loses 1 Effort: 4 -> 3.",
                actor_id=target.actor_id,
                target_id=target.actor_id,
                effect_type="resource",
                resource="effort",
                label="EF -1",
                delta=-1,
                before=4,
                after=3,
                source_kind="skill",
                source_id="effort_drain",
                emphasis="bad",
            ),
            StatusChangedEvent(
                message=f"{target.name} loses 1 Effort: 4 -> 3 Effort.",
                actor_id=target.actor_id,
                status="effort",
                added=False,
            ),
        ),
        title="Enemy Response",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_frame=2,
    )
    plain = Text.from_markup(text).plain

    assert "EF -1" in plain
    assert "loses 1 Effort: 4 -> 3" in plain
    assert "deals 0 damage" not in plain
    assert plain.count("loses 1 Effort") == 1


def test_combat_beat_renderer_surfaces_blood_hot_effort_restore() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    actor = view.current_actor
    assert actor is not None
    target = view.enemies[0]
    text = CombatPanel.render_combat_beat(
        view,
        (
            SkillUsedEvent(
                message=f"{actor.name} uses Test Strike on {target.name}.",
                actor_id=actor.actor_id,
                skill_id="test_strike",
                target_id=target.actor_id,
            ),
            DamageEvent(
                message=f"{actor.name} deals 4 damage to {target.name}.",
                source_id=actor.actor_id,
                target_id=target.actor_id,
                amount=4,
            ),
            CombatEffectEvent(
                message=f"Blood Hot restores 1 Effort to {actor.name}: 2 -> 3.",
                actor_id=actor.actor_id,
                target_id=actor.actor_id,
                effect_type="resource",
                resource="effort",
                label="EF +1",
                delta=1,
                before=2,
                after=3,
                source_kind="quirk",
                source_id="blood_hot",
                emphasis="good",
            ),
        ),
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "attack", actor.actor_id: "heal"},
        animation_frame=4,
    )
    plain = Text.from_markup(text).plain

    assert "EF +1" in plain
    assert "Effects:" in plain
    assert "Blood Hot: EF +1" in plain


def test_combat_beat_renderer_prioritizes_resource_effect_over_mitigation_callout() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    actor = view.enemies[0]
    target = view.current_actor
    assert target is not None
    text = CombatPanel.render_combat_beat(
        view,
        (
            SkillUsedEvent(
                message=f"{actor.name} uses Effort Drain on {target.name}.",
                actor_id=actor.actor_id,
                skill_id="effort_drain",
                target_id=target.actor_id,
            ),
            DamageEvent(
                message=f"{actor.name} deals 0 damage to {target.name}.",
                source_id=actor.actor_id,
                target_id=target.actor_id,
                amount=0,
            ),
            CombatEffectEvent(
                message=f"{target.name}'s Guard reduces damage by 3.",
                actor_id=target.actor_id,
                target_id=target.actor_id,
                effect_type="mitigation",
                label="Guard -3",
                delta=-3,
                source_kind="tag",
                source_id="guarded",
                emphasis="good",
            ),
            CombatEffectEvent(
                message=f"{target.name} loses 1 Effort: 4 -> 3.",
                actor_id=target.actor_id,
                target_id=target.actor_id,
                effect_type="resource",
                resource="effort",
                label="EF -1",
                delta=-1,
                before=4,
                after=3,
                source_kind="skill",
                source_id="effort_drain",
                emphasis="bad",
            ),
        ),
        title="Enemy Response",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_frame=3,
    )
    plain = Text.from_markup(text).plain

    assert "EF -1" in plain
    assert "loses 1 Effort" in plain
    assert "deals 0 damage" not in plain
    assert "Guard -3" in plain


def test_combat_beat_renderer_adds_status_line_under_damage() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    actor = view.current_actor
    assert actor is not None
    target = view.enemies[0]
    text = CombatPanel.render_combat_beat(
        view,
        (
            SkillUsedEvent(
                message=f"{actor.name} uses Test Strike on {target.name}.",
                actor_id=actor.actor_id,
                skill_id="test_strike",
                target_id=target.actor_id,
            ),
            DamageEvent(
                message=f"{actor.name} deals 2 damage to {target.name}.",
                source_id=actor.actor_id,
                target_id=target.actor_id,
                amount=2,
            ),
            StatusChangedEvent(
                message=f"{target.name} is Marked.",
                actor_id=target.actor_id,
                status="marked",
                added=True,
            ),
        ),
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_frame=4,
    )
    plain = Text.from_markup(text).plain

    assert "deals 2 damage" in plain
    assert "Effects:" in plain
    assert f"{target.name} is Marked." in plain


def test_combat_beat_renderer_keeps_guard_effect_when_guarded_target_dies() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    actor = view.current_actor
    assert actor is not None
    target = view.enemies[0]
    text = CombatPanel.render_combat_beat(
        view,
        (
            SkillUsedEvent(
                message=f"{actor.name} uses Test Strike on {target.name}.",
                actor_id=actor.actor_id,
                skill_id="test_strike",
                target_id=target.actor_id,
            ),
            StatusChangedEvent(
                message=f"{target.name}'s Guard absorbs the blow.",
                actor_id=target.actor_id,
                status="guarded",
                added=False,
            ),
            CombatEffectEvent(
                message=f"{target.name}'s Guard absorbs 3 damage.",
                actor_id=target.actor_id,
                target_id=target.actor_id,
                effect_type="mitigation",
                label="Guard -3",
                delta=-3,
                source_kind="tag",
                source_id="guarded",
                emphasis="good",
            ),
            DamageEvent(
                message=f"{actor.name} deals 5 damage to {target.name}.",
                source_id=actor.actor_id,
                target_id=target.actor_id,
                amount=5,
            ),
            DeathEvent(
                message=f"{target.name} dies: took 5 damage, reduced to 0 HP.",
                actor_id=target.actor_id,
            ),
        ),
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_frame=4,
    )
    plain = Text.from_markup(text).plain

    assert f"{target.name} dies: took 5 damage." in plain
    assert "Effects:" in plain
    assert "Guard -3" in plain
    assert "Guard absorbs the blow" not in plain


def test_combat_beat_renderer_staggers_attack_then_hurt_and_hp_drop() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    actor = next(combatant for combatant in view.party if combatant.class_id == "watchman")
    target = next(combatant for combatant in view.enemies if combatant.class_id == "bone_soldier")
    frameless_actor = replace(actor, art_frames={}, art_frame_impacts={})
    post_target = replace(target, hp=max(0, target.hp - 4))
    post_view = replace(
        view,
        party=tuple(
            frameless_actor if combatant.actor_id == actor.actor_id else combatant
            for combatant in view.party
        ),
        enemies=tuple(
            post_target if combatant.actor_id == target.actor_id else combatant
            for combatant in view.enemies
        ),
    )
    events = (
        SkillUsedEvent(
            message=f"{actor.name} uses Test Strike on {target.name}.",
            actor_id=actor.actor_id,
            skill_id="test_strike",
            target_id=target.actor_id,
        ),
        DamageEvent(
            message=f"{target.name} takes 4 damage.",
            source_id=actor.actor_id,
            target_id=target.actor_id,
            amount=4,
        ),
    )

    def line_index(text: str, needle: str) -> int:
        return next(index for index, line in enumerate(text.splitlines()) if needle in line)

    def line_offset(text: str, needle: str) -> int:
        line = next(line for line in text.splitlines() if needle in line)
        return line.index(needle)

    anticipation_text = CombatPanel.render_combat_beat(
        post_view,
        events,
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_frame=0,
    )
    attack_text = CombatPanel.render_combat_beat(
        post_view,
        events,
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_frame=1,
    )
    bridge_text = CombatPanel.render_combat_beat(
        post_view,
        events,
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_frame=2,
    )
    hurt_text = CombatPanel.render_combat_beat(
        post_view,
        events,
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_frame=3,
    )
    settle_text = CombatPanel.render_combat_beat(
        post_view,
        events,
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_frame=4,
    )
    idle_texts = {
        CombatPanel.render_combat_beat(
            post_view,
            events,
            title="Hero Action",
            source_actor_ids={actor.actor_id},
            target_intents={target.actor_id: "attack"},
            animation_frame=frame,
        )
        for frame in range(5, 17)
    }

    assert "[bright_cyan]" not in anticipation_text
    assert "-4" not in anticipation_text
    assert "takes 4 damage" not in anticipation_text
    actor_head = actor.art_lines[0].strip()
    assert line_offset(attack_text, actor_head) > line_offset(anticipation_text, actor_head)
    assert "[bold cyan]" not in attack_text
    hurt_marker = "__0  !"
    assert hurt_marker not in attack_text
    assert "-4" not in attack_text
    assert "takes 4 damage" not in attack_text
    assert f"HP {max(0, target.hp - 4)}/{target.max_hp}" not in attack_text
    assert hurt_marker not in bridge_text
    assert "-4" not in bridge_text
    assert "takes 4 damage" not in bridge_text
    assert hurt_marker in hurt_text
    assert "/|#|_" not in hurt_text
    assert "-4" not in hurt_text
    assert "takes 4 damage" in hurt_text
    assert "[bold red]" in hurt_text
    assert any("[bold red]" in line and hurt_marker in line for line in hurt_text.splitlines())
    parsed_hurt = Text.from_markup(hurt_text)
    hurt_hp_index = parsed_hurt.plain.find(f"HP {max(0, target.hp - 4)}/{target.max_hp}")
    assert hurt_hp_index >= 0
    assert not any(
        span.start <= hurt_hp_index < span.end and "red" in str(span.style)
        for span in parsed_hurt.spans
    )
    assert f"[bold red]{hurt_marker}" not in hurt_text
    assert f"HP {max(0, target.hp - 4)}/{target.max_hp}" in hurt_text
    assert line_offset(hurt_text, hurt_marker) > line_offset(bridge_text, "_0_")
    assert line_offset(hurt_text, hurt_marker) > line_offset(settle_text, "_0_")
    assert line_index(attack_text, "_0_") == line_index(bridge_text, "_0_")
    assert line_index(bridge_text, "_0_") == line_index(hurt_text, hurt_marker)
    assert line_index(hurt_text, hurt_marker) == line_index(settle_text, "_0_")
    assert len(idle_texts) > 1
    assert all("-4" not in text for text in idle_texts)
    assert all("takes 4 damage" in text for text in idle_texts)
    assert "Actor" not in hurt_text
    assert "Target" not in hurt_text


def test_combat_beat_overkill_uses_hp_before_instead_of_raw_damage() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    actor = next(combatant for combatant in view.party if combatant.class_id == "watchman")
    target = next(combatant for combatant in view.enemies if combatant.class_id == "bone_soldier")
    frameless_actor = replace(actor, art_frames={}, art_frame_impacts={})
    dead_target = replace(target, hp=0)
    post_view = replace(
        view,
        party=tuple(
            frameless_actor if combatant.actor_id == actor.actor_id else combatant
            for combatant in view.party
        ),
        enemies=tuple(
            dead_target if combatant.actor_id == target.actor_id else combatant
            for combatant in view.enemies
        ),
    )
    events = (
        SkillUsedEvent(
            message=f"{actor.name} uses Test Strike on {target.name}.",
            actor_id=actor.actor_id,
            skill_id="test_strike",
            target_id=target.actor_id,
        ),
        DamageEvent(
            message=f"{target.name} takes 5 damage.",
            source_id=actor.actor_id,
            target_id=target.actor_id,
            amount=5,
            hp_before=1,
        ),
    )

    before_impact = Text.from_markup(
        CombatPanel.render_combat_beat(
            post_view,
            events,
            title="Hero Action",
            source_actor_ids={actor.actor_id},
            target_intents={target.actor_id: "attack"},
            animation_frame=0,
        )
    ).plain
    impact = Text.from_markup(
        CombatPanel.render_combat_beat(
            post_view,
            events,
            title="Hero Action",
            source_actor_ids={actor.actor_id},
            target_intents={target.actor_id: "attack"},
            animation_frame=3,
        )
    ).plain

    assert f"HP 1/{target.max_hp}" in before_impact
    assert f"HP 5/{target.max_hp}" not in before_impact
    assert f"HP 0/{target.max_hp}" in impact


def test_combat_beat_enemy_attack_keeps_enemy_on_right() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    enemy = next(combatant for combatant in view.enemies if combatant.class_id == "bone_soldier")
    target = next(combatant for combatant in view.party if combatant.class_id == "watchman")
    text = CombatPanel.render_enemy_turn(
        view,
        (
            SkillUsedEvent(
                message=f"{enemy.name} uses Test Strike on {target.name}.",
                actor_id=enemy.actor_id,
                skill_id="test_strike",
                target_id=target.actor_id,
            ),
        ),
        source_actor_ids={enemy.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_frame=0,
    )
    name_line = next(
        line for line in text.splitlines() if enemy.name in line and target.name in line
    )

    assert "Target" not in text
    assert "Actor" not in text
    assert "VS" in text
    assert "->" not in text
    assert "<-" not in text
    assert name_line.find(target.name) < name_line.find(enemy.name)
    assert enemy.art_frames["attack"][0][0].strip() in text


def test_combat_beat_death_summary_and_status_badge_are_not_duplicated() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    actor = next(combatant for combatant in view.party if combatant.class_id == "watchman")
    target = next(combatant for combatant in view.enemies if combatant.class_id == "maze_leech")
    frameless_actor = replace(actor, art_frames={}, art_frame_impacts={})
    dead_target = replace(target, hp=0, statuses=("dead",))
    post_view = replace(
        view,
        party=tuple(
            frameless_actor if combatant.actor_id == actor.actor_id else combatant
            for combatant in view.party
        ),
        enemies=tuple(
            dead_target if combatant.actor_id == target.actor_id else combatant
            for combatant in view.enemies
        ),
    )
    events = (
        SkillUsedEvent(
            message=f"{actor.name} uses Test Strike on {target.name}.",
            actor_id=actor.actor_id,
            skill_id="test_strike",
            target_id=target.actor_id,
        ),
        DamageEvent(
            message=f"{actor.name} deals 5 damage to {target.name}.",
            source_id=actor.actor_id,
            target_id=target.actor_id,
            amount=5,
        ),
        DeathEvent(
            message=f"{target.name} dies: took 5 damage, reduced to 0 HP.",
            actor_id=target.actor_id,
        ),
        StatusChangedEvent(
            message=f"{target.name} is dead.",
            actor_id=target.actor_id,
            status="dead",
            added=True,
        ),
        CombatEndedEvent(message="The company wins the fight.", victor="heroes"),
        ExpeditionEvent(message="Wolf Hollow ends in victory.", node_id="wolf_hollow"),
    )
    anticipation_text = CombatPanel.render_combat_beat(
        post_view,
        events,
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_frame=0,
    )
    hurt_text = CombatPanel.render_combat_beat(
        post_view,
        events,
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_frame=3,
    )
    settle_text = CombatPanel.render_combat_beat(
        post_view,
        events,
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_frame=4,
    )

    assert "DEAD" not in anticipation_text
    assert "-5" not in anticipation_text
    assert "-5" not in hurt_text
    assert "DEAD" not in hurt_text
    assert "deals 5 damage" in hurt_text
    assert f"{target.name} dies." not in hurt_text
    assert "-5" not in settle_text
    assert "deals 5 damage" not in settle_text
    assert ".:::::." in settle_text
    assert f"{target.name} dies: took 5 damage." in settle_text
    assert "reduced to 0 HP" not in settle_text
    assert f"{target.name} is dead" not in settle_text
    assert "Victory." in settle_text
    assert "Wolf Hollow ends in victory" not in settle_text
    assert settle_text.count("DEAD") == 1
    assert "+DEAD" not in settle_text


def test_combat_beat_idle_frames_breathe_without_shifting_portrait_base() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    actor = next(combatant for combatant in view.party if combatant.class_id == "watchman")
    target = next(combatant for combatant in view.enemies if combatant.class_id == "bone_soldier")
    text = CombatPanel.render_combat_beat(
        view,
        (
            SkillUsedEvent(
                message=f"{actor.name} waits on {target.name}.",
                actor_id=actor.actor_id,
                skill_id="test_strike",
                target_id=target.actor_id,
            ),
        ),
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_cues={},
        animation_frame=0,
    )
    breathing_texts = [
        CombatPanel.render_combat_beat(
            view,
            (
                SkillUsedEvent(
                    message=f"{actor.name} waits on {target.name}.",
                    actor_id=actor.actor_id,
                    skill_id="test_strike",
                    target_id=target.actor_id,
                ),
            ),
            title="Hero Action",
            source_actor_ids={actor.actor_id},
            target_intents={target.actor_id: "attack"},
            animation_cues={},
            animation_frame=frame,
        )
        for frame in range(12)
    ]

    plain_texts = [Text.from_markup(frame_text).plain for frame_text in breathing_texts]

    assert "   [#]   " in Text.from_markup(text).plain
    assert any(frame[0].strip() in text for frame in target.art_frames["idle"])
    assert any("  [|#|>" in plain_text for plain_text in plain_texts)
    assert any("  [|#|\\" in plain_text for plain_text in plain_texts)
    assert len(set(breathing_texts)) > 1
    assert all("[I]" not in frame_text for frame_text in breathing_texts)


def test_combat_beat_idle_speed_slows_when_actor_is_hurt() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    actor = next(combatant for combatant in view.party if combatant.class_id == "watchman")
    target = next(combatant for combatant in view.enemies if combatant.class_id == "bone_soldier")
    critical_actor = replace(actor, hp=1)
    critical_target = replace(target, hp=1)
    critical_view = replace(
        view,
        party=tuple(
            critical_actor if combatant.actor_id == actor.actor_id else combatant
            for combatant in view.party
        ),
        enemies=tuple(
            critical_target if combatant.actor_id == target.actor_id else combatant
            for combatant in view.enemies
        ),
    )
    events = (
        SkillUsedEvent(
            message=f"{actor.name} waits on {target.name}.",
            actor_id=actor.actor_id,
            skill_id="test_strike",
            target_id=target.actor_id,
        ),
    )

    full_health_frames = {
        CombatPanel.render_combat_beat(
            view,
            events,
            title="Hero Action",
            source_actor_ids={actor.actor_id},
            target_intents={target.actor_id: "attack"},
            animation_cues={},
            animation_frame=frame,
        )
        for frame in range(5, 13)
    }
    critical_frames = {
        CombatPanel.render_combat_beat(
            critical_view,
            events,
            title="Hero Action",
            source_actor_ids={actor.actor_id},
            target_intents={target.actor_id: "attack"},
            animation_cues={},
            animation_frame=frame,
        )
        for frame in range(5, 13)
    }

    assert len(full_health_frames) > len(critical_frames)


def test_combat_beat_renderer_uses_authored_cast_frames() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    actor = next(combatant for combatant in view.party if combatant.class_id == "field_surgeon")
    target = next(combatant for combatant in view.party if combatant.class_id == "watchman")
    caster = replace(
        actor,
        art_lines=("READY",),
        art_frames={"cast": (("CHANT",), ("GLOW",), ("RELEASE",))},
        art_frame_holds={"cast": (1, 1, 1)},
        art_frame_impacts={"cast": 2},
    )
    cast_view = replace(
        view,
        party=tuple(
            caster if combatant.actor_id == actor.actor_id else combatant
            for combatant in view.party
        ),
    )
    text = CombatPanel.render_combat_beat(
        cast_view,
        (
            SkillUsedEvent(
                message=f"{actor.name} uses Emergency Dressing on {target.name}.",
                actor_id=actor.actor_id,
                skill_id="emergency_stitch",
                target_id=target.actor_id,
            ),
            HealingEvent(
                message=f"{target.name} recovers 2 HP.",
                source_id=actor.actor_id,
                target_id=target.actor_id,
                amount=2,
            ),
        ),
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "heal"},
        animation_frame=0,
    )
    pre_heal_text = CombatPanel.render_combat_beat(
        cast_view,
        (
            SkillUsedEvent(
                message=f"{actor.name} uses Emergency Dressing on {target.name}.",
                actor_id=actor.actor_id,
                skill_id="emergency_stitch",
                target_id=target.actor_id,
            ),
            HealingEvent(
                message=f"{target.name} recovers 2 HP.",
                source_id=actor.actor_id,
                target_id=target.actor_id,
                amount=2,
            ),
        ),
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "heal"},
        animation_frame=1,
    )
    heal_text = CombatPanel.render_combat_beat(
        cast_view,
        (
            SkillUsedEvent(
                message=f"{actor.name} uses Emergency Dressing on {target.name}.",
                actor_id=actor.actor_id,
                skill_id="emergency_stitch",
                target_id=target.actor_id,
            ),
            HealingEvent(
                message=f"{target.name} recovers 2 HP.",
                source_id=actor.actor_id,
                target_id=target.actor_id,
                amount=2,
            ),
        ),
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "heal"},
        animation_frame=2,
    )

    assert "CHANT" in text
    assert "recovers 2 HP" not in text
    assert "GLOW" in pre_heal_text
    assert "+2" not in pre_heal_text
    assert "RELEASE" in heal_text
    assert "+2" not in heal_text
    assert "recovers 2 HP" in heal_text
    assert "[bold green]" in heal_text
    assert any("[bold green]" in line and "\\[#]" in line for line in heal_text.splitlines())
    parsed_heal = Text.from_markup(heal_text)
    heal_hp_index = parsed_heal.plain.find(f"HP {target.hp}/{target.max_hp}")
    assert heal_hp_index >= 0
    assert not any(
        span.start <= heal_hp_index < span.end and "green" in str(span.style)
        for span in parsed_heal.spans
    )


def test_combat_beat_damage_flash_keeps_surgeon_and_scribe_portraits_clean() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    enemy = next(combatant for combatant in view.enemies if combatant.class_id == "bone_soldier")
    for class_id in ("field_surgeon", "scribe"):
        target = next(combatant for combatant in view.party if combatant.class_id == class_id)
        damaged_target = replace(target, hp=max(0, target.hp - 3))
        damaged_view = replace(
            view,
            party=tuple(
                damaged_target if combatant.actor_id == target.actor_id else combatant
                for combatant in view.party
            ),
        )
        text = CombatPanel.render_enemy_turn(
            damaged_view,
            (
                SkillUsedEvent(
                    message=f"{enemy.name} uses Test Strike on {target.name}.",
                    actor_id=enemy.actor_id,
                    skill_id="test_strike",
                    target_id=target.actor_id,
                ),
                DamageEvent(
                    message=f"{target.name} takes 3 damage.",
                    source_id=enemy.actor_id,
                    target_id=target.actor_id,
                    amount=3,
                ),
            ),
            source_actor_ids={enemy.actor_id},
            target_intents={target.actor_id: "attack"},
            animation_frame=1,
        )

        plain_text = Text.from_markup(text).plain
        portrait_block = plain_text.split("\n\n")[1]

        assert "-3" not in portrait_block
        assert "\\\\" not in portrait_block
        assert f"HP {damaged_target.hp}/{target.max_hp}" in portrait_block
        assert f"{target.name} takes 3 damage." in plain_text


def test_combat_beat_renderer_syncs_four_frame_attack_impact() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    actor = next(combatant for combatant in view.party if combatant.class_id == "watchman")
    target = next(combatant for combatant in view.enemies if combatant.class_id == "bone_soldier")
    animated_actor = replace(
        actor,
        art_lines=("BASE",),
        art_frames={
            "attack": (
                ("WINDUP",),
                ("SWING",),
                ("IMPACT",),
                ("RECOVER",),
            )
        },
        art_frame_holds={"attack": (1, 1, 1, 1)},
        art_frame_impacts={"attack": 2},
    )
    hurt_target = replace(
        target,
        hp=max(0, target.hp - 4),
        art_lines=("TARGET",),
        art_frames={"hurt": (("HURT",),)},
        art_frame_impacts={},
    )
    animated_view = replace(
        view,
        party=tuple(
            animated_actor if combatant.actor_id == actor.actor_id else combatant
            for combatant in view.party
        ),
        enemies=tuple(
            hurt_target if combatant.actor_id == target.actor_id else combatant
            for combatant in view.enemies
        ),
    )
    events = (
        SkillUsedEvent(
            message=f"{actor.name} uses Test Strike on {target.name}.",
            actor_id=actor.actor_id,
            skill_id="test_strike",
            target_id=target.actor_id,
        ),
        DamageEvent(
            message=f"{target.name} takes 4 damage.",
            source_id=actor.actor_id,
            target_id=target.actor_id,
            amount=4,
        ),
    )

    frames = [
        CombatPanel.render_combat_beat(
            animated_view,
            events,
            title="Hero Action",
            source_actor_ids={actor.actor_id},
            target_intents={target.actor_id: "attack"},
            animation_frame=frame,
        )
        for frame in range(5)
    ]

    assert (
        CombatPanel.beat_animation_last_frame(
            animated_view,
            events,
            source_actor_ids={actor.actor_id},
            target_intents={target.actor_id: "attack"},
        )
        == 4
    )
    long_actor = replace(
        animated_actor,
        art_frames={
            "attack": (
                ("WINDUP",),
                ("SWING",),
                ("IMPACT",),
                ("RECOVER",),
                ("RESET",),
            )
        },
        art_frame_holds={"attack": (1, 1, 1, 1, 1)},
    )
    long_view = replace(
        animated_view,
        party=tuple(
            long_actor if combatant.actor_id == actor.actor_id else combatant
            for combatant in animated_view.party
        ),
    )
    assert (
        CombatPanel.beat_animation_last_frame(
            long_view,
            events,
            source_actor_ids={actor.actor_id},
            target_intents={target.actor_id: "attack"},
        )
        == 5
    )
    assert "WINDUP" in frames[0]
    assert "SWING" in frames[1]
    assert "IMPACT" in frames[2]
    assert "RECOVER" in frames[3]
    assert "BASE" in frames[4]
    assert "-4" not in frames[0]
    assert "-4" not in frames[1]
    assert "takes 4 damage" not in frames[1]
    assert f"HP {target.hp}/{target.max_hp}" in frames[1]
    assert "-4" not in frames[2]
    assert "HURT" in frames[2]
    assert "takes 4 damage" in frames[2]
    assert f"HP {max(0, target.hp - 4)}/{target.max_hp}" in frames[2]
    assert "HURT" not in frames[3]
    assert "HURT" not in frames[4]


def test_combat_beat_renderer_honors_authored_frame_holds() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    actor = next(combatant for combatant in view.party if combatant.class_id == "watchman")
    target = next(combatant for combatant in view.enemies if combatant.class_id == "bone_soldier")
    animated_actor = replace(
        actor,
        art_lines=("BASE",),
        art_frames={
            "attack": (
                ("WINDUP",),
                ("SWING",),
                ("IMPACT",),
                ("RECOVER",),
            )
        },
        art_frame_holds={"attack": (4, 2, 2, 2)},
        art_frame_impacts={"attack": 2},
    )
    hurt_target = replace(
        target,
        hp=max(0, target.hp - 4),
        art_lines=("TARGET",),
        art_frames={"hurt": (("HURT",),)},
        art_frame_impacts={},
    )
    held_view = replace(
        view,
        party=tuple(
            animated_actor if combatant.actor_id == actor.actor_id else combatant
            for combatant in view.party
        ),
        enemies=tuple(
            hurt_target if combatant.actor_id == target.actor_id else combatant
            for combatant in view.enemies
        ),
    )
    events = (
        SkillUsedEvent(
            message=f"{actor.name} uses Test Strike on {target.name}.",
            actor_id=actor.actor_id,
            skill_id="test_strike",
            target_id=target.actor_id,
        ),
        DamageEvent(
            message=f"{target.name} takes 4 damage.",
            source_id=actor.actor_id,
            target_id=target.actor_id,
            amount=4,
        ),
    )
    frames = [
        CombatPanel.render_combat_beat(
            held_view,
            events,
            title="Hero Action",
            source_actor_ids={actor.actor_id},
            target_intents={target.actor_id: "attack"},
            animation_frame=frame,
        )
        for frame in range(9)
    ]

    assert (
        CombatPanel.beat_animation_last_frame(
            held_view,
            events,
            source_actor_ids={actor.actor_id},
            target_intents={target.actor_id: "attack"},
        )
        == 5
    )
    assert "WINDUP" in frames[0]
    assert "WINDUP" in frames[1]
    assert "SWING" in frames[2]
    assert "IMPACT" in frames[3]
    assert "RECOVER" in frames[4]
    assert "BASE" in frames[5]
    assert "HURT" in frames[3]
    assert "takes 4 damage" in frames[3]
    assert f"HP {max(0, target.hp - 4)}/{target.max_hp}" in frames[3]
    assert "HURT" not in frames[4]


def test_combat_beat_renderer_procedurally_animates_when_frames_are_missing() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    actor = next(combatant for combatant in view.party if combatant.class_id == "watchman")
    target = next(combatant for combatant in view.enemies if combatant.class_id == "bone_soldier")
    frameless_view = replace(
        view,
        party=tuple(
            replace(combatant, art_frames={}) if combatant.actor_id == actor.actor_id else combatant
            for combatant in view.party
        ),
        enemies=tuple(
            replace(combatant, art_frames={})
            if combatant.actor_id == target.actor_id
            else combatant
            for combatant in view.enemies
        ),
    )
    events = [
        SkillUsedEvent(
            message=f"{actor.name} uses Test Strike on {target.name}.",
            actor_id=actor.actor_id,
            skill_id="test_strike",
            target_id=target.actor_id,
        ),
        DamageEvent(
            message=f"{target.name} takes 4 damage.",
            source_id=actor.actor_id,
            target_id=target.actor_id,
            amount=4,
        ),
    ]

    base_text = CombatPanel.render_combat_beat(
        frameless_view,
        events,
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_frame=0,
    )
    animated_text = CombatPanel.render_combat_beat(
        frameless_view,
        events,
        title="Hero Action",
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_frame=1,
    )

    assert animated_text != base_text
    assert "Actor" not in animated_text
    assert "Target" not in animated_text


def test_combat_beat_pulse_style_does_not_color_entire_padded_cell() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("cave_mini_boss"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    actor = next(combatant for combatant in view.enemies if combatant.class_id == "cave_maw_brute")
    target = next(combatant for combatant in view.party if combatant.class_id == "watchman")
    text = CombatPanel.render_enemy_turn(
        view,
        (
            SkillUsedEvent(
                message=f"{actor.name} uses Maw Slam on {target.name}.",
                actor_id=actor.actor_id,
                skill_id="maw_slam",
                target_id=target.actor_id,
            ),
        ),
        source_actor_ids={actor.actor_id},
        target_intents={target.actor_id: "debuff"},
        animation_frame=1,
    )

    assert "[bold cyan]" in text
    assert "VS" in text
    assert "[bold cyan]    * |#M#| *[/]" in text
    assert "[bold cyan]    * |#M#| *            " not in text
    assert "    * |#M#| *" in Text.from_markup(text).plain
    assert "@#@" not in text


def test_combat_event_beats_group_enemy_actions() -> None:
    app = CharterApp()
    events = [
        EnemyIntentEvent(
            message="Bone Skulker rears back. Scratch.",
            enemy_id="bone_skulker_1",
            enemy_name="Bone Skulker",
            skill_id="scratch",
            skill_name="Scratch",
            label="Scratch",
            target_id="hero_watchman",
            target_name="Mara Vell",
            threat_level="normal",
            obvious_effect="Incoming strike",
        ),
        SkillUsedEvent(
            message="Bone Skulker uses Scratch on Mara Vell.",
            actor_id="bone_skulker_1",
            skill_id="scratch",
            target_id="hero_watchman",
        ),
        MissEvent(
            message="Bone Skulker misses Mara Vell.",
            actor_id="bone_skulker_1",
            target_id="hero_watchman",
        ),
        SkillUsedEvent(
            message="Cave Maw uses Maw Slam on Ilyra Penn.",
            actor_id="cave_maw_brute_1",
            skill_id="maw_slam",
            target_id="hero_field_surgeon",
        ),
        DamageEvent(
            message="Ilyra Penn takes 4 damage.",
            source_id="cave_maw_brute_1",
            target_id="hero_field_surgeon",
            amount=4,
        ),
        RoundEndedEvent(
            message="Round 1 ends.",
            encounter_id="cave_mini_boss",
            round_number=1,
        ),
    ]

    beats = app._combat_event_beats(events)

    assert [event.message for event in beats[0]] == [
        "Bone Skulker rears back. Scratch.",
        "Bone Skulker uses Scratch on Mara Vell.",
        "Bone Skulker misses Mara Vell.",
    ]
    assert [event.message for event in beats[1]] == [
        "Cave Maw uses Maw Slam on Ilyra Penn.",
        "Ilyra Penn takes 4 damage.",
        "Round 1 ends.",
    ]


def test_recruited_hero_action_stays_in_hero_resolution_beat() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    view_result = controller.handle(ViewCombat())
    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    base_actor = view.party[0]
    recruited_actor = replace(
        base_actor,
        actor_id="recruit_alden",
        name="Alden Reed",
        team="hero",
    )
    target = view.enemies[0]
    app = CharterApp(controller=controller)
    app.current_combat_view = replace(
        view,
        party=(
            recruited_actor,
            *(actor for actor in view.party if actor.actor_id != base_actor.actor_id),
        ),
    )
    events = [
        SkillUsedEvent(
            message=f"{recruited_actor.name} uses Test Strike on {target.name}.",
            actor_id=recruited_actor.actor_id,
            skill_id="test_strike",
            target_id=target.actor_id,
        ),
        DamageEvent(
            message=f"{target.name} takes 4 damage.",
            source_id=recruited_actor.actor_id,
            target_id=target.actor_id,
            amount=4,
        ),
    ]

    hero_events, enemy_events = app._split_turn_events(events)

    assert hero_events == events
    assert enemy_events == []


def test_combat_beat_defers_future_downed_status_until_that_enemy_beat() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("cave_mini_boss"))

    view_result = controller.handle(ViewCombat())

    assert isinstance(view_result.value, CombatView)
    view = view_result.value
    first_enemy, second_enemy = view.enemies[:2]
    target = next(combatant for combatant in view.party if combatant.class_id == "watchman")
    downed_target = replace(target, hp=0, statuses=("downed",))
    post_view = replace(
        view,
        party=tuple(
            downed_target if combatant.actor_id == target.actor_id else combatant
            for combatant in view.party
        ),
    )
    first_events = (
        SkillUsedEvent(
            message=f"{first_enemy.name} uses Scratch on {target.name}.",
            actor_id=first_enemy.actor_id,
            skill_id="scratch",
            target_id=target.actor_id,
        ),
        MissEvent(
            message=f"{first_enemy.name} misses {target.name}.",
            actor_id=first_enemy.actor_id,
            target_id=target.actor_id,
        ),
    )
    second_events = (
        SkillUsedEvent(
            message=f"{second_enemy.name} uses Maw Slam on {target.name}.",
            actor_id=second_enemy.actor_id,
            skill_id="maw_slam",
            target_id=target.actor_id,
        ),
        DamageEvent(
            message=f"{target.name} takes {target.hp} damage.",
            source_id=second_enemy.actor_id,
            target_id=target.actor_id,
            amount=target.hp,
        ),
        DownedEvent(message=f"{target.name} is Downed.", actor_id=target.actor_id),
        StatusChangedEvent(
            message=f"{target.name} cannot act or protect their lane.",
            actor_id=target.actor_id,
            status="downed",
            added=True,
        ),
    )

    first_text = CombatPanel.render_enemy_turn(
        post_view,
        first_events,
        source_actor_ids={first_enemy.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_frame=1,
        deferred_events=second_events,
    )
    second_text = CombatPanel.render_enemy_turn(
        post_view,
        second_events,
        source_actor_ids={second_enemy.actor_id},
        target_intents={target.actor_id: "attack"},
        animation_frame=3,
    )

    assert "MISS" in first_text
    assert "DOWNED" not in first_text
    assert f"HP {target.hp}/{target.max_hp}" in first_text
    assert "DOWNED" in second_text
    assert f"HP 0/{target.max_hp}" in second_text


@pytest.mark.anyio
async def test_combat_idle_tick_updates_live_body_and_detail_without_changing_state() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    app = CharterApp(controller=controller)

    async with app.run_test():
        app._show_combat_command()
        before_body = app._live_body_text()
        before_detail = app._detail_text()
        actions = app.actions

        assert "TURN" not in before_detail
        assert "[bold black on cyan]" not in before_detail
        after_detail = before_detail
        for _ in range(4):
            app._tick_idle_animation()
            after_detail = app._detail_text()
            if after_detail != before_detail:
                break

        assert app.screen_state == "combat"
        assert app.actions == actions
        assert app._live_body_text() != before_body
        assert after_detail != before_detail


@pytest.mark.anyio
async def test_combat_turn_handoff_flashes_live_glyph_once() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    app = CharterApp(controller=controller)

    async with app.run_test():
        app._show_combat_command()
        assert app.current_combat_view is not None
        view = app.current_combat_view
        current_actor = view.current_actor
        assert current_actor is not None
        next_actor = next(actor for actor in view.party if actor.actor_id != current_actor.actor_id)
        handoff_view = replace(
            view,
            current_actor=replace(next_actor, acting=True),
            party=tuple(
                replace(actor, acting=actor.actor_id == next_actor.actor_id) for actor in view.party
            ),
        )

        app._show_combat_view(handoff_view, phase="command")

        assert app.turn_flash_actor_id == next_actor.actor_id
        assert "[bold black on bright_cyan]" in app._live_body_text()
        assert "TURN" not in app._live_body_text()

        for _ in range(4):
            app._tick_turn_flash_animation()

        assert app.turn_flash_actor_id == ""
        assert "TURN" not in app._live_body_text()


@pytest.mark.anyio
async def test_beat_tick_updates_live_body_without_advancing_resolution() -> None:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    controller.handle(StartManualCombat("shallow_cave"))
    app = CharterApp(controller=controller)

    async with app.run_test():
        app._show_combat_command()
        assert app.current_combat_view is not None
        actor = next(
            combatant
            for combatant in app.current_combat_view.party
            if combatant.class_id == "watchman"
        )
        target = next(
            combatant
            for combatant in app.current_combat_view.enemies
            if combatant.class_id == "bone_soldier"
        )
        app._show_resolution(
            [
                SkillUsedEvent(
                    message=f"{actor.name} uses Test Strike on {target.name}.",
                    actor_id=actor.actor_id,
                    skill_id="test_strike",
                    target_id=target.actor_id,
                ),
                DamageEvent(
                    message=f"{target.name} takes 4 damage.",
                    source_id=actor.actor_id,
                    target_id=target.actor_id,
                    amount=4,
                ),
            ]
        )
        before = app._live_body_text()
        actions = app.actions
        assert app.beat_animation_frame == -1

        app._tick_beat_animation()

        assert app.screen_state == "resolution"
        assert app.actions == actions
        assert app.beat_animation_frame == 0
        assert app._live_body_text() != before

        for _ in range(4):
            app._tick_beat_animation()

        assert app.beat_animation_frame == 4
        assert app.screen_state == "resolution"
        assert app.actions == actions

        app._tick_beat_animation()

        settled = app._live_body_text()
        assert app.beat_animation_frame == 5
        app._tick_beat_animation()

        assert app.beat_animation_frame == 6
        assert app.screen_state == "resolution"
        assert app.actions == actions

        for _ in range(2):
            app._tick_beat_animation()

        assert app.beat_animation_frame == 8
        assert app._live_body_text() != settled


def test_dungeon_room_panel_keeps_action_hints_out_of_body() -> None:
    controller = _started_interactive_dungeon_controller()
    _move_dungeon(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
        "shallow_cave_room_1",
        "cave_fork",
        "fungus_chamber",
        "old_works_cache",
    )

    text = _dungeon_room_text(controller)

    assert "Old Works Cache" in text
    assert "Recover Gate Key" not in text

    controller = _started_interactive_dungeon_controller()
    _move_dungeon(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
        "shallow_cave_room_1",
        "cave_fork",
        "narrow_crawl",
        "stone_gate",
    )
    assert controller.company is not None
    controller.company.supplies["rope"] = 0

    text = _dungeon_room_text(controller)

    assert "Black Stone Gate" in text
    assert "Unlock Black Gate" not in text
    assert "Force Black Gate" not in text


def test_dungeon_room_panel_renders_room_art_without_hiding_text() -> None:
    controller = _started_interactive_dungeon_controller()

    text = _dungeon_room_text(controller)
    lines = text.splitlines()

    assert any("_____" in line for line in lines[:8])
    assert "Haven East Gate" in text
    assert "Haven's lamps are still close enough to retreat by." in text
    assert "Dust trails east under the trees" in text
    assert "road | cleared | safe return" not in text


def test_dungeon_minimap_draws_elbow_between_distant_linked_nodes() -> None:
    origin = DungeonMapNodeView(
        node_id="room_a",
        name="Alpha",
        node_type="maze",
        status="visited",
        current=True,
        known=True,
        visited=True,
        cleared=True,
        safe_return=False,
        map_id="test_map",
        exit_node_ids=("room_b",),
        map_x=0,
        map_y=0,
    )
    distant = DungeonMapNodeView(
        node_id="room_b",
        name="Beta",
        node_type="maze",
        status="known",
        current=False,
        known=True,
        visited=False,
        cleared=False,
        safe_return=False,
        map_id="test_map",
        exit_node_ids=("room_a",),
        map_x=2,
        map_y=0,
    )
    view = DungeonView(
        expedition_id="exp_test",
        dungeon_id="test_map",
        current_map_id="test_map",
        current_room=DungeonRoomView(
            node_id="room_a",
            name="Alpha",
            node_type="maze",
            text="Test room.",
            safe_return=False,
            cleared=True,
        ),
        map_nodes=(origin, distant),
        exits=(distant,),
        room_actions=(),
        actions=(),
    )

    map_text = DungeonMapPanel.render_minimap_text(view)
    map_body = "\n".join(
        line
        for line in map_text.splitlines()[3:]
        if line and not line.startswith("Legend:")
    )

    assert "-" in map_body
    assert "@" in map_body
    assert "?" in map_body


def test_dungeon_minimap_is_bounded_and_centered_on_player() -> None:
    controller = _started_interactive_dungeon_controller()
    _move_dungeon(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
        "shallow_cave_room_1",
        "cave_fork",
    )
    result = controller.handle(ViewDungeon())
    assert result.success
    assert isinstance(result.value, DungeonView)

    text = DungeonMapPanel.render_minimap_text(result.value)
    map_lines = text.splitlines()[3:]
    legend = "Legend: @ you   o known   ? unknown   ! quest"
    while map_lines and map_lines[-1] != legend:
        map_lines.pop()
    if map_lines:
        map_lines.pop()
    while map_lines and map_lines[-1] == "":
        map_lines.pop()

    assert "Mini Map" in text
    assert "\nFork\n" in text
    assert "@" in text
    assert text.count(legend) == 1
    assert len(map_lines) <= 9
    assert all(len(line) <= 31 for line in map_lines)

    styled = DungeonMapPanel.render_minimap(
        result.value,
        highlighted_node_id=result.value.exits[0].node_id,
    )
    styles = [str(span.style) for span in styled.spans]
    assert "bold cyan" in styles
    assert "bold yellow" in styles
    assert "black on yellow" not in styles


def test_dungeon_minimap_draws_longer_trails_on_old_road() -> None:
    controller = _started_interactive_dungeon_controller()
    _move_dungeon(controller, "old_road", "hunters_trail")
    result = controller.handle(ViewDungeon())
    assert result.success
    assert isinstance(result.value, DungeonView)

    map_text = DungeonMapPanel.render_minimap_text(result.value)
    map_body = "\n".join(
        line
        for line in map_text.splitlines()[3:]
        if line and not line.startswith("Legend:")
    )

    assert "@" in map_body
    assert "|" in map_body
    assert "-" in map_body

    controller = _started_interactive_dungeon_controller()
    _move_dungeon(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
    )
    result = controller.handle(ViewDungeon())
    assert result.success
    map_text = DungeonMapPanel.render_minimap_text(result.value)
    map_body = "\n".join(
        line
        for line in map_text.splitlines()[3:]
        if line and not line.startswith("Legend:")
    )
    assert "@" in map_body
    assert "|" in map_body


def _started_interactive_dungeon_controller() -> AppController:
    controller = AppController(definitions=get_definitions())
    controller.handle(StartNewCompany())
    result = controller.handle(StartExpedition(manual_combat=True, interactive_dungeon=True))
    assert result.success
    return controller


def _move_dungeon(controller: AppController, *node_ids: str) -> None:
    for node_id in node_ids:
        result = controller.handle(MoveDungeon(node_id))
        assert result.success, result.error


def _use_dungeon_action(controller: AppController, action_id: str) -> None:
    result = controller.handle(UseDungeonAction(action_id))
    assert result.success, result.error


def _reach_fungus_chamber(controller: AppController) -> None:
    _move_dungeon(controller, *OPENING_DUNGEON_TO_WORKS_CACHE)


def _controller_at_old_works_cache_combat() -> AppController:
    controller = _started_interactive_dungeon_controller()
    _reach_fungus_chamber(controller)
    result = controller.handle(MoveDungeon("old_works_cache"))
    assert result.success, result.error
    assert controller.manual_combat is not None
    return controller


def _controller_at_stone_gate() -> AppController:
    controller = _started_interactive_dungeon_controller()
    _reach_fungus_chamber(controller)
    _move_dungeon(controller, "stone_gate")
    return controller


def _win_active_manual_combat(controller: AppController):
    final_result = None
    while controller.manual_combat is not None:
        session = controller.manual_combat
        if session.pending_enemy_intent is not None:
            final_result = controller.handle(ResolveCombatReaction(None))
            assert final_result.success, final_result.error
            continue
        skill_ids = legal_skill_ids(session, controller.definitions)
        if not skill_ids:
            final_result = controller.handle(PassCombatTurn())
            assert final_result.success, final_result.error
            continue
        skill_id = skill_ids[0]
        target_id = legal_target_ids(session, controller.definitions, skill_id)[0]
        final_result = controller.handle(ResolveCombatAction(skill_id, target_id))
        assert final_result.success, final_result.error
    assert final_result is not None
    return final_result


def _dungeon_room_text(controller: AppController) -> str:
    result = controller.handle(ViewDungeon())
    assert result.success
    assert isinstance(result.value, DungeonView)
    return DungeonRoomPanel.render_text(result.value)


def test_world_view_resolves_location_id_and_old_save_fallback() -> None:
    controller = AppController(definitions=get_definitions())
    start_result = controller.handle(StartNewCompany())
    assert start_result.success
    assert controller.company is not None
    assert controller.company.town_state["location_id"] == "haven"

    result = controller.handle(ViewWorld())

    assert result.success
    assert isinstance(result.value, WorldView)
    assert result.value.current_location_id == "haven"

    raw = controller.company.to_dict()
    raw["town_state"] = {"location": "Haven Town"}
    loaded = CompanyState.from_dict(raw)
    assert loaded.town_state["location_id"] == "haven"

    controller.company = loaded
    result = controller.handle(ViewWorld())

    assert result.success
    assert isinstance(result.value, WorldView)
    assert result.value.current_location_id == "haven"


async def start_default_company(app: CharterApp, pilot: object) -> None:
    await pilot.press("1")
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    assert app.controller.company is not None
    assert app.screen_state == "town"


async def start_opening_wilderness(app: CharterApp, pilot: object) -> None:
    await pilot.press("1")
    await pilot.pause()
    assert app.screen_state == "regional_place"
    old_road_action = next(action for action in app.actions if action.value == "old_road")
    await pilot.press(old_road_action.number)
    await pilot.pause()
    assert app.screen_state == "playback"


async def advance_playback_to(
    app: CharterApp,
    pilot: object,
    expected_screen: str,
) -> None:
    while app.screen_state == "playback":
        continue_action = next(
            action for action in app.actions if action.value == "continue"
        )
        await pilot.press(continue_action.number)
        await pilot.pause()
    while expected_screen == "combat" and app.screen_state == "enemy_turn":
        await pilot.press("enter")
        await pilot.pause()
    assert app.screen_state == expected_screen


async def choose_dungeon_action(
    app: CharterApp,
    pilot: object,
    key: str,
    *,
    expected_screen: str = "dungeon",
    expected_playback: bool = True,
) -> None:
    await pilot.press(key)
    await pilot.pause()
    if not expected_playback:
        assert app.screen_state == expected_screen
        return
    if app.screen_state == expected_screen:
        return
    await advance_playback_to(app, pilot, expected_screen)


async def choose_dungeon_value(
    app: CharterApp,
    pilot: object,
    value: str,
    *,
    expected_screen: str = "dungeon",
    expected_playback: bool | None = None,
) -> None:
    action = next((action for action in app.actions if action.value == value), None)
    if action is None and value.startswith("action:"):
        interact = next(action for action in app.actions if action.value == "interact")
        await choose_dungeon_action(
            app,
            pilot,
            interact.number,
            expected_screen="dungeon_interact",
            expected_playback=False,
        )
        action = next(action for action in app.actions if action.value == value)
    if action is None:
        raise AssertionError(f"No dungeon action found for value: {value}")
    if expected_playback is None:
        expected_playback = not value.startswith("action:")
    await choose_dungeon_action(
        app,
        pilot,
        action.number,
        expected_screen=expected_screen,
        expected_playback=expected_playback,
    )


async def reach_old_works_cache_combat(app: CharterApp, pilot: object) -> None:
    for value in OPENING_DUNGEON_TO_WORKS_CACHE:
        await choose_dungeon_value(app, pilot, value)
    await choose_dungeon_value(app, pilot, "old_works_cache", expected_screen="combat")


def hci_transcript_entry(app: CharterApp) -> tuple[str, str, str]:
    action = app.focused_action
    return (
        app.screen_state,
        action.label if action is not None else "none",
        app._detail_text(),
    )


@pytest.mark.anyio
async def test_tui_mounts_and_supports_focus_enter_number_and_hotkey(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.screen_state == "main"
        assert app.focused_action is not None
        assert app.focused_action.value == "start"
        assert isinstance(app.query_one("#header"), StatusHeader)
        assert isinstance(app.query_one("#body"), BodyPane)
        assert isinstance(app.query_one("#detail"), DetailPane)
        assert isinstance(app.query_one("#log"), LogPane)
        assert isinstance(app.query_one("#dock"), CommandDock)

        await pilot.press("3")
        await pilot.pause()
        assert app.screen_state == "help"

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "main"

        await start_default_company(app, pilot)
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "system"
        await pilot.press("6")
        await pilot.pause()
        assert app.screen_state == "town"

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "system"


@pytest.mark.anyio
async def test_tui_start_company_default_and_custom_name(tmp_path: Path) -> None:
    default_app = CharterApp(save_path=tmp_path / "default.json")
    async with default_app.run_test() as pilot:
        await start_default_company(default_app, pilot)
        assert default_app.controller.company is not None
        assert default_app.controller.company.name == "Haven Charter"

    custom_app = CharterApp(save_path=tmp_path / "custom.json")
    async with custom_app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("1")
        await pilot.pause()
        custom_app.query_one("#name-input", Input).value = "Amber Gate"
        await pilot.press("enter")
        await pilot.pause()

        assert custom_app.controller.company is not None
        assert custom_app.controller.company.name == "Amber Gate"


@pytest.mark.anyio
async def test_tui_save_load_confirmation_flow(tmp_path: Path) -> None:
    save_path = tmp_path / "company.json"
    save_path.write_text("existing", encoding="utf-8")
    app = CharterApp(save_path=save_path)

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        await pilot.press("escape", "1")
        await pilot.pause()
        assert app.screen_state == "confirm"
        assert app.pending_confirm == "overwrite_save"

        await pilot.press("1")
        await pilot.pause()
        assert save_path.read_text(encoding="utf-8") == "existing"
        assert app.screen_state == "system"

        await pilot.press("1", "2")
        await pilot.pause()
        assert "Haven Charter" in save_path.read_text(encoding="utf-8")

    load_app = CharterApp(save_path=save_path)
    async with load_app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("2")
        await pilot.pause()

        assert load_app.controller.company is not None
        assert load_app.controller.company.name == "Haven Charter"
        assert load_app.screen_state == "town"


@pytest.mark.anyio
async def test_tui_town_disabled_actions_are_blocked(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        assert app.controller.company is not None
        app.controller.company.coin = 0

        assert app.screen_state == "town"
        assert "Town Actions" not in app.body_text
        assert tuple(action.value for action in app.actions) == (
            "east_gate",
            "town_charter",
            "town_yard",
            "town_market",
            "town_recovery",
            "system",
        )

        await pilot.press("2")
        await pilot.pause()
        assert app.screen_state == "town_charter"
        assert "Charter Office" in app.body_text
        upgrade_action = next(action for action in app.actions if action.value == "town_upgrades")
        await pilot.press(upgrade_action.number)
        await pilot.pause()
        assert app.screen_state == "town_upgrades"
        assert "Quartermaster Shelf" in app.body_text
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "town_charter"
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "town"

        await pilot.press("4")
        await pilot.pause()
        assert app.screen_state == "town_market"
        recruit_action = next(action for action in app.actions if action.value == "recruit")
        assert not recruit_action.enabled

        await pilot.press(recruit_action.number)
        await pilot.pause()
        assert app.screen_state == "town_market"
        assert "unavailable" in app.message


@pytest.mark.anyio
async def test_tui_hci_safe_focus_and_disabled_reason_contract(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        assert_focused_action_is_safe(app)

        assert app.controller.company is not None
        app.controller.company.coin = 0
        await pilot.press("4")
        await pilot.pause()
        assert app.screen_state == "town_market"
        assert_focused_action_is_safe(app)
        assert app.focused_action is not None
        assert app.focused_action.value == "back"

        recruit_index = next(
            index for index, action in enumerate(app.actions) if action.value == "recruit"
        )
        app.focused_command_index = recruit_index
        app.message = ""
        app._render()

        recruit = app.focused_action
        assert recruit is not None
        assert not recruit.enabled
        assert recruit.unavailable_reason
        assert "Need" in recruit.unavailable_reason
        detail = app._detail_text()
        assert "Risk: Costly" in detail
        assert "Cost:" in detail
        assert "Need" in detail

        await pilot.press(recruit.number)
        await pilot.pause()
        assert "Need" in app.message


@pytest.mark.anyio
async def test_tui_confirmations_default_to_safe_cancel(tmp_path: Path) -> None:
    save_path = tmp_path / "company.json"
    save_path.write_text("existing", encoding="utf-8")
    app = CharterApp(save_path=save_path)

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        await pilot.press("escape", "1")
        await pilot.pause()

        assert app.screen_state == "confirm"
        assert app.focused_action is not None
        assert app.focused_action.value == "cancel"
        assert_focused_action_is_safe(app)
        assert any(
            action.value == "confirm" and str(action.risk) == "irreversible"
            for action in app.actions
        )

        await pilot.press("enter")
        await pilot.pause()
        assert save_path.read_text(encoding="utf-8") == "existing"
        assert app.screen_state == "system"


@pytest.mark.anyio
async def test_tui_load_over_active_company_requires_confirmation(tmp_path: Path) -> None:
    save_path = tmp_path / "company.json"
    app = CharterApp(save_path=save_path)

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("1")
        await pilot.pause()
        assert save_path.exists()

        assert app.controller.company is not None
        app.controller.company.name = "Unsaved Name"
        await pilot.press("2")
        await pilot.pause()

        assert app.screen_state == "confirm"
        assert app.pending_confirm == "load_company"
        assert app.focused_action is not None
        assert app.focused_action.value == "cancel"
        assert "replace the current in-memory company" in app._detail_text()


@pytest.mark.anyio
async def test_tui_world_shell_uses_place_travel_map_and_system(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)

        assert app.screen_state == "town"
        assert "World > Haven" in app.body_text
        assert tuple(action.value for action in app.actions) == (
            "east_gate",
            "town_charter",
            "town_yard",
            "town_market",
            "town_recovery",
            "system",
        )

        await pilot.press("1")
        await pilot.pause()
        assert app.screen_state == "regional_place"

        survey_action = next(action for action in app.actions if action.value == "survey_route")
        assert survey_action.label == "Open Roadbook"
        await pilot.press(survey_action.number)
        await pilot.pause()
        assert app.screen_state == "regional_map"
        assert "Company Roadbook" in app.body_text
        assert "Charted Route Survey" in app.body_text
        assert "ZOOMED OUT: choose a known destination" in app.body_text
        assert any(action.label == "Fold Roadbook" for action in app.actions)
        assert app.current_regional_view is not None
        assert app.current_regional_view.anchor_kind == "east_gate"

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "regional_place"

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "town"

        await pilot.press("6")
        await pilot.pause()
        assert app.screen_state == "system"
        assert tuple(action.label for action in app.actions) == (
            "Save Company",
            "Load Company",
            "Enemy AI: Learned Static",
            "Help",
            "Quit",
            "Back",
        )

        await pilot.press("6")
        await pilot.pause()
        assert app.screen_state == "town"

        await pilot.press("1")
        await pilot.pause()
        assert app.screen_state == "regional_place"
        assert any(action.value == "survey_route" for action in app.actions)
        assert any(action.label == "Open Roadbook" for action in app.actions)
        assert "East Gate" in app.body_text


@pytest.mark.anyio
async def test_tui_system_menu_toggles_enemy_ai_mode(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        await pilot.press("escape")
        await pilot.pause()

        assert app.screen_state == "system"
        assert app.controller.enemy_ai_mode == "learned_static"
        assert "Enemy AI" in app.body_text
        assert "Current: Learned Static" in app.body_text
        assert "Heuristic" in app.body_text

        await pilot.press("3")
        await pilot.pause()

        assert app.controller.enemy_ai_mode == "heuristic"
        assert "Current: Heuristic" in app.body_text
        assert any(action.label == "Enemy AI: Heuristic" for action in app.actions)


@pytest.mark.anyio
async def test_tui_system_menu_enemy_ai_mode_bundles_wait_and_move_timing(
    tmp_path: Path,
) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        await pilot.press("escape")
        await pilot.pause()

        assert app.screen_state == "system"
        assert app.controller.enemy_ai_mode == "learned_static"
        assert app.controller.enemy_wait_mode == "package_only"
        assert app.controller.enemy_movement_mode == "package_only"
        assert "Enemy AI" in app.body_text
        assert "Current: Learned Static" in app.body_text
        assert "Timing: wait Package Only, move Package Only" in app.body_text
        assert "Dev Enemy Timing" not in app.body_text

        await pilot.press("3")
        await pilot.pause()

        assert app.controller.enemy_ai_mode == "heuristic"
        assert app.controller.enemy_wait_mode == "none"
        assert app.controller.enemy_movement_mode == "recovery_only"
        assert "Current: Heuristic" in app.body_text
        assert "Timing: wait None, move Recovery Only" in app.body_text
        assert any(action.label == "Enemy AI: Heuristic" for action in app.actions)


@pytest.mark.anyio
async def test_tui_haven_service_rooms_use_focused_structured_bodies(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)

        assert app.screen_state == "town"
        assert "Company Status\n" in app.body_text
        assert "Current Objective" in app.body_text
        assert "Districts" in app.body_text

        await pilot.press("5")
        await pilot.pause()
        assert app.screen_state == "town_recovery"
        assert "Purpose\nRestore HP and Effort" in app.body_text
        assert "Status\n- Wounded:" in app.body_text
        assert "Treatment\n- Recovery cost:" in app.body_text
        assert "Active Party\n" not in app.body_text
        assert any(action.label == "Back to Haven" for action in app.actions)

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "town"

        await pilot.press("2")
        await pilot.pause()
        assert app.screen_state == "town_charter"
        assert "Purpose\nTurn dangerous places" in app.body_text
        assert "Contract Board" in app.body_text
        assert "Active Party\n" not in app.body_text

        records_action = next(action for action in app.actions if action.value == "town_records")
        await pilot.press(records_action.number)
        await pilot.pause()
        assert app.screen_state == "town_records"
        assert "Purpose\nReview company ledger" in app.body_text
        assert "Filed records:" in app.body_text
        assert any(action.label == "Back to Charter Office" for action in app.actions)

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "town_charter"


@pytest.mark.anyio
async def test_tui_contract_board_accepts_unlocked_scout(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        assert app.controller.company is not None
        app.controller.company.completed_contract_ids.add("blackwood_road_charter")
        app.controller.company.active_contract_ids.discard("blackwood_road_charter")
        app.controller.company.known_breaches.add("shallow_cave_breach")
        app._show_town()

        await pilot.press("2")
        await pilot.pause()
        assert app.screen_state == "town_charter"
        assert "Available\n  Shallow Cave Breach Scout" in app.body_text
        assert "Breach Stalker Hunt" not in app.body_text
        assert "Complete the breach scout first." not in app.body_text
        scout_action = next(
            action for action in app.actions if action.value == "accept:shallow_cave_breach_scout"
        )
        assert not any(action.value == "accept:shallow_cave_breach_hunt" for action in app.actions)
        assert scout_action.label == "Shallow Cave Breach Scout"
        assert "x Breach Stalker Hunt" not in CommandDock.render_text(
            app.actions,
            app.focused_command_index,
        )

        await pilot.press(scout_action.number)
        await pilot.pause()

        assert "shallow_cave_breach_scout" in app.controller.company.active_contract_ids
        assert app.screen_state == "town_charter"
        assert "Active\n  Shallow Cave Breach Scout" in app.body_text


@pytest.mark.anyio
async def test_tui_roster_opens_scoped_hero_sheet(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        assert app.controller.company is not None
        company = app.controller.company
        hero = company.roster[0]
        other_hero = company.roster[1]
        company.completed_contract_ids.add("blackwood_road_charter")
        company.active_contract_ids.discard("blackwood_road_charter")
        company.gear_inventory["reinforced_vest"] = 1
        hero.fresh_memories = [
            FreshMemoryState(
                family_id="killing_blow",
                display_name="Killing Blow",
                intensity=2,
                source_summary="Mara ended the fight.",
            )
        ]
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
        app._show_town()

        await pilot.press("3")
        await pilot.pause()
        assert app.screen_state == "town_yard"

        await pilot.press("2")
        await pilot.pause()
        assert app.screen_state == "roster"
        hero_action = next(
            action for action in app.actions if action.value == f"hero:{hero.hero_id}"
        )
        app.focused_command_index = app.actions.index(hero_action)
        assert hero.name in app._detail_text()
        assert "Vitals" in app._detail_text()
        assert "Memory:" in app._detail_text()

        await pilot.press(hero_action.number)
        await pilot.pause()
        assert app.screen_state == "hero_sheet"
        assert "Vitals" in app.body_text
        assert "At a Glance" in app.body_text
        assert tuple(action.value for action in app.actions) == ("memories", "gear", "back")

        memories_action = next(action for action in app.actions if action.value == "memories")
        app.focused_command_index = app.actions.index(memories_action)
        assert "Active Memory Pressure" in app._detail_text()
        await pilot.press(memories_action.number)
        await pilot.pause()
        assert app.screen_state == "hero_memories"
        assert "Quirks" in app.body_text
        assert "Active Memory Pressure" in app.body_text
        assert "Recent Records" in app.body_text
        assert "Killing Blow" in app.body_text
        assert "first marched" in app.body_text

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "hero_sheet"

        gear_action = next(action for action in app.actions if action.value == "gear")
        app.focused_command_index = app.actions.index(gear_action)
        assert "Available Kits" in app._detail_text()
        await pilot.press(gear_action.number)
        await pilot.pause()
        assert app.screen_state == "hero_gear"
        assert "Available Kits" in app.body_text
        assert any(
            action.value == f"gear:equip:{hero.hero_id}:reinforced_vest" for action in app.actions
        )
        assert not any(other_hero.hero_id in action.value for action in app.actions)
        equip_action = next(
            action for action in app.actions
            if action.value == f"gear:equip:{hero.hero_id}:reinforced_vest"
        )
        await pilot.press(equip_action.number)
        await pilot.pause()
        assert app.screen_state == "hero_gear"
        assert hero.equipped_gear_id == "reinforced_vest"

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "hero_sheet"
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "roster"


@pytest.mark.anyio
async def test_tui_golden_path_transcript_has_stage_cues(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")
    transcript: list[tuple[str, str, str]] = []

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        transcript.append(hci_transcript_entry(app))

        await start_opening_wilderness(app, pilot)
        transcript.append(hci_transcript_entry(app))

        await advance_playback_to(app, pilot, "dungeon")
        transcript.append(hci_transcript_entry(app))

        await reach_old_works_cache_combat(app, pilot)
        transcript.append(hci_transcript_entry(app))

    states = [state for state, _action, _detail in transcript]
    details = "\n".join(detail for _state, _action, detail in transcript)

    assert states == ["town", "playback", "dungeon", "combat"]
    assert "Stage Focus" in details
    assert "Risk:" in details
    assert "Preview" in details
    assert "Command Focus" in details


@pytest.mark.anyio
async def test_tui_known_route_opens_cave_entrance_then_dungeon(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        assert app.controller.company is not None
        app.controller.company.known_route_ids.add("shallow_cave")

        await pilot.press("1")
        await pilot.pause()
        assert app.screen_state == "regional_place"
        survey_action = next(action for action in app.actions if action.value == "survey_route")
        assert survey_action.label == "Open Roadbook"
        assert any(action.value == "old_road" for action in app.actions)
        await pilot.press(survey_action.number)
        await pilot.pause()
        assert app.screen_state == "regional_map"
        assert "Company Roadbook" in app.body_text
        assert "Charted Route Survey" in app.body_text
        assert "known roads - cleared ground - fast travel" in app.body_text
        assert "No new discoveries on charted travel" in app.body_text
        assert any(action.label == "Fold Roadbook" for action in app.actions)
        charted_hop = next(
            action for action in app.actions if action.value == "shallow_cave"
        )
        assert charted_hop.label == "Take Charted Road to Shallow Cave"
        app.focused_command_index = app.actions.index(charted_hop)
        detail = app._detail_text()
        assert "Destination: Shallow Cave." in detail
        assert "Route: charted road." in detail
        assert "Cost: 1 ration when available" in detail
        assert "Risk: Low" in detail
        assert "Skips cleared Old Road beats." in detail
        assert "No new discoveries on this route." in detail
        await pilot.press(charted_hop.number)
        await pilot.pause()
        assert app.screen_state == "regional_place"
        assert "bypassing cleared stretches of Old Road" in app.message
        assert app.controller.company.town_state["location_id"] == "shallow_cave"
        assert app.controller.company.town_state["regional_node_id"] == "shallow_cave_entrance"
        enter_cave = next(action for action in app.actions if action.value == "enter_cave")
        await choose_dungeon_action(
            app,
            pilot,
            enter_cave.number,
            expected_screen="dungeon",
            expected_playback=False,
        )
        assert "Cave Mouth" in app.body_text


@pytest.mark.anyio
async def test_tui_regional_walk_from_east_gate_after_route_charted(
    tmp_path: Path,
) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        assert app.controller.company is not None
        app.controller.company.known_route_ids.add("shallow_cave")
        memory = app.controller.company.dungeon_memory.setdefault(
            "shallow_cave",
            DungeonMemoryState(dungeon_id="shallow_cave"),
        )
        memory.visited_node_ids.extend(["town_gate", "old_road"])
        memory.cleared_node_ids.extend(["town_gate", "old_road"])

        await pilot.press("1")
        await pilot.pause()
        assert app.screen_state == "regional_place"
        walk_action = next(action for action in app.actions if action.value == "old_road")
        await pilot.press(walk_action.number)
        await pilot.pause()
        assert app.screen_state == "regional_place"
        assert app.controller.company.town_state["regional_node_id"] == "old_road"
        assert app.current_regional_view is not None
        assert app.current_regional_view.current_node_id == "old_road"
        walk_exits = {
            action.value
            for action in app.actions
            if action.value not in {"back", "survey_route", "interact"}
        }
        assert "town_gate" in walk_exits
        assert not any(action.value == "survey_route" for action in app.actions)


@pytest.mark.anyio
async def test_tui_regional_map_back_returns_to_latest_place(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        assert app.controller.company is not None
        app.controller.company.known_route_ids.add("shallow_cave")
        memory = app.controller.company.dungeon_memory.setdefault(
            "shallow_cave",
            DungeonMemoryState(dungeon_id="shallow_cave"),
        )
        memory.visited_node_ids.extend(["town_gate", "old_road"])
        memory.cleared_node_ids.extend(["town_gate", "old_road"])

        await pilot.press("1")
        await pilot.pause()
        walk_action = next(action for action in app.actions if action.value == "old_road")
        await pilot.press(walk_action.number)
        await pilot.pause()
        assert app.controller.company.town_state["regional_node_id"] == "old_road"

        gate_action = next(action for action in app.actions if action.value == "town_gate")
        await pilot.press(gate_action.number)
        await pilot.pause()
        assert app.controller.company.town_state["regional_node_id"] == "town_gate"

        survey_action = next(action for action in app.actions if action.value == "survey_route")
        await pilot.press(survey_action.number)
        await pilot.pause()
        assert app.screen_state == "regional_map"
        assert any(action.label == "Fold Roadbook" for action in app.actions)
        assert any(action.value == "shallow_cave" for action in app.actions)
        assert not any(
            action.value in {"old_road", "town_gate"}
            for action in app.actions
            if action.value != "back"
        )

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "regional_place"
        assert "roadbook is folded away" in app.message.lower()
        assert "East Gate" in app.body_text
        assert app.controller.company.town_state["regional_node_id"] == "town_gate"


@pytest.mark.anyio
async def test_tui_regional_bramble_shortcut_from_place(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        assert app.controller.company is not None
        company = app.controller.company
        company.known_route_ids.add("shallow_cave")
        memory = company.dungeon_memory.setdefault(
            "shallow_cave",
            DungeonMemoryState(dungeon_id="shallow_cave"),
        )
        memory.visited_node_ids.extend(["town_gate", "old_road", "bramble_shrine"])
        memory.cleared_node_ids.extend(["town_gate", "old_road", "bramble_shrine"])
        company.town_state["regional_node_id"] = "bramble_shrine"

        result = app.controller.handle(ViewRegionalMap())
        assert result.success
        app.current_regional_view = result.value
        app._show_regional_place(view=result.value)
        await pilot.pause()

        interact = next(action for action in app.actions if action.value == "interact")
        await pilot.press(interact.number)
        await pilot.pause()
        clear_action = next(
            action for action in app.actions if action.value == "action:clear_bramble_path"
        )
        await pilot.press(clear_action.number)
        await pilot.pause()
        assert app.screen_state == "regional_place"
        assert "hidden deer path" in app.message.lower()
        assert "bramble_shrine->hidden_deer_path" in memory.revealed_exit_ids


@pytest.mark.anyio
async def test_tui_overworld_shortcuts_toggle_and_company_manages_heroes(
    tmp_path: Path,
) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        assert app.controller.company is not None
        company = app.controller.company
        hero = company.roster[0]
        company.known_route_ids.add("shallow_cave")
        company.completed_contract_ids.add("blackwood_road_charter")
        company.active_contract_ids.discard("blackwood_road_charter")
        company.supplies["ration"] = 2
        company.gear_inventory["reinforced_vest"] = 1

        await pilot.press("1")
        await pilot.pause()
        assert app.screen_state == "regional_place"
        survey_action = next(action for action in app.actions if action.value == "survey_route")
        assert survey_action.label == "Open Roadbook"
        await pilot.press(survey_action.number)
        await pilot.pause()
        assert app.screen_state == "regional_map"
        charted_hop = next(
            action for action in app.actions if action.value == "shallow_cave"
        )
        await pilot.press(charted_hop.number)
        await pilot.pause()
        assert app.screen_state == "regional_place"
        assert app.controller.company.town_state["location_id"] == "shallow_cave"
        assert app.controller.company.town_state["regional_node_id"] == "shallow_cave_entrance"
        assert any(action.value == "survey_route" for action in app.actions)
        assert not any(action.value == "company_inventory" for action in app.actions)

        await pilot.press("p")
        await pilot.pause()
        assert app.screen_state == "pack"
        assert "Pack" in app.body_text
        assert "Supplies" in app.body_text
        assert "Ration" in app.body_text
        assert "x2" in app.body_text
        assert "Equipped" in app.body_text
        assert "Purpose" not in app.body_text
        assert not any(action.value.startswith("hero:") for action in app.actions)
        assert not any(action.value.startswith("gear:buy:") for action in app.actions)

        gear_action = next(action for action in app.actions if action.value == "gear")
        await pilot.press(gear_action.number)
        await pilot.pause()
        assert app.screen_state == "gear"
        assert "Gear purchases are only available in Haven." in app.body_text
        assert not any(action.value.startswith("gear:buy:") for action in app.actions)

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "pack"

        await pilot.press("p")
        await pilot.pause()
        assert app.screen_state == "regional_place"

        await pilot.press("m")
        await pilot.pause()
        assert app.screen_state == "regional_map"

        await pilot.press("m")
        await pilot.pause()
        assert app.screen_state == "regional_place"

        await pilot.press("?")
        await pilot.pause()
        assert app.screen_state == "help"

        await pilot.press("?")
        await pilot.pause()
        assert app.screen_state == "regional_place"

        await pilot.press("c")
        await pilot.pause()
        assert app.screen_state == "company_summary"
        assert "Company" in app.body_text
        assert company.name in app.body_text
        assert "Party Formation" in app.body_text
        assert "Characters" in app.body_text
        assert hero.name in app.body_text
        assert "Active Contracts" not in app.body_text
        assert "Purpose" not in app.body_text

        formation_action = next(action for action in app.actions if action.value == "formation")
        await pilot.press(formation_action.number)
        await pilot.pause()
        assert app.screen_state == "formation"

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "company_summary"

        hero_action = next(
            action for action in app.actions if action.value == f"hero:{hero.hero_id}"
        )
        await pilot.press(hero_action.number)
        await pilot.pause()
        assert app.screen_state == "hero_sheet"
        assert "Vitals" in app.body_text
        gear_action = next(action for action in app.actions if action.value == "gear")
        await pilot.press(gear_action.number)
        await pilot.pause()
        assert app.screen_state == "hero_gear"
        assert "Available Kits" in app.body_text
        assert any(
            action.value == f"gear:equip:{hero.hero_id}:reinforced_vest" for action in app.actions
        )

        equip_action = next(
            action
            for action in app.actions
            if action.value == f"gear:equip:{hero.hero_id}:reinforced_vest"
        )
        await pilot.press(equip_action.number)
        await pilot.pause()
        assert hero.equipped_gear_id == "reinforced_vest"

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "hero_sheet"
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "company_summary"

        await pilot.press("c")
        await pilot.pause()
        assert app.screen_state == "regional_place"


@pytest.mark.anyio
async def test_tui_service_screens_use_verb_first_docks(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        assert app.controller.company is not None
        app.controller.company.coin = 10

        await pilot.press("4")
        await pilot.pause()
        assert app.screen_state == "town_market"

        recruit_action = next(action for action in app.actions if action.value == "recruit")
        await pilot.press(recruit_action.number)
        await pilot.pause()
        assert app.screen_state == "recruiting"
        assert tuple(action.value for action in app.actions) == ("hire", "back")
        assert "Purpose\nReview a short slate" in app.body_text
        assert "Status\n- Reputation" in app.body_text
        assert "Candidates\n1." in app.body_text
        offer_names = [offer.name for offer in app.controller.recruit_offers]
        assert len(offer_names) == len(set(offer_names))
        assert "Back to Market Row" in tuple(action.label for action in app.actions)

        await pilot.press("1")
        await pilot.pause()
        assert app.screen_state == "recruiting_hire"
        assert any(action.value == "0" for action in app.actions)
        assert any(action.label == "Back to Recruitment Desk" for action in app.actions)
        first_offer = next(action for action in app.actions if action.value == "0")
        app.focused_command_index = app.actions.index(first_offer)
        assert "Candidate" in app._detail_text()
        assert "Background" in app._detail_text()
        assert "Roster Fit" in app._detail_text()

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "recruiting"
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "town_market"

        buy_action = next(action for action in app.actions if action.value == "buy")
        await pilot.press(buy_action.number)
        await pilot.pause()
        assert app.screen_state == "supply_shop"
        assert tuple(action.value for action in app.actions) == ("buy_supplies", "back")
        assert "Quartermaster" in app.body_text
        assert "Coin 10" in app.body_text
        assert "Purchase quantity: 1" in app.body_text
        assert "Stock" in app.body_text
        assert "Purpose" not in app.body_text
        assert "Back to Market Row" in tuple(action.label for action in app.actions)

        await pilot.press("1")
        await pilot.pause()
        assert app.screen_state == "supply_buy"
        assert any(action.value == "rations" for action in app.actions)
        assert any(action.label == "Back to Quartermaster" for action in app.actions)

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "supply_shop"


@pytest.mark.anyio
async def test_tui_opening_expedition_reaches_manual_combat_and_resolves_turn(
    tmp_path: Path,
) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        await start_opening_wilderness(app, pilot)
        assert app.screen_state == "playback"
        assert "Expedition Progress" in app.body_text
        assert "[>]" in app.body_text
        assert tuple(action.label for action in app.actions) == ("Continue",)
        assert "Mini Map" in app.log_text
        assert "@" in app.log_text
        assert "Haven East Gate" not in app.body_text

        await advance_playback_to(app, pilot, "dungeon")
        assert app.screen_state == "dungeon"
        assert "Haven East Gate" in app.body_text
        assert "The cave is past the dry creek" in app.body_text
        assert "Local Actions" not in app.body_text
        assert "Company Map" not in app.body_text
        assert "Ways Out" not in app.body_text
        assert "Mini Map" in app.log_text
        assert "\nGate\n\n" in app.log_text
        assert "1?" in app.log_text
        assert "Company Map" not in app.log_text
        assert "Current Routes" not in app.log_text
        assert "Old Road" not in app.log_text

        await reach_old_works_cache_combat(app, pilot)
        assert app.screen_state == "combat"
        assert app.current_combat_phase == "command"
        assert app.current_combat_view is not None
        assert "Combat Command" in app._live_body_text()
        assert "PARTY" in app._live_body_text()
        assert "ENEMIES" in app._live_body_text()
        assert "Commands\n1. Skills" not in app._live_body_text()
        assert "Mode:" not in app._live_body_text()
        assert "Target:" not in app._live_body_text()

        await pilot.press("enter")
        await pilot.pause()
        assert app.screen_state == "combat"
        assert app.current_combat_phase == "skill"

        await pilot.press("enter")
        await pilot.pause()
        assert app.screen_state == "combat"
        assert app.current_combat_phase == "target"
        assert app.selected_skill_id is not None
        assert "Target Focus" in app._detail_text()
        assert "Combat Command" in app._live_body_text()
        assert "Turns" in app._live_body_text()
        assert "TARGET PREVIEW" not in app._live_body_text()
        assert "<b>" not in app._live_body_text()
        assert "Turn" in app._live_body_text()
        assert "TARGET" in app._live_body_text()
        assert "[bold black on yellow]" not in app._live_body_text()
        assert "[bold white on red]" not in app._live_body_text()

        await pilot.press("enter")
        await pilot.pause()
        assert app.screen_state == "resolution"
        assert app.recent_events
        assert "Hero Action" in app.body_text
        had_enemy_response = bool(app.pending_enemy_events)
        if had_enemy_response:
            assert app.actions[0].label == "Enemy Response"

        await pilot.press("enter")
        await pilot.pause()
        enemy_screens_seen = 0
        while app.screen_state == "enemy_turn":
            enemy_screens_seen += 1
            assert app.screen_state == "enemy_turn"
            assert "Enemy Response" in app.body_text
            assert "Actor" not in app.body_text
            assert "Target" not in app.body_text
            assert "VS" in app.body_text
            assert "<-" not in app.body_text
            assert "Summary" not in app.body_text
            assert "Pressure" not in app.body_text
            assert "Log" not in app.body_text
            await pilot.press("enter")
            await pilot.pause()
        if had_enemy_response:
            assert enemy_screens_seen >= 1
        assert app.screen_state in {"combat", "dungeon", "breach", "expedition_report"}


@pytest.mark.anyio
async def test_tui_dungeon_enter_activates_focused_route(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        await start_opening_wilderness(app, pilot)
        await advance_playback_to(app, pilot, "dungeon")
        assert app.focused_action is not None
        assert app.focused_action.value == "old_road"
        assert app.focused_action.default
        route_detail = app._detail_text()
        assert "Old Road" in route_detail
        assert "North - Unexplored" in route_detail
        assert "Risk:" not in route_detail
        assert "Threat:" not in route_detail
        assert "Kind:" not in route_detail
        assert "Exit Focus" not in route_detail
        assert "Status:" not in route_detail
        assert "Type:" not in route_detail
        dock_text = CommandDock.render_text(
            app.actions,
            app.focused_command_index,
            shortcut_text=app._global_shortcut_text(),
        )
        assert "Commands" in dock_text
        assert "Exits" not in dock_text
        assert "Actions" not in dock_text
        assert "  1   Old Road" in dock_text
        assert "  2   Return to East Gate" in dock_text
        assert "[M] Map" in dock_text

        await pilot.press("enter")
        await pilot.pause()
        assert app.screen_state == "playback"

        await advance_playback_to(app, pilot, "dungeon")
        assert "Old Road" in app.body_text
        assert not any(action.default for action in app.actions)
        route_actions = [action for action in app.actions if str(action.kind) == "travel"]
        assert [action.value for action in route_actions] == [
            "bramble_shrine",
            "hunters_trail",
            "town_gate",
            "abandoned_toll_post",
        ]
        assert "\nRoad\n\n" in app.log_text
        assert "@" in app.log_text
        assert "3o" in app.log_text
        assert "2?" in app.log_text
        assert "1?" in app.log_text
        assert "4?" in app.log_text
        live_map = app._live_log_content()
        assert isinstance(live_map, Text)
        assert ">1?" in live_map.plain
        assert "Toll Post" not in app.log_text
        assert "Hunter's Trail" not in app.log_text

        haven_route = next(action for action in route_actions if action.value == "town_gate")
        await pilot.press(haven_route.number)
        await pilot.pause()
        assert app.screen_state == "dungeon"
        assert "Haven East Gate" in app.body_text
        assert "east gate stands open" in app.body_text


def test_tui_meaningful_playback_filter_keeps_major_revisits() -> None:
    app = CharterApp()
    ordinary_revisit = ExpeditionEvent(
        message="Returned to Old Road.",
        node_id="old_road",
        first_visit=False,
    )
    major_revisit = ExpeditionEvent(
        message="The breach is still wrong.",
        node_id="maze_breach",
        first_visit=False,
        major_beat=True,
    )

    assert app._meaningful_playback_events([ordinary_revisit, major_revisit]) == [major_revisit]


def test_tui_dungeon_focus_detail_explains_blocked_movement() -> None:
    controller = _controller_at_old_works_cache_combat()
    app = CharterApp(controller=controller)
    result = app.controller.handle(ViewDungeon())
    assert result.success
    assert isinstance(result.value, DungeonView)
    app.current_dungeon_view = result.value
    route_action = next(
        action for action in result.value.actions if action.value == "fungus_chamber"
    )

    detail = app._dungeon_detail_text(route_action)

    assert "Fungus-Lit Gallery" in detail
    assert "Clear this room before moving." in detail
    assert "Exit Focus" not in detail
    assert "Status:" not in detail
    assert "Type:" not in detail


@pytest.mark.anyio
async def test_tui_dungeon_focus_shows_concrete_route_preview(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        await start_opening_wilderness(app, pilot)
        await advance_playback_to(app, pilot, "dungeon")

        assert app.focused_action is not None
        assert_focused_action_is_safe(app)
        detail = app._detail_text()

        assert "Old Road" in detail
        assert "North - Unexplored" in detail
        assert "Expect a new room, obstacle, or discovery." in detail
        assert detail.count("Old Road") == 1
        assert "Risk:" not in detail
        assert "Threat:" not in detail
        assert "Kind:" not in detail
        assert "direction " not in detail
        help_text = app._dock_help_text()
        assert "Take the focused route." in help_text
        assert "Use Up/Down" in help_text
        assert "unexplored" not in help_text


def test_tui_dungeon_hazard_route_avoids_generic_visible_risk() -> None:
    controller = _started_interactive_dungeon_controller()
    _move_dungeon(
        controller,
        "old_road",
        "hunters_trail",
        "dry_creek_bed",
        "black_stone_sinkhole",
        "shallow_cave_entrance",
        "shallow_cave_room_1",
        "cave_fork",
    )
    app = CharterApp(controller=controller)
    result = app.controller.handle(ViewDungeon())
    assert result.success
    assert isinstance(result.value, DungeonView)
    app.current_dungeon_view = result.value
    route_action = next(action for action in result.value.actions if action.value == "narrow_crawl")

    detail = app._dungeon_detail_text(route_action)

    assert "Narrow Crawl" in detail
    assert "South - Unexplored" in detail
    assert "Expect a new room, obstacle, or discovery." in detail
    assert "Risk: Caution" not in detail
    assert "Risk:" not in detail
    assert "Threat: Hazard" not in detail
    assert "Warning:" not in detail
    assert route_action.risk == "costly"
    assert not app._is_safe_default(route_action)


def test_tui_dungeon_boss_route_keeps_exceptional_warning() -> None:
    controller = _controller_at_stone_gate()
    assert controller.company is not None
    controller.company.inventory["cave_key"] = 1
    _use_dungeon_action(controller, "unlock_black_gate")
    app = CharterApp(controller=controller)
    result = app.controller.handle(ViewDungeon())
    assert result.success
    assert isinstance(result.value, DungeonView)
    app.current_dungeon_view = result.value
    route_action = next(
        action for action in result.value.actions if action.value == "maze_touched_lair"
    )

    detail = app._dungeon_detail_text(route_action)

    assert "Maze-Touched Lair" in detail
    assert "South - Unexplored - Boss" in detail
    assert "Warning: serious danger ahead." in detail
    assert "Risk:" not in detail
    assert "Threat:" not in detail
    assert route_action.risk == "risky"
    assert not app._is_safe_default(route_action)
    dock = CommandDock.render_text(app._dungeon_navigation_actions(result.value), 2)
    assert "!Maze-Touched Lair" in dock
    map_text = DungeonMapPanel.render_minimap_text(
        result.value,
        actions=app._dungeon_navigation_actions(result.value),
        highlighted_node_id="maze_touched_lair",
    )
    assert ">3!" in map_text


def test_tui_combat_focus_shows_stage_consequence() -> None:
    controller = _controller_at_old_works_cache_combat()
    app = CharterApp(controller=controller)
    result = app.controller.handle(ViewCombat())
    assert result.success
    assert isinstance(result.value, CombatView)
    app.screen_state = "combat"
    app.current_combat_view = result.value
    app.current_combat_phase = "command"
    app.actions = result.value.commands

    assert app.focused_action is not None
    assert_focused_action_is_safe(app)

    detail = app._detail_text()
    help_text = app._dock_help_text()
    assert "Command Focus" in detail
    assert "Preview" in detail
    assert "Consequence" in detail
    assert "Choose a skill" in help_text
    assert "Risk" not in help_text


def test_tui_dungeon_blocked_actions_live_in_interact_detail() -> None:
    controller = _controller_at_stone_gate()
    assert controller.company is not None
    controller.company.supplies["rope"] = 0
    app = CharterApp(controller=controller)
    result = app.controller.handle(ViewDungeon())
    assert result.success
    assert isinstance(result.value, DungeonView)
    app.screen_state = "dungeon"
    app.current_dungeon_view = result.value
    app.body_text = app._dungeon_place_text(result.value)
    app.actions = app._dungeon_navigation_actions(result.value)

    interact = next(action for action in app.actions if action.value == "interact")
    assert not interact.enabled
    assert "Unlock Black Gate" not in app.body_text
    assert "Force Black Gate" not in app.body_text
    assert "Focus: Black Gate" in app.body_text
    assert "Action: Interact to handle it." in app.body_text

    detail = app._dungeon_detail_text(interact)

    assert "Unlock Black Gate - Needs 1 cave key." in detail
    assert "Force Black Gate - Needs 1 rope." in detail
    assert "missing item" not in detail
    assert "missing supplies" not in detail


def test_tui_dungeon_place_text_includes_party_watch() -> None:
    controller = _started_interactive_dungeon_controller()
    app = CharterApp(controller=controller)
    result = app.controller.handle(ViewDungeon())
    assert result.success
    assert isinstance(result.value, DungeonView)

    body = app._dungeon_place_text(result.value)

    assert "Party Watch" in body
    assert "Mara" in body
    assert "Senn" in body


def test_tui_dungeon_minimap_uses_renumbered_navigation_actions() -> None:
    controller = _started_interactive_dungeon_controller()
    _move_dungeon(controller, "old_road", "abandoned_toll_post")
    app = CharterApp(controller=controller)
    result = app.controller.handle(ViewDungeon())
    assert result.success
    assert isinstance(result.value, DungeonView)
    view = result.value

    actions = app._dungeon_navigation_actions(view)
    text = DungeonMapPanel.render_minimap_text(view, actions=actions)

    assert [action.value for action in actions[:3]] == [
        "old_road",
        "bandit_camp",
        "interact",
    ]
    assert "1o" in text
    assert "2?" in text
    assert "3?" not in text
    focused_text = DungeonMapPanel.render_minimap_text(
        view,
        actions=actions,
        highlighted_node_id="bandit_camp",
    )
    assert ">2?" in focused_text
    assert text.count("Legend: @ you   o known   ? unknown   ! quest") == 1
    assert ">2?" not in text


def test_tui_dungeon_command_dock_keeps_flat_numbered_actions() -> None:
    controller = _started_interactive_dungeon_controller()
    _move_dungeon(controller, "old_road", "abandoned_toll_post")
    app = CharterApp(controller=controller)
    result = app.controller.handle(ViewDungeon())
    assert result.success
    assert isinstance(result.value, DungeonView)

    actions = app._dungeon_navigation_actions(result.value)
    dock = CommandDock.render_text(actions, 1)

    assert "Commands" in dock
    assert "Exits" not in dock
    assert "Actions" not in dock
    assert "  1  " in dock
    assert ">  2  " in dock
    assert "  3  " in dock
    assert dock.index("Old Road") < dock.index("Bandit Camp")
    assert dock.index("Bandit Camp") < dock.index("Interact")


@pytest.mark.anyio
async def test_tui_dungeon_map_page_keeps_full_map_detail(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        await start_opening_wilderness(app, pilot)
        await advance_playback_to(app, pilot, "dungeon")
        assert not any(action.value == "map" for action in app.actions)
        assert app._global_shortcut_text() == (
            "Shortcuts: [M] Map  [P] Pack  [C] Company  [?] Help"
        )

        await pilot.press("m")
        await pilot.pause()
        assert app.screen_state == "dungeon_map"
        assert "Company Map" in app.body_text
        assert "Survey" in app.body_text
        assert "Legend" in app.body_text
        assert "Inventory" in app.body_text
        assert "Known Places" in app.body_text
        assert "Current Routes" in app.body_text
        assert "Haven East Gate" in app.body_text
        assert "Mini Map" in app.log_text

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "dungeon"


def test_dungeon_map_page_surfaces_node_inventory_options() -> None:
    controller = _controller_at_stone_gate()
    result = controller.handle(ViewDungeon())
    assert result.success
    assert isinstance(result.value, DungeonView)

    text = DungeonMapPanel.render_text(result.value)

    assert "Black Stone Gate" in text
    assert "inventory: needs cave key x1" in text
    assert "costs rope x1" in text
    assert "action: Unlock Black Gate: needs 1 cave key" in text
    assert "action: Force Black Gate: costs 1 rope" in text


@pytest.mark.anyio
async def test_tui_dungeon_safe_return_shows_arrival_brief(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        await start_opening_wilderness(app, pilot)
        await advance_playback_to(app, pilot, "dungeon")

        return_action = next(action for action in app.actions if action.value == "return")
        await pilot.press(return_action.number)
        await pilot.pause()
        assert app.screen_state == "regional_place"
        assert "Returned to Haven East Gate" in app.body_text
        assert "What Changed" in app.body_text
        assert "Company record filed." in app.body_text
        assert "Records" in app.body_text
        assert app.controller.company is not None
        assert app.controller.company.last_expedition_report is not None
        assert app.controller.company.town_state["regional_node_id"] == "town_gate"
        assert any(action.value == "enter_haven" for action in app.actions)

        app._show_town_submenu("town_records")
        assert any(action.value == "latest_record" for action in app.actions)
        app._handle_town_submenu_action("latest_record")
        assert app.screen_state == "expedition_report"
        assert "Filed Company Record" in app.body_text
        assert "Outcome: Returned To Haven" in app.body_text


@pytest.mark.anyio
async def test_tui_dungeon_room_action_plays_back_and_updates_room(tmp_path: Path) -> None:
    controller = _controller_at_old_works_cache_combat()
    final_combat_result = _win_active_manual_combat(controller)
    app = CharterApp(controller=controller, save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        app.current_combat_view = final_combat_result.value
        app._show_resolution(final_combat_result.events, final_combat_result.hci)
        assert app.screen_state == "resolution"
        assert app.actions[0].label == "Return to Dungeon"
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen_state == "dungeon"
        await choose_dungeon_value(
            app,
            pilot,
            "action:recover_gate_key",
            expected_playback=False,
        )
        await pilot.pause()
        assert app.screen_state == "dungeon"
        assert "brass cave key drops" in app._detail_text()
        assert "Local Actions" not in app.body_text
        assert "Recover Gate Key" not in app.body_text
        assert app.controller.company is not None
        assert app.controller.company.inventory["cave_key"] == 1
        assert not any(action.value == "interact" for action in app.actions)

        await choose_dungeon_value(app, pilot, "fungus_chamber")
        await choose_dungeon_value(app, pilot, "stone_gate")
        assert app.screen_state == "dungeon"
        assert any(action.value == "interact" for action in app.actions)
        assert not any(action.value == "action:unlock_black_gate" for action in app.actions)
        assert not any(action.value == "action:force_black_gate" for action in app.actions)
        interact_detail = app._dungeon_detail_text(
            next(action for action in app.actions if action.value == "interact")
        )
        assert "Unlock Black Gate" in interact_detail
        assert "Force Black Gate" in interact_detail

        await pilot.press("i")
        await pilot.pause()
        assert app.screen_state == "dungeon_interact"
        assert not any(
            action.default for action in app.actions if action.value.startswith("action:")
        )
        assert any(action.value == "action:unlock_black_gate" for action in app.actions)
        assert any(action.value == "action:force_black_gate" for action in app.actions)
        force_action = next(
            action for action in app.actions if action.value == "action:force_black_gate"
        )
        force_detail = app._dungeon_detail_text(force_action)
        assert "Cost: rope" in force_detail
        assert "Risk:" not in force_detail


@pytest.mark.anyio
async def test_tui_formation_uses_party_grid(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await start_default_company(app, pilot)
        await pilot.press("3")
        await pilot.pause()
        assert app.screen_state == "town_yard"

        await pilot.press("1")
        await pilot.pause()

        assert app.screen_state == "formation"
        assert "Party Formation" in app.body_text
        assert "Front Left" in app.body_text
        assert "FRONT LEFT" not in app.body_text
        assert "Available" not in app.body_text
        assert any(action.label.startswith("Back Left:") for action in app.actions)
        assert not any(action.label.startswith("Edit ") for action in app.actions)

        await pilot.press("1")
        await pilot.pause()

        assert app.screen_state == "assign_hero"
        assert "Slot: Back Left" in app.body_text
        assert "Put Mara Vell in Back Left" in app.actions[0].preview
        assert "Before:" in app.actions[0].description


@pytest.mark.anyio
async def test_tui_escape_and_backspace_return_where_available(tmp_path: Path) -> None:
    app = CharterApp(save_path=tmp_path / "company.json")

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("1")
        await pilot.pause()
        assert app.screen_state == "name_company"

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_state == "main"
