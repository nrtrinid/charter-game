"""Rich renderers for terminal UI screens."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from game.app.views import (
    CombatActorView,
    CombatView,
    ExpeditionReportView,
    FormationView,
    GearInventoryView,
    HeroListEntry,
    MemorialEntryView,
    RecruitOffersView,
    RosterSectionView,
    ScreenAction,
    ShellStatusView,
    SupplyShopView,
    TownDashboardView,
    build_shell_status,
)
from game.campaign.company import CompanyState, HeroState
from game.core.events import (
    CombatEndedEvent,
    CombatRetreatDeclaredEvent,
    CombatRetreatedEvent,
    DamageEvent,
    DeathEvent,
    DownedEvent,
    EncounterEndedEvent,
    EncounterStartedEvent,
    EnemyIntentEvent,
    GameEvent,
    HealingEvent,
    MissEvent,
    MoveEvent,
    ReactionSkippedEvent,
    ReactionUsedEvent,
    RoundEndedEvent,
    RoundStartedEvent,
    SkillUsedEvent,
    StatusChangedEvent,
    TurnDelayedEvent,
    TurnPassedEvent,
)
from game.core.hci import EventBeat, HciResultAnalysis, build_event_beats
from game.ui.hci_text import action_dock_detail, hci_summary_lines
from game.ui.wounds import mortal_wound_badge, mortal_wound_count

MAIN_MENU_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("1", "Haven Town", "town, t"),
    ("2", "Expedition", "expedition, x"),
    ("3", "Save / Load", "save, load"),
    ("4", "Help", "help, ?"),
    ("5", "Quit", "quit, q"),
)

COMPANY_MENU_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("1", "Start New Company", "start, new"),
    ("2", "Haven Town", "town, t"),
    ("3", "Roster", "roster, r"),
    ("4", "Supplies", "supplies, s"),
    ("5", "Ledger", "ledger"),
    ("6", "Back", "back, b"),
)

TOWN_MENU_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("1", "Expedition", "expedition, x"),
    ("2", "Formation", "formation, party"),
    ("3", "Recruiting", "recruit, hire"),
    ("4", "Quartermaster", "buy, supplies"),
    ("5", "Roster", "roster, r"),
    ("6", "Back to Main", "back, b"),
    ("7", "Recovery Ward", "recover"),
    ("8", "Ledger", "ledger, l"),
    ("9", "Memorial", "memorial"),
)

EXPEDITION_MENU_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("1", "Begin / Resume Opening Expedition", "begin, expedition, x"),
    ("2", "Back", "back, b"),
)

SAVE_MENU_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("1", "Save", "save"),
    ("2", "Load", "load"),
    ("3", "Back", "back, b"),
)

PROMPT_RESERVED_LINES = 2
MIN_BODY_LINES = 3
MIN_LOG_LINES = 3
MAX_LOG_LINES = 6


@dataclass(frozen=True)
class ScreenFrame:
    title: str
    body: Any
    log: Any | None = None
    command_dock: Any | None = None
    hint: str = ""


def render_screen(
    company: CompanyState | None,
    save_path: Path,
    title: str,
    body: Any,
    *,
    hint: str = "",
    command_dock: Any | None = None,
    log: Any | None = None,
    console: Console | None = None,
    enable_spacer: bool = False,
) -> Group:
    return render_screen_frame(
        company,
        save_path,
        ScreenFrame(title=title, body=body, log=log, command_dock=command_dock, hint=hint),
        console=console,
        enable_spacer=enable_spacer,
    )


def render_screen_frame(
    company: CompanyState | None,
    save_path: Path,
    frame: ScreenFrame,
    *,
    console: Console | None = None,
    enable_spacer: bool = False,
) -> Group:
    header = render_status_header(company, save_path)
    body = Panel(frame.body, title=frame.title, border_style="cyan")
    if _use_fixed_viewport(console, enable_spacer):
        return _render_fixed_viewport_frame(
            header,
            body,
            frame,
            console=console,
        )
    parts: list[Any] = [
        header,
        body,
    ]
    if frame.log is not None:
        parts.append(frame.log)
    if frame.command_dock is not None:
        spacer_lines = _spacer_line_count(
            (*parts, frame.command_dock),
            console=console,
            enabled=enable_spacer,
        )
        parts.extend(Text("") for _ in range(spacer_lines))
        parts.append(frame.command_dock)
    elif frame.hint:
        parts.append(render_footer(frame.hint))
    return Group(*parts)


def render_status_header(company: CompanyState | None, save_path: Path) -> Panel:
    return render_status_view(
        build_shell_status(company, str(save_path), save_exists=save_path.exists())
    )


def render_status_view(status_view: ShellStatusView) -> Panel:
    if status_view.company_name is None:
        status = Text("No company loaded", style="yellow")
    else:
        status = Text.assemble(
            (status_view.company_name, "bold cyan"),
            "   ",
            ("Location ", "dim"),
            status_view.location,
            "   ",
            ("Rep ", "dim"),
            str(status_view.reputation),
            "   ",
            ("Coin ", "dim"),
            str(status_view.coin),
            "\n",
            ("Active ", "dim"),
            str(status_view.active_count),
            "   ",
            ("Reserves ", "dim"),
            str(status_view.reserve_count),
            "   ",
            ("Wounded ", "dim"),
            str(status_view.wounded_count),
            "   ",
            ("Downed ", "dim"),
            str(status_view.downed_count),
            "   ",
            ("Breaches ", "dim"),
            status_view.breaches,
        )
    status.append(
        f"\nSave slot: {status_view.save_path} ({status_view.save_status})",
        style="dim",
    )
    return Panel(status, title="The Charter v0.1", border_style="cyan")


def render_footer(hint: str = "") -> Text:
    return Text(hint or "Numbers select actions. Aliases work as shortcuts.", style="dim")


def render_menu(options: tuple[tuple[str, str, str], ...]) -> Table:
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Action", style="white")
    table.add_column("Aliases", style="dim")
    for number, action, aliases in options:
        table.add_row(number, action, aliases)
    return table


def render_action_list(actions: tuple[ScreenAction, ...]) -> Panel:
    return render_command_dock(actions, prompt="Action")


def render_command_dock(
    actions: tuple[ScreenAction, ...],
    *,
    prompt: str,
    guidance: str = "",
) -> Panel:
    table = Table(
        show_header=True,
        header_style="bold",
        expand=True,
        box=None,
        padding=(0, 1),
        pad_edge=False,
    )
    table.add_column("#", justify="right", style="bright_yellow", no_wrap=True)
    table.add_column("Command", style="white", ratio=2)
    table.add_column("Hotkey", style="bright_yellow", ratio=1)
    table.add_column("State", style="dim", ratio=1)
    table.add_column("Detail", style="dim", ratio=3)
    for action in actions:
        state = _command_state_label(action)
        style = "dim" if not action.enabled else "bold" if action.default else None
        table.add_row(
            action.number,
            action.label,
            _primary_hotkey(action),
            state,
            action_dock_detail(action),
            style=style,
        )
    has_default = any(action.default and action.enabled for action in actions)
    dock_guidance = guidance
    if not dock_guidance:
        dock_guidance = (
            "Enter accepts the default. Numbers select commands. Aliases work as shortcuts."
            if has_default
            else "Numbers select commands. Aliases work as shortcuts."
        )
    guidance_line = Text(dock_guidance, style="dim")
    return Panel(
        Group(table, guidance_line),
        title="Command Dock",
        border_style="bright_yellow",
        box=box.HEAVY,
        padding=(0, 1),
    )


def render_main_menu() -> Table:
    return render_menu(MAIN_MENU_OPTIONS)


def render_company_menu() -> Table:
    return render_menu(COMPANY_MENU_OPTIONS)


def render_town_menu() -> Table:
    return render_menu(TOWN_MENU_OPTIONS)


def render_expedition_menu(*, breach_pending: bool = False) -> Table:
    table = render_menu(EXPEDITION_MENU_OPTIONS)
    if breach_pending:
        table.caption = "A discovered breach is waiting for a return-or-descend decision."
    return table


def render_save_menu() -> Table:
    return render_menu(SAVE_MENU_OPTIONS)


def render_choice_list(options: tuple[tuple[str, str, str], ...]) -> Panel:
    actions = tuple(
        ScreenAction(
            number,
            label,
            label.lower().replace(" ", "_"),
            tuple(alias for alias in aliases.split(", ") if alias),
        )
        for number, label, aliases in options
    )
    return render_command_dock(actions, prompt="Choice")


def render_help() -> Panel:
    return Panel(
        "Menus accept numbers and aliases. Expedition playback advances by scene or encounter: "
        "choose Continue, Auto, or Stop to control the current section "
        "to a safe menu. Interactive expeditions pause only for hero skill and target choices.",
        border_style="blue",
    )


def render_roster(roster: list[HeroState]) -> Table | Panel:
    if not roster:
        return Panel("No living roster members.", border_style="yellow")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Slot", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Class")
    table.add_column("HP", justify="right")
    table.add_column("Effort", justify="right")
    table.add_column("Morale")
    table.add_column("Strain")
    table.add_column("Mortal Wounds", justify="right")
    table.add_column("Status")
    table.add_column("Traits")
    for hero in roster:
        statuses = hero.life_state.value if hero.life_state.value != "alive" else "ready"
        table.add_row(
            hero.formation_slot.value,
            hero.name,
            hero.class_id,
            f"{hero.hp}/{hero.max_hp}",
            f"{hero.effort}/{hero.max_effort}",
            hero.morale.name.title(),
            hero.strain.name.title(),
            mortal_wound_count(hero.mortal_wounds),
            statuses,
            _raw_hero_trait_summary(hero),
        )
    return table


def render_roster_sections(sections: tuple[RosterSectionView, ...]) -> Group:
    return Group(*(_render_hero_entries(section.title, section.heroes) for section in sections))


def render_supplies(supplies: dict[str, int]) -> Table | Panel:
    if not supplies:
        return Panel("No supplies.", border_style="yellow")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Supply", style="cyan")
    table.add_column("Quantity", justify="right")
    for name, quantity in sorted(supplies.items()):
        table.add_row(name, str(quantity))
    return table


def render_gear_inventory(view: GearInventoryView) -> Group:
    status = Table(show_header=False, box=None, padding=(0, 2))
    status.add_column("Field", style="dim")
    status.add_column("Value", style="bold")
    status.add_row("Reputation", str(view.reputation))
    status.add_row("Coin", str(view.coin))
    status.add_row("Management", "available" if view.can_manage else view.manage_reason)
    status.add_row("Purchases", "available" if view.can_purchase else view.purchase_reason)

    items = Table(show_header=True, header_style="bold")
    items.add_column("Gear", style="cyan")
    items.add_column("State")
    items.add_column("Owned", justify="right")
    items.add_column("Available", justify="right")
    items.add_column("Cost", justify="right")
    items.add_column("Effect")
    items.add_column("Note")
    for item in view.items:
        items.add_row(
            item.name,
            item.state,
            str(item.owned_count),
            str(item.available_count),
            "reward" if item.cost is None else str(item.cost),
            item.effect_summary,
            item.unavailable_reason,
        )

    heroes = Table(show_header=True, header_style="bold")
    heroes.add_column("Hero", style="cyan")
    heroes.add_column("Class")
    heroes.add_column("Equipped")
    heroes.add_column("Condition")
    for hero in view.heroes:
        heroes.add_row(
            hero.name,
            hero.class_id,
            hero.equipped_gear_name or "none",
            hero.condition,
        )

    return Group(
        Panel(status, title="Armory", border_style="cyan"),
        Panel(items, title="Company Gear", border_style="cyan"),
        Panel(heroes, title="Equipped Kits", border_style="cyan"),
    )


def render_ledger(ledger: dict[str, object]) -> Table:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for key, value in ledger.items():
        table.add_row(str(key), _display_value(value))
    return table


def render_town(view: TownDashboardView) -> Group:
    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column("Field", style="dim")
    summary.add_column("Value", style="bold")
    summary.add_row("Reputation", str(view.reputation))
    summary.add_row("Coin", str(view.coin))
    summary.add_row("Roster", f"{view.active_count + view.reserve_count} / {view.roster_cap}")
    summary.add_row("Active", str(view.active_count))
    summary.add_row("Reserves", str(view.reserve_count))
    summary.add_row("Wounded", str(view.wounded_count))
    summary.add_row("Downed", str(view.downed_count))
    summary.add_row("Memorial", str(view.deceased_count))
    return Group(
        Panel(summary, title="Haven Town", border_style="cyan"),
        _lines_panel("Current Objective", _objective_lines(view.objective)),
        _lines_panel(
            "Posted Contracts",
            tuple(
                _contract_summary_line(entry)
                for entry in view.contract_board
            ),
        ),
        _lines_panel(
            "Company Upgrades",
            tuple(_upgrade_summary_line(entry) for entry in view.upgrades),
        ),
        _render_hero_entries("Active Party", view.active_party),
        _render_hero_entries("Reserves", view.reserves),
    )


def render_memorial(
    heroes: tuple[MemorialEntryView, ...] | list[MemorialEntryView],
) -> Table | Panel:
    if not heroes:
        return Panel("No fallen heroes are recorded.", border_style="cyan")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="bold")
    table.add_column("Class")
    table.add_column("Mortal Wounds", justify="right")
    table.add_column("Final Memory")
    for hero in heroes:
        table.add_row(
            hero.name,
            hero.class_id,
            mortal_wound_count(hero.mortal_wounds),
            hero.final_memory or "none",
        )
    return table


def render_expedition_report(view: ExpeditionReportView) -> Group:
    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column("Field", style="dim")
    summary.add_column("Value", style="bold")
    summary.add_row("Outcome", view.outcome.replace("_", " ").title())
    summary.add_row("Route", view.expedition_id)
    summary.add_row("Dungeon", view.dungeon_id or "none")
    summary.add_row(
        "Reputation",
        f"{view.reputation_start}->{view.reputation_end} ({_signed(view.reputation_delta)})",
    )
    summary.add_row(
        "Coin",
        f"{view.coin_start}->{view.coin_end} ({_signed(view.coin_delta)})",
    )
    return Group(
        Panel(summary, title="Filed Company Record", border_style="cyan"),
        _lines_panel("Record Brief", _report_brief_lines(view)),
        _lines_panel("What Changed", view.what_changed),
        _lines_panel("Next Objective", _objective_lines(view.objective)),
        _lines_panel("Rooms Entered", view.rooms_entered),
        _lines_panel("Encounters Resolved", view.encounters_resolved),
        _delta_panel("Supplies", view.supply_deltas),
        _delta_panel("Inventory", view.inventory_deltas),
        _delta_panel("Gear", view.gear_deltas),
        _lines_panel("Hero Outcomes", view.hero_outcomes),
        _lines_panel("Notable Moments", view.notable_moments),
    )


def render_recruit_offers(view: RecruitOffersView) -> Group | Panel:
    if not view.offers:
        return Panel("No recruits are currently waiting.", border_style="yellow")
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Class")
    table.add_column("Background")
    table.add_column("Cost", justify="right")
    for index, offer in enumerate(view.offers, start=1):
        table.add_row(
            str(index),
            offer.name,
            offer.class_id,
            offer.background or "unknown",
            str(offer.cost),
        )
    summary = Panel(
        f"Reputation: {view.reputation}   Coin: {view.coin}   "
        f"Roster: {view.roster_count}/{view.roster_cap}",
        border_style="cyan",
    )
    return Group(summary, table)


def render_supply_shop(view: SupplyShopView) -> Group:
    return Group(Panel(f"Coin available: {view.coin}", border_style="cyan"))


def render_formation(view: FormationView) -> Group:
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Slot", style="cyan")
    table.add_column("Hero", style="bold")
    table.add_column("Condition")
    for index, slot_view in enumerate(view.slots, start=1):
        table.add_row(
            str(index),
            slot_view.slot_label,
            slot_view.hero_name,
            slot_view.condition,
        )
    return Group(table)


def render_combat_view(view: CombatView) -> Group:
    parts: list[Any] = [
        _render_combat_header(view),
    ]
    focus_art = _render_combat_focus_art(view)
    if focus_art is not None:
        parts.append(focus_art)
    parts.extend(
        (
            _render_combat_formation("Party", view.party),
            _render_combat_formation("Enemies", view.enemies),
        )
    )
    if view.pending_enemy_intent is not None:
        parts.extend(
            (
                _render_enemy_intent(view),
                _render_reaction_options(view),
            )
        )
    else:
        parts.extend(
            (
                _render_combat_commands(view),
                _render_skill_options(view),
                _render_target_options(view),
                _render_move_options(view),
            )
        )
    return Group(*parts)


def _render_combat_focus_art(view: CombatView) -> Panel | None:
    actors = _combat_focus_actors(view)
    if not actors:
        return None
    lines = _portrait_pair_lines(actors)
    if not lines:
        return None
    return Panel("\n".join(lines), title="Focus", border_style="magenta")


def _combat_focus_actors(view: CombatView) -> tuple[CombatActorView, ...]:
    if view.pending_enemy_intent is not None:
        source = _actor_by_id(view, view.pending_enemy_intent.enemy_id)
        target = _actor_by_id(view, view.pending_enemy_intent.target_id)
        return tuple(actor for actor in (source, target) if actor is not None)
    if view.selected_skill_id is not None and view.targets:
        target_option = next(
            (option for option in view.targets if option.action.default),
            view.targets[0],
        )
        target_actor = _actor_by_id(view, target_option.target_id)
        return tuple(actor for actor in (view.current_actor, target_actor) if actor is not None)
    return (view.current_actor,) if view.current_actor is not None else ()


def _actor_by_id(view: CombatView, actor_id: str) -> CombatActorView | None:
    return next(
        (actor for actor in (*view.party, *view.enemies) if actor.actor_id == actor_id),
        None,
    )


def _portrait_pair_lines(actors: tuple[CombatActorView, ...]) -> list[str]:
    visible = tuple(actor for actor in actors if actor.art_lines)
    if not visible:
        return []
    if len(visible) == 1:
        actor = visible[0]
        return [
            actor.name,
            *_compact_art_lines(actor.art_lines, max_lines=6, max_width=18),
            f"HP {actor.hp}/{actor.max_hp}  Effort {actor.effort}/{actor.max_effort}",
        ]
    left, right = visible[:2]
    width = 18
    left_art = _padded_art_lines(left.art_lines, width=width, max_lines=6)
    right_art = _padded_art_lines(right.art_lines, width=width, max_lines=6)
    lines = [
        f"{left.name[:width]:<{width}}  ->  {right.name[:width]:<{width}}",
    ]
    lines.extend(
        f"{left_line:<{width}}      {right_line:<{width}}"
        for left_line, right_line in zip(left_art, right_art, strict=True)
    )
    lines.append(
        f"HP {left.hp}/{left.max_hp:<{width - 4}}  ->  HP {right.hp}/{right.max_hp}"
    )
    return lines


def _padded_art_lines(
    art_lines: tuple[str, ...],
    *,
    width: int,
    max_lines: int,
) -> list[str]:
    lines = _compact_art_lines(art_lines, max_lines=max_lines, max_width=width)
    while len(lines) < max_lines:
        lines.append("")
    return lines


def _compact_art_lines(
    art_lines: tuple[str, ...],
    *,
    max_lines: int,
    max_width: int,
) -> list[str]:
    return [line[:max_width].rstrip() for line in art_lines[:max_lines]]


def render_recent_log(
    events: tuple[GameEvent, ...] | list[GameEvent],
    *,
    title: str = "Recent Log",
    max_events: int = 6,
) -> Panel:
    compact_events = list(events)[-max_events:]
    return Panel(render_events(compact_events), title=title, border_style="dim")


def render_save_slot(save_path: Path) -> Panel:
    status = "present" if save_path.exists() else "empty"
    style = "green" if save_path.exists() else "yellow"
    return Panel(f"{save_path}\nStatus: {status}", border_style=style)


def render_breach_prompt(company: CompanyState) -> Panel:
    wounded = [
        (
            f"{hero.name} {hero.hp}/{hero.max_hp} HP"
            + (f", {mortal_wound_badge(hero.mortal_wounds)}" if hero.mortal_wounds else "")
        )
        for hero in company.roster
        if hero.hp < hero.max_hp or hero.mortal_wounds > 0
    ]
    condition = "; ".join(wounded) if wounded else "No recorded wounds."
    return Panel(
        Text.assemble(
            ("Shallow Cave Breach discovered.\n", "bold cyan"),
            ("Company condition: ", "dim"),
            condition,
            "\nChoose from the numbered options below.",
        ),
        border_style="cyan",
    )


def render_event_beat(beat: EventBeat, *, show_title: bool = True) -> Panel:
    if beat.combat:
        lines = _summarize_combat(beat.events)
    else:
        lines = [event.message for event in beat.events]
    title = beat.title if show_title else None
    return Panel("\n".join(lines), title=title, border_style=beat.style)


def render_events(events: list[GameEvent]) -> Group:
    beats = build_event_beats(events)
    panels = [render_event_beat(beat) for beat in beats]
    if not panels:
        panels.append(Panel("Nothing happens.", title="Log", border_style="dim"))
    return Group(*panels)


def render_events_text(events: list[GameEvent]) -> str:
    return "\n".join(event.message for event in events)


def render_hci_summary(
    hci: HciResultAnalysis,
    *,
    title: str = "Outcome",
    style: str = "cyan",
) -> Panel:
    lines = hci_summary_lines(hci)
    return Panel("\n".join(lines) if lines else "Done.", title=title, border_style=style)


def render_resolution_card(
    events: list[GameEvent],
    hci: HciResultAnalysis | None = None,
) -> Panel:
    if hci is not None:
        summary_lines = hci_summary_lines(hci)
        if summary_lines:
            return Panel("\n".join(summary_lines), title="Resolution", border_style="red")
    lines: list[str] = []
    for beat in build_event_beats(events):
        if beat.combat:
            lines.extend(_summarize_combat(beat.events))
        else:
            lines.extend(event.message for event in beat.events)
    if not lines:
        lines.append("The turn resolves.")
    return Panel("\n".join(lines), title="Resolution", border_style="red")


def render_notice(message: str, *, title: str = "Notice", style: str = "blue") -> Panel:
    return Panel(message, border_style=style)


def _summarize_combat(events: list[GameEvent]) -> list[str]:
    lines: list[str] = []
    current_action: str | None = None
    has_structured_ending = any(isinstance(event, EncounterEndedEvent) for event in events)

    def flush_action() -> None:
        nonlocal current_action
        if current_action is not None:
            lines.append(f"  - {current_action}.")
            current_action = None

    for event in events:
        if isinstance(event, EncounterStartedEvent):
            flush_action()
            lines.append(event.message)
        elif isinstance(event, RoundStartedEvent):
            flush_action()
            lines.append(f"Round {event.round_number}")
        elif isinstance(event, RoundEndedEvent):
            flush_action()
        elif isinstance(event, SkillUsedEvent):
            flush_action()
            current_action = event.message.rstrip(".")
        elif isinstance(event, DamageEvent) and current_action is not None:
            current_action += f"; hits for {event.amount}"
        elif isinstance(event, HealingEvent) and current_action is not None:
            current_action += f"; restores {event.amount} HP"
        elif isinstance(event, MissEvent) and current_action is not None:
            current_action += "; misses"
        elif isinstance(event, MoveEvent):
            flush_action()
            lines.append(f"  - {event.message}")
        elif isinstance(event, TurnDelayedEvent | TurnPassedEvent):
            flush_action()
            lines.append(f"  - {event.message}")
        elif isinstance(event, CombatRetreatDeclaredEvent):
            flush_action()
            lines.append(f"  - {event.message}")
        elif isinstance(event, CombatRetreatedEvent):
            flush_action()
            lines.append(f"  - {event.message}")
        elif isinstance(event, EnemyIntentEvent):
            flush_action()
            lines.append(f"  - {event.message}")
        elif isinstance(event, ReactionUsedEvent | ReactionSkippedEvent):
            flush_action()
            lines.append(f"  - {event.message}")
        elif isinstance(event, DeathEvent | DownedEvent) and current_action is not None:
            current_action += f"; {event.message.rstrip('.')}"
        elif isinstance(event, StatusChangedEvent):
            continue
        elif isinstance(event, CombatEndedEvent) and not has_structured_ending:
            flush_action()
            lines.append(f"Outcome: {event.message}")
        elif isinstance(event, EncounterEndedEvent):
            flush_action()
            lines.append(f"Outcome: {event.message}")
        else:
            flush_action()
            lines.append(f"  - {event.message}")

    flush_action()
    return lines


def _render_combat_header(view: CombatView) -> Panel:
    order_text = _turn_order_text(view)
    if view.pending_enemy_intent is not None:
        intent = view.pending_enemy_intent
        text = Text.assemble(
            ("Enemy Intent ", "dim"),
            (intent.enemy_name, "bold red"),
            " -> ",
            (intent.target_name, "bold cyan"),
            "\n",
            (intent.label, "bold"),
            "   ",
            ("Threat ", "dim"),
            intent.threat_level,
            "   ",
            intent.obvious_effect,
        )
        if order_text.plain:
            text.append("\n")
            text.append(order_text)
        return Panel(text, title=view.encounter_name, border_style="bright_yellow")
    actor = view.current_actor
    if actor is None:
        text = Text("The encounter is resolving.", style="yellow")
    else:
        text = Text.assemble(
            ("Round ", "dim"),
            str(view.round_number),
            "   ",
            ("Cohesion ", "dim"),
            view.cohesion,
            "   ",
            ("Acting ", "dim"),
            (actor.name, "bold cyan"),
            "   ",
            ("Effort ", "dim"),
            f"{actor.effort}/{actor.max_effort}",
        )
    if order_text.plain:
        text.append("\n")
        text.append(order_text)
    return Panel(text, title=view.encounter_name, border_style="red")


def _turn_order_text(view: CombatView) -> Text:
    if not view.turn_order:
        return Text("")
    text = Text("Turns  ", style="dim")
    for index, entry in enumerate(view.turn_order[:8]):
        if index:
            text.append(" > ", style="dim")
        style = "cyan" if entry.team == "hero" else "red"
        if entry.active:
            style = "bold black on cyan" if entry.team == "hero" else "bold white on red"
        elif entry.acted:
            style = "dim"
        if "dead" in entry.statuses:
            style = "dim strike"
        elif "downed" in entry.statuses:
            style = "bold white on dark_magenta"
        text.append(_short_order_name(entry.name, team=entry.team), style=style)
    if len(view.turn_order) > 8:
        text.append(" > ...", style="dim")
    return text


def _short_order_name(name: str, *, team: str = "") -> str:
    parts = name.replace("-", " ").split()
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:10]
    if team == "enemy" and any(part.lower() == "maw" for part in parts):
        return "Maw"
    if team == "enemy":
        return parts[-1][:10]
    if parts[0].lower() in {"the", "a", "an"}:
        return parts[1][:10]
    return parts[0][:10]


def _render_enemy_intent(view: CombatView) -> Panel:
    intent = view.pending_enemy_intent
    if intent is None:
        return Panel("No pending enemy intent.", title="Enemy Intent", border_style="dim")
    lines = [
        f"Enemy: {intent.enemy_name}",
        f"Action: {intent.label}",
        f"Target: {intent.target_name}",
        f"Threat: {intent.threat_level}",
        f"Effect: {intent.obvious_effect}",
    ]
    if intent.debug_hit_chance is not None:
        lines.append(f"Debug hit: {intent.debug_hit_chance}%")
    if intent.debug_damage_estimate is not None:
        damage = intent.debug_damage_label or str(intent.debug_damage_estimate)
        lines.append(f"Debug damage: {damage}")
    return Panel("\n".join(lines), title="Enemy Intent", border_style="bright_yellow")


def _render_reaction_options(view: CombatView) -> Table:
    table = Table(title="Class Reactions", show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Reaction", style="bold")
    table.add_column("Cost", justify="right")
    table.add_column("Detail", style="dim")
    for option in view.reaction_options:
        table.add_row(
            option.action.number,
            option.action.label,
            str(option.cost),
            option.summary,
        )
    return table


def _render_combat_commands(view: CombatView) -> Table:
    table = Table(title="Commands", show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Command", style="bold")
    table.add_column("State")
    table.add_column("Detail", style="dim")
    for action in view.commands:
        label = f"{action.label} [default]" if action.default else action.label
        table.add_row(
            action.number,
            label,
            "available" if action.enabled else "disabled",
            action.description,
        )
    return table


def _render_combat_formation(
    title: str,
    combatants: tuple[CombatActorView, ...],
) -> Table:
    show_art = any(combatant.art_lines for combatant in combatants)
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Slot", style="cyan", no_wrap=True)
    if show_art:
        table.add_column("Art", style="dim", no_wrap=True)
    table.add_column("Actor", style="bold")
    table.add_column("HP", justify="right")
    table.add_column("Effort", justify="right")
    table.add_column("Morale")
    table.add_column("Strain")
    table.add_column("Tags")
    table.add_column("Traits")
    table.add_column("Status")
    for combatant in combatants:
        statuses = ", ".join(combatant.statuses)
        tags = ", ".join(combatant.tags) or "-"
        traits = _actor_trait_summary(combatant)
        name = f"> {combatant.name}" if combatant.acting else combatant.name
        row = [
            combatant.slot,
        ]
        if show_art:
            row.append(_compact_art_block(combatant.art_lines, max_lines=3, max_width=9))
        row.extend(
            [
                name,
                f"{combatant.hp}/{combatant.max_hp}",
                f"{combatant.effort}/{combatant.max_effort}",
                combatant.morale,
                combatant.strain,
                tags,
                traits,
                statuses,
            ]
        )
        table.add_row(*row)
    return table


def _compact_art_block(
    art_lines: tuple[str, ...],
    *,
    max_lines: int,
    max_width: int,
) -> str:
    if not art_lines:
        return ""
    lines = [line[:max_width].rstrip() for line in art_lines[:max_lines]]
    return "\n".join(lines)


def _render_skill_options(view: CombatView) -> Table:
    table = Table(title="Skills", show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Skill", style="bold")
    table.add_column("Effort", justify="right")
    table.add_column("Type")
    table.add_column("Row")
    table.add_column("Effect", justify="right")
    table.add_column("Targets", justify="right")
    for option in view.skills:
        label = f"{option.name} [default]" if option.action.default else option.name
        if not option.action.enabled and option.unavailable_reason:
            label = f"{label} [dim]({option.unavailable_reason})[/dim]"
        table.add_row(
            option.action.number,
            label,
            str(option.effort_cost),
            option.attack_type,
            option.usable_from_label,
            option.damage_label,
            str(option.target_count),
        )
    return table


def _render_target_options(view: CombatView) -> Table | Panel:
    if view.selected_skill_id is None:
        return Panel("Choose a skill to list legal targets.", title="Targets", border_style="dim")
    table = Table(title="Targets", show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Slot", style="cyan", no_wrap=True)
    table.add_column("Target", style="bold")
    table.add_column("HP", justify="right")
    table.add_column("Hit", justify="right")
    table.add_column("Effect", justify="right")
    table.add_column("Why", style="dim")
    table.add_column("Status")
    for target in view.targets:
        label = f"{target.name} [default]" if target.action.default else target.name
        table.add_row(
            target.action.number,
            target.slot,
            label,
            f"{target.hp}/{target.max_hp}",
            f"{target.hit_chance}%",
            target.damage_label,
            target.legality_reason,
            ", ".join(target.statuses),
        )
    return table


def _render_move_options(view: CombatView) -> Table | Panel:
    if not view.moves:
        return Panel("No adjacent movement is available.", title="Moves", border_style="dim")
    table = Table(title="Moves", show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Move", style="bold")
    table.add_column("From", style="cyan", no_wrap=True)
    table.add_column("To", style="cyan", no_wrap=True)
    table.add_column("Detail", style="dim")
    for move in view.moves:
        table.add_row(
            move.action.number,
            move.action.label,
            move.from_slot,
            move.to_slot,
            move.description,
        )
    return table


def _render_hero_entries(
    title: str,
    heroes: tuple[HeroListEntry, ...],
) -> Table | Panel:
    if not heroes:
        return Panel("None.", title=title, border_style="dim")
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Slot", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Class")
    table.add_column("HP", justify="right")
    table.add_column("Effort", justify="right")
    table.add_column("Morale")
    table.add_column("Strain")
    table.add_column("Mortal Wounds", justify="right")
    table.add_column("Status")
    table.add_column("Gear")
    table.add_column("Traits")
    table.add_column("Memory", no_wrap=True)
    for hero in heroes:
        table.add_row(
            hero.slot,
            hero.name,
            hero.class_id,
            f"{hero.hp}/{hero.max_hp}",
            f"{hero.effort}/{hero.max_effort}",
            hero.morale,
            hero.strain,
            mortal_wound_count(hero.mortal_wounds),
            ", ".join(hero.statuses),
            _hero_gear_summary(hero),
            _hero_trait_summary(hero),
            hero.latest_memory or f"{hero.memory_count} records",
        )
    return table


def _hero_gear_summary(hero: HeroListEntry) -> str:
    if not hero.equipped_gear:
        return "none"
    if hero.stat_bonus:
        return f"{hero.equipped_gear} ({hero.stat_bonus})"
    return hero.equipped_gear


def _actor_trait_summary(actor: CombatActorView) -> str:
    pieces: list[str] = []
    if actor.personal_quirk:
        pieces.append(actor.personal_quirk)
    if actor.quirks:
        pieces.append("Q: " + ", ".join(actor.quirks))
    if actor.strain_marks:
        pieces.append("Marks: " + ", ".join(actor.strain_marks))
    return "; ".join(pieces) or "-"


def _hero_trait_summary(hero: HeroListEntry) -> str:
    pieces: list[str] = []
    if hero.personal_quirk:
        pieces.append(hero.personal_quirk)
    if hero.quirks:
        pieces.append("Q: " + ", ".join(hero.quirks))
    if hero.strain_marks:
        pieces.append("Marks: " + ", ".join(hero.strain_marks))
    return "; ".join(pieces) or "-"


def _raw_hero_trait_summary(hero: HeroState) -> str:
    pieces: list[str] = []
    if hero.personal_quirk:
        pieces.append(_trait_label(hero.personal_quirk))
    if hero.quirks:
        pieces.append("Q: " + ", ".join(_trait_label(quirk) for quirk in hero.quirks))
    if hero.strain_marks:
        marks = (
            _trait_label(mark.value)
            for mark in sorted(hero.strain_marks, key=lambda item: item.value)
        )
        pieces.append(
            "Marks: "
            + ", ".join(marks)
        )
    return "; ".join(pieces) or "-"


def _trait_label(trait_id: str) -> str:
    return trait_id.replace("_", " ").title()


def _display_value(value: Any) -> str:
    if isinstance(value, list | tuple | set):
        return ", ".join(str(item) for item in value) or "none"
    return str(value)


def _lines_panel(title: str, values: tuple[str, ...]) -> Panel:
    text = "\n".join(f"- {value}" for value in values) if values else "none"
    return Panel(text, title=title, border_style="dim")


def _command_state_label(action: ScreenAction) -> str:
    if not action.enabled:
        return "locked"
    if action.default:
        return "default"
    return "ready"


def _contract_summary_line(entry: Any) -> str:
    line = (
        f"{entry.state.title()}: {entry.name} "
        f"(D{entry.difficulty}, {_contract_reward_summary(entry)})"
    )
    if entry.unavailable_reason:
        line += f" - {entry.unavailable_reason}"
    return line


def _contract_reward_summary(entry: Any) -> str:
    pieces: list[str] = []
    if getattr(entry, "reward_reputation", 0):
        pieces.append(f"+{entry.reward_reputation} rep")
    if getattr(entry, "coin_reward", 0):
        pieces.append(f"+{entry.coin_reward} Coin")
    return ", ".join(pieces) or "no payout"


def _upgrade_summary_line(entry: Any) -> str:
    line = f"{entry.state.title()}: {entry.name} (cost {entry.cost})"
    if entry.effect_summary:
        line += f" - {entry.effect_summary}"
    if entry.unavailable_reason and entry.state != "installed":
        line += f" - {entry.unavailable_reason}"
    return line


def _objective_lines(objective: Any) -> tuple[str, ...]:
    lines = [
        f"{objective.title} [{objective.status}]",
        objective.summary,
    ]
    if getattr(objective, "progress", ""):
        lines.append(f"Progress: {objective.progress}")
    lines.extend(
        [
            f"Next: {objective.next_step}",
            f"Chapter: {objective.chapter_status}",
        ]
    )
    lines.extend(
        f"{step.state.title()}: {step.name}"
        for step in objective.steps
    )
    return tuple(line for line in lines if line)


def _report_brief_lines(view: ExpeditionReportView) -> tuple[str, ...]:
    lines = [
        f"Outcome: {view.outcome.replace('_', ' ').title()}.",
        (
            f"Reputation: {view.reputation_start}->{view.reputation_end} "
            f"({_signed(view.reputation_delta)})."
        ),
        (
            f"Coin: {view.coin_start}->{view.coin_end} "
            f"({_signed(view.coin_delta)})."
        ),
    ]
    if view.wounded_count or view.downed_count or view.deceased_count:
        lines.append(
            "Condition: "
            f"{view.wounded_count} wounded, "
            f"{view.downed_count} downed, "
            f"{view.deceased_count} memorialized."
        )
    if view.supplies_spent:
        spent = ", ".join(
            f"{supply_id} x{quantity}" for supply_id, quantity in view.supplies_spent
        )
        lines.append(f"Supplies spent: {spent}.")
    if view.loot:
        loot = ", ".join(f"{item_id} x{quantity}" for item_id, quantity in view.loot)
        lines.append(f"Loot secured: {loot}.")
    if getattr(view, "gear", ()):
        gear = ", ".join(f"{gear_id} x{quantity}" for gear_id, quantity in view.gear)
        lines.append(f"Gear secured: {gear}.")
    if view.breaches_discovered:
        breaches = ", ".join(view.breaches_discovered)
        lines.append(f"Breach discovered: {breaches}.")
    lines.extend(view.hero_outcomes[:2])
    lines.extend(view.notable_moments[:3])
    return tuple(lines)


def _delta_panel(title: str, values: tuple[tuple[str, int, int, int], ...]) -> Panel:
    if not values:
        return Panel("none", title=title, border_style="dim")
    text = "\n".join(
        f"- {item_id}: {start}->{end} ({_signed(delta)})"
        for item_id, start, end, delta in values
    )
    return Panel(text, title=title, border_style="dim")


def _signed(value: int) -> str:
    return f"+{value}" if value > 0 else str(value)


def _primary_hotkey(action: ScreenAction) -> str:
    for alias in action.aliases:
        if len(alias) == 1:
            return alias
    return ""


def _use_fixed_viewport(console: Console | None, enabled: bool) -> bool:
    return bool(enabled and console is not None and console.is_terminal)


def _render_fixed_viewport_frame(
    header: Any,
    body: Any,
    frame: ScreenFrame,
    *,
    console: Console | None,
) -> Group:
    if console is None:
        return Group(header, body)

    command_dock = frame.command_dock
    footer = render_footer(frame.hint) if command_dock is None and frame.hint else None
    bottom = command_dock or footer
    viewport_lines = max(1, console.size.height - PROMPT_RESERVED_LINES)
    header_lines = _rendered_line_count(header, console)
    bottom_lines = _rendered_line_count(bottom, console) if bottom is not None else 0
    remaining = viewport_lines - header_lines - bottom_lines

    if remaining < MIN_BODY_LINES:
        body = _terminal_too_small_panel(console.size.height)
        too_small_parts = [header, body]
        if bottom is not None:
            too_small_parts.append(bottom)
        return Group(*too_small_parts)

    log = frame.log
    if log is not None:
        log_budget = min(MAX_LOG_LINES, remaining - MIN_BODY_LINES)
        if log_budget >= MIN_LOG_LINES:
            log = _fit_renderable_to_lines(log, console, log_budget, title="Log")
            remaining -= _rendered_line_count(log, console)
        else:
            log = None

    if remaining < MIN_BODY_LINES:
        log = None
        remaining = viewport_lines - header_lines - bottom_lines

    body = _fit_body_to_lines(frame.body, frame.title, console, remaining)
    parts: list[Any] = [header, body]
    if log is not None:
        parts.append(log)
    used_lines = sum(_rendered_line_count(part, console) for part in parts)
    used_lines += bottom_lines
    spacer_lines = max(0, viewport_lines - used_lines)
    parts.extend(Text("") for _ in range(spacer_lines))
    if bottom is not None:
        parts.append(bottom)
    return Group(*parts)


def _terminal_too_small_panel(height: int) -> Panel:
    return Panel(
        f"Terminal height is {height} lines. Enlarge the window to at least 24 lines.",
        title="Terminal Too Small",
        border_style="yellow",
    )


def _fit_renderable_to_lines(
    renderable: Any,
    console: Console,
    max_lines: int,
    *,
    title: str,
) -> Any:
    if max_lines <= 0:
        return Text("")
    if _rendered_line_count(renderable, console) <= max_lines:
        return renderable
    if max_lines <= 2:
        return Text("...")
    inner_lines = max(1, max_lines - 2)
    return Panel(
        _capture_renderable_text(renderable, console, inner_lines),
        title=title,
        border_style="yellow",
    )


def _fit_body_to_lines(
    body: Any,
    title: str,
    console: Console,
    max_lines: int,
) -> Any:
    body_panel = Panel(body, title=title, border_style="cyan")
    if _rendered_line_count(body_panel, console) <= max_lines:
        return body_panel
    if max_lines <= 2:
        return Text("...")
    inner_lines = max(1, max_lines - 2)
    return Panel(
        _capture_renderable_text(body, console, inner_lines, width_padding=4),
        title=title,
        border_style="yellow",
    )


def _capture_renderable_text(
    renderable: Any,
    console: Console,
    max_lines: int,
    *,
    width_padding: int = 0,
) -> Text:
    lines = _plain_rendered_lines(renderable, console, width_padding=width_padding)
    if len(lines) > max_lines:
        if max_lines <= 1:
            lines = ["..."]
        else:
            lines = lines[: max_lines - 1] + ["..."]
    return Text("\n".join(lines))


def _spacer_line_count(
    parts: tuple[Any, ...],
    *,
    console: Console | None,
    enabled: bool,
) -> int:
    if not enabled or console is None or not console.is_terminal:
        return 0
    available_height = console.size.height - 1
    if available_height <= 0:
        return 0
    used_lines = sum(_rendered_line_count(part, console) for part in parts)
    return max(0, available_height - used_lines)


def _rendered_line_count(renderable: Any, console: Console) -> int:
    if renderable is None:
        return 0
    with console.capture() as capture:
        console.print(renderable)
    output = capture.get()
    if not output:
        return 0
    return len(output.splitlines())


def _plain_rendered_lines(
    renderable: Any,
    console: Console,
    *,
    width_padding: int = 0,
) -> list[str]:
    output = StringIO()
    plain_console = Console(
        file=output,
        force_terminal=False,
        no_color=True,
        width=max(20, console.size.width - width_padding),
    )
    plain_console.print(renderable)
    return output.getvalue().splitlines()
