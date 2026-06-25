"""TUI screen rendering for dungeon."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from game.app.actions import (
    room_action_state_reason,
)
from game.app.commands import (
    StartExpedition,
    ViewDungeon,
    ViewExpeditionReport,
)
from game.app.views import (
    DungeonView,
    ExpeditionReportView,
    ScreenAction,
    ScreenActionKind,
    ScreenActionRisk,
)
from game.core.events import (
    DamageEvent,
    DeathEvent,
    DownedEvent,
    EnemyIntentEvent,
    ExpeditionEvent,
    GameEvent,
    HealingEvent,
    MissEvent,
    MoveEvent,
    ReactionSkippedEvent,
    ReactionUsedEvent,
    SkillUsedEvent,
    StatusChangedEvent,
    TurnDelayedEvent,
    TurnPassedEvent,
)
from game.core.hci import HciResultAnalysis, build_event_beats
from game.ui.tui_constants import (
    _route_summary_line,
    _route_warning_line,
)
from game.ui.tui_render.protocol import TuiRenderHost
from game.ui.tui_widgets import (
    DungeonMapPanel,
    DungeonRoomPanel,
    ExpeditionProgressStrip,
    ExpeditionReportPanel,
)


@dataclass
class DungeonRender:
    app: TuiRenderHost

    def show_expedition(self, message: str = "") -> None:
        has_company = self.app.controller.company is not None
        if self.app._breach_pending():
            body = "The opening breach is pending. Resolve Return or Descend before beginning."
        elif (
            has_company
            and self.app.controller.company is not None
            and self.app.controller.company.active_expedition
        ):
            body = "An opening expedition is active inside the Shallow Cave dungeon."
        else:
            body = "Begin the opening expedition. Travel is paced; dungeon rooms are interactive."
        actions = (
            ScreenAction(
                "1",
                "Begin / Resume Opening Expedition",
                "begin",
                ("x",),
                enabled=has_company,
                default=has_company,
            ),
            ScreenAction("2", "Back", "back", ("b",), default=not has_company),
        )
        self.app._show_screen("expedition", "Expedition", body, actions, message=message)

    def _begin_expedition(
        self,
        *,
        use_known_route: bool = True,
        skip_known_route_playback: bool = False,
        direct_to_dungeon: bool = False,
    ) -> None:
        if self.app._breach_pending():
            self.show_breach()
            return
        result = self.app.controller.handle(
            StartExpedition(
                stop_at_breach=True,
                manual_combat=True,
                interactive_dungeon=True,
                use_known_route=use_known_route,
                skip_known_route_playback=skip_known_route_playback,
            )
        )
        if not result.success:
            self.show_expedition(result.error or "The expedition cannot begin.")
            return
        self.app._record_events(result.events)
        if direct_to_dungeon:
            if (
                self.app.controller.company is not None
                and self.app.controller.company.active_expedition is not None
            ):
                self.show_dungeon(hci=result.hci)
            return
        if not result.events and self.app.controller.company is not None:
            if self.app.controller.company.active_expedition is not None:
                self.show_dungeon()
                return
        self._start_playback(result.events, result.hci)

    def _start_playback(
        self,
        events: list[GameEvent],
        hci: HciResultAnalysis | None = None,
    ) -> None:
        self.app.playback_beats = build_event_beats(self._meaningful_playback_events(events))
        self.app.playback_index = 0
        self.app.playback_hci = hci
        if not self.app.playback_beats:
            if (
                self.app.controller.company is not None
                and self.app.controller.company.active_expedition
            ):
                self.show_dungeon()
                return
            self._finish_playback("Nothing happens.")
            return
        self.show_playback()

    def _meaningful_playback_events(self, events: list[GameEvent]) -> list[GameEvent]:
        hidden_event_ids = self._opening_enemy_response_playback_event_ids(events)
        return [
            event
            for event in events
            if id(event) not in hidden_event_ids
            if not isinstance(event, ExpeditionEvent) or event.first_visit or event.major_beat
        ]

    def _opening_enemy_response_playback_event_ids(
        self,
        events: list[GameEvent],
    ) -> set[int]:
        if self.app.controller.manual_combat is None:
            return set()
        _hero_events, enemy_events = self.app._split_turn_events(events)
        enemy_beats = self.app._enemy_response_beats(enemy_events)
        hidden_types = (
            EnemyIntentEvent,
            SkillUsedEvent,
            MoveEvent,
            TurnDelayedEvent,
            TurnPassedEvent,
            DamageEvent,
            HealingEvent,
            MissEvent,
            ReactionUsedEvent,
            ReactionSkippedEvent,
            DownedEvent,
            DeathEvent,
            StatusChangedEvent,
        )
        return {
            id(event) for beat in enemy_beats for event in beat if isinstance(event, hidden_types)
        }

    def show_playback(self, message: str = "") -> None:
        if self.app.playback_index >= len(self.app.playback_beats):
            self._finish_playback()
            return
        beat = self.app.playback_beats[self.app.playback_index]
        actions = (ScreenAction("1", "Continue", "continue", ("c",), default=True),)
        progress = ExpeditionProgressStrip.render_text(
            self.app.playback_beats, self.app.playback_index
        )
        self.app._show_screen(
            "playback",
            "Expedition Playback",
            f"{progress}\n\nCurrent Beat\n{self.app._beat_text(beat)}",
            actions,
            message=message,
            log=self._playback_log_text(beat.events),
        )

    def _finish_playback(self, message: str = "") -> None:
        if self.app.controller.manual_combat is not None:
            if self.app._show_opening_enemy_response_if_needed():
                return
            self.app._show_combat_command(message)
            return
        if self.app._breach_pending():
            self.show_breach(message)
            return
        company = self.app.controller.company
        if company is not None and company.active_expedition is not None:
            self.show_dungeon(message)
            return
        if company is not None and company.last_expedition_report is not None:
            self.app._show_current_place(message or "Company record filed.")
            return
        if self.app.controller.return_to_regional_place:
            self.app.controller.return_to_regional_place = False
            self.app._show_regional_place(message)
            return
        self.show_expedition(message or "Expedition section complete.")

    def show_dungeon(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        result = self.app.controller.handle(ViewDungeon())
        if not result.success:
            self.show_expedition(result.error or "No active dungeon expedition.")
            return
        view = cast(DungeonView, result.value)
        self.app.current_dungeon_view = view
        title = "Wilderness" if view.current_map_id == "old_road_wilderness" else "Dungeon"
        actions = self._dungeon_navigation_actions(view)
        self.app._show_screen(
            "dungeon",
            title,
            self._dungeon_place_text(view),
            actions,
            message=message,
            log=self._dungeon_log_text(view, hci, actions=actions),
        )

    def show_dungeon_map(self, message: str = "") -> None:
        result = self.app.controller.handle(ViewDungeon())
        if not result.success:
            self.show_expedition(result.error or "No active dungeon expedition.")
            return
        view = cast(DungeonView, result.value)
        self.app.current_dungeon_view = view
        navigation_actions = self._dungeon_navigation_actions(view)
        self.app._show_screen(
            "dungeon_map",
            "Map",
            DungeonMapPanel.render_text(view, actions=navigation_actions),
            (ScreenAction("1", "Back to Place", "back", ("b",), default=True),),
            message=message,
            log=DungeonMapPanel.render_minimap_text(view, actions=navigation_actions),
        )

    def show_dungeon_interactions(self, message: str = "") -> None:
        result = self.app.controller.handle(ViewDungeon())
        if not result.success:
            self.show_expedition(result.error or "No active dungeon expedition.")
            return
        view = cast(DungeonView, result.value)
        self.app.current_dungeon_view = view
        commands = self._available_room_action_commands(view)
        if not commands:
            self.show_dungeon(message or "Nothing here needs handling.")
            return
        actions = tuple(
            self._renumbered_action(command, index, default=False)
            for index, command in enumerate(commands, start=1)
        )
        actions = (
            *actions,
            ScreenAction(str(len(actions) + 1), "Back to Room", "back", ("b",)),
        )
        self.app._show_screen(
            "dungeon_interact",
            "Interact",
            self._dungeon_place_text(view),
            actions,
            message=message,
            log=DungeonMapPanel.render_minimap_text(view, actions=actions),
        )

    def _dungeon_navigation_actions(
        self,
        view: DungeonView,
    ) -> tuple[ScreenAction, ...]:
        action_by_value = {action.value: action for action in view.actions}
        action_list: list[ScreenAction] = []
        for exit_view in view.exits:
            source = action_by_value.get(exit_view.node_id)
            if source is None:
                continue
            action_list.append(
                self._renumbered_action(
                    source,
                    len(action_list) + 1,
                    default=False,
                )
            )
        room_action_commands = self._available_room_action_commands(view)
        blocked_room_actions = self._blocked_room_actions(view)
        if room_action_commands or blocked_room_actions:
            action_list.append(
                ScreenAction(
                    str(len(action_list) + 1),
                    "Interact",
                    "interact",
                    ("i",),
                    enabled=bool(room_action_commands),
                    default=bool(room_action_commands)
                    and not any(action.enabled for action in action_list),
                    description="Handle something in this room."
                    if room_action_commands
                    else "Room actions are blocked.",
                    kind=ScreenActionKind.DUNGEON,
                    risk=ScreenActionRisk.COSTLY
                    if any(command.cost for command in room_action_commands)
                    else ScreenActionRisk.LOW,
                    unavailable_reason=self._blocked_room_actions_reason(view),
                    preview="Open room-specific actions and blocked requirements.",
                    result_hint="Room actions can reveal routes, spend supplies, or claim loot.",
                )
            )
        primary_indexes = [
            index
            for index, action in enumerate(action_list)
            if action.enabled and action.value != "interact"
        ]
        if len(primary_indexes) == 1:
            index = primary_indexes[0]
            action_list[index] = self._renumbered_action(
                action_list[index],
                int(action_list[index].number),
                default=True,
            )
        for value in (
            "enter_generated_maze",
            "retrace_generated_maze",
            "withdraw_generated_maze",
            "retreat_generated_maze",
        ):
            generated_action = action_by_value.get(value)
            if generated_action is not None:
                action_list.append(self._renumbered_action(generated_action, len(action_list) + 1))
        return_action = action_by_value.get("return")
        if return_action is not None and return_action.enabled:
            action_list.append(self._renumbered_action(return_action, len(action_list) + 1))
        return tuple(action_list)

    def _available_room_action_commands(self, view: DungeonView) -> tuple[ScreenAction, ...]:
        command_by_action_id = {
            action.value.removeprefix("action:"): action
            for action in view.actions
            if action.value.startswith("action:")
        }
        return tuple(
            command
            for room_action in view.room_actions
            if (command := command_by_action_id.get(room_action.action_id)) is not None
            and command.enabled
        )

    def _blocked_room_actions(self, view: DungeonView) -> tuple[Any, ...]:
        return tuple(
            room_action
            for room_action in view.room_actions
            if room_action.state not in {"available", "completed"}
        )

    def _blocked_room_actions_reason(self, view: DungeonView) -> str:
        blocked = self._blocked_room_actions(view)
        if not blocked:
            return ""
        return "; ".join(
            f"{room_action.label}: {self._room_action_state_reason(room_action)}"
            for room_action in blocked
        )

    def _renumbered_action(
        self,
        action: ScreenAction,
        number: int,
        *,
        label: str | None = None,
        default: bool | None = None,
    ) -> ScreenAction:
        return ScreenAction(
            str(number),
            label or action.label,
            action.value,
            action.aliases,
            enabled=action.enabled,
            default=action.default if default is None else default,
            description=action.description,
            kind=action.kind,
            risk=action.risk,
            cost=action.cost,
            unavailable_reason=action.unavailable_reason,
            preview=action.preview,
            result_hint=action.result_hint,
            confirm=action.confirm,
            route_warning=action.route_warning,
        )

    def show_expedition_report(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        result = self.app.controller.handle(ViewExpeditionReport())
        if not result.success:
            self.show_expedition(result.error or "No expedition report is available.")
            return
        view = cast(ExpeditionReportView, result.value)
        self.show_report_view(view, message=message, hci=hci)

    def show_report_view(
        self,
        view: ExpeditionReportView,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        self.app.current_report_view = view
        self.app._show_screen(
            "expedition_report",
            "Filed Company Record",
            ExpeditionReportPanel.render_text(view),
            view.actions,
            message=message,
            log=self.app._result_log_text(self.app.recent_events, hci)
            if hci is not None
            else self.app._events_text(self.app.recent_events),
        )

    def show_breach(self, message: str = "") -> None:
        company = self.app.controller.company
        if company is None:
            self.show_expedition("Start or load a company first.")
            return
        body = (
            "The Shallow Cave Breach is open.\n\n"
            f"Company: {company.name}\n"
            f"Reputation: {company.reputation}\n"
            f"Coin: {company.coin}\n"
            f"Location: {company.town_state.get('location', 'Haven Town')}\n"
            f"Active party: {len([slot for slot in company.active_party_slots.values() if slot])}"
        )
        actions = (
            ScreenAction(
                "1",
                "Return to Haven",
                "return_to_haven",
                ("r",),
                default=True,
                kind=ScreenActionKind.TRAVEL,
                risk=ScreenActionRisk.LOW,
                preview="Report the breach and bring the company back to Haven.",
                result_hint="Ends the expedition with proof and survivors.",
            ),
            ScreenAction(
                "2",
                "Descend to Maze Depth 1",
                "descend_maze_depth_1",
                ("d",),
                kind=ScreenActionKind.TRAVEL,
                risk=ScreenActionRisk.RISKY,
                preview="Push beyond the cave into the first impossible rooms.",
                result_hint="Requires confirmation before the company descends.",
                confirm="Descend into Maze Depth 1?",
            ),
        )
        self.app._show_screen(
            "breach",
            "Breach",
            body,
            actions,
            message=message,
            log=self.app._events_text(self.app.recent_events),
        )

    def _playback_log_text(self, events: Sequence[GameEvent]) -> str:
        pieces = []
        if self.app.playback_hci is not None:
            pieces.append(self.app._result_log_text(events, self.app.playback_hci))
        map_text = self._active_dungeon_minimap_text()
        if map_text:
            pieces.append(map_text)
        if not pieces:
            pieces.append(self.app._events_text(self.app.recent_events))
        return "\n\n".join(piece for piece in pieces if piece)

    def _dungeon_log_text(
        self,
        view: DungeonView,
        hci: HciResultAnalysis | None,
        *,
        actions: Sequence[ScreenAction] | None = None,
    ) -> str:
        map_text = DungeonMapPanel.render_minimap_text(view, actions=actions)
        if hci is None:
            return map_text
        brief = self.app._result_log_text([], hci)
        return "\n\n".join(piece for piece in (brief, map_text) if piece)

    def _dungeon_place_text(self, view: DungeonView) -> str:
        if view.current_map_id == "old_road_wilderness":
            breadcrumb = "World > Old Road Wilds"
        elif view.current_map_id == "shallow_cave":
            breadcrumb = "World > Shallow Cave > Dungeon"
        elif view.current_map_id == "shallow_cave_breach":
            breadcrumb = "World > Shallow Cave Breach"
        elif view.current_map_id == "maze_depth_1":
            breadcrumb = "World > Pandora's Maze Depth 1"
        else:
            breadcrumb = f"World > {view.current_map_id.replace('_', ' ').title()}"
        body = f"{breadcrumb}\n\n{DungeonRoomPanel.render_text(view)}"
        party_watch = self.app._party_watch_text()
        if party_watch:
            body = f"{body}\n\n{party_watch}"
        return body

    def _dungeon_detail_text(self, action: ScreenAction) -> str:
        view = self.app.current_dungeon_view
        if view is None:
            return action.description or action.label
        if action.value == "interact":
            return self._dungeon_interact_detail_text(view)
        exit_node = next((node for node in view.exits if node.node_id == action.value), None)
        if exit_node is not None:
            return self._dungeon_route_detail_text(action, exit_node)
        if action.value.startswith("action:"):
            action_id = action.value.removeprefix("action:")
            room_action = next(
                (candidate for candidate in view.room_actions if candidate.action_id == action_id),
                None,
            )
            if room_action is not None:
                return self._dungeon_room_action_detail_text(action, room_action)
        return self.app._generic_action_detail(action)

    def _dungeon_interact_detail_text(self, view: DungeonView) -> str:
        lines = ["Interact"]
        available = [
            room_action for room_action in view.room_actions if room_action.state == "available"
        ]
        blocked = self._blocked_room_actions(view)
        entries = [room_action.label for room_action in available]
        entries.extend(
            f"{room_action.label} - {self._room_action_state_reason(room_action)}"
            for room_action in blocked
        )
        if entries:
            lines.extend(("", *entries))
        else:
            lines.extend(("", "Nothing here needs handling."))
        return "\n".join(lines)

    def _dungeon_route_detail_text(self, action: ScreenAction, exit_node: Any) -> str:
        view = self.app.current_dungeon_view
        lines = [
            exit_node.name,
            "",
            _route_summary_line(exit_node),
        ]
        if view is not None and exit_node.map_id != view.current_map_id:
            lines.append(f"Area: {str(exit_node.map_id).replace('_', ' ').title()}")
        if not action.enabled:
            lines.extend(("", action.unavailable_reason or "Clear this room before moving."))
            if "Clear this room" not in (action.unavailable_reason or ""):
                lines.append("Clear this room before moving.")
        else:
            detail_lines = [line for line in (_route_warning_line(exit_node),) if line]
            if detail_lines:
                lines.extend(("", *detail_lines))
        if view is not None and view.current_room.safe_return:
            lines.append("Safe return is available here.")
        elif exit_node.safe_return:
            lines.append("Destination has safe return.")
        return "\n".join(lines)

    def _dungeon_room_action_detail_text(self, action: ScreenAction, room_action: Any) -> str:
        lines = [room_action.label]
        if room_action.description:
            lines.extend(("", room_action.description))
        reason = self._room_action_state_reason(room_action)
        if reason and room_action.state != "available":
            lines.extend(("", self._sentence_case(reason)))
        if room_action.cost:
            lines.append(f"Cost: {self._quantity_list(room_action.cost)}")
        if room_action.reward:
            lines.append("Reward: " + ", ".join(room_action.reward))
        if action.result_hint:
            lines.append("Expected: " + action.result_hint)
        return "\n".join(lines)

    def _active_dungeon_minimap_text(self) -> str:
        company = self.app.controller.company
        if company is None or company.active_expedition is None:
            return ""
        result = self.app.controller.handle(ViewDungeon())
        if not result.success:
            return ""
        view = cast(DungeonView, result.value)
        self.app.current_dungeon_view = view
        return DungeonMapPanel.render_minimap_text(view)

    def _focused_dungeon_node_id(self) -> str:
        action = self.app.focused_action
        view = self.app.current_dungeon_view
        if action is None or view is None:
            return ""
        node_ids = {node.node_id for node in view.map_nodes}
        return action.value if action.value in node_ids else ""

    def _post_combat_continue_label(self) -> str:
        if self.app.controller.manual_combat is not None:
            return "Next Command"
        if self.app._breach_pending():
            return "Resolve Breach"
        company = self.app.controller.company
        if company is not None and company.active_expedition is not None:
            return "Return to Dungeon"
        if company is not None and company.last_expedition_report is not None:
            return "Return to Place"
        if self.app.controller.return_to_regional_place:
            return "Return to Place"
        return "Continue"

    def _post_combat_continue_description(self) -> str:
        if self.app.controller.manual_combat is not None:
            return "Return to the next hero command."
        if self.app._breach_pending():
            return "Open the breach decision."
        company = self.app.controller.company
        if company is not None and company.active_expedition is not None:
            return "Commit combat results and return to room actions."
        if company is not None and company.last_expedition_report is not None:
            return "Return to the current place with the company record filed."
        return "Continue."

    def _expedition_progress_strip(self) -> str:
        if not self.app.playback_beats:
            return "Expedition Progress\n(no route)"
        lines = ["Expedition Progress"]
        for index, beat in enumerate(self.app.playback_beats):
            if index < self.app.playback_index:
                marker = "[x]"
            elif index == self.app.playback_index:
                marker = "[>]"
            else:
                marker = "[ ]"
            kind = "combat" if beat.combat else beat.title.lower()
            lines.append(f"{marker} {index + 1:02}. {beat.title} ({kind})")
        return "\n".join(lines)

    def _room_action_notice(self, events: Sequence[GameEvent]) -> str:
        return events[-1].message if events else "Room action resolved."

    def _room_action_state_reason(self, room_action: Any) -> str:
        return room_action_state_reason(room_action)

    def _quantity_list(self, values: Sequence[tuple[str, int]]) -> str:
        return ", ".join(self._quantity_label(item_id, quantity) for item_id, quantity in values)

    def _quantity_label(self, item_id: str, quantity: int) -> str:
        label = item_id.replace("_", " ")
        if quantity == 1:
            return label
        return f"{quantity} {label}"

    def _sentence_case(self, text: str) -> str:
        if not text:
            return text
        return f"{text[0].upper()}{text[1:]}."

    def _route_dock_help_text(self, action: ScreenAction) -> str:
        if action.value == "map":
            return "Open the full map.\nNo route is taken."
        if action.value == "interact":
            return "Open room actions.\nBlocked actions explain their requirements."
        if action.value == "back":
            return "Return to the current place."
        if action.value == "return":
            return "Follow the road back to Haven.\nArrives with the latest company record filed."
        if action.value == "enter_generated_maze":
            return (
                "Enter a breach route that extends as you travel.\n"
                "The Maze does not end until the company withdraws."
            )
        if action.value == "retrace_generated_maze":
            return "Follow marks back to the Maze threshold.\nKeeps the route active."
        if action.value == "withdraw_generated_maze":
            return (
                "Company withdraws to the Shallow Cave breach.\n"
                "This run collapses. The breach remains exploitable."
            )
        if action.value == "retreat_generated_maze":
            return "Follow marks back to the Maze threshold.\nKeeps the route active."
        if action.value.startswith("action:"):
            return action.result_hint or "Resolve this room action."
        if str(action.kind) == "travel":
            return "Take the focused route.\nUse Up/Down to compare exits. M opens the map."
        return action.result_hint or action.preview or action.description
