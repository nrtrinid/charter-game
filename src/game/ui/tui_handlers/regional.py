"""Regional map and place action handlers for the TUI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from game.app.actions import (
    quantity_detail,
    room_action_preview,
    room_action_result_hint,
    room_action_state_reason,
)
from game.app.commands import (
    MarkRegionalRoute,
    MoveRegional,
    ReturnToHavenTown,
    TravelRegional,
    UseRegionalAction,
    ViewRegionalMap,
    VisitEastGate,
)
from game.app.views import (
    RegionalMapView,
    ScreenAction,
    ScreenActionKind,
    ScreenActionRisk,
    build_regional_render_view,
)
from game.core.events import ExpeditionEvent, GameEvent
from game.ui.tui_handlers.protocol import TuiHandlerHost


@dataclass
class RegionalHandlers:
    app: TuiHandlerHost

    def visit_east_gate_from_town(self) -> None:
        result = self.app.controller.handle(VisitEastGate())
        if not result.success:
            self.app._show_town(result.error or "Could not reach East Gate.")
            return
        self.app._record_events(result.events)
        self.app._show_regional_place(
            self.app._result_log_text(result.events, result.hci) or "",
            result.hci,
            view=cast(RegionalMapView, result.value),
            return_to="town",
        )

    def handle_place_action(self, value: str) -> None:
        if value == "interact":
            self.show_interactions()
            return
        if value.startswith("action:"):
            self.resolve_action(value.removeprefix("action:"))
            return
        if value == "enter_haven":
            result = self.app.controller.handle(ReturnToHavenTown())
            if not result.success:
                self.app._show_regional_place(result.error or "Could not return to Haven.")
                return
            self.app._record_events(result.events)
            self.app._show_town(
                self.app._result_log_text(result.events, result.hci) or "",
                result.hci,
            )
            return
        if value == "enter_cave":
            self.app._begin_expedition(
                skip_known_route_playback=True,
                direct_to_dungeon=True,
            )
            return
        if value == "latest_record":
            self.app._show_expedition_report()
            return
        if value == "mark_route":
            result = self.app.controller.handle(MarkRegionalRoute())
            if not result.success:
                self.app._show_regional_place(result.error or "Could not mark the route.")
                return
            self.app._record_events(result.events)
            notice = self.app._result_log_text(result.events, result.hci) or "Route charted."
            self.app._show_regional_place(notice, result.hci, view=result.value)
            return
        if value in {"survey_route", "map"}:
            self.app._show_regional_map()
            return
        if value == "world_map":
            self.app._show_world_map(return_to="regional_place")
            return
        if value == "system":
            self.app._show_system()
            return
        if value in {"haven", "shallow_cave"}:
            self.handle_travel(value)
            return
        from game.expedition.travel import REGIONAL_OVERWORLD_NODE_IDS

        if value in REGIONAL_OVERWORLD_NODE_IDS:
            view = self.app.current_regional_view
            if (
                view is not None
                and view.anchor_kind == "east_gate"
                and not view.route_charted
                and value == "old_road"
            ):
                self.app._begin_expedition(use_known_route=False)
                return
            self.handle_walk(value)
            return
        self.app._show_regional_place("Choose a listed regional action.")

    def handle_map_action(self, value: str) -> None:
        if value == "back":
            self.fold_roadbook()
            return
        if value == "begin_opening_route":
            self.app._begin_expedition(use_known_route=False)
            return
        if value in {"haven", "shallow_cave"}:
            self.handle_travel(value)
            return
        from game.expedition.travel import REGIONAL_OVERWORLD_NODE_IDS

        if value in REGIONAL_OVERWORLD_NODE_IDS:
            self.handle_walk(value)
            return
        self.app._show_regional_map("Choose a listed regional action.")

    def handle_walk(self, node_id: str) -> None:
        on_map = self.app.screen_state == "regional_map"
        result = self.app.controller.handle(MoveRegional(node_id))
        if not result.success:
            if on_map:
                self.app._show_regional_map(result.error or "Walk failed.")
            else:
                self.app._show_regional_place(result.error or "Walk failed.")
            return
        self.app._record_events(result.events)
        if self.app.controller.manual_combat is not None:
            self.app._start_playback(result.events, result.hci)
            return
        travel_view = cast(RegionalMapView, result.value)
        message = f"Arrived at {travel_view.current_node_name}."
        if self.move_lands_on_place(travel_view, result.events):
            self.app._show_regional_place(message, result.hci, view=travel_view)
        elif on_map:
            self.app._show_regional_map(message, result.hci, view=travel_view)
        else:
            self.app._show_regional_place(message, result.hci, view=travel_view)

    def move_lands_on_place(
        self,
        view: RegionalMapView,
        events: Sequence[GameEvent],
    ) -> bool:
        from game.expedition.travel import (
            regional_move_lands_on_place,
            regional_overworld_nodes,
        )

        first_visit = any(
            isinstance(event, ExpeditionEvent) and event.first_visit for event in events
        )
        nodes = regional_overworld_nodes(self.app.controller.definitions)
        node = nodes[view.current_node_id]
        return regional_move_lands_on_place(node, first_visit=first_visit)

    def available_room_action_commands(
        self,
        view: RegionalMapView,
    ) -> tuple[ScreenAction, ...]:
        return tuple(
            ScreenAction(
                str(index),
                room_action.label,
                f"action:{room_action.action_id}",
                (room_action.action_id,),
                enabled=room_action.state == "available",
                description=room_action.description,
                kind=ScreenActionKind.DUNGEON,
                risk=ScreenActionRisk.COSTLY if room_action.cost else ScreenActionRisk.LOW,
                cost=quantity_detail(room_action.cost),
                unavailable_reason=room_action_state_reason(room_action),
                preview=room_action_preview(room_action),
                result_hint=room_action_result_hint(room_action),
            )
            for index, room_action in enumerate(
                (
                    room_action
                    for room_action in view.room_actions
                    if room_action.state == "available"
                ),
                start=1,
            )
        )

    def show_interactions(self, message: str = "") -> None:
        view = self.app.current_regional_view
        if view is None:
            result = self.app.controller.handle(ViewRegionalMap())
            if not result.success:
                self.app._show_regional_place(result.error or "Regional map unavailable.")
                return
            view = cast(RegionalMapView, result.value)
            self.app.current_regional_view = view
        commands = self.available_room_action_commands(view)
        if not commands:
            self.app._show_regional_place(message or "Nothing here needs handling.")
            return
        actions = tuple(
            self.app._renumbered_action(command, index, default=False)
            for index, command in enumerate(commands, start=1)
        )
        actions = (
            *actions,
            ScreenAction(str(len(actions) + 1), "Back to Place", "back", ("b",)),
        )
        render_view = build_regional_render_view(view)
        self.app._show_screen(
            "regional_interact",
            "Interact",
            self.app._regional_place_text(view),
            actions,
            message=message,
            log=self.app._regional_log_text(render_view, None, actions=actions),
        )

    def resolve_action(self, action_id: str) -> None:
        return_state = self.app.screen_state
        result = self.app.controller.handle(UseRegionalAction(action_id))
        if not result.success:
            if return_state == "regional_interact":
                self.show_interactions(result.error or "Room action failed.")
            else:
                self.app._show_regional_place(result.error or "Room action failed.")
            return
        self.app._record_events(result.events)
        notice = self.app._room_action_notice(result.events)
        travel_view = cast(RegionalMapView, result.value)
        if return_state == "regional_map":
            self.app._show_regional_map(notice, result.hci, view=travel_view)
        else:
            self.app._show_regional_place(notice, result.hci, view=travel_view)

    def handle_interaction_action(self, value: str) -> None:
        if value.startswith("action:"):
            self.resolve_action(value.removeprefix("action:"))
            return
        self.show_interactions("Choose a listed regional action.")

    def handle_travel(self, destination_id: str) -> None:
        result = self.app.controller.handle(TravelRegional(destination_id))
        if result.success:
            self.app._record_events(result.events)
            travel_view = cast(RegionalMapView, result.value)
            self.app._show_regional_place(
                result.events[0].message
                if result.events
                else f"Travelled to {self.app._place_name(destination_id)}.",
                result.hci,
                view=travel_view,
            )
        elif self.app.screen_state == "regional_map":
            self.app._show_regional_map(result.error or "Travel failed.")
        else:
            self.app._show_regional_place(result.error or "Travel failed.")

    def back_from_place(self) -> None:
        if self.app.pending_regional_return_state == "town":
            result = self.app.controller.handle(ReturnToHavenTown())
            if not result.success:
                self.app._show_regional_place(result.error or "Could not return to Haven.")
                return
            self.app._record_events(result.events)
            self.app._show_town(
                self.app._result_log_text(result.events, result.hci) or "",
                result.hci,
            )
            return
        self.app._show_current_place()

    def fold_roadbook(self) -> None:
        self.app._show_regional_place(
            "The roadbook is folded away. The gate returns: mud, lanterns, wagon ruts."
        )

    def back_from_world_map(self) -> None:
        if self.app.pending_world_map_return_state == "regional_place":
            self.app._show_regional_place()
            return
        self.app._show_current_place()
