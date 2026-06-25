"""TUI screen rendering for regional."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

from game.app.actions import (
    ActionProvider,
)
from game.app.commands import (
    ViewRegionalMap,
    ViewWorld,
)
from game.app.views import (
    ArrivalBriefView,
    DungeonView,
    RegionalMapView,
    ScreenAction,
    ScreenActionKind,
    WorldView,
    build_regional_render_view,
)
from game.core.events import (
    GameEvent,
)
from game.core.hci import HciResultAnalysis
from game.ui.tui_render.protocol import TuiRenderHost
from game.ui.tui_widgets import (
    DungeonMapPanel,
)


@dataclass
class RegionalRender:
    app: TuiRenderHost

    def show_regional_place(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
        *,
        view: RegionalMapView | None = None,
        return_to: str | None = None,
    ) -> None:
        if return_to is not None:
            self.app.pending_regional_return_state = return_to
        cached_arrival = (
            view.arrival_context
            if view is not None
            else (
                self.app.current_regional_view.arrival_context
                if self.app.current_regional_view is not None
                else None
            )
        )
        if view is None:
            result = self.app.controller.handle(ViewRegionalMap())
            if not result.success:
                self.app._show_main(result.error or "Regional map unavailable.")
                return
            view = cast(RegionalMapView, result.value)
            if cached_arrival is not None and view.arrival_context is None:
                view = replace(view, arrival_context=cached_arrival)
        self.app.current_regional_view = view
        render_view = build_regional_render_view(view)
        title = view.place_title or view.current_node_name
        actions = ActionProvider.regional_place_actions(view)
        self.app._show_screen(
            "regional_place",
            title,
            self._regional_place_text(view),
            actions,
            message=message,
            log=self._regional_log_text(render_view, hci, actions=actions),
        )

    def show_regional_map(
        self,
        message: str = "",
        hci: HciResultAnalysis | None = None,
        *,
        view: RegionalMapView | None = None,
    ) -> None:
        if view is None:
            if self.app.current_regional_view is not None:
                view = self.app.current_regional_view
            else:
                result = self.app.controller.handle(ViewRegionalMap())
                if not result.success:
                    self.app._show_main(result.error or "Regional map unavailable.")
                    return
                view = cast(RegionalMapView, result.value)
        self.app.current_regional_view = view
        render_view = build_regional_render_view(view)
        navigation_actions = self._regional_navigation_actions(view)
        map_actions = self._regional_map_display_actions(navigation_actions)
        self.app._show_screen(
            "regional_map",
            "Company Roadbook",
            self._regional_map_text(
                view,
                render_view,
                actions=navigation_actions,
                map_actions=map_actions,
            ),
            (
                *navigation_actions,
                ScreenAction(
                    str(len(navigation_actions) + 1),
                    "Fold Roadbook",
                    "back",
                    ("b",),
                    default=True,
                ),
            ),
            message=message
            or (
                "The company roadbook is unrolled. Cleared ground becomes lines, "
                "costs, and destinations."
            ),
            log=self._regional_log_text(render_view, hci, actions=map_actions),
        )

    def show_world_map(
        self,
        message: str = "",
        *,
        return_to: str = "current_place",
    ) -> None:
        self.app.pending_world_map_return_state = return_to
        result = self.app.controller.handle(ViewWorld())
        if not result.success:
            self.app._show_current_place(result.error or "Map is unavailable.")
            return
        view = cast(WorldView, result.value)
        self.app._show_screen(
            "world_map",
            "World Map",
            self._world_map_text(view),
            (
                ScreenAction(
                    "1",
                    "Back",
                    "back",
                    ("b",),
                    default=True,
                    kind=ScreenActionKind.NAVIGATE,
                ),
            ),
            message=message,
            log=self.app._events_text(self.app.recent_events),
        )

    def show_regional_interactions(self, message: str = "") -> None:
        self.app._regional_handlers.show_interactions(message)

    def show_arrival_brief(
        self,
        view: ArrivalBriefView,
        message: str = "",
        hci: HciResultAnalysis | None = None,
    ) -> None:
        if view.location_id == "haven":
            world_result = self.app.controller.handle(ViewRegionalMap())
            if not world_result.success:
                self.app._show_current_place(world_result.error or "Arrival view unavailable.")
                return
            regional_view = cast(RegionalMapView, world_result.value)
            self.show_regional_place(
                message,
                hci,
                view=replace(regional_view, arrival_context=view),
            )
            return
        world_result = self.app.controller.handle(ViewRegionalMap())
        if not world_result.success:
            self.app._show_current_place(world_result.error or "Arrival view unavailable.")
            return
        regional_view = cast(RegionalMapView, world_result.value)
        self.show_regional_place(
            message,
            hci,
            view=replace(regional_view, arrival_context=view),
        )

    def _regional_place_text(self, view: RegionalMapView) -> str:
        lines = [
            view.breadcrumb,
            "",
            view.place_text or view.location_description,
        ]
        if view.arrival_context is not None:
            lines.extend(["", self._arrival_brief_text(view.arrival_context)])
        lines.extend(
            (
                "",
                "Current Objective",
                f"- {view.objective.title}: {view.objective.next_step}",
            )
        )
        if view.objective.progress:
            lines.append(f"  Progress: {view.objective.progress}")
        if view.travel_flavor:
            lines.extend(("", view.travel_flavor))
        if view.route_charted:
            lines.extend(("", f"Roadbook route available. Cost: {view.travel_cost}."))
        elif view.anchor_kind == "east_gate" and not view.route_charted:
            lines.extend(
                (
                    "",
                    (
                        "The road to Shallow Cave is uncharted. Take the Old Road "
                        "to begin the opening expedition."
                    ),
                )
            )
        if view.active_contracts:
            lines.extend(("", "Active Contracts"))
            lines.extend(
                f"- {name}: {summary}" if summary else f"- {name}"
                for _contract_id, name, summary in view.active_contracts
            )
        party_watch = self.app._party_watch_text()
        if party_watch:
            lines.extend(("", party_watch))
        return "\n".join(lines)

    def _regional_log_text(
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

    def _regional_map_text(
        self,
        view: RegionalMapView,
        render_view: DungeonView,
        *,
        actions: Sequence[ScreenAction],
        map_actions: Sequence[ScreenAction],
    ) -> str:
        lines = [
            "Company Roadbook",
            "Old Road Wilderness",
            "",
            "Charted Route Survey",
            "known roads - cleared ground - fast travel",
            "",
            "ZOOMED OUT: choose a known destination. No new discoveries on charted travel.",
            "",
            "The company roadbook is unrolled across a crate.",
            "Cleared ground collapses into lines, costs, and destinations.",
        ]
        if view.travel_flavor:
            lines.extend(("", view.travel_flavor))
        lines.extend(
            (
                "",
                "Current Objective",
                f"- {view.objective.title}: {view.objective.next_step}",
            )
        )
        if view.objective.progress:
            lines.append(f"  Progress: {view.objective.progress}")
        lines.extend(("", "Route Commands"))
        if actions:
            for action in actions:
                destination = action.label.removeprefix("Take Charted Road to ")
                lines.append(f"{action.number} = {destination}")
        else:
            lines.append("- No charted road is available from here.")
        lines.extend(
            (
                "",
                DungeonMapPanel.render_text(
                    replace(render_view, exits=()),
                    actions=map_actions,
                    title="Roadbook Map",
                    survey_label="Charted Route Survey",
                    legend_line=(
                        "@ current  |  o visited  |  ? known  |  number = charted route command"
                    ),
                ),
            )
        )
        return "\n".join(lines)

    def _regional_navigation_actions(
        self,
        view: RegionalMapView,
    ) -> tuple[ScreenAction, ...]:
        travel_actions = ActionProvider.regional_map_travel_actions(view)
        return tuple(
            self.app._renumbered_action(action, index + 1, default=index == 0)
            for index, action in enumerate(travel_actions)
        )

    def _regional_map_display_actions(
        self,
        actions: Sequence[ScreenAction],
    ) -> tuple[ScreenAction, ...]:
        regional_view = self.app.current_regional_view
        node_by_location_id = {
            node.location_id: node.node_id
            for node in (regional_view.map_nodes if regional_view else ())
        }
        return tuple(
            replace(action, value=node_by_location_id.get(action.value, action.value))
            for action in actions
        )

    def _arrival_brief_text(self, view: ArrivalBriefView) -> str:
        lines = [
            view.title,
            "",
            " -- ".join(f"[{label}]" for label in view.path),
            "",
            f"From: {view.origin_name}",
            f"Arrived: {view.location_name}",
            "",
        ]
        lines.extend(view.flavor_lines)
        lines.extend(("", "What Changed"))
        lines.extend(f"- {line}" for line in view.what_changed)
        lines.extend(("", "Next"))
        lines.append(view.next_objective or "Choose the company's next route.")
        lines.extend(("", "Records"))
        lines.append(f"{view.record_label} is available in Haven Records.")
        return "\n".join(lines)

    def _world_map_text(self, view: WorldView) -> str:
        lines = [
            "World Map",
            view.breadcrumb,
            "",
            "Current Objective",
            f"- {view.objective.title}: {view.objective.next_step}",
        ]
        if view.objective.progress:
            lines.append(f"  Progress: {view.objective.progress}")
        lines.extend(("", "Known Places"))
        if view.known_locations:
            for location in view.known_locations:
                if not location.known:
                    continue
                lines.append(f"- {location.name} ({location.kind})")
                if location.memory_summary:
                    lines.append(f"  {location.memory_summary}")
                for contract_line in location.related_contracts:
                    lines.append(f"  {contract_line}")
        else:
            lines.append("- none")
        lines.extend(("", "Charted Approaches"))
        if view.known_routes:
            lines.extend(
                f"- {route_id.replace('_', ' ').title()}" for route_id in view.known_routes
            )
        else:
            lines.append("- none")
        lines.extend(("", "Active Contracts"))
        if view.active_contracts:
            lines.extend(
                f"- {name}: {summary}" if summary else f"- {name}"
                for _contract_id, name, summary in view.active_contracts
            )
        else:
            lines.append("- none")
        return "\n".join(lines)

    def _travel_destination_label(self, view: WorldView, destination_id: str) -> str:
        destination = next(
            (
                candidate
                for candidate in view.travel_destinations
                if candidate.destination_id == destination_id
            ),
            None,
        )
        if destination is None:
            return destination_id.replace("_", " ").title()
        if destination.location_id == "haven" and view.current_location_id != "haven":
            return "Enter Haven"
        return f"Travel {destination.label}"

    def _place_name(self, location_id: str) -> str:
        location = self.app.controller.definitions.locations.get(location_id)
        if location is not None:
            return location.name
        return location_id.replace("_", " ").title()

    def _available_regional_room_action_commands(
        self,
        view: RegionalMapView,
    ) -> tuple[ScreenAction, ...]:
        return self.app._regional_handlers.available_room_action_commands(view)

    def _regional_move_lands_on_place(
        self,
        view: RegionalMapView,
        events: Sequence[GameEvent],
    ) -> bool:
        return self.app._regional_handlers.move_lands_on_place(view, events)

    def _focused_regional_node_id(self) -> str:
        action = self.app.focused_action
        if action is None or self.app.current_regional_view is None:
            return ""
        if action.value in {
            self.app.current_regional_view.current_node_id,
            "back",
            "survey_route",
            "mark_route",
            "map",
            "world_map",
            "system",
        }:
            return self.app.current_regional_view.current_node_id
        from game.expedition.travel import REGIONAL_OVERWORLD_NODE_IDS

        if action.value in REGIONAL_OVERWORLD_NODE_IDS:
            return action.value
        if action.value in {"haven", "shallow_cave"}:
            node_by_location_id = {
                node.location_id: node.node_id for node in self.app.current_regional_view.map_nodes
            }
            return node_by_location_id.get(action.value, action.value)
        if action.value == "begin_opening_route":
            return self.app.current_regional_view.other_node_id
        return self.app.current_regional_view.current_node_id
