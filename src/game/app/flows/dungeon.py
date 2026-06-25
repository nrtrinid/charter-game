"""Application orchestration flows used by AppController."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from game.app.commands import (
    ClearExpeditionReport,
    EnterGeneratedMaze,
    InspectDungeonRoom,
    MoveDungeon,
    RetraceGeneratedMaze,
    RetreatGeneratedMaze,
    ReturnFromDungeon,
    UseDungeonAction,
    ViewDungeon,
    ViewExpeditionReport,
    WithdrawGeneratedMaze,
)
from game.app.flows.base import ControllerFlow
from game.app.manual_combat import (
    start_manual_session,
)
from game.app.views import (
    build_dungeon_view,
    build_expedition_report_view,
    build_regional_arrival_context,
    build_regional_map_view,
)
from game.campaign.company import CompanyState
from game.core.events import (
    GameEvent,
)
from game.core.result import Result
from game.expedition.cave import create_encounter_combat
from game.expedition.dungeon import (
    active_dungeon_nodes,
    enter_dungeon_node,
    enter_generated_maze,
    generated_maze_frontier_exit_ids,
    move_generated_maze_if_needed,
    retrace_generated_maze,
    return_from_dungeon,
    revealed_exit_node_ids,
    start_interactive_opening_dungeon,
    use_dungeon_action,
    withdraw_generated_maze,
)
from game.expedition.travel import (
    event_for_node,
    regional_return_flavor,
)


@dataclass
class DungeonFlow(ControllerFlow):
    def handle(self, command: object) -> Result[Any]:
        if isinstance(command, ViewDungeon):
            return self._view_dungeon()
        if isinstance(command, InspectDungeonRoom):
            return self._inspect_dungeon_room()
        if isinstance(command, MoveDungeon):
            return self._move_dungeon(command.node_id)
        if isinstance(command, UseDungeonAction):
            return self._use_dungeon_action(command.action_id)
        if isinstance(command, EnterGeneratedMaze):
            return self._enter_generated_maze(command.seed)
        if isinstance(command, RetraceGeneratedMaze):
            return self._retrace_generated_maze()
        if isinstance(command, WithdrawGeneratedMaze):
            return self._withdraw_generated_maze()
        if isinstance(command, RetreatGeneratedMaze):
            return self._retrace_generated_maze()
        if isinstance(command, ReturnFromDungeon):
            return self._return_from_dungeon()
        if isinstance(command, ViewExpeditionReport):
            return self._view_expedition_report()
        if isinstance(command, ClearExpeditionReport):
            return self._clear_expedition_report()
        return Result.fail(f"Unsupported dungeon command: {command!r}")

    def _view_dungeon(self) -> Result[Any]:
        def view(company: CompanyState) -> Result[Any]:
            try:
                return Result.ok(
                    build_dungeon_view(
                        company,
                        self.definitions,
                        first_visit_node_id=self._current_first_visit_node_id(company),
                    )
                )
            except ValueError as exc:
                return Result.fail(str(exc))

        return self._require_company(view)

    def _inspect_dungeon_room(self) -> Result[Any]:
        return self._require_company(self._inspect_active_dungeon_room)

    def _move_dungeon(self, node_id: str) -> Result[Any]:
        return self._require_company(lambda company: self._move_active_dungeon(company, node_id))

    def _use_dungeon_action(self, action_id: str) -> Result[Any]:
        return self._require_company(
            lambda company: self._use_active_dungeon_action(company, action_id)
        )

    def _return_from_dungeon(self) -> Result[Any]:
        return self._require_company(self._return_from_active_dungeon)

    def _view_expedition_report(self) -> Result[Any]:
        def view(company: CompanyState) -> Result[Any]:
            try:
                return Result.ok(build_expedition_report_view(company, self.definitions))
            except ValueError as exc:
                return Result.fail(str(exc))

        return self._require_company(view)

    def _clear_expedition_report(self) -> Result[Any]:
        def clear(company: CompanyState) -> Result[Any]:
            company.last_expedition_report = None
            return Result.ok(company)

        return self._require_company(clear)

    def _start_interactive_opening_dungeon(
        self,
        company: CompanyState,
        *,
        use_known_route: bool = True,
        skip_known_route_playback: bool = False,
    ) -> Result[Any]:
        if company.active_expedition is not None:
            return Result.ok(
                build_dungeon_view(
                    company,
                    self.definitions,
                    first_visit_node_id=self._current_first_visit_node_id(company),
                )
            )
        events = start_interactive_opening_dungeon(
            company,
            self.definitions,
            use_known_route=use_known_route,
            skip_known_route_playback=skip_known_route_playback,
        )
        self.manual_combat = None
        self.opening_manual_stage = None
        self._remember_dungeon_entry(company, events)
        return Result.ok(
            build_dungeon_view(
                company,
                self.definitions,
                events,
                first_visit_node_id=self._current_first_visit_node_id(company),
            ),
            events,
        )

    def _inspect_active_dungeon_room(self, company: CompanyState) -> Result[Any]:
        session = company.active_expedition
        if session is None:
            return Result.fail("No active dungeon expedition.")
        node = active_dungeon_nodes(self.definitions, session)[session.current_node_id]
        events: list[GameEvent] = [event_for_node(node, first_visit=False)]
        self._remember_dungeon_entry(company, events)
        return Result.ok(
            build_dungeon_view(
                company,
                self.definitions,
                events,
                first_visit_node_id=self._current_first_visit_node_id(company),
            ),
            events,
        )

    def _move_active_dungeon(self, company: CompanyState, node_id: str) -> Result[Any]:
        session = company.active_expedition
        if session is None:
            return Result.fail("No active dungeon expedition.")
        if session.pending_combat_node_id is not None:
            return Result.fail("Resolve the pending room combat first.")
        nodes = active_dungeon_nodes(self.definitions, session)
        current = nodes[session.current_node_id]
        current_exit_ids = [
            *current.exits,
            *revealed_exit_node_ids(session, current.id),
            *generated_maze_frontier_exit_ids(session),
        ]
        if current.id not in session.cleared_node_ids:
            return Result.fail("Clear or inspect this room before moving on.")
        if node_id not in current_exit_ids:
            return Result.fail("Choose a listed dungeon exit.")
        try:
            generated_events = move_generated_maze_if_needed(
                company,
                self.definitions,
                node_id,
            )
        except ValueError as exc:
            return Result.fail(str(exc))
        if generated_events is not None:
            events = generated_events
            nodes = active_dungeon_nodes(self.definitions, session)
            destination = nodes[session.current_node_id]
        else:
            events = enter_dungeon_node(company, self.definitions, node_id)
            destination = nodes[node_id]
        if destination.encounter is not None and node_id not in session.cleared_node_ids:
            combat = create_encounter_combat(company, self.definitions, destination.encounter)
            self.manual_combat, combat_events = start_manual_session(
                destination.encounter,
                destination.name,
                combat,
                self.definitions,
                self.rng,
                enemy_ai_mode=self.controller.enemy_ai_mode,
            )
            self.opening_manual_stage = None
            events.extend(combat_events)
        self._remember_dungeon_entry(company, events)
        return Result.ok(
            build_dungeon_view(
                company,
                self.definitions,
                events,
                first_visit_node_id=self._current_first_visit_node_id(company),
            ),
            events,
        )

    def _use_active_dungeon_action(
        self,
        company: CompanyState,
        action_id: str,
    ) -> Result[Any]:
        if company.active_expedition is None:
            return Result.fail("No active dungeon expedition.")
        try:
            events = use_dungeon_action(company, self.definitions, action_id)
        except ValueError as exc:
            return Result.fail(str(exc))
        return Result.ok(
            build_dungeon_view(
                company,
                self.definitions,
                events,
                first_visit_node_id=self._current_first_visit_node_id(company),
            ),
            events,
        )

    def _enter_generated_maze(self, seed: int | None) -> Result[Any]:
        def enter(company: CompanyState) -> Result[Any]:
            try:
                events = enter_generated_maze(
                    company,
                    self.definitions,
                    self.rng,
                    seed=seed,
                )
            except ValueError as exc:
                return Result.fail(str(exc))
            self.manual_combat = None
            self.opening_manual_stage = None
            self._remember_dungeon_entry(company, events)
            return Result.ok(
                build_dungeon_view(
                    company,
                    self.definitions,
                    events,
                    first_visit_node_id=self._current_first_visit_node_id(company),
                ),
                events,
            )

        return self._require_company(enter)

    def _retrace_generated_maze(self) -> Result[Any]:
        def retrace(company: CompanyState) -> Result[Any]:
            try:
                events = retrace_generated_maze(company, self.definitions)
            except ValueError as exc:
                return Result.fail(str(exc))
            self.manual_combat = None
            self.opening_manual_stage = None
            self._remember_dungeon_entry(company, events)
            return Result.ok(
                build_dungeon_view(
                    company,
                    self.definitions,
                    events,
                    first_visit_node_id=self._current_first_visit_node_id(company),
                ),
                events,
            )

        return self._require_company(retrace)

    def _withdraw_generated_maze(self) -> Result[Any]:
        def withdraw(company: CompanyState) -> Result[Any]:
            try:
                events = withdraw_generated_maze(company, self.definitions)
            except ValueError as exc:
                return Result.fail(str(exc))
            self.manual_combat = None
            self.opening_manual_stage = None
            self._remember_dungeon_entry(company, events)
            return Result.ok(
                build_dungeon_view(
                    company,
                    self.definitions,
                    events,
                    first_visit_node_id=self._current_first_visit_node_id(company),
                ),
                events,
            )

        return self._require_company(withdraw)

    def _return_from_active_dungeon(self, company: CompanyState) -> Result[Any]:
        session = company.active_expedition
        if session is None:
            return Result.fail("No active dungeon expedition.")
        node = active_dungeon_nodes(self.definitions, session)[session.current_node_id]
        if not node.safe_return:
            return Result.fail("Return is only available from safe return rooms.")
        events = return_from_dungeon(
            company,
            self.definitions,
            origin_node_id=node.id,
        )
        arrival_context = build_regional_arrival_context(
            company,
            self.definitions,
            origin_name=node.name,
            flavor_line=regional_return_flavor(node.id),
        )
        return Result.ok(
            build_regional_map_view(
                company,
                self.definitions,
                arrival_context=arrival_context,
            ),
            events,
        )
