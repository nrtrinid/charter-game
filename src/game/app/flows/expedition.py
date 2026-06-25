"""Application orchestration flows used by AppController."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from game.app.commands import (
    StartExpedition,
    TakeExpeditionChoice,
)
from game.app.flows.base import ControllerFlow
from game.app.flows.dungeon import DungeonFlow
from game.app.flows.manual_combat import MANUAL_STAGE_SHALLOW_CAVE
from game.app.manual_combat import (
    start_manual_session,
)
from game.campaign.company import CompanyState
from game.campaign.roster import active_roster
from game.core.events import (
    GameEvent,
)
from game.core.result import Result
from game.expedition.cave import create_encounter_combat
from game.expedition.dungeon import (
    active_report,
    finish_report,
    record_events,
)
from game.expedition.expedition import (
    OPENING_EXPEDITION_ID,
    descend_from_breach,
    return_to_haven_from_breach,
    run_opening_route,
)
from game.expedition.travel import (
    apply_node_rewards,
    event_for_node,
    opening_nodes,
    set_company_node_location,
    spend_ration,
)


@dataclass
class ExpeditionFlow(ControllerFlow):
    def handle(self, command: object) -> Result[Any]:
        if isinstance(command, StartExpedition):
            return self._start_expedition(command)
        if isinstance(command, TakeExpeditionChoice):
            return self._take_expedition_choice(command)
        return Result.fail(f"Unsupported expedition command: {command!r}")

    def _start_expedition(self, command: StartExpedition) -> Result[Any]:
        if command.expedition_id != OPENING_EXPEDITION_ID:
            return Result.fail(f"Unknown expedition: {command.expedition_id}")
        if self.company is not None and not active_roster(self.company):
            return Result.fail("Assign at least one living hero to the active party first.")
        if command.interactive_dungeon:
            return self._require_company(
                lambda company: DungeonFlow(self.controller)._start_interactive_opening_dungeon(
                    company,
                    use_known_route=command.use_known_route,
                    skip_known_route_playback=command.skip_known_route_playback,
                )
            )
        if command.manual_combat:
            return self._require_company(self._start_manual_opening_route)
        return self._require_company(
            lambda company: Result.ok(
                company,
                run_opening_route(
                    company,
                    self.definitions,
                    self.rng,
                    enter_maze=command.enter_maze,
                    stop_at_breach=command.stop_at_breach,
                    enemy_ai_mode=self.controller.enemy_ai_mode,
                    enemy_wait_mode=self.controller.enemy_wait_mode,
                    enemy_movement_mode=self.controller.enemy_movement_mode,
                ),
            )
        )

    def _take_expedition_choice(self, command: TakeExpeditionChoice) -> Result[Any]:
        if command.choice_id == "return_to_haven":
            return self._require_company(self._return_from_breach)
        if command.choice_id == "descend_maze_depth_1":
            return self._require_company(self._descend_from_breach)
        return Result.fail(f"Unknown expedition choice: {command.choice_id}")

    def _return_from_breach(self, company: CompanyState) -> Result[Any]:
        report = active_report(company)
        events = return_to_haven_from_breach(company, self.definitions)
        if report is not None:
            record_events(report, events)
            finish_report(company, "returned_to_haven")
        return Result.ok(company, events)

    def _descend_from_breach(self, company: CompanyState) -> Result[Any]:
        report = active_report(company)
        events = descend_from_breach(
            company,
            self.definitions,
            self.rng,
            enemy_ai_mode=self.controller.enemy_ai_mode,
        )
        if report is not None:
            record_events(report, events)
            finish_report(company, "descended_maze_depth_1")
        return Result.ok(company, events)

    def _start_manual_opening_route(self, company: CompanyState) -> Result[Any]:
        nodes = opening_nodes(self.definitions)
        events: list[GameEvent] = []

        set_company_node_location(company, nodes["old_road"])
        spend_ration(company.supplies)
        events.append(event_for_node(nodes["old_road"]))
        events.extend(apply_node_rewards(company, nodes["old_road"], self.definitions))

        events.append(event_for_node(nodes["old_road_cache"]))
        events.extend(apply_node_rewards(company, nodes["old_road_cache"], self.definitions))

        set_company_node_location(company, nodes["blackwood_forest"])
        events.append(event_for_node(nodes["blackwood_forest"]))
        events.extend(apply_node_rewards(company, nodes["blackwood_forest"], self.definitions))

        set_company_node_location(company, nodes["shallow_cave_room_1"])
        events.append(event_for_node(nodes["shallow_cave_room_1"]))
        events.extend(apply_node_rewards(company, nodes["shallow_cave_room_1"], self.definitions))

        set_company_node_location(company, nodes["shallow_cave_room_2"])
        events.append(event_for_node(nodes["shallow_cave_room_2"]))
        combat = create_encounter_combat(company, self.definitions, MANUAL_STAGE_SHALLOW_CAVE)
        self.manual_combat, combat_events = start_manual_session(
            MANUAL_STAGE_SHALLOW_CAVE,
            nodes["shallow_cave_room_2"].name,
            combat,
            self.definitions,
            self.rng,
            enemy_ai_mode=self.controller.enemy_ai_mode,
        )
        self.opening_manual_stage = MANUAL_STAGE_SHALLOW_CAVE
        events.extend(combat_events)
        return Result.ok(company, events)
