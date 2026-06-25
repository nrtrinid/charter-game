"""Application orchestration flows used by AppController."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from game.app.commands import (
    ChooseCombatSkill,
    ChooseCombatTarget,
    DelayCombatTurn,
    MoveCombatActor,
    PassCombatTurn,
    ResolveCombatAction,
    ResolveCombatReaction,
    Retreat,
    RetreatCombat,
    SelectTarget,
    StartManualCombat,
    UseCombatSkill,
    ViewCombat,
)
from game.app.flows.base import ControllerFlow
from game.app.manual_combat import (
    ManualCombatSession,
    legal_skill_ids,
    legal_target_ids,
    resolve_enemy_reaction,
    resolve_hero_action,
    resolve_hero_delay,
    resolve_hero_move,
    resolve_hero_pass,
    resolve_hero_retreat,
    start_manual_session,
)
from game.app.views import (
    build_combat_view,
)
from game.campaign.company import CompanyState
from game.campaign.roster import sync_company_from_combat
from game.core.events import (
    CombatRetreatedEvent,
    ExpeditionReturnedEvent,
    GameEvent,
)
from game.core.result import Result
from game.expedition.cave import create_cave_boss_combat, create_encounter_combat
from game.expedition.dungeon import (
    SHALLOW_CAVE_BOSS_NODE_ID,
    active_dungeon_nodes,
    finish_shallow_cave_boss,
    mark_pending_combat_cleared,
    open_opening_breach_room,
    record_events,
    return_from_dungeon,
)
from game.expedition.expedition import (
    OPENING_BREACH_PENDING_FLAG,
    OPENING_EXPEDITION_ID,
)
from game.expedition.travel import (
    apply_node_rewards,
    event_for_node,
    mark_regional_combat_cleared,
    opening_nodes,
    set_company_location,
    set_company_node_location,
)

MANUAL_STAGE_SHALLOW_CAVE = "shallow_cave"
MANUAL_STAGE_CAVE_BOSS = "cave_mini_boss"


@dataclass
class ManualCombatFlow(ControllerFlow):
    def handle(self, command: object) -> Result[Any]:
        if isinstance(command, StartManualCombat):
            return self._start_manual_combat(command.encounter_id)
        if isinstance(command, ViewCombat):
            if self.manual_combat is None:
                return Result.fail("No manual combat is active.")
            return Result.ok(
                build_combat_view(
                    self.manual_combat,
                    self.definitions,
                    retreat_available=self._combat_retreat_available(),
                    debug_combat_preview=self.debug_combat_preview,
                )
            )
        if isinstance(command, ChooseCombatSkill):
            return self._choose_combat_skill(command)
        if isinstance(command, ChooseCombatTarget):
            return self._choose_combat_target(command)
        if isinstance(command, ResolveCombatAction):
            return self._resolve_combat_action(command)
        if isinstance(command, UseCombatSkill):
            return self._use_combat_skill(command)
        if isinstance(command, SelectTarget):
            return self._select_target(command)
        if isinstance(command, MoveCombatActor):
            return self._move_combat_actor(command)
        if isinstance(command, PassCombatTurn):
            return self._pass_combat_turn()
        if isinstance(command, DelayCombatTurn):
            return self._delay_combat_turn()
        if isinstance(command, RetreatCombat):
            return self._retreat_combat()
        if isinstance(command, Retreat):
            return self._retreat_combat()
        if isinstance(command, ResolveCombatReaction):
            return self._resolve_combat_reaction(command)
        return Result.fail(f"Unsupported combat command: {command!r}")

    def _combat_retreat_available(self) -> bool:
        return bool(
            self.company is not None
            and self.company.active_expedition is not None
            and self.manual_combat is not None
            and self.manual_combat.pending_hero() is not None
        )

    def _start_manual_combat(self, encounter_id: str) -> Result[Any]:
        return self._require_company(
            lambda company: self._create_manual_session(
                company,
                encounter_id,
                encounter_id.replace("_", " ").title(),
            )
        )

    def _choose_combat_skill(self, command: ChooseCombatSkill) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        if command.skill_id not in legal_skill_ids(self.manual_combat, self.definitions):
            return Result.fail("Choose a listed skill.")
        self.manual_combat.selected_skill_id = command.skill_id
        self.manual_combat.selected_target_id = None
        return Result.ok(
            build_combat_view(
                self.manual_combat,
                self.definitions,
                retreat_available=self._combat_retreat_available(),
                debug_combat_preview=self.debug_combat_preview,
            )
        )

    def _choose_combat_target(self, command: ChooseCombatTarget) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        skill_id = self.manual_combat.selected_skill_id
        if skill_id is None:
            return Result.fail("Choose a skill first.")
        if command.target_id not in legal_target_ids(
            self.manual_combat,
            self.definitions,
            skill_id,
        ):
            return Result.fail("Choose a listed target.")
        self.manual_combat.selected_target_id = command.target_id
        return Result.ok(
            build_combat_view(
                self.manual_combat,
                self.definitions,
                retreat_available=self._combat_retreat_available(),
                debug_combat_preview=self.debug_combat_preview,
            )
        )

    def _resolve_combat_action(self, command: ResolveCombatAction) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        if command.skill_id not in legal_skill_ids(self.manual_combat, self.definitions):
            return Result.fail("Choose a listed skill.")
        if command.target_id not in legal_target_ids(
            self.manual_combat,
            self.definitions,
            command.skill_id,
        ):
            return Result.fail("Choose a listed target.")

        events = resolve_hero_action(
            self.manual_combat,
            self.definitions,
            self.rng,
            command.skill_id,
            command.target_id,
        )
        return self._complete_combat_turn(events)

    def _use_combat_skill(self, command: UseCombatSkill) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        actor = self.manual_combat.pending_hero()
        if actor is None or actor.actor_id != command.actor_id:
            return Result.fail("Choose the active hero.")
        return self._resolve_combat_action(ResolveCombatAction(command.skill_id, command.target_id))

    def _select_target(self, command: SelectTarget) -> Result[Any]:
        return self._choose_combat_target(ChooseCombatTarget(command.target_id))

    def _resolve_combat_reaction(self, command: ResolveCombatReaction) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        events = resolve_enemy_reaction(
            self.manual_combat,
            self.definitions,
            self.rng,
            command.reaction_id,
        )
        if not events:
            return Result.fail("Choose a listed reaction.")
        return self._complete_combat_turn(events)

    def _move_combat_actor(self, command: MoveCombatActor) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        events = resolve_hero_move(
            self.manual_combat,
            self.definitions,
            self.rng,
            command.to_slot,
        )
        if not events:
            return Result.fail("Choose a listed movement option.")
        return self._complete_combat_turn(events)

    def _pass_combat_turn(self) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        events = resolve_hero_pass(
            self.manual_combat,
            self.definitions,
            self.rng,
        )
        if not events:
            return Result.fail("No hero can pass right now.")
        return self._complete_combat_turn(events)

    def _delay_combat_turn(self) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        events = resolve_hero_delay(
            self.manual_combat,
            self.definitions,
            self.rng,
        )
        if not events:
            return Result.fail("No later turn slot is available this round.")
        return self._complete_combat_turn(events)

    def _complete_combat_turn(self, events: list[GameEvent]) -> Result[Any]:
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        view = build_combat_view(
            self.manual_combat,
            self.definitions,
            retreat_available=self._combat_retreat_available(),
            debug_combat_preview=self.debug_combat_preview,
        )
        if self.manual_combat.ended:
            events.extend(self._finish_manual_encounter(events))
        elif self.manual_combat is not None:
            view = build_combat_view(
                self.manual_combat,
                self.definitions,
                retreat_available=self._combat_retreat_available(),
                debug_combat_preview=self.debug_combat_preview,
            )
        return Result.ok(view, events)

    def _retreat_combat(self) -> Result[Any]:
        if self.company is None:
            return Result.fail("Start or load a company first.")
        if self.manual_combat is None:
            return Result.fail("No manual combat is active.")
        if self.manual_combat.pending_hero() is None:
            return Result.fail("Retreat is only available on a hero command.")
        if self.company.active_expedition is None:
            return Result.fail("Retreat is only available during active dungeon combat.")

        events = resolve_hero_retreat(
            self.manual_combat,
            self.definitions,
            self.rng,
        )
        if not events:
            return Result.fail("Retreat is already pending.")
        return self._complete_combat_turn(events)

    def _finish_dungeon_retreat(self, session: ManualCombatSession) -> list[GameEvent]:
        if self.company is None or self.company.active_expedition is None:
            return []
        active_session = self.company.active_expedition
        nodes = active_dungeon_nodes(self.definitions, active_session)
        to_node_id = self._combat_retreat_destination_node_id(active_session)
        from_node_id = active_session.pending_combat_node_id or active_session.current_node_id
        if from_node_id != to_node_id:
            active_session.previous_node_id = from_node_id
        active_session.current_node_id = to_node_id
        active_session.pending_combat_node_id = None
        if to_node_id not in active_session.visited_node_ids:
            active_session.visited_node_ids.append(to_node_id)
        to_node = nodes[to_node_id]
        set_company_node_location(self.company, to_node)
        actor_id = session.retreat_actor_id or ""
        event = CombatRetreatedEvent(
            message=(
                "You withdraw from the fight. "
                "The enemy still holds this place. "
                "Re-entering will restart the encounter."
            ),
            actor_id=actor_id,
            encounter_id=session.encounter_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
        )
        if active_session.report is not None:
            record_events(
                active_session.report,
                [
                    *self._combat_report_context_events(session, from_node_id),
                    event,
                ],
            )
        events: list[GameEvent] = [event]
        self.manual_combat = None
        self.opening_manual_stage = None
        return events

    def _combat_retreat_destination_node_id(
        self,
        active_session: Any,
    ) -> str:
        generated = active_session.generated_dungeon
        if (
            generated is not None
            and not generated.collapsed
            and active_session.pending_combat_node_id
            in {node.id for node in generated.nodes}
        ):
            return generated.entry_node_id
        return self._latest_safe_return_node_id()

    def _create_manual_session(
        self,
        company: CompanyState,
        encounter_id: str,
        encounter_name: str,
    ) -> Result[Any]:
        combat = create_encounter_combat(company, self.definitions, encounter_id)
        self.manual_combat, events = start_manual_session(
            encounter_id,
            encounter_name,
            combat,
            self.definitions,
            self.rng,
            enemy_ai_mode=self.controller.enemy_ai_mode,
        )
        self.opening_manual_stage = None
        return Result.ok(
            build_combat_view(
                self.manual_combat,
                self.definitions,
                retreat_available=self._combat_retreat_available(),
                debug_combat_preview=self.debug_combat_preview,
            ),
            events,
        )

    def _finish_manual_encounter(
        self,
        resolved_events: list[GameEvent] | None = None,
    ) -> list[GameEvent]:
        if self.company is None or self.manual_combat is None:
            return []

        events: list[GameEvent] = []
        session = self.manual_combat
        resolved_events = resolved_events or []
        sync_company_from_combat(
            self.company,
            session.state.heroes,
            session.state.party_formation,
        )

        if session.outcome == "retreat":
            if self.company.active_expedition is not None:
                events.extend(self._finish_dungeon_retreat(session))
            else:
                self.manual_combat = None
                self.opening_manual_stage = None
            return events

        if self.company.active_expedition is not None:
            events.extend(self._finish_dungeon_manual_encounter(session, resolved_events))
            return events

        if self.opening_manual_stage == MANUAL_STAGE_SHALLOW_CAVE:
            if session.state.is_defeat():
                self.company.expedition_history.append("opening_failed_shallow_cave")
                events.extend(self._manual_return_to_haven())
                return events
            events.extend(self._start_opening_boss_combat())
            return events

        if self.opening_manual_stage == MANUAL_STAGE_CAVE_BOSS:
            if session.state.is_defeat():
                self.company.expedition_history.append("opening_failed_cave_boss")
                events.extend(self._manual_return_to_haven())
                return events
            events.extend(self._finish_manual_opening_to_breach())
            return events

        pending_regional = self.company.town_state.get("pending_regional_combat_node_id")
        if pending_regional:
            if session.state.is_defeat():
                self.company.town_state.pop("pending_regional_combat_node_id", None)
                self.controller.return_to_regional_place = True
                self.manual_combat = None
                return events
            mark_regional_combat_cleared(self.company, self.definitions)
            self.controller.return_to_regional_place = True
            self.manual_combat = None
            return events

        self.manual_combat = None
        return events

    def _finish_dungeon_manual_encounter(
        self,
        session: ManualCombatSession,
        _resolved_events: list[GameEvent],
    ) -> list[GameEvent]:
        if self.company is None or self.company.active_expedition is None:
            return []

        events: list[GameEvent] = []
        active_session = self.company.active_expedition
        pending_node_id = active_session.pending_combat_node_id
        combat_report_events = self._combat_report_context_events(session, pending_node_id)
        if session.state.is_defeat():
            if session.encounter_id == "shallow_cave":
                self.company.expedition_history.append("opening_failed_shallow_cave")
            elif session.encounter_id == "cave_mini_boss":
                self.company.expedition_history.append("opening_failed_cave_boss")
            if active_session.report is not None:
                record_events(active_session.report, combat_report_events)
            events.extend(
                return_from_dungeon(
                    self.company,
                    self.definitions,
                    outcome="defeat",
                    message="The company is carried back to Haven after defeat.",
                )
            )
            self.manual_combat = None
            self.opening_manual_stage = None
            return events

        mark_pending_combat_cleared(
            self.company,
            self.definitions,
            session.encounter_id,
            combat_report_events,
        )
        self.manual_combat = None
        self.opening_manual_stage = None

        if pending_node_id == SHALLOW_CAVE_BOSS_NODE_ID:
            events.extend(finish_shallow_cave_boss(self.company, self.definitions))
        return events

    def _combat_report_context_events(
        self,
        session: ManualCombatSession,
        node_id: str | None,
    ) -> list[GameEvent]:
        if self.company is not None and self.company.active_expedition is not None:
            nodes = active_dungeon_nodes(self.definitions, self.company.active_expedition)
        else:
            nodes = opening_nodes(self.definitions)
        events: list[GameEvent] = []
        if node_id is not None and node_id in nodes:
            events.append(event_for_node(nodes[node_id]))
        events.extend(session.event_log)
        return events

    def _start_opening_boss_combat(self) -> list[GameEvent]:
        if self.company is None:
            return []
        nodes = opening_nodes(self.definitions)
        events: list[GameEvent] = [event_for_node(nodes["shallow_cave_room_3"])]
        combat = create_cave_boss_combat(self.company, self.definitions)
        self.manual_combat, combat_events = start_manual_session(
            MANUAL_STAGE_CAVE_BOSS,
            nodes["shallow_cave_room_3"].name,
            combat,
            self.definitions,
            self.rng,
            enemy_ai_mode=self.controller.enemy_ai_mode,
        )
        self.opening_manual_stage = MANUAL_STAGE_CAVE_BOSS
        events.extend(combat_events)
        return events

    def _finish_manual_opening_to_breach(self) -> list[GameEvent]:
        if self.company is None:
            return []
        nodes = opening_nodes(self.definitions)
        events: list[GameEvent] = [event_for_node(nodes["cave_mini_boss"])]
        events.extend(apply_node_rewards(self.company, nodes["cave_mini_boss"], self.definitions))
        events.extend(open_opening_breach_room(self.company, self.definitions))
        self.manual_combat = None
        self.opening_manual_stage = None
        return events

    def _manual_return_to_haven(self) -> list[GameEvent]:
        if self.company is None:
            return []
        set_company_location(self.company, "haven", "Haven Town")
        self.company.flags[OPENING_BREACH_PENDING_FLAG] = False
        self.manual_combat = None
        self.opening_manual_stage = None
        return [
            ExpeditionReturnedEvent(
                message="The company returns to Haven.",
                expedition_id=OPENING_EXPEDITION_ID,
                location="Haven Town",
            )
        ]
