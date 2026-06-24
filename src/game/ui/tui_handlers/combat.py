"""Combat action handlers for the TUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from game.app.commands import (
    ChooseCombatSkill,
    ChooseCombatTarget,
    DelayCombatTurn,
    MoveCombatActor,
    PassCombatTurn,
    ResolveCombatAction,
    ResolveCombatReaction,
    RetreatCombat,
    ViewCombat,
)
from game.app.views import CombatView
from game.combat.formation import FormationSlot
from game.ui.tui_handlers.protocol import TuiHandlerHost


@dataclass
class CombatHandlers:
    app: TuiHandlerHost

    def activate_combat_action(self, value: str) -> None:
        if self.app.current_combat_phase == "command":
            self.handle_combat_command(value)
        elif self.app.current_combat_phase == "skill":
            self.choose_skill(value)
        elif self.app.current_combat_phase == "move":
            self.choose_move(value)
        elif self.app.current_combat_phase == "reaction":
            self.choose_reaction(value)
        else:
            self.choose_target(value)

    def advance_resolution(self) -> None:
        if self.app.pending_enemy_events:
            self.app._show_enemy_turn()
        else:
            self.app._finish_playback()

    def advance_enemy_turn(self) -> None:
        if self.app.pending_enemy_beats:
            self.app._show_enemy_turn()
        else:
            self.app.pending_enemy_events = []
            self.app.pending_enemy_beats = []
            self.app.pending_enemy_view = None
            self.app._finish_playback()

    def show_opening_enemy_response_if_needed(self) -> bool:
        session = self.app.controller.manual_combat
        if session is None:
            return False
        session_key = f"{session.encounter_id}:{id(session)}"
        if self.app.opening_enemy_response_session_key == session_key:
            return False

        _hero_events, enemy_events = self.app._split_turn_events(list(session.event_log))
        enemy_beats = self.app._enemy_response_beats(enemy_events)
        if not enemy_beats:
            self.app.opening_enemy_response_session_key = session_key
            return False

        result = self.app.controller.handle(ViewCombat())
        if not result.success or not isinstance(result.value, CombatView):
            return False

        self.app.current_combat_view = result.value
        self.app.current_combat_phase = "command"
        self.app.pending_enemy_view = result.value
        self.app.pending_enemy_beats = enemy_beats
        self.app.pending_enemy_events = [event for beat in enemy_beats for event in beat]
        self.app.pending_resolution_hci = None
        self.app.opening_enemy_response_session_key = session_key
        self.app._show_enemy_turn()
        return True

    def handle_combat_command(self, value: str) -> None:
        view = self.app.current_combat_view
        if value == "skill":
            if view is None:
                self.app._show_combat_skill()
            else:
                self.app._show_combat_view(view, phase="skill")
            return
        if value == "move":
            if view is None:
                self.app._show_combat_command("Move options are unavailable.")
            else:
                self.app._show_combat_view(view, phase="move")
            return
        if value == "pass":
            result = self.app.controller.handle(PassCombatTurn())
            if not result.success:
                self.app._show_combat_command(result.error or "Pass failed.")
                return
            self.app.current_combat_view = cast(CombatView | None, result.value)
            self.app._show_resolution(result.events, result.hci)
            return
        if value == "delay":
            result = self.app.controller.handle(DelayCombatTurn())
            if not result.success:
                self.app._show_combat_command(result.error or "Delay failed.")
                return
            self.app.current_combat_view = cast(CombatView | None, result.value)
            self.app._show_resolution(result.events, result.hci)
            return
        if value == "retreat":
            result = self.app.controller.handle(RetreatCombat())
            if not result.success:
                self.app._show_combat_command(result.error or "Retreat failed.")
                return
            self.app.current_combat_view = None
            self.app._show_resolution(result.events, result.hci)
            return
        self.app._show_combat_command("Choose a listed combat command.")

    def choose_skill(self, skill_id: str) -> None:
        result = self.app.controller.handle(ChooseCombatSkill(skill_id))
        if not result.success:
            self.app._show_combat_skill(result.error or "Choose a listed skill.")
            return
        view = cast(CombatView, result.value)
        self.app._show_combat_target(view)

    def choose_move(self, slot_id: str) -> None:
        try:
            to_slot = FormationSlot(slot_id)
        except ValueError:
            self.app._show_combat_command("Choose a listed movement option.")
            return
        result = self.app.controller.handle(MoveCombatActor(to_slot))
        if not result.success:
            view_result = self.app.controller.handle(ViewCombat())
            if view_result.success and isinstance(view_result.value, CombatView):
                self.app._show_combat_view(
                    view_result.value,
                    phase="move",
                    message=result.error or "Move failed.",
                )
            else:
                self.app._show_combat_command(result.error or "Move failed.")
            return
        self.app.current_combat_view = cast(CombatView | None, result.value)
        self.app._show_resolution(result.events, result.hci)

    def choose_reaction(self, value: str) -> None:
        reaction_id = None if value == "skip" else value
        result = self.app.controller.handle(ResolveCombatReaction(reaction_id))
        if not result.success:
            self.app._show_combat_command(result.error or "Choose a listed reaction.")
            return
        self.app.current_combat_view = cast(CombatView | None, result.value)
        self.app._show_resolution(result.events, result.hci)

    def choose_target(self, target_id: str) -> None:
        skill_id = self.app.selected_skill_id
        if skill_id is None:
            self.app._show_combat_skill("Choose a skill first.")
            return
        target_result = self.app.controller.handle(ChooseCombatTarget(target_id))
        if not target_result.success:
            view = self.app.current_combat_view
            if view is not None:
                self.app._show_combat_target(view, target_result.error or "Choose a listed target.")
            return
        result = self.app.controller.handle(ResolveCombatAction(skill_id, target_id))
        if not result.success:
            self.app._show_combat_skill(result.error or "The action could not resolve.")
            return
        self.app.current_combat_view = cast(CombatView | None, result.value)
        self.app._show_resolution(result.events, result.hci)

    def back_from_combat(self) -> None:
        if self.app.current_combat_phase == "target":
            self.app._show_combat_skill()
        elif self.app.current_combat_phase in {"skill", "move"}:
            self.app._show_combat_command()
        elif self.app.current_combat_phase == "reaction":
            self.app.message = "Choose a reaction or Skip Reaction to resolve the enemy intent."
            self.app._render()
        else:
            self.app.message = "Use Retreat to leave combat from a safe dungeon fight."
            self.app._render()

    def blocked_resolution_back(self) -> None:
        self.app.message = "Combat resolution is already in motion."
        self.app._render()
