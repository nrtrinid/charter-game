"""TUI screen rendering for combat."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from game.app.commands import (
    ViewCombat,
)
from game.app.views import (
    CombatView,
    ScreenAction,
)
from game.core.events import (
    GameEvent,
)
from game.core.hci import EventBeat, HciResultAnalysis
from game.ui.hci_text import (
    format_meta_line,
)
from game.ui.tui_render.protocol import TuiRenderHost
from game.ui.tui_widgets import (
    CombatPanel,
)


@dataclass
class CombatRender:
    app: TuiRenderHost

    def show_combat_command(self, message: str = "") -> None:
        result = self.app.controller.handle(ViewCombat())
        if not result.success:
            company = self.app.controller.company
            if company is not None and company.active_expedition is not None:
                self.app._show_dungeon(result.error or "No combat is active.")
            else:
                self.app._show_expedition(result.error or "No combat is active.")
            return
        view = cast(CombatView, result.value)
        phase = "reaction" if view.pending_enemy_intent is not None else "command"
        self.show_combat_view(view, phase=phase, message=message)

    def show_combat_skill(self, message: str = "") -> None:
        result = self.app.controller.handle(ViewCombat())
        if not result.success:
            company = self.app.controller.company
            if company is not None and company.active_expedition is not None:
                self.app._show_dungeon(result.error or "No combat is active.")
            else:
                self.app._show_expedition(result.error or "No combat is active.")
            return
        view = cast(CombatView, result.value)
        self.show_combat_view(view, phase="skill", message=message)

    def show_combat_view(
        self,
        view: CombatView,
        *,
        phase: str,
        message: str = "",
    ) -> None:
        self._track_combat_turn_handoff(view, phase)
        self.app.current_combat_view = view
        self.app.current_combat_phase = phase
        self.app.selected_skill_id = view.selected_skill_id if phase == "target" else None
        if phase == "command":
            actions = view.commands
        elif phase == "skill":
            actions = tuple(skill.action for skill in view.skills)
        elif phase == "move":
            actions = tuple(move.action for move in view.moves)
        elif phase == "reaction":
            actions = tuple(option.action for option in view.reaction_options)
        else:
            actions = tuple(target.action for target in view.targets)
        if not actions:
            actions = (ScreenAction("1", "Continue", "continue", ("c",), default=True),)
        subtitles = {
            "command": "Choose Command",
            "skill": "Choose Skill",
            "target": "Choose Target",
            "move": "Choose Move",
            "reaction": "Reaction",
        }
        subtitle = subtitles.get(phase, "Choose Command")
        self.app._show_screen(
            "combat",
            f"Combat - {subtitle}",
            self._combat_text(view, mode=phase),
            actions,
            message=message,
            log=self.app._events_text(view.recent_events),
        )

    def _track_combat_turn_handoff(self, view: CombatView, phase: str) -> None:
        if phase != "command" or view.current_actor is None:
            return
        actor_id = view.current_actor.actor_id
        if self.app.last_combat_actor_id and actor_id != self.app.last_combat_actor_id:
            self.app.turn_flash_actor_id = actor_id
            self.app.turn_flash_frame = 0
        self.app.last_combat_actor_id = actor_id

    def show_combat_target(self, view: CombatView, message: str = "") -> None:
        self.show_combat_view(view, phase="target", message=message)

    def show_resolution(
        self,
        events: list[GameEvent],
        hci: HciResultAnalysis | None = None,
    ) -> None:
        hero_events, enemy_events = self.app._split_turn_events(events)
        self.app.pending_enemy_events = enemy_events
        self.app.pending_enemy_beats = self._combat_event_beats(enemy_events)
        self.app.pending_enemy_view = self.app.current_combat_view
        self.app.pending_resolution_hci = hci
        display_events = hero_events or events
        self.app._record_events(display_events)
        title = "Combat Complete" if self.app.controller.manual_combat is None else "Hero Action"
        self.app._set_current_beat(
            title,
            display_events,
            self.app.current_combat_view,
            deferred_events=enemy_events,
        )
        continue_label = self.app._post_combat_continue_label()
        actions = (
            ScreenAction(
                "1",
                "Enemy Response" if enemy_events else continue_label,
                "enemy_response" if enemy_events else "continue",
                ("c",),
                default=True,
                description=(
                    "Resolve enemy actions."
                    if enemy_events
                    else self.app._post_combat_continue_description()
                ),
            ),
        )
        self.app._show_screen(
            "resolution",
            title,
            self._combat_beat_text(
                title,
                display_events,
                view=self.app.current_combat_view,
                animation_frame=self.app.beat_animation_frame,
                deferred_events=enemy_events,
            ),
            actions,
            log=self.app._result_log_text(display_events, hci),
        )

    def show_enemy_turn(self) -> None:
        if not self.app.pending_enemy_beats:
            self.app.pending_enemy_events = []
            self.app.pending_enemy_view = None
            self.app.pending_resolution_hci = None
            self.app._finish_playback()
            return

        events = self.app.pending_enemy_beats.pop(0)
        remaining_events = [event for beat in self.app.pending_enemy_beats for event in beat]
        self.app.pending_enemy_events = [*events, *remaining_events]
        self.app._record_events(events)
        self.app._set_current_beat(
            "Enemy Response",
            events,
            self.app.pending_enemy_view,
            deferred_events=remaining_events,
        )
        continue_label = self.app._post_combat_continue_label()
        has_more = bool(self.app.pending_enemy_beats)
        actions = (
            ScreenAction(
                "1",
                "Next Enemy Action" if has_more else continue_label,
                "enemy_response" if has_more else "continue",
                ("c",),
                default=True,
                description=(
                    "Resolve the next enemy action."
                    if has_more
                    else self.app._post_combat_continue_description()
                ),
            ),
        )
        self.app._show_screen(
            "enemy_turn",
            "Enemy Response",
            CombatPanel.render_enemy_turn(
                self.app.pending_enemy_view,
                events,
                source_actor_ids=self.app._event_source_actor_ids(events),
                target_intents=self.app._event_target_intents(events),
                animation_frame=self.app.beat_animation_frame,
                deferred_events=remaining_events,
            ),
            actions,
            log=self.app._result_log_text(events, self.app.pending_resolution_hci),
        )

    def _show_opening_enemy_response_if_needed(self) -> bool:
        return self.app._combat_handlers.show_opening_enemy_response_if_needed()

    def _combat_text(self, view: CombatView, *, mode: str, idle_frame: int = 0) -> str:
        return CombatPanel.render_text(
            view,
            phase=mode,
            focused_action=self.app.focused_action,
            idle_frame=idle_frame,
            turn_flash_actor_id=self.app.turn_flash_actor_id,
            turn_flash_frame=self.app.turn_flash_frame,
        )

    def _combat_beat_text(
        self,
        title: str,
        events: list[GameEvent],
        *,
        view: CombatView | None = None,
        animation_frame: int = 0,
        deferred_events: list[GameEvent] | None = None,
    ) -> str:
        if not events:
            return "The turn resolves."
        return CombatPanel.render_combat_beat(
            view if view is not None else self.app.current_combat_view,
            events,
            title=title,
            source_actor_ids=self.app._event_source_actor_ids(events),
            target_intents=self.app._event_target_intents(events),
            animation_frame=animation_frame,
            deferred_events=deferred_events or [],
        )

    def _combat_event_beats(self, events: list[GameEvent]) -> list[list[GameEvent]]:
        beats: list[list[GameEvent]] = []
        current: list[GameEvent] = []
        for event in events:
            if self.app._is_combat_beat_start(event):
                if self.app._continues_danger_beat(current, event):
                    current.append(event)
                    continue
                if current:
                    beats.append(current)
                current = [event]
                continue
            if current:
                current.append(event)
            else:
                current = [event]
                if self.app._is_combat_footer_event(event):
                    beats.append(current)
                    current = []
        if current:
            beats.append(current)
        return beats

    def _current_beat_text(self) -> str:
        if self.app.screen_state == "enemy_turn":
            return CombatPanel.render_enemy_turn(
                self.app.current_beat_view,
                self.app.current_beat_events,
                source_actor_ids=self.app._event_source_actor_ids(self.app.current_beat_events),
                target_intents=self.app._event_target_intents(self.app.current_beat_events),
                animation_frame=self.app.beat_animation_frame,
                deferred_events=self.app.current_beat_deferred_events,
            )
        return self._combat_beat_text(
            self.app.current_beat_title,
            self.app.current_beat_events,
            view=self.app.current_beat_view,
            animation_frame=self.app.beat_animation_frame,
            deferred_events=self.app.current_beat_deferred_events,
        )

    def _current_beat_animation_last_frame(self) -> int:
        return CombatPanel.beat_animation_last_frame(
            self.app.current_beat_view,
            self.app.current_beat_events,
            source_actor_ids=self.app._event_source_actor_ids(self.app.current_beat_events),
            target_intents=self.app._event_target_intents(self.app.current_beat_events),
        )

    def _combat_detail_text(self, action: ScreenAction) -> str:
        view = self.app.current_combat_view
        if view is None:
            return "Combat detail unavailable."
        return CombatPanel.detail_text(
            view,
            self.app.current_combat_phase,
            action,
            idle_frame=self.app.idle_animation_frame,
        )

    def _combatant_lines(self, actors: Sequence[Any]) -> str:
        if not actors:
            return "none"
        lines = []
        for actor in actors:
            marker = ">" if actor.acting else " "
            status = ", ".join(actor.statuses)
            marks = ", ".join(getattr(actor, "strain_marks", ()))
            detail = format_meta_line(
                actor.name,
                f"{actor.hp}/{actor.max_hp} HP",
                f"{actor.effort}/{actor.max_effort} Effort",
                f"Strain {getattr(actor, 'strain', '')}",
                f"Marks {marks}" if marks else "",
                status,
            )
            lines.append(f"{marker} {actor.slot}: {detail}")
        return "\n".join(lines)

    def _beat_text(self, beat: EventBeat) -> str:
        return f"{beat.title}\n" + "\n".join(event.message for event in beat.events)
