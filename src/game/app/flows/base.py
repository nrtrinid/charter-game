"""Application orchestration flows used by AppController."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from game.app.manual_combat import (
    ManualCombatSession,
)
from game.campaign.company import CompanyState
from game.campaign.recruitment import RecruitChoice
from game.content.definitions import GameDefinitions
from game.core.events import (
    ExpeditionEvent,
    GameEvent,
)
from game.core.result import Result
from game.core.rng import GameRng
from game.expedition.dungeon import (
    active_dungeon_nodes,
)


@dataclass
class ControllerFlow:
    controller: Any

    @property
    def definitions(self) -> GameDefinitions:
        return self.controller.definitions

    @property
    def rng(self) -> GameRng:
        return self.controller.rng

    @property
    def company(self) -> CompanyState | None:
        return self.controller.company

    @company.setter
    def company(self, value: CompanyState | None) -> None:
        self.controller.company = value

    @property
    def recruit_offers(self) -> list[RecruitChoice]:
        return self.controller.recruit_offers

    @recruit_offers.setter
    def recruit_offers(self, value: list[RecruitChoice]) -> None:
        self.controller.recruit_offers = value

    @property
    def manual_combat(self) -> ManualCombatSession | None:
        return self.controller.manual_combat

    @manual_combat.setter
    def manual_combat(self, value: ManualCombatSession | None) -> None:
        self.controller.manual_combat = value

    @property
    def opening_manual_stage(self) -> str | None:
        return self.controller.opening_manual_stage

    @opening_manual_stage.setter
    def opening_manual_stage(self, value: str | None) -> None:
        self.controller.opening_manual_stage = value

    @property
    def debug_combat_preview(self) -> bool:
        return self.controller.debug_combat_preview

    @property
    def dungeon_first_visit_node_id(self) -> str | None:
        return self.controller.dungeon_first_visit_node_id

    @dungeon_first_visit_node_id.setter
    def dungeon_first_visit_node_id(self, value: str | None) -> None:
        self.controller.dungeon_first_visit_node_id = value

    def _require_company(self, action: Callable[[CompanyState], Result[Any]]) -> Result[Any]:
        return self.controller._require_company(action)

    def _current_first_visit_node_id(self, company: CompanyState) -> str | None:
        session = company.active_expedition
        if session is None:
            return None
        if self.dungeon_first_visit_node_id == session.current_node_id:
            return self.dungeon_first_visit_node_id
        return None

    def _remember_dungeon_entry(self, company: CompanyState, events: list[GameEvent]) -> None:
        session = company.active_expedition
        if session is None:
            self.dungeon_first_visit_node_id = None
            return
        for event in events:
            if isinstance(event, ExpeditionEvent) and event.node_id == session.current_node_id:
                self.dungeon_first_visit_node_id = event.node_id if event.first_visit else None
                return

    def _pending_generated_combat(self) -> bool:
        if self.company is None or self.company.active_expedition is None:
            return False
        session = self.company.active_expedition
        generated = session.generated_dungeon
        if generated is None or generated.collapsed:
            return False
        return session.pending_combat_node_id in {node.id for node in generated.nodes}

    def _latest_safe_return_node_id(self) -> str:
        if self.company is None or self.company.active_expedition is None:
            return "town_gate"
        session = self.company.active_expedition
        nodes = active_dungeon_nodes(self.definitions, session)
        for node_id in reversed(session.visited_node_ids):
            node = nodes.get(node_id)
            if node is not None and node.safe_return:
                return node_id
        node = nodes.get(session.current_node_id)
        if node is not None and node.safe_return:
            return session.current_node_id
        return "town_gate"
