"""Research-only mechanic perturbations for CP8 route balance analysis.

Production modules do not import this file.  Each condition is evaluated in an
isolated worker process so temporary hard-coded mechanic overrides cannot leak
between variants or into normal gameplay.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any

import game.combat.actions as combat_actions
import game.combat.damage as combat_damage
import game.combat.damage_range as damage_range
import game.combat.enemy_learning as enemy_learning
from game.combat.combat_state import Combatant, CombatState, Tag, Team
from game.combat.enemy_decision import (
    EnemyDecisionCandidate,
    EnemyDecisionPolicy,
    EnemyDecisionRuntimeContext,
    explain_enemy_decision,
)
from game.content.definitions import GameDefinitions
from game.core.events import CombatEffectEvent, MoveEvent
from game.ml.counterfactual import decision_state_digest
from game.ml.outcome_features import CP3_FEATURE_NAMES, CP3_FEATURE_SCHEMA


@dataclass(frozen=True)
class PerturbationSpec:
    condition_id: str
    mechanic: str
    hypothesis: str
    description: str
    observability: str
    observable_feature_changes: tuple[str, ...] = ()
    enemy_mark_cost: int | None = None
    enemy_guard_cost: int | None = None
    effort_drain_cost: int | None = None
    drag_damage_delta: int = 0
    marked_vulnerable_bonus_delta: int = 0
    drag_mark_turns: int | None = None
    enemy_guard_mitigation: int | None = None


PERTURBATIONS: dict[str, PerturbationSpec] = {
    "BASELINE": PerturbationSpec(
        "BASELINE",
        "BASELINE",
        "production baseline",
        "Exact production mechanics.",
        "FULLY_OBSERVABLE_TO_FROZEN_POLICY",
    ),
    "M1": PerturbationSpec(
        "M1",
        "MARK",
        "payoff_strength",
        "Enemy vulnerable-bonus attacks gain +1 additional damage specifically "
        "against Marked targets.",
        "FULLY_OBSERVABLE_TO_FROZEN_POLICY",
        ("projected_damage_min", "projected_damage_max"),
        marked_vulnerable_bonus_delta=1,
    ),
    "M2": PerturbationSpec(
        "M2",
        "MARK",
        "setup_economics",
        "Enemy Mark-tagged skills cost 0 Effort instead of 1.",
        "FULLY_OBSERVABLE_TO_FROZEN_POLICY",
        ("skill_effort_cost",),
        enemy_mark_cost=0,
    ),
    "G1": PerturbationSpec(
        "G1",
        "GRAB",
        "immediate_value",
        "Enemy drag-forward skills gain +1 base damage without changing formation movement.",
        "FULLY_OBSERVABLE_TO_FROZEN_POLICY",
        ("skill_damage_min", "skill_damage_max", "projected_damage_min", "projected_damage_max"),
        drag_damage_delta=1,
    ),
    "G2": PerturbationSpec(
        "G2",
        "GRAB",
        "positional_payoff",
        "A Mark applied by an enemy drag lasts 3 target turns instead of 2.",
        "NOT_OBSERVABLE",
        (),
        drag_mark_turns=3,
    ),
    "D1": PerturbationSpec(
        "D1",
        "GUARD",
        "mitigation_value",
        "Guard on enemies prevents up to 4 damage on the consuming hit instead of 3.",
        "NOT_OBSERVABLE",
        (),
        enemy_guard_mitigation=4,
    ),
    "D2": PerturbationSpec(
        "D2",
        "GUARD",
        "economics",
        "Enemy Guard-tagged support skills cost 0 Effort instead of 1.",
        "FULLY_OBSERVABLE_TO_FROZEN_POLICY",
        ("skill_effort_cost",),
        enemy_guard_cost=0,
    ),
    "E1": PerturbationSpec(
        "E1",
        "EFFORT_DRAIN",
        "economics",
        "Effort Drain costs the enemy 0 Effort instead of 1; drain amount remains 1.",
        "FULLY_OBSERVABLE_TO_FROZEN_POLICY",
        ("skill_effort_cost",),
        effort_drain_cost=0,
    ),
}

CONDITION_IDS = tuple(PERTURBATIONS)
MECHANIC_CONDITIONS = {
    "MARK": ("M1", "M2"),
    "GRAB": ("G1", "G2"),
    "GUARD": ("D1", "D2"),
    "EFFORT_DRAIN": ("E1",),
}

_current_skill: ContextVar[tuple[str, Team] | None] = ContextVar("cp8_current_skill", default=None)


def cloned_definitions(definitions: GameDefinitions, spec: PerturbationSpec) -> GameDefinitions:
    """Return an isolated definitions copy with only data-driven CP8 edits."""
    cloned = copy.deepcopy(definitions)
    for skill_id, skill in list(cloned.skills.items()):
        tags = set(skill.tags)
        update: dict[str, Any] = {}
        if spec.enemy_mark_cost is not None and "enemy" in tags and "mark" in tags:
            update["effort_cost"] = spec.enemy_mark_cost
        if spec.enemy_guard_cost is not None and "enemy" in tags and "guard" in tags:
            update["effort_cost"] = spec.enemy_guard_cost
        if spec.effort_drain_cost is not None and "enemy" in tags and "effort_drain" in tags:
            update["effort_cost"] = spec.effort_drain_cost
        if spec.drag_damage_delta and "enemy" in tags and "drag_forward" in tags:
            low = skill.damage if skill.damage_min is None else skill.damage_min
            high = skill.damage if skill.damage_max is None else skill.damage_max
            update.update(
                damage=skill.damage + spec.drag_damage_delta,
                damage_min=low + spec.drag_damage_delta,
                damage_max=high + spec.drag_damage_delta,
            )
        if update:
            cloned.skills[skill_id] = skill.model_copy(update=update)
    return cloned


def _slot(state: CombatState, actor_id: str) -> str | None:
    actor = state.actor(actor_id)
    slot = state.formation_for(actor.team).slot_of(actor_id)
    return slot.value if slot is not None else None


def _protected(state: CombatState, actor_id: str) -> bool:
    actor = state.actor(actor_id)
    formation = state.formation_for(actor.team)
    return formation.is_protected(actor_id, state.side_for(actor.team))


def _event_payload(event: object) -> dict[str, Any]:
    row: dict[str, Any] = {"type": type(event).__name__}
    for name in ("actor_id", "source_id", "target_id", "status", "added", "amount", "delta"):
        if hasattr(event, name):
            value = getattr(event, name)
            row[name] = value.value if hasattr(value, "value") else value
    if isinstance(event, MoveEvent):
        row["from_slot"] = event.from_slot
        row["to_slot"] = event.to_slot
    return row


class ResolutionRecorder:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record_use_skill(
        self,
        original: Callable[..., Any],
        state: CombatState,
        actor_id: str,
        skill: Any,
        target_id: str,
        rng: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        actor = state.actor(actor_id)
        target = state.actor(target_id)
        before: dict[str, Any] = {
            "actor_effort": actor.effort,
            "target_hp": target.hp,
            "target_effort": target.effort,
            "target_life_state": target.life_state.value,
            "target_marked": Tag.MARKED in target.tags,
            "target_mark_turns": target.tag_turns.get(Tag.MARKED, 0),
            "target_guarded": Tag.GUARDED in target.tags,
            "target_slot": _slot(state, target_id),
            "target_protected": _protected(state, target_id),
        }
        token = _current_skill.set((str(skill.id), actor.team))
        try:
            result = original(state, actor_id, skill, target_id, rng, *args, **kwargs)
        finally:
            _current_skill.reset(token)
        after_target = state.actor(target_id)
        events = list(getattr(result, "events", ()))
        mitigation = sum(
            int(getattr(event, "amount", 0))
            for event in events
            if isinstance(event, CombatEffectEvent) and event.effect_type == "mitigation"
        )
        self.rows.append(
            {
                "sequence": len(self.rows),
                "round": state.round_number,
                "actor_id": actor_id,
                "actor_team": actor.team.value,
                "actor_class": actor.class_id,
                "skill_id": str(skill.id),
                "skill_tags": sorted(str(tag) for tag in skill.tags),
                "target_id": target_id,
                "target_team": target.team.value,
                "target_class": target.class_id,
                "before": before,
                "after": {
                    "actor_effort": state.actor(actor_id).effort,
                    "target_hp": after_target.hp,
                    "target_effort": after_target.effort,
                    "target_life_state": after_target.life_state.value,
                    "target_marked": Tag.MARKED in after_target.tags,
                    "target_mark_turns": after_target.tag_turns.get(Tag.MARKED, 0),
                    "target_guarded": Tag.GUARDED in after_target.tags,
                    "target_slot": _slot(state, target_id),
                    "target_protected": _protected(state, target_id),
                },
                "damage_dealt": max(0, int(before["target_hp"]) - after_target.hp),
                "effort_removed": max(0, int(before["target_effort"]) - after_target.effort),
                "guard_damage_prevented": mitigation,
                "events": [_event_payload(event) for event in events],
                "success": bool(getattr(result, "success", True)),
            }
        )
        return result


class MechanicOverrideContext(AbstractContextManager["MechanicOverrideContext"]):
    """Process-local temporary overrides for hard-coded CP8 mechanics."""

    def __init__(self, spec: PerturbationSpec, recorder: ResolutionRecorder) -> None:
        self.spec = spec
        self.recorder = recorder
        self._original_use_skill = enemy_learning.use_skill
        self._original_actions_use_skill = combat_actions.use_skill
        self._original_actions_bonus = combat_actions.skill_damage_bonus
        self._original_range_bonus = damage_range.skill_damage_bonus
        self._original_apply_marked = combat_actions.apply_marked
        self._original_apply_damage = combat_actions.apply_damage

    def __enter__(self) -> MechanicOverrideContext:
        spec = self.spec
        original_bonus = self._original_range_bonus
        original_mark = self._original_apply_marked
        original_damage = self._original_apply_damage

        def bonus(state: CombatState, actor: Any, target: Any, skill_tags: Any) -> int:
            value = original_bonus(state, actor, target, skill_tags)
            if (
                spec.marked_vulnerable_bonus_delta
                and actor.team == Team.ENEMY
                and "vulnerable_bonus" in skill_tags
                and Tag.MARKED in target.tags
            ):
                value += spec.marked_vulnerable_bonus_delta
            return value

        def apply_marked(combatant: Combatant, turns: int = 2) -> None:
            current = _current_skill.get()
            if current is not None:
                skill_id, team = current
                if (
                    spec.drag_mark_turns is not None
                    and team == Team.ENEMY
                    and skill_id in {"drag_forward", "stalker_hook"}
                ):
                    turns = spec.drag_mark_turns
            original_mark(combatant, turns)

        def apply_damage(state: CombatState, source_id: str, target_id: str, amount: int) -> Any:
            target = state.actor(target_id)
            if (
                spec.enemy_guard_mitigation is not None
                and target.team == Team.ENEMY
                and Tag.GUARDED in target.tags
            ):
                old = combat_damage.GUARDED_DAMAGE_REDUCTION
                combat_damage.GUARDED_DAMAGE_REDUCTION = spec.enemy_guard_mitigation
                try:
                    return original_damage(state, source_id, target_id, amount)
                finally:
                    combat_damage.GUARDED_DAMAGE_REDUCTION = old
            return original_damage(state, source_id, target_id, amount)

        def use_skill(*args: Any, **kwargs: Any) -> Any:
            return self.recorder.record_use_skill(self._original_use_skill, *args, **kwargs)

        damage_range.skill_damage_bonus = bonus
        combat_actions.skill_damage_bonus = bonus
        combat_actions.apply_marked = apply_marked
        combat_actions.apply_damage = apply_damage
        combat_actions.use_skill = use_skill
        enemy_learning.use_skill = use_skill
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        damage_range.skill_damage_bonus = self._original_range_bonus
        combat_actions.skill_damage_bonus = self._original_actions_bonus
        combat_actions.apply_marked = self._original_apply_marked
        combat_actions.apply_damage = self._original_apply_damage
        combat_actions.use_skill = self._original_actions_use_skill
        enemy_learning.use_skill = self._original_use_skill
        return None


_DEFAULT_RUNTIME_CONTEXT = EnemyDecisionRuntimeContext()


class PerturbationAuditPolicy:
    """Record legal opportunities and the frozen policy's actual choice."""

    def __init__(self, policy_name: str, policy: EnemyDecisionPolicy) -> None:
        self.policy_name = policy_name
        self.policy = policy
        self.decisions: list[dict[str, Any]] = []

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME_CONTEXT,
    ) -> EnemyDecisionCandidate | None:
        trace = explain_enemy_decision(state, definitions, enemy_id, runtime_context)
        chosen = self.policy.choose(state, definitions, enemy_id, runtime_context)
        if chosen is None:
            return None
        if trace is None or not trace.candidates:
            raise ValueError("Frozen policy chose without a legal decision trace.")
        keys = {(candidate.skill_id, candidate.target_id) for candidate in trace.candidates}
        if (chosen.skill_id, chosen.target_id) not in keys:
            raise ValueError("Frozen policy selected outside simulator legality.")
        candidate_rows: list[dict[str, Any]] = []
        for candidate in trace.candidates:
            encoded = CP3_FEATURE_SCHEMA.encode(state, definitions, candidate, runtime_context)
            obs = dict(zip(CP3_FEATURE_NAMES, (float(value) for value in encoded), strict=True))
            candidate_rows.append(
                {
                    "skill_id": candidate.skill_id,
                    "target_id": candidate.target_id,
                    "skill_tags": sorted(candidate.skill_tags),
                    "projected_damage_min": obs["projected_damage_min"],
                    "projected_damage_max": obs["projected_damage_max"],
                    "skill_effort_cost": obs["skill_effort_cost"],
                    "target_hp": obs["target_hp"],
                    "target_effort": obs["target_effort"],
                    "target_mark_turns": obs["target_mark_turns"],
                    "target_protected": bool(obs["target_is_protected"]),
                    "target_slot": next(
                        (
                            slot
                            for slot in ("front_left", "front_right", "back_left", "back_right")
                            if obs[f"target_slot_{slot}"] > 0.5
                        ),
                        "none",
                    ),
                }
            )
        direct = [
            row
            for row in candidate_rows
            if float(row["projected_damage_max"]) > 0
            and row["skill_id"] not in {"drag_forward", "stalker_hook", "effort_drain"}
            and "mark" not in row["skill_tags"]
            and "guard" not in row["skill_tags"]
        ]
        best_direct = max(
            direct,
            key=lambda row: (
                float(row["projected_damage_max"]),
                -float(row["skill_effort_cost"]),
                str(row["skill_id"]),
                str(row["target_id"]),
            ),
            default=None,
        )
        chosen_row = next(
            row
            for row in candidate_rows
            if row["skill_id"] == chosen.skill_id and row["target_id"] == chosen.target_id
        )
        self.decisions.append(
            {
                "action_index": len(self.decisions),
                "round": state.round_number,
                "state_digest": decision_state_digest(state, runtime_context),
                "enemy_id": enemy_id,
                "enemy_class": state.actor(enemy_id).class_id,
                "chosen_skill": chosen.skill_id,
                "chosen_target": chosen.target_id,
                "chosen_skill_tags": sorted(chosen.skill_tags),
                "chosen_projected_damage_max": chosen_row["projected_damage_max"],
                "chosen_target_class": state.actor(chosen.target_id).class_id,
                "chosen_target_hp": chosen_row["target_hp"],
                "chosen_target_effort": chosen_row["target_effort"],
                "chosen_target_protected": chosen_row["target_protected"],
                "chosen_target_slot": chosen_row["target_slot"],
                "actor_hp": state.actor(enemy_id).hp,
                "actor_max_hp": state.actor(enemy_id).max_hp,
                "legal_candidates": candidate_rows,
                "legal_mark": any("mark" in row["skill_tags"] for row in candidate_rows),
                "legal_grab": any("drag_forward" in row["skill_tags"] for row in candidate_rows),
                "legal_guard": any("guard" in row["skill_tags"] for row in candidate_rows),
                "legal_effort_drain": any(
                    "effort_drain" in row["skill_tags"] for row in candidate_rows
                ),
                "selected_mark": "mark" in chosen.skill_tags,
                "selected_grab": "drag_forward" in chosen.skill_tags,
                "selected_guard": "guard" in chosen.skill_tags,
                "selected_effort_drain": "effort_drain" in chosen.skill_tags,
                "best_direct_alternative": best_direct,
                "party_hp_total": sum(max(0, hero.hp) for hero in state.heroes.values()),
                "party_max_hp_total": sum(hero.max_hp for hero in state.heroes.values()),
                "party_effort_total": sum(max(0, hero.effort) for hero in state.heroes.values()),
                "party_max_effort_total": sum(hero.max_effort for hero in state.heroes.values()),
                "marked_hero_count": sum(Tag.MARKED in hero.tags for hero in state.heroes.values()),
            }
        )
        return chosen


def perturbation_matrix_payload() -> dict[str, Any]:
    return {
        "conditions": {key: asdict(value) for key, value in PERTURBATIONS.items()},
        "no_combined_primary_condition": True,
        "direct_pressure_perturbed": False,
    }


def run_perturbed_route(
    production_definitions: GameDefinitions,
    route: Any,
    *,
    root_seed: int,
    enemy_policy_name: str,
    enemy_policy: EnemyDecisionPolicy,
    hero_policy_id: str,
    spec: PerturbationSpec,
    safety_budget: Any,
) -> dict[str, Any]:
    """Run one stateful route under one immutable research perturbation."""
    from game.campaign.company import create_new_company
    from game.campaign.roster import sync_company_from_combat
    from game.combat.enemy_learning import create_hero_policy
    from game.core.rng import GameRng
    from game.expedition.cave import create_encounter_combat
    from game.ml.route_termination import (
        CombatTerminationReason,
        RouteTerminalStatus,
        run_research_combat_with_termination,
    )
    from game.ml.stateful_route import (
        derive_encounter_seed,
        snapshot_combat_entry,
        snapshot_company,
    )

    definitions = cloned_definitions(production_definitions, spec)
    company = create_new_company(definitions)
    initial = snapshot_company(company, definitions)
    encounters: list[dict[str, Any]] = []
    completed_count = 0
    terminal_status: Any = None
    stopped_at: str | None = None

    for encounter_index, encounter_id in enumerate(route.encounter_ids):
        company_before = snapshot_company(company, definitions)
        combat = create_encounter_combat(company, definitions, encounter_id)
        combat_entry = snapshot_combat_entry(combat)
        seed_namespace = route.seed_namespace or route.route_id
        seed_index = encounter_index
        if route.seed_by_encounter_identity:
            seed_index = sum(
                1 for prior in route.encounter_ids[:encounter_index] if prior == encounter_id
            )
        encounter_seed = derive_encounter_seed(
            root_seed,
            route_id=seed_namespace,
            encounter_index=seed_index,
            encounter_id=encounter_id,
            hero_policy_id=hero_policy_id,
        )
        audit_policy = PerturbationAuditPolicy(enemy_policy_name, enemy_policy)
        recorder = ResolutionRecorder()
        with MechanicOverrideContext(spec, recorder):
            combat_result = run_research_combat_with_termination(
                combat,
                definitions,
                GameRng(encounter_seed),
                enemy_policy_name=enemy_policy_name,
                enemy_policy=audit_policy,
                hero_policy=create_hero_policy(
                    hero_policy_id,
                    encounter_id=encounter_id,
                    seed=encounter_seed,
                ),
                encounter_id=encounter_id,
                encounter_name=encounter_id.replace("_", " ").title(),
                safety_budget=safety_budget,
                enemy_wait_mode="none",
                enemy_movement_mode="recovery_only",
            )

        terminal = combat_result.termination_reason
        company_after: dict[str, Any] | None = None
        deterioration: dict[str, int] | None = None
        if terminal in {
            CombatTerminationReason.HERO_VICTORY,
            CombatTerminationReason.ENEMY_VICTORY,
            CombatTerminationReason.PRODUCTION_RETREAT,
        }:
            sync_company_from_combat(company, combat.heroes, combat.party_formation)
            company_after = snapshot_company(company, definitions)
            deterioration = {
                "hp_loss": int(combat_entry["hp_total"]) - int(company_after["hp_total"]),
                "effort_loss": int(combat_entry["effort_total"])
                - int(company_after["effort_total"]),
                "downed_delta": int(company_after["downed_count"])
                - int(combat_entry["downed_count"]),
                "mortal_wounds_delta": int(company_after["mortal_wounds_total"])
                - int(combat_entry["mortal_wounds_total"]),
                "active_hero_delta": int(company_after["active_hero_count"])
                - int(combat_entry["active_hero_count"]),
            }

        resolution_rows = [
            row for row in recorder.rows if str(row["actor_team"]) == Team.ENEMY.value
        ]
        down_events = sum(
            event.get("type") == "DownedEvent" for row in resolution_rows for event in row["events"]
        )
        mortal_events = sum(
            event.get("type") == "StatusChangedEvent"
            and event.get("status") == "mortal_wound"
            and bool(event.get("added"))
            for row in resolution_rows
            for event in row["events"]
        )
        encounters.append(
            {
                "encounter_index": encounter_index,
                "route_length": len(route.encounter_ids),
                "encounter_id": encounter_id,
                "encounter_seed": encounter_seed,
                "company_before_create": company_before,
                "combat_entry": combat_entry,
                "termination_reason": terminal.value,
                "production_terminal": combat_result.is_production_terminal,
                "censored": combat_result.is_censored,
                "company_after_sync": company_after,
                "deterioration": deterioration,
                "rounds": combat_result.rounds_elapsed,
                "transitions": combat_result.transitions,
                "enemy_guarded_at_end": sum(
                    Tag.GUARDED in enemy.tags
                    for enemy in combat.enemies.values()
                    if enemy.is_alive()
                ),
                "downs_caused": down_events,
                "mortal_wounds_caused": mortal_events,
                "decision_audit": audit_policy.decisions,
                "resolution_audit": recorder.rows,
                "enemy_action_sequence": [
                    list(item) for item in combat_result.enemy_action_sequence
                ],
                "hero_action_sequence": [list(item) for item in combat_result.hero_action_sequence],
                "enemy_event_digest": combat_result.enemy_event_digest,
                "final_state_digest": combat_result.final_state_digest,
                "final_rng_digest": combat_result.final_rng_digest,
            }
        )

        if terminal == CombatTerminationReason.HERO_VICTORY:
            completed_count += 1
            continue
        stopped_at = encounter_id
        if terminal == CombatTerminationReason.ENEMY_VICTORY:
            terminal_status = RouteTerminalStatus.ROUTE_FAILED_ENEMY_VICTORY
        elif terminal == CombatTerminationReason.PRODUCTION_RETREAT:
            terminal_status = RouteTerminalStatus.ROUTE_ENDED_PRODUCTION_RETREAT
        elif terminal == CombatTerminationReason.DETECTED_CYCLE:
            terminal_status = RouteTerminalStatus.ROUTE_CENSORED_CYCLE
        elif terminal == CombatTerminationReason.SAFETY_BOUND_EXCEEDED:
            terminal_status = RouteTerminalStatus.ROUTE_CENSORED_SAFETY_BOUND
        else:
            terminal_status = RouteTerminalStatus.ROUTE_INVALID
        break

    if terminal_status is None:
        terminal_status = RouteTerminalStatus.ROUTE_COMPLETED
    final_company = snapshot_company(company, definitions)
    return {
        "condition_id": spec.condition_id,
        "mechanic": spec.mechanic,
        "case_id": f"{route.route_id}|{hero_policy_id}|{root_seed}",
        "route_id": route.route_id,
        "route_structure": route.structure,
        "route_length": len(route.encounter_ids),
        "root_seed": root_seed,
        "enemy_policy": enemy_policy_name,
        "hero_policy_id": hero_policy_id,
        "encounter_sequence": list(route.encounter_ids),
        "route_terminal_status": terminal_status.value,
        "route_completed": terminal_status == RouteTerminalStatus.ROUTE_COMPLETED,
        "route_failed_enemy_victory": (
            terminal_status == RouteTerminalStatus.ROUTE_FAILED_ENEMY_VICTORY
        ),
        "route_censored": terminal_status
        in {
            RouteTerminalStatus.ROUTE_CENSORED_CYCLE,
            RouteTerminalStatus.ROUTE_CENSORED_SAFETY_BOUND,
        },
        "route_invalid": terminal_status == RouteTerminalStatus.ROUTE_INVALID,
        "encounters_completed": completed_count,
        "stopped_at_encounter_id": stopped_at,
        "initial_state": initial,
        "final_company_state": final_company,
        "encounters": encounters,
        "total_rounds": sum(int(row["rounds"]) for row in encounters),
        "total_transitions": sum(int(row["transitions"]) for row in encounters),
    }
