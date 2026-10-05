"""Neutral observation schema and CP2 feature audit for outcome-ranking research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from game.combat.combat_state import CombatState, LifeState, Tag
from game.combat.damage_range import (
    projected_damage_range,
    skill_base_damage_max,
    skill_base_damage_min,
)
from game.combat.enemy_decision import EnemyDecisionCandidate, EnemyDecisionRuntimeContext
from game.combat.formation import FormationSlot
from game.content.definitions import GameDefinitions

CP3_FEATURE_SCHEMA_ID = "enemy-outcome-candidate-v1"
CP3_FEATURE_SCHEMA_VERSION = 1
FeatureClass = Literal["neutral_factual", "derived_factual", "prescriptive_excluded"]


@dataclass(frozen=True)
class FeatureAuditEntry:
    name: str
    classification: FeatureClass
    rationale: str


CP2_FEATURE_AUDIT: tuple[FeatureAuditEntry, ...] = (
    FeatureAuditEntry(
        "formation", "prescriptive_excluded", "Weighted formation preference, not raw geometry."
    ),
    FeatureAuditEntry(
        "marked_focus", "prescriptive_excluded", "Authored bonus for focusing Marked targets."
    ),
    FeatureAuditEntry(
        "damage_pressure",
        "prescriptive_excluded",
        "Composite heuristic score mixing damage and tactical bonuses.",
    ),
    FeatureAuditEntry(
        "boss_pressure", "prescriptive_excluded", "Boss-package heuristic preference."
    ),
    FeatureAuditEntry("guard", "prescriptive_excluded", "Authored guard preference score."),
    FeatureAuditEntry(
        "vulnerable_payoff", "prescriptive_excluded", "Authored payoff preference score."
    ),
    FeatureAuditEntry(
        "maw_grab_setup", "prescriptive_excluded", "Designer-authored Maw setup value."
    ),
    FeatureAuditEntry(
        "maw_bite_payoff", "prescriptive_excluded", "Designer-authored Maw payoff value."
    ),
    FeatureAuditEntry(
        "target_has_heal_skill",
        "derived_factual",
        "Target capability derived from currently authored legal skills.",
    ),
    FeatureAuditEntry(
        "target_has_recovery_skill",
        "derived_factual",
        "Target recovery capability derived from current skills.",
    ),
    FeatureAuditEntry(
        "target_has_support_skill",
        "derived_factual",
        "Target support capability derived from current skills.",
    ),
    FeatureAuditEntry(
        "target_has_effort_for_support",
        "derived_factual",
        "Current resource sufficiency for a support action.",
    ),
    FeatureAuditEntry(
        "target_is_support_actor",
        "prescriptive_excluded",
        "Role label collapses capabilities into an authored archetype.",
    ),
    FeatureAuditEntry(
        "target_is_backline_support",
        "prescriptive_excluded",
        "Role/archetype label combined with preferred positioning.",
    ),
    FeatureAuditEntry(
        "grab_disrupts_support_actor",
        "prescriptive_excluded",
        "Authored conclusion that a grab is tactically desirable.",
    ),
    FeatureAuditEntry(
        "grab_action_tax_value",
        "prescriptive_excluded",
        "Hand-weighted tactical value of action denial.",
    ),
    FeatureAuditEntry(
        "grab_target_has_not_acted",
        "derived_factual",
        "Candidate-gated fact about target turn order.",
    ),
    FeatureAuditEntry(
        "grab_creates_support_denial",
        "prescriptive_excluded",
        "Authored tactical conclusion rather than raw state.",
    ),
    FeatureAuditEntry(
        "maw_grab_high_value_support",
        "prescriptive_excluded",
        "Designer valuation of a support target.",
    ),
    FeatureAuditEntry(
        "maw_grab_down_threat",
        "prescriptive_excluded",
        "Handcrafted estimate specialized to the Maw combo.",
    ),
    FeatureAuditEntry(
        "maw_grab_bite_expected_collapse",
        "prescriptive_excluded",
        "Designer-authored combo conclusion.",
    ),
    FeatureAuditEntry(
        "boss_guard_package", "prescriptive_excluded", "Designer-authored boss protection package."
    ),
    FeatureAuditEntry(
        "bandit_mark_collapse", "prescriptive_excluded", "Handcrafted Mark package preference."
    ),
    FeatureAuditEntry(
        "bandit_mark_kill_lane", "prescriptive_excluded", "Handcrafted Mark package preference."
    ),
    FeatureAuditEntry(
        "mark_ally_reach_count", "derived_factual", "Count of allies that could reach the target."
    ),
    FeatureAuditEntry(
        "mark_expected_followup_damage",
        "derived_factual",
        "Simulator-derived estimate of follow-up damage.",
    ),
    FeatureAuditEntry(
        "mark_payoff_attacker_count",
        "derived_factual",
        "Count of allies with relevant payoff capability.",
    ),
    FeatureAuditEntry(
        "mark_acts_before_target_count",
        "derived_factual",
        "Turn-order fact for potential follow-up actors.",
    ),
    FeatureAuditEntry(
        "mark_followup_can_down",
        "derived_factual",
        "Derived consequence estimate without a style bonus.",
    ),
    FeatureAuditEntry(
        "mark_target_low_hp",
        "derived_factual",
        "Thresholded target-state fact; raw HP is preferred in CP3.",
    ),
    FeatureAuditEntry("mark_target_guarded", "derived_factual", "Current target status fact."),
    FeatureAuditEntry(
        "mark_target_protected", "derived_factual", "Current formation-protection fact."
    ),
    FeatureAuditEntry(
        "bandit_marked_attack",
        "prescriptive_excluded",
        "Authored reward for attacking a marked target.",
    ),
    FeatureAuditEntry(
        "bandit_marked_payoff", "prescriptive_excluded", "Authored Mark payoff preference."
    ),
    FeatureAuditEntry(
        "bandit_ignored_marked_legal",
        "prescriptive_excluded",
        "Explicit penalty signal for violating authored behavior.",
    ),
    FeatureAuditEntry(
        "effort_drain",
        "prescriptive_excluded",
        "Weighted preference; CP3 exposes effect tag and resources directly.",
    ),
    FeatureAuditEntry(
        "mark", "prescriptive_excluded", "Composite authored Mark desirability score."
    ),
    FeatureAuditEntry(
        "drag_forward",
        "prescriptive_excluded",
        "Composite authored Drag Forward desirability score.",
    ),
    FeatureAuditEntry(
        "role_flavor", "prescriptive_excluded", "Explicit archetype-flavor preference."
    ),
)

_MECHANIC_SKILL_TAGS = (
    "drag_forward",
    "effort_drain",
    "formation",
    "guard",
    "mark",
    "support",
    "vulnerable_bonus",
)
_STATUS_TAGS = tuple(Tag)
_SLOTS = tuple(FormationSlot)
_CATEGORIES = ("basic", "special", "support")
_ATTACK_TYPES = ("melee", "ranged", "reach", "magic")

CP3_FEATURE_NAMES: tuple[str, ...] = (
    "round_number",
    "living_hero_count",
    "living_enemy_count",
    "party_hp_total",
    "party_effort_total",
    "enemy_hp_total",
    "enemy_effort_total",
    "marked_hero_count",
    "remaining_hero_activations",
    "remaining_enemy_activations",
    "actor_hp",
    "actor_max_hp",
    "actor_effort",
    "actor_max_effort",
    "actor_speed",
    "actor_accuracy",
    "actor_defense",
    "actor_damage",
    "actor_mortal_wounds",
    "actor_morale",
    "actor_strain",
    *(f"actor_slot_{slot.value.lower()}" for slot in _SLOTS),
    "target_is_hero",
    "target_hp",
    "target_max_hp",
    "target_effort",
    "target_max_effort",
    "target_speed",
    "target_accuracy",
    "target_defense",
    "target_damage",
    "target_mortal_wounds",
    "target_morale",
    "target_strain",
    "target_is_alive",
    "target_is_downed",
    "target_is_dead",
    *(f"target_slot_{slot.value.lower()}" for slot in _SLOTS),
    *(f"target_tag_{tag.name.lower()}" for tag in _STATUS_TAGS),
    "target_mark_turns",
    "target_is_protected",
    "target_heal_skill_count",
    "target_support_skill_count",
    "target_recovery_skill_count",
    "target_min_support_effort_cost",
    "target_has_not_acted",
    "skill_effort_cost",
    "skill_accuracy",
    "skill_damage_min",
    "skill_damage_max",
    "projected_damage_min",
    "projected_damage_max",
    *(f"skill_category_{category}" for category in _CATEGORIES),
    *(f"skill_attack_type_{attack_type}" for attack_type in _ATTACK_TYPES),
    "skill_front_only",
    "skill_reaction_window",
    *(f"skill_tag_{tag}" for tag in _MECHANIC_SKILL_TAGS),
)


CP3_DERIVED_FEATURE_NAMES: tuple[str, ...] = (
    "living_hero_count",
    "living_enemy_count",
    "party_hp_total",
    "party_effort_total",
    "enemy_hp_total",
    "enemy_effort_total",
    "marked_hero_count",
    "remaining_hero_activations",
    "remaining_enemy_activations",
    "target_is_protected",
    "target_heal_skill_count",
    "target_support_skill_count",
    "target_recovery_skill_count",
    "target_min_support_effort_cost",
    "target_has_not_acted",
    "projected_damage_min",
    "projected_damage_max",
)
CP3_NEUTRAL_FACTUAL_FEATURE_NAMES: tuple[str, ...] = tuple(
    name for name in CP3_FEATURE_NAMES if name not in CP3_DERIVED_FEATURE_NAMES
)


@dataclass(frozen=True)
class OutcomeFeatureSchema:
    schema_id: str = CP3_FEATURE_SCHEMA_ID
    version: int = CP3_FEATURE_SCHEMA_VERSION
    names: tuple[str, ...] = CP3_FEATURE_NAMES

    def encode(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        candidate: EnemyDecisionCandidate,
        runtime: EnemyDecisionRuntimeContext,
    ) -> tuple[float, ...]:
        values = _feature_mapping(state, definitions, candidate, runtime)
        actual = set(values)
        expected = set(self.names)
        if actual != expected:
            missing = tuple(name for name in self.names if name not in actual)
            extra = tuple(sorted(actual - expected))
            raise ValueError(f"CP3 feature schema mismatch: missing={missing}, extra={extra}")
        return tuple(float(values[name]) for name in self.names)


CP3_FEATURE_SCHEMA = OutcomeFeatureSchema()


def _feature_mapping(
    state: CombatState,
    definitions: GameDefinitions,
    candidate: EnemyDecisionCandidate,
    runtime: EnemyDecisionRuntimeContext,
) -> dict[str, int]:
    enemy_id = _candidate_actor_id(state, definitions, candidate, runtime)
    enemy = state.actor(enemy_id)
    target = state.actor(candidate.target_id)
    skill = definitions.skills[candidate.skill_id]
    projected_min, projected_max = projected_damage_range(
        state, skill, enemy, target, set(skill.tags)
    )
    remaining_ids = runtime.initiative_actor_ids[runtime.current_turn_index + 1 :]
    target_skills = [definitions.skills[skill_id] for skill_id in target.skills]
    support_skills = [
        item
        for item in target_skills
        if item.category == "support" or "support" in item.tags or "rally" in item.tags
    ]
    heal_skills = [
        item for item in target_skills if "treatment" in item.tags or "brink_heal" in item.tags
    ]
    recovery_skills = [item for item in target_skills if "effort" in item.tags]
    min_support_effort = min((item.effort_cost for item in support_skills), default=0)
    target_formation = state.formation_for(target.team)
    target_protected = int(
        target_formation.is_protected(target.actor_id, state.side_for(target.team))
    )
    values: dict[str, int] = {
        "round_number": state.round_number,
        "living_hero_count": len(state.living_heroes()),
        "living_enemy_count": len(state.living_enemies()),
        "party_hp_total": sum(max(0, hero.hp) for hero in state.heroes.values()),
        "party_effort_total": sum(max(0, hero.effort) for hero in state.heroes.values()),
        "enemy_hp_total": sum(max(0, item.hp) for item in state.enemies.values()),
        "enemy_effort_total": sum(max(0, item.effort) for item in state.enemies.values()),
        "marked_hero_count": sum(Tag.MARKED in hero.tags for hero in state.heroes.values()),
        "remaining_hero_activations": sum(actor_id in state.heroes for actor_id in remaining_ids),
        "remaining_enemy_activations": sum(actor_id in state.enemies for actor_id in remaining_ids),
        "actor_hp": enemy.hp,
        "actor_max_hp": enemy.max_hp,
        "actor_effort": enemy.effort,
        "actor_max_effort": enemy.max_effort,
        "actor_speed": enemy.speed,
        "actor_accuracy": enemy.accuracy,
        "actor_defense": enemy.defense,
        "actor_damage": enemy.damage,
        "actor_mortal_wounds": enemy.mortal_wounds,
        "actor_morale": int(enemy.morale.value),
        "actor_strain": int(enemy.strain.value),
        "target_is_hero": int(candidate.target_id in state.heroes),
        "target_hp": target.hp,
        "target_max_hp": target.max_hp,
        "target_effort": target.effort,
        "target_max_effort": target.max_effort,
        "target_speed": target.speed,
        "target_accuracy": target.accuracy,
        "target_defense": target.defense,
        "target_damage": target.damage,
        "target_mortal_wounds": target.mortal_wounds,
        "target_morale": int(target.morale.value),
        "target_strain": int(target.strain.value),
        "target_is_alive": int(target.life_state == LifeState.ALIVE),
        "target_is_downed": int(target.life_state == LifeState.DOWNED),
        "target_is_dead": int(target.life_state == LifeState.DEAD),
        "target_mark_turns": target.tag_turns.get(Tag.MARKED, 0),
        "target_is_protected": target_protected,
        "target_heal_skill_count": len(heal_skills),
        "target_support_skill_count": len(support_skills),
        "target_recovery_skill_count": len(recovery_skills),
        "target_min_support_effort_cost": min_support_effort,
        "target_has_not_acted": int(candidate.target_id in remaining_ids),
        "skill_effort_cost": skill.effort_cost,
        "skill_accuracy": skill.accuracy,
        "skill_damage_min": skill_base_damage_min(skill),
        "skill_damage_max": skill_base_damage_max(skill),
        "projected_damage_min": projected_min,
        "projected_damage_max": projected_max,
        "skill_front_only": int(str(skill.usable_from) == "front_only"),
        "skill_reaction_window": int(skill.reaction_window),
    }
    for slot in _SLOTS:
        values[f"actor_slot_{slot.value.lower()}"] = int(enemy.formation_slot == slot)
        values[f"target_slot_{slot.value.lower()}"] = int(target.formation_slot == slot)
    for status_tag in _STATUS_TAGS:
        values[f"target_tag_{status_tag.name.lower()}"] = int(status_tag in target.tags)
    for category in _CATEGORIES:
        values[f"skill_category_{category}"] = int(skill.category == category)
    attack_type_value = str(skill.attack_type)
    for attack_type in _ATTACK_TYPES:
        values[f"skill_attack_type_{attack_type}"] = int(attack_type_value == attack_type)
    for mechanic_tag in _MECHANIC_SKILL_TAGS:
        values[f"skill_tag_{mechanic_tag}"] = int(mechanic_tag in skill.tags)
    return values


def _candidate_actor_id(
    state: CombatState,
    definitions: GameDefinitions,
    candidate: EnemyDecisionCandidate,
    runtime: EnemyDecisionRuntimeContext,
) -> str:
    del definitions, candidate
    if 0 <= runtime.current_turn_index < len(runtime.initiative_actor_ids):
        actor_id = runtime.initiative_actor_ids[runtime.current_turn_index]
        if actor_id in state.enemies:
            return actor_id
    raise ValueError("CP3 observation could not identify the acting enemy from runtime context.")
