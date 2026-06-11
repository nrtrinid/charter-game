"""Deterministic enemy skill and target selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from game.combat.combat_state import Combatant, CombatState, Tag, Team
from game.combat.damage_range import combatant_damage_max, skill_base_damage_max
from game.combat.formation import FormationSlot, is_back, is_front, lane_of
from game.combat.targeting import can_use_skill_from_position, cover_penalty, legal_targets
from game.content.definitions import GameDefinitions
from game.data.schemas import SkillDefinition

_TARGET_SLOT_ORDER: dict[FormationSlot, int] = {
    FormationSlot.FRONT_LEFT: 0,
    FormationSlot.FRONT_RIGHT: 1,
    FormationSlot.BACK_LEFT: 2,
    FormationSlot.BACK_RIGHT: 3,
}
_UNFORMED_TARGET_ORDER = len(_TARGET_SLOT_ORDER)
_HIGH_VALUE_MARK_CLASSES = {"field_surgeon", "scribe", "cutpurse"}
_BACKLINE_DISRUPTION_CLASSES = {"field_surgeon", "scribe"}
_FRONTLINE_ANCHOR_CLASSES = {"watchman"}
_PAYOFF_TAGS = {"exploit_vulnerable", "vulnerable_bonus"}
_MARKED_FOCUS_SCORE = 8
_MARK_ALLY_ACCESS_SCORE = 8
_LATE_FIGHT_ROUND = 3
_LEARNING_ONLY_FEATURES = {
    "maw_grab_setup",
    "maw_bite_payoff",
    "boss_guard_package",
    "target_has_heal_skill",
    "target_has_recovery_skill",
    "target_has_support_skill",
    "target_has_effort_for_support",
    "target_is_support_actor",
    "target_is_backline_support",
    "grab_disrupts_support_actor",
    "grab_action_tax_value",
    "grab_target_has_not_acted",
    "grab_creates_support_denial",
    "maw_grab_high_value_support",
    "maw_grab_down_threat",
    "maw_grab_bite_expected_collapse",
    "bandit_mark_kill_lane",
    "bandit_marked_attack",
    "bandit_marked_payoff",
    "bandit_ignored_marked_legal",
    "mark_ally_reach_count",
    "mark_expected_followup_damage",
    "mark_payoff_attacker_count",
    "mark_acts_before_target_count",
    "mark_followup_can_down",
    "mark_target_low_hp",
    "mark_target_guarded",
    "mark_target_protected",
}
_PRODUCTION_LEARNED_WEIGHTS: dict[str, int] = {
    "damage_pressure": 60,
    "vulnerable_payoff": 55,
    "bandit_mark_collapse": 36,
    "maw_bite_payoff": 36,
    "drag_forward": 27,
    "boss_pressure": 14,
    "bandit_mark_kill_lane": 21,
    "mark": 12,
    "boss_guard_package": 16,
    "maw_grab_setup": 15,
    "marked_focus": 9,
    "maw_grab_bite_expected_collapse": 15,
    "bandit_marked_attack": 15,
    "bandit_marked_payoff": 15,
    "maw_grab_high_value_support": 15,
    "maw_grab_down_threat": 15,
    "mark_ally_reach_count": 15,
    "mark_expected_followup_damage": 15,
    "mark_payoff_attacker_count": 15,
    "formation": 2,
    "role_flavor": 1,
    "effort_drain": 1,
}
SUPPORTED_PRODUCTION_ENEMY_AI_MODES = ("learned_static", "heuristic")
PRODUCTION_ENEMY_AI_MODE_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "learned_static": "Learned Static",
        "heuristic": "Heuristic",
    }
)
PRODUCTION_ENEMY_AI_WAIT_MODES: Mapping[str, str] = MappingProxyType(
    {
        "learned_static": "package_only",
        "heuristic": "none",
    }
)
PRODUCTION_ENEMY_AI_MOVEMENT_MODES: Mapping[str, str] = MappingProxyType(
    {
        "learned_static": "package_only",
        "heuristic": "recovery_only",
    }
)
PRODUCTION_ENEMY_AI_MODE_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "learned_static": (
            "Uses fixed learned feature weights from the AI lab with package wait/move "
            "timing so enemies can delay for marks and reposition for payoff lanes."
        ),
        "heuristic": (
            "Uses the hand-authored deterministic feature scorer without package "
            "timing; enemies act immediately and only reposition for recovery."
        ),
    }
)


def production_enemy_wait_mode(ai_mode: str) -> str:
    if ai_mode not in SUPPORTED_PRODUCTION_ENEMY_AI_MODES:
        raise ValueError(f"Unknown enemy AI mode: {ai_mode}")
    return PRODUCTION_ENEMY_AI_WAIT_MODES[ai_mode]


def production_enemy_movement_mode(ai_mode: str) -> str:
    if ai_mode not in SUPPORTED_PRODUCTION_ENEMY_AI_MODES:
        raise ValueError(f"Unknown enemy AI mode: {ai_mode}")
    return PRODUCTION_ENEMY_AI_MOVEMENT_MODES[ai_mode]
EnemyDecisionFeatureVector = Mapping[str, int]


@dataclass(frozen=True)
class EnemyDecisionRuntimeContext:
    initiative_actor_ids: tuple[str, ...] = ()
    current_turn_index: int = 0


_DEFAULT_RUNTIME_CONTEXT = EnemyDecisionRuntimeContext()


@dataclass(frozen=True)
class EnemyDecisionContext:
    round_number: int
    living_enemy_count: int
    living_hero_count: int
    ally_count: int
    is_last_enemy: bool
    enemy_low_hp: bool
    allies_have_payoff_skills: bool
    marked_hero_count: int
    wounded_hero_count: int

    @property
    def late_fight(self) -> bool:
        return self.round_number >= _LATE_FIGHT_ROUND


@dataclass(frozen=True)
class BanditMarkCollapseProfile:
    ally_reach_count: int = 0
    expected_followup_damage: int = 0
    payoff_attacker_count: int = 0
    acts_before_target_count: int = 0
    followup_can_down: bool = False
    target_low_hp: bool = False
    target_wounded: bool = False
    target_high_value: bool = False
    target_guarded: bool = False
    target_protected: bool = False
    already_marked: bool = False
    marked_turns_remaining: int = 0


@dataclass(frozen=True)
class _BanditMarkFollowup:
    expected_damage: int = 0
    has_payoff: bool = False
    acts_before_target: bool = False


@dataclass(frozen=True)
class TargetSupportProfile:
    has_heal_skill: bool = False
    has_recovery_skill: bool = False
    has_support_skill: bool = False
    has_effort_for_support: bool = False
    is_support_actor: bool = False
    is_backline_support: bool = False
    has_not_acted: bool = False
    urgent_support_available: bool = False


@dataclass(frozen=True)
class EnemyDecisionCandidate:
    skill_id: str
    target_id: str
    score: int
    skill_order: int
    target_order: int
    skill_tags: frozenset[str]
    features: EnemyDecisionFeatureVector = MappingProxyType({})


@dataclass(frozen=True)
class EnemyDecisionTrace:
    enemy_id: str
    runtime_context: EnemyDecisionRuntimeContext
    candidates: tuple[EnemyDecisionCandidate, ...]
    chosen: EnemyDecisionCandidate | None


class EnemyDecisionPolicy(Protocol):
    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME_CONTEXT,
    ) -> EnemyDecisionCandidate | None:
        """Choose one legal enemy action candidate."""
        ...


class HeuristicEnemyDecisionPolicy:
    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME_CONTEXT,
    ) -> EnemyDecisionCandidate | None:
        enemy = state.actor(enemy_id)
        if enemy.team != Team.ENEMY or not enemy.can_act():
            return None

        context = build_enemy_decision_context(state, definitions, enemy)
        candidates = enumerate_enemy_decision_candidates(
            state,
            definitions,
            enemy,
            context,
            runtime_context,
        )
        if not candidates:
            return None

        return _choose_best_candidate(candidates)


class StaticLearnedEnemyDecisionPolicy:
    """Deterministic production policy using fixed learned feature weights."""

    def __init__(self, weights: Mapping[str, int] | None = None) -> None:
        self.weights = weights or _PRODUCTION_LEARNED_WEIGHTS

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME_CONTEXT,
    ) -> EnemyDecisionCandidate | None:
        enemy = state.actor(enemy_id)
        if enemy.team != Team.ENEMY or not enemy.can_act():
            return None

        context = build_enemy_decision_context(state, definitions, enemy)
        candidates = enumerate_enemy_decision_candidates(
            state,
            definitions,
            enemy,
            context,
            runtime_context,
        )
        if not candidates:
            return None
        return _choose_weighted_candidate(candidates, self.weights)


_PRODUCTION_ENEMY_DECISION_POLICY = StaticLearnedEnemyDecisionPolicy()
_PRODUCTION_HEURISTIC_ENEMY_DECISION_POLICY = HeuristicEnemyDecisionPolicy()


def production_enemy_decision_policy(mode: str = "learned_static") -> EnemyDecisionPolicy:
    if mode == "learned_static":
        return _PRODUCTION_ENEMY_DECISION_POLICY
    if mode == "heuristic":
        return _PRODUCTION_HEURISTIC_ENEMY_DECISION_POLICY
    raise ValueError(f"Unknown enemy AI mode: {mode}")


def choose_enemy_skill_and_target(
    state: CombatState,
    definitions: GameDefinitions,
    enemy_id: str,
    runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME_CONTEXT,
    policy: EnemyDecisionPolicy | None = None,
) -> tuple[str, str] | None:
    """Choose the highest-scored legal enemy action with stable tie-breaking."""

    selected_policy = policy or HeuristicEnemyDecisionPolicy()
    choice = selected_policy.choose(state, definitions, enemy_id, runtime_context)
    if choice is None:
        return None
    return choice.skill_id, choice.target_id


def explain_enemy_decision(
    state: CombatState,
    definitions: GameDefinitions,
    enemy_id: str,
    runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME_CONTEXT,
) -> EnemyDecisionTrace | None:
    enemy = state.actor(enemy_id)
    if enemy.team != Team.ENEMY or not enemy.can_act():
        return None

    context = build_enemy_decision_context(state, definitions, enemy)
    candidates = tuple(
        enumerate_enemy_decision_candidates(
            state,
            definitions,
            enemy,
            context,
            runtime_context,
        )
    )
    return EnemyDecisionTrace(
        enemy_id=enemy_id,
        runtime_context=runtime_context,
        candidates=candidates,
        chosen=_choose_best_candidate(candidates),
    )


def build_enemy_decision_context(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
) -> EnemyDecisionContext:
    living_enemies = [candidate for candidate in state.enemies.values() if candidate.is_alive()]
    living_heroes = [candidate for candidate in state.heroes.values() if candidate.is_alive()]
    living_allies = [
        candidate
        for candidate in living_enemies
        if candidate.actor_id != enemy.actor_id and candidate.can_protect()
    ]
    return EnemyDecisionContext(
        round_number=state.round_number,
        living_enemy_count=len(living_enemies),
        living_hero_count=len(living_heroes),
        ally_count=len(living_allies),
        is_last_enemy=len(living_enemies) <= 1,
        enemy_low_hp=_is_low_hp(enemy),
        allies_have_payoff_skills=any(
            _combatant_has_payoff_skill(definitions, ally) for ally in living_allies
        ),
        marked_hero_count=sum(1 for hero in living_heroes if Tag.MARKED in hero.tags),
        wounded_hero_count=sum(1 for hero in living_heroes if _is_wounded(hero)),
    )


def enumerate_enemy_decision_candidates(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
    context: EnemyDecisionContext,
    runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME_CONTEXT,
) -> list[EnemyDecisionCandidate]:
    candidates: list[EnemyDecisionCandidate] = []
    for skill_order, skill_id in enumerate(enemy.skills):
        skill = definitions.skills[skill_id]
        if enemy.effort < skill.effort_cost:
            continue
        if not can_use_skill_from_position(state, enemy.actor_id, skill):
            continue
        for target_id in _legal_enemy_targets(state, enemy, skill):
            target = state.actor(target_id)
            skill_tags = frozenset(skill.tags)
            features = _feature_vector(
                state,
                definitions,
                context,
                runtime_context,
                enemy,
                skill,
                target,
            )
            candidates.append(
                EnemyDecisionCandidate(
                    skill_id=skill_id,
                    target_id=target_id,
                    score=_score_features(features),
                    skill_order=skill_order,
                    target_order=_target_order(state, target),
                    skill_tags=skill_tags,
                    features=MappingProxyType(features),
                )
            )
    return candidates


def _choose_best_candidate(
    candidates: tuple[EnemyDecisionCandidate, ...] | list[EnemyDecisionCandidate],
) -> EnemyDecisionCandidate | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            candidate.skill_order,
            candidate.target_order,
            candidate.target_id,
        ),
    )


def _choose_weighted_candidate(
    candidates: tuple[EnemyDecisionCandidate, ...] | list[EnemyDecisionCandidate],
    weights: Mapping[str, int],
) -> EnemyDecisionCandidate | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (
            -_weighted_score(candidate.features, weights),
            -candidate.score,
            candidate.skill_order,
            candidate.target_order,
            candidate.target_id,
        ),
    )


def _weighted_score(features: EnemyDecisionFeatureVector, weights: Mapping[str, int]) -> int:
    return sum(weights.get(name, 0) * value for name, value in features.items())


def _score_features(features: EnemyDecisionFeatureVector) -> int:
    return sum(
        value
        for name, value in features.items()
        if name not in _LEARNING_ONLY_FEATURES
    )


def _feature_vector(
    state: CombatState,
    definitions: GameDefinitions,
    context: EnemyDecisionContext,
    runtime_context: EnemyDecisionRuntimeContext,
    enemy: Combatant,
    skill: SkillDefinition,
    target: Combatant,
) -> dict[str, int]:
    skill_tags = set(skill.tags)
    enemy_role_tags = _enemy_role_tags(definitions, enemy)
    bandit_mark_profile = _bandit_mark_collapse_profile(
        state,
        definitions,
        runtime_context,
        enemy,
        skill_tags,
        target,
        enemy_role_tags,
    )
    support_profile = _target_support_profile(state, definitions, runtime_context, target)
    return {
        "formation": _formation_score(state, enemy, target),
        "marked_focus": _score_marked_focus(target),
        "damage_pressure": _score_damage_pressure(skill, skill_tags, enemy, target, context),
        "boss_pressure": _score_boss_pressure(state, skill, skill_tags, target),
        "guard": _score_guard(state, definitions, context, enemy, skill, target),
        "vulnerable_payoff": _score_vulnerable_payoff(skill_tags, target, enemy_role_tags),
        "maw_grab_setup": _score_maw_grab_setup(
            state,
            enemy_role_tags,
            skill_tags,
            target,
            support_profile,
        ),
        "maw_bite_payoff": _score_maw_bite_payoff(
            state,
            definitions,
            enemy,
            enemy_role_tags,
            skill,
            skill_tags,
            target,
            support_profile,
        ),
        "target_has_heal_skill": int(support_profile.has_heal_skill),
        "target_has_recovery_skill": int(support_profile.has_recovery_skill),
        "target_has_support_skill": int(support_profile.has_support_skill),
        "target_has_effort_for_support": int(support_profile.has_effort_for_support),
        "target_is_support_actor": int(support_profile.is_support_actor),
        "target_is_backline_support": int(support_profile.is_backline_support),
        "grab_disrupts_support_actor": _score_grab_disrupts_support_actor(
            state,
            enemy_role_tags,
            skill_tags,
            target,
            support_profile,
        ),
        "grab_action_tax_value": _score_grab_action_tax_value(
            state,
            enemy_role_tags,
            skill_tags,
            target,
            support_profile,
        ),
        "grab_target_has_not_acted": int(
            _is_maw_drag_candidate(state, enemy_role_tags, skill_tags, target)
            and support_profile.has_not_acted
        ),
        "grab_creates_support_denial": _score_grab_support_denial(
            state,
            enemy_role_tags,
            skill_tags,
            target,
            support_profile,
        ),
        "maw_grab_high_value_support": _score_maw_grab_high_value_support(
            state,
            enemy_role_tags,
            skill_tags,
            target,
            support_profile,
        ),
        "maw_grab_down_threat": _score_maw_grab_down_threat(
            state,
            definitions,
            enemy,
            enemy_role_tags,
            skill,
            skill_tags,
            target,
        ),
        "maw_grab_bite_expected_collapse": _score_maw_grab_bite_expected_collapse(
            state,
            definitions,
            enemy,
            enemy_role_tags,
            skill,
            skill_tags,
            target,
            support_profile,
        ),
        "boss_guard_package": _score_boss_guard_package(definitions, skill, target),
        "bandit_mark_collapse": _score_bandit_mark_collapse(bandit_mark_profile),
        "bandit_mark_kill_lane": _score_bandit_mark_kill_lane(bandit_mark_profile),
        "mark_ally_reach_count": bandit_mark_profile.ally_reach_count,
        "mark_expected_followup_damage": bandit_mark_profile.expected_followup_damage,
        "mark_payoff_attacker_count": bandit_mark_profile.payoff_attacker_count,
        "mark_acts_before_target_count": bandit_mark_profile.acts_before_target_count,
        "mark_followup_can_down": int(bandit_mark_profile.followup_can_down),
        "mark_target_low_hp": int(bandit_mark_profile.target_low_hp),
        "mark_target_guarded": int(bandit_mark_profile.target_guarded),
        "mark_target_protected": int(bandit_mark_profile.target_protected),
        "bandit_marked_attack": _score_bandit_marked_attack(
            skill,
            target,
            enemy_role_tags,
        ),
        "bandit_marked_payoff": _score_bandit_marked_payoff(
            skill_tags,
            target,
            enemy_role_tags,
        ),
        "bandit_ignored_marked_legal": _score_bandit_ignored_marked_legal(
            state,
            enemy,
            skill,
            target,
            enemy_role_tags,
        ),
        "effort_drain": _score_effort_drain(skill_tags, target),
        "mark": _score_mark(
            state,
            definitions,
            context,
            runtime_context,
            enemy,
            skill_tags,
            target,
            enemy_role_tags,
        ),
        "drag_forward": _score_drag_forward(
            state,
            definitions,
            context,
            enemy,
            enemy_role_tags,
            skill,
            skill_tags,
            target,
            support_profile,
        ),
        "role_flavor": _score_role_flavor(state, enemy_role_tags, enemy, target),
    }


def _score_pair(
    state: CombatState,
    definitions: GameDefinitions,
    context: EnemyDecisionContext,
    enemy: Combatant,
    skill: SkillDefinition,
    target: Combatant,
) -> int:
    return _score_features(
        _feature_vector(
            state,
            definitions,
            context,
            EnemyDecisionRuntimeContext(),
            enemy,
            skill,
            target,
        )
    )


def _score_marked_focus(target: Combatant) -> int:
    if target.team == Team.HERO and Tag.MARKED in target.tags:
        return _MARKED_FOCUS_SCORE
    return 0


def _score_damage_pressure(
    skill: SkillDefinition,
    skill_tags: set[str],
    enemy: Combatant,
    target: Combatant,
    context: EnemyDecisionContext,
) -> int:
    skill_damage_max = skill_base_damage_max(skill)
    if skill_damage_max <= 0:
        return 0
    score = max(0, skill_damage_max)
    if _estimated_damage(skill, skill_tags, enemy, target) >= target.hp:
        score += 22
    if context.is_last_enemy:
        score += 8
    if context.late_fight:
        score += 5
    if Tag.MARKED in target.tags:
        score += 3
    if _is_wounded(target):
        score += 3
    return score


def _score_boss_pressure(
    state: CombatState,
    skill: SkillDefinition,
    skill_tags: set[str],
    target: Combatant,
) -> int:
    if "boss" not in skill_tags or skill_base_damage_max(skill) <= 0:
        return 0
    if skill.id == "maw_slam" and _is_exposed_disruption_target(state, target):
        return 24
    return 0


def _score_guard(
    state: CombatState,
    definitions: GameDefinitions,
    context: EnemyDecisionContext,
    enemy: Combatant,
    skill: SkillDefinition,
    target: Combatant,
) -> int:
    if not _is_guard_skill(skill):
        return 0
    if target.team != enemy.team:
        return -100
    if target.actor_id == enemy.actor_id:
        return -20
    if context.is_last_enemy:
        return -40

    score = 20
    if _is_backline(state, target):
        score += 6
    if _is_wounded(target):
        score += 8
    if Tag.GUARDED in target.tags:
        score -= 35
    if _combatant_has_payoff_skill(definitions, target):
        score += 6
    if _enemy_role_tags(definitions, target) & {
        "alpha",
        "bandit",
        "beast",
        "boss",
        "horror",
        "leak",
        "maze",
        "maze_leak",
    }:
        score += 3
    if context.late_fight:
        score -= 8
    return score


def _score_vulnerable_payoff(
    skill_tags: set[str],
    target: Combatant,
    enemy_role_tags: set[str],
) -> int:
    if not _has_any(skill_tags, *_PAYOFF_TAGS):
        return 0
    score = 0
    if Tag.MARKED in target.tags:
        score += 26
    if _is_wounded(target):
        score += 12
    if _is_low_hp(target):
        score += 8
    if score <= 0:
        return -6
    if enemy_role_tags & {"bandit", "wolf"}:
        score += 4
    return score


def _score_maw_grab_setup(
    state: CombatState,
    enemy_role_tags: set[str],
    skill_tags: set[str],
    target: Combatant,
    support_profile: TargetSupportProfile,
) -> int:
    if "boss" not in enemy_role_tags or "drag_forward" not in skill_tags:
        return 0
    if not _is_backline(state, target):
        return 0
    score = 20
    if _is_backline_disruption_target(target) or support_profile.is_backline_support:
        score += 10
    if _is_protected_backline(state, target):
        score += 8
    if support_profile.has_effort_for_support:
        score += 8
    if support_profile.urgent_support_available:
        score += 10
    if support_profile.has_not_acted:
        score += 6
    return score


def _score_maw_bite_payoff(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
    enemy_role_tags: set[str],
    skill: SkillDefinition,
    skill_tags: set[str],
    target: Combatant,
    support_profile: TargetSupportProfile,
) -> int:
    if "boss" not in enemy_role_tags:
        return 0
    if "maw_slam" not in skill.id and "boss_special" not in skill_tags:
        return 0
    if not _is_frontline(state, target):
        return 0
    score = 8
    if _is_exposed_disruption_target(state, target):
        score += 12
    if _is_support_disruption_target(definitions, state, target, support_profile):
        score += 8
    if _estimated_damage(skill, skill_tags, enemy, target) >= target.hp:
        score += 8
    return score


def _target_support_profile(
    state: CombatState,
    definitions: GameDefinitions,
    runtime_context: EnemyDecisionRuntimeContext,
    target: Combatant,
) -> TargetSupportProfile:
    has_heal_skill = False
    has_recovery_skill = False
    has_support_skill = False
    has_effort_for_support = False
    for skill_id in target.skills:
        skill = definitions.skills.get(skill_id)
        if skill is None:
            continue
        skill_tags = set(skill.tags)
        is_heal = _has_any(skill_tags, "treatment", "heal")
        is_recovery = _has_any(skill_tags, "brink_heal")
        is_guard_support = "guard" in skill_tags and skill_base_damage_max(skill) <= 0
        is_support = is_heal or is_recovery or is_guard_support or _has_any(
            skill_tags,
            "support",
            "rally",
        )
        has_heal_skill = has_heal_skill or is_heal
        has_recovery_skill = has_recovery_skill or is_recovery
        has_support_skill = has_support_skill or is_support
        if is_support and target.effort >= skill.effort_cost:
            has_effort_for_support = True

    is_support_actor = has_heal_skill or has_recovery_skill or has_support_skill
    return TargetSupportProfile(
        has_heal_skill=has_heal_skill,
        has_recovery_skill=has_recovery_skill,
        has_support_skill=has_support_skill,
        has_effort_for_support=has_effort_for_support,
        is_support_actor=is_support_actor,
        is_backline_support=is_support_actor and _is_backline(state, target),
        has_not_acted=_has_not_acted_yet(runtime_context, target.actor_id),
        urgent_support_available=_urgent_hero_support_available(state),
    )


def _score_grab_disrupts_support_actor(
    state: CombatState,
    enemy_role_tags: set[str],
    skill_tags: set[str],
    target: Combatant,
    support_profile: TargetSupportProfile,
) -> int:
    if not _is_maw_drag_candidate(state, enemy_role_tags, skill_tags, target):
        return 0
    if not support_profile.is_support_actor:
        return 0
    score = 8
    if support_profile.is_backline_support:
        score += 8
    if support_profile.has_effort_for_support:
        score += 6
    return score


def _score_grab_action_tax_value(
    state: CombatState,
    enemy_role_tags: set[str],
    skill_tags: set[str],
    target: Combatant,
    support_profile: TargetSupportProfile,
) -> int:
    if not _is_maw_drag_candidate(state, enemy_role_tags, skill_tags, target):
        return 0
    if not support_profile.has_not_acted:
        return 0
    if support_profile.has_effort_for_support:
        return 12
    if support_profile.is_support_actor:
        return 5
    return 0


def _score_grab_support_denial(
    state: CombatState,
    enemy_role_tags: set[str],
    skill_tags: set[str],
    target: Combatant,
    support_profile: TargetSupportProfile,
) -> int:
    if not _is_maw_drag_candidate(state, enemy_role_tags, skill_tags, target):
        return 0
    if not support_profile.urgent_support_available:
        return 0
    if support_profile.has_effort_for_support:
        return 16
    if support_profile.is_support_actor:
        return 6
    return 0


def _score_maw_grab_high_value_support(
    state: CombatState,
    enemy_role_tags: set[str],
    skill_tags: set[str],
    target: Combatant,
    support_profile: TargetSupportProfile,
) -> int:
    if not _is_maw_drag_candidate(state, enemy_role_tags, skill_tags, target):
        return 0
    if not support_profile.is_support_actor:
        return 0
    score = 12
    if support_profile.has_heal_skill:
        score += 10
    if support_profile.has_recovery_skill:
        score += 8
    if support_profile.has_effort_for_support:
        score += 10
    if support_profile.has_not_acted:
        score += 8
    if support_profile.urgent_support_available:
        score += 10
    return score


def _score_maw_grab_down_threat(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
    enemy_role_tags: set[str],
    skill: SkillDefinition,
    skill_tags: set[str],
    target: Combatant,
) -> int:
    if not _is_maw_drag_candidate(state, enemy_role_tags, skill_tags, target):
        return 0
    total_damage = _maw_grab_plus_bite_damage(
        state,
        definitions,
        enemy,
        skill,
        skill_tags,
        target,
    )
    return 20 if total_damage >= target.hp else 0


def _score_maw_grab_bite_expected_collapse(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
    enemy_role_tags: set[str],
    skill: SkillDefinition,
    skill_tags: set[str],
    target: Combatant,
    support_profile: TargetSupportProfile,
) -> int:
    if not _is_maw_drag_candidate(state, enemy_role_tags, skill_tags, target):
        return 0
    total_damage = _maw_grab_plus_bite_damage(
        state,
        definitions,
        enemy,
        skill,
        skill_tags,
        target,
    )
    score = 0
    if total_damage >= target.hp:
        score += 28
    elif total_damage * 2 >= target.hp:
        score += 10
    if support_profile.is_support_actor and total_damage > 0:
        score += 8
    if support_profile.has_effort_for_support and total_damage > 0:
        score += 8
    return score


def _is_maw_drag_candidate(
    state: CombatState,
    enemy_role_tags: set[str],
    skill_tags: set[str],
    target: Combatant,
) -> bool:
    return (
        "boss" in enemy_role_tags
        and "drag_forward" in skill_tags
        and target.team == Team.HERO
        and _is_backline(state, target)
    )


def _maw_grab_plus_bite_damage(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
    drag_skill: SkillDefinition,
    drag_skill_tags: set[str],
    target: Combatant,
) -> int:
    drag_damage = _estimated_damage(drag_skill, drag_skill_tags, enemy, target)
    return drag_damage + _best_maw_bite_damage(state, definitions, enemy, target)


def _best_maw_bite_damage(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
    target: Combatant,
) -> int:
    best = 0
    for skill_id in enemy.skills:
        skill = definitions.skills.get(skill_id)
        if skill is None:
            continue
        skill_tags = set(skill.tags)
        if "maw_slam" not in skill.id and "boss_special" not in skill_tags:
            continue
        if enemy.effort < skill.effort_cost:
            continue
        if not can_use_skill_from_position(state, enemy.actor_id, skill):
            continue
        best = max(best, _estimated_damage(skill, skill_tags, enemy, target))
    return best


def _score_boss_guard_package(
    definitions: GameDefinitions,
    skill: SkillDefinition,
    target: Combatant,
) -> int:
    if not _is_guard_skill(skill):
        return 0
    if "boss" not in _enemy_role_tags(definitions, target):
        return 0
    if Tag.GUARDED in target.tags:
        return 0
    return 80


def _bandit_mark_collapse_profile(
    state: CombatState,
    definitions: GameDefinitions,
    runtime_context: EnemyDecisionRuntimeContext,
    enemy: Combatant,
    skill_tags: set[str],
    target: Combatant,
    enemy_role_tags: set[str],
) -> BanditMarkCollapseProfile:
    if (
        "bandit" not in enemy_role_tags
        or not _has_any(skill_tags, "mark_target", "mark")
        or target.team != Team.HERO
    ):
        return BanditMarkCollapseProfile()

    ally_reach_count = 0
    expected_followup_damage = 0
    payoff_attacker_count = 0
    acts_before_target_count = 0
    for ally in state.enemies.values():
        if ally.actor_id == enemy.actor_id or not ally.can_act():
            continue
        followup = _best_marked_followup_for_ally(
            state,
            definitions,
            ally,
            target.actor_id,
            runtime_context,
        )
        if followup.expected_damage <= 0:
            continue
        ally_reach_count += 1
        expected_followup_damage += followup.expected_damage
        if followup.has_payoff:
            payoff_attacker_count += 1
        if followup.acts_before_target:
            acts_before_target_count += 1

    return BanditMarkCollapseProfile(
        ally_reach_count=ally_reach_count,
        expected_followup_damage=expected_followup_damage,
        payoff_attacker_count=payoff_attacker_count,
        acts_before_target_count=acts_before_target_count,
        followup_can_down=expected_followup_damage >= target.hp,
        target_low_hp=_is_low_hp(target),
        target_wounded=_is_wounded(target),
        target_high_value=target.class_id in _HIGH_VALUE_MARK_CLASSES,
        target_guarded=Tag.GUARDED in target.tags,
        target_protected=_is_protected_backline(state, target),
        already_marked=Tag.MARKED in target.tags,
        marked_turns_remaining=target.tag_turns.get(Tag.MARKED, 0),
    )


def _score_bandit_mark_collapse(profile: BanditMarkCollapseProfile) -> int:
    if profile.ally_reach_count <= 0:
        return 0

    score = 0
    score += 8 * profile.ally_reach_count
    score += 2 * profile.expected_followup_damage
    score += 12 * profile.payoff_attacker_count
    score += 6 * profile.acts_before_target_count
    if profile.followup_can_down:
        score += 28
    if profile.target_low_hp:
        score += 8
    if profile.target_wounded:
        score += 5
    if profile.target_high_value:
        score += 5
    if profile.target_guarded:
        score -= 18
    if profile.target_protected:
        score -= 10
    if profile.already_marked:
        score -= 35
        if profile.marked_turns_remaining <= 1 and profile.followup_can_down:
            score += 24
    return score


def _score_bandit_mark_kill_lane(profile: BanditMarkCollapseProfile) -> int:
    if profile.ally_reach_count <= 0 or profile.already_marked:
        return 0
    score = 10 * profile.ally_reach_count
    if profile.payoff_attacker_count > 0:
        score += 20
    if profile.followup_can_down:
        score += 20
    if profile.target_guarded:
        score -= 10
    if profile.target_protected:
        score -= 6
    return max(0, score)


def _best_marked_followup_for_ally(
    state: CombatState,
    definitions: GameDefinitions,
    ally: Combatant,
    target_id: str,
    runtime_context: EnemyDecisionRuntimeContext,
) -> _BanditMarkFollowup:
    target = state.actor(target_id)
    best = _BanditMarkFollowup()
    for skill_id in ally.skills:
        skill = definitions.skills[skill_id]
        skill_tags = set(skill.tags)
        if _is_guard_skill(skill):
            continue
        if ally.effort < skill.effort_cost:
            continue
        if not can_use_skill_from_position(state, ally.actor_id, skill):
            continue
        if target_id not in legal_targets(state, ally.actor_id, skill.attack_type):
            continue

        candidate = _BanditMarkFollowup(
            expected_damage=_expected_damage_against_marked_target(
                state,
                skill,
                skill_tags,
                ally,
                target,
            ),
            has_payoff=_has_any(skill_tags, *_PAYOFF_TAGS),
            acts_before_target=_acts_before_target(runtime_context, ally.actor_id, target_id),
        )
        if (candidate.expected_damage, candidate.has_payoff, candidate.acts_before_target) > (
            best.expected_damage,
            best.has_payoff,
            best.acts_before_target,
        ):
            best = candidate
    return best


def _expected_damage_against_marked_target(
    state: CombatState,
    skill: SkillDefinition,
    skill_tags: set[str],
    enemy: Combatant,
    target: Combatant,
) -> int:
    damage = combatant_damage_max(skill, enemy)
    if _has_any(skill_tags, *_PAYOFF_TAGS):
        damage += 2
    if "basic" in skill_tags:
        damage += 1
    hit_chance = max(
        0,
        min(
            100,
            skill.accuracy
            + enemy.accuracy
            - target.defense
            + cover_penalty(state, target.actor_id, skill.attack_type),
        ),
    )
    return (damage * hit_chance) // 100


def _score_bandit_marked_attack(
    skill: SkillDefinition,
    target: Combatant,
    enemy_role_tags: set[str],
) -> int:
    if "bandit" not in enemy_role_tags or skill_base_damage_max(skill) <= 0:
        return 0
    return 14 if Tag.MARKED in target.tags else 0


def _score_bandit_marked_payoff(
    skill_tags: set[str],
    target: Combatant,
    enemy_role_tags: set[str],
) -> int:
    if "bandit" not in enemy_role_tags or not _has_any(skill_tags, *_PAYOFF_TAGS):
        return 0
    return 30 if Tag.MARKED in target.tags else 0


def _score_bandit_ignored_marked_legal(
    state: CombatState,
    enemy: Combatant,
    skill: SkillDefinition,
    target: Combatant,
    enemy_role_tags: set[str],
) -> int:
    if "bandit" not in enemy_role_tags or skill_base_damage_max(skill) <= 0:
        return 0
    if Tag.MARKED in target.tags:
        return 0
    return 16 if _legal_marked_targets_for_skill(state, enemy, skill) else 0


def _score_effort_drain(skill_tags: set[str], target: Combatant) -> int:
    if not _has_any(skill_tags, "drain_effort", "effort_drain"):
        return 0
    if target.effort <= 0:
        return -18
    score = 8 + (4 * target.effort)
    if target.max_effort > 0 and target.effort >= target.max_effort:
        score += 2
    return score


def _score_mark(
    state: CombatState,
    definitions: GameDefinitions,
    context: EnemyDecisionContext,
    runtime_context: EnemyDecisionRuntimeContext,
    enemy: Combatant,
    skill_tags: set[str],
    target: Combatant,
    enemy_role_tags: set[str],
) -> int:
    if not _has_any(skill_tags, "mark_target", "mark"):
        return 0
    score = 0
    followup_value = _targetable_by_enemy_ally_followup_value(
        state,
        definitions,
        runtime_context,
        enemy,
        target.actor_id,
    )
    if Tag.MARKED in target.tags:
        score -= 35
        if target.tag_turns.get(Tag.MARKED, 0) <= 1 and followup_value >= 24:
            score += 28
    else:
        score += 12
    if context.ally_count > 0:
        score += 6
    if context.allies_have_payoff_skills:
        score += 12
    if "bandit" not in enemy_role_tags:
        score += followup_value
    if (
        enemy_role_tags & {"alpha", "wolf"} == {"alpha", "wolf"}
        and context.marked_hero_count == 0
        and context.allies_have_payoff_skills
    ):
        score += 12
    if context.is_last_enemy:
        score -= 28
    if context.late_fight:
        score -= 10
    if _is_backline(state, target):
        score += 8
    if _is_wounded(target):
        score += 3
    if target.class_id in _HIGH_VALUE_MARK_CLASSES:
        score += 5
    return score


def _score_drag_forward(
    state: CombatState,
    definitions: GameDefinitions,
    context: EnemyDecisionContext,
    enemy: Combatant,
    enemy_role_tags: set[str],
    skill: SkillDefinition,
    skill_tags: set[str],
    target: Combatant,
    support_profile: TargetSupportProfile,
) -> int:
    if not _has_any(skill_tags, "pull_forward", "drag_forward"):
        return 0
    score = 0
    if _is_backline(state, target):
        score += 10
    if _is_protected_backline(state, target):
        score += 6
    if _is_backline_disruption_target(target) or support_profile.is_backline_support:
        score += 16
    else:
        score -= 18
    if "boss" in enemy_role_tags and "drag_forward" in skill_tags:
        if support_profile.is_backline_support:
            score += 8
        if support_profile.has_effort_for_support:
            score += 8
        if support_profile.urgent_support_available:
            score += 8
        if support_profile.has_not_acted:
            score += 6
        total_damage = _maw_grab_plus_bite_damage(
            state,
            definitions,
            enemy,
            skill,
            skill_tags,
            target,
        )
        if total_damage >= target.hp:
            score += 56
        elif total_damage * 2 >= target.hp:
            score += 8
    if _has_exposed_disruption_target(state):
        score -= 28
    if target.class_id in _FRONTLINE_ANCHOR_CLASSES:
        score -= 12
    if context.ally_count > 0:
        score += 4
    if context.is_last_enemy:
        score -= 10
    if context.late_fight:
        score -= 6
    return score


def _score_role_flavor(
    state: CombatState,
    enemy_role_tags: set[str],
    enemy: Combatant,
    target: Combatant,
) -> int:
    score = 0
    if "beast" in enemy_role_tags and (
        _same_lane(state, enemy, target) or _is_frontline(state, target)
    ):
        score += 1
    if "wolf" in enemy_role_tags and (_is_wounded(target) or Tag.MARKED in target.tags):
        score += 2
    if "bandit" in enemy_role_tags and (_is_wounded(target) or Tag.MARKED in target.tags):
        score += 2
    if enemy_role_tags & {"maze", "horror"} and _is_backline(state, target):
        score += 1
    if "boss" in enemy_role_tags and (_is_wounded(target) or Tag.MARKED in target.tags):
        score += 1
    return score


def _legal_enemy_targets(
    state: CombatState,
    enemy: Combatant,
    skill: SkillDefinition,
) -> list[str]:
    if _is_guard_skill(skill):
        ally_ids = [
            ally.actor_id
            for ally in state.side_for(enemy.team).values()
            if ally.actor_id != enemy.actor_id and ally.can_protect()
        ]
        if ally_ids:
            return sorted(ally_ids, key=lambda ally_id: _target_order(state, state.actor(ally_id)))
        return [enemy.actor_id]
    return legal_targets(state, enemy.actor_id, skill.attack_type)


def _is_guard_skill(skill: SkillDefinition) -> bool:
    return "guard" in skill.tags and skill_base_damage_max(skill) <= 0


def _combatant_has_payoff_skill(
    definitions: GameDefinitions,
    combatant: Combatant,
) -> bool:
    return any(
        bool(set(definitions.skills[skill_id].tags) & _PAYOFF_TAGS)
        and combatant.effort >= definitions.skills[skill_id].effort_cost
        for skill_id in combatant.skills
    if skill_id in definitions.skills
    )


def _targetable_by_enemy_ally_count(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
    target_id: str,
) -> int:
    count = 0
    for ally in state.enemies.values():
        if ally.actor_id == enemy.actor_id or not ally.can_act():
            continue
        if _can_enemy_target_with_any_offensive_skill(state, definitions, ally, target_id):
            count += 1
    return count


def _targetable_by_enemy_ally_followup_value(
    state: CombatState,
    definitions: GameDefinitions,
    runtime_context: EnemyDecisionRuntimeContext,
    enemy: Combatant,
    target_id: str,
) -> int:
    value = 0
    for ally in state.enemies.values():
        if ally.actor_id == enemy.actor_id or not ally.can_act():
            continue
        followup = _best_enemy_ally_followup(
            state,
            definitions,
            runtime_context,
            ally,
            target_id,
        )
        if followup <= 0:
            continue
        value += followup
    return value


def _best_enemy_ally_followup(
    state: CombatState,
    definitions: GameDefinitions,
    runtime_context: EnemyDecisionRuntimeContext,
    ally: Combatant,
    target_id: str,
) -> int:
    target = state.actor(target_id)
    best = 0
    for skill_id in ally.skills:
        skill = definitions.skills[skill_id]
        skill_tags = set(skill.tags)
        if _is_guard_skill(skill):
            continue
        if ally.effort < skill.effort_cost:
            continue
        if not can_use_skill_from_position(state, ally.actor_id, skill):
            continue
        if target_id not in legal_targets(state, ally.actor_id, skill.attack_type):
            continue

        value = _MARK_ALLY_ACCESS_SCORE
        if _has_any(skill_tags, *_PAYOFF_TAGS):
            value += 10
        if _estimated_damage(skill, skill_tags, ally, target) >= target.hp:
            value += 12
        if _acts_before_target(runtime_context, ally.actor_id, target_id):
            value += 6
        best = max(best, value)
    return best


def _acts_before_target(
    runtime_context: EnemyDecisionRuntimeContext,
    actor_id: str,
    target_id: str,
) -> bool:
    initiative = runtime_context.initiative_actor_ids
    if not initiative:
        return False
    try:
        actor_index = initiative.index(actor_id)
        target_index = initiative.index(target_id)
    except ValueError:
        return False
    current_index = max(0, runtime_context.current_turn_index)
    if actor_index < current_index <= target_index:
        return False
    if target_index < current_index <= actor_index:
        return True
    return actor_index < target_index


def _can_enemy_target_with_any_offensive_skill(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
    target_id: str,
) -> bool:
    for skill_id in enemy.skills:
        skill = definitions.skills[skill_id]
        if _is_guard_skill(skill):
            continue
        if enemy.effort < skill.effort_cost:
            continue
        if not can_use_skill_from_position(state, enemy.actor_id, skill):
            continue
        if target_id in legal_targets(state, enemy.actor_id, skill.attack_type):
            return True
    return False


def _estimated_damage_from_enemy_allies(
    state: CombatState,
    definitions: GameDefinitions,
    marker: Combatant,
    target_id: str,
) -> int:
    target = state.actor(target_id)
    total = 0
    for ally in state.enemies.values():
        if ally.actor_id == marker.actor_id or not ally.can_act():
            continue
        best = 0
        for skill_id in ally.skills:
            skill = definitions.skills[skill_id]
            skill_tags = set(skill.tags)
            if _is_guard_skill(skill):
                continue
            if ally.effort < skill.effort_cost:
                continue
            if not can_use_skill_from_position(state, ally.actor_id, skill):
                continue
            if target_id not in legal_targets(state, ally.actor_id, skill.attack_type):
                continue
            best = max(best, _estimated_damage(skill, skill_tags, ally, target))
        total += best
    return total


def _legal_marked_targets_for_skill(
    state: CombatState,
    enemy: Combatant,
    skill: SkillDefinition,
) -> list[str]:
    return [
        target_id
        for target_id in legal_targets(state, enemy.actor_id, skill.attack_type)
        if target_id in state.heroes and Tag.MARKED in state.heroes[target_id].tags
    ]


def _estimated_damage(
    skill: SkillDefinition,
    skill_tags: set[str],
    enemy: Combatant,
    target: Combatant,
) -> int:
    damage = combatant_damage_max(skill, enemy)
    if _has_any(skill_tags, *_PAYOFF_TAGS) and _is_vulnerable(target):
        damage += 2
    if "basic" in skill_tags and Tag.MARKED in target.tags:
        damage += 1
    return damage


def _has_any(tags: set[str], *candidates: str) -> bool:
    return any(candidate in tags for candidate in candidates)


def _formation_score(state: CombatState, enemy: Combatant, target: Combatant) -> int:
    score = 0
    if _same_lane(state, enemy, target):
        score += 2
    if _is_frontline(state, target):
        score += 1
    return score


def _enemy_role_tags(definitions: GameDefinitions, enemy: Combatant) -> set[str]:
    definition = definitions.enemies.get(enemy.class_id)
    if definition is None:
        return set()
    return set(definition.tags)


def _target_order(state: CombatState, target: Combatant) -> int:
    slot = state.formation_for(target.team).slot_of(target.actor_id)
    if slot is None:
        return _UNFORMED_TARGET_ORDER
    return _TARGET_SLOT_ORDER[slot]


def _same_lane(state: CombatState, enemy: Combatant, target: Combatant) -> bool:
    enemy_slot = state.formation_for(enemy.team).slot_of(enemy.actor_id)
    target_slot = state.formation_for(target.team).slot_of(target.actor_id)
    return (
        enemy_slot is not None
        and target_slot is not None
        and lane_of(enemy_slot) == lane_of(target_slot)
    )


def _is_frontline(state: CombatState, target: Combatant) -> bool:
    slot = state.formation_for(target.team).slot_of(target.actor_id)
    return slot is not None and is_front(slot)


def _is_backline(state: CombatState, target: Combatant) -> bool:
    slot = state.formation_for(target.team).slot_of(target.actor_id)
    return slot is not None and is_back(slot)


def _is_protected_backline(state: CombatState, target: Combatant) -> bool:
    formation = state.formation_for(target.team)
    slot = formation.slot_of(target.actor_id)
    return slot is not None and is_back(slot) and formation.is_protected(
        target.actor_id,
        state.side_for(target.team),
    )


def _is_exposed_disruption_target(state: CombatState, target: Combatant) -> bool:
    return _is_frontline(state, target) and _is_backline_disruption_target(target)


def _is_support_disruption_target(
    definitions: GameDefinitions,
    state: CombatState,
    target: Combatant,
    support_profile: TargetSupportProfile | None = None,
) -> bool:
    profile = support_profile or _target_support_profile(
        state,
        definitions,
        EnemyDecisionRuntimeContext(),
        target,
    )
    return _is_frontline(state, target) and profile.is_support_actor


def _has_exposed_disruption_target(state: CombatState) -> bool:
    return any(
        hero.is_alive() and _is_exposed_disruption_target(state, hero)
        for hero in state.heroes.values()
    )


def _is_backline_disruption_target(target: Combatant) -> bool:
    return target.class_id in _BACKLINE_DISRUPTION_CLASSES


def _urgent_hero_support_available(state: CombatState) -> bool:
    return any(
        hero.is_alive() and (hero.is_downed() or _is_low_hp(hero))
        for hero in state.heroes.values()
    )


def _has_not_acted_yet(
    runtime_context: EnemyDecisionRuntimeContext,
    target_id: str,
) -> bool:
    initiative = runtime_context.initiative_actor_ids
    if not initiative:
        return False
    try:
        target_index = initiative.index(target_id)
    except ValueError:
        return False
    return target_index > max(0, runtime_context.current_turn_index)


def _is_vulnerable(target: Combatant) -> bool:
    return Tag.MARKED in target.tags or _is_wounded(target)


def _is_wounded(target: Combatant) -> bool:
    return target.hp < target.max_hp


def _is_low_hp(target: Combatant) -> bool:
    return target.hp * 2 <= target.max_hp
