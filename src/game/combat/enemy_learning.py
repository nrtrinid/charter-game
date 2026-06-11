"""Dev-only offline enemy decision learning helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from game.combat.actions import use_skill
from game.combat.combat_state import Combatant, CombatState, Tag, Team
from game.combat.damage import GUARDED_DAMAGE_REDUCTION, heal_combatant
from game.combat.damage_range import (
    combatant_damage_max,
    roll_skill_base_damage,
    skill_base_damage_max,
)
from game.combat.enemy_actions import (
    SUPPORTED_ENEMY_MOVEMENT_MODES,
    SUPPORTED_ENEMY_WAIT_MODES,
    enemy_proactive_move,
    enemy_recovery_events,
    enemy_wait_reason,
    extract_move_timing_features,
    extract_wait_timing_features,
)
from game.combat.enemy_decision import (
    EnemyDecisionCandidate,
    EnemyDecisionPolicy,
    EnemyDecisionRuntimeContext,
    EnemyDecisionTrace,
    explain_enemy_decision,
)
from game.combat.formation import is_back, lane_of
from game.combat.morale import raise_morale
from game.combat.targeting import can_use_skill_from_position, cover_penalty, legal_targets
from game.combat.traits import PERSONAL_GENTLE_HANDS, PERSONAL_OPPORTUNIST, has_trait
from game.combat.turn_order import roll_initiative
from game.content.definitions import GameDefinitions
from game.core.events import (
    CombatEndedEvent,
    DamageEvent,
    DeathEvent,
    DownedEvent,
    GameEvent,
    HealingEvent,
    MissEvent,
    MoveEvent,
    SkillUsedEvent,
    StatusChangedEvent,
    TurnDelayedEvent,
)
from game.core.rng import GameRng
from game.data.schemas import SkillDefinition


@dataclass(frozen=True)
class PartyCollapseRewardWeights:
    enemy_victory: int = 1000
    forced_retreat: int = 900
    hero_death: int = 450
    hero_downed: int = 180
    mortal_wound: int = 220
    kill_range: int = 80
    hero_damage: int = 8
    effort_drained: int = 60
    setup_created: int = 90
    setup_converted: int = 160
    miss_penalty: int = -20
    maw_grab_setup: int = 1500
    maw_bite_conversion: int = 2500
    boss_guard_sequence: int = 2500
    bandit_mark_lane: int = 700
    bandit_mark_collapse: int = 700
    bandit_mark_down_threat: int = 900
    bandit_marked_attack: int = 400
    bandit_marked_payoff: int = 1000
    ignored_mark_penalty: int = 500
    waited_then_payoff: int = 120
    waited_then_marked_hit: int = 80
    waited_then_down: int = 100
    waited_then_no_attack: int = -60
    waited_then_no_payoff: int = -40
    wait_when_current_action_good: int = -80
    move_then_payoff: int = 120
    move_then_marked_hit: int = 80
    move_then_down: int = 100
    move_wasted_no_followup: int = -50
    move_when_current_action_good: int = -80
    recovery_move_then_attack: int = 40
    recovery_move_then_payoff: int = 80
    recovery_move_wasted: int = -40
    all_enemy_wait_round_penalty: int = -30


_DEFAULT_RUNTIME_CONTEXT = EnemyDecisionRuntimeContext()
_DEFAULT_REWARD_WEIGHTS = PartyCollapseRewardWeights()
_PENALTY_FEATURES = frozenset({"bandit_ignored_marked_legal"})
_LINEAR_POLICY_PRIOR_WEIGHTS = {
    "bandit_ignored_marked_legal": -20_000_000,
}
SUPPORTED_HERO_POLICY_IDS = (
    "naive",
    "damage_race",
    "survival",
    "anti_mark",
    "conservative",
    "tactical",
    "company_survival",
    "mixed",
)
HeroActionScore = tuple[int | str, ...]


@dataclass(frozen=True)
class HeroActionCandidate:
    skill_id: str
    target_id: str
    skill_order: int
    target_order: int
    skill_tags: frozenset[str]
    effort_cost: int
    estimated_damage: int
    expected_damage: int
    killable: bool
    is_support: bool
    is_heal: bool


@dataclass(frozen=True)
class CompanySurvivalScoreEntry:
    skill_id: str
    target_id: str
    tier: int
    subscore: int
    score: HeroActionScore
    killable: bool
    effort_cost: int
    package_target: str
    killable_opportunities: int


@dataclass(frozen=True)
class HeroPolicyActionRecord:
    hero_id: str
    round_number: int
    skill_id: str
    target_id: str
    effort_cost: int
    estimated_damage: int
    killable: bool
    is_heal: bool
    killable_opportunities: int
    ignored_killable_opportunity: bool
    package_target: str
    marked_hero_present: bool
    produced_kill: bool
    target_hp_remaining: int


class HeroDecisionPolicy(Protocol):
    @property
    def policy_id(self) -> str: ...

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        hero_id: str,
    ) -> tuple[str, str] | None:
        """Choose one legal dev-harness hero action."""
        ...


class NaiveHeroPolicy:
    policy_id = "naive"

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        hero_id: str,
    ) -> tuple[str, str] | None:
        return _first_usable_skill_and_target(state, definitions, hero_id)


class DamageRaceHeroPolicy:
    policy_id = "damage_race"

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        hero_id: str,
    ) -> tuple[str, str] | None:
        candidates = _hero_action_candidates(state, definitions, hero_id)
        kill_candidates = [candidate for candidate in candidates if candidate.killable]
        if kill_candidates:
            return _choice(
                min(
                    kill_candidates,
                    key=lambda candidate: (
                        candidate.effort_cost,
                        -candidate.expected_damage,
                        candidate.skill_order,
                        candidate.target_order,
                        candidate.target_id,
                    ),
                )
            )
        return _choice(_best_damage_candidate(candidates))


class SurvivalHeroPolicy:
    policy_id = "survival"

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        hero_id: str,
    ) -> tuple[str, str] | None:
        candidates = _hero_action_candidates(state, definitions, hero_id)
        urgent_heals = [
            candidate
            for candidate in candidates
            if candidate.is_heal and _is_urgent_ally(state.actor(candidate.target_id))
        ]
        if urgent_heals:
            return _choice(
                min(
                    urgent_heals,
                    key=lambda candidate: (
                        -_heal_priority(state.actor(candidate.target_id)),
                        candidate.effort_cost,
                        candidate.skill_order,
                        candidate.target_order,
                        candidate.target_id,
                    ),
                )
            )
        return DamageRaceHeroPolicy().choose(state, definitions, hero_id)


class AntiMarkHeroPolicy:
    policy_id = "anti_mark"

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        hero_id: str,
    ) -> tuple[str, str] | None:
        marked_heroes = [
            hero for hero in state.heroes.values() if hero.is_alive() and Tag.MARKED in hero.tags
        ]
        if not marked_heroes:
            return SurvivalHeroPolicy().choose(state, definitions, hero_id)

        marked_hero = min(marked_heroes, key=lambda hero: (hero.hp, hero.actor_id))
        candidates = _hero_action_candidates(state, definitions, hero_id)
        marked_killer = _best_killable_target(candidates, state, definitions, _enemy_can_mark)
        if marked_killer is not None:
            return _choice(marked_killer)

        payoff_killer = _best_killable_target(
            candidates,
            state,
            definitions,
            lambda enemy, _: _enemy_can_pay_off_mark(
                state,
                definitions,
                enemy,
                marked_hero.actor_id,
            ),
        )
        if payoff_killer is not None:
            return _choice(payoff_killer)

        support = _best_marked_hero_support(candidates, marked_hero.actor_id)
        if support is not None:
            return _choice(support)

        return SurvivalHeroPolicy().choose(state, definitions, hero_id)


class ConservativeHeroPolicy:
    policy_id = "conservative"

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        hero_id: str,
    ) -> tuple[str, str] | None:
        candidates = _hero_action_candidates(state, definitions, hero_id)
        zero_effort_kills = [
            candidate
            for candidate in candidates
            if candidate.killable and candidate.effort_cost == 0
        ]
        if zero_effort_kills:
            return _choice(_best_conservative_damage(zero_effort_kills))

        urgent_heals = [
            candidate
            for candidate in candidates
            if candidate.is_heal and _is_critical_ally(state.actor(candidate.target_id))
        ]
        if urgent_heals:
            return _choice(
                min(
                    urgent_heals,
                    key=lambda candidate: (
                        -_heal_priority(state.actor(candidate.target_id)),
                        candidate.effort_cost,
                        candidate.skill_order,
                        candidate.target_order,
                        candidate.target_id,
                    ),
                )
            )

        kill_candidates = [candidate for candidate in candidates if candidate.killable]
        if kill_candidates:
            return _choice(_best_conservative_damage(kill_candidates))

        anti_mark = _best_conservative_anti_mark(candidates, state, definitions)
        if anti_mark is not None:
            return _choice(anti_mark)

        zero_effort_damage = [
            candidate
            for candidate in candidates
            if candidate.effort_cost == 0 and candidate.estimated_damage > 0
        ]
        return _choice(_best_conservative_damage(zero_effort_damage))


class TacticalHeroPolicy:
    policy_id = "tactical"

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        hero_id: str,
    ) -> tuple[str, str] | None:
        candidates = _hero_action_candidates(state, definitions, hero_id)
        marked_heroes = [
            hero for hero in state.heroes.values() if hero.is_alive() and Tag.MARKED in hero.tags
        ]
        marked_hero = (
            min(marked_heroes, key=lambda hero: (hero.hp, hero.actor_id))
            if marked_heroes
            else None
        )
        scored: list[tuple[HeroActionScore, HeroActionCandidate]] = []
        for candidate in candidates:
            score = _tactical_action_score(
                candidate,
                state,
                definitions,
                marked_hero=marked_hero,
                candidates=candidates,
            )
            if score is not None:
                scored.append((score, candidate))
        if not scored:
            return None
        return _choice(max(scored, key=lambda item: item[0])[1])


class CompanySurvivalHeroPolicy:
    policy_id = "company_survival"

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        hero_id: str,
    ) -> tuple[str, str] | None:
        choice, _entries = _explain_company_survival_choice(state, definitions, hero_id)
        return choice


@dataclass(frozen=True)
class MixedHeroPolicy:
    encounter_id: str = "combat"
    seed: int | None = None

    @property
    def policy_id(self) -> str:
        return "mixed"

    @property
    def selected_policy_id(self) -> str:
        options = ("naive", "damage_race", "survival", "anti_mark", "conservative")
        return options[_stable_policy_index(self.encounter_id, self.seed) % len(options)]

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        hero_id: str,
    ) -> tuple[str, str] | None:
        return create_hero_policy(
            self.selected_policy_id,
            encounter_id=self.encounter_id,
            seed=self.seed,
        ).choose(state, definitions, hero_id)


_DEFAULT_HERO_POLICY = NaiveHeroPolicy()


@dataclass(frozen=True)
class EnemyDecisionRecord:
    enemy_id: str
    round_number: int
    action_index: int
    trace: EnemyDecisionTrace
    chosen_skill_id: str
    chosen_target_id: str
    chosen_features: Mapping[str, int]
    events: tuple[GameEvent, ...]
    action_reward: int = 0
    enemy_class_id: str = ""
    enemy_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PendingEnemyTiming:
    kind: str
    unlocked_future_skill: bool = False
    moved_into_marked_lane: bool = False
    features: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _TimingOutcome:
    attacked: bool
    marked_hit: bool
    payoff: bool
    down: bool

    @property
    def useful(self) -> bool:
        return self.marked_hit or self.payoff or self.down


@dataclass(frozen=True)
class _PendingGuard:
    guarding_actor_id: str
    guard_skill_id: str
    damage_blocked: int = 0
    consumed: bool = False


@dataclass(frozen=True)
class EnemyPressureMetrics:
    rounds_elapsed: int = 0
    enemy_decisions: int = 0
    total_hero_damage: int = 0
    lowest_hero_hp_reached: int = 0
    final_hero_hp_total: int = 0
    final_hero_effort_total: int = 0
    hero_downs: int = 0
    hero_deaths: int = 0
    mortal_wounds: int = 0
    healing_actions: int = 0
    healing_amount: int = 0
    marks_applied: int = 0
    marks_exploited: int = 0
    guard_actions: int = 0
    forced_movement: int = 0
    enemy_recovery_moves: int = 0
    enemy_proactive_moves: int = 0
    enemy_waits: int = 0
    waited_then_attacked_next_activation: int = 0
    waited_then_marked_hit: int = 0
    waited_then_payoff: int = 0
    waited_then_down: int = 0
    waited_then_no_payoff: int = 0
    waited_then_no_attack: int = 0
    waits_without_payoff: int = 0
    move_then_attack_next_activation: int = 0
    move_then_marked_hit: int = 0
    move_then_payoff: int = 0
    move_then_down: int = 0
    move_unlocks_future_skill: int = 0
    move_into_marked_lane: int = 0
    move_wasted_no_followup: int = 0
    moves_without_payoff: int = 0
    recovery_move_then_attack: int = 0
    recovery_move_then_payoff: int = 0
    recovery_move_wasted: int = 0
    recovery_moves_without_followup: int = 0
    all_enemy_wait_rounds: int = 0
    end_round_enemy_burst_damage: int = 0
    end_round_enemy_burst_downs: int = 0
    boss_sequence: BossSequenceMetrics = field(default_factory=lambda: BossSequenceMetrics())
    boss_targeting: BossTargetingMetrics = field(default_factory=lambda: BossTargetingMetrics())
    mark_flow: MarkFlowMetrics = field(default_factory=lambda: MarkFlowMetrics())
    guard_flow: GuardFlowMetrics = field(default_factory=lambda: GuardFlowMetrics())
    skill_uses: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class BossSequenceMetrics:
    grab_uses: int = 0
    bite_uses: int = 0
    grab_damage: int = 0
    bite_damage: int = 0
    grab_to_bite_same_target: int = 0
    grab_to_bite_any_target: int = 0
    grabbed_target_escaped_before_bite: int = 0
    grabbed_target_remained_in_bite_range: int = 0
    bite_hit_dragged_target: int = 0
    bite_hit_frontliner: int = 0
    bite_downs: int = 0
    boss_killed_before_first_bite: int = 0
    boss_killed_after_grab_before_bite: int = 0
    boss_actions_before_death: int = 0
    boss_rounds_survived: int = 0
    bone_soldier_guarded_boss: int = 0


@dataclass(frozen=True)
class BossTargetingMetrics:
    grab_target_classes: Mapping[str, int] = field(default_factory=dict)
    bite_target_classes: Mapping[str, int] = field(default_factory=dict)
    support_grabs: int = 0
    support_grabs_with_effort: int = 0
    support_grabs_not_acted: int = 0
    support_grab_to_bite_same_target: int = 0
    support_grab_downs: int = 0
    direct_front_bites: int = 0


@dataclass(frozen=True)
class MarkFlowMetrics:
    marks_applied: int = 0
    marks_refreshed: int = 0
    marks_applied_to_already_marked: int = 0
    exploited_by_enemy_hit: int = 0
    multi_hit_focus: int = 0
    vulnerable_payoffs: int = 0
    total_damage_to_marked: int = 0
    marked_downs: int = 0
    marked_deaths: int = 0
    marked_mortal_wounds: int = 0
    mark_ally_reach_total: int = 0
    mark_ally_reach_count: int = 0
    best_focus_marks: int = 0
    attacks_against_marked_when_legal: int = 0
    ignored_marked_legal_attacks: int = 0

    @property
    def average_ally_reach(self) -> float:
        if self.mark_ally_reach_count == 0:
            return 0.0
        return self.mark_ally_reach_total / self.mark_ally_reach_count


@dataclass(frozen=True)
class GuardFlowMetrics:
    guard_uses: int = 0
    dead_guard_uses: int = 0
    guard_targets: Mapping[str, int] = field(default_factory=dict)
    guard_targets_by_enemy_id: Mapping[str, int] = field(default_factory=dict)
    guard_damage_blocked: int = 0
    guarded_ally_survived_to_next_activation: int = 0
    guarded_ally_acted_after_guard: int = 0
    guarded_ally_used_payoff_after_guard: int = 0
    guarded_ally_downs_after_guard: int = 0
    guard_expired_or_consumed_without_payoff: int = 0
    guard_wasted_no_followup: int = 0


@dataclass(frozen=True)
class EnemyDecisionEpisode:
    encounter_id: str
    encounter_name: str
    seed: int | None
    records: tuple[EnemyDecisionRecord, ...]
    final_victor: str
    total_reward: int
    metrics: EnemyPressureMetrics = field(default_factory=EnemyPressureMetrics)
    hero_actions: tuple[HeroPolicyActionRecord, ...] = ()


@dataclass(frozen=True)
class LinearEnemyDecisionPolicy:
    weights: Mapping[str, int]

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME_CONTEXT,
    ) -> EnemyDecisionCandidate | None:
        trace = explain_enemy_decision(state, definitions, enemy_id, runtime_context)
        if trace is None or not trace.candidates:
            return None
        return _choose_weighted_candidate(trace.candidates, self.weights)


def run_enemy_learning_episode(
    state: CombatState,
    definitions: GameDefinitions,
    rng: GameRng,
    policy: EnemyDecisionPolicy | None = None,
    max_rounds: int = 20,
    *,
    encounter_id: str = "combat",
    encounter_name: str = "Combat Encounter",
    reward_weights: PartyCollapseRewardWeights = _DEFAULT_REWARD_WEIGHTS,
    hero_policy: HeroDecisionPolicy | None = None,
    enemy_wait_mode: str = "none",
    enemy_movement_mode: str = "recovery_only",
) -> EnemyDecisionEpisode:
    _validate_enemy_wait_mode(enemy_wait_mode)
    _validate_enemy_movement_mode(enemy_movement_mode)
    records: list[EnemyDecisionRecord] = []
    hero_actions: list[HeroPolicyActionRecord] = []
    episode_events: list[GameEvent] = []
    final_victor = "withdrawal"
    action_index = 0
    enemy_recovery_moves = 0
    enemy_proactive_moves = 0
    enemy_waits = 0
    waited_then_attacked_next_activation = 0
    waited_then_marked_hit = 0
    waited_then_payoff = 0
    waited_then_down = 0
    waited_then_no_payoff = 0
    waited_then_no_attack = 0
    waits_without_payoff = 0
    move_then_attack_next_activation = 0
    move_then_marked_hit = 0
    move_then_payoff = 0
    move_then_down = 0
    move_unlocks_future_skill = 0
    move_into_marked_lane = 0
    move_wasted_no_followup = 0
    moves_without_payoff = 0
    recovery_move_then_attack = 0
    recovery_move_then_payoff = 0
    recovery_move_wasted = 0
    recovery_moves_without_followup = 0
    all_enemy_wait_rounds = 0
    end_round_enemy_burst_damage = 0
    end_round_enemy_burst_downs = 0
    selected_hero_policy = hero_policy or _DEFAULT_HERO_POLICY
    initial_round_number = state.round_number
    initial_hero_hp = {hero_id: hero.hp for hero_id, hero in state.heroes.items()}
    initial_marked_heroes = frozenset(
        hero_id for hero_id, hero in state.heroes.items() if Tag.MARKED in hero.tags
    )
    diagnostic_tracker = _DiagnosticTracker(state, definitions)
    pending_timing: dict[str, _PendingEnemyTiming] = {}

    while not state.is_victory() and not state.is_defeat() and state.round_number <= max_rounds:
        initiative = roll_initiative(state, rng)
        round_enemy_actor_ids = tuple(
            entry.actor_id
            for entry in initiative
            if entry.actor_id in state.enemies and state.actor(entry.actor_id).can_act()
        )
        waited_actor_ids: set[str] = set()
        failed_move_actor_ids: set[str] = set()
        turn_index = 0
        while turn_index < len(initiative):
            entry = initiative[turn_index]
            actor = state.actor(entry.actor_id)
            if not actor.can_act() or state.is_victory() or state.is_defeat():
                turn_index += 1
                continue
            if actor.team == Team.ENEMY:
                initiative_actor_ids = tuple(entry.actor_id for entry in initiative)
                runtime_context = EnemyDecisionRuntimeContext(
                    initiative_actor_ids=initiative_actor_ids,
                    current_turn_index=turn_index,
                )
                if actor.actor_id not in pending_timing:
                    wait_reason = enemy_wait_reason(
                        state,
                        definitions,
                        actor,
                        runtime_context,
                        enemy_wait_mode,
                        waited_actor_ids,
                    )
                    if wait_reason is not None and turn_index + 1 < len(initiative):
                        delayed_entry = initiative.pop(turn_index)
                        initiative.append(delayed_entry)
                        waited_actor_ids.add(actor.actor_id)
                        wait_features = extract_wait_timing_features(
                            state,
                            definitions,
                            actor,
                            runtime_context,
                        )
                        pending_timing[actor.actor_id] = _PendingEnemyTiming(
                            kind="wait",
                            features=wait_features,
                        )
                        enemy_waits += 1
                        episode_events.append(
                            TurnDelayedEvent(
                                message=f"{actor.name} waits for {wait_reason}.",
                                actor_id=actor.actor_id,
                                encounter_id=encounter_id,
                            )
                        )
                        continue

                    had_legal_choice_before_move = _enemy_has_legal_offensive_choice(
                        state,
                        definitions,
                        actor,
                    )
                    pre_move_features = extract_move_timing_features(
                        state,
                        definitions,
                        actor,
                        runtime_context,
                        had_offensive_before=had_legal_choice_before_move,
                    )
                    movement_events = enemy_proactive_move(
                        state,
                        definitions,
                        actor,
                        enemy_movement_mode,
                        runtime_context,
                        failed_move_actor_ids,
                    )
                    if movement_events:
                        enemy_proactive_moves += 1
                        move_features = dict(pre_move_features)
                        unlocked = (
                            not had_legal_choice_before_move
                            and _enemy_has_legal_offensive_choice(
                                state,
                                definitions,
                                actor,
                            )
                        )
                        moved_into_marked_lane = _enemy_in_marked_lane(state, actor)
                        move_features["move_unlocks_future_skill"] = int(unlocked)
                        move_features["move_into_marked_lane"] = int(moved_into_marked_lane)
                        if unlocked:
                            move_unlocks_future_skill += 1
                        if moved_into_marked_lane:
                            move_into_marked_lane += 1
                        pending_timing[actor.actor_id] = _PendingEnemyTiming(
                            kind="move",
                            unlocked_future_skill=unlocked,
                            moved_into_marked_lane=moved_into_marked_lane,
                            features=move_features,
                        )
                        episode_events.extend(movement_events)
                        turn_index += 1
                        continue

                trace = _trace_for_policy(
                    state,
                    definitions,
                    actor.actor_id,
                    runtime_context,
                    policy,
                )
                if trace is None or trace.chosen is None:
                    pending = pending_timing.pop(actor.actor_id, None)
                    if pending is not None:
                        if pending.kind == "wait":
                            waited_then_no_attack += 1
                            waits_without_payoff += 1
                        elif pending.kind == "move":
                            move_wasted_no_followup += 1
                            moves_without_payoff += 1
                            failed_move_actor_ids.add(actor.actor_id)
                        elif pending.kind == "recovery":
                            recovery_move_wasted += 1
                            recovery_moves_without_followup += 1
                    recovery_events = enemy_recovery_events(
                        state,
                        definitions,
                        actor,
                        enemy_movement_mode,
                    )
                    if recovery_events:
                        enemy_recovery_moves += 1
                        recovery_unlocked = _enemy_has_legal_offensive_choice(
                            state,
                            definitions,
                            actor,
                        )
                        pending_timing[actor.actor_id] = _PendingEnemyTiming(
                            kind="recovery",
                            unlocked_future_skill=recovery_unlocked,
                            features={
                                "move_unlocks_future_skill": int(recovery_unlocked),
                            },
                        )
                        episode_events.extend(recovery_events)
                    turn_index += 1
                    continue
                chosen = trace.chosen
                action_snapshot = diagnostic_tracker.before_action(
                    state,
                    actor.actor_id,
                    definitions.skills[chosen.skill_id],
                    chosen.target_id,
                    runtime_context,
                )
                result = use_skill(
                    state,
                    actor.actor_id,
                    definitions.skills[chosen.skill_id],
                    chosen.target_id,
                    rng,
                )
                action_events = tuple(result.events)
                episode_events.extend(action_events)
                package_reward = diagnostic_tracker.after_action(
                    state,
                    action_snapshot,
                    action_events,
                    state.round_number,
                    reward_weights,
                )
                pending = pending_timing.pop(actor.actor_id, None)
                timing_reward = 0
                if pending is not None:
                    outcome = _timing_outcome(action_snapshot, action_events)
                    timing_reward = score_enemy_timing_outcome(
                        pending,
                        outcome,
                        weights=reward_weights,
                    )
                    if pending.kind == "wait":
                        if outcome.attacked:
                            waited_then_attacked_next_activation += 1
                        if outcome.marked_hit:
                            waited_then_marked_hit += 1
                        if outcome.payoff:
                            waited_then_payoff += 1
                            end_round_enemy_burst_damage += _hero_damage_amount(action_events)
                            end_round_enemy_burst_downs += _hero_down_count(action_events)
                        if outcome.down:
                            waited_then_down += 1
                        if outcome.attacked and not outcome.useful:
                            waited_then_no_payoff += 1
                        if not outcome.attacked:
                            waited_then_no_attack += 1
                        if not outcome.useful:
                            waits_without_payoff += 1
                    elif pending.kind == "move":
                        if outcome.attacked:
                            move_then_attack_next_activation += 1
                        if outcome.marked_hit:
                            move_then_marked_hit += 1
                        if outcome.payoff:
                            move_then_payoff += 1
                        if outcome.down:
                            move_then_down += 1
                        if not outcome.useful:
                            move_wasted_no_followup += 1
                            moves_without_payoff += 1
                            failed_move_actor_ids.add(actor.actor_id)
                    elif pending.kind == "recovery":
                        if outcome.attacked:
                            recovery_move_then_attack += 1
                        if outcome.payoff:
                            recovery_move_then_payoff += 1
                        if not outcome.useful:
                            recovery_move_wasted += 1
                            recovery_moves_without_followup += 1
                enemy_definition = definitions.enemies.get(actor.class_id)
                merged_features = dict(chosen.features)
                if pending is not None and pending.features:
                    for feature_name, feature_value in pending.features.items():
                        if feature_value != 0:
                            merged_features[feature_name] = feature_value
                action_reward = (
                    score_enemy_action_events(
                        action_events,
                        trace,
                        weights=reward_weights,
                    )
                    + package_reward
                    + timing_reward
                )
                records.append(
                    EnemyDecisionRecord(
                        enemy_id=actor.actor_id,
                        round_number=state.round_number,
                        action_index=action_index,
                        trace=trace,
                        chosen_skill_id=chosen.skill_id,
                        chosen_target_id=chosen.target_id,
                        chosen_features=merged_features,
                        events=action_events,
                        action_reward=action_reward,
                        enemy_class_id=actor.class_id,
                        enemy_tags=tuple(enemy_definition.tags) if enemy_definition else (),
                    )
                )
                action_index += 1
                turn_index += 1
            else:
                hero_candidates = _hero_action_candidates(
                    state,
                    definitions,
                    actor.actor_id,
                )
                skill_and_target = selected_hero_policy.choose(
                    state,
                    definitions,
                    actor.actor_id,
                )
                if skill_and_target is None:
                    turn_index += 1
                    continue
                skill_id, target_id = skill_and_target
                action_snapshot = diagnostic_tracker.before_action(
                    state,
                    actor.actor_id,
                    definitions.skills[skill_id],
                    target_id,
                )
                hero_events = _resolve_hero_policy_action(
                    state,
                    definitions,
                    rng,
                    actor.actor_id,
                    skill_id,
                    target_id,
                )
                episode_events.extend(hero_events)
                hero_actions.append(
                    _hero_policy_action_record(
                        state,
                        definitions,
                        hero_id=actor.actor_id,
                        round_number=state.round_number,
                        skill_id=skill_id,
                        target_id=target_id,
                        candidates=hero_candidates,
                        hero_events=hero_events,
                    )
                )
                diagnostic_tracker.after_action(
                    state,
                    action_snapshot,
                    hero_events,
                    state.round_number,
                    reward_weights,
                )
                turn_index += 1
        if round_enemy_actor_ids and len(waited_actor_ids) >= len(round_enemy_actor_ids):
            all_enemy_wait_rounds += 1
            if reward_weights.all_enemy_wait_round_penalty != 0:
                penalty = reward_weights.all_enemy_wait_round_penalty
                for index, record in enumerate(records):
                    if record.round_number == state.round_number:
                        records[index] = replace(
                            record,
                            action_reward=record.action_reward + penalty,
                        )
        for actor_id, pending in tuple(pending_timing.items()):
            if pending.kind == "wait":
                waited_then_no_attack += 1
                waits_without_payoff += 1
                pending_timing.pop(actor_id, None)
        state.round_number += 1

    for pending in pending_timing.values():
        if pending.kind == "wait":
            waited_then_no_attack += 1
            waits_without_payoff += 1
        elif pending.kind == "move":
            move_wasted_no_followup += 1
            moves_without_payoff += 1
        elif pending.kind == "recovery":
            recovery_move_wasted += 1
            recovery_moves_without_followup += 1

    if state.is_victory():
        final_victor = "heroes"
    elif state.is_defeat():
        final_victor = "enemies"

    return score_enemy_episode(
        EnemyDecisionEpisode(
            encounter_id=encounter_id,
            encounter_name=encounter_name,
            seed=rng.seed,
            records=tuple(records),
            final_victor=final_victor,
            total_reward=sum(record.action_reward for record in records),
            hero_actions=tuple(hero_actions),
            metrics=_enemy_pressure_metrics(
                state,
                definitions,
                tuple(episode_events),
                enemy_decision_count=len(records),
                rounds_elapsed=max(0, state.round_number - initial_round_number),
                initial_hero_hp=initial_hero_hp,
                initial_marked_heroes=initial_marked_heroes,
                boss_sequence=diagnostic_tracker.boss_metrics(state, state.round_number),
                boss_targeting=diagnostic_tracker.boss_targeting_metrics(),
                mark_flow=diagnostic_tracker.mark_metrics(),
                guard_flow=diagnostic_tracker.guard_metrics(),
                enemy_recovery_moves=enemy_recovery_moves,
                enemy_proactive_moves=enemy_proactive_moves,
                enemy_waits=enemy_waits,
                waited_then_attacked_next_activation=waited_then_attacked_next_activation,
                waited_then_marked_hit=waited_then_marked_hit,
                waited_then_payoff=waited_then_payoff,
                waited_then_down=waited_then_down,
                waited_then_no_payoff=waited_then_no_payoff,
                waited_then_no_attack=waited_then_no_attack,
                waits_without_payoff=waits_without_payoff,
                move_then_attack_next_activation=move_then_attack_next_activation,
                move_then_marked_hit=move_then_marked_hit,
                move_then_payoff=move_then_payoff,
                move_then_down=move_then_down,
                move_unlocks_future_skill=move_unlocks_future_skill,
                move_into_marked_lane=move_into_marked_lane,
                move_wasted_no_followup=move_wasted_no_followup,
                moves_without_payoff=moves_without_payoff,
                recovery_move_then_attack=recovery_move_then_attack,
                recovery_move_then_payoff=recovery_move_then_payoff,
                recovery_move_wasted=recovery_move_wasted,
                recovery_moves_without_followup=recovery_moves_without_followup,
                all_enemy_wait_rounds=all_enemy_wait_rounds,
                end_round_enemy_burst_damage=end_round_enemy_burst_damage,
                end_round_enemy_burst_downs=end_round_enemy_burst_downs,
            ),
        ),
        weights=reward_weights,
    )


def score_enemy_timing_outcome(
    pending: _PendingEnemyTiming,
    outcome: _TimingOutcome,
    *,
    weights: PartyCollapseRewardWeights = _DEFAULT_REWARD_WEIGHTS,
) -> int:
    reward = 0
    features = pending.features

    if pending.kind == "wait":
        if outcome.marked_hit:
            reward += weights.waited_then_marked_hit
        if outcome.payoff:
            reward += weights.waited_then_payoff
        if outcome.down:
            reward += weights.waited_then_down
        if not outcome.attacked:
            reward += weights.waited_then_no_attack
        elif not outcome.useful:
            reward += weights.waited_then_no_payoff
        if features.get("wait_when_current_action_good", 0) > 0:
            reward += weights.wait_when_current_action_good
    elif pending.kind == "move":
        if outcome.marked_hit:
            reward += weights.move_then_marked_hit
        if outcome.payoff:
            reward += weights.move_then_payoff
        if outcome.down:
            reward += weights.move_then_down
        if not outcome.useful:
            reward += weights.move_wasted_no_followup
        if features.get("move_when_current_action_good", 0) > 0:
            reward += weights.move_when_current_action_good
    elif pending.kind == "recovery":
        if outcome.attacked:
            reward += weights.recovery_move_then_attack
        if outcome.payoff:
            reward += weights.recovery_move_then_payoff
        if not outcome.useful:
            reward += weights.recovery_move_wasted

    return reward


def score_enemy_action_events(
    events: Sequence[GameEvent],
    trace: EnemyDecisionTrace | None = None,
    *,
    weights: PartyCollapseRewardWeights = _DEFAULT_REWARD_WEIGHTS,
) -> int:
    reward = 0
    damaged_hero_ids: set[str] = set()
    mark_added_ids: set[str] = set()
    payoff_hit = False
    chosen_features = (
        trace.chosen.features
        if trace is not None and trace.chosen is not None
        else {}
    )

    for event in events:
        if isinstance(event, DamageEvent):
            damaged_hero_ids.add(event.target_id)
            reward += event.amount * weights.hero_damage
            if event.hp_before is not None and event.amount > 0:
                hp_after = max(0, event.hp_before - event.amount)
                if hp_after > 0 and hp_after * 2 <= event.hp_before:
                    reward += weights.kill_range
        elif isinstance(event, DownedEvent):
            reward += weights.hero_downed
        elif isinstance(event, DeathEvent):
            reward += weights.hero_death
        elif isinstance(event, MissEvent):
            reward += weights.miss_penalty
        elif isinstance(event, StatusChangedEvent):
            if event.status == "mortal_wound" and event.added:
                reward += weights.mortal_wound
            elif event.status == "effort" and not event.added:
                reward += weights.effort_drained
            elif event.status == "marked" and event.added:
                mark_added_ids.add(event.actor_id)
        elif isinstance(event, CombatEndedEvent) and event.victor == "enemies":
            reward += weights.enemy_victory

    if mark_added_ids and chosen_features.get("mark", 0) > 0:
        reward += weights.setup_created
    if mark_added_ids and chosen_features.get("bandit_mark_kill_lane", 0) > 0:
        reward += weights.bandit_mark_lane
    if mark_added_ids and chosen_features.get("bandit_mark_collapse", 0) > 0:
        reward += weights.bandit_mark_collapse
    if mark_added_ids and chosen_features.get("mark_followup_can_down", 0) > 0:
        reward += weights.bandit_mark_down_threat
    if damaged_hero_ids and chosen_features.get("vulnerable_payoff", 0) > 0:
        payoff_hit = True
    if damaged_hero_ids and chosen_features.get("bandit_marked_attack", 0) > 0:
        reward += weights.bandit_marked_attack
    if damaged_hero_ids and chosen_features.get("bandit_marked_payoff", 0) > 0:
        reward += weights.bandit_marked_payoff
    if payoff_hit:
        reward += weights.setup_converted
    if chosen_features.get("bandit_ignored_marked_legal", 0) > 0:
        reward -= weights.ignored_mark_penalty
    return reward


def score_enemy_episode(
    episode: EnemyDecisionEpisode,
    *,
    weights: PartyCollapseRewardWeights = _DEFAULT_REWARD_WEIGHTS,
) -> EnemyDecisionEpisode:
    encounter_reward = 0
    if episode.final_victor == "enemies":
        encounter_reward += weights.enemy_victory
    elif episode.final_victor == "withdrawal":
        encounter_reward += weights.forced_retreat

    if not episode.records:
        return replace(episode, total_reward=encounter_reward)

    per_record_bonus = encounter_reward // len(episode.records)
    records = tuple(
        replace(record, action_reward=record.action_reward + per_record_bonus)
        for record in episode.records
    )
    return replace(
        episode,
        records=records,
        total_reward=sum(record.action_reward for record in records),
    )


def learn_linear_enemy_weights(
    episodes: Sequence[EnemyDecisionEpisode],
) -> dict[str, int]:
    learned: dict[str, int] = {}
    for episode in episodes:
        for record in episode.records:
            if record.action_reward == 0:
                continue
            for feature_name, feature_value in record.chosen_features.items():
                if feature_value == 0:
                    continue
                if feature_name in _PENALTY_FEATURES:
                    learned[feature_name] = learned.get(feature_name, 0) - (
                        feature_value * abs(record.action_reward)
                    )
                    continue
                if record.action_reward <= 0:
                    continue
                learned[feature_name] = learned.get(feature_name, 0) + (
                    feature_value * record.action_reward
                )
    return learned


def create_hero_policy(
    policy_id: str,
    *,
    encounter_id: str = "combat",
    seed: int | None = None,
) -> HeroDecisionPolicy:
    if policy_id == "naive":
        return NaiveHeroPolicy()
    if policy_id == "damage_race":
        return DamageRaceHeroPolicy()
    if policy_id == "survival":
        return SurvivalHeroPolicy()
    if policy_id == "anti_mark":
        return AntiMarkHeroPolicy()
    if policy_id == "conservative":
        return ConservativeHeroPolicy()
    if policy_id == "tactical":
        return TacticalHeroPolicy()
    if policy_id == "company_survival":
        return CompanySurvivalHeroPolicy()
    if policy_id == "mixed":
        return MixedHeroPolicy(encounter_id=encounter_id, seed=seed)
    raise ValueError(f"Unknown hero policy: {policy_id}")


@dataclass(frozen=True)
class _ActionDiagnosticSnapshot:
    actor_id: str
    actor_team: Team
    actor_class_id: str
    actor_tags: frozenset[str]
    skill_id: str
    skill_tags: frozenset[str]
    target_id: str
    target_class_id: str
    target_was_marked: bool
    target_was_front: bool
    target_was_support_actor: bool
    target_had_effort_for_support: bool
    target_had_not_acted: bool
    marked_legal_targets: frozenset[str]
    mark_ally_reach: int = 0
    mark_is_best_focus: bool = False


class _DiagnosticTracker:
    def __init__(self, state: CombatState, definitions: GameDefinitions) -> None:
        self._definitions = definitions
        self._boss_ids = frozenset(
            enemy.actor_id
            for enemy in state.enemies.values()
            if "boss" in self._enemy_tags(enemy.class_id)
        )
        self._boss_start_round = state.round_number
        self._boss_death_rounds: dict[str, int] = {}
        self._boss_had_grab: set[str] = set()
        self._boss_had_bite: set[str] = set()
        self._pending_grab_target_by_boss: dict[str, str] = {}
        self._boss_actions = 0
        self._boss_grab_uses = 0
        self._boss_bite_uses = 0
        self._boss_grab_damage = 0
        self._boss_bite_damage = 0
        self._boss_grab_target_classes: dict[str, int] = {}
        self._boss_bite_target_classes: dict[str, int] = {}
        self._support_grabs = 0
        self._support_grabs_with_effort = 0
        self._support_grabs_not_acted = 0
        self._support_grab_to_bite_same_target = 0
        self._support_grab_downs = 0
        self._direct_front_bites = 0
        self._grab_to_bite_same_target = 0
        self._grab_to_bite_any_target = 0
        self._grabbed_target_escaped_before_bite = 0
        self._grabbed_target_remained_in_bite_range = 0
        self._bite_hit_dragged_target = 0
        self._bite_hit_frontliner = 0
        self._bite_downs = 0
        self._boss_killed_before_first_bite = 0
        self._boss_killed_after_grab_before_bite = 0
        self._bone_soldier_guarded_boss = 0
        self._pending_grab_support_by_boss: dict[str, bool] = {}

        self._marked_targets: set[str] = {
            hero.actor_id for hero in state.heroes.values() if Tag.MARKED in hero.tags
        }
        self._marked_hit_counts: dict[str, int] = dict.fromkeys(self._marked_targets, 0)
        self._marks_applied = 0
        self._marks_refreshed = 0
        self._marks_applied_to_already_marked = 0
        self._exploited_by_enemy_hit = 0
        self._multi_hit_focus = 0
        self._vulnerable_payoffs = 0
        self._total_damage_to_marked = 0
        self._marked_downs = 0
        self._marked_deaths = 0
        self._marked_mortal_wounds = 0
        self._mark_ally_reach_total = 0
        self._mark_ally_reach_count = 0
        self._best_focus_marks = 0
        self._attacks_against_marked_when_legal = 0
        self._ignored_marked_legal_attacks = 0

        self._pending_guards: dict[str, _PendingGuard] = {}
        self._guard_uses = 0
        self._dead_guard_uses = 0
        self._guard_targets: dict[str, int] = {}
        self._guard_targets_by_enemy_id: dict[str, int] = {}
        self._guard_damage_blocked = 0
        self._guarded_ally_survived_to_next_activation = 0
        self._guarded_ally_acted_after_guard = 0
        self._guarded_ally_used_payoff_after_guard = 0
        self._guarded_ally_downs_after_guard = 0
        self._guard_expired_or_consumed_without_payoff = 0
        self._guard_wasted_no_followup = 0

    def before_action(
        self,
        state: CombatState,
        actor_id: str,
        skill: SkillDefinition,
        target_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME_CONTEXT,
    ) -> _ActionDiagnosticSnapshot:
        actor = state.actor(actor_id)
        target = state.actor(target_id)
        skill_tags = frozenset(skill.tags)
        target_was_support_actor = _combatant_has_support_skill(self._definitions, target)
        target_had_effort_for_support = _combatant_has_affordable_support_skill(
            self._definitions,
            target,
        )
        marked_legal_targets: frozenset[str] = frozenset()
        mark_ally_reach = 0
        mark_is_best_focus = False
        if actor.team == Team.ENEMY:
            legal = legal_targets(state, actor_id, skill.attack_type)
            marked_legal_targets = frozenset(
                hero_id
                for hero_id in legal
                if hero_id in state.heroes and Tag.MARKED in state.heroes[hero_id].tags
            )
            if marked_legal_targets:
                if target_id in marked_legal_targets:
                    self._attacks_against_marked_when_legal += 1
                elif skill.damage > 0:
                    self._ignored_marked_legal_attacks += 1
            if "mark" in skill_tags and target.team == Team.HERO:
                mark_ally_reach = _enemy_ally_reach_count(
                    state,
                    self._definitions,
                    actor_id,
                    target_id,
                )
                mark_is_best_focus = mark_ally_reach >= _best_enemy_ally_reach_count(
                    state,
                    self._definitions,
                    actor_id,
                )
                self._mark_ally_reach_total += mark_ally_reach
                self._mark_ally_reach_count += 1
                if mark_is_best_focus:
                    self._best_focus_marks += 1
                if Tag.MARKED in target.tags:
                    self._marks_refreshed += 1
                    self._marks_applied_to_already_marked += 1
        return _ActionDiagnosticSnapshot(
            actor_id=actor_id,
            actor_team=actor.team,
            actor_class_id=actor.class_id,
            actor_tags=self._enemy_tags(actor.class_id),
            skill_id=skill.id,
            skill_tags=skill_tags,
            target_id=target_id,
            target_class_id=target.class_id,
            target_was_marked=Tag.MARKED in target.tags,
            target_was_front=_is_front_target(state, target_id),
            target_was_support_actor=target_was_support_actor,
            target_had_effort_for_support=target_had_effort_for_support,
            target_had_not_acted=_has_not_acted_yet(runtime_context, target_id),
            marked_legal_targets=marked_legal_targets,
            mark_ally_reach=mark_ally_reach,
            mark_is_best_focus=mark_is_best_focus,
        )

    def after_action(
        self,
        state: CombatState,
        snapshot: _ActionDiagnosticSnapshot,
        events: Sequence[GameEvent],
        round_number: int,
        weights: PartyCollapseRewardWeights,
    ) -> int:
        reward = 0
        if snapshot.actor_id in self._boss_ids:
            reward += self._observe_boss_action(state, snapshot, events, weights)
        elif "guard" in snapshot.skill_tags and snapshot.target_id in self._boss_ids:
            self._bone_soldier_guarded_boss += 1
            reward += weights.boss_guard_sequence

        if snapshot.actor_team == Team.ENEMY:
            self._observe_enemy_mark_flow(snapshot, events)
            self._observe_enemy_guard_flow(snapshot, events)
        self._observe_guard_damage_and_losses(events)
        self._resolve_guarded_actor_activation(snapshot, events)
        self._observe_mark_outcomes(events)
        self._observe_boss_deaths(events, round_number)
        return reward

    def boss_metrics(self, state: CombatState, final_round: int) -> BossSequenceMetrics:
        boss_rounds_survived = 0
        for boss_id in self._boss_ids:
            death_round = self._boss_death_rounds.get(boss_id)
            if (
                death_round is None
                and boss_id in state.enemies
                and state.enemies[boss_id].is_alive()
            ):
                death_round = final_round
            elif death_round is None:
                death_round = final_round
            boss_rounds_survived += max(0, death_round - self._boss_start_round + 1)
        return BossSequenceMetrics(
            grab_uses=self._boss_grab_uses,
            bite_uses=self._boss_bite_uses,
            grab_damage=self._boss_grab_damage,
            bite_damage=self._boss_bite_damage,
            grab_to_bite_same_target=self._grab_to_bite_same_target,
            grab_to_bite_any_target=self._grab_to_bite_any_target,
            grabbed_target_escaped_before_bite=self._grabbed_target_escaped_before_bite,
            grabbed_target_remained_in_bite_range=self._grabbed_target_remained_in_bite_range,
            bite_hit_dragged_target=self._bite_hit_dragged_target,
            bite_hit_frontliner=self._bite_hit_frontliner,
            bite_downs=self._bite_downs,
            boss_killed_before_first_bite=self._boss_killed_before_first_bite,
            boss_killed_after_grab_before_bite=self._boss_killed_after_grab_before_bite,
            boss_actions_before_death=self._boss_actions,
            boss_rounds_survived=boss_rounds_survived,
            bone_soldier_guarded_boss=self._bone_soldier_guarded_boss,
        )

    def boss_targeting_metrics(self) -> BossTargetingMetrics:
        return BossTargetingMetrics(
            grab_target_classes=dict(self._boss_grab_target_classes),
            bite_target_classes=dict(self._boss_bite_target_classes),
            support_grabs=self._support_grabs,
            support_grabs_with_effort=self._support_grabs_with_effort,
            support_grabs_not_acted=self._support_grabs_not_acted,
            support_grab_to_bite_same_target=self._support_grab_to_bite_same_target,
            support_grab_downs=self._support_grab_downs,
            direct_front_bites=self._direct_front_bites,
        )

    def mark_metrics(self) -> MarkFlowMetrics:
        return MarkFlowMetrics(
            marks_applied=self._marks_applied,
            marks_refreshed=self._marks_refreshed,
            marks_applied_to_already_marked=self._marks_applied_to_already_marked,
            exploited_by_enemy_hit=self._exploited_by_enemy_hit,
            multi_hit_focus=self._multi_hit_focus,
            vulnerable_payoffs=self._vulnerable_payoffs,
            total_damage_to_marked=self._total_damage_to_marked,
            marked_downs=self._marked_downs,
            marked_deaths=self._marked_deaths,
            marked_mortal_wounds=self._marked_mortal_wounds,
            mark_ally_reach_total=self._mark_ally_reach_total,
            mark_ally_reach_count=self._mark_ally_reach_count,
            best_focus_marks=self._best_focus_marks,
            attacks_against_marked_when_legal=self._attacks_against_marked_when_legal,
            ignored_marked_legal_attacks=self._ignored_marked_legal_attacks,
        )

    def guard_metrics(self) -> GuardFlowMetrics:
        return GuardFlowMetrics(
            guard_uses=self._guard_uses,
            dead_guard_uses=self._dead_guard_uses,
            guard_targets=dict(self._guard_targets),
            guard_targets_by_enemy_id=dict(self._guard_targets_by_enemy_id),
            guard_damage_blocked=self._guard_damage_blocked,
            guarded_ally_survived_to_next_activation=(
                self._guarded_ally_survived_to_next_activation
            ),
            guarded_ally_acted_after_guard=self._guarded_ally_acted_after_guard,
            guarded_ally_used_payoff_after_guard=self._guarded_ally_used_payoff_after_guard,
            guarded_ally_downs_after_guard=self._guarded_ally_downs_after_guard,
            guard_expired_or_consumed_without_payoff=(
                self._guard_expired_or_consumed_without_payoff
                + len(self._pending_guards)
            ),
            guard_wasted_no_followup=self._guard_wasted_no_followup + len(self._pending_guards),
        )

    def _observe_boss_action(
        self,
        state: CombatState,
        snapshot: _ActionDiagnosticSnapshot,
        events: Sequence[GameEvent],
        weights: PartyCollapseRewardWeights,
    ) -> int:
        reward = 0
        self._boss_actions += 1
        if "drag_forward" in snapshot.skill_tags:
            self._boss_grab_uses += 1
            self._boss_had_grab.add(snapshot.actor_id)
            self._pending_grab_target_by_boss[snapshot.actor_id] = snapshot.target_id
            self._pending_grab_support_by_boss[snapshot.actor_id] = (
                snapshot.target_was_support_actor
            )
            _increment_count(self._boss_grab_target_classes, snapshot.target_class_id)
            self._boss_grab_damage += _damage_to_target(events, snapshot.target_id)
            if snapshot.target_was_support_actor:
                self._support_grabs += 1
                if snapshot.target_had_effort_for_support:
                    self._support_grabs_with_effort += 1
                if snapshot.target_had_not_acted:
                    self._support_grabs_not_acted += 1
                if any(
                    isinstance(event, DownedEvent) and event.actor_id == snapshot.target_id
                    for event in events
                ):
                    self._support_grab_downs += 1
            if any(isinstance(event, MoveEvent) for event in events):
                reward += weights.maw_grab_setup
        elif _is_boss_bite_skill(snapshot.skill_id, snapshot.skill_tags):
            self._boss_bite_uses += 1
            self._boss_had_bite.add(snapshot.actor_id)
            _increment_count(self._boss_bite_target_classes, snapshot.target_class_id)
            self._boss_bite_damage += _damage_to_target(events, snapshot.target_id)
            pending_target = self._pending_grab_target_by_boss.pop(snapshot.actor_id, None)
            pending_support = self._pending_grab_support_by_boss.pop(snapshot.actor_id, False)
            if pending_target is not None:
                self._grab_to_bite_any_target += 1
                reward += weights.maw_bite_conversion
                bite_skill = self._definitions.skills[snapshot.skill_id]
                if pending_target in legal_targets(
                    state,
                    snapshot.actor_id,
                    bite_skill.attack_type,
                ):
                    self._grabbed_target_remained_in_bite_range += 1
                else:
                    self._grabbed_target_escaped_before_bite += 1
                if pending_target == snapshot.target_id:
                    self._grab_to_bite_same_target += 1
                    self._bite_hit_dragged_target += 1
                    if pending_support:
                        self._support_grab_to_bite_same_target += 1
                    reward += weights.maw_bite_conversion
            elif snapshot.target_was_front:
                self._bite_hit_frontliner += 1
                self._direct_front_bites += 1
            if any(
                isinstance(event, DownedEvent) and event.actor_id == snapshot.target_id
                for event in events
            ):
                self._bite_downs += 1
                if pending_support and pending_target == snapshot.target_id:
                    self._support_grab_downs += 1
        return reward

    def _observe_enemy_mark_flow(
        self,
        snapshot: _ActionDiagnosticSnapshot,
        events: Sequence[GameEvent],
    ) -> None:
        for event in events:
            if isinstance(event, StatusChangedEvent) and event.status == "marked" and event.added:
                if event.actor_id in self._marked_targets:
                    self._marks_refreshed += 1
                    self._marks_applied_to_already_marked += 1
                else:
                    self._marks_applied += 1
                    self._marked_targets.add(event.actor_id)
                    self._marked_hit_counts[event.actor_id] = 0
            elif isinstance(event, DamageEvent) and event.target_id in self._marked_targets:
                self._exploited_by_enemy_hit += 1
                self._total_damage_to_marked += event.amount
                hits = self._marked_hit_counts.get(event.target_id, 0) + 1
                self._marked_hit_counts[event.target_id] = hits
                if hits == 2:
                    self._multi_hit_focus += 1
                if _has_payoff_tag(snapshot.skill_tags):
                    self._vulnerable_payoffs += 1

    def _observe_enemy_guard_flow(
        self,
        snapshot: _ActionDiagnosticSnapshot,
        events: Sequence[GameEvent],
    ) -> None:
        if not _is_zero_damage_guard_skill(snapshot.skill_id, snapshot.skill_tags):
            return
        if not any(
            isinstance(event, StatusChangedEvent)
            and event.actor_id == snapshot.target_id
            and event.status == "guarded"
            and event.added
            for event in events
        ):
            return
        self._guard_uses += 1
        if snapshot.skill_id == "shielding_dead":
            self._dead_guard_uses += 1
        _increment_count(self._guard_targets, snapshot.target_class_id)
        _increment_count(self._guard_targets_by_enemy_id, snapshot.target_id)
        self._pending_guards[snapshot.target_id] = _PendingGuard(
            guarding_actor_id=snapshot.actor_id,
            guard_skill_id=snapshot.skill_id,
        )

    def _observe_guard_damage_and_losses(self, events: Sequence[GameEvent]) -> None:
        consumed_guard_target: str | None = None
        for event in events:
            if (
                isinstance(event, StatusChangedEvent)
                and event.status == "guarded"
                and not event.added
            ):
                consumed_guard_target = event.actor_id
                pending = self._pending_guards.get(event.actor_id)
                if pending is not None:
                    self._pending_guards[event.actor_id] = replace(pending, consumed=True)
                continue
            if (
                isinstance(event, DamageEvent)
                and consumed_guard_target == event.target_id
                and event.target_id in self._pending_guards
            ):
                blocked = GUARDED_DAMAGE_REDUCTION
                pending = self._pending_guards[event.target_id]
                self._pending_guards[event.target_id] = replace(
                    pending,
                    damage_blocked=pending.damage_blocked + blocked,
                    consumed=True,
                )
                self._guard_damage_blocked += blocked
                consumed_guard_target = None
                continue
            if isinstance(event, DeathEvent) and event.actor_id in self._pending_guards:
                self._clear_guard_without_payoff(event.actor_id)

    def _resolve_guarded_actor_activation(
        self,
        snapshot: _ActionDiagnosticSnapshot,
        events: Sequence[GameEvent],
    ) -> None:
        pending = self._pending_guards.pop(snapshot.actor_id, None)
        if pending is None:
            return
        self._guarded_ally_survived_to_next_activation += 1
        self._guarded_ally_acted_after_guard += 1
        attacked = _hero_damage_amount(events) > 0
        payoff = attacked and _has_payoff_tag(snapshot.skill_tags)
        downed = _hero_down_count(events) > 0
        if payoff or downed:
            self._guarded_ally_used_payoff_after_guard += int(payoff)
            self._guarded_ally_downs_after_guard += int(downed)
            return
        self._guard_wasted_no_followup += 1
        if pending.consumed:
            self._guard_expired_or_consumed_without_payoff += 1

    def _clear_guard_without_payoff(self, actor_id: str) -> None:
        if self._pending_guards.pop(actor_id, None) is None:
            return
        self._guard_expired_or_consumed_without_payoff += 1
        self._guard_wasted_no_followup += 1

    def _observe_mark_outcomes(self, events: Sequence[GameEvent]) -> None:
        for event in events:
            if isinstance(event, DownedEvent) and event.actor_id in self._marked_targets:
                self._marked_downs += 1
            elif isinstance(event, DeathEvent) and event.actor_id in self._marked_targets:
                self._marked_deaths += 1
            elif (
                isinstance(event, StatusChangedEvent)
                and event.actor_id in self._marked_targets
                and event.status == "mortal_wound"
                and event.added
            ):
                self._marked_mortal_wounds += 1
            elif (
                isinstance(event, StatusChangedEvent)
                and event.status == "marked"
                and not event.added
            ):
                self._marked_targets.discard(event.actor_id)
                self._marked_hit_counts.pop(event.actor_id, None)

    def _observe_boss_deaths(self, events: Sequence[GameEvent], round_number: int) -> None:
        for event in events:
            if not isinstance(event, DeathEvent) or event.actor_id not in self._boss_ids:
                continue
            if event.actor_id in self._boss_death_rounds:
                continue
            self._boss_death_rounds[event.actor_id] = round_number
            if event.actor_id not in self._boss_had_bite:
                self._boss_killed_before_first_bite += 1
            if (
                event.actor_id in self._boss_had_grab
                and event.actor_id not in self._boss_had_bite
            ):
                self._boss_killed_after_grab_before_bite += 1

    def _enemy_tags(self, enemy_class_id: str) -> frozenset[str]:
        enemy_definition = self._definitions.enemies.get(enemy_class_id)
        if enemy_definition is None:
            return frozenset()
        return frozenset(enemy_definition.tags)


def _enemy_pressure_metrics(
    state: CombatState,
    definitions: GameDefinitions,
    events: Sequence[GameEvent],
    *,
    enemy_decision_count: int,
    rounds_elapsed: int,
    initial_hero_hp: Mapping[str, int],
    initial_marked_heroes: frozenset[str],
    boss_sequence: BossSequenceMetrics,
    boss_targeting: BossTargetingMetrics,
    mark_flow: MarkFlowMetrics,
    guard_flow: GuardFlowMetrics,
    enemy_recovery_moves: int = 0,
    enemy_proactive_moves: int = 0,
    enemy_waits: int = 0,
    waited_then_attacked_next_activation: int = 0,
    waited_then_marked_hit: int = 0,
    waited_then_payoff: int = 0,
    waited_then_down: int = 0,
    waited_then_no_payoff: int = 0,
    waited_then_no_attack: int = 0,
    waits_without_payoff: int = 0,
    move_then_attack_next_activation: int = 0,
    move_then_marked_hit: int = 0,
    move_then_payoff: int = 0,
    move_then_down: int = 0,
    move_unlocks_future_skill: int = 0,
    move_into_marked_lane: int = 0,
    move_wasted_no_followup: int = 0,
    moves_without_payoff: int = 0,
    recovery_move_then_attack: int = 0,
    recovery_move_then_payoff: int = 0,
    recovery_move_wasted: int = 0,
    recovery_moves_without_followup: int = 0,
    all_enemy_wait_rounds: int = 0,
    end_round_enemy_burst_damage: int = 0,
    end_round_enemy_burst_downs: int = 0,
) -> EnemyPressureMetrics:
    total_hero_damage = 0
    lowest_hero_hp_reached = min(initial_hero_hp.values(), default=0)
    hero_downs = 0
    hero_deaths = 0
    mortal_wounds = 0
    healing_actions = 0
    healing_amount = 0
    marks_applied = 0
    marks_exploited = 0
    guard_actions = 0
    forced_movement = 0
    skill_uses: dict[str, int] = {}
    marked_heroes = set(initial_marked_heroes)

    for event in events:
        if isinstance(event, SkillUsedEvent):
            skill_uses[event.skill_id] = skill_uses.get(event.skill_id, 0) + 1
            skill = definitions.skills.get(event.skill_id)
            if (
                skill is not None
                and "guard" in skill.tags
                and event.actor_id in state.heroes
            ):
                guard_actions += 1
        elif isinstance(event, DamageEvent):
            if event.target_id in state.heroes:
                total_hero_damage += event.amount
                if event.hp_before is not None:
                    lowest_hero_hp_reached = min(
                        lowest_hero_hp_reached,
                        max(0, event.hp_before - event.amount),
                    )
                if event.source_id in state.enemies and event.target_id in marked_heroes:
                    marks_exploited += 1
        elif isinstance(event, HealingEvent):
            if event.target_id in state.heroes:
                healing_actions += 1
                healing_amount += event.amount
        elif isinstance(event, DownedEvent):
            if event.actor_id in state.heroes:
                hero_downs += 1
        elif isinstance(event, DeathEvent):
            if event.actor_id in state.heroes:
                hero_deaths += 1
        elif isinstance(event, MoveEvent):
            if event.actor_id in state.heroes:
                forced_movement += 1
        elif isinstance(event, StatusChangedEvent):
            if event.actor_id in state.heroes and event.status == "mortal_wound" and event.added:
                mortal_wounds += 1
            elif event.actor_id in state.heroes and event.status == "marked":
                if event.added:
                    marks_applied += 1
                    marked_heroes.add(event.actor_id)
                else:
                    marked_heroes.discard(event.actor_id)

    for hero in state.heroes.values():
        lowest_hero_hp_reached = min(lowest_hero_hp_reached, hero.hp)

    return EnemyPressureMetrics(
        rounds_elapsed=rounds_elapsed,
        enemy_decisions=enemy_decision_count,
        total_hero_damage=total_hero_damage,
        lowest_hero_hp_reached=lowest_hero_hp_reached,
        final_hero_hp_total=sum(hero.hp for hero in state.heroes.values()),
        final_hero_effort_total=sum(hero.effort for hero in state.heroes.values()),
        hero_downs=hero_downs,
        hero_deaths=hero_deaths,
        mortal_wounds=mortal_wounds,
        healing_actions=healing_actions,
        healing_amount=healing_amount,
        marks_applied=marks_applied,
        marks_exploited=marks_exploited,
        guard_actions=guard_actions,
        forced_movement=forced_movement,
        enemy_recovery_moves=enemy_recovery_moves,
        enemy_proactive_moves=enemy_proactive_moves,
        enemy_waits=enemy_waits,
        waited_then_attacked_next_activation=waited_then_attacked_next_activation,
        waited_then_marked_hit=waited_then_marked_hit,
        waited_then_payoff=waited_then_payoff,
        waited_then_down=waited_then_down,
        waited_then_no_payoff=waited_then_no_payoff,
        waited_then_no_attack=waited_then_no_attack,
        waits_without_payoff=waits_without_payoff,
        move_then_attack_next_activation=move_then_attack_next_activation,
        move_then_marked_hit=move_then_marked_hit,
        move_then_payoff=move_then_payoff,
        move_then_down=move_then_down,
        move_unlocks_future_skill=move_unlocks_future_skill,
        move_into_marked_lane=move_into_marked_lane,
        move_wasted_no_followup=move_wasted_no_followup,
        moves_without_payoff=moves_without_payoff,
        recovery_move_then_attack=recovery_move_then_attack,
        recovery_move_then_payoff=recovery_move_then_payoff,
        recovery_move_wasted=recovery_move_wasted,
        recovery_moves_without_followup=recovery_moves_without_followup,
        all_enemy_wait_rounds=all_enemy_wait_rounds,
        end_round_enemy_burst_damage=end_round_enemy_burst_damage,
        end_round_enemy_burst_downs=end_round_enemy_burst_downs,
        boss_sequence=boss_sequence,
        boss_targeting=boss_targeting,
        mark_flow=mark_flow,
        guard_flow=guard_flow,
        skill_uses=skill_uses,
    )


def _resolve_hero_policy_action(
    state: CombatState,
    definitions: GameDefinitions,
    rng: GameRng,
    hero_id: str,
    skill_id: str,
    target_id: str,
) -> tuple[GameEvent, ...]:
    skill = definitions.skills[skill_id]
    if _is_treatment_skill(skill.tags):
        return tuple(_resolve_hero_treatment(state, rng, hero_id, skill, target_id))
    result = use_skill(state, hero_id, skill, target_id, rng)
    return tuple(result.events)


def _validate_enemy_wait_mode(mode: str) -> None:
    if mode not in SUPPORTED_ENEMY_WAIT_MODES:
        raise ValueError(f"Unknown enemy wait mode: {mode}")


def _validate_enemy_movement_mode(mode: str) -> None:
    if mode not in SUPPORTED_ENEMY_MOVEMENT_MODES:
        raise ValueError(f"Unknown enemy movement mode: {mode}")


def _action_damaged_marked_target(
    snapshot: _ActionDiagnosticSnapshot,
    events: Sequence[GameEvent],
) -> bool:
    return snapshot.target_was_marked and _hero_damage_amount(events) > 0


def _action_had_payoff(
    snapshot: _ActionDiagnosticSnapshot,
    events: Sequence[GameEvent],
) -> bool:
    return (
        snapshot.target_was_marked
        and _hero_damage_amount(events) > 0
        and (
            _has_payoff_tag(snapshot.skill_tags)
            or "basic" in snapshot.skill_tags
            or "boss_special" in snapshot.skill_tags
        )
    )


def _timing_outcome(
    snapshot: _ActionDiagnosticSnapshot,
    events: Sequence[GameEvent],
) -> _TimingOutcome:
    attacked = any(
        (
            isinstance(event, DamageEvent)
            and event.source_id == snapshot.actor_id
        )
        or (
            isinstance(event, MissEvent)
            and event.actor_id == snapshot.actor_id
        )
        for event in events
    )
    marked_hit = _action_damaged_marked_target(snapshot, events)
    down = _hero_down_count(events) > 0
    payoff = marked_hit or _action_had_payoff(snapshot, events) or down
    return _TimingOutcome(
        attacked=attacked,
        marked_hit=marked_hit,
        payoff=payoff,
        down=down,
    )


def _enemy_has_legal_offensive_choice(
    state: CombatState,
    definitions: GameDefinitions,
    actor: Combatant,
) -> bool:
    for skill_id in actor.skills:
        skill = definitions.skills[skill_id]
        if actor.effort < skill.effort_cost:
            continue
        if skill_base_damage_max(skill) <= 0:
            continue
        if not can_use_skill_from_position(state, actor.actor_id, skill):
            continue
        if legal_targets(state, actor.actor_id, skill.attack_type):
            return True
    return False


def _enemy_in_marked_lane(state: CombatState, actor: Combatant) -> bool:
    actor_slot = state.enemy_formation.slot_of(actor.actor_id)
    if actor_slot is None:
        return False
    actor_lane = lane_of(actor_slot)
    return any(
        lane_of(slot) == actor_lane
        for hero in state.heroes.values()
        if hero.is_alive() and Tag.MARKED in hero.tags
        for slot in [state.party_formation.slot_of(hero.actor_id)]
        if slot is not None
    )


def _hero_damage_amount(events: Sequence[GameEvent]) -> int:
    return sum(
        event.amount
        for event in events
        if isinstance(event, DamageEvent) and event.target_id
    )


def _hero_down_count(events: Sequence[GameEvent]) -> int:
    return sum(1 for event in events if isinstance(event, DownedEvent))


def _damage_to_target(events: Sequence[GameEvent], target_id: str) -> int:
    return sum(
        event.amount
        for event in events
        if isinstance(event, DamageEvent) and event.target_id == target_id
    )


def _has_payoff_tag(tags: frozenset[str]) -> bool:
    return bool({"vulnerable_bonus", "exploit_vulnerable"}.intersection(tags))


def _is_zero_damage_guard_skill(skill_id: str, tags: frozenset[str]) -> bool:
    return "guard" in tags and skill_id != "guard_strike"


def _is_boss_bite_skill(skill_id: str, tags: frozenset[str]) -> bool:
    return "maw_slam" in skill_id or ("boss_special" in tags and "drag_forward" not in tags)


def _is_front_target(state: CombatState, target_id: str) -> bool:
    target = state.actor(target_id)
    slot = state.formation_for(target.team).slot_of(target_id)
    if slot is None:
        return False
    return not is_back(slot)


def _combatant_has_support_skill(
    definitions: GameDefinitions,
    combatant: Combatant,
) -> bool:
    return any(
        _is_support_diagnostic_skill(definitions.skills[skill_id])
        for skill_id in combatant.skills
        if skill_id in definitions.skills
    )


def _combatant_has_affordable_support_skill(
    definitions: GameDefinitions,
    combatant: Combatant,
) -> bool:
    return any(
        combatant.effort >= definitions.skills[skill_id].effort_cost
        and _is_support_diagnostic_skill(definitions.skills[skill_id])
        for skill_id in combatant.skills
        if skill_id in definitions.skills
    )


def _is_support_diagnostic_skill(skill: SkillDefinition) -> bool:
    tags = set(skill.tags)
    return (
        "treatment" in tags
        or "heal" in tags
        or "brink_heal" in tags
        or "support" in tags
        or "rally" in tags
        or ("guard" in tags and skill_base_damage_max(skill) <= 0)
    )


def _has_not_acted_yet(
    runtime_context: EnemyDecisionRuntimeContext,
    actor_id: str,
) -> bool:
    initiative = runtime_context.initiative_actor_ids
    if not initiative:
        return False
    try:
        actor_index = initiative.index(actor_id)
    except ValueError:
        return False
    return actor_index > max(0, runtime_context.current_turn_index)


def _increment_count(counts: dict[str, int], key: str) -> None:
    label = key or "unknown"
    counts[label] = counts.get(label, 0) + 1


def _enemy_ally_reach_count(
    state: CombatState,
    definitions: GameDefinitions,
    marker_id: str,
    target_id: str,
) -> int:
    count = 0
    for enemy in state.enemies.values():
        if enemy.actor_id == marker_id or not enemy.can_act():
            continue
        if _enemy_can_legally_hit(state, definitions, enemy.actor_id, target_id):
            count += 1
    return count


def _best_enemy_ally_reach_count(
    state: CombatState,
    definitions: GameDefinitions,
    marker_id: str,
) -> int:
    living_heroes = [hero.actor_id for hero in state.heroes.values() if hero.is_alive()]
    if not living_heroes:
        return 0
    return max(
        _enemy_ally_reach_count(state, definitions, marker_id, hero_id)
        for hero_id in living_heroes
    )


def _enemy_can_legally_hit(
    state: CombatState,
    definitions: GameDefinitions,
    enemy_id: str,
    target_id: str,
) -> bool:
    enemy = state.actor(enemy_id)
    for skill_id in enemy.skills:
        skill = definitions.skills[skill_id]
        if skill.damage <= 0:
            continue
        if enemy.effort < skill.effort_cost:
            continue
        if not can_use_skill_from_position(state, enemy_id, skill):
            continue
        if target_id in legal_targets(state, enemy_id, skill.attack_type):
            return True
    return False


def _resolve_hero_treatment(
    state: CombatState,
    rng: GameRng,
    hero_id: str,
    skill: SkillDefinition,
    target_id: str,
) -> list[GameEvent]:
    actor = state.actor(hero_id)
    target = state.actor(target_id)
    if not actor.can_act() or actor.effort < skill.effort_cost:
        return []
    if target.team != Team.HERO or not target.is_alive():
        return []
    actor.effort -= skill.effort_cost
    events: list[GameEvent] = [
        SkillUsedEvent(
            message=f"{actor.name} uses {skill.name} on {target.name}.",
            actor_id=hero_id,
            skill_id=skill.id,
            target_id=target_id,
        )
    ]
    amount = max(1, roll_skill_base_damage(skill, rng) + actor.damage)
    if "brink_heal" in skill.tags and _is_urgent_ally(target):
        amount += 2
    amount = max(0, min(amount, target.max_hp - target.hp))
    events.append(
        HealingEvent(
            message=f"{target.name} recovers {amount} HP.",
            source_id=hero_id,
            target_id=target_id,
            amount=amount,
        )
    )
    events.extend(heal_combatant(state, target_id, amount))
    if actor.personal_quirk == PERSONAL_GENTLE_HANDS and raise_morale(target):
        events.append(
            StatusChangedEvent(
                message=f"{actor.name}'s Gentle Hands steady {target.name}.",
                actor_id=target_id,
                status=target.morale.name.lower(),
                added=True,
            )
        )
    return events


def _trace_for_policy(
    state: CombatState,
    definitions: GameDefinitions,
    enemy_id: str,
    runtime_context: EnemyDecisionRuntimeContext,
    policy: EnemyDecisionPolicy | None,
) -> EnemyDecisionTrace | None:
    trace = explain_enemy_decision(state, definitions, enemy_id, runtime_context)
    if trace is None or policy is None:
        return trace
    chosen = policy.choose(state, definitions, enemy_id, runtime_context)
    return replace(trace, chosen=chosen)


def _choose_weighted_candidate(
    candidates: Sequence[EnemyDecisionCandidate],
    weights: Mapping[str, int],
) -> EnemyDecisionCandidate | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (
            -_weighted_score(candidate.features, weights),
            candidate.skill_order,
            candidate.target_order,
            candidate.target_id,
        ),
    )


def _weighted_score(features: Mapping[str, int], weights: Mapping[str, int]) -> int:
    return sum(
        (weights.get(name, 0) + _LINEAR_POLICY_PRIOR_WEIGHTS.get(name, 0)) * value
        for name, value in features.items()
    )


def _hero_action_candidates(
    state: CombatState,
    definitions: GameDefinitions,
    hero_id: str,
) -> list[HeroActionCandidate]:
    hero = state.actor(hero_id)
    candidates: list[HeroActionCandidate] = []
    for skill_order, skill_id in enumerate(hero.skills):
        skill = definitions.skills[skill_id]
        if hero.effort < skill.effort_cost:
            continue
        if not can_use_skill_from_position(state, hero_id, skill):
            continue
        tags = frozenset(skill.tags)
        is_support = _is_support_skill(skill)
        is_heal = _is_treatment_skill(skill.tags)
        targets = _legal_hero_support_targets(state) if is_support else sorted(
            legal_targets(state, hero_id, skill.attack_type)
        )
        for target_order, target_id in enumerate(targets):
            target = state.actor(target_id)
            estimated_damage = (
                _estimated_hero_damage(state, skill, set(tags), hero, target)
                if target.team == Team.ENEMY
                else 0
            )
            expected_damage = _expected_hero_damage(
                state,
                skill,
                hero,
                target,
                estimated_damage,
            )
            candidates.append(
                HeroActionCandidate(
                    skill_id=skill_id,
                    target_id=target_id,
                    skill_order=skill_order,
                    target_order=target_order,
                    skill_tags=tags,
                    effort_cost=skill.effort_cost,
                    estimated_damage=estimated_damage,
                    expected_damage=expected_damage,
                    killable=target.team == Team.ENEMY and estimated_damage >= target.hp,
                    is_support=is_support,
                    is_heal=is_heal,
                )
            )
    return candidates


def _best_damage_candidate(
    candidates: Sequence[HeroActionCandidate],
) -> HeroActionCandidate | None:
    offensive = [candidate for candidate in candidates if candidate.estimated_damage > 0]
    if not offensive:
        return None
    return min(
        offensive,
        key=lambda candidate: (
            -candidate.expected_damage,
            candidate.effort_cost,
            candidate.skill_order,
            candidate.target_order,
            candidate.target_id,
        ),
    )


def _best_conservative_damage(
    candidates: Sequence[HeroActionCandidate],
) -> HeroActionCandidate | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (
            candidate.effort_cost,
            -candidate.expected_damage,
            candidate.skill_order,
            candidate.target_order,
            candidate.target_id,
        ),
    )


def _best_conservative_anti_mark(
    candidates: Sequence[HeroActionCandidate],
    state: CombatState,
    definitions: GameDefinitions,
) -> HeroActionCandidate | None:
    marked_heroes = [
        hero for hero in state.heroes.values() if hero.is_alive() and Tag.MARKED in hero.tags
    ]
    if not marked_heroes:
        return None
    marked_hero = min(marked_heroes, key=lambda hero: (hero.hp, hero.actor_id))
    marker = _best_killable_target(candidates, state, definitions, _enemy_can_mark)
    if marker is not None:
        return marker
    return _best_killable_target(
        candidates,
        state,
        definitions,
        lambda enemy, _: _enemy_can_pay_off_mark(
            state,
            definitions,
            enemy,
            marked_hero.actor_id,
        ),
    )


def _choice(candidate: HeroActionCandidate | None) -> tuple[str, str] | None:
    if candidate is None:
        return None
    return candidate.skill_id, candidate.target_id


def _best_killable_target(
    candidates: Sequence[HeroActionCandidate],
    state: CombatState,
    definitions: GameDefinitions,
    predicate,
) -> HeroActionCandidate | None:
    matches = [
        candidate
        for candidate in candidates
        if candidate.killable and predicate(state.actor(candidate.target_id), definitions)
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda candidate: (
            candidate.effort_cost,
            -candidate.expected_damage,
            candidate.skill_order,
            candidate.target_order,
            candidate.target_id,
        ),
    )


def _best_marked_hero_support(
    candidates: Sequence[HeroActionCandidate],
    marked_hero_id: str,
) -> HeroActionCandidate | None:
    support = [
        candidate
        for candidate in candidates
        if candidate.target_id == marked_hero_id
        and (candidate.is_heal or "guard" in candidate.skill_tags)
    ]
    if not support:
        return None
    return min(
        support,
        key=lambda candidate: (
            0 if candidate.is_heal else 1,
            candidate.effort_cost,
            candidate.skill_order,
            candidate.target_order,
            candidate.target_id,
        ),
    )


def _first_usable_skill_and_target(
    state: CombatState,
    definitions: GameDefinitions,
    actor_id: str,
) -> tuple[str, str] | None:
    actor = state.actor(actor_id)
    for skill_id in actor.skills:
        skill = definitions.skills[skill_id]
        if actor.effort < skill.effort_cost:
            continue
        if not can_use_skill_from_position(state, actor_id, skill):
            continue
        targets = legal_targets(state, actor_id, skill.attack_type)
        if targets:
            return skill_id, sorted(targets)[0]
    return None


def _is_support_skill(skill: SkillDefinition) -> bool:
    return _is_treatment_skill(skill.tags) or "rally" in skill.tags or (
        "guard" in skill.tags and skill_base_damage_max(skill) <= 0
    )


def _is_treatment_skill(tags: Sequence[str]) -> bool:
    return "treatment" in tags or "heal" in tags


def _legal_hero_support_targets(state: CombatState) -> list[str]:
    return sorted(hero.actor_id for hero in state.heroes.values() if hero.is_alive())


def _is_urgent_ally(hero: Combatant) -> bool:
    return hero.is_downed() or hero.hp * 2 <= hero.max_hp


def _is_critical_ally(hero: Combatant) -> bool:
    return hero.is_downed() or hero.hp * 4 <= hero.max_hp


def _heal_priority(hero: Combatant) -> int:
    if hero.is_downed():
        return 1000
    return hero.max_hp - hero.hp


def _estimated_hero_damage(
    state: CombatState,
    skill: SkillDefinition,
    skill_tags: set[str],
    hero: Combatant,
    target: Combatant,
) -> int:
    damage = combatant_damage_max(skill, hero)
    if has_trait(hero, PERSONAL_OPPORTUNIST) and Tag.MARKED in target.tags:
        damage += 1
    if "vulnerable_bonus" in skill_tags and _is_vulnerable(target):
        damage += 2
    if "exposed_bonus" in skill_tags and _is_exposed_backliner(state, target.actor_id):
        damage += 2
    if "basic" in skill_tags and Tag.MARKED in target.tags:
        damage += 1
    if "shock" in skill_tags and Tag.WET in target.tags:
        damage += 2
    return max(0, damage)


def _expected_hero_damage(
    state: CombatState,
    skill: SkillDefinition,
    hero: Combatant,
    target: Combatant,
    estimated_damage: int,
) -> int:
    if target.team != Team.ENEMY or estimated_damage <= 0:
        return 0
    hit_chance = skill.accuracy + hero.accuracy - target.defense + cover_penalty(
        state,
        target.actor_id,
        skill.attack_type,
    )
    return estimated_damage * max(0, min(100, hit_chance))


def _enemy_can_mark(enemy: Combatant, definitions: GameDefinitions) -> bool:
    return _enemy_has_skill_tag(enemy, definitions, {"mark", "mark_target"})


def _enemy_can_pay_off_mark(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
    marked_hero_id: str,
) -> bool:
    if not _enemy_has_skill_tag(enemy, definitions, {"vulnerable_bonus", "exploit_vulnerable"}):
        return False
    for skill_id in enemy.skills:
        skill = definitions.skills[skill_id]
        if not {"vulnerable_bonus", "exploit_vulnerable"}.intersection(skill.tags):
            continue
        if enemy.effort < skill.effort_cost:
            continue
        if not can_use_skill_from_position(state, enemy.actor_id, skill):
            continue
        if marked_hero_id in legal_targets(state, enemy.actor_id, skill.attack_type):
            return True
    return False


def _enemy_has_skill_tag(
    enemy: Combatant,
    definitions: GameDefinitions,
    tags: set[str],
) -> bool:
    for skill_id in enemy.skills:
        if tags.intersection(definitions.skills[skill_id].tags):
            return True
    return False


def _is_vulnerable(target: Combatant) -> bool:
    return Tag.MARKED in target.tags or target.hp < target.max_hp


def _is_exposed_backliner(state: CombatState, target_id: str) -> bool:
    target = state.actor(target_id)
    formation = state.formation_for(target.team)
    slot = formation.slot_of(target_id)
    return slot is not None and is_back(slot) and formation.is_exposed(
        target_id,
        state.side_for(target.team),
    )


_TACTICAL_TIER_HEAL = 100
_TACTICAL_TIER_KILL_URGENT_PAYOFF = 90
_TACTICAL_TIER_KILL_PAYOFF = 85
_TACTICAL_TIER_KILL_SETUP = 82
_TACTICAL_TIER_KILL_GENERIC = 80
_TACTICAL_TIER_NONLETHAL_URGENT_PAYOFF = 50
_TACTICAL_TIER_NONLETHAL_PACKAGE_VALUE = 35
_TACTICAL_TIER_ZERO_EFFORT_DAMAGE = 30
_TACTICAL_TIER_EFFORT_KILL = 20


def _enemy_max_payoff_damage_to_hero(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
    hero_id: str,
) -> int:
    max_damage = 0
    for skill_id in enemy.skills:
        skill = definitions.skills[skill_id]
        if not {"vulnerable_bonus", "exploit_vulnerable"}.intersection(skill.tags):
            continue
        if enemy.effort < skill.effort_cost:
            continue
        if not can_use_skill_from_position(state, enemy.actor_id, skill):
            continue
        if hero_id not in legal_targets(state, enemy.actor_id, skill.attack_type):
            continue
        max_damage = max(max_damage, combatant_damage_max(skill, enemy))
    return max_damage


def _payoff_threat_is_urgent(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
    marked_hero_id: str,
) -> bool:
    if not _enemy_can_pay_off_mark(state, definitions, enemy, marked_hero_id):
        return False
    marked_hero = state.actor(marked_hero_id)
    return _enemy_max_payoff_damage_to_hero(
        state,
        definitions,
        enemy,
        marked_hero_id,
    ) >= marked_hero.hp


def _enemy_action_economy_value(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
) -> int:
    value = enemy.speed
    if _enemy_has_skill_tag(
        enemy,
        definitions,
        {"vulnerable_bonus", "exploit_vulnerable"},
    ):
        value += 15
    if _is_exposed_backliner(state, enemy.actor_id):
        value += 5
    return value


def _any_zero_effort_kill(candidates: Sequence[HeroActionCandidate]) -> bool:
    return any(
        candidate.killable and candidate.effort_cost == 0 and candidate.estimated_damage > 0
        for candidate in candidates
    )


def _any_killable_candidate(candidates: Sequence[HeroActionCandidate]) -> bool:
    return any(candidate.killable and candidate.estimated_damage > 0 for candidate in candidates)


def _is_decisive_effort_use(
    candidate: HeroActionCandidate,
    candidates: Sequence[HeroActionCandidate],
) -> bool:
    if candidate.effort_cost == 0:
        return True
    if candidate.killable and candidate.estimated_damage > 0:
        return True
    return False


def _is_nonlethal_package_target(
    state: CombatState,
    definitions: GameDefinitions,
    target: Combatant,
    marked_hero_id: str | None,
) -> bool:
    if target.team != Team.ENEMY:
        return False
    if marked_hero_id is not None and _enemy_can_pay_off_mark(
        state,
        definitions,
        target,
        marked_hero_id,
    ):
        return True
    return _enemy_can_mark(target, definitions)


def _nonlethal_package_leaves_low_hp(
    candidate: HeroActionCandidate,
    target: Combatant,
) -> bool:
    remaining = target.hp - candidate.estimated_damage
    return remaining in (1, 2)


def _has_comparable_generic_zero_effort_damage(
    candidate: HeroActionCandidate,
    candidates: Sequence[HeroActionCandidate],
    state: CombatState,
    definitions: GameDefinitions,
    *,
    marked_hero_id: str | None,
) -> bool:
    for other in candidates:
        if other is candidate or other.is_heal or other.estimated_damage <= 0:
            continue
        if other.effort_cost != 0:
            continue
        other_target = state.actor(other.target_id)
        if other_target.team != Team.ENEMY:
            continue
        if _is_nonlethal_package_target(
            state,
            definitions,
            other_target,
            marked_hero_id,
        ):
            continue
        if other.expected_damage >= candidate.expected_damage:
            return True
    return False


def _nonlethal_package_beats_generic(
    candidate: HeroActionCandidate,
    state: CombatState,
    definitions: GameDefinitions,
    *,
    marked_hero_id: str | None,
    candidates: Sequence[HeroActionCandidate],
) -> bool:
    # Future tuning: party-wide kill setup visibility.
    target = state.actor(candidate.target_id)
    if _nonlethal_package_leaves_low_hp(candidate, target):
        return True
    if (
        marked_hero_id is not None
        and _payoff_threat_is_urgent(state, definitions, target, marked_hero_id)
    ):
        return True
    return not _has_comparable_generic_zero_effort_damage(
        candidate,
        candidates,
        state,
        definitions,
        marked_hero_id=marked_hero_id,
    )


def _is_nonlethal_package_fixation(
    candidate: HeroActionCandidate,
    candidates: Sequence[HeroActionCandidate],
    state: CombatState,
    definitions: GameDefinitions,
    marked_hero_id: str | None,
) -> bool:
    if candidate.killable or candidate.estimated_damage <= 0:
        return False
    target = state.actor(candidate.target_id)
    if not _is_nonlethal_package_target(state, definitions, target, marked_hero_id):
        return False
    if not _any_killable_candidate(candidates) and not _any_zero_effort_kill(candidates):
        if (
            marked_hero_id is not None
            and _payoff_threat_is_urgent(state, definitions, target, marked_hero_id)
            and _enemy_can_pay_off_mark(state, definitions, target, marked_hero_id)
            and candidate.effort_cost == 0
        ):
            return False
        return False
    return True


def _tactical_score_tiebreak(candidate: HeroActionCandidate) -> tuple[int, int, int, str]:
    return (
        candidate.skill_order,
        candidate.target_order,
        -candidate.expected_damage,
        candidate.target_id,
    )


def _tactical_action_score(
    candidate: HeroActionCandidate,
    state: CombatState,
    definitions: GameDefinitions,
    *,
    marked_hero: Combatant | None,
    candidates: Sequence[HeroActionCandidate],
) -> HeroActionScore | None:
    target = state.actor(candidate.target_id)
    marked_hero_id = marked_hero.actor_id if marked_hero is not None else None

    if candidate.is_heal and _is_urgent_ally(target):
        return (
            _TACTICAL_TIER_HEAL,
            _heal_priority(target),
            -candidate.effort_cost,
            *_tactical_score_tiebreak(candidate),
        )

    if candidate.estimated_damage <= 0:
        return None

    if candidate.effort_cost > 0 and not _is_decisive_effort_use(candidate, candidates):
        return None

    if _is_nonlethal_package_fixation(
        candidate,
        candidates,
        state,
        definitions,
        marked_hero_id,
    ):
        return None

    is_payoff = (
        marked_hero_id is not None
        and target.team == Team.ENEMY
        and _enemy_can_pay_off_mark(state, definitions, target, marked_hero_id)
    )
    is_setup = target.team == Team.ENEMY and _enemy_can_mark(target, definitions)
    is_urgent_payoff = (
        marked_hero_id is not None
        and target.team == Team.ENEMY
        and _payoff_threat_is_urgent(state, definitions, target, marked_hero_id)
    )

    if candidate.killable:
        if is_urgent_payoff:
            tier = _TACTICAL_TIER_KILL_URGENT_PAYOFF
        elif is_payoff:
            tier = _TACTICAL_TIER_KILL_PAYOFF
        elif is_setup:
            tier = _TACTICAL_TIER_KILL_SETUP
        else:
            tier = _TACTICAL_TIER_KILL_GENERIC
        subscore = _enemy_action_economy_value(state, definitions, target)
        if tier == _TACTICAL_TIER_KILL_SETUP and marked_hero_id is not None:
            subscore += 10
        return (
            tier,
            subscore,
            -candidate.effort_cost,
            *_tactical_score_tiebreak(candidate),
        )

    if (
        marked_hero_id is not None
        and is_urgent_payoff
        and is_payoff
        and not _any_killable_candidate(candidates)
        and not _any_zero_effort_kill(candidates)
        and candidate.effort_cost == 0
    ):
        return (
            _TACTICAL_TIER_NONLETHAL_URGENT_PAYOFF,
            candidate.expected_damage,
            *_tactical_score_tiebreak(candidate),
        )

    if candidate.effort_cost == 0:
        if _is_nonlethal_package_target(
            state,
            definitions,
            target,
            marked_hero_id,
        ) and not candidate.killable:
            if not _nonlethal_package_beats_generic(
                candidate,
                state,
                definitions,
                marked_hero_id=marked_hero_id,
                candidates=candidates,
            ):
                return None
            subscore = candidate.expected_damage + _enemy_action_economy_value(
                state,
                definitions,
                target,
            )
            return (
                _TACTICAL_TIER_NONLETHAL_PACKAGE_VALUE,
                subscore,
                *_tactical_score_tiebreak(candidate),
            )
        subscore = candidate.expected_damage
        if target.team == Team.ENEMY and not _is_nonlethal_package_target(
            state,
            definitions,
            target,
            marked_hero_id,
        ):
            subscore += _enemy_action_economy_value(state, definitions, target)
        return (
            _TACTICAL_TIER_ZERO_EFFORT_DAMAGE,
            subscore,
            *_tactical_score_tiebreak(candidate),
        )

    return None


_COMPANY_TIER_HEAL = 100
_COMPANY_TIER_KILL_IMMEDIATE_THREAT = 90
_COMPANY_TIER_KILL_URGENT_PAYOFF = 90
_COMPANY_TIER_KILL_PAYOFF = 85
_COMPANY_TIER_KILL_GENERIC = 82
_COMPANY_TIER_KILL_SETUP = 78
_COMPANY_TIER_EFFORT_KILL = 70
_COMPANY_TIER_LOW_HP_SETUP = 55
_COMPANY_TIER_ZERO_EFFORT_DAMAGE = 45
_COMPANY_TIER_FALLBACK_DAMAGE = 20
_COMPANY_SETUP_KILL_BONUS = 12
_COMPANY_PAYOFF_KILL_BONUS = 6
_COMPANY_MAX_SUBSCORE = 40


def _company_survival_marked_hero(state: CombatState) -> Combatant | None:
    marked_heroes = [
        hero for hero in state.heroes.values() if hero.is_alive() and Tag.MARKED in hero.tags
    ]
    if not marked_heroes:
        return None
    return min(marked_heroes, key=lambda hero: (hero.hp, hero.actor_id))


def _is_company_survival_heal_urgent(
    hero: Combatant,
    _state: CombatState,
    _definitions: GameDefinitions,
) -> bool:
    return hero.is_downed() or _is_critical_ally(hero)


def _company_survival_should_score_heal(
    candidate: HeroActionCandidate,
    state: CombatState,
    definitions: GameDefinitions,
    *,
    candidates: Sequence[HeroActionCandidate],
) -> bool:
    target = state.actor(candidate.target_id)
    if not candidate.is_heal or not _is_company_survival_heal_urgent(target, state, definitions):
        return False
    if target.is_downed():
        return True
    return not _any_killable_candidate(candidates)


def _is_vulnerable_hero_for_immediate_threat(
    hero: Combatant,
    *,
    enemy_damage: int,
) -> bool:
    if enemy_damage >= hero.hp:
        return True
    return _is_critical_ally(hero) and enemy_damage > 0


def _enemy_max_damage_to_hero(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
    hero_id: str,
) -> int:
    max_damage = 0
    for skill_id in enemy.skills:
        skill = definitions.skills[skill_id]
        if enemy.effort < skill.effort_cost:
            continue
        if not can_use_skill_from_position(state, enemy.actor_id, skill):
            continue
        if hero_id not in legal_targets(state, enemy.actor_id, skill.attack_type):
            continue
        max_damage = max(max_damage, combatant_damage_max(skill, enemy))
    return max_damage


def _enemy_threatens_immediate_kill(
    state: CombatState,
    definitions: GameDefinitions,
    enemy: Combatant,
) -> bool:
    for hero in state.heroes.values():
        if not hero.is_alive():
            continue
        enemy_damage = _enemy_max_damage_to_hero(state, definitions, enemy, hero.actor_id)
        if _is_vulnerable_hero_for_immediate_threat(hero, enemy_damage=enemy_damage):
            return True
    return False


def _company_survival_kill_subscore(
    state: CombatState,
    definitions: GameDefinitions,
    target: Combatant,
    *,
    is_setup: bool,
    is_payoff: bool,
    marked_hero_id: str | None,
) -> int:
    subscore = _enemy_action_economy_value(state, definitions, target)
    if is_setup:
        subscore += _COMPANY_SETUP_KILL_BONUS
    elif is_payoff and marked_hero_id is not None:
        subscore += _COMPANY_PAYOFF_KILL_BONUS
    return min(subscore, _COMPANY_MAX_SUBSCORE)


def _company_survival_score_tuple(
    tier: int,
    subscore: int,
    candidate: HeroActionCandidate,
) -> HeroActionScore:
    return (
        tier,
        subscore,
        -candidate.effort_cost,
        *_tactical_score_tiebreak(candidate),
    )


def _company_survival_score_detail(
    candidate: HeroActionCandidate,
    state: CombatState,
    definitions: GameDefinitions,
    *,
    marked_hero: Combatant | None,
    candidates: Sequence[HeroActionCandidate],
) -> tuple[int, int, HeroActionScore] | None:
    target = state.actor(candidate.target_id)
    marked_hero_id = marked_hero.actor_id if marked_hero is not None else None

    if _company_survival_should_score_heal(
        candidate,
        state,
        definitions,
        candidates=candidates,
    ):
        subscore = _heal_priority(target)
        tier = _COMPANY_TIER_HEAL
        return (tier, subscore, _company_survival_score_tuple(tier, subscore, candidate))

    if candidate.estimated_damage <= 0:
        return None

    if candidate.effort_cost > 0 and not _is_decisive_effort_use(candidate, candidates):
        return None

    if _is_nonlethal_package_fixation(
        candidate,
        candidates,
        state,
        definitions,
        marked_hero_id,
    ):
        return None

    is_payoff = (
        marked_hero_id is not None
        and target.team == Team.ENEMY
        and _enemy_can_pay_off_mark(state, definitions, target, marked_hero_id)
    )
    is_setup = target.team == Team.ENEMY and _enemy_can_mark(target, definitions)
    is_urgent_payoff = (
        marked_hero_id is not None
        and target.team == Team.ENEMY
        and _payoff_threat_is_urgent(state, definitions, target, marked_hero_id)
    )

    if candidate.killable:
        tier = _COMPANY_TIER_KILL_GENERIC
        if target.team == Team.ENEMY and _enemy_threatens_immediate_kill(
            state,
            definitions,
            target,
        ):
            tier = _COMPANY_TIER_KILL_IMMEDIATE_THREAT
        elif is_urgent_payoff and is_payoff:
            tier = _COMPANY_TIER_KILL_URGENT_PAYOFF
        elif is_payoff:
            tier = _COMPANY_TIER_KILL_PAYOFF
        elif is_setup and marked_hero_id is not None:
            tier = _COMPANY_TIER_KILL_GENERIC
        elif is_setup:
            tier = _COMPANY_TIER_KILL_SETUP
        elif candidate.effort_cost > 0:
            tier = _COMPANY_TIER_EFFORT_KILL
        subscore = _company_survival_kill_subscore(
            state,
            definitions,
            target,
            is_setup=is_setup,
            is_payoff=is_payoff,
            marked_hero_id=marked_hero_id,
        )
        return (tier, subscore, _company_survival_score_tuple(tier, subscore, candidate))

    if _any_killable_candidate(candidates):
        return None

    if candidate.effort_cost > 0:
        return None

    if (
        target.team == Team.ENEMY
        and _nonlethal_package_leaves_low_hp(candidate, target)
        and _enemy_action_economy_value(state, definitions, target) >= 15
        and not _any_zero_effort_kill(candidates)
    ):
        tier = _COMPANY_TIER_LOW_HP_SETUP
        subscore = min(
            candidate.expected_damage
            + _enemy_action_economy_value(state, definitions, target),
            _COMPANY_MAX_SUBSCORE,
        )
        return (tier, subscore, _company_survival_score_tuple(tier, subscore, candidate))

    if candidate.effort_cost == 0 and candidate.estimated_damage > 0:
        if _is_nonlethal_package_target(
            state,
            definitions,
            target,
            marked_hero_id,
        ) and not candidate.killable:
            return None
        tier = _COMPANY_TIER_ZERO_EFFORT_DAMAGE
        subscore = candidate.expected_damage
        if target.team == Team.ENEMY:
            subscore += _enemy_action_economy_value(state, definitions, target)
        subscore = min(subscore, _COMPANY_MAX_SUBSCORE)
        return (tier, subscore, _company_survival_score_tuple(tier, subscore, candidate))

    tier = _COMPANY_TIER_FALLBACK_DAMAGE
    subscore = min(candidate.expected_damage, _COMPANY_MAX_SUBSCORE)
    return (tier, subscore, _company_survival_score_tuple(tier, subscore, candidate))


def _company_survival_action_score(
    candidate: HeroActionCandidate,
    state: CombatState,
    definitions: GameDefinitions,
    *,
    marked_hero: Combatant | None,
    candidates: Sequence[HeroActionCandidate],
) -> HeroActionScore | None:
    detail = _company_survival_score_detail(
        candidate,
        state,
        definitions,
        marked_hero=marked_hero,
        candidates=candidates,
    )
    if detail is None:
        return None
    return detail[2]


def _company_survival_score_entries(
    state: CombatState,
    definitions: GameDefinitions,
    hero_id: str,
) -> tuple[CompanySurvivalScoreEntry, ...]:
    candidates = _hero_action_candidates(state, definitions, hero_id)
    marked_hero = _company_survival_marked_hero(state)
    marked_hero_id = marked_hero.actor_id if marked_hero is not None else None
    killable_opportunities = sum(1 for candidate in candidates if candidate.killable)
    entries: list[CompanySurvivalScoreEntry] = []
    for candidate in candidates:
        detail = _company_survival_score_detail(
            candidate,
            state,
            definitions,
            marked_hero=marked_hero,
            candidates=candidates,
        )
        if detail is None:
            continue
        tier, subscore, score = detail
        entries.append(
            CompanySurvivalScoreEntry(
                skill_id=candidate.skill_id,
                target_id=candidate.target_id,
                tier=tier,
                subscore=subscore,
                score=score,
                killable=candidate.killable,
                effort_cost=candidate.effort_cost,
                package_target=_hero_package_target_kind(
                    state,
                    definitions,
                    candidate.target_id,
                    marked_hero_id,
                ),
                killable_opportunities=killable_opportunities,
            )
        )
    return tuple(sorted(entries, key=lambda entry: entry.score, reverse=True))


def _explain_company_survival_choice(
    state: CombatState,
    definitions: GameDefinitions,
    hero_id: str,
) -> tuple[tuple[str, str] | None, tuple[CompanySurvivalScoreEntry, ...]]:
    entries = _company_survival_score_entries(state, definitions, hero_id)
    if not entries:
        return None, ()
    chosen = entries[0]
    return (chosen.skill_id, chosen.target_id), entries[:3]


def _stable_policy_index(encounter_id: str, seed: int | None) -> int:
    return (seed or 0) + sum(ord(character) for character in encounter_id)


def _marked_hero_id_for_package(state: CombatState) -> str | None:
    marked_heroes = [
        hero for hero in state.heroes.values() if hero.is_alive() and Tag.MARKED in hero.tags
    ]
    if not marked_heroes:
        return None
    return min(marked_heroes, key=lambda hero: (hero.hp, hero.actor_id)).actor_id


def _hero_package_target_kind(
    state: CombatState,
    definitions: GameDefinitions,
    target_id: str,
    marked_hero_id: str | None,
) -> str:
    target = state.actor(target_id)
    if target.team != Team.ENEMY:
        return ""
    if marked_hero_id is not None and _enemy_can_pay_off_mark(
        state,
        definitions,
        target,
        marked_hero_id,
    ):
        return "payoff"
    if _enemy_can_mark(target, definitions):
        return "setup"
    return ""


def _hero_policy_action_record(
    state: CombatState,
    definitions: GameDefinitions,
    *,
    hero_id: str,
    round_number: int,
    skill_id: str,
    target_id: str,
    candidates: Sequence[HeroActionCandidate],
    hero_events: Sequence[GameEvent],
) -> HeroPolicyActionRecord:
    chosen = next(
        (
            candidate
            for candidate in candidates
            if candidate.skill_id == skill_id and candidate.target_id == target_id
        ),
        None,
    )
    if chosen is not None:
        effort_cost = chosen.effort_cost
        estimated_damage = chosen.estimated_damage
        killable = chosen.killable
        is_heal = chosen.is_heal
    else:
        skill = definitions.skills[skill_id]
        effort_cost = skill.effort_cost
        target = state.actor(target_id)
        is_heal = _is_treatment_skill(skill.tags)
        estimated_damage = 0
        killable = False
        if target.team == Team.ENEMY and not is_heal:
            estimated_damage = _estimated_hero_damage(
                state,
                skill,
                set(skill.tags),
                state.actor(hero_id),
                target,
            )
            killable = estimated_damage >= target.hp

    killable_opportunities = sum(1 for candidate in candidates if candidate.killable)
    marked_hero_id = _marked_hero_id_for_package(state)
    marked_hero_present = marked_hero_id is not None
    produced_kill = any(
        isinstance(event, DeathEvent) and event.actor_id == target_id for event in hero_events
    )
    target = state.actor(target_id)
    target_hp_remaining = target.hp if target.team == Team.ENEMY else 0
    package_target = _hero_package_target_kind(
        state,
        definitions,
        target_id,
        marked_hero_id,
    )
    ignored_killable_opportunity = (
        killable_opportunities > 0 and not killable and not is_heal
    )
    return HeroPolicyActionRecord(
        hero_id=hero_id,
        round_number=round_number,
        skill_id=skill_id,
        target_id=target_id,
        effort_cost=effort_cost,
        estimated_damage=estimated_damage,
        killable=killable,
        is_heal=is_heal,
        killable_opportunities=killable_opportunities,
        ignored_killable_opportunity=ignored_killable_opportunity,
        package_target=package_target,
        marked_hero_present=marked_hero_present,
        produced_kill=produced_kill,
        target_hp_remaining=target_hp_remaining,
    )


__all__ = [
    "AntiMarkHeroPolicy",
    "DamageRaceHeroPolicy",
    "EnemyDecisionEpisode",
    "EnemyDecisionRecord",
    "EnemyPressureMetrics",
    "GuardFlowMetrics",
    "HeroActionCandidate",
    "HeroDecisionPolicy",
    "HeroPolicyActionRecord",
    "LinearEnemyDecisionPolicy",
    "MixedHeroPolicy",
    "NaiveHeroPolicy",
    "PartyCollapseRewardWeights",
    "SUPPORTED_ENEMY_MOVEMENT_MODES",
    "SUPPORTED_ENEMY_WAIT_MODES",
    "SUPPORTED_HERO_POLICY_IDS",
    "CompanySurvivalHeroPolicy",
    "ConservativeHeroPolicy",
    "SurvivalHeroPolicy",
    "TacticalHeroPolicy",
    "create_hero_policy",
    "learn_linear_enemy_weights",
    "run_enemy_learning_episode",
    "score_enemy_action_events",
    "score_enemy_episode",
    "score_enemy_timing_outcome",
]
