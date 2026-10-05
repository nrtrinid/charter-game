# ruff: noqa: E501
"""Checkpoint 5 evaluation-only audit helpers.

This module records factual decision context around frozen enemy policies without
changing simulator legality, combat resolution, or policy choices.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean, median
from typing import Any

from game.combat.combat_state import CombatState, Team
from game.combat.enemy_decision import (
    EnemyDecisionCandidate,
    EnemyDecisionPolicy,
    EnemyDecisionRuntimeContext,
    explain_enemy_decision,
)
from game.combat.enemy_learning import EnemyDecisionEpisode
from game.content.definitions import GameDefinitions
from game.core.events import DeathEvent, DownedEvent, StatusChangedEvent
from game.ml.counterfactual import decision_state_digest, outcome_from_episode
from game.ml.outcome_features import CP3_FEATURE_NAMES, CP3_FEATURE_SCHEMA

_DEFAULT_CONTEXT = EnemyDecisionRuntimeContext()
_Z95 = 1.959963984540054


def choice_key(choice: EnemyDecisionCandidate | None) -> str | None:
    if choice is None:
        return None
    return f"{choice.skill_id}|{choice.target_id}"


def wilson_interval(successes: int, count: int) -> list[float]:
    if count <= 0:
        return [0.0, 0.0]
    p = successes / count
    denom = 1.0 + _Z95 * _Z95 / count
    center = (p + _Z95 * _Z95 / (2 * count)) / denom
    half = _Z95 * math.sqrt(p * (1 - p) / count + _Z95 * _Z95 / (4 * count * count)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def mcnemar_exact(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    smaller = min(first_only, second_only)
    tail = sum(math.comb(discordant, i) for i in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_difference_ci(differences: Sequence[int | float]) -> list[float]:
    """Normal 95% CI for the mean paired binary difference, clipped to [-1, 1]."""
    values = [float(value) for value in differences]
    if not values:
        return [0.0, 0.0]
    center = mean(values)
    if len(values) == 1:
        return [center, center]
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    half = _Z95 * math.sqrt(variance / len(values))
    return [max(-1.0, center - half), min(1.0, center + half)]


def classify_advantage(comparison: Mapping[str, Any]) -> str:
    diff = float(comparison["win_rate_difference"])
    low, high = (float(x) for x in comparison["paired_difference_ci_95"])
    p = float(comparison["mcnemar_exact_two_sided_p"])
    if diff > 0 and low > 0 and p < 0.05:
        return "CONFIRMED"
    if diff > 0:
        return "DIRECTIONALLY_REPLICATED"
    if diff < 0 and high < 0 and p < 0.05:
        return "CONTRADICTED"
    if diff < 0:
        return "NOT_REPLICATED"
    return "INCONCLUSIVE"


def classify_equivalence(comparison: Mapping[str, Any], *, margin: float = 0.10) -> str:
    diff = float(comparison["win_rate_difference"])
    low, high = (float(x) for x in comparison["paired_difference_ci_95"])
    p = float(comparison["mcnemar_exact_two_sided_p"])
    if diff > 0 and low > 0 and p < 0.05:
        return "CLEARLY_STRONGER"
    if diff < 0 and high < 0 and p < 0.05:
        return "CLEARLY_WEAKER"
    if low >= -margin and high <= margin:
        return "APPROXIMATELY_EQUIVALENT"
    return "INCONCLUSIVE"


def classify_reduced_behavior(pi0: float, pi1: float, pi2: float) -> str:
    if pi0 <= 0:
        return "INCONCLUSIVE"
    ratios = (pi1 / pi0, pi2 / pi0)
    if max(ratios) <= 0.75:
        return "CONFIRMED"
    if max(ratios) < 1.0:
        return "DIRECTIONALLY_REPLICATED"
    if min(ratios) < 1.0:
        return "INCONCLUSIVE"
    return "NOT_REPLICATED"


def classify_direct_pressure(pi0: float, pi1: float, pi2: float) -> str:
    if pi1 >= pi0 and pi2 >= pi0:
        return "CONFIRMED"
    if (pi1 + pi2) / 2 >= pi0:
        return "DIRECTIONALLY_REPLICATED"
    if max(pi1, pi2) >= pi0:
        return "INCONCLUSIVE"
    return "NOT_REPLICATED"


def _band(value: float, maximum: float) -> str:
    if maximum <= 0:
        return "none"
    ratio = value / maximum
    if ratio <= 1 / 3:
        return "low"
    if ratio <= 2 / 3:
        return "mid"
    return "high"


class AuditRecordingPolicy:
    """Decorate one frozen policy while evaluating all frozen choices on each state."""

    def __init__(
        self,
        *,
        executed_name: str,
        policies: Mapping[str, EnemyDecisionPolicy],
    ) -> None:
        if executed_name not in policies:
            raise ValueError(f"Unknown executed policy: {executed_name}")
        self.executed_name = executed_name
        self.policies = dict(policies)
        self.decisions: list[dict[str, Any]] = []
        self._action_index = 0

    def choose(
        self,
        state: CombatState,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_CONTEXT,
    ) -> EnemyDecisionCandidate | None:
        trace = explain_enemy_decision(state, definitions, enemy_id, runtime_context)
        choices = {
            name: policy.choose(state, definitions, enemy_id, runtime_context)
            for name, policy in self.policies.items()
        }
        chosen = choices[self.executed_name]
        if chosen is None:
            return None
        if trace is None or not trace.candidates:
            raise ValueError("Frozen policy chose outside an available decision trace.")
        legal_keys = {choice_key(candidate) for candidate in trace.candidates}
        if choice_key(chosen) not in legal_keys:
            raise ValueError("Frozen policy chose outside the simulator legal candidate set.")
        if any(choice is not None and choice_key(choice) not in legal_keys for choice in choices.values()):
            raise ValueError("Comparison policy chose outside the simulator legal candidate set.")

        observation_by_key: dict[str, dict[str, float]] = {}
        for candidate in trace.candidates:
            encoded = CP3_FEATURE_SCHEMA.encode(state, definitions, candidate, runtime_context)
            observation_by_key[str(choice_key(candidate))] = {
                name: float(value) for name, value in zip(CP3_FEATURE_NAMES, encoded, strict=True)
            }
        chosen_obs = observation_by_key[str(choice_key(chosen))]
        hero_candidate_obs = [
            observation_by_key[str(choice_key(candidate))]
            for candidate in trace.candidates
            if observation_by_key[str(choice_key(candidate))]["target_is_hero"] > 0.5
        ]
        min_legal_hp = min((item["target_hp"] for item in hero_candidate_obs), default=chosen_obs["target_hp"])
        target = state.actor(chosen.target_id)
        party_hp_max = sum(hero.max_hp for hero in state.heroes.values())
        party_effort_max = sum(hero.max_effort for hero in state.heroes.values())
        self.decisions.append(
            {
                "action_index": self._action_index,
                "round": state.round_number,
                "state_digest": decision_state_digest(state, runtime_context),
                "enemy_id": enemy_id,
                "enemy_class": state.actor(enemy_id).class_id,
                "chosen_skill": chosen.skill_id,
                "chosen_target": chosen.target_id,
                "chosen_target_class": target.class_id,
                "chosen_target_team": target.team.value,
                "choices": {name: choice_key(choice) for name, choice in choices.items()},
                "legal_candidate_count": len(trace.candidates),
                "party_hp": chosen_obs["party_hp_total"],
                "party_hp_band": _band(chosen_obs["party_hp_total"], party_hp_max),
                "party_effort": chosen_obs["party_effort_total"],
                "party_effort_band": _band(chosen_obs["party_effort_total"], party_effort_max),
                "marked_hero_count": chosen_obs["marked_hero_count"],
                "target_hp": chosen_obs["target_hp"],
                "target_life_state": (
                    "dead" if chosen_obs["target_is_dead"] else "downed" if chosen_obs["target_is_downed"] else "alive"
                ),
                "target_marked": bool(chosen_obs["target_tag_marked"]),
                "target_protected": bool(chosen_obs["target_is_protected"]),
                "target_support_capable": bool(
                    chosen_obs["target_heal_skill_count"]
                    or chosen_obs["target_support_skill_count"]
                    or chosen_obs["target_recovery_skill_count"]
                ),
                "target_is_lowest_hp_legal": bool(chosen_obs["target_is_hero"] and chosen_obs["target_hp"] <= min_legal_hp),
                "target_slot": _slot_name(chosen_obs),
                "direct_pressure": bool(chosen_obs["projected_damage_max"] > 0),
                "effort_drain_action": bool(chosen_obs["skill_tag_effort_drain"]),
                "mark_action": bool(chosen_obs["skill_tag_mark"]),
                "guard_action": bool(chosen_obs["skill_tag_guard"]),
                "grab_action": bool(chosen_obs["skill_tag_drag_forward"]),
                "projected_damage_max": chosen_obs["projected_damage_max"],
                "down_caused": False,
                "death_caused": False,
                "mortal_wounds_caused": 0,
            }
        )
        self._action_index += 1
        return chosen


def _slot_name(obs: Mapping[str, float]) -> str:
    for name in ("back_left", "back_right", "front_left", "front_right"):
        if obs[f"target_slot_{name}"] > 0.5:
            return name
    return "none"


def attach_decision_events(
    decisions: list[dict[str, Any]],
    episode: EnemyDecisionEpisode,
) -> None:
    by_index = {int(record.action_index): record for record in episode.records}
    for decision in decisions:
        record = by_index.get(int(decision["action_index"]))
        if record is None:
            continue
        target_id = str(decision["chosen_target"])
        decision["down_caused"] = any(
            isinstance(event, DownedEvent) and event.actor_id == target_id for event in record.events
        )
        decision["death_caused"] = any(
            isinstance(event, DeathEvent) and event.actor_id == target_id for event in record.events
        )
        decision["mortal_wounds_caused"] = sum(
            1
            for event in record.events
            if isinstance(event, StatusChangedEvent)
            and event.actor_id == target_id
            and event.status == "mortal_wound"
            and event.added
        )


def summarize_case(
    *,
    case_id: str,
    group: str,
    encounter_id: str,
    hero_policy_id: str,
    seed: int,
    policy_name: str,
    state: CombatState,
    episode: EnemyDecisionEpisode,
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    outcome = outcome_from_episode(state, episode)
    hostile = [row for row in decisions if row["chosen_target_team"] == Team.HERO.value]
    target_counts = Counter(str(row["chosen_target"]) for row in hostile)
    top_share = max(target_counts.values(), default=0) / len(hostile) if hostile else 0.0
    hhi = sum((count / len(hostile)) ** 2 for count in target_counts.values()) if hostile else 0.0
    repeats = 0
    switches = 0
    last_by_enemy: dict[str, str] = {}
    for row in hostile:
        actor = str(row["enemy_id"])
        target = str(row["chosen_target"])
        if actor in last_by_enemy:
            if last_by_enemy[actor] == target:
                repeats += 1
            else:
                switches += 1
        last_by_enemy[actor] = target
    direct_pressure = sum(bool(row["direct_pressure"]) for row in decisions)
    target_classes = Counter(str(row["chosen_target_class"]) for row in hostile)
    skill_uses = Counter(str(row["chosen_skill"]) for row in decisions)
    return {
        "case_id": case_id,
        "group": group,
        "encounter_id": encounter_id,
        "hero_policy_id": hero_policy_id,
        "seed": seed,
        "policy": policy_name,
        "enemy_win": int(outcome.enemy_win),
        "terminal_outcome": episode.final_victor,
        "final_hero_hp": outcome.final_hero_hp,
        "final_hero_effort": outcome.final_hero_effort,
        "hero_deaths": outcome.hero_deaths,
        "terminal_hero_downs": outcome.terminal_hero_downs,
        "mortal_wounds": outcome.mortal_wounds,
        "enemy_caused_effort_drain": outcome.enemy_caused_effort_drain,
        "rounds": outcome.rounds_elapsed,
        "enemy_actions": episode.metrics.enemy_decisions,
        "marks_applied": episode.metrics.marks_applied,
        "guard_uses": episode.metrics.guard_actions,
        "grab_uses": episode.metrics.boss_sequence.grab_uses,
        "bite_uses": episode.metrics.boss_sequence.bite_uses,
        "effort_drain_uses": int(episode.metrics.skill_uses.get("effort_drain", 0)),
        "direct_pressure_uses": direct_pressure,
        "direct_pressure_rate": direct_pressure / len(decisions) if decisions else 0.0,
        "enemy_waits": episode.metrics.enemy_waits,
        "enemy_moves": episode.metrics.enemy_recovery_moves + episode.metrics.enemy_proactive_moves,
        "mark_payoffs": episode.metrics.mark_flow.exploited_by_enemy_hit,
        "mark_multi_hit_focus": episode.metrics.mark_flow.multi_hit_focus,
        "grab_to_bite_same_target": episode.metrics.boss_sequence.grab_to_bite_same_target,
        "guard_payoffs": episode.metrics.guard_flow.guarded_ally_used_payoff_after_guard,
        "target_top_share": top_share,
        "target_hhi": hhi,
        "repeat_target_transitions": repeats,
        "target_switches": switches,
        "lowest_hp_attacks": sum(bool(row["target_is_lowest_hp_legal"]) for row in hostile),
        "support_target_attacks": sum(bool(row["target_support_capable"]) for row in hostile),
        "downed_target_attacks": sum(row["target_life_state"] == "downed" for row in hostile),
        "protected_target_attacks": sum(bool(row["target_protected"]) for row in hostile),
        "hostile_target_actions": len(hostile),
        "downs_caused": sum(bool(row["down_caused"]) for row in hostile),
        "deaths_caused": sum(bool(row["death_caused"]) for row in hostile),
        "mortal_wounds_caused": sum(int(row["mortal_wounds_caused"]) for row in hostile),
        "target_class_counts": dict(sorted(target_classes.items())),
        "skill_uses": dict(sorted(skill_uses.items())),
    }


def aggregate_cases(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    if not items:
        return {"case_count": 0}
    wins = sum(int(row["enemy_win"]) for row in items)
    scalar_mean_keys = (
        "final_hero_hp", "final_hero_effort", "hero_deaths", "terminal_hero_downs",
        "mortal_wounds", "enemy_caused_effort_drain", "rounds", "enemy_actions",
        "marks_applied", "guard_uses", "grab_uses", "bite_uses", "effort_drain_uses",
        "direct_pressure_uses", "direct_pressure_rate", "enemy_waits", "enemy_moves",
        "mark_payoffs", "mark_multi_hit_focus", "grab_to_bite_same_target", "guard_payoffs",
        "target_top_share", "target_hhi", "repeat_target_transitions", "target_switches",
        "lowest_hp_attacks", "support_target_attacks", "downed_target_attacks",
        "protected_target_attacks", "hostile_target_actions", "downs_caused", "deaths_caused",
        "mortal_wounds_caused",
    )
    skill_totals: Counter[str] = Counter()
    class_totals: Counter[str] = Counter()
    for row in items:
        skill_totals.update({str(k): int(v) for k, v in row["skill_uses"].items()})
        class_totals.update({str(k): int(v) for k, v in row["target_class_counts"].items()})
    hostile = sum(int(row["hostile_target_actions"]) for row in items)
    return {
        "case_count": len(items),
        "enemy_win_count": wins,
        "enemy_win_rate": wins / len(items),
        "enemy_win_rate_wilson_95": wilson_interval(wins, len(items)),
        "mean_final_hero_hp": mean(float(row["final_hero_hp"]) for row in items),
        "median_final_hero_hp": median(float(row["final_hero_hp"]) for row in items),
        "terminal_outcome_counts": dict(sorted(Counter(str(row["terminal_outcome"]) for row in items).items())),
        **{f"mean_{key}": mean(float(row[key]) for row in items) for key in scalar_mean_keys if key != "final_hero_hp"},
        "targeting_rates": {
            "lowest_hp_legal": sum(int(row["lowest_hp_attacks"]) for row in items) / hostile if hostile else 0.0,
            "support_capable": sum(int(row["support_target_attacks"]) for row in items) / hostile if hostile else 0.0,
            "downed": sum(int(row["downed_target_attacks"]) for row in items) / hostile if hostile else 0.0,
            "protected": sum(int(row["protected_target_attacks"]) for row in items) / hostile if hostile else 0.0,
        },
        "mean_skill_uses_per_case": {key: value / len(items) for key, value in sorted(skill_totals.items())},
        "target_class_distribution": {
            key: value / hostile for key, value in sorted(class_totals.items())
        } if hostile else {},
    }


def paired_policy_comparison(
    baseline_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    first = {str(row["case_id"]): row for row in baseline_rows}
    second = {str(row["case_id"]): row for row in comparison_rows}
    if first.keys() != second.keys():
        raise ValueError("Paired policy cases differ.")
    both_loss = both_win = first_only = second_only = 0
    differences: list[int] = []
    for case_id in sorted(first):
        a = bool(first[case_id]["enemy_win"])
        b = bool(second[case_id]["enemy_win"])
        differences.append(int(b) - int(a))
        if a and b:
            both_win += 1
        elif a:
            first_only += 1
        elif b:
            second_only += 1
        else:
            both_loss += 1
    n = len(differences)
    return {
        "case_count": n,
        "both_loss": both_loss,
        "baseline_only_win": first_only,
        "comparison_only_win": second_only,
        "both_win": both_win,
        "discordant_pair_count": first_only + second_only,
        "win_rate_difference": mean(differences) if differences else 0.0,
        "paired_difference_ci_95": paired_difference_ci(differences),
        "mcnemar_exact_two_sided_p": mcnemar_exact(first_only, second_only),
        "transition_matrix": {
            "baseline_loss/comparison_loss": both_loss,
            "baseline_loss/comparison_win": second_only,
            "baseline_win/comparison_loss": first_only,
            "baseline_win/comparison_win": both_win,
        },
    }


def disagreement_audit(decision_rows: Sequence[Mapping[str, Any]], pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for a, b in pairs:
        total = 0
        disagreements = 0
        by_encounter: Counter[str] = Counter()
        by_round: Counter[str] = Counter()
        by_actor: Counter[str] = Counter()
        by_skill_motif: Counter[str] = Counter()
        by_target_change: Counter[str] = Counter()
        by_hp_band: Counter[str] = Counter()
        by_effort_band: Counter[str] = Counter()
        by_marked: Counter[str] = Counter()
        by_life: Counter[str] = Counter()
        by_protection: Counter[str] = Counter()
        samples: list[dict[str, Any]] = []
        for row in decision_rows:
            choices = row["choices"]
            ca = choices.get(a)
            cb = choices.get(b)
            if ca is None or cb is None:
                continue
            total += 1
            if ca == cb:
                continue
            disagreements += 1
            by_encounter[str(row["encounter_id"])] += 1
            by_round[str(row["round"])] += 1
            by_actor[str(row["enemy_class"])] += 1
            skill_a, target_a = str(ca).split("|", 1)
            skill_b, target_b = str(cb).split("|", 1)
            by_skill_motif[f"{skill_a}->{skill_b}"] += 1
            by_target_change["same_target" if target_a == target_b else "different_target"] += 1
            by_hp_band[str(row["party_hp_band"])] += 1
            by_effort_band[str(row["party_effort_band"])] += 1
            by_marked["marked_present" if float(row["marked_hero_count"]) > 0 else "no_mark"] += 1
            by_life[str(row["target_life_state"])] += 1
            by_protection["protected" if row["target_protected"] else "unprotected"] += 1
            samples.append(dict(row))
        result[f"{a}_vs_{b}"] = {
            "comparable_states": total,
            "disagreements": disagreements,
            "disagreement_rate": disagreements / total if total else 0.0,
            "by_encounter": dict(by_encounter.most_common()),
            "by_round": dict(by_round.most_common()),
            "by_actor": dict(by_actor.most_common()),
            "skill_motifs": dict(by_skill_motif.most_common(25)),
            "target_change": dict(by_target_change),
            "party_hp_band": dict(by_hp_band),
            "party_effort_band": dict(by_effort_band),
            "marked_state": dict(by_marked),
            "target_life_state": dict(by_life),
            "protection_state": dict(by_protection),
        }
    return result


def stable_disagreement_sample(
    decision_rows: Sequence[Mapping[str, Any]],
    *,
    per_group: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    seen: set[str] = set()
    for row in decision_rows:
        choices = [value for value in row["choices"].values() if value is not None]
        if len(set(choices)) <= 1:
            continue
        provenance = "|".join(
            str(row[key])
            for key in ("group", "trajectory_policy", "encounter_id", "hero_policy_id", "seed", "action_index", "state_digest")
        )
        if provenance in seen:
            continue
        seen.add(provenance)
        digest = hashlib.sha256(provenance.encode("utf-8")).hexdigest()
        grouped[str(row["group"])].append((digest, dict(row)))
    selected: list[dict[str, Any]] = []
    for group in sorted(grouped):
        for digest, row in sorted(grouped[group], key=lambda item: item[0])[:per_group]:
            selected.append({"sample_hash": digest, **row})
    return selected
