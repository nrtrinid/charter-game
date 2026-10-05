"""Research-only stateful route evaluation over production combat transitions.

Routes own only encounter order and deterministic seed authority. Hero persistence
is delegated to the production create/sync abstractions used by interactive play.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any

from game.campaign.company import CompanyState, HeroState, create_new_company
from game.campaign.gear import effective_hero_stats
from game.campaign.roster import active_roster, sync_company_from_combat
from game.combat.combat_state import CombatState, LifeState
from game.combat.enemy_decision import EnemyDecisionPolicy
from game.combat.enemy_learning import create_hero_policy, run_enemy_learning_episode
from game.content.definitions import GameDefinitions
from game.core.rng import GameRng
from game.expedition.cave import create_encounter_combat
from game.ml.confirmation_audit import (
    AuditRecordingPolicy,
    attach_decision_events,
    mcnemar_exact,
    paired_difference_ci,
    summarize_case,
    wilson_interval,
)

MAX_ROUNDS = 20
SEED_METHOD = "sha256-root-route-encounter-index-v1"


@dataclass(frozen=True)
class ResearchRoute:
    route_id: str
    encounter_ids: tuple[str, ...]
    structure: str
    purpose: str
    metadata: Mapping[str, str] = field(default_factory=dict)
    seed_namespace: str | None = None
    seed_by_encounter_identity: bool = False


def derive_encounter_seed(
    root_seed: int,
    *,
    route_id: str,
    encounter_index: int,
    encounter_id: str,
    hero_policy_id: str,
) -> int:
    """Derive policy-independent per-encounter randomness from one route seed."""
    payload = (
        f"cp6-route-v1|{root_seed}|{route_id}|{encounter_index}|{encounter_id}|{hero_policy_id}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF


def _hero_snapshot(hero: HeroState, definitions: GameDefinitions) -> dict[str, Any]:
    stats = effective_hero_stats(hero, definitions)
    return {
        "hero_id": hero.hero_id,
        "class_id": hero.class_id,
        "hp": hero.hp,
        "max_hp": stats.max_hp,
        "effort": hero.effort,
        "max_effort": stats.max_effort,
        "life_state": hero.life_state.value,
        "morale": hero.morale.name,
        "strain": hero.strain.name,
        "strain_marks": sorted(mark.value for mark in hero.strain_marks),
        "mortal_wounds": hero.mortal_wounds,
        "tags": sorted(tag.name for tag in hero.tags),
        "quirks": sorted(hero.quirks),
        "personal_quirk": hero.personal_quirk,
        "formation_slot": hero.formation_slot.value,
        "equipped_gear_id": hero.equipped_gear_id,
    }


def snapshot_company(company: CompanyState, definitions: GameDefinitions) -> dict[str, Any]:
    party = active_roster(company)
    hero_rows = [
        _hero_snapshot(hero, definitions) for hero in sorted(party, key=lambda h: h.hero_id)
    ]
    roster_living = [hero for hero in company.roster if hero.life_state != LifeState.DEAD]
    max_hp = sum(int(row["max_hp"]) for row in hero_rows)
    max_effort = sum(int(row["max_effort"]) for row in hero_rows)
    morale_counts = Counter(str(row["morale"]) for row in hero_rows)
    strain_counts = Counter(str(row["strain"]) for row in hero_rows)
    return {
        "party_heroes": hero_rows,
        "active_party_slots": {
            slot.value: company.active_party_slots.get(slot)
            for slot in sorted(company.active_party_slots, key=lambda item: item.value)
        },
        "deceased_hero_ids": sorted(hero.hero_id for hero in company.deceased_heroes),
        "active_hero_count": len(hero_rows),
        "living_roster_count": len(roster_living),
        "downed_count": sum(row["life_state"] == LifeState.DOWNED.value for row in hero_rows),
        "hp_total": sum(int(row["hp"]) for row in hero_rows),
        "max_hp_total": max_hp,
        "hp_fraction": (sum(int(row["hp"]) for row in hero_rows) / max_hp) if max_hp else 0.0,
        "effort_total": sum(int(row["effort"]) for row in hero_rows),
        "max_effort_total": max_effort,
        "effort_fraction": (
            sum(int(row["effort"]) for row in hero_rows) / max_effort if max_effort else 0.0
        ),
        "mortal_wounds_total": sum(int(row["mortal_wounds"]) for row in hero_rows),
        "strain_marks_total": sum(len(row["strain_marks"]) for row in hero_rows),
        "morale_counts": dict(sorted(morale_counts.items())),
        "strain_counts": dict(sorted(strain_counts.items())),
    }


def _combat_formation_slot(state: CombatState, hero_id: str) -> str | None:
    slot = state.party_formation.slot_of(hero_id)
    return slot.value if slot is not None else None


def snapshot_combat_entry(state: CombatState) -> dict[str, Any]:
    heroes = [state.heroes[key] for key in sorted(state.heroes)]
    max_hp = sum(hero.max_hp for hero in heroes)
    max_effort = sum(hero.max_effort for hero in heroes)
    morale_counts = Counter(hero.morale.name for hero in heroes)
    strain_counts = Counter(hero.strain.name for hero in heroes)
    return {
        "heroes": [
            {
                "hero_id": hero.actor_id,
                "class_id": hero.class_id,
                "hp": hero.hp,
                "max_hp": hero.max_hp,
                "effort": hero.effort,
                "max_effort": hero.max_effort,
                "life_state": hero.life_state.value,
                "morale": hero.morale.name,
                "strain": hero.strain.name,
                "strain_marks": sorted(mark.value for mark in hero.strain_marks),
                "mortal_wounds": hero.mortal_wounds,
                "tags": sorted(tag.name for tag in hero.tags),
                "quirks": sorted(hero.quirks),
                "personal_quirk": hero.personal_quirk,
                "formation_slot": _combat_formation_slot(state, hero.actor_id),
            }
            for hero in heroes
        ],
        "active_hero_count": sum(hero.life_state != LifeState.DEAD for hero in heroes),
        "downed_count": sum(hero.life_state == LifeState.DOWNED for hero in heroes),
        "hp_total": sum(hero.hp for hero in heroes),
        "max_hp_total": max_hp,
        "hp_fraction": sum(hero.hp for hero in heroes) / max_hp if max_hp else 0.0,
        "effort_total": sum(hero.effort for hero in heroes),
        "max_effort_total": max_effort,
        "effort_fraction": sum(hero.effort for hero in heroes) / max_effort if max_effort else 0.0,
        "mortal_wounds_total": sum(hero.mortal_wounds for hero in heroes),
        "strain_marks_total": sum(len(hero.strain_marks) for hero in heroes),
        "morale_counts": dict(sorted(morale_counts.items())),
        "strain_counts": dict(sorted(strain_counts.items())),
    }


def _numeric_deterioration(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "hp_delta": int(after["hp_total"]) - int(before["hp_total"]),
        "hp_loss": int(before["hp_total"]) - int(after["hp_total"]),
        "effort_delta": int(after["effort_total"]) - int(before["effort_total"]),
        "effort_loss": int(before["effort_total"]) - int(after["effort_total"]),
        "active_hero_delta": int(after["active_hero_count"]) - int(before["active_hero_count"]),
        "downed_delta": int(after["downed_count"]) - int(before["downed_count"]),
        "mortal_wounds_delta": (
            int(after["mortal_wounds_total"]) - int(before["mortal_wounds_total"])
        ),
        "strain_marks_delta": int(after["strain_marks_total"]) - int(before["strain_marks_total"]),
    }


def run_stateful_route(
    definitions: GameDefinitions,
    route: ResearchRoute,
    *,
    root_seed: int,
    enemy_policy_name: str,
    enemy_policy: EnemyDecisionPolicy,
    hero_policy_id: str,
    max_rounds: int = MAX_ROUNDS,
) -> dict[str, Any]:
    """Run one stateful route while delegating all persistence to production helpers."""
    company = create_new_company(definitions)
    initial = snapshot_company(company, definitions)
    encounters: list[dict[str, Any]] = []
    completed_count = 0
    failure_kind: str | None = None
    failed_at: str | None = None

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
        audit_policy = AuditRecordingPolicy(
            executed_name=enemy_policy_name,
            policies={enemy_policy_name: enemy_policy},
        )
        episode = run_enemy_learning_episode(
            combat,
            definitions,
            GameRng(encounter_seed),
            policy=audit_policy,
            max_rounds=max_rounds,
            encounter_id=encounter_id,
            encounter_name=encounter_id.replace("_", " ").title(),
            hero_policy=create_hero_policy(
                hero_policy_id,
                encounter_id=encounter_id,
                seed=encounter_seed,
            ),
            enemy_wait_mode="none",
            enemy_movement_mode="recovery_only",
        )
        attach_decision_events(audit_policy.decisions, episode)
        case_id = f"{route.route_id}|{hero_policy_id}|{root_seed}|{encounter_index}"
        encounter_summary = summarize_case(
            case_id=case_id,
            group="route",
            encounter_id=encounter_id,
            hero_policy_id=hero_policy_id,
            seed=encounter_seed,
            policy_name=enemy_policy_name,
            state=combat,
            episode=episode,
            decisions=audit_policy.decisions,
        )
        sync_company_from_combat(company, combat.heroes, combat.party_formation)
        company_after = snapshot_company(company, definitions)
        deterioration = _numeric_deterioration(combat_entry, company_after)
        hero_effort_spent = sum(int(action.effort_cost) for action in episode.hero_actions)
        encounters.append(
            {
                "encounter_index": encounter_index,
                "encounter_id": encounter_id,
                "encounter_seed": encounter_seed,
                "company_before_create": company_before,
                "combat_entry": combat_entry,
                "company_after_sync": company_after,
                "deterioration": deterioration,
                "enemy_win": int(combat.is_defeat()),
                "hero_win": int(combat.is_victory()),
                "final_victor": episode.final_victor,
                "rounds": episode.metrics.rounds_elapsed,
                "total_hero_damage": episode.metrics.total_hero_damage,
                "hero_effort_spent": hero_effort_spent,
                "enemy_caused_effort_drain": encounter_summary["enemy_caused_effort_drain"],
                "downs": episode.metrics.hero_downs,
                "deaths": episode.metrics.hero_deaths,
                "mortal_wounds": episode.metrics.mortal_wounds,
                "marks_applied": episode.metrics.marks_applied,
                "guard_uses": episode.metrics.guard_actions,
                "grab_uses": episode.metrics.boss_sequence.grab_uses,
                "bite_uses": episode.metrics.boss_sequence.bite_uses,
                "effort_drain_uses": encounter_summary["effort_drain_uses"],
                "direct_pressure_uses": encounter_summary["direct_pressure_uses"],
                "enemy_actions": episode.metrics.enemy_decisions,
                "enemy_waits": episode.metrics.enemy_waits,
                "enemy_moves": (
                    episode.metrics.enemy_recovery_moves + episode.metrics.enemy_proactive_moves
                ),
                "mark_payoffs": episode.metrics.mark_flow.exploited_by_enemy_hit,
                "grab_to_bite_same_target": episode.metrics.boss_sequence.grab_to_bite_same_target,
                "guard_payoffs": episode.metrics.guard_flow.guarded_ally_used_payoff_after_guard,
                "target_top_share": encounter_summary["target_top_share"],
                "target_switches": encounter_summary["target_switches"],
                "hostile_target_actions": encounter_summary["hostile_target_actions"],
                "lowest_hp_attacks": encounter_summary["lowest_hp_attacks"],
                "downed_target_attacks": encounter_summary["downed_target_attacks"],
                "protected_target_attacks": encounter_summary["protected_target_attacks"],
                "skill_uses": encounter_summary["skill_uses"],
                "target_class_counts": encounter_summary["target_class_counts"],
            }
        )
        if combat.is_victory():
            completed_count += 1
            continue
        failed_at = encounter_id
        failure_kind = "enemy_victory" if combat.is_defeat() else "unresolved_withdrawal"
        break

    final = snapshot_company(company, definitions)
    route_completed = completed_count == len(route.encounter_ids)
    return {
        "case_id": f"{route.route_id}|{hero_policy_id}|{root_seed}",
        "route_id": route.route_id,
        "route_structure": route.structure,
        "root_seed": root_seed,
        "enemy_policy": enemy_policy_name,
        "hero_policy_id": hero_policy_id,
        "encounter_sequence": list(route.encounter_ids),
        "route_completed": route_completed,
        "route_failed": not route_completed,
        "encounters_completed": completed_count,
        "failed_at_encounter_id": failed_at,
        "failure_kind": failure_kind,
        "initial_state": initial,
        "final_state": final,
        "encounters": encounters,
        "total_rounds": sum(int(row["rounds"]) for row in encounters),
        "cumulative_damage_taken": sum(int(row["total_hero_damage"]) for row in encounters),
        "hero_effort_spent": sum(int(row["hero_effort_spent"]) for row in encounters),
        "cumulative_enemy_effort_drain": sum(
            int(row["enemy_caused_effort_drain"]) for row in encounters
        ),
        "total_downs": sum(int(row["downs"]) for row in encounters),
        "total_deaths": sum(int(row["deaths"]) for row in encounters),
        "total_mortal_wounds": sum(int(row["mortal_wounds"]) for row in encounters),
    }


def aggregate_route_cases(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    if not items:
        return {"case_count": 0}
    failures = sum(bool(row["route_failed"]) for row in items)
    failed_at = Counter(
        str(row["failed_at_encounter_id"])
        for row in items
        if row["failed_at_encounter_id"] is not None
    )
    final_hp = [float(row["final_state"]["hp_total"]) for row in items]
    final_hp_fraction = [float(row["final_state"]["hp_fraction"]) for row in items]
    final_effort = [float(row["final_state"]["effort_total"]) for row in items]
    final_living = [float(row["final_state"]["active_hero_count"]) for row in items]
    final_mortal = [float(row["final_state"]["mortal_wounds_total"]) for row in items]
    return {
        "case_count": len(items),
        "route_failure_count": failures,
        "route_failure_rate": failures / len(items),
        "route_failure_wilson_95": wilson_interval(failures, len(items)),
        "completed_count": len(items) - failures,
        "failed_at_counts": dict(sorted(failed_at.items())),
        "mean_encounters_completed": mean(float(row["encounters_completed"]) for row in items),
        "mean_final_living_heroes": mean(final_living),
        "mean_final_hp": mean(final_hp),
        "median_final_hp": median(final_hp),
        "mean_final_hp_fraction": mean(final_hp_fraction),
        "median_final_hp_fraction": median(final_hp_fraction),
        "mean_final_effort": mean(final_effort),
        "median_final_effort": median(final_effort),
        "mean_final_mortal_wounds": mean(final_mortal),
        "mean_total_downs": mean(float(row["total_downs"]) for row in items),
        "mean_total_deaths": mean(float(row["total_deaths"]) for row in items),
        "mean_total_mortal_wounds": mean(float(row["total_mortal_wounds"]) for row in items),
        "mean_total_rounds": mean(float(row["total_rounds"]) for row in items),
        "mean_cumulative_damage_taken": mean(
            float(row["cumulative_damage_taken"]) for row in items
        ),
        "mean_hero_effort_spent": mean(float(row["hero_effort_spent"]) for row in items),
        "mean_cumulative_enemy_effort_drain": mean(
            float(row["cumulative_enemy_effort_drain"]) for row in items
        ),
    }


def _mean_difference_ci(values: Sequence[float]) -> list[float]:
    samples = [float(value) for value in values]
    if not samples:
        return [0.0, 0.0]
    center = mean(samples)
    if len(samples) == 1:
        return [center, center]
    variance = sum((value - center) ** 2 for value in samples) / (len(samples) - 1)
    half = 1.959963984540054 * (variance / len(samples)) ** 0.5
    return [center - half, center + half]


def paired_route_comparison(
    baseline_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    first = {str(row["case_id"]): row for row in baseline_rows}
    second = {str(row["case_id"]): row for row in comparison_rows}
    if first.keys() != second.keys():
        raise ValueError("Paired route cases differ.")
    failure_diffs: list[int] = []
    hp_diffs: list[float] = []
    effort_diffs: list[float] = []
    mortal_diffs: list[float] = []
    baseline_only = comparison_only = both_fail = both_complete = 0
    for case_id in sorted(first):
        a = bool(first[case_id]["route_failed"])
        b = bool(second[case_id]["route_failed"])
        failure_diffs.append(int(b) - int(a))
        hp_diffs.append(
            float(second[case_id]["final_state"]["hp_fraction"])
            - float(first[case_id]["final_state"]["hp_fraction"])
        )
        effort_diffs.append(
            float(second[case_id]["final_state"]["effort_total"])
            - float(first[case_id]["final_state"]["effort_total"])
        )
        mortal_diffs.append(
            float(second[case_id]["final_state"]["mortal_wounds_total"])
            - float(first[case_id]["final_state"]["mortal_wounds_total"])
        )
        if a and b:
            both_fail += 1
        elif a:
            baseline_only += 1
        elif b:
            comparison_only += 1
        else:
            both_complete += 1
    return {
        "case_count": len(failure_diffs),
        "route_failure_rate_difference": mean(failure_diffs) if failure_diffs else 0.0,
        "paired_failure_difference_ci_95": paired_difference_ci(failure_diffs),
        "mcnemar_exact_two_sided_p": mcnemar_exact(baseline_only, comparison_only),
        "transition_matrix": {
            "both_complete": both_complete,
            "baseline_only_failure": baseline_only,
            "comparison_only_failure": comparison_only,
            "both_fail": both_fail,
        },
        "mean_final_hp_fraction_difference": mean(hp_diffs) if hp_diffs else 0.0,
        "final_hp_fraction_difference_ci_95": _mean_difference_ci(hp_diffs),
        "mean_final_effort_difference": mean(effort_diffs) if effort_diffs else 0.0,
        "final_effort_difference_ci_95": _mean_difference_ci(effort_diffs),
        "mean_final_mortal_wounds_difference": mean(mortal_diffs) if mortal_diffs else 0.0,
        "final_mortal_wounds_difference_ci_95": _mean_difference_ci(mortal_diffs),
    }


def route_digest(row: Mapping[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hp_fraction_band(value: float) -> str:
    if value <= 0.25:
        return "0_25"
    if value <= 0.50:
        return "25_50"
    if value <= 0.75:
        return "50_75"
    return "75_100"


def encounter_attrition_rows(route_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for route in route_rows:
        for encounter in route["encounters"]:
            entry = encounter["combat_entry"]
            after = encounter["company_after_sync"]
            result.append(
                {
                    "case_id": route["case_id"],
                    "route_id": route["route_id"],
                    "root_seed": route["root_seed"],
                    "enemy_policy": route["enemy_policy"],
                    "hero_policy_id": route["hero_policy_id"],
                    "encounter_index": encounter["encounter_index"],
                    "encounter_id": encounter["encounter_id"],
                    "enemy_win": encounter["enemy_win"],
                    "entry_hp": entry["hp_total"],
                    "entry_max_hp": entry["max_hp_total"],
                    "entry_hp_fraction": entry["hp_fraction"],
                    "entry_hp_band": hp_fraction_band(float(entry["hp_fraction"])),
                    "entry_effort": entry["effort_total"],
                    "entry_max_effort": entry["max_effort_total"],
                    "entry_living": entry["active_hero_count"],
                    "entry_downed": entry["downed_count"],
                    "entry_mortal_wounds": entry["mortal_wounds_total"],
                    "entry_strain_marks": entry["strain_marks_total"],
                    "exit_hp": after["hp_total"],
                    "exit_hp_fraction": after["hp_fraction"],
                    "exit_effort": after["effort_total"],
                    "exit_living": after["active_hero_count"],
                    "exit_downed": after["downed_count"],
                    "exit_mortal_wounds": after["mortal_wounds_total"],
                    "exit_strain_marks": after["strain_marks_total"],
                    "hp_loss": encounter["deterioration"]["hp_loss"],
                    "effort_loss": encounter["deterioration"]["effort_loss"],
                    "total_hero_damage": encounter["total_hero_damage"],
                    "hero_effort_spent": encounter["hero_effort_spent"],
                    "enemy_effort_drain": encounter["enemy_caused_effort_drain"],
                    "downs": encounter["downs"],
                    "deaths": encounter["deaths"],
                    "mortal_wounds": encounter["mortal_wounds"],
                    "rounds": encounter["rounds"],
                    "marks_applied": encounter["marks_applied"],
                    "guard_uses": encounter["guard_uses"],
                    "grab_uses": encounter["grab_uses"],
                    "bite_uses": encounter["bite_uses"],
                    "effort_drain_uses": encounter["effort_drain_uses"],
                    "direct_pressure_uses": encounter["direct_pressure_uses"],
                    "enemy_actions": encounter["enemy_actions"],
                    "enemy_waits": encounter["enemy_waits"],
                    "enemy_moves": encounter["enemy_moves"],
                }
            )
    return result
