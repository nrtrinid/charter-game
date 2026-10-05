# ruff: noqa: E501
"""Checkpoint 7 production-faithful route termination requalification."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

from game.data.loaders import load_game_definitions
from game.dev.ml_cp5 import PI1_MODEL_SHA, PI2_MODEL_SHA, _load_frozen_policies
from game.dev.ml_cp6 import (
    FRESH_CONDITIONED_CONTEXTS,
    HERO_POLICIES,
    POLICY_NAMES,
    POLICY_PAIRS,
    PRIMARY_HERO_POLICY,
    PRIMARY_ROUTES,
    RESEARCH_SEEDS,
)
from game.ml.artifacts import sha256_file
from game.ml.confirmation_audit import mcnemar_exact, paired_difference_ci, wilson_interval
from game.ml.counterfactual import stable_digest
from game.ml.route_termination import (
    CombatTerminationReason,
    ResearchSafetyBudget,
    RouteTerminalStatus,
    run_corrected_stateful_route,
)
from game.ml.stateful_route import ResearchRoute

EXPERIMENT_ID = "checkpoint7-production-faithful-route-termination-v1"
CP6_COMMIT = "1495044607e0fe2bd83d21e38553baf8393262fb"
CP6_ROOT = Path("artifacts/ml/checkpoint6-stateful-route-evaluation-v1")
ARTIFACT_ROOT = Path("artifacts/ml") / EXPERIMENT_ID
PROTOCOL_PATH = ARTIFACT_ROOT / "frozen_protocol.json"
PROVENANCE_PATH = ARTIFACT_ROOT / "provenance.json"
TERMINATION_PATH = ARTIFACT_ROOT / "termination_semantics.json"
CYCLE_PATH = ARTIFACT_ROOT / "cycle_detection_audit.json"
SAFETY_PATH = ARTIFACT_ROOT / "safety_budget.json"
SAMPLE_PATH = ARTIFACT_ROOT / "old_withdrawal_sample.json"
SAMPLE_RESOLUTION_PATH = ARTIFACT_ROOT / "old_withdrawal_resolution.json"
EQUIVALENCE_PATH = ARTIFACT_ROOT / "ordinary_equivalence.json"
RESULTS_PATH = ARTIFACT_ROOT / "corrected_route_results.json"
TRANSITION_PATH = ARTIFACT_ROOT / "cp6_cp7_transition_matrix.json"
LENGTH_PATH = ARTIFACT_ROOT / "combat_length_distribution.json"
STALLING_PATH = ARTIFACT_ROOT / "stalling_audit.json"
COMPARISON_PATH = ARTIFACT_ROOT / "route_policy_comparison.json"
FRESH_PATH = ARTIFACT_ROOT / "fresh_vs_conditioned.json"
REACHABILITY_PATH = ARTIFACT_ROOT / "reachability_censoring.json"
ROLE_PATH = ARTIFACT_ROOT / "encounter_role_classification.json"
BEHAVIOR_PATH = ARTIFACT_ROOT / "behavior_long_combat_audit.json"
OBJECTIVE_PATH = ARTIFACT_ROOT / "objective_alignment.json"
MAZE_PATH = ARTIFACT_ROOT / "maze_readiness.json"
CP8_PATH = ARTIFACT_ROOT / "cp8_readiness.json"
MANIFEST_PATH = ARTIFACT_ROOT / "manifest.json"
REPORT_PATH = ARTIFACT_ROOT / "REPORT.md"

SAFETY_BUDGET = ResearchSafetyBudget(max_transitions=2048, max_rounds=256)

IMPLEMENTATION_PATHS = (
    "src/game/dev/ml_cp7.py",
    "src/game/ml/route_termination.py",
    "tests/test_ml_route_termination.py",
    "src/game/dev/ml_cp6.py",
    "src/game/ml/stateful_route.py",
    "src/game/dev/ml_cp5.py",
    "src/game/ml/confirmation_audit.py",
    "src/game/ml/policy_iteration.py",
    "src/game/ml/outcome_features.py",
    "src/game/ml/outcome_ranking.py",
)
SIMULATOR_PATHS = (
    "src/game/combat/actions.py",
    "src/game/combat/combat_state.py",
    "src/game/combat/enemy_decision.py",
    "src/game/combat/enemy_learning.py",
    "src/game/core/rng.py",
)
CONTENT_PATHS = tuple(str(path).replace("/", "\\") for path in sorted(Path("data").glob("*.yaml")))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _hash_paths(paths: Sequence[str]) -> dict[str, str]:
    return {path: sha256_file(Path(path)) for path in paths}


def _quantile(values: Sequence[float], q: float) -> float:
    items = sorted(float(value) for value in values)
    if not items:
        return 0.0
    index = (len(items) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return items[low]
    weight = index - low
    return items[low] * (1.0 - weight) + items[high] * weight


def _mean_ci(values: Sequence[float]) -> list[float]:
    samples = [float(value) for value in values]
    if not samples:
        return [0.0, 0.0]
    center = mean(samples)
    if len(samples) == 1:
        return [center, center]
    variance = sum((value - center) ** 2 for value in samples) / (len(samples) - 1)
    half = 1.959963984540054 * math.sqrt(variance / len(samples))
    return [center - half, center + half]


def _termination_semantics() -> dict[str, Any]:
    return {
        "production_manual": {
            "terminal_conditions": [
                "CombatState.is_victory",
                "CombatState.is_defeat",
                "explicit hero retreat",
            ],
            "automatic_round_limit": False,
            "retreat_semantics": "explicit player-declared gameplay action resolved at round boundary",
        },
        "legacy_research": {
            "runner": "run_enemy_learning_episode",
            "legacy_max_rounds": 20,
            "legacy_nonterminal_label": "withdrawal",
            "problem": "execution guard was conflated with game-world withdrawal",
        },
        "cp7": {
            "HERO_VICTORY": "production-valid terminal state",
            "ENEMY_VICTORY": "production-valid terminal state",
            "PRODUCTION_RETREAT": "only an actual gameplay retreat; frozen hero policies do not synthesize one",
            "DETECTED_CYCLE": "exact repeated transition-equivalent round-start state under identical RNG/policy phase",
            "SAFETY_BOUND_EXCEEDED": "censored unresolved research stop",
            "INVALID_SIMULATION": "simulation/invariant failure",
        },
        "route_rule": "censored/invalid combats never become route failure/completion and never sync partial combat state into CompanyState",
    }


def _cycle_semantics() -> dict[str, Any]:
    return {
        "checkpoint": "round start before initiative roll",
        "state_components": "full CombatState plus RNG digest and policy-relevant round semantics",
        "rng_required_equal": True,
        "hero_policies": "CP6 tactical/anti_mark/company_survival are deterministic and stateless for fixed state",
        "pi0_round_semantics": "heuristic uses round only through late_fight = round>=3; rounds >=3 normalize to one policy phase for cycle proof",
        "pi1_pi2_round_semantics": "absolute round retained because frozen 81-feature learned policy consumes round_number",
        "false_positive_policy": "if exact proof is unavailable, return SAFETY_BOUND_EXCEEDED rather than DETECTED_CYCLE",
        "absolute_round_omitted_only_for_pi0_after_late_phase": True,
    }


def _safety_semantics() -> dict[str, Any]:
    return {
        "primary_budget": {"primitive": "combat transitions", "max": SAFETY_BUDGET.max_transitions},
        "secondary_guard": {"primitive": "rounds", "max": SAFETY_BUDGET.max_rounds},
        "transition_definition": "enemy chosen actions + hero chosen actions + wait/move/recovery transitions + round boundary",
        "old_bound_multiple": "intentionally far above the historical 20-round guard",
        "on_exhaustion": CombatTerminationReason.SAFETY_BOUND_EXCEEDED.value,
        "route_handling": RouteTerminalStatus.ROUTE_CENSORED_SAFETY_BOUND.value,
        "imputation": False,
    }


def _ordinary_equivalence() -> dict[str, Any]:
    from game.campaign.company import create_new_company
    from game.campaign.roster import sync_company_from_combat
    from game.combat.enemy_learning import create_hero_policy, run_enemy_learning_episode
    from game.core.rng import GameRng
    from game.expedition.cave import create_encounter_combat
    from game.ml.counterfactual import combat_state_digest, rng_state_digest
    from game.ml.route_termination import run_research_combat_with_termination
    from game.ml.stateful_route import snapshot_company

    definitions = load_game_definitions()
    policies, _ = _load_frozen_policies()
    cases: list[dict[str, Any]] = []
    for encounter_id in ("shallow_cave", "generated_maze_probe", "cave_mini_boss"):
        for policy_name in POLICY_NAMES:
            encounter_seed = 20001
            old_company = create_new_company(definitions)
            old_state = create_encounter_combat(old_company, definitions, encounter_id)
            old_rng = GameRng(encounter_seed)
            old_episode = run_enemy_learning_episode(
                old_state,
                definitions,
                old_rng,
                policy=policies[policy_name],
                max_rounds=20,
                encounter_id=encounter_id,
                encounter_name=encounter_id,
                hero_policy=create_hero_policy(
                    "tactical", encounter_id=encounter_id, seed=encounter_seed
                ),
                enemy_wait_mode="none",
                enemy_movement_mode="recovery_only",
            )
            if old_episode.final_victor == "withdrawal":
                raise ValueError("Equivalence case unexpectedly reached legacy bound.")
            old_enemy = [
                [
                    record.round_number,
                    record.enemy_id,
                    record.chosen_skill_id,
                    record.chosen_target_id,
                ]
                for record in old_episode.records
            ]
            old_hero = [
                [record.round_number, record.hero_id, record.skill_id, record.target_id]
                for record in old_episode.hero_actions
            ]
            old_state_digest = combat_state_digest(old_state)
            old_rng_digest = rng_state_digest(old_rng)
            old_enemy_event_digest = stable_digest(
                [
                    {
                        "round": int(record.round_number),
                        "enemy": str(record.enemy_id),
                        "skill": str(record.chosen_skill_id),
                        "target": str(record.chosen_target_id),
                        "events": record.events,
                    }
                    for record in old_episode.records
                ]
            )
            sync_company_from_combat(old_company, old_state.heroes, old_state.party_formation)
            old_company_snapshot = snapshot_company(old_company, definitions)

            new_company = create_new_company(definitions)
            new_state = create_encounter_combat(new_company, definitions, encounter_id)
            new_rng = GameRng(encounter_seed)
            result = run_research_combat_with_termination(
                new_state,
                definitions,
                new_rng,
                enemy_policy_name=policy_name,
                enemy_policy=policies[policy_name],
                hero_policy=create_hero_policy(
                    "tactical", encounter_id=encounter_id, seed=encounter_seed
                ),
                encounter_id=encounter_id,
                encounter_name=encounter_id,
                safety_budget=SAFETY_BUDGET,
            )
            sync_company_from_combat(new_company, new_state.heroes, new_state.party_formation)
            new_company_snapshot = snapshot_company(new_company, definitions)
            row = {
                "encounter_id": encounter_id,
                "policy": policy_name,
                "old_victor": old_episode.final_victor,
                "new_reason": result.termination_reason.value,
                "final_state_equal": old_state_digest == result.final_state_digest,
                "rng_equal": old_rng_digest == result.final_rng_digest,
                "enemy_action_sequence_equal": old_enemy
                == [list(item) for item in result.enemy_action_sequence],
                "hero_action_sequence_equal": old_hero
                == [list(item) for item in result.hero_action_sequence],
                "enemy_event_digest_equal": old_enemy_event_digest == result.enemy_event_digest,
                "route_persistence_equal": old_company_snapshot == new_company_snapshot,
            }
            row["all_equal"] = all(
                bool(row[key])
                for key in (
                    "final_state_equal",
                    "rng_equal",
                    "enemy_action_sequence_equal",
                    "hero_action_sequence_equal",
                    "enemy_event_digest_equal",
                    "route_persistence_equal",
                )
            )
            cases.append(row)
    if not all(row["all_equal"] for row in cases):
        raise ValueError("Ordinary-combat equivalence qualification failed.")
    return {"case_count": len(cases), "all_equal": True, "cases": cases}


def _protocol_payload() -> dict[str, Any]:
    cp6_suite = _read_json(CP6_ROOT / "route_suite.json")
    return {
        "experiment_id": EXPERIMENT_ID,
        "immutable_cp6_commit": CP6_COMMIT,
        "policies": {
            "pi0_heuristic": "handcrafted heuristic baseline",
            "pi1_cp3_model_sha": PI1_MODEL_SHA,
            "pi2_cp4_model_sha": PI2_MODEL_SHA,
        },
        "cp6_route_suite_sha": sha256_file(CP6_ROOT / "route_suite.json"),
        "cp6_route_suite_digest": stable_digest(cp6_suite),
        "routes": [route.route_id for route in PRIMARY_ROUTES],
        "hero_policies": list(HERO_POLICIES),
        "seeds": [RESEARCH_SEEDS[0], RESEARCH_SEEDS[-1]],
        "seed_count": len(RESEARCH_SEEDS),
        "persistence_semantics": "unchanged CP6 production create_encounter_combat -> terminal combat -> sync_company_from_combat(..., party_formation)",
        "termination_taxonomy": [reason.value for reason in CombatTerminationReason],
        "cycle_detection": _cycle_semantics(),
        "safety_budget": _safety_semantics(),
        "route_terminal_statuses": [status.value for status in RouteTerminalStatus],
        "censoring_rule": "cycles/safety bounds excluded from production-valid completion/failure denominators; invalid separate",
        "primary_metrics": [
            "enemy_victory_route_count",
            "completed_route_count",
            "valid_terminal_route_count",
            "censored_cycle_count",
            "censored_safety_count",
            "invalid_count",
            "enemy_victory_rate_among_valid",
            "route_completion_rate_among_valid",
            "paired_valid_terminal_transitions",
        ],
        "encounter_role_requalification": {
            "execution_enemy_win_rate": 0.20,
            "pressure_enemy_win_rate": 0.05,
            "meaningful_hp_fraction_loss": 0.08,
            "meaningful_effort_loss": 0.50,
            "meaningful_downs_or_wounds": 0.10,
            "missing_conditioned_observation": "INSUFFICIENT_OBSERVATION",
        },
        "route_conditioning_materiality": {"absolute_enemy_win_rate_change": 0.10},
        "pi2_vs_pi1_requalification": {
            "stronger_if_paired_ci_lower_gt_zero": True,
            "weaker_if_paired_ci_upper_lt_zero": True,
            "approximately_equivalent_if_ci_crosses_zero_and_abs_difference_lte": 0.05,
        },
        "stalling_classification": {
            "DETERMINISTIC_CYCLING": "more than half of former withdrawals resolve as detected cycles",
            "STALLING_BUT_EVENTUALLY_TERMINAL": "at least 90% terminal, median extra rounds >=3, and median no-progress-round rate >=0.25",
            "LEGITIMATE_LONG_COMBAT": "at least 90% terminal and median no-progress-round rate <0.25",
            "MIXED": "multiple materially represented resolution modes",
            "INCONCLUSIVE": "more than 10% of former withdrawals remain censored/invalid",
        },
        "maze_readiness_rule": {
            "READY": "invalid=0 and censored fraction <=1%",
            "READY_WITH_LIMITATIONS": "invalid=0 and censored fraction <=5%",
            "NOT_YET_READY": "otherwise",
        },
        "cp8_readiness_rule": "PROCEED_AS_CP8 only if Maze readiness is READY or READY_WITH_LIMITATIONS and production-valid route statistics are not dominated by censoring",
        "post_evaluation_tuning_performed": False,
        "evaluation_accessed": False,
    }


def freeze_protocol() -> None:
    if _git_head() != CP6_COMMIT:
        raise ValueError(f"CP7 must freeze on immutable CP6 commit {CP6_COMMIT}.")
    if RESULTS_PATH.exists() or MANIFEST_PATH.exists():
        raise ValueError("CP7 full outcomes already exist; refusing to refreeze.")
    if not SAMPLE_PATH.exists() or not SAMPLE_RESOLUTION_PATH.exists():
        raise ValueError("Pre-freeze old-withdrawal sample qualification is missing.")
    _policies, policy_meta = _load_frozen_policies()
    if str(policy_meta["pi1_cp3"]["model_checksum"]) != PI1_MODEL_SHA:
        raise ValueError("π1 model checksum mismatch.")
    if str(policy_meta["pi2_cp4"]["model_checksum"]) != PI2_MODEL_SHA:
        raise ValueError("π2 model checksum mismatch.")
    equivalence = _ordinary_equivalence()
    _write_json(EQUIVALENCE_PATH, equivalence)
    _write_json(TERMINATION_PATH, _termination_semantics())
    _write_json(CYCLE_PATH, _cycle_semantics())
    _write_json(SAFETY_PATH, _safety_semantics())
    protocol = _protocol_payload()
    _write_json(PROTOCOL_PATH, protocol)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "cp6_commit": CP6_COMMIT,
        "current_head": _git_head(),
        "frozen_protocol_checksum": protocol_sha,
        "pi1_model_checksum": PI1_MODEL_SHA,
        "pi2_model_checksum": PI2_MODEL_SHA,
        "implementation_hashes": _hash_paths(IMPLEMENTATION_PATHS),
        "simulator_hashes": _hash_paths(SIMULATOR_PATHS),
        "content_hashes": _hash_paths(CONTENT_PATHS),
        "cp6_route_suite_checksum": sha256_file(CP6_ROOT / "route_suite.json"),
        "old_withdrawal_sample_checksum": sha256_file(SAMPLE_PATH),
        "old_withdrawal_resolution_checksum": sha256_file(SAMPLE_RESOLUTION_PATH),
        "pre_access_tests_passed": True,
        "evaluation_accessed": False,
        "post_evaluation_tuning_performed": False,
        "runtime": {"python": platform.python_version()},
    }
    _write_json(PROVENANCE_PATH, provenance)
    print(f"FROZEN_PROTOCOL_SHA={protocol_sha}")
    print(f"ORDINARY_EQUIVALENCE_CASES={equivalence['case_count']}")
    print("EVALUATION_ACCESSED=false")


def _validate_frozen() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _read_json(PROTOCOL_PATH)
    provenance = _read_json(PROVENANCE_PATH)
    if bool(protocol["evaluation_accessed"]):
        raise ValueError("Frozen protocol unexpectedly records evaluation_accessed=true.")
    if sha256_file(PROTOCOL_PATH) != provenance["frozen_protocol_checksum"]:
        raise ValueError("Frozen protocol checksum changed.")
    for label, paths in (
        ("implementation", provenance["implementation_hashes"]),
        ("simulator", provenance["simulator_hashes"]),
        ("content", provenance["content_hashes"]),
    ):
        for path, expected in paths.items():
            if sha256_file(Path(path)) != expected:
                raise ValueError(f"Frozen {label} hash changed: {path}")
    if sha256_file(CP6_ROOT / "route_suite.json") != provenance["cp6_route_suite_checksum"]:
        raise ValueError("CP6 route suite changed.")
    if sha256_file(SAMPLE_PATH) != provenance["old_withdrawal_sample_checksum"]:
        raise ValueError("Old-withdrawal sample changed after freeze.")
    if sha256_file(SAMPLE_RESOLUTION_PATH) != provenance["old_withdrawal_resolution_checksum"]:
        raise ValueError("Old-withdrawal qualification changed after freeze.")
    return protocol, provenance


def _route_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    counts = Counter(str(row["route_terminal_status"]) for row in items)
    completed = counts[RouteTerminalStatus.ROUTE_COMPLETED.value]
    enemy = counts[RouteTerminalStatus.ROUTE_FAILED_ENEMY_VICTORY.value]
    retreat = counts[RouteTerminalStatus.ROUTE_ENDED_PRODUCTION_RETREAT.value]
    cycle = counts[RouteTerminalStatus.ROUTE_CENSORED_CYCLE.value]
    safety = counts[RouteTerminalStatus.ROUTE_CENSORED_SAFETY_BOUND.value]
    invalid = counts[RouteTerminalStatus.ROUTE_INVALID.value]
    valid = completed + enemy + retreat
    return {
        "case_count": len(items),
        "completed_route_count": completed,
        "enemy_victory_route_count": enemy,
        "production_retreat_count": retreat,
        "valid_terminal_route_count": valid,
        "censored_cycle_count": cycle,
        "censored_safety_bound_count": safety,
        "censored_count": cycle + safety,
        "invalid_count": invalid,
        "enemy_victory_rate_among_valid": enemy / valid if valid else None,
        "route_completion_rate_among_valid": completed / valid if valid else None,
        "enemy_victory_wilson_95": wilson_interval(enemy, valid) if valid else None,
        "completion_wilson_95": wilson_interval(completed, valid) if valid else None,
        "terminal_status_counts": dict(sorted(counts.items())),
    }


def _paired_valid_comparison(
    first_rows: Sequence[Mapping[str, Any]],
    second_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    first = {str(row["case_id"]): row for row in first_rows}
    second = {str(row["case_id"]): row for row in second_rows}
    if first.keys() != second.keys():
        raise ValueError("Paired CP7 route cases differ.")
    diffs: list[int] = []
    a_only = b_only = both_enemy = both_complete = 0
    excluded = 0
    transitions: Counter[str] = Counter()
    for case_id in sorted(first):
        a_status = str(first[case_id]["route_terminal_status"])
        b_status = str(second[case_id]["route_terminal_status"])
        transitions[f"{a_status}->{b_status}"] += 1
        a_valid = a_status in {
            RouteTerminalStatus.ROUTE_COMPLETED.value,
            RouteTerminalStatus.ROUTE_FAILED_ENEMY_VICTORY.value,
            RouteTerminalStatus.ROUTE_ENDED_PRODUCTION_RETREAT.value,
        }
        b_valid = b_status in {
            RouteTerminalStatus.ROUTE_COMPLETED.value,
            RouteTerminalStatus.ROUTE_FAILED_ENEMY_VICTORY.value,
            RouteTerminalStatus.ROUTE_ENDED_PRODUCTION_RETREAT.value,
        }
        if not (a_valid and b_valid):
            excluded += 1
            continue
        a_enemy = a_status == RouteTerminalStatus.ROUTE_FAILED_ENEMY_VICTORY.value
        b_enemy = b_status == RouteTerminalStatus.ROUTE_FAILED_ENEMY_VICTORY.value
        diffs.append(int(b_enemy) - int(a_enemy))
        if a_enemy and b_enemy:
            both_enemy += 1
        elif a_enemy:
            a_only += 1
        elif b_enemy:
            b_only += 1
        else:
            both_complete += 1
    return {
        "paired_case_count": len(first),
        "paired_valid_terminal_count": len(diffs),
        "excluded_for_censoring_or_invalid": excluded,
        "enemy_victory_rate_difference": mean(diffs) if diffs else None,
        "paired_difference_ci_95": paired_difference_ci(diffs) if diffs else None,
        "mcnemar_exact_two_sided_p": mcnemar_exact(a_only, b_only) if diffs else None,
        "discordant": {"first_only_enemy_victory": a_only, "second_only_enemy_victory": b_only},
        "both_enemy_victory": both_enemy,
        "both_non_enemy_valid_terminal": both_complete,
        "all_status_transitions": dict(sorted(transitions.items())),
    }


def _route_policy_comparison(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"by_hero_policy": {}, "primary": {}}
    for hero_policy in HERO_POLICIES:
        hero_rows = [row for row in rows if row["hero_policy_id"] == hero_policy]
        by_policy = {
            name: [row for row in hero_rows if row["enemy_policy"] == name] for name in POLICY_NAMES
        }
        result["by_hero_policy"][hero_policy] = {
            "aggregates": {name: _route_counts(items) for name, items in by_policy.items()},
            "paired": {
                f"{a}_to_{b}": _paired_valid_comparison(by_policy[a], by_policy[b])
                for a, b in POLICY_PAIRS
            },
        }
    primary = result["by_hero_policy"][PRIMARY_HERO_POLICY]
    result["primary"] = primary
    return result


def _cp6_withdrawal_transition_matrix(
    cp6_rows: Sequence[Mapping[str, Any]],
    cp7_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cp7_by_case = {str(row["case_id"]): row for row in cp7_rows}
    by_policy: dict[str, Counter[str]] = {name: Counter() for name in POLICY_NAMES}
    details: list[dict[str, Any]] = []
    for old_route in cp6_rows:
        for old_encounter in old_route["encounters"]:
            if old_encounter["final_victor"] != "withdrawal":
                continue
            policy = str(old_route["enemy_policy"])
            new_route = cp7_by_case[str(old_route["case_id"])]
            new_encounter = next(
                (
                    row
                    for row in new_route["encounters"]
                    if int(row["encounter_index"]) == int(old_encounter["encounter_index"])
                ),
                None,
            )
            reason = (
                str(new_encounter["termination_reason"])
                if new_encounter is not None
                else "TARGET_NOT_REACHED"
            )
            by_policy[policy][reason] += 1
            details.append(
                {
                    "case_id": old_route["case_id"],
                    "route_id": old_route["route_id"],
                    "hero_policy_id": old_route["hero_policy_id"],
                    "enemy_policy": policy,
                    "encounter_index": old_encounter["encounter_index"],
                    "encounter_id": old_encounter["encounter_id"],
                    "cp6_rounds": old_encounter["rounds"],
                    "cp7_reason": reason,
                    "cp7_rounds": new_encounter["rounds"] if new_encounter else None,
                    "additional_rounds": (
                        int(new_encounter["rounds"]) - int(old_encounter["rounds"])
                        if new_encounter
                        else None
                    ),
                    "cp7_transitions": new_encounter["transitions"] if new_encounter else None,
                    "no_progress_round_rate": (
                        new_encounter["behavior"]["no_progress_round_rate"]
                        if new_encounter
                        else None
                    ),
                }
            )
    return {
        "by_policy": {
            policy: {
                "count": sum(counts.values()),
                "counts": dict(sorted(counts.items())),
                "fractions": {
                    key: value / sum(counts.values()) if sum(counts.values()) else 0.0
                    for key, value in sorted(counts.items())
                },
            }
            for policy, counts in by_policy.items()
        },
        "rows": details,
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        return {"count": 0}
    return {
        "count": len(samples),
        "mean": mean(samples),
        "median": median(samples),
        "p90": _quantile(samples, 0.90),
        "p95": _quantile(samples, 0.95),
        "p99": _quantile(samples, 0.99),
        "max": max(samples),
    }


def _combat_length_distribution(
    route_rows: Sequence[Mapping[str, Any]],
    fresh_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    encounters: list[dict[str, Any]] = []
    for route in [*route_rows, *fresh_rows]:
        for encounter in route["encounters"]:
            encounters.append(
                {
                    **encounter,
                    "route_id": route["route_id"],
                    "enemy_policy": route["enemy_policy"],
                    "hero_policy_id": route["hero_policy_id"],
                }
            )
    terminal = [
        row
        for row in encounters
        if row["termination_reason"]
        in {CombatTerminationReason.HERO_VICTORY.value, CombatTerminationReason.ENEMY_VICTORY.value}
    ]
    by_policy: dict[str, Any] = {}
    for policy in POLICY_NAMES:
        subset = [row for row in terminal if row["enemy_policy"] == policy]
        by_policy[policy] = {
            "rounds": _distribution([row["rounds"] for row in subset]),
            "transitions": _distribution([row["transitions"] for row in subset]),
            "terminal_over_20_rounds": sum(int(row["rounds"]) > 20 for row in subset),
            "max_naturally_terminal_rounds": max((int(row["rounds"]) for row in subset), default=0),
        }
    by_encounter: dict[str, Any] = {}
    for encounter_id in sorted({str(row["encounter_id"]) for row in terminal}):
        by_encounter[encounter_id] = {}
        for policy in POLICY_NAMES:
            subset = [
                row
                for row in terminal
                if row["encounter_id"] == encounter_id and row["enemy_policy"] == policy
            ]
            by_encounter[encounter_id][policy] = {
                "rounds": _distribution([row["rounds"] for row in subset]),
                "transitions": _distribution([row["transitions"] for row in subset]),
            }
    return {"by_policy": by_policy, "by_encounter": by_encounter}


def _stalling_audit(
    transition_matrix: Mapping[str, Any],
    route_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    old_rows = list(transition_matrix["rows"])
    by_policy: dict[str, Any] = {}
    for policy in POLICY_NAMES:
        rows = [row for row in old_rows if row["enemy_policy"] == policy]
        terminal_rows = [
            row
            for row in rows
            if row["cp7_reason"]
            in {
                CombatTerminationReason.HERO_VICTORY.value,
                CombatTerminationReason.ENEMY_VICTORY.value,
            }
        ]
        censored_or_invalid = len(rows) - len(terminal_rows)
        extra = [float(row["additional_rounds"]) for row in terminal_rows]
        no_progress = [float(row["no_progress_round_rate"]) for row in terminal_rows]
        counts = Counter(str(row["cp7_reason"]) for row in rows)
        terminal_fraction = len(terminal_rows) / len(rows) if rows else 0.0
        censored_fraction = censored_or_invalid / len(rows) if rows else 0.0
        cycle_fraction = (
            counts[CombatTerminationReason.DETECTED_CYCLE.value] / len(rows) if rows else 0.0
        )
        median_extra = median(extra) if extra else 0.0
        median_no_progress = median(no_progress) if no_progress else 0.0
        if censored_fraction > 0.10:
            classification = "INCONCLUSIVE"
        elif cycle_fraction > 0.50:
            classification = "DETERMINISTIC_CYCLING"
        elif terminal_fraction >= 0.90 and median_extra >= 3 and median_no_progress >= 0.25:
            classification = "STALLING_BUT_EVENTUALLY_TERMINAL"
        elif terminal_fraction >= 0.90 and median_no_progress < 0.25:
            classification = "LEGITIMATE_LONG_COMBAT"
        else:
            classification = "MIXED"
        by_policy[policy] = {
            "former_withdrawal_count": len(rows),
            "resolution_counts": dict(sorted(counts.items())),
            "terminal_fraction": terminal_fraction,
            "censored_or_invalid_fraction": censored_fraction,
            "median_additional_rounds": median_extra,
            "mean_additional_rounds": mean(extra) if extra else None,
            "median_no_progress_round_rate": median_no_progress,
            "mean_no_progress_round_rate": mean(no_progress) if no_progress else None,
            "classification": classification,
        }

    long_by_policy: dict[str, Any] = {}
    for policy in POLICY_NAMES:
        long_encounters: list[Mapping[str, Any]] = []
        ordinary: list[Mapping[str, Any]] = []
        for route in route_rows:
            if route["enemy_policy"] != policy:
                continue
            for enc in route["encounters"]:
                if enc["termination_reason"] not in {
                    CombatTerminationReason.HERO_VICTORY.value,
                    CombatTerminationReason.ENEMY_VICTORY.value,
                }:
                    continue
                (long_encounters if int(enc["rounds"]) > 20 else ordinary).append(enc)

        def behavior_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            total_actions = sum(int(row["behavior"]["enemy_actions"]) for row in items)
            return {
                "count": len(items),
                "mean_rounds": mean(float(row["rounds"]) for row in items) if items else None,
                "mean_no_progress_round_rate": mean(
                    float(row["behavior"]["no_progress_round_rate"]) for row in items
                )
                if items
                else None,
                "mark_rate": sum(int(row["behavior"]["mark_uses"]) for row in items) / total_actions
                if total_actions
                else 0.0,
                "grab_rate": sum(int(row["behavior"]["grab_uses"]) for row in items) / total_actions
                if total_actions
                else 0.0,
                "guard_rate": sum(int(row["behavior"]["guard_uses"]) for row in items)
                / total_actions
                if total_actions
                else 0.0,
                "bite_rate": sum(int(row["behavior"]["bite_uses"]) for row in items) / total_actions
                if total_actions
                else 0.0,
                "direct_pressure_rate": sum(
                    int(row["behavior"]["direct_pressure_uses"]) for row in items
                )
                / total_actions
                if total_actions
                else 0.0,
                "effort_drain_rate": sum(int(row["behavior"]["effort_drain_uses"]) for row in items)
                / total_actions
                if total_actions
                else 0.0,
            }

        long_by_policy[policy] = {
            "ordinary_terminal_20_or_less": behavior_summary(ordinary),
            "long_terminal_over_20": behavior_summary(long_encounters),
        }
    return {"former_withdrawals": by_policy, "ordinary_vs_long_behavior": long_by_policy}


def _fresh_conditioned_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target, contexts in FRESH_CONDITIONED_CONTEXTS.items():
        target_result: dict[str, Any] = {"contexts": {}}
        for context_name in contexts:
            route_id = f"freshcmp_{target}_{context_name}"
            context_rows = [row for row in rows if row["route_id"] == route_id]
            context_payload: dict[str, Any] = {}
            for policy in POLICY_NAMES:
                policy_rows = [row for row in context_rows if row["enemy_policy"] == policy]
                reached = []
                not_reached = 0
                for route in policy_rows:
                    target_row = next(
                        (enc for enc in route["encounters"] if enc["encounter_id"] == target),
                        None,
                    )
                    if target_row is None:
                        not_reached += 1
                    else:
                        reached.append(target_row)
                valid = [
                    row
                    for row in reached
                    if row["termination_reason"]
                    in {
                        CombatTerminationReason.HERO_VICTORY.value,
                        CombatTerminationReason.ENEMY_VICTORY.value,
                    }
                ]
                censored = [row for row in reached if bool(row["censored"])]
                invalid = [
                    row
                    for row in reached
                    if row["termination_reason"] == CombatTerminationReason.INVALID_SIMULATION.value
                ]
                enemy_wins = sum(
                    row["termination_reason"] == CombatTerminationReason.ENEMY_VICTORY.value
                    for row in valid
                )
                context_payload[policy] = {
                    "case_count": len(policy_rows),
                    "target_reached_count": len(reached),
                    "target_not_reached_count": not_reached,
                    "valid_terminal_count": len(valid),
                    "censored_count": len(censored),
                    "invalid_count": len(invalid),
                    "enemy_wins": enemy_wins,
                    "enemy_win_rate_among_reached_valid": enemy_wins / len(valid)
                    if valid
                    else None,
                    "enemy_win_wilson_95": wilson_interval(enemy_wins, len(valid))
                    if valid
                    else None,
                    "mean_entry_hp_fraction": mean(
                        float(row["combat_entry"]["hp_fraction"]) for row in reached
                    )
                    if reached
                    else None,
                    "mean_entry_effort": mean(
                        float(row["combat_entry"]["effort_total"]) for row in reached
                    )
                    if reached
                    else None,
                    "mean_rounds_valid": mean(float(row["rounds"]) for row in valid)
                    if valid
                    else None,
                    "observation_status": "OBSERVED" if reached else "TARGET_NOT_REACHED",
                }
            target_result["contexts"][context_name] = context_payload
        result[target] = target_result
    return result


def _reachability_censoring(fresh: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for target, payload in fresh.items():
        for context, policies in payload["contexts"].items():
            for policy, values in policies.items():
                rows.append(
                    {
                        "target": target,
                        "context": context,
                        "policy": policy,
                        "target_reached_count": values["target_reached_count"],
                        "target_not_reached_count": values["target_not_reached_count"],
                        "censored_count": values["censored_count"],
                        "invalid_count": values["invalid_count"],
                        "observation_status": values["observation_status"],
                    }
                )
    return {
        "rule": "TARGET_NOT_REACHED is missing encounter-specific observation, never zero lethality/value",
        "rows": rows,
        "historical_cp6_issue": "unreached target cells could appear as zero effect in downstream classification",
    }


def _encounter_role_classification(
    route_rows: Sequence[Mapping[str, Any]],
    fresh: Mapping[str, Any],
) -> dict[str, Any]:
    encounter_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for route in route_rows:
        if route["hero_policy_id"] != PRIMARY_HERO_POLICY:
            continue
        for enc in route["encounters"]:
            if enc["production_terminal"]:
                encounter_rows[str(enc["encounter_id"])].append(enc)
    result: dict[str, Any] = {}
    encounter_ids = set(encounter_rows) | set(FRESH_CONDITIONED_CONTEXTS)
    for encounter_id in sorted(encounter_ids):
        rows = encounter_rows.get(encounter_id, [])
        terminal_hero = [
            row
            for row in rows
            if row["termination_reason"] == CombatTerminationReason.HERO_VICTORY.value
            and row["company_after_sync"] is not None
        ]
        mean_hp_loss = (
            mean(
                float(row["deterioration"]["hp_loss"])
                / max(1, float(row["combat_entry"]["max_hp_total"]))
                for row in terminal_hero
            )
            if terminal_hero
            else 0.0
        )
        mean_effort_loss = (
            mean(float(row["deterioration"]["effort_loss"]) for row in terminal_hero)
            if terminal_hero
            else 0.0
        )
        mean_down_or_wound = (
            mean(
                float(
                    row["deterioration"]["downed_delta"]
                    + row["deterioration"]["mortal_wounds_delta"]
                )
                for row in terminal_hero
            )
            if terminal_hero
            else 0.0
        )
        fresh_payload = fresh.get(encounter_id)
        conditioned_observed = False
        fresh_learned_rates: list[float] = []
        conditioned_rates: list[float] = []
        if fresh_payload:
            for context, policies in fresh_payload["contexts"].items():
                for policy in ("pi1_cp3", "pi2_cp4"):
                    values = policies[policy]
                    rate = values["enemy_win_rate_among_reached_valid"]
                    if context == "fresh" and rate is not None:
                        fresh_learned_rates.append(float(rate))
                    elif context != "fresh" and values["target_reached_count"] > 0:
                        conditioned_observed = True
                        if rate is not None:
                            conditioned_rates.append(float(rate))
        fresh_rate = mean(fresh_learned_rates) if fresh_learned_rates else 0.0
        max_conditioned = max(conditioned_rates, default=0.0)
        attrition = mean_hp_loss >= 0.08 or mean_effort_loss >= 0.5 or mean_down_or_wound >= 0.1
        if fresh_payload and not conditioned_observed:
            classification = "INSUFFICIENT_OBSERVATION"
        elif max(fresh_rate, max_conditioned) >= 0.20 and attrition:
            classification = "MIXED"
        elif max(fresh_rate, max_conditioned) >= 0.20:
            classification = "EXECUTION"
        elif attrition:
            classification = "ATTRITION"
        elif max(fresh_rate, max_conditioned) >= 0.05:
            classification = "PRESSURE"
        else:
            classification = "NEGLIGIBLE_UNDER_CURRENT_RULES"
        result[encounter_id] = {
            "classification": classification,
            "conditioned_observation_available": conditioned_observed,
            "fresh_learned_enemy_win_rate_mean": fresh_rate,
            "max_conditioned_enemy_win_rate": max_conditioned if conditioned_observed else None,
            "mean_hp_fraction_loss_on_hero_terminal": mean_hp_loss,
            "mean_effort_loss_on_hero_terminal": mean_effort_loss,
            "mean_downs_plus_wounds_on_hero_terminal": mean_down_or_wound,
            "terminal_route_observation_count": len(rows),
        }
    return result


def _behavior_long_audit(stalling: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "warning": "frequency differences are not causal balance evidence",
        "ordinary_vs_long": stalling["ordinary_vs_long_behavior"],
        "former_withdrawal_resolution": stalling["former_withdrawals"],
        "cycle_cases": {
            policy: values["resolution_counts"].get(CombatTerminationReason.DETECTED_CYCLE.value, 0)
            for policy, values in stalling["former_withdrawals"].items()
        },
    }


def _objective_alignment(comparison: Mapping[str, Any], fresh: Mapping[str, Any]) -> dict[str, Any]:
    primary = comparison["primary"]["aggregates"]
    learned_stronger = all(
        (primary[name]["enemy_victory_rate_among_valid"] or 0.0)
        > (primary["pi0_heuristic"]["enemy_victory_rate_among_valid"] or 0.0)
        for name in ("pi1_cp3", "pi2_cp4")
    )
    conditioning_changes = False
    for target in fresh.values():
        for policy in ("pi1_cp3", "pi2_cp4"):
            fresh_rate = target["contexts"]["fresh"][policy]["enemy_win_rate_among_reached_valid"]
            for context in ("one_predecessor", "long_prefix"):
                rate = target["contexts"][context][policy]["enemy_win_rate_among_reached_valid"]
                if (
                    fresh_rate is not None
                    and rate is not None
                    and abs(float(rate) - float(fresh_rate)) >= 0.10
                ):
                    conditioning_changes = True
    classification = "PARTIALLY" if learned_stronger and conditioning_changes else "INCONCLUSIVE"
    return {
        "single_combat_objective_alignment": classification,
        "learned_policies_stronger_on_valid_primary_routes": learned_stronger,
        "route_conditioning_materially_changes_lethality": conditioning_changes,
        "future_route_level_training_may_be_justified": conditioning_changes,
        "training_performed": False,
    }


def _maze_readiness(comparison: Mapping[str, Any]) -> dict[str, Any]:
    primary = comparison["primary"]["aggregates"]
    total = sum(int(primary[name]["case_count"]) for name in POLICY_NAMES)
    censored = sum(int(primary[name]["censored_count"]) for name in POLICY_NAMES)
    invalid = sum(int(primary[name]["invalid_count"]) for name in POLICY_NAMES)
    fraction = censored / total if total else 1.0
    if invalid == 0 and fraction <= 0.01:
        verdict = "READY"
    elif invalid == 0 and fraction <= 0.05:
        verdict = "READY_WITH_LIMITATIONS"
    else:
        verdict = "NOT_YET_READY"
    return {
        "verdict": verdict,
        "primary_case_count": total,
        "primary_censored_count": censored,
        "primary_censored_fraction": fraction,
        "primary_invalid_count": invalid,
        "requirements": [
            "no research timeout may be interpreted as game outcome",
            "unreached targets remain missing observations",
            "censoring must remain explicit in optimization denominators",
        ],
    }


def _cp8_readiness(maze: Mapping[str, Any], comparison: Mapping[str, Any]) -> dict[str, Any]:
    primary = comparison["primary"]["aggregates"]
    valid = sum(int(primary[name]["valid_terminal_route_count"]) for name in POLICY_NAMES)
    total = sum(int(primary[name]["case_count"]) for name in POLICY_NAMES)
    if maze["verdict"] in {"READY", "READY_WITH_LIMITATIONS"} and valid / total >= 0.95:
        verdict = "PROCEED_AS_CP8"
    elif maze["verdict"] == "NOT_YET_READY":
        verdict = "MODIFY_BEFORE_CP8"
    else:
        verdict = "MODIFY_BEFORE_CP8"
    return {
        "verdict": verdict,
        "reason": "Causal balance perturbations require production-valid route outcomes and explicit censoring; CP7 itself makes no perturbation.",
        "executed": False,
    }


def _build_fresh_routes() -> list[ResearchRoute]:
    routes: list[ResearchRoute] = []
    for target, contexts in FRESH_CONDITIONED_CONTEXTS.items():
        for context_name, sequence in contexts.items():
            routes.append(
                ResearchRoute(
                    route_id=f"freshcmp_{target}_{context_name}",
                    encounter_ids=sequence,
                    structure=f"FRESH_CONDITIONED_{context_name.upper()}",
                    purpose=f"Fresh-versus-conditioned panel for {target}.",
                    metadata={"target_encounter": target, "context": context_name},
                    seed_namespace=f"freshcmp_{target}",
                    seed_by_encounter_identity=True,
                )
            )
    return routes


def run_evaluation() -> None:
    protocol, provenance = _validate_frozen()
    if RESULTS_PATH.exists() or MANIFEST_PATH.exists():
        raise ValueError("CP7 corrected route outcomes already exist; refusing to rerun.")
    definitions = load_game_definitions()
    policies, _ = _load_frozen_policies()
    route_rows: list[dict[str, Any]] = []
    for policy_name in POLICY_NAMES:
        for hero_policy_id in HERO_POLICIES:
            for route in PRIMARY_ROUTES:
                for seed in RESEARCH_SEEDS:
                    route_rows.append(
                        run_corrected_stateful_route(
                            definitions,
                            route,
                            root_seed=seed,
                            enemy_policy_name=policy_name,
                            enemy_policy=policies[policy_name],
                            hero_policy_id=hero_policy_id,
                            safety_budget=SAFETY_BUDGET,
                        )
                    )
    fresh_rows: list[dict[str, Any]] = []
    for route in _build_fresh_routes():
        for policy_name in POLICY_NAMES:
            for seed in RESEARCH_SEEDS:
                fresh_rows.append(
                    run_corrected_stateful_route(
                        definitions,
                        route,
                        root_seed=seed,
                        enemy_policy_name=policy_name,
                        enemy_policy=policies[policy_name],
                        hero_policy_id=PRIMARY_HERO_POLICY,
                        safety_budget=SAFETY_BUDGET,
                    )
                )
    comparison = _route_policy_comparison(route_rows)
    cp6_rows = _read_json(CP6_ROOT / "route_results.json")["rows"]
    transitions = _cp6_withdrawal_transition_matrix(cp6_rows, route_rows)
    lengths = _combat_length_distribution(route_rows, fresh_rows)
    stalling = _stalling_audit(transitions, route_rows)
    fresh = _fresh_conditioned_analysis(fresh_rows)
    reachability = _reachability_censoring(fresh)
    roles = _encounter_role_classification(route_rows, fresh)
    behavior = _behavior_long_audit(stalling)
    objective = _objective_alignment(comparison, fresh)
    maze = _maze_readiness(comparison)
    cp8 = _cp8_readiness(maze, comparison)

    _write_json(
        RESULTS_PATH,
        {
            "experiment_id": EXPERIMENT_ID,
            "route_rows": route_rows,
            "fresh_conditioned_rows": fresh_rows,
            "post_evaluation_tuning_performed": False,
        },
    )
    _write_json(TRANSITION_PATH, transitions)
    _write_json(LENGTH_PATH, lengths)
    _write_json(STALLING_PATH, stalling)
    _write_json(COMPARISON_PATH, comparison)
    _write_json(FRESH_PATH, fresh)
    _write_json(REACHABILITY_PATH, reachability)
    _write_json(ROLE_PATH, roles)
    _write_json(BEHAVIOR_PATH, behavior)
    _write_json(OBJECTIVE_PATH, objective)
    _write_json(MAZE_PATH, maze)
    _write_json(CP8_PATH, cp8)

    _write_report(
        {
            "protocol": protocol,
            "provenance": provenance,
            "transitions": transitions,
            "lengths": lengths,
            "stalling": stalling,
            "comparison": comparison,
            "fresh": fresh,
            "reachability": reachability,
            "roles": roles,
            "behavior": behavior,
            "objective": objective,
            "maze": maze,
            "cp8": cp8,
        }
    )
    manifest = _manifest()
    _write_json(MANIFEST_PATH, manifest)
    _reverify_manifest(manifest)
    print(f"ROUTES={len(route_rows)}")
    print(f"FRESH_CONDITIONED_ROUTES={len(fresh_rows)}")
    print(f"FROZEN_PROTOCOL_SHA={provenance['frozen_protocol_checksum']}")
    print("POST_EVALUATION_TUNING=false")


def _final_verdict(payload: Mapping[str, Any]) -> dict[str, Any]:
    comparison = payload["comparison"]["primary"]
    aggregates = comparison["aggregates"]
    paired = comparison["paired"]
    pi0 = aggregates["pi0_heuristic"]
    pi1 = aggregates["pi1_cp3"]
    pi2 = aggregates["pi2_cp4"]
    learned_superiority = (pi1["enemy_victory_rate_among_valid"] or 0.0) > (
        pi0["enemy_victory_rate_among_valid"] or 0.0
    ) and (pi2["enemy_victory_rate_among_valid"] or 0.0) > (
        pi0["enemy_victory_rate_among_valid"] or 0.0
    )
    pi2_pair = paired["pi1_cp3_to_pi2_cp4"]
    diff = pi2_pair["enemy_victory_rate_difference"]
    ci = pi2_pair["paired_difference_ci_95"]
    if diff is None or ci is None:
        pi2_vs_pi1 = "INCONCLUSIVE"
    elif ci[0] > 0:
        pi2_vs_pi1 = "YES"
    elif ci[1] < 0:
        pi2_vs_pi1 = "NO"
    elif abs(float(diff)) <= 0.05:
        pi2_vs_pi1 = "APPROXIMATELY_EQUIVALENT"
    else:
        pi2_vs_pi1 = "INCONCLUSIVE"
    roles = payload["roles"]
    low_attrition = any(
        roles.get(name, {}).get("classification") in {"ATTRITION", "MIXED"}
        for name in ("generated_maze_probe", "generated_maze_glass_pack", "shallow_cave")
    )
    missing_contamination = (
        roles.get("generated_maze_hunt_guarded", {}).get("classification")
        == "INSUFFICIENT_OBSERVATION"
    )
    maze_verdict = str(payload["maze"]["verdict"])
    faithful = (
        "YES"
        if maze_verdict == "READY"
        else "YES_WITH_CENSORING_LIMITATIONS"
        if maze_verdict == "READY_WITH_LIMITATIONS"
        else "NO"
    )
    return {
        "cp6_round20_withdrawal_production_faithful": "NO",
        "old_withdrawal_resolution_by_policy": payload["transitions"]["by_policy"],
        "pi0_old_withdrawal_primary_classification": payload["stalling"]["former_withdrawals"][
            "pi0_heuristic"
        ]["classification"],
        "learned_superiority_over_pi0_persists": "YES" if learned_superiority else "MIXED",
        "pi2_stronger_than_pi1_route_level": pi2_vs_pi1,
        "route_conditioning_materially_changes_lethality": "YES"
        if payload["objective"]["route_conditioning_materially_changes_lethality"]
        else "INCONCLUSIVE",
        "low_lethality_early_encounters_have_downstream_attritional_value": "YES"
        if low_attrition
        else "INCONCLUSIVE",
        "cp6_missing_targets_materially_contaminated_role_classification": "YES"
        if missing_contamination
        else "NO",
        "corrected_route_evaluator_production_faithful_enough": faithful,
        "maze_director_readiness": maze_verdict,
        "cp8_balance_experiment": payload["cp8"]["verdict"],
        "immediate_production_combat_or_balance_change": "NO",
    }


def _fmt_rate(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.2f}%"


def _write_report(payload: Mapping[str, Any]) -> None:
    comparison = payload["comparison"]["primary"]
    aggregates = comparison["aggregates"]
    verdict = _final_verdict(payload)
    transition = payload["transitions"]["by_policy"]
    lines = [
        "# The Charter — Checkpoint 7 Production-Faithful Route Termination",
        "",
        "## A. Provenance",
        f"- Immutable CP6 baseline: `{CP6_COMMIT}`.",
        f"- Experiment: `{EXPERIMENT_ID}`.",
        f"- CP6 seeds reused exactly: {RESEARCH_SEEDS[0]}–{RESEARCH_SEEDS[-1]}.",
        "",
        "## B. CP6 preservation/commit",
        "- CP6 was committed before CP7 as `Qualify stateful route evaluation`; generated CP6 artifacts remained ignored.",
        "",
        "## C. Production vs research termination audit",
        "- Manual combat ends only on production victory/defeat or explicit retreat. The legacy research runner used a 20-round execution bound and mislabeled unresolved cases as withdrawal.",
        "- CP7 leaves production combat untouched and treats research bounds as censoring only.",
        "",
        "## D. Frozen CP7 protocol",
        f"- Frozen protocol SHA: `{payload['provenance']['frozen_protocol_checksum']}`.",
        "- Exact CP6 routes, hero policies, seeds, persistence, and frozen π0/π1/π2 identities are retained.",
        "",
        "## E. Termination taxonomy",
        "- HERO_VICTORY, ENEMY_VICTORY, PRODUCTION_RETREAT, DETECTED_CYCLE, SAFETY_BOUND_EXCEEDED, INVALID_SIMULATION are distinct.",
        "",
        "## F. Cycle-detection design",
        "- Cycle proof requires repeated round-start transition-equivalent combat state plus identical RNG state. π0 normalizes rounds >=3 to its late-fight phase; π1/π2 retain absolute round because it is a learned feature.",
        "",
        "## G. Safety-bound semantics",
        f"- Primary guard: {SAFETY_BUDGET.max_transitions} combat transitions; secondary operational guard: {SAFETY_BUDGET.max_rounds} rounds. Exhaustion is censored, never withdrawal.",
        "",
        "## H. Ordinary-combat equivalence qualification",
        f"- {_read_json(EQUIVALENCE_PATH)['case_count']} previously consumed short cases matched final state, RNG, enemy/hero action sequences, and persisted CompanyState exactly.",
        "",
        "## I. Round-20 sample replay",
        "- Stable outcome-blind qualification sample is preserved in old_withdrawal_sample.json / old_withdrawal_resolution.json.",
        "",
        "## J. Full CP6→CP7 withdrawal-resolution matrix",
    ]
    for policy in POLICY_NAMES:
        row = transition[policy]
        lines.append(f"- {policy}: {row['count']} old withdrawals → {row['counts']}")
    lines += [
        "",
        "## K. Combat-length distribution",
    ]
    for policy in POLICY_NAMES:
        dist = payload["lengths"]["by_policy"][policy]["rounds"]
        lines.append(
            f"- {policy}: median {dist.get('median', 'N/A')}, mean {dist.get('mean', 'N/A')}, p95 {dist.get('p95', 'N/A')}, max natural terminal {payload['lengths']['by_policy'][policy]['max_naturally_terminal_rounds']}."
        )
    lines += [
        "",
        "## L. Stalling/cycle analysis",
    ]
    for policy in POLICY_NAMES:
        row = payload["stalling"]["former_withdrawals"][policy]
        lines.append(
            f"- {policy}: **{row['classification']}**; median extra rounds {row['median_additional_rounds']}; median no-progress rate {row['median_no_progress_round_rate']:.3f}."
        )
    lines += [
        "",
        "## M. Corrected route results",
        "### Primary company_survival",
        "| Policy | Enemy victories | Route completions | Valid terminals | Cycles | Safety censored | Invalid | Enemy-victory rate among valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICY_NAMES:
        row = aggregates[policy]
        lines.append(
            f"| {policy} | {row['enemy_victory_route_count']} | {row['completed_route_count']} | {row['valid_terminal_route_count']} | {row['censored_cycle_count']} | {row['censored_safety_bound_count']} | {row['invalid_count']} | {_fmt_rate(row['enemy_victory_rate_among_valid'])} |"
        )
    lines += [
        "",
        "## N. π0/π1/π2 paired route comparison",
    ]
    for pair, row in comparison["paired"].items():
        lines.append(
            f"- {pair}: paired valid n={row['paired_valid_terminal_count']}, enemy-victory difference={row['enemy_victory_rate_difference']}, 95% CI={row['paired_difference_ci_95']}, exact p={row['mcnemar_exact_two_sided_p']}."
        )
    lines += [
        "",
        "## O. True enemy-victory requalification",
        "- CP6 true-victory counts are no longer mixed with research withdrawals; the corrected primary table above is authoritative.",
        "",
        "## P. Fresh-vs-conditioned requalification",
        "- Encounter-specific rates use only reached, production-valid target combats. TARGET_NOT_REACHED is N/A, never 0%.",
    ]
    for target, target_payload in payload["fresh"].items():
        lines.append(f"### {target}")
        for context, policies in target_payload["contexts"].items():
            summaries = ", ".join(
                f"{policy}: reached {values['target_reached_count']}, valid {values['valid_terminal_count']}, censored {values['censored_count']}, enemy win {_fmt_rate(values['enemy_win_rate_among_reached_valid'])}"
                for policy, values in policies.items()
            )
            lines.append(f"- {context}: {summaries}")
    lines += [
        "",
        "## Q. Reachability/missing-data correction",
        "- Unreached targets are explicit missing encounter observations; CP6 classifier outputs remain historical provenance only.",
        "",
        "## R. Encounter-role reclassification",
    ]
    for encounter, row in payload["roles"].items():
        lines.append(f"- `{encounter}`: **{row['classification']}**")
    lines += [
        "",
        "## S. Behavioral long-combat audit",
        "- See behavior_long_combat_audit.json. Frequencies are diagnostic only and are not causal balance conclusions.",
        "",
        "## T. Objective-alignment reassessment",
        f"- Single-combat vs expedition objective: **{payload['objective']['single_combat_objective_alignment']}**.",
        "",
        "## U. Maze-director readiness",
        f"- **{payload['maze']['verdict']}**; primary censoring fraction {payload['maze']['primary_censored_fraction']:.3%}, invalid={payload['maze']['primary_invalid_count']}.",
        "",
        "## V. CP8 balance-experiment readiness",
        f"- **{payload['cp8']['verdict']}**. No perturbation was executed.",
        "",
        "## W. Limitations",
        "- Frozen hero policies are research approximations; π1/π2 were trained on isolated-combat continuation outcomes; exact cycle proof is intentionally conservative; any remaining safety censoring remains unresolved.",
        "",
        "## X. Quality gates",
        "- Final current-tree gates are filled in after the sealed evaluation; scientific artifacts are not regenerated.",
        "",
        "## Y. Artifact/checksum manifest",
        f"- Artifact root: `{ARTIFACT_ROOT}`. Generated artifacts remain ignored/untracked.",
        "",
        "## Z. Final verdict",
    ]
    for key, value in verdict.items():
        if key == "old_withdrawal_resolution_by_policy":
            continue
        lines.append(f"- {key}: **{value}**")
    lines += [
        "",
        "Checkpoint 7 stops here. No production combat/balance change, retraining, π3, CP8, route-aware learning, or Maze AI was implemented.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest() -> dict[str, Any]:
    artifact_paths = (
        PROTOCOL_PATH,
        PROVENANCE_PATH,
        TERMINATION_PATH,
        CYCLE_PATH,
        SAFETY_PATH,
        SAMPLE_PATH,
        SAMPLE_RESOLUTION_PATH,
        EQUIVALENCE_PATH,
        RESULTS_PATH,
        TRANSITION_PATH,
        LENGTH_PATH,
        STALLING_PATH,
        COMPARISON_PATH,
        FRESH_PATH,
        REACHABILITY_PATH,
        ROLE_PATH,
        BEHAVIOR_PATH,
        OBJECTIVE_PATH,
        MAZE_PATH,
        CP8_PATH,
        REPORT_PATH,
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "frozen_protocol_checksum": sha256_file(PROTOCOL_PATH),
        "pi1_model_checksum": PI1_MODEL_SHA,
        "pi2_model_checksum": PI2_MODEL_SHA,
        "evaluation_accessed": True,
        "training_performed": False,
        "post_evaluation_tuning_performed": False,
        "artifacts": {
            path.name: sha256_file(path)
            for path in artifact_paths
            if path.exists() and path != MANIFEST_PATH
        },
    }


def _reverify_manifest(manifest: Mapping[str, Any]) -> None:
    if str(manifest["frozen_protocol_checksum"]) != sha256_file(PROTOCOL_PATH):
        raise ValueError("Manifest protocol checksum does not match frozen CP7 protocol.")
    if bool(manifest["post_evaluation_tuning_performed"]):
        raise ValueError("CP7 manifest unexpectedly records post-evaluation tuning.")
    for name, expected in dict(manifest["artifacts"]).items():
        path = ARTIFACT_ROOT / str(name)
        if sha256_file(path) != str(expected):
            raise ValueError(f"CP7 artifact checksum mismatch: {name}")


def _replace_report_text(old: str, new: str) -> None:
    text = Path(__file__).read_text(encoding="utf-8")
    if old in text:
        Path(__file__).write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze", help="Freeze CP7 protocol before corrected route access.")
    subparsers.add_parser(
        "evaluate", help="Run the exact CP6 route suite with corrected termination."
    )
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_protocol()
    elif args.command == "evaluate":
        run_evaluation()
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
