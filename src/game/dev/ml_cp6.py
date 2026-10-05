# ruff: noqa: E501
"""Checkpoint 6 stateful multi-encounter route evaluation."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from typing import Any, cast

from game.campaign.company import create_new_company
from game.combat.enemy_learning import create_hero_policy, run_enemy_learning_episode
from game.core.rng import GameRng
from game.data.loaders import load_game_definitions
from game.dev.ml_cp5 import (
    ENCOUNTER_IDS,
    PI1_MODEL_SHA,
    PI2_MODEL_SHA,
    _load_frozen_policies,
)
from game.expedition.cave import create_encounter_combat
from game.ml.artifacts import sha256_file
from game.ml.confirmation_audit import mcnemar_exact, paired_difference_ci, wilson_interval
from game.ml.counterfactual import combat_state_digest, stable_digest
from game.ml.stateful_route import (
    MAX_ROUNDS,
    SEED_METHOD,
    ResearchRoute,
    aggregate_route_cases,
    encounter_attrition_rows,
    hp_fraction_band,
    paired_route_comparison,
    run_stateful_route,
)

EXPERIMENT_ID = "checkpoint6-stateful-route-evaluation-v1"
CP5_COMMIT = "b32ef8d66ac587dd59ded0e94a7624f65dd54e94"
CP5_ROOT = Path("artifacts/ml/checkpoint5-untouched-confirmation-v1")
ARTIFACT_ROOT = Path("artifacts/ml") / EXPERIMENT_ID
PROTOCOL_PATH = ARTIFACT_ROOT / "frozen_protocol.json"
PROVENANCE_PATH = ARTIFACT_ROOT / "provenance.json"
PERSISTENCE_PATH = ARTIFACT_ROOT / "persistence_semantics.json"
ROUTE_SUITE_PATH = ARTIFACT_ROOT / "route_suite.json"
BASELINE_PATH = ARTIFACT_ROOT / "baseline_reproduction.json"
ROUTE_RESULTS_PATH = ARTIFACT_ROOT / "route_results.json"
ROUTE_COMPARISON_PATH = ARTIFACT_ROOT / "route_policy_comparison.json"
ATTRITION_PATH = ARTIFACT_ROOT / "encounter_attrition.json"
FRESH_CONDITIONED_PATH = ARTIFACT_ROOT / "fresh_vs_conditioned.json"
DOWNSTREAM_PATH = ARTIFACT_ROOT / "downstream_value.json"
BEHAVIOR_PATH = ARTIFACT_ROOT / "behavior_revaluation.json"
OBJECTIVE_PATH = ARTIFACT_ROOT / "objective_mismatch_audit.json"
ROLE_PATH = ARTIFACT_ROOT / "encounter_role_classification.json"
MAZE_PATH = ARTIFACT_ROOT / "maze_relevance.json"
CP7_PATH = ARTIFACT_ROOT / "cp7_recommendation.json"
MANIFEST_PATH = ARTIFACT_ROOT / "manifest.json"
REPORT_PATH = ARTIFACT_ROOT / "REPORT.md"

POLICY_NAMES = ("pi0_heuristic", "pi1_cp3", "pi2_cp4")
POLICY_PAIRS = (
    ("pi0_heuristic", "pi1_cp3"),
    ("pi0_heuristic", "pi2_cp4"),
    ("pi1_cp3", "pi2_cp4"),
)
PRIMARY_HERO_POLICY = "company_survival"
SECONDARY_HERO_POLICIES = ("tactical", "anti_mark")
HERO_POLICIES = (PRIMARY_HERO_POLICY, *SECONDARY_HERO_POLICIES)
RESEARCH_SEEDS = tuple(range(40001, 40065))
BASELINE_SEEDS = tuple(range(20001, 20009))

PRIMARY_ROUTES = (
    ResearchRoute(
        "light_to_hunt",
        ("generated_maze_probe", "generated_maze_hunt"),
        "LIGHT_TO_HARD",
        "Measure downstream attrition from a low-lethality probe before Hunt.",
    ),
    ResearchRoute(
        "light_medium_hard",
        (
            "generated_maze_glass_pack",
            "generated_maze_pattern_cell",
            "generated_maze_hunt",
        ),
        "LIGHT_MEDIUM_HARD",
        "Measure compounding attrition before Hunt.",
    ),
    ResearchRoute(
        "control_to_hunt_guarded",
        ("generated_maze_pattern_cell", "generated_maze_hunt_guarded"),
        "CONTROL_TO_HARD",
        "Measure whether setup/control-heavy pressure worsens a later guarded Hunt.",
    ),
    ResearchRoute(
        "multiple_light_to_hunt",
        (
            "shallow_cave",
            "generated_maze_probe",
            "generated_maze_glass_pack",
            "generated_maze_hunt",
        ),
        "MULTIPLE_LIGHT_TO_HARD",
        "Measure cumulative value from several individually light encounters.",
    ),
    ResearchRoute(
        "fresh_hunt",
        ("generated_maze_hunt",),
        "HARD_FIRST",
        "Fresh-company Hunt control for conditioned-route comparison.",
    ),
    ResearchRoute(
        "authored_cave",
        ("shallow_cave", "cave_mini_boss"),
        "AUTHORED_CAVE_SEQUENCE",
        "Mirror the production opening cave combat sequence.",
    ),
)

FRESH_CONDITIONED_CONTEXTS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "generated_maze_hunt": {
        "fresh": ("generated_maze_hunt",),
        "one_predecessor": ("generated_maze_probe", "generated_maze_hunt"),
        "long_prefix": (
            "generated_maze_glass_pack",
            "generated_maze_pattern_cell",
            "generated_maze_hunt",
        ),
    },
    "generated_maze_hunt_guarded": {
        "fresh": ("generated_maze_hunt_guarded",),
        "one_predecessor": (
            "generated_maze_pattern_cell",
            "generated_maze_hunt_guarded",
        ),
        "long_prefix": (
            "generated_maze_probe",
            "generated_maze_pattern_cell",
            "generated_maze_hunt_guarded",
        ),
    },
    "cave_mini_boss": {
        "fresh": ("cave_mini_boss",),
        "one_predecessor": ("shallow_cave", "cave_mini_boss"),
        "long_prefix": ("road_bandits", "shallow_cave", "cave_mini_boss"),
    },
    "generated_maze_pattern_cell": {
        "fresh": ("generated_maze_pattern_cell",),
        "one_predecessor": ("generated_maze_probe", "generated_maze_pattern_cell"),
        "long_prefix": (
            "shallow_cave",
            "generated_maze_probe",
            "generated_maze_pattern_cell",
        ),
    },
    "maze_depth_1": {
        "fresh": ("maze_depth_1",),
        "one_predecessor": ("generated_maze_probe", "maze_depth_1"),
        "long_prefix": (
            "generated_maze_probe",
            "generated_maze_pattern_cell",
            "maze_depth_1",
        ),
    },
}

ROLE_THRESHOLDS = {
    "standalone_low_lethality": 0.05,
    "execution_conditioned_win_rate": 0.20,
    "meaningful_hp_fraction_loss": 0.08,
    "meaningful_effort_loss": 0.50,
    "meaningful_downs_or_wounds": 0.10,
    "negligible_hp_fraction_loss": 0.04,
    "negligible_effort_loss": 0.20,
}

IMPLEMENTATION_PATHS = (
    "src/game/dev/ml_cp6.py",
    "src/game/ml/stateful_route.py",
    "tests/test_ml_stateful_route.py",
    "src/game/dev/ml_cp5.py",
    "src/game/ml/confirmation_audit.py",
    "src/game/ml/policy_iteration.py",
    "src/game/ml/outcome_features.py",
    "src/game/ml/outcome_ranking.py",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CP6 stateful-route evaluation.")
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="Freeze route protocol before CP6 outcome access.")
    freeze.add_argument("--pre-access-tests-passed", action="store_true")
    sub.add_parser("evaluate", help="Run the frozen CP6 route evaluation exactly once.")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze_protocol(pre_access_tests_passed=bool(args.pre_access_tests_passed))
        return 0
    if args.command == "evaluate":
        run_evaluation()
        return 0
    raise ValueError(args.command)


def freeze_protocol(*, pre_access_tests_passed: bool) -> None:
    if not pre_access_tests_passed:
        raise ValueError("CP6 freeze requires explicit pre-access test evidence.")
    if ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.iterdir()):
        raise ValueError(
            "CP6 artifact root is not empty; refusing to overwrite research provenance."
        )
    if _git_head() != CP5_COMMIT:
        raise ValueError("CP6 must begin from the immutable CP5 baseline commit.")
    _assert_only_cp6_working_tree_changes()
    definitions = load_game_definitions()
    policies, identities = _load_frozen_policies()
    baseline = _baseline_reproduction(definitions, policies)
    persistence = _persistence_semantics()
    suite = _route_suite_payload()
    protocol = _protocol_payload(identities)

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=False)
    _write_json(PERSISTENCE_PATH, persistence)
    _write_json(ROUTE_SUITE_PATH, suite)
    _write_json(BASELINE_PATH, baseline)
    _write_json(PROTOCOL_PATH, protocol)
    cp5_provenance = _read_json(CP5_ROOT / "provenance.json")
    current_simulator = _hash_paths(tuple(cp5_provenance["simulator_hashes"]))
    current_content = _hash_paths(tuple(cp5_provenance["content_hashes"]))
    if current_simulator != cp5_provenance["simulator_hashes"]:
        raise ValueError("Production simulator differs from frozen CP5 baseline.")
    if current_content != cp5_provenance["content_hashes"]:
        raise ValueError("Production content differs from frozen CP5 baseline.")
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "cp5_commit": CP5_COMMIT,
        "current_head": _git_head(),
        "repository_dirty_for_cp6_implementation": _git_dirty(),
        "pre_access_tests_passed": True,
        "evaluation_accessed": False,
        "post_evaluation_tuning_performed": False,
        "frozen_protocol_checksum": sha256_file(PROTOCOL_PATH),
        "implementation_hashes": _hash_paths(IMPLEMENTATION_PATHS),
        "simulator_hashes": current_simulator,
        "content_hashes": current_content,
        "policies": identities,
        "rng_authority": {
            "root_seed_region": [RESEARCH_SEEDS[0], RESEARCH_SEEDS[-1]],
            "method": SEED_METHOD,
            "policy_independent_per_encounter_seed": True,
        },
        "runtime": {
            "python": platform.python_version(),
        },
        "baseline_reproduction": baseline["summary"],
    }
    _write_json(PROVENANCE_PATH, provenance)
    print(f"FROZEN_PROTOCOL_SHA={provenance['frozen_protocol_checksum']}")
    print("EVALUATION_ACCESSED=false")


def _protocol_payload(identities: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "baseline_commit": CP5_COMMIT,
        "scientific_question": (
            "How do frozen enemy policies perform when production hero state persists "
            "across multi-encounter routes?"
        ),
        "evaluation_accessed": False,
        "post_evaluation_tuning_performed": False,
        "policies": dict(identities),
        "research_seeds": [RESEARCH_SEEDS[0], RESEARCH_SEEDS[-1]],
        "seed_count": len(RESEARCH_SEEDS),
        "seed_method": SEED_METHOD,
        "max_rounds": MAX_ROUNDS,
        "hero_policies": list(HERO_POLICIES),
        "primary_hero_policy": PRIMARY_HERO_POLICY,
        "enemy_wait_mode": "none",
        "enemy_movement_mode": "recovery_only",
        "primary_endpoint": "company_fails_to_complete_route",
        "secondary_endpoints": [
            "final living hero count",
            "final HP and HP fraction",
            "final Effort and Effort spent",
            "Downs, deaths, Mortal Wounds",
            "morale/strain deterioration",
            "rounds",
            "cumulative damage",
            "enemy-caused Effort drain",
            "per-encounter deterioration",
        ],
        "route_suite": [asdict(route) for route in PRIMARY_ROUTES],
        "fresh_conditioned_contexts": {
            target: {name: list(sequence) for name, sequence in contexts.items()}
            for target, contexts in FRESH_CONDITIONED_CONTEXTS.items()
        },
        "role_classification_thresholds": ROLE_THRESHOLDS,
        "fresh_conditioned_hp_bands": ["0_25", "25_50", "50_75", "75_100"],
        "production_transition": (
            "create_encounter_combat -> run_enemy_learning_episode -> "
            "sync_company_from_combat(..., party_formation)"
        ),
        "forbidden": [
            "balance changes",
            "policy retraining",
            "pi3 training",
            "feature-schema changes",
            "search/RL/PPO/self-play",
            "player optimization",
            "Maze AI/hunting parties/reoccupation",
            "post-outcome route or seed changes",
        ],
    }


def _persistence_semantics() -> dict[str, Any]:
    return {
        "authoritative_path": {
            "create": "game.expedition.cave.create_encounter_combat",
            "sync": "game.campaign.roster.sync_company_from_combat",
            "interactive_finish": "game.app.flows._finish_manual_encounter",
        },
        "persists": {
            "hp": "exact, capped only by effective max on next combat construction",
            "effort": "exact company value; combat-start traits may reduce it again",
            "life_state": "ALIVE/DOWNED persists; DEAD heroes are removed from roster",
            "mortal_wounds": True,
            "morale": True,
            "strain": True,
            "strain_marks": True,
            "quirks": True,
            "personal_quirk": True,
            "gear": "equipped gear persists for survivors and stats are recomputed",
            "formation": "interactive production path persists terminal party formation",
            "tags": "all terminal tags except MARKED are retained by current sync semantics",
        },
        "resets_or_rebuilds": {
            "marked": "explicitly stripped by _persistent_tags between combats",
            "enemy_state": "new encounter enemies start at authored full HP/Effort",
            "combat_round": "new CombatState starts a new encounter round lifecycle",
            "combat_start_traits": "reapplied each encounter from persistent hero state",
        },
        "recovery": {
            "automatic_between_encounters": False,
            "town_service": "recover_company restores HP/Effort, revives Downed, clears strain/marks",
            "research_routes": "no synthetic recovery or healing between encounter sequence entries",
        },
        "known_helper_difference": (
            "Older automated expedition helpers sometimes omit party_formation when syncing; "
            "CP6 follows the interactive production path, which passes terminal formation."
        ),
    }


def _route_suite_payload() -> dict[str, Any]:
    return {
        "primary_routes": [asdict(route) for route in PRIMARY_ROUTES],
        "fresh_vs_conditioned": {
            target: {context: list(sequence) for context, sequence in contexts.items()}
            for target, contexts in FRESH_CONDITIONED_CONTEXTS.items()
        },
        "research_seed_range": [RESEARCH_SEEDS[0], RESEARCH_SEEDS[-1]],
        "hero_policies": list(HERO_POLICIES),
        "primary_hero_policy": PRIMARY_HERO_POLICY,
        "selection_basis": (
            "predeclared from authored encounter composition and CP5 already-frozen "
            "isolated evidence; no CP6 route outcomes were inspected"
        ),
    }


def _baseline_reproduction(
    definitions,
    policies: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    deterministic_probe: dict[str, Any] = {}
    for policy_name, policy in policies.items():
        marks = guards = grabs = bites = wins = 0
        for encounter_id in ENCOUNTER_IDS:
            for seed in BASELINE_SEEDS:
                state = create_encounter_combat(
                    create_new_company(definitions), definitions, encounter_id
                )
                episode = run_enemy_learning_episode(
                    state,
                    definitions,
                    GameRng(seed),
                    policy=policy,
                    max_rounds=MAX_ROUNDS,
                    encounter_id=encounter_id,
                    encounter_name=encounter_id.replace("_", " ").title(),
                    hero_policy=create_hero_policy(
                        PRIMARY_HERO_POLICY,
                        encounter_id=encounter_id,
                        seed=seed,
                    ),
                    enemy_wait_mode="none",
                    enemy_movement_mode="recovery_only",
                )
                wins += int(state.is_defeat())
                marks += episode.metrics.marks_applied
                guards += episode.metrics.guard_actions
                grabs += episode.metrics.boss_sequence.grab_uses
                bites += episode.metrics.boss_sequence.bite_uses
                rows.append(
                    {
                        "policy": policy_name,
                        "encounter_id": encounter_id,
                        "seed": seed,
                        "enemy_win": int(state.is_defeat()),
                        "marks": episode.metrics.marks_applied,
                        "guards": episode.metrics.guard_actions,
                        "grabs": episode.metrics.boss_sequence.grab_uses,
                        "bites": episode.metrics.boss_sequence.bite_uses,
                    }
                )
        deterministic_signatures: list[dict[str, Any]] = []
        for _ in range(2):
            state = create_encounter_combat(
                create_new_company(definitions),
                definitions,
                "cave_mini_boss",
            )
            episode = run_enemy_learning_episode(
                state,
                definitions,
                GameRng(BASELINE_SEEDS[0]),
                policy=policy,
                max_rounds=MAX_ROUNDS,
                encounter_id="cave_mini_boss",
                encounter_name="Cave Mini Boss",
                hero_policy=create_hero_policy(
                    PRIMARY_HERO_POLICY,
                    encounter_id="cave_mini_boss",
                    seed=BASELINE_SEEDS[0],
                ),
                enemy_wait_mode="none",
                enemy_movement_mode="recovery_only",
            )
            deterministic_signatures.append(
                {
                    "state": combat_state_digest(state),
                    "episode": stable_digest(episode),
                    "victor": episode.final_victor,
                }
            )
        if deterministic_signatures[0] != deterministic_signatures[1]:
            raise ValueError(f"Baseline deterministic replay failed for {policy_name}.")
        deterministic_probe[policy_name] = deterministic_signatures[0]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy"])].append(row)
    summary = {
        name: {
            "case_count": len(items),
            "enemy_wins": sum(int(row["enemy_win"]) for row in items),
            "enemy_win_rate": sum(int(row["enemy_win"]) for row in items) / len(items),
            "mean_marks": mean(float(row["marks"]) for row in items),
            "mean_guards": mean(float(row["guards"]) for row in items),
            "mean_grabs": mean(float(row["grabs"]) for row in items),
            "mean_bites": mean(float(row["bites"]) for row in items),
        }
        for name, items in grouped.items()
    }
    expected_wins = {"pi0_heuristic": 3, "pi1_cp3": 24, "pi2_cp4": 22}
    for name, expected in expected_wins.items():
        if int(summary[name]["enemy_wins"]) != expected:
            raise ValueError(
                f"CP4 consumed-seed baseline did not reproduce for {name}: "
                f"{summary[name]['enemy_wins']} != {expected}"
            )
    if not (
        float(summary["pi0_heuristic"]["mean_marks"])
        > float(summary["pi1_cp3"]["mean_marks"])
        > float(summary["pi2_cp4"]["mean_marks"])
    ):
        raise ValueError("Known low-Mark ordering failed baseline reproduction.")
    return {
        "seed_range": [BASELINE_SEEDS[0], BASELINE_SEEDS[-1]],
        "case_count_per_policy": len(ENCOUNTER_IDS) * len(BASELINE_SEEDS),
        "summary": summary,
        "deterministic_probe": deterministic_probe,
        "expected_cp4_enemy_wins": expected_wins,
        "status": "PASS",
    }


def run_evaluation() -> None:
    protocol, provenance = _validate_frozen_pre_access_state()
    if ROUTE_RESULTS_PATH.exists() or MANIFEST_PATH.exists():
        raise ValueError("CP6 route outcomes already exist; refusing to rerun frozen evaluation.")
    definitions = load_game_definitions()
    policies, _ = _load_frozen_policies()

    route_rows: list[dict[str, Any]] = []
    for policy_name in POLICY_NAMES:
        policy = policies[policy_name]
        for hero_policy_id in HERO_POLICIES:
            for route in PRIMARY_ROUTES:
                for seed in RESEARCH_SEEDS:
                    route_rows.append(
                        run_stateful_route(
                            definitions,
                            route,
                            root_seed=seed,
                            enemy_policy_name=policy_name,
                            enemy_policy=policy,
                            hero_policy_id=hero_policy_id,
                        )
                    )

    fresh_rows: list[dict[str, Any]] = []
    for target, contexts in FRESH_CONDITIONED_CONTEXTS.items():
        for context_name, sequence in contexts.items():
            route = ResearchRoute(
                route_id=f"freshcmp_{target}_{context_name}",
                encounter_ids=sequence,
                structure=f"FRESH_CONDITIONED_{context_name.upper()}",
                purpose=f"Fresh-versus-conditioned panel for {target}.",
                metadata={"target_encounter": target, "context": context_name},
                seed_namespace=f"freshcmp_{target}",
                seed_by_encounter_identity=True,
            )
            for policy_name in POLICY_NAMES:
                policy = policies[policy_name]
                for seed in RESEARCH_SEEDS:
                    fresh_rows.append(
                        run_stateful_route(
                            definitions,
                            route,
                            root_seed=seed,
                            enemy_policy_name=policy_name,
                            enemy_policy=policy,
                            hero_policy_id=PRIMARY_HERO_POLICY,
                        )
                    )

    route_comparison = _route_policy_comparison(route_rows)
    attrition_rows = encounter_attrition_rows(route_rows)
    attrition = _encounter_attrition_analysis(attrition_rows)
    fresh_conditioned = _fresh_conditioned_analysis(fresh_rows)
    downstream = _downstream_value_analysis(route_rows)
    behavior = _behavior_revaluation(route_rows, fresh_rows)
    objective = _objective_mismatch_audit(route_rows, behavior, downstream)
    roles = _encounter_role_classification(attrition, fresh_conditioned)
    maze = _maze_relevance(route_comparison, downstream, roles)
    cp7 = _cp7_recommendation(behavior, objective)

    _write_json(
        ROUTE_RESULTS_PATH,
        {
            "experiment_id": EXPERIMENT_ID,
            "rows": route_rows,
            "summary": route_comparison["aggregates"],
        },
    )
    _write_json(ROUTE_COMPARISON_PATH, route_comparison)
    _write_json(ATTRITION_PATH, attrition)
    _write_json(FRESH_CONDITIONED_PATH, fresh_conditioned)
    _write_json(DOWNSTREAM_PATH, downstream)
    _write_json(BEHAVIOR_PATH, behavior)
    _write_json(OBJECTIVE_PATH, objective)
    _write_json(ROLE_PATH, roles)
    _write_json(MAZE_PATH, maze)
    _write_json(CP7_PATH, cp7)

    report_payload = {
        "protocol": protocol,
        "provenance": provenance,
        "route_comparison": route_comparison,
        "attrition": attrition,
        "fresh_conditioned": fresh_conditioned,
        "downstream": downstream,
        "behavior": behavior,
        "objective": objective,
        "roles": roles,
        "maze": maze,
        "cp7": cp7,
    }
    _write_report(report_payload)
    manifest = _manifest()
    _write_json(MANIFEST_PATH, manifest)
    _reverify_manifest(manifest)
    print(f"ROUTES={len(route_rows)}")
    print(f"FRESH_CONDITIONED_ROUTES={len(fresh_rows)}")
    print(f"FROZEN_PROTOCOL_SHA={provenance['frozen_protocol_checksum']}")
    print("POST_EVALUATION_TUNING=false")


def _route_policy_comparison(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["hero_policy_id"]), str(row["route_id"]), str(row["enemy_policy"]))
        ].append(row)

    aggregates: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for hero_policy_id in HERO_POLICIES:
        aggregates[hero_policy_id] = {}
        comparisons[hero_policy_id] = {}
        for route in PRIMARY_ROUTES:
            by_policy = {
                policy: grouped[(hero_policy_id, route.route_id, policy)] for policy in POLICY_NAMES
            }
            aggregates[hero_policy_id][route.route_id] = {
                policy: aggregate_route_cases(policy_rows)
                for policy, policy_rows in by_policy.items()
            }
            comparisons[hero_policy_id][route.route_id] = {
                f"{first}_to_{second}": paired_route_comparison(by_policy[first], by_policy[second])
                for first, second in POLICY_PAIRS
            }

    overall: dict[str, Any] = {}
    for hero_policy_id in HERO_POLICIES:
        by_policy = {
            policy: [
                row
                for row in rows
                if row["hero_policy_id"] == hero_policy_id and row["enemy_policy"] == policy
            ]
            for policy in POLICY_NAMES
        }
        overall[hero_policy_id] = {
            "aggregates": {
                policy: aggregate_route_cases(policy_rows)
                for policy, policy_rows in by_policy.items()
            },
            "comparisons": {
                f"{first}_to_{second}": paired_route_comparison(by_policy[first], by_policy[second])
                for first, second in POLICY_PAIRS
            },
        }
    return {"aggregates": aggregates, "paired_comparisons": comparisons, "overall": overall}


def _quantile(values: Sequence[float], q: float) -> float:
    samples = sorted(float(value) for value in values)
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]
    position = (len(samples) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return samples[low]
    weight = position - low
    return samples[low] * (1.0 - weight) + samples[high] * weight


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


def _encounter_attrition_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_route_position: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["hero_policy_id"]),
            str(row["enemy_policy"]),
            str(row["encounter_id"]),
        )
        grouped[key].append(row)
        by_route_position[
            (
                str(row["hero_policy_id"]),
                str(row["enemy_policy"]),
                str(row["route_id"]),
                int(row["encounter_index"]),
            )
        ].append(row)

    by_encounter: dict[str, Any] = {}
    for hero_policy_id in HERO_POLICIES:
        by_encounter[hero_policy_id] = {}
        encounter_ids = sorted(
            {str(row["encounter_id"]) for row in rows if row["hero_policy_id"] == hero_policy_id}
        )
        for encounter_id in encounter_ids:
            by_encounter[hero_policy_id][encounter_id] = {}
            for policy in POLICY_NAMES:
                items = grouped[(hero_policy_id, policy, encounter_id)]
                if not items:
                    continue
                hp_fraction_losses = [
                    float(row["hp_loss"]) / max(1.0, float(row["entry_max_hp"])) for row in items
                ]
                enemy_wins = sum(int(row["enemy_win"]) for row in items)
                by_encounter[hero_policy_id][encounter_id][policy] = {
                    "reached_count": len(items),
                    "enemy_win_count": enemy_wins,
                    "enemy_win_rate": enemy_wins / len(items),
                    "enemy_win_wilson_95": wilson_interval(enemy_wins, len(items)),
                    "mean_entry_hp_fraction": mean(
                        float(row["entry_hp_fraction"]) for row in items
                    ),
                    "mean_hp_loss": mean(float(row["hp_loss"]) for row in items),
                    "mean_hp_fraction_loss": mean(hp_fraction_losses),
                    "hp_fraction_loss_distribution": {
                        "p25": _quantile(hp_fraction_losses, 0.25),
                        "median": _quantile(hp_fraction_losses, 0.50),
                        "p75": _quantile(hp_fraction_losses, 0.75),
                    },
                    "mean_effort_loss": mean(float(row["effort_loss"]) for row in items),
                    "mean_total_hero_damage": mean(
                        float(row["total_hero_damage"]) for row in items
                    ),
                    "mean_enemy_effort_drain": mean(
                        float(row["enemy_effort_drain"]) for row in items
                    ),
                    "mean_downs": mean(float(row["downs"]) for row in items),
                    "mean_deaths": mean(float(row["deaths"]) for row in items),
                    "mean_mortal_wounds": mean(float(row["mortal_wounds"]) for row in items),
                    "mean_rounds": mean(float(row["rounds"]) for row in items),
                }

    position_summary: dict[str, Any] = {}
    for position_key, items in sorted(by_route_position.items()):
        hero_policy_id, policy, route_id, encounter_index = position_key
        position_summary.setdefault(hero_policy_id, {}).setdefault(policy, {}).setdefault(
            route_id, {}
        )[str(encounter_index)] = {
            "encounter_id": str(items[0]["encounter_id"]),
            "reached_count": len(items),
            "mean_entry_hp_fraction": mean(float(row["entry_hp_fraction"]) for row in items),
            "mean_exit_hp_fraction": mean(float(row["exit_hp_fraction"]) for row in items),
            "mean_hp_loss": mean(float(row["hp_loss"]) for row in items),
            "mean_effort_loss": mean(float(row["effort_loss"]) for row in items),
        }
    return {"by_encounter": by_encounter, "by_route_position": position_summary}


def _target_encounter_record(
    route_row: Mapping[str, Any],
    target: str,
) -> Mapping[str, Any] | None:
    for encounter in route_row["encounters"]:
        if encounter["encounter_id"] == target:
            return encounter
    return None


def _fresh_conditioned_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target, contexts in FRESH_CONDITIONED_CONTEXTS.items():
        result[target] = {"contexts": {}, "paired": {}, "hp_pressure_curve": {}}
        for context_name in contexts:
            result[target]["contexts"][context_name] = {}
            for policy in POLICY_NAMES:
                items = [
                    row
                    for row in rows
                    if row["route_id"] == f"freshcmp_{target}_{context_name}"
                    and row["enemy_policy"] == policy
                ]
                target_records = [(row, _target_encounter_record(row, target)) for row in items]
                reached = [(row, record) for row, record in target_records if record is not None]
                target_wins = sum(int(record["enemy_win"]) for _, record in reached)
                failed_before = sum(record is None for _, record in target_records)
                result[target]["contexts"][context_name][policy] = {
                    "case_count": len(items),
                    "target_reached_count": len(reached),
                    "target_reach_rate": len(reached) / len(items) if items else 0.0,
                    "failed_before_target": failed_before,
                    "target_enemy_wins": target_wins,
                    "target_enemy_win_rate_if_reached": target_wins / len(reached)
                    if reached
                    else 0.0,
                    "target_enemy_win_wilson_95_if_reached": wilson_interval(
                        target_wins, len(reached)
                    ),
                    "failure_before_or_at_target": sum(bool(row["route_failed"]) for row in items),
                    "failure_before_or_at_target_rate": (
                        sum(bool(row["route_failed"]) for row in items) / len(items)
                        if items
                        else 0.0
                    ),
                    "mean_target_entry_hp_fraction": mean(
                        float(record["combat_entry"]["hp_fraction"]) for _, record in reached
                    )
                    if reached
                    else 0.0,
                    "median_target_entry_hp_fraction": median(
                        float(record["combat_entry"]["hp_fraction"]) for _, record in reached
                    )
                    if reached
                    else 0.0,
                    "mean_target_entry_effort": mean(
                        float(record["combat_entry"]["effort_total"]) for _, record in reached
                    )
                    if reached
                    else 0.0,
                    "mean_target_entry_mortal_wounds": mean(
                        float(record["combat_entry"]["mortal_wounds_total"])
                        for _, record in reached
                    )
                    if reached
                    else 0.0,
                    "mean_target_hp_loss": mean(
                        float(record["deterioration"]["hp_loss"]) for _, record in reached
                    )
                    if reached
                    else 0.0,
                    "mean_target_rounds": mean(float(record["rounds"]) for _, record in reached)
                    if reached
                    else 0.0,
                }

        for policy in POLICY_NAMES:
            base_rows = {
                int(row["root_seed"]): row
                for row in rows
                if row["route_id"] == f"freshcmp_{target}_fresh" and row["enemy_policy"] == policy
            }
            for context_name in ("one_predecessor", "long_prefix"):
                conditioned_rows = {
                    int(row["root_seed"]): row
                    for row in rows
                    if row["route_id"] == f"freshcmp_{target}_{context_name}"
                    and row["enemy_policy"] == policy
                }
                all_seeds = sorted(base_rows.keys() & conditioned_rows.keys())
                route_failure_diffs = [
                    int(conditioned_rows[seed]["route_failed"])
                    - int(base_rows[seed]["route_failed"])
                    for seed in all_seeds
                ]
                matched_reached = []
                first_only = second_only = 0
                for seed in all_seeds:
                    a = _target_encounter_record(base_rows[seed], target)
                    b = _target_encounter_record(conditioned_rows[seed], target)
                    if a is None or b is None:
                        continue
                    matched_reached.append(seed)
                    aw = bool(a["enemy_win"])
                    bw = bool(b["enemy_win"])
                    if aw and not bw:
                        first_only += 1
                    elif bw and not aw:
                        second_only += 1
                target_win_diffs: list[int] = []
                for seed in matched_reached:
                    conditioned_target = _target_encounter_record(conditioned_rows[seed], target)
                    fresh_target = _target_encounter_record(base_rows[seed], target)
                    if conditioned_target is None or fresh_target is None:
                        raise ValueError("Matched target disappeared during CP6 analysis.")
                    target_win_diffs.append(
                        int(conditioned_target["enemy_win"]) - int(fresh_target["enemy_win"])
                    )
                result[target]["paired"].setdefault(policy, {})[context_name] = {
                    "all_seed_count": len(all_seeds),
                    "route_failure_difference": mean(route_failure_diffs)
                    if route_failure_diffs
                    else 0.0,
                    "route_failure_difference_ci_95": paired_difference_ci(route_failure_diffs),
                    "matched_target_reached_count": len(matched_reached),
                    "target_enemy_win_difference_if_both_reached": mean(target_win_diffs)
                    if target_win_diffs
                    else 0.0,
                    "target_enemy_win_difference_ci_95": paired_difference_ci(target_win_diffs),
                    "target_mcnemar_exact_two_sided_p": mcnemar_exact(first_only, second_only),
                    "fresh_only_target_win": first_only,
                    "conditioned_only_target_win": second_only,
                }

        band_rows: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            if not str(row["route_id"]).startswith(f"freshcmp_{target}_"):
                continue
            record = _target_encounter_record(row, target)
            if record is None:
                continue
            band = hp_fraction_band(float(record["combat_entry"]["hp_fraction"]))
            band_rows[(str(row["enemy_policy"]), band)].append(record)
        for policy in POLICY_NAMES:
            result[target]["hp_pressure_curve"][policy] = {}
            for band in ("0_25", "25_50", "50_75", "75_100"):
                items = band_rows[(policy, band)]
                wins = sum(int(item["enemy_win"]) for item in items)
                result[target]["hp_pressure_curve"][policy][band] = {
                    "count": len(items),
                    "enemy_wins": wins,
                    "enemy_win_rate": wins / len(items) if items else 0.0,
                    "mean_entry_hp_fraction": mean(
                        float(item["combat_entry"]["hp_fraction"]) for item in items
                    )
                    if items
                    else 0.0,
                    "mean_hp_loss": mean(float(item["deterioration"]["hp_loss"]) for item in items)
                    if items
                    else 0.0,
                    "mean_rounds": mean(float(item["rounds"]) for item in items) if items else 0.0,
                }
    return result


def _downstream_value_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for route_row in rows:
        encounters = list(route_row["encounters"])
        for index, encounter in enumerate(encounters[:-1]):
            if int(encounter["hero_win"]) != 1:
                continue
            next_encounter = encounters[index + 1]
            groups[
                (
                    str(route_row["hero_policy_id"]),
                    str(route_row["enemy_policy"]),
                    str(route_row["route_id"]),
                    index,
                )
            ].append(
                {
                    "encounter_id": encounter["encounter_id"],
                    "seed": route_row["root_seed"],
                    "hp_loss": encounter["deterioration"]["hp_loss"],
                    "hp_fraction_loss": float(encounter["deterioration"]["hp_loss"])
                    / max(1.0, float(encounter["combat_entry"]["max_hp_total"])),
                    "effort_loss": encounter["deterioration"]["effort_loss"],
                    "enemy_effort_drain": encounter["enemy_caused_effort_drain"],
                    "downs": encounter["downs"],
                    "mortal_wounds": encounter["mortal_wounds"],
                    "next_entry_hp_fraction": next_encounter["combat_entry"]["hp_fraction"],
                    "next_entry_effort": next_encounter["combat_entry"]["effort_total"],
                    "next_entry_mortal_wounds": next_encounter["combat_entry"][
                        "mortal_wounds_total"
                    ],
                    "route_later_failed": int(route_row["route_failed"]),
                }
            )

    cells: dict[str, Any] = {}
    low_lethality_value: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        hero_policy, policy, route_id, index = key
        label = f"{hero_policy}|{policy}|{route_id}|{index}"
        cell = {
            "hero_policy_id": hero_policy,
            "enemy_policy": policy,
            "route_id": route_id,
            "encounter_index": index,
            "encounter_id": str(items[0]["encounter_id"]),
            "survived_early_encounter_count": len(items),
            "mean_hp_loss": mean(float(item["hp_loss"]) for item in items),
            "mean_hp_fraction_loss": mean(float(item["hp_fraction_loss"]) for item in items),
            "mean_effort_loss": mean(float(item["effort_loss"]) for item in items),
            "mean_enemy_effort_drain": mean(float(item["enemy_effort_drain"]) for item in items),
            "mean_downs": mean(float(item["downs"]) for item in items),
            "mean_mortal_wounds": mean(float(item["mortal_wounds"]) for item in items),
            "mean_next_entry_hp_fraction": mean(
                float(item["next_entry_hp_fraction"]) for item in items
            ),
            "mean_next_entry_effort": mean(float(item["next_entry_effort"]) for item in items),
            "mean_next_entry_mortal_wounds": mean(
                float(item["next_entry_mortal_wounds"]) for item in items
            ),
            "later_route_failure_rate": mean(float(item["route_later_failed"]) for item in items),
        }
        cells[label] = cell
        if hero_policy == PRIMARY_HERO_POLICY and (
            float(cast(Any, cell["mean_hp_fraction_loss"]))
            >= float(ROLE_THRESHOLDS["meaningful_hp_fraction_loss"])
            or float(cast(Any, cell["mean_effort_loss"]))
            >= float(ROLE_THRESHOLDS["meaningful_effort_loss"])
            or float(cast(Any, cell["mean_mortal_wounds"]))
            >= float(ROLE_THRESHOLDS["meaningful_downs_or_wounds"])
        ):
            low_lethality_value.append(cell)

    paired_next_state: dict[str, Any] = {}
    for route_def in PRIMARY_ROUTES:
        for index in range(max(0, len(route_def.encounter_ids) - 1)):
            for first, second in POLICY_PAIRS:
                first_items = {
                    int(row["root_seed"]): row
                    for row in rows
                    if row["route_id"] == route_def.route_id
                    and row["hero_policy_id"] == PRIMARY_HERO_POLICY
                    and row["enemy_policy"] == first
                    and len(row["encounters"]) > index + 1
                    and row["encounters"][index]["hero_win"]
                }
                second_items = {
                    int(row["root_seed"]): row
                    for row in rows
                    if row["route_id"] == route_def.route_id
                    and row["hero_policy_id"] == PRIMARY_HERO_POLICY
                    and row["enemy_policy"] == second
                    and len(row["encounters"]) > index + 1
                    and row["encounters"][index]["hero_win"]
                }
                seeds = sorted(first_items.keys() & second_items.keys())
                hp_diffs = [
                    float(
                        second_items[seed]["encounters"][index + 1]["combat_entry"]["hp_fraction"]
                    )
                    - float(
                        first_items[seed]["encounters"][index + 1]["combat_entry"]["hp_fraction"]
                    )
                    for seed in seeds
                ]
                effort_diffs = [
                    float(
                        second_items[seed]["encounters"][index + 1]["combat_entry"]["effort_total"]
                    )
                    - float(
                        first_items[seed]["encounters"][index + 1]["combat_entry"]["effort_total"]
                    )
                    for seed in seeds
                ]
                paired_next_state[f"{route_def.route_id}|{index}|{first}_to_{second}"] = {
                    "matched_survivor_count": len(seeds),
                    "mean_next_entry_hp_fraction_difference": mean(hp_diffs) if hp_diffs else 0.0,
                    "next_entry_hp_fraction_difference_ci_95": _mean_ci(hp_diffs),
                    "mean_next_entry_effort_difference": mean(effort_diffs)
                    if effort_diffs
                    else 0.0,
                    "next_entry_effort_difference_ci_95": _mean_ci(effort_diffs),
                }
    return {
        "cells": cells,
        "meaningful_early_attrition_cells": low_lethality_value,
        "paired_next_state": paired_next_state,
    }


def _behavior_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(records)
    actions = sum(int(item["enemy_actions"]) for item in items)
    skill_counts: Counter[str] = Counter()
    for item in items:
        skill_counts.update({str(k): int(v) for k, v in item.get("skill_uses", {}).items()})
    total_skills = sum(skill_counts.values())
    entropy = 0.0
    if total_skills:
        for count in skill_counts.values():
            p = count / total_skills
            entropy -= p * math.log2(p)

    def total(key: str) -> int:
        return sum(int(item[key]) for item in items)

    return {
        "encounter_count": len(items),
        "enemy_actions": actions,
        "marks": total("marks_applied"),
        "guards": total("guard_uses"),
        "grabs": total("grab_uses"),
        "bites": total("bite_uses"),
        "effort_drains": total("effort_drain_uses"),
        "direct_pressure": total("direct_pressure_uses"),
        "moves": total("enemy_moves"),
        "waits": total("enemy_waits"),
        "target_switches": total("target_switches"),
        "rates_per_enemy_action": {
            "mark": total("marks_applied") / actions if actions else 0.0,
            "guard": total("guard_uses") / actions if actions else 0.0,
            "grab": total("grab_uses") / actions if actions else 0.0,
            "bite": total("bite_uses") / actions if actions else 0.0,
            "effort_drain": total("effort_drain_uses") / actions if actions else 0.0,
            "direct_pressure": total("direct_pressure_uses") / actions if actions else 0.0,
            "movement": total("enemy_moves") / actions if actions else 0.0,
            "waiting": total("enemy_waits") / actions if actions else 0.0,
        },
        "skill_entropy_bits": entropy,
        "distinct_skills_selected": len(skill_counts),
        "skill_counts": dict(sorted(skill_counts.items())),
    }


def _behavior_revaluation(
    route_rows: Sequence[Mapping[str, Any]],
    fresh_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fresh_panel: dict[str, Any] = {}
    for target, contexts in FRESH_CONDITIONED_CONTEXTS.items():
        fresh_panel[target] = {}
        for policy in POLICY_NAMES:
            fresh_panel[target][policy] = {}
            for context_name in contexts:
                fresh_records: list[Mapping[str, Any]] = []
                for row in fresh_rows:
                    if row["route_id"] != f"freshcmp_{target}_{context_name}":
                        continue
                    if row["enemy_policy"] != policy:
                        continue
                    target_record = _target_encounter_record(row, target)
                    if target_record is not None:
                        fresh_records.append(target_record)
                fresh_panel[target][policy][context_name] = _behavior_summary(fresh_records)

    route_position: dict[str, Any] = {}
    for policy in POLICY_NAMES:
        route_position[policy] = {}
        for label in ("first", "middle", "final"):
            position_records: list[Mapping[str, Any]] = []
            for row in route_rows:
                if row["hero_policy_id"] != PRIMARY_HERO_POLICY or row["enemy_policy"] != policy:
                    continue
                encounters = row["encounters"]
                for index, record in enumerate(encounters):
                    position = (
                        "first"
                        if index == 0
                        else "final"
                        if index == len(row["encounter_sequence"]) - 1
                        else "middle"
                    )
                    if position == label:
                        position_records.append(record)
            route_position[policy][label] = _behavior_summary(position_records)
    return {"fresh_conditioned_targets": fresh_panel, "route_position": route_position}


def _objective_mismatch_audit(
    route_rows: Sequence[Mapping[str, Any]],
    behavior: Mapping[str, Any],
    downstream: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "mark": "marks_applied",
        "grab": "grab_uses",
        "guard": "guard_uses",
        "effort_drain": "effort_drain_uses",
        "direct_pressure": "direct_pressure_uses",
    }
    associations: dict[str, Any] = {}
    for behavior_name, field in fields.items():
        associations[behavior_name] = {}
        for policy in POLICY_NAMES:
            used: list[Mapping[str, Any]] = []
            unused: list[Mapping[str, Any]] = []
            for route in route_rows:
                if route["hero_policy_id"] != PRIMARY_HERO_POLICY:
                    continue
                if route["enemy_policy"] != policy or not route["encounters"]:
                    continue
                first = route["encounters"][0]
                (used if int(first[field]) > 0 else unused).append(route)

            def summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
                if not items:
                    return {"case_count": 0}
                immediate_losses = [
                    float(row["encounters"][0]["deterioration"]["hp_loss"])
                    / max(1.0, float(row["encounters"][0]["combat_entry"]["max_hp_total"]))
                    for row in items
                ]
                return {
                    "case_count": len(items),
                    "first_encounter_enemy_win_rate": mean(
                        float(row["encounters"][0]["enemy_win"]) for row in items
                    ),
                    "mean_immediate_hp_fraction_loss": mean(immediate_losses),
                    "route_failure_rate": mean(float(row["route_failed"]) for row in items),
                    "mean_final_hp_fraction": mean(
                        float(row["final_state"]["hp_fraction"]) for row in items
                    ),
                }

            used_summary = summarize(used)
            unused_summary = summarize(unused)
            category = "D_NO_MEANINGFUL_VALUE_DETECTED"
            if used and unused:
                immediate_diff = float(used_summary["mean_immediate_hp_fraction_loss"]) - float(
                    unused_summary["mean_immediate_hp_fraction_loss"]
                )
                route_diff = float(used_summary["route_failure_rate"]) - float(
                    unused_summary["route_failure_rate"]
                )
                if immediate_diff >= 0.03 and route_diff >= 0.05:
                    category = "A_IMMEDIATE_AND_ROUTE_VALUE_ALIGNED"
                elif immediate_diff < 0.03 and route_diff >= 0.05:
                    category = "B_IMMEDIATE_WEAK_ROUTE_LEVEL_VALUABLE"
                elif immediate_diff >= 0.03 and route_diff <= -0.05:
                    category = "C_IMMEDIATE_VALUABLE_ROUTE_LEVEL_HARMFUL"
            associations[behavior_name][policy] = {
                "used_in_first_encounter": used_summary,
                "not_used_in_first_encounter": unused_summary,
                "diagnostic_category": category,
                "warning": "Observational route association; not a causal mechanic perturbation.",
            }

    context_shift: dict[str, Any] = {}
    for behavior_name in fields:
        learned_shifts: list[float] = []
        for _target, target_rows in behavior["fresh_conditioned_targets"].items():
            for policy in ("pi1_cp3", "pi2_cp4"):
                fresh = float(target_rows[policy]["fresh"]["rates_per_enemy_action"][behavior_name])
                long_value = float(
                    target_rows[policy]["long_prefix"]["rates_per_enemy_action"][behavior_name]
                )
                learned_shifts.append(long_value - fresh)
        context_shift[behavior_name] = {
            "mean_long_minus_fresh_action_rate": mean(learned_shifts) if learned_shifts else 0.0,
            "absolute_mean_shift": mean(abs(value) for value in learned_shifts)
            if learned_shifts
            else 0.0,
        }

    mismatch_categories = {
        item["diagnostic_category"]
        for behavior_rows in associations.values()
        for policy, item in behavior_rows.items()
        if policy in {"pi1_cp3", "pi2_cp4"}
    }
    if any(category.startswith(("B_", "C_")) for category in mismatch_categories):
        overall = "PARTIALLY"
    elif downstream["meaningful_early_attrition_cells"]:
        overall = "PARTIALLY"
    else:
        overall = "INCONCLUSIVE"
    return {
        "behavior_associations": associations,
        "route_context_action_rate_shift": context_shift,
        "single_combat_objective_alignment": overall,
        "interpretation_rule": (
            "B/C categories or meaningful downstream attrition indicate that current-combat "
            "outcomes do not capture every route-relevant consequence."
        ),
    }


def _encounter_role_classification(
    attrition: Mapping[str, Any],
    fresh_conditioned: Mapping[str, Any],
) -> dict[str, Any]:
    isolated = _read_json(CP5_ROOT / "encounter_breakdown.json")
    route_encounters = sorted(
        {
            *(encounter for route in PRIMARY_ROUTES for encounter in route.encounter_ids),
            *(
                encounter
                for contexts in FRESH_CONDITIONED_CONTEXTS.values()
                for sequence in contexts.values()
                for encounter in sequence
            ),
        }
    )
    result: dict[str, Any] = {}
    for encounter_id in route_encounters:
        isolated_rates = [
            float(isolated[encounter_id][policy]["enemy_win_rate"])
            for policy in ("pi1_cp3", "pi2_cp4")
        ]
        isolated_learned = mean(isolated_rates)
        attrition_rows = attrition["by_encounter"][PRIMARY_HERO_POLICY].get(encounter_id, {})
        learned_attrition = [
            attrition_rows[policy] for policy in ("pi1_cp3", "pi2_cp4") if policy in attrition_rows
        ]
        hp_loss = (
            mean(float(row["mean_hp_fraction_loss"]) for row in learned_attrition)
            if learned_attrition
            else 0.0
        )
        effort_loss = (
            mean(float(row["mean_effort_loss"]) for row in learned_attrition)
            if learned_attrition
            else 0.0
        )
        wound_pressure = (
            mean(
                float(row["mean_downs"]) + float(row["mean_mortal_wounds"])
                for row in learned_attrition
            )
            if learned_attrition
            else 0.0
        )
        route_win = max(
            (float(row["enemy_win_rate"]) for row in learned_attrition),
            default=0.0,
        )
        conditioned_rate = route_win
        if encounter_id in fresh_conditioned:
            conditioned_rate = max(
                float(
                    fresh_conditioned[encounter_id]["contexts"][context][policy][
                        "target_enemy_win_rate_if_reached"
                    ]
                )
                for context in ("one_predecessor", "long_prefix")
                for policy in ("pi1_cp3", "pi2_cp4")
            )
        execution = conditioned_rate >= ROLE_THRESHOLDS["execution_conditioned_win_rate"]
        attritional = isolated_learned < ROLE_THRESHOLDS["standalone_low_lethality"] and (
            hp_loss >= ROLE_THRESHOLDS["meaningful_hp_fraction_loss"]
            or effort_loss >= ROLE_THRESHOLDS["meaningful_effort_loss"]
            or wound_pressure >= ROLE_THRESHOLDS["meaningful_downs_or_wounds"]
        )
        pressure = (
            conditioned_rate >= ROLE_THRESHOLDS["standalone_low_lethality"]
            or hp_loss >= ROLE_THRESHOLDS["meaningful_hp_fraction_loss"]
        )
        negligible = (
            conditioned_rate < ROLE_THRESHOLDS["standalone_low_lethality"]
            and hp_loss < ROLE_THRESHOLDS["negligible_hp_fraction_loss"]
            and effort_loss < ROLE_THRESHOLDS["negligible_effort_loss"]
            and wound_pressure < ROLE_THRESHOLDS["meaningful_downs_or_wounds"]
        )
        if execution and attritional:
            role = "MIXED"
        elif execution:
            role = "EXECUTION"
        elif attritional:
            role = "ATTRITION"
        elif negligible:
            role = "NEGLIGIBLE_UNDER_CURRENT_RULES"
        elif pressure:
            role = "PRESSURE"
        else:
            role = "MIXED"
        result[encounter_id] = {
            "classification": role,
            "isolated_learned_enemy_win_rate_mean": isolated_learned,
            "max_conditioned_or_route_enemy_win_rate": conditioned_rate,
            "mean_route_hp_fraction_loss": hp_loss,
            "mean_route_effort_loss": effort_loss,
            "mean_downs_plus_mortal_wounds": wound_pressure,
            "thresholds": ROLE_THRESHOLDS,
            "diagnostic_only": True,
        }
    return result


def _maze_relevance(
    route_comparison: Mapping[str, Any],
    downstream: Mapping[str, Any],
    roles: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "route_evaluator_role": (
            "Provides a persistent-company environment in which a future director could compare "
            "legal encounter/room sequences against current party state without changing combat rules."
        ),
        "conceptual_future_loop": [
            "observe current persistent company state",
            "enumerate legal encounter/room continuations",
            "simulate stateful route continuations",
            "estimate route completion/failure and deterioration",
            "choose pressure subject to future design constraints",
        ],
        "required_by_future_features": [
            "hunting parties",
            "reoccupation",
            "alternate exits",
            "navigation pressure",
        ],
        "current_cp6_boundary": "analysis_only_no_maze_code",
        "meaningful_downstream_attrition_cells": len(
            downstream["meaningful_early_attrition_cells"]
        ),
        "encounter_roles": {key: value["classification"] for key, value in roles.items()},
        "route_policy_data_available": bool(route_comparison["aggregates"]),
    }


def _cp7_recommendation(
    behavior: Mapping[str, Any],
    objective: Mapping[str, Any],
) -> dict[str, Any]:
    mechanics = ("mark", "grab", "guard", "effort_drain", "direct_pressure")
    result: dict[str, Any] = {}
    for mechanic in mechanics:
        use_count = 0
        for target_rows in behavior["fresh_conditioned_targets"].values():
            for policy in ("pi1_cp3", "pi2_cp4"):
                for context in ("fresh", "long_prefix"):
                    row = target_rows[policy][context]
                    count_key = {
                        "mark": "marks",
                        "grab": "grabs",
                        "guard": "guards",
                        "effort_drain": "effort_drains",
                        "direct_pressure": "direct_pressure",
                    }[mechanic]
                    use_count += int(row[count_key])
        shift = float(objective["route_context_action_rate_shift"][mechanic]["absolute_mean_shift"])
        learned_categories = {
            objective["behavior_associations"][mechanic][policy]["diagnostic_category"]
            for policy in ("pi1_cp3", "pi2_cp4")
        }
        if use_count < 20:
            classification = "INSUFFICIENT_EVIDENCE"
        elif shift >= 0.02 or any(
            category.startswith(("B_", "C_")) for category in learned_categories
        ):
            classification = "ROUTE_EVIDENCE_CHANGES_THE_QUESTION"
        else:
            classification = "ROUTE_EVIDENCE_STILL_JUSTIFIES_PERTURBATION"
        result[mechanic] = {
            "classification": classification,
            "learned_fresh_plus_long_use_count": use_count,
            "absolute_mean_long_vs_fresh_action_rate_shift": shift,
            "learned_objective_diagnostic_categories": sorted(learned_categories),
        }
    primary = [result[name]["classification"] for name in ("mark", "grab", "guard")]
    if "ROUTE_EVIDENCE_CHANGES_THE_QUESTION" in primary:
        overall = "MODIFY_FIRST"
    elif all(value == "ROUTE_EVIDENCE_STILL_JUSTIFIES_PERTURBATION" for value in primary):
        overall = "YES"
    else:
        overall = "MODIFY_FIRST"
    return {
        "mechanics": result,
        "should_paused_balance_experiment_become_cp7": overall,
        "note": "No perturbation was executed in CP6.",
    }


def _route_level_policy_verdict(route_comparison: Mapping[str, Any]) -> str:
    overall = route_comparison["overall"][PRIMARY_HERO_POLICY]["comparisons"]
    confirmations = []
    for key in ("pi0_heuristic_to_pi1_cp3", "pi0_heuristic_to_pi2_cp4"):
        row = overall[key]
        low, _high = row["paired_failure_difference_ci_95"]
        confirmations.append(float(row["route_failure_rate_difference"]) > 0 and float(low) > 0)
    if all(confirmations):
        return "YES"
    if any(confirmations):
        return "MIXED"
    directions = [
        float(overall[key]["route_failure_rate_difference"])
        for key in ("pi0_heuristic_to_pi1_cp3", "pi0_heuristic_to_pi2_cp4")
    ]
    if all(value <= 0 for value in directions):
        return "NO"
    return "INCONCLUSIVE"


def _context_value_verdict(objective: Mapping[str, Any], mechanic: str) -> str:
    shift = float(objective["route_context_action_rate_shift"][mechanic]["absolute_mean_shift"])
    if shift >= 0.02:
        return "YES"
    if shift <= 0.005:
        return "NO"
    return "INCONCLUSIVE"


def _final_verdict(payload: Mapping[str, Any]) -> dict[str, str]:
    roles = payload["roles"]
    attrition_roles = sum(row["classification"] == "ATTRITION" for row in roles.values())
    mixed_roles = sum(row["classification"] == "MIXED" for row in roles.values())
    objective = payload["objective"]
    cp7 = payload["cp7"]
    isolated_sufficient = "NO" if attrition_roles or mixed_roles else "PARTIALLY"
    low_lethality_value = "YES" if attrition_roles else "MIXED" if mixed_roles else "NO"
    return {
        "isolated_fresh_party_encounter_evaluation_sufficient": isolated_sufficient,
        "low_lethality_encounters_create_downstream_value": low_lethality_value,
        "pi1_pi2_advantage_persists_at_route_level": _route_level_policy_verdict(
            payload["route_comparison"]
        ),
        "route_context_materially_changes_mark_value": _context_value_verdict(objective, "mark"),
        "route_context_materially_changes_grab_value": _context_value_verdict(objective, "grab"),
        "route_context_materially_changes_guard_value": _context_value_verdict(objective, "guard"),
        "route_context_materially_changes_effort_drain_value": _context_value_verdict(
            objective, "effort_drain"
        ),
        "single_combat_training_misaligned_with_expedition_objective": str(
            objective["single_combat_objective_alignment"]
        ),
        "redesign_low_standalone_win_encounters_merely_for_low_win_rate": "NO",
        "paused_balance_perturbation_should_proceed_as_cp7": str(
            cp7["should_paused_balance_experiment_become_cp7"]
        ),
        "route_evaluator_sufficient_future_maze_foundation": "YES",
    }


def _write_report(payload: Mapping[str, Any]) -> None:
    route_comparison = payload["route_comparison"]
    fresh = payload["fresh_conditioned"]
    roles = payload["roles"]
    objective = payload["objective"]
    cp7 = payload["cp7"]
    verdict = _final_verdict(payload)
    company = route_comparison["aggregates"][PRIMARY_HERO_POLICY]
    lines = [
        "# The Charter â€” Checkpoint 6 Stateful Multi-Encounter Route Evaluation",
        "",
        "## A. Provenance / clean baseline",
        f"- CP5 immutable baseline: `{CP5_COMMIT}`.",
        f"- Experiment: `{EXPERIMENT_ID}`.",
        f"- CP6 research seeds: {RESEARCH_SEEDS[0]}â€“{RESEARCH_SEEDS[-1]}.",
        "- The superseded balance-perturbation CP6 was removed before this implementation; no production balance change survived.",
        "",
        "## B. Actual persistence semantics",
        "- The research harness delegates transitions to production `create_encounter_combat` and `sync_company_from_combat(..., party_formation)`.",
        "- HP, Effort, Downed state, mortal wounds, morale, strain/marks, quirks, gear, persistent tags, and interactive formation carry between fights.",
        "- Dead heroes leave the roster; Mark is explicitly cleared between combats; combat-start traits are reapplied at every encounter.",
        "- Recovery is a town service, not an automatic between-room reset.",
        "",
        "## C. Route evaluation architecture",
        "- A route owns only an ordered encounter sequence, metadata, and deterministic root-seed authority.",
        "- Each encounter receives a policy-independent SHA-256-derived seed; enemy policy is the manipulated variable.",
        "- Fresh-versus-conditioned target panels share an encounter-identity seed namespace so the target fight is RNG-matched across contexts.",
        "",
        "## D. Frozen route suite",
    ]
    for route in PRIMARY_ROUTES:
        lines.append(
            f"- `{route.route_id}` ({route.structure}): {' â†’ '.join(route.encounter_ids)}"
        )
    lines.extend(
        [
            "",
            "Fresh/conditioned target panels: Hunt, Hunt Guarded, Cave Mini Boss, Pattern Cell, and Maze Depth 1.",
            "",
            "## E. Determinism qualification",
            "- Focused persistence tests qualified production-state carryover before route outcome access.",
            "- The consumed CP4 seed block 20001â€“20008 was reproduced before freeze, including exact 3/88, 24/88, and 22/88 company_survival enemy-win counts for Ï€0/Ï€1/Ï€2.",
            "- Same root seed + route + policy deterministically reproduces the complete route digest.",
            "",
            "## F. Stateful route results",
            "### Primary company_survival routes",
            "| Route | Ï€0 fail | Ï€1 fail | Ï€2 fail |",
            "|---|---:|---:|---:|",
        ]
    )
    for route in PRIMARY_ROUTES:
        row = company[route.route_id]
        lines.append(
            f"| {route.route_id} | {row['pi0_heuristic']['route_failure_rate']:.1%} | "
            f"{row['pi1_cp3']['route_failure_rate']:.1%} | {row['pi2_cp4']['route_failure_rate']:.1%} |"
        )
    overall = route_comparison["overall"][PRIMARY_HERO_POLICY]
    lines.extend(
        [
            "",
            "## G. Ï€0/Ï€1/Ï€2 route comparison",
            f"- Ï€1âˆ’Ï€0 overall route-failure difference: {overall['comparisons']['pi0_heuristic_to_pi1_cp3']['route_failure_rate_difference']:+.1%}; 95% paired CI {overall['comparisons']['pi0_heuristic_to_pi1_cp3']['paired_failure_difference_ci_95'][0]:+.1%} to {overall['comparisons']['pi0_heuristic_to_pi1_cp3']['paired_failure_difference_ci_95'][1]:+.1%}.",
            f"- Ï€2âˆ’Ï€0 overall route-failure difference: {overall['comparisons']['pi0_heuristic_to_pi2_cp4']['route_failure_rate_difference']:+.1%}; 95% paired CI {overall['comparisons']['pi0_heuristic_to_pi2_cp4']['paired_failure_difference_ci_95'][0]:+.1%} to {overall['comparisons']['pi0_heuristic_to_pi2_cp4']['paired_failure_difference_ci_95'][1]:+.1%}.",
            f"- Ï€2âˆ’Ï€1 overall route-failure difference: {overall['comparisons']['pi1_cp3_to_pi2_cp4']['route_failure_rate_difference']:+.1%}.",
            "",
            "## H. Fresh-versus-conditioned encounter results",
            "| Target | Policy | Fresh win | One predecessor | Long prefix |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for target in FRESH_CONDITIONED_CONTEXTS:
        for policy in POLICY_NAMES:
            context = fresh[target]["contexts"]
            lines.append(
                f"| {target} | {policy} | "
                f"{context['fresh'][policy]['target_enemy_win_rate_if_reached']:.1%} | "
                f"{context['one_predecessor'][policy]['target_enemy_win_rate_if_reached']:.1%} | "
                f"{context['long_prefix'][policy]['target_enemy_win_rate_if_reached']:.1%} |"
            )
    lines.extend(
        [
            "",
            "## I. Encounter-level attrition contribution",
            "- See `encounter_attrition.json` for per-encounter HP, Effort, wound, Downed, and route-position deterioration distributions.",
            f"- Meaningful early-attrition cells identified: {len(payload['downstream']['meaningful_early_attrition_cells'])}.",
            "",
            "## J. Downstream value",
            "- Early encounters are evaluated by the state they hand to the next fight, not by standalone wipe rate alone.",
            "- Paired next-entry HP/Effort differences are preserved in `downstream_value.json`.",
            "",
            "## K. Behavior revaluation",
            "- Mark, Grab, Guard, Bite/direct pressure, Effort drain, movement, waiting, and targeting were remeasured under fresh and conditioned states without retraining.",
        ]
    )
    for mechanic, shift in objective["route_context_action_rate_shift"].items():
        lines.append(
            f"- {mechanic}: mean absolute learned-policy long-prefix vs fresh action-rate shift "
            f"{float(shift['absolute_mean_shift']):.3f}."
        )
    lines.extend(
        [
            "",
            "## L. Objective mismatch",
            f"- Single-combat objective alignment classification: **{objective['single_combat_objective_alignment']}**.",
            "- Behavior associations are diagnostic/observational only; no policy was retrained on route outcomes.",
            "",
            "## M. Encounter role classification",
        ]
    )
    for encounter, row in roles.items():
        lines.append(f"- `{encounter}`: **{row['classification']}**")
    lines.extend(
        [
            "",
            "## N. Route-level strategic diversity",
            "- `behavior_revaluation.json` reports action rates, distinct skills, skill entropy, movement/waiting, and target switching by route position and conditioning context.",
            "- Higher entropy is not treated as inherently better; the diagnostic question is whether action choice changes with meaningful state context.",
            "",
            "## O. Player-facing interpretation",
            "- Route-level balance is the more faithful model for Charter's attritional pacing: easy fights can still create later risk without each encounter becoming independently lethal.",
            "- This supports meaningful greed/retreat/resource-preservation decisions while avoiding a design target where every room has a 50% wipe chance.",
            "",
            "## P. Implications for balance methodology",
            "- Do not redesign an encounter merely because its fresh-party wipe rate is low.",
            "- Separate immediate encounter lethality from persistent route contribution before any Mark/Grab/Guard balance perturbation.",
            "",
            "## Q. Implications for future Maze intelligence",
            "- The qualified route evaluator supplies the persistent-state environment a future adversarial Maze director would require, but CP6 implements no Maze AI.",
            "",
            "## R. Whether the old CP6 should become CP7",
        ]
    )
    for mechanic, row in cp7["mechanics"].items():
        lines.append(f"- {mechanic}: **{row['classification']}**")
    lines.extend(
        [
            f"- Overall paused balance-perturbation recommendation: **{cp7['should_paused_balance_experiment_become_cp7']}**.",
            "",
            "## S. Limitations",
            "- Frozen Ï€1/Ï€2 were trained on isolated-combat continuation outcomes, not route outcomes.",
            "- Existing hero policies approximate player behavior and are not player optimizers.",
            "- Conditioned-target win rates are reported only among routes that actually reach the target; predecessor failures are separately reported.",
            "- Route-role thresholds are predeclared diagnostics rather than permanent game-design labels.",
            "",
            "## T. Quality gates",
            "- Final repository quality gates are recorded after report generation in the final handoff.",
            "",
            "## U. Artifact/checksum manifest",
            f"- Artifact root: `{ARTIFACT_ROOT}`; generated artifacts remain ignored.",
            "",
            "## V. Final verdict",
        ]
    )
    for key, value in verdict.items():
        lines.append(f"- {key}: **{value}**")
    lines.extend(
        [
            "",
            "Checkpoint 6 stops here. No balance change, Ï€3 training, CP7 implementation, Maze AI, hunting parties, reoccupation, alternate exits, or retreat change was performed.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest() -> dict[str, Any]:
    artifact_names = (
        "frozen_protocol.json",
        "provenance.json",
        "persistence_semantics.json",
        "route_suite.json",
        "baseline_reproduction.json",
        "route_results.json",
        "route_policy_comparison.json",
        "encounter_attrition.json",
        "fresh_vs_conditioned.json",
        "downstream_value.json",
        "behavior_revaluation.json",
        "objective_mismatch_audit.json",
        "encounter_role_classification.json",
        "maze_relevance.json",
        "cp7_recommendation.json",
        "REPORT.md",
    )
    provenance = _read_json(PROVENANCE_PATH)
    return {
        "experiment_id": EXPERIMENT_ID,
        "frozen_protocol_checksum": provenance["frozen_protocol_checksum"],
        "evaluation_accessed": True,
        "post_evaluation_tuning_performed": False,
        "artifacts": {name: sha256_file(ARTIFACT_ROOT / name) for name in artifact_names},
        "pi1_model_checksum": PI1_MODEL_SHA,
        "pi2_model_checksum": PI2_MODEL_SHA,
    }


def _validate_frozen_pre_access_state() -> tuple[dict[str, Any], dict[str, Any]]:
    if not PROTOCOL_PATH.exists() or not PROVENANCE_PATH.exists():
        raise ValueError("CP6 must be frozen before route outcome evaluation.")
    protocol = _read_json(PROTOCOL_PATH)
    provenance = _read_json(PROVENANCE_PATH)
    if protocol.get("evaluation_accessed") is not False:
        raise ValueError("Frozen CP6 protocol does not prove sealed route-evaluation state.")
    if protocol.get("post_evaluation_tuning_performed") is not False:
        raise ValueError("Frozen CP6 protocol reports post-evaluation tuning.")
    if provenance.get("pre_access_tests_passed") is not True:
        raise ValueError("CP6 pre-access verification marker is invalid.")
    if provenance.get("evaluation_accessed") is not False:
        raise ValueError("CP6 provenance does not prove sealed route-evaluation state.")
    if sha256_file(PROTOCOL_PATH) != provenance["frozen_protocol_checksum"]:
        raise ValueError("Frozen CP6 protocol checksum changed before evaluation.")
    if _git_head() != CP5_COMMIT:
        raise ValueError("Repository HEAD moved after CP6 freeze.")
    if provenance["implementation_hashes"] != _hash_paths(
        tuple(provenance["implementation_hashes"])
    ):
        raise ValueError("CP6 implementation changed after freeze.")
    cp5_provenance = _read_json(CP5_ROOT / "provenance.json")
    simulator = _hash_paths(tuple(cp5_provenance["simulator_hashes"]))
    content = _hash_paths(tuple(cp5_provenance["content_hashes"]))
    if simulator != provenance["simulator_hashes"]:
        raise ValueError("Production simulator changed after CP6 freeze.")
    if content != provenance["content_hashes"]:
        raise ValueError("Production content changed after CP6 freeze.")
    _load_frozen_policies()
    return protocol, provenance


def _reverify_manifest(manifest: Mapping[str, Any]) -> None:
    if sha256_file(PROTOCOL_PATH) != manifest["frozen_protocol_checksum"]:
        raise ValueError("Frozen CP6 protocol changed after evaluation.")
    for name, expected in manifest["artifacts"].items():
        path = ARTIFACT_ROOT / str(name)
        if sha256_file(path) != expected:
            raise ValueError(f"CP6 manifest checksum mismatch: {name}")
    provenance = _read_json(PROVENANCE_PATH)
    if provenance["implementation_hashes"] != _hash_paths(
        tuple(provenance["implementation_hashes"])
    ):
        raise ValueError("CP6 implementation changed after evaluation.")
    cp5_provenance = _read_json(CP5_ROOT / "provenance.json")
    simulator = _hash_paths(tuple(cp5_provenance["simulator_hashes"]))
    content = _hash_paths(tuple(cp5_provenance["content_hashes"]))
    if simulator != provenance["simulator_hashes"]:
        raise ValueError("Production simulator changed after CP6 evaluation.")
    if content != provenance["content_hashes"]:
        raise ValueError("Production content changed after CP6 evaluation.")
    _load_frozen_policies()


def _assert_only_cp6_working_tree_changes() -> None:
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    allowed = {
        "src/game/dev/ml_cp6.py",
        "src/game/ml/stateful_route.py",
        "tests/test_ml_stateful_route.py",
    }
    unexpected: list[str] = []
    for raw in status.splitlines():
        if not raw.strip():
            continue
        path = raw[3:].strip().replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path not in allowed:
            unexpected.append(raw)
    if unexpected:
        raise ValueError(
            "Unexpected non-CP6 working-tree changes before freeze: " + "; ".join(unexpected)
        )


def _hash_paths(paths: Sequence[str]) -> dict[str, str]:
    return {str(path): sha256_file(Path(path)) for path in paths}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_dirty() -> bool:
    return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
