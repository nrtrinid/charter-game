# ruff: noqa: E501
"""Checkpoint 8 route-aware frozen-policy causal mechanic perturbation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from game.combat.combat_state import DEFAULT_MARKED_TURNS
from game.combat.damage import GUARDED_DAMAGE_REDUCTION
from game.data.loaders import load_game_definitions
from game.dev.ml_cp5 import PI1_MODEL_SHA, PI2_MODEL_SHA, _load_frozen_policies
from game.dev.ml_cp6 import HERO_POLICIES, POLICY_NAMES, PRIMARY_HERO_POLICY, PRIMARY_ROUTES
from game.dev.ml_cp7 import SAFETY_BUDGET
from game.ml.artifacts import sha256_file
from game.ml.balance_analysis import (
    EQUIVALENCE_MARGIN,
    adoption_change,
    causal_classification,
    classify_effect,
    dominance_audit,
    encounter_horizon_analysis,
    horizon_alignment,
    mechanic_usage,
    opportunity_cost,
    paired_local_effect,
    paired_route_effect,
    policy_ordering,
    state_niches,
    strategy_space,
)
from game.ml.balance_perturbation import (
    CONDITION_IDS,
    MECHANIC_CONDITIONS,
    PERTURBATIONS,
    perturbation_matrix_payload,
    run_perturbed_route,
)
from game.ml.confirmation_audit import wilson_interval
from game.ml.outcome_features import CP3_FEATURE_NAMES, CP3_FEATURE_SCHEMA
from game.ml.route_termination import run_corrected_stateful_route

EXPERIMENT_ID = "checkpoint8-route-aware-balance-perturbation-v1"
CP7_COMMIT = "95d3a846a1ed0230789d9f284170a87b17395000"
CP7_ROOT = Path("artifacts/ml/checkpoint7-production-faithful-route-termination-v1")
ARTIFACT_ROOT = Path("artifacts/ml") / EXPERIMENT_ID
WORKER_ROOT = ARTIFACT_ROOT / "workers"
PROTOCOL_PATH = ARTIFACT_ROOT / "frozen_protocol.json"
PROVENANCE_PATH = ARTIFACT_ROOT / "provenance.json"
BASELINE_MECHANICS_PATH = ARTIFACT_ROOT / "baseline_mechanics.json"
MATRIX_PATH = ARTIFACT_ROOT / "perturbation_matrix.json"
OBSERVABILITY_PATH = ARTIFACT_ROOT / "observability_audit.json"
BASELINE_EQUIVALENCE_PATH = ARTIFACT_ROOT / "baseline_equivalence.json"
ROUTE_RESULTS_PATH = ARTIFACT_ROOT / "route_results.json"
PAIRED_ROUTE_PATH = ARTIFACT_ROOT / "paired_route_effects.json"
LOCAL_EFFECTS_PATH = ARTIFACT_ROOT / "local_combat_effects.json"
HORIZON_PATH = ARTIFACT_ROOT / "horizon_alignment.json"
MARK_PATH = ARTIFACT_ROOT / "mark_audit.json"
GRAB_PATH = ARTIFACT_ROOT / "grab_audit.json"
GUARD_PATH = ARTIFACT_ROOT / "guard_audit.json"
DRAIN_PATH = ARTIFACT_ROOT / "effort_drain_audit.json"
OPPORTUNITY_PATH = ARTIFACT_ROOT / "opportunity_cost.json"
ROUTE_POSITION_PATH = ARTIFACT_ROOT / "route_position_analysis.json"
ENCOUNTER_HORIZON_PATH = ARTIFACT_ROOT / "encounter_horizon_analysis.json"
NICHE_PATH = ARTIFACT_ROOT / "state_niche_analysis.json"
POLICY_RESPONSE_PATH = ARTIFACT_ROOT / "policy_response_comparison.json"
STRATEGY_PATH = ARTIFACT_ROOT / "strategy_space_analysis.json"
DOMINANCE_PATH = ARTIFACT_ROOT / "dominance_audit.json"
CAUSAL_PATH = ARTIFACT_ROOT / "causal_classification.json"
DESIGN_PATH = ARTIFACT_ROOT / "design_implications.json"
MAZE_PATH = ARTIFACT_ROOT / "maze_implications.json"
NEXT_PATH = ARTIFACT_ROOT / "next_research_recommendation.json"
MANIFEST_PATH = ARTIFACT_ROOT / "manifest.json"
REPORT_PATH = ARTIFACT_ROOT / "REPORT.md"

CP8_SEEDS = tuple(range(50001, 50065))
PRIMARY_POLICY = "pi2_cp4"
MAX_PARALLEL_WORKERS = 4

IMPLEMENTATION_PATHS = (
    "src/game/dev/ml_cp8.py",
    "src/game/ml/balance_perturbation.py",
    "src/game/ml/balance_analysis.py",
    "tests/test_ml_balance_perturbation.py",
    "src/game/dev/ml_cp7.py",
    "src/game/ml/route_termination.py",
    "src/game/dev/ml_cp6.py",
    "src/game/ml/stateful_route.py",
    "src/game/dev/ml_cp5.py",
    "src/game/ml/outcome_features.py",
    "src/game/ml/outcome_ranking.py",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _hash_paths(paths: Sequence[str]) -> dict[str, str]:
    return {path: sha256_file(Path(path)) for path in paths}


def _current_production_hashes() -> tuple[dict[str, str], dict[str, str]]:
    cp7 = _read_json(CP7_ROOT / "provenance.json")
    simulator = _hash_paths(tuple(cp7["simulator_hashes"]))
    content = _hash_paths(tuple(cp7["content_hashes"]))
    if simulator != cp7["simulator_hashes"]:
        raise ValueError("Production simulator changed since CP7.")
    if content != cp7["content_hashes"]:
        raise ValueError("Production content changed since CP7.")
    return simulator, content


def _enemy_encounters(definitions: Any, enemy_id: str) -> list[str]:
    result: list[str] = []
    for encounter_id, encounter in definitions.encounters.items():
        raw = getattr(encounter, "enemy_ids", None) or getattr(encounter, "enemies", None) or ()
        ids = [str(getattr(item, "enemy_id", item)) for item in raw]
        if enemy_id in ids:
            result.append(str(encounter_id))
    return sorted(result)


def _skill_rows(definitions: Any, tag: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for skill in definitions.skills.values():
        if "enemy" not in skill.tags or tag not in skill.tags:
            continue
        enemy_ids = [enemy.id for enemy in definitions.enemies.values() if skill.id in enemy.skills]
        rows.append(
            {
                "skill_id": skill.id,
                "name": skill.name,
                "effort_cost": skill.effort_cost,
                "accuracy": skill.accuracy,
                "damage_min": skill.damage if skill.damage_min is None else skill.damage_min,
                "damage_max": skill.damage if skill.damage_max is None else skill.damage_max,
                "tags": sorted(skill.tags),
                "enemy_classes": sorted(enemy_ids),
                "encounters": sorted(
                    {
                        encounter
                        for enemy_id in enemy_ids
                        for encounter in _enemy_encounters(definitions, enemy_id)
                    }
                ),
            }
        )
    return sorted(rows, key=lambda row: row["skill_id"])


def _baseline_mechanics(definitions: Any) -> dict[str, Any]:
    drag_rows = _skill_rows(definitions, "drag_forward")
    mark_rows = _skill_rows(definitions, "mark")
    guard_rows = _skill_rows(definitions, "guard")
    drain_rows = _skill_rows(definitions, "effort_drain")
    direct = []
    for skill in definitions.skills.values():
        if "enemy" not in skill.tags:
            continue
        high = skill.damage if skill.damage_max is None else skill.damage_max
        if high <= 0 or any(
            tag in skill.tags for tag in ("mark", "guard", "drag_forward", "effort_drain")
        ):
            continue
        enemy_ids = [enemy.id for enemy in definitions.enemies.values() if skill.id in enemy.skills]
        direct.append(
            {
                "skill_id": skill.id,
                "effort_cost": skill.effort_cost,
                "damage_min": skill.damage if skill.damage_min is None else skill.damage_min,
                "damage_max": high,
                "tags": sorted(skill.tags),
                "enemy_classes": sorted(enemy_ids),
            }
        )
    return {
        "MARK": {
            "enemy_mark_skills": mark_rows,
            "drag_also_applies_mark": [row["skill_id"] for row in drag_rows],
            "duration_target_turns": DEFAULT_MARKED_TURNS,
            "removal": "decrements at marked target turn; clears when remaining turns reaches zero",
            "between_combats": "MARKED is explicitly removed by production CompanyState persistence",
            "payoffs": {
                "basic_attack_bonus": 1,
                "vulnerable_bonus_against_marked_or_wounded": 2,
                "personal_opportunist_marked_bonus": 1,
                "predator_marked_bonus": 1,
                "predator_unmarked_penalty": -1,
            },
            "opportunity_cost": "enemy Mark skills generally spend one action and 1 Effort",
        },
        "GRAB": {
            "enemy_drag_skills": drag_rows,
            "movement": "living backline target swaps with corresponding front-slot occupant",
            "mark_application": f"successful living target is Marked for {DEFAULT_MARKED_TURNS} target turns",
            "same_turn": "1-2 baseline base damage plus formation swap and Mark",
            "later_turn": "changed exposure/access and Mark payoff can affect subsequent actions",
        },
        "GUARD": {
            "enemy_guard_skills": guard_rows,
            "damage_reduction": GUARDED_DAMAGE_REDUCTION,
            "consumption": "first positive-damage hit removes Guard and prevents up to the reduction amount",
            "persistence": "no round expiry; survives until consumed or encounter ends",
            "stacking": False,
        },
        "EFFORT_DRAIN": {
            "enemy_drain_skills": drain_rows,
            "drain_amount": 1,
            "prerequisite": "successful hit, living target, target Effort > 0",
            "secondary_effect": "actor heals up to 1 HP after stealing Effort",
            "between_combats": "hero Effort persists through the production CompanyState route transition",
        },
        "DIRECT_PRESSURE": {
            "comparator_skills": sorted(direct, key=lambda row: row["skill_id"]),
            "perturbed_in_primary_experiment": False,
        },
    }


def _observability_audit() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for condition_id, spec in PERTURBATIONS.items():
        rows[condition_id] = {
            "mechanic": spec.mechanic,
            "hypothesis": spec.hypothesis,
            "classification": spec.observability,
            "frozen_feature_changes": list(spec.observable_feature_changes),
            "projected_damage_represented": condition_id in {"M1", "G1"},
            "effort_cost_represented": condition_id in {"M2", "D2", "E1"},
            "status_duration_value_represented_prospectively": False,
            "formation_consequences_represented": "current slots/protection are visible, future drag consequences are not directly encoded",
            "future_cross_encounter_effects_represented": False,
            "interpretation_rule": (
                "lack of frozen-policy adoption is interpretable"
                if spec.observability == "FULLY_OBSERVABLE_TO_FROZEN_POLICY"
                else "lack of frozen-policy adoption is NOT evidence the mechanic is weak"
            ),
        }
    return {
        "schema_id": CP3_FEATURE_SCHEMA.schema_id,
        "schema_version": CP3_FEATURE_SCHEMA.version,
        "feature_count": len(CP3_FEATURE_NAMES),
        "conditions": rows,
    }


def _equivalence_case_equal(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    if expected["route_terminal_status"] != actual["route_terminal_status"]:
        return False
    if expected["final_company_state"] != actual["final_company_state"]:
        return False
    if len(expected["encounters"]) != len(actual["encounters"]):
        return False
    for first, second in zip(expected["encounters"], actual["encounters"], strict=True):
        for key in (
            "termination_reason",
            "final_state_digest",
            "final_rng_digest",
            "enemy_event_digest",
            "enemy_action_sequence",
            "hero_action_sequence",
        ):
            if first[key] != second[key]:
                return False
    return True


def _baseline_equivalence(definitions: Any, policies: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for route in PRIMARY_ROUTES:
        for hero_policy_id in HERO_POLICIES:
            for policy_name in POLICY_NAMES:
                expected = run_corrected_stateful_route(
                    definitions,
                    route,
                    root_seed=40001,
                    enemy_policy_name=policy_name,
                    enemy_policy=policies[policy_name],
                    hero_policy_id=hero_policy_id,
                    safety_budget=SAFETY_BUDGET,
                )
                actual = run_perturbed_route(
                    definitions,
                    route,
                    root_seed=40001,
                    enemy_policy_name=policy_name,
                    enemy_policy=policies[policy_name],
                    hero_policy_id=hero_policy_id,
                    spec=PERTURBATIONS["BASELINE"],
                    safety_budget=SAFETY_BUDGET,
                )
                rows.append(
                    {
                        "route_id": route.route_id,
                        "hero_policy_id": hero_policy_id,
                        "enemy_policy": policy_name,
                        "seed": 40001,
                        "equal": _equivalence_case_equal(expected, actual),
                    }
                )
    if not all(row["equal"] for row in rows):
        raise ValueError("CP8 research baseline failed CP7 production-equivalence qualification.")
    return {
        "consumed_cp7_seed_only": 40001,
        "case_count": len(rows),
        "all_equal": True,
        "rows": rows,
    }


def _protocol_payload(policy_identities: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "immutable_cp7_commit": CP7_COMMIT,
        "primary_policy": PRIMARY_POLICY,
        "policies": policy_identities,
        "conditions": perturbation_matrix_payload()["conditions"],
        "seed_range": [CP8_SEEDS[0], CP8_SEEDS[-1]],
        "seed_count": len(CP8_SEEDS),
        "seed_authority": "new disjoint exploratory CP8 block; CP3 used 1-256/10001-10064/20001-20128, CP5 used 30001-30128, CP6/CP7 used 40001-40064",
        "routes": [
            {
                "route_id": route.route_id,
                "encounters": list(route.encounter_ids),
                "structure": route.structure,
            }
            for route in PRIMARY_ROUTES
        ],
        "route_subset_rationale": "exact CP7 six-route primary structural suite; retains fresh control, light-to-hard, attrition-to-execution, multiple-light, authored sequence, and longer prefixes while omitting the separate CP7 fresh-conditioned diagnostic panel for bounded runtime",
        "hero_policies": list(HERO_POLICIES),
        "primary_hero_policy": PRIMARY_HERO_POLICY,
        "route_endpoint": "production-valid ROUTE_FAILED_ENEMY_VICTORY under CP7 termination/censoring semantics",
        "local_endpoint": "enemy victory on matched reached mechanic-relevant encounters plus paired HP/Effort/Down/wound/round effects",
        "horizon_effect_rule": {
            "POSITIVE": "paired enemy-win difference >0 and lower 95% CI >0",
            "NEGATIVE": "paired enemy-win difference <0 and upper 95% CI <0",
            "NEUTRAL": f"entire paired 95% CI inside +/-{EQUIVALENCE_MARGIN:.2f}",
            "INCONCLUSIVE": "otherwise",
        },
        "adoption_materiality": {
            "absolute_conditional_selection_rate_gain": 0.10,
            "relative_conditional_selection_rate_gain": 0.20,
        },
        "route_position": {
            "SINGLE": "one-encounter route",
            "EARLY": "first encounter of multi-encounter route",
            "LATE": "last encounter of multi-encounter route",
            "MIDDLE": "all other encounters",
        },
        "duration_bands_from_baseline": {"SHORT": "<=5 rounds", "MEDIUM": "6-15", "LONG": ">=16"},
        "state_bands": {"LOW": "<=0.50 fraction", "MID": "(0.50,0.75]", "HIGH": ">0.75"},
        "dominance_rule": "POTENTIAL_STRATEGY_COLLAPSE if selection>=90% when legal overall with >=20 opportunities and >=80% in at least three encounter|route-position strata having >=10 opportunities",
        "causal_classification_precedence": [
            "LOCAL_ROUTE_TRADEOFF if any variant has opposite significant local/route signs",
            "CURRENT_INCENTIVE_LIKELY_WEAK if a fully observable variant materially increases pi2 adoption and route enemy victory",
            "CONTEXTUAL_ROUTE_VALUE if local neutral but route positive",
            "CONTEXTUAL_LOCAL_VALUE if local positive but route neutral",
            "HEURISTIC_OVERVALUATION if pi0 baseline use >=2x pi2 and no fully observable variant is route-positive",
            "FROZEN_POLICY_OBSERVABILITY_LIMITATION if only nonobservable variants create route-positive evidence",
            "MIXED_INCONCLUSIVE otherwise",
        ],
        "production_balance_change_rule": "YES only if at least one mechanic is CURRENT_INCENTIVE_LIKELY_WEAK, its supporting pi2 route effect is positive by frozen CI rule, and that condition does not trigger strategy collapse; otherwise NO",
        "next_direction_rule": [
            "PRODUCTION_BALANCE_EXPERIMENT if production balance change rule is YES",
            "ROUTE_AWARE_POLICY_LEARNING if any local-route mismatch exists or route-level policy ordering changes materially across conditions",
            "ADVERSARIAL_MAZE_DIRECTOR if route-position/context effects are material without prior rules",
            "ANOTHER_CAUSAL_MECHANIC_EXPERIMENT otherwise",
        ],
        "evaluation_accessed": False,
        "training_performed": False,
        "post_evaluation_tuning_performed": False,
    }


def freeze_protocol(*, pre_access_tests_passed: bool) -> None:
    if not pre_access_tests_passed:
        raise ValueError("CP8 freeze requires explicit pre-access test evidence.")
    if _git_head() != CP7_COMMIT:
        raise ValueError(f"CP8 must begin from immutable CP7 commit {CP7_COMMIT}.")
    if ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.iterdir()):
        raise ValueError("CP8 artifact root is not pristine.")
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).splitlines()
    allowed = {
        "src/game/dev/ml_cp8.py",
        "src/game/ml/balance_perturbation.py",
        "src/game/ml/balance_analysis.py",
        "tests/test_ml_balance_perturbation.py",
    }
    unexpected = [line for line in status if line[3:].replace("\\", "/") not in allowed]
    if unexpected:
        raise ValueError(f"Unexpected pre-freeze working-tree changes: {unexpected}")
    definitions = load_game_definitions()
    policies, identities = _load_frozen_policies()
    simulator, content = _current_production_hashes()
    baseline_mechanics = _baseline_mechanics(definitions)
    observability = _observability_audit()
    equivalence = _baseline_equivalence(definitions, policies)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=False)
    _write_json(BASELINE_MECHANICS_PATH, baseline_mechanics)
    _write_json(MATRIX_PATH, perturbation_matrix_payload())
    _write_json(OBSERVABILITY_PATH, observability)
    _write_json(BASELINE_EQUIVALENCE_PATH, equivalence)
    protocol = _protocol_payload(identities)
    _write_json(PROTOCOL_PATH, protocol)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "cp7_commit": CP7_COMMIT,
        "current_head": _git_head(),
        "frozen_protocol_checksum": protocol_sha,
        "pi1_model_checksum": PI1_MODEL_SHA,
        "pi2_model_checksum": PI2_MODEL_SHA,
        "cp7_protocol_checksum": sha256_file(CP7_ROOT / "frozen_protocol.json"),
        "cp7_route_termination_checksum": sha256_file(Path("src/game/ml/route_termination.py")),
        "implementation_hashes": _hash_paths(IMPLEMENTATION_PATHS),
        "simulator_hashes": simulator,
        "content_hashes": content,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__},
        "pre_access_tests_passed": True,
        "evaluation_accessed": False,
        "training_performed": False,
        "post_evaluation_tuning_performed": False,
    }
    _write_json(PROVENANCE_PATH, provenance)
    print(f"CP8_FROZEN_PROTOCOL_SHA={protocol_sha}")
    print(f"BASELINE_EQUIVALENCE_CASES={equivalence['case_count']}")
    print("EVALUATION_ACCESSED=false")


def _validate_frozen_state() -> tuple[dict[str, Any], dict[str, Any]]:
    if not PROTOCOL_PATH.exists() or not PROVENANCE_PATH.exists():
        raise ValueError("CP8 must be frozen before worker/evaluation access.")
    protocol = _read_json(PROTOCOL_PATH)
    provenance = _read_json(PROVENANCE_PATH)
    if protocol.get("evaluation_accessed") is not False:
        raise ValueError("Frozen protocol no longer proves sealed pre-access state.")
    if provenance.get("evaluation_accessed") is not False:
        raise ValueError("Frozen provenance no longer proves sealed pre-access state.")
    if sha256_file(PROTOCOL_PATH) != provenance["frozen_protocol_checksum"]:
        raise ValueError("Frozen CP8 protocol changed.")
    if _git_head() != CP7_COMMIT:
        raise ValueError("Repository HEAD moved after CP8 freeze.")
    if provenance["implementation_hashes"] != _hash_paths(
        tuple(provenance["implementation_hashes"])
    ):
        raise ValueError("CP8 implementation changed after freeze.")
    simulator, content = _current_production_hashes()
    if simulator != provenance["simulator_hashes"] or content != provenance["content_hashes"]:
        raise ValueError("Simulator/content changed after CP8 freeze.")
    _load_frozen_policies()
    return protocol, provenance


def _compact_decision(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "action_index",
            "round",
            "state_digest",
            "enemy_id",
            "enemy_class",
            "chosen_skill",
            "chosen_target",
            "chosen_skill_tags",
            "chosen_projected_damage_max",
            "chosen_target_class",
            "chosen_target_hp",
            "chosen_target_effort",
            "chosen_target_protected",
            "chosen_target_slot",
            "actor_hp",
            "actor_max_hp",
            "legal_mark",
            "legal_grab",
            "legal_guard",
            "legal_effort_drain",
            "selected_mark",
            "selected_grab",
            "selected_guard",
            "selected_effort_drain",
            "best_direct_alternative",
            "party_hp_total",
            "party_max_hp_total",
            "party_effort_total",
            "party_max_effort_total",
            "marked_hero_count",
        )
    }


def _compact_resolution(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "sequence",
            "round",
            "actor_id",
            "actor_team",
            "actor_class",
            "skill_id",
            "skill_tags",
            "target_id",
            "target_team",
            "target_class",
            "before",
            "after",
            "damage_dealt",
            "effort_removed",
            "guard_damage_prevented",
            "success",
        )
    }


def _compact_route(row: Mapping[str, Any]) -> dict[str, Any]:
    encounters: list[dict[str, Any]] = []
    for encounter in row["encounters"]:
        encounters.append(
            {
                key: encounter[key]
                for key in (
                    "encounter_index",
                    "route_length",
                    "encounter_id",
                    "encounter_seed",
                    "combat_entry",
                    "termination_reason",
                    "production_terminal",
                    "censored",
                    "company_after_sync",
                    "deterioration",
                    "rounds",
                    "transitions",
                    "enemy_guarded_at_end",
                    "downs_caused",
                    "mortal_wounds_caused",
                )
            }
            | {
                "decision_audit": [
                    _compact_decision(decision) for decision in encounter["decision_audit"]
                ],
                "resolution_audit": [
                    _compact_resolution(resolution) for resolution in encounter["resolution_audit"]
                ],
            }
        )
    return {
        key: row[key]
        for key in (
            "condition_id",
            "mechanic",
            "case_id",
            "route_id",
            "route_structure",
            "route_length",
            "root_seed",
            "enemy_policy",
            "hero_policy_id",
            "encounter_sequence",
            "route_terminal_status",
            "route_completed",
            "route_failed_enemy_victory",
            "route_censored",
            "route_invalid",
            "encounters_completed",
            "stopped_at_encounter_id",
            "initial_state",
            "final_company_state",
            "total_rounds",
            "total_transitions",
        )
    } | {"encounters": encounters}


def run_worker(condition_id: str) -> None:
    _validate_frozen_state()
    if condition_id not in PERTURBATIONS:
        raise ValueError(f"Unknown CP8 condition: {condition_id}")
    output = WORKER_ROOT / f"{condition_id}.jsonl"
    if output.exists():
        raise ValueError(f"Worker output already exists: {output}")
    WORKER_ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    definitions = load_game_definitions()
    policies, _ = _load_frozen_policies()
    spec = PERTURBATIONS[condition_id]
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for hero_policy_id in HERO_POLICIES:
            for route in PRIMARY_ROUTES:
                for seed in CP8_SEEDS:
                    for policy_name in POLICY_NAMES:
                        result = run_perturbed_route(
                            definitions,
                            route,
                            root_seed=seed,
                            enemy_policy_name=policy_name,
                            enemy_policy=policies[policy_name],
                            hero_policy_id=hero_policy_id,
                            spec=spec,
                            safety_budget=SAFETY_BUDGET,
                        )
                        handle.write(
                            json.dumps(
                                _compact_route(result), sort_keys=True, separators=(",", ":")
                            )
                            + "\n"
                        )
                        count += 1
    _write_json(
        WORKER_ROOT / f"{condition_id}.done.json",
        {
            "condition_id": condition_id,
            "route_case_count": count,
            "output_sha256": sha256_file(output),
            "frozen_protocol_checksum": sha256_file(PROTOCOL_PATH),
            "training_performed": False,
        },
    )
    print(f"CP8_WORKER={condition_id}")
    print(f"ROUTE_CASES={count}")


def _worker_rows(condition_id: str) -> list[dict[str, Any]]:
    path = WORKER_ROOT / f"{condition_id}.jsonl"
    done = _read_json(WORKER_ROOT / f"{condition_id}.done.json")
    if sha256_file(path) != done["output_sha256"]:
        raise ValueError(f"Worker checksum mismatch: {condition_id}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run_workers() -> None:
    pending = list(CONDITION_IDS)
    active: dict[subprocess.Popen[str], str] = {}
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    while pending or active:
        while pending and len(active) < MAX_PARALLEL_WORKERS:
            condition_id = pending.pop(0)
            process = subprocess.Popen(
                [sys.executable, "-m", "game.dev.ml_cp8", "worker", "--condition", condition_id],
                text=True,
                env=env,
            )
            active[process] = condition_id
        finished: list[subprocess.Popen[str]] = []
        for process, condition_id in active.items():
            code = process.poll()
            if code is None:
                continue
            if code != 0:
                raise RuntimeError(f"CP8 worker failed: {condition_id} exit={code}")
            finished.append(process)
        for process in finished:
            active.pop(process)
        if active and not finished:
            import time

            time.sleep(1.0)


def _aggregate_routes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row["route_censored"] and not row["route_invalid"]]
    wins = sum(bool(row["route_failed_enemy_victory"]) for row in valid)
    completions = sum(bool(row["route_completed"]) for row in valid)
    return {
        "case_count": len(rows),
        "valid_count": len(valid),
        "censored_count": sum(bool(row["route_censored"]) for row in rows),
        "invalid_count": sum(bool(row["route_invalid"]) for row in rows),
        "enemy_victories": wins,
        "enemy_victory_rate": wins / len(valid) if valid else None,
        "enemy_victory_wilson_95": wilson_interval(wins, len(valid)) if valid else None,
        "route_completions": completions,
        "route_completion_rate": completions / len(valid) if valid else None,
        "mean_final_hp": mean(float(row["final_company_state"]["hp_total"]) for row in valid)
        if valid
        else None,
        "mean_final_effort": mean(
            float(row["final_company_state"]["effort_total"]) for row in valid
        )
        if valid
        else None,
        "mean_final_mortal_wounds": mean(
            float(row["final_company_state"]["mortal_wounds_total"]) for row in valid
        )
        if valid
        else None,
        "mean_total_rounds": mean(float(row["total_rounds"]) for row in valid) if valid else None,
        "failure_encounters": dict(
            Counter(
                str(row["stopped_at_encounter_id"])
                for row in valid
                if row["stopped_at_encounter_id"]
            )
        ),
    }


def _select_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: str | None = None,
    hero_policy: str | None = None,
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if (policy is None or row["enemy_policy"] == policy)
        and (hero_policy is None or row["hero_policy_id"] == hero_policy)
    ]


def _flatten_decisions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for route in rows:
        for encounter in route["encounters"]:
            if int(route["route_length"]) <= 1:
                position = "SINGLE"
            elif int(encounter["encounter_index"]) == 0:
                position = "EARLY"
            elif int(encounter["encounter_index"]) == int(route["route_length"]) - 1:
                position = "LATE"
            else:
                position = "MIDDLE"
            for decision in encounter["decision_audit"]:
                item = dict(decision)
                item.update(
                    route_id=route["route_id"],
                    root_seed=route["root_seed"],
                    enemy_policy=route["enemy_policy"],
                    hero_policy_id=route["hero_policy_id"],
                    encounter_id=encounter["encounter_id"],
                    encounter_index=encounter["encounter_index"],
                    route_position=position,
                    encounter_rounds=encounter["rounds"],
                )
                result.append(item)
    return result


def _flatten_resolutions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for route in rows:
        for encounter in route["encounters"]:
            if int(route["route_length"]) <= 1:
                position = "SINGLE"
            elif int(encounter["encounter_index"]) == 0:
                position = "EARLY"
            elif int(encounter["encounter_index"]) == int(route["route_length"]) - 1:
                position = "LATE"
            else:
                position = "MIDDLE"
            for resolution in encounter["resolution_audit"]:
                item = dict(resolution)
                item.update(
                    route_id=route["route_id"],
                    root_seed=route["root_seed"],
                    enemy_policy=route["enemy_policy"],
                    hero_policy_id=route["hero_policy_id"],
                    encounter_id=encounter["encounter_id"],
                    encounter_index=encounter["encounter_index"],
                    route_position=position,
                )
                result.append(item)
    return result


def _mechanic_selected_context(rows: Sequence[Mapping[str, Any]], mechanic: str) -> dict[str, Any]:
    selected_field = {
        "MARK": "selected_mark",
        "GRAB": "selected_grab",
        "GUARD": "selected_guard",
        "EFFORT_DRAIN": "selected_effort_drain",
    }[mechanic]
    decisions = [row for row in _flatten_decisions(rows) if bool(row[selected_field])]
    hp = [
        float(row["party_hp_total"]) / max(1.0, float(row["party_max_hp_total"]))
        for row in decisions
    ]
    effort = [
        float(row["party_effort_total"]) / max(1.0, float(row["party_max_effort_total"]))
        for row in decisions
    ]
    return {
        "selection_count": len(decisions),
        "mean_round": mean(float(row["round"]) for row in decisions) if decisions else None,
        "mean_incoming_party_hp_fraction": mean(hp) if hp else None,
        "mean_incoming_party_effort_fraction": mean(effort) if effort else None,
        "target_class_counts": dict(
            Counter(str(row["chosen_target_class"]) for row in decisions).most_common()
        ),
        "target_slot_counts": dict(
            Counter(str(row["chosen_target_slot"]) for row in decisions).most_common()
        ),
        "route_position_counts": dict(
            Counter(str(row["route_position"]) for row in decisions).most_common()
        ),
    }


def _resolution_followup(rows: Sequence[Mapping[str, Any]], mechanic: str) -> dict[str, Any]:
    all_rows = _flatten_resolutions(rows)
    by_encounter: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        key = (
            str(row["route_id"]),
            int(row["root_seed"]),
            str(row["enemy_policy"]),
            f"{row['hero_policy_id']}|{row['encounter_index']}",
        )
        by_encounter[key].append(row)
    selected: list[dict[str, Any]] = []
    tag = {
        "MARK": "mark",
        "GRAB": "drag_forward",
        "GUARD": "guard",
        "EFFORT_DRAIN": "effort_drain",
    }[mechanic]
    for item in all_rows:
        if item["actor_team"] != "enemy" or tag not in item["skill_tags"]:
            continue
        selected.append(item)
    if mechanic == "MARK":
        payoffs = 0
        payoff_damage: list[float] = []
        survived = 0
        for item in selected:
            key = (
                str(item["route_id"]),
                int(item["root_seed"]),
                str(item["enemy_policy"]),
                f"{item['hero_policy_id']}|{item['encounter_index']}",
            )
            later = [
                row
                for row in by_encounter[key]
                if int(row["sequence"]) > int(item["sequence"])
                and row["actor_team"] == "enemy"
                and row["target_id"] == item["target_id"]
                and bool(row["before"].get("target_marked"))
            ]
            if later:
                survived += 1
                payoffs += len(later)
                payoff_damage.extend(float(row["damage_dealt"]) for row in later)
        return {
            "applications": len(selected),
            "marked_target_survived_to_enemy_payoff_count": survived,
            "subsequent_marked_target_enemy_hits": payoffs,
            "mean_payoff_hit_damage": mean(payoff_damage) if payoff_damage else None,
        }
    if mechanic == "GRAB":
        moved = sum(row["before"]["target_slot"] != row["after"]["target_slot"] for row in selected)
        protection_changed = sum(
            row["before"]["target_protected"] != row["after"]["target_protected"]
            for row in selected
        )
        followups = 0
        for item in selected:
            key = (
                str(item["route_id"]),
                int(item["root_seed"]),
                str(item["enemy_policy"]),
                f"{item['hero_policy_id']}|{item['encounter_index']}",
            )
            later = [
                row
                for row in by_encounter[key]
                if int(row["sequence"]) > int(item["sequence"])
                and row["actor_team"] == "enemy"
                and row["target_id"] == item["target_id"]
            ]
            followups += int(bool(later))
        return {
            "uses": len(selected),
            "mean_immediate_damage": mean(float(row["damage_dealt"]) for row in selected)
            if selected
            else None,
            "target_relocated_count": moved,
            "protection_state_changed_count": protection_changed,
            "same_target_later_enemy_followup_count": followups,
        }
    if mechanic == "GUARD":
        mitigations = [row for row in all_rows if float(row.get("guard_damage_prevented", 0)) > 0]
        return {
            "uses": len(selected),
            "guard_consumptions": len(mitigations),
            "mean_damage_prevented_when_consumed": mean(
                float(row["guard_damage_prevented"]) for row in mitigations
            )
            if mitigations
            else None,
            "total_damage_prevented": sum(
                float(row["guard_damage_prevented"]) for row in mitigations
            ),
        }
    return {
        "uses": len(selected),
        "total_effort_removed": sum(float(row["effort_removed"]) for row in selected),
        "mean_effort_removed": mean(float(row["effort_removed"]) for row in selected)
        if selected
        else None,
        "target_class_counts": dict(
            Counter(str(row["target_class"]) for row in selected).most_common()
        ),
    }


def _condition_analysis(
    baseline_rows: Sequence[Mapping[str, Any]],
    condition_rows: Sequence[Mapping[str, Any]],
    condition_id: str,
) -> dict[str, Any]:
    spec = PERTURBATIONS[condition_id]
    if spec.mechanic == "BASELINE":
        raise ValueError("Baseline has no perturbation analysis.")
    policies: dict[str, Any] = {}
    for policy in POLICY_NAMES:
        base = _select_rows(baseline_rows, policy=policy, hero_policy=PRIMARY_HERO_POLICY)
        changed = _select_rows(condition_rows, policy=policy, hero_policy=PRIMARY_HERO_POLICY)
        base_usage = mechanic_usage(base, spec.mechanic)
        changed_usage = mechanic_usage(changed, spec.mechanic)
        route_effect = paired_route_effect(base, changed)
        local_effect = paired_local_effect(base, changed, mechanic=spec.mechanic)
        policies[policy] = {
            "route_effect": route_effect,
            "local_effect": local_effect,
            "horizon_alignment": horizon_alignment(local_effect, route_effect),
            "baseline_usage": base_usage,
            "perturbed_usage": changed_usage,
            "adoption_change": adoption_change(base_usage, changed_usage),
            "selected_context": _mechanic_selected_context(changed, spec.mechanic),
            "mechanic_resolution": _resolution_followup(changed, spec.mechanic),
            "opportunity_cost": opportunity_cost(changed, spec.mechanic),
            "strategy_space": strategy_space(base, changed, spec.mechanic),
            "dominance": dominance_audit(changed, spec.mechanic),
            "state_niches": state_niches(changed, spec.mechanic),
            "encounter_horizon": encounter_horizon_analysis(base, changed, spec.mechanic),
        }
    return {
        "condition_id": condition_id,
        "mechanic": spec.mechanic,
        "hypothesis": spec.hypothesis,
        "description": spec.description,
        "observability": spec.observability,
        "policies": policies,
    }


def _route_result_payload(
    rows_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition_id, rows in rows_by_condition.items():
        result[condition_id] = {
            "all": _aggregate_routes(rows),
            "policy_ordering_company_survival": policy_ordering(rows),
            "by_policy_and_hero_policy": {
                policy: {
                    hero: _aggregate_routes(_select_rows(rows, policy=policy, hero_policy=hero))
                    for hero in HERO_POLICIES
                }
                for policy in POLICY_NAMES
            },
            "raw_worker": str(WORKER_ROOT / f"{condition_id}.jsonl"),
            "raw_worker_sha256": sha256_file(WORKER_ROOT / f"{condition_id}.jsonl"),
        }
    return result


def _direct_pressure_explanation(
    condition_analysis: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    evidence: set[str] = set()
    for condition_id, row in condition_analysis.items():
        primary = row["policies"][PRIMARY_POLICY]
        route_sign = classify_effect(primary["route_effect"]["enemy_victory"])
        if route_sign != "POSITIVE":
            continue
        if condition_id in {"M2", "D2", "E1"}:
            evidence.update({"TEMPO", "RESOURCE_ECONOMICS"})
        if condition_id in {"M1", "G1", "D1"}:
            evidence.add("RAW_DAMAGE_VALUE")
        if condition_id == "G2":
            evidence.add("TARGET_ACCESS")
        if primary["horizon_alignment"] in {
            "LOCAL_NEUTRAL_ROUTE_POSITIVE",
            "LOCAL_NEGATIVE_ROUTE_POSITIVE",
        }:
            evidence.add("ROUTE_STATE")
        horizon = primary["encounter_horizon"]
        long_rate = horizon["perturbed"].get("LONG", {}).get("rate")
        short_rate = horizon["perturbed"].get("SHORT", {}).get("rate")
        if (
            long_rate is not None
            and short_rate is not None
            and float(long_rate) > float(short_rate)
        ):
            evidence.add("SHORT_ENCOUNTER_HORIZON")
    classification = (
        next(iter(evidence)) if len(evidence) == 1 else "MIXED" if evidence else "INCONCLUSIVE"
    )
    return {"classification": classification, "evidence_dimensions": sorted(evidence)}


def analyze_results() -> dict[str, Any]:
    rows_by_condition = {condition: _worker_rows(condition) for condition in CONDITION_IDS}
    expected_per_condition = (
        len(PRIMARY_ROUTES) * len(HERO_POLICIES) * len(CP8_SEEDS) * len(POLICY_NAMES)
    )
    if any(len(rows) != expected_per_condition for rows in rows_by_condition.values()):
        raise ValueError("CP8 worker route coverage is incomplete.")
    baseline_rows = rows_by_condition["BASELINE"]
    condition_analysis = {
        condition: _condition_analysis(baseline_rows, rows_by_condition[condition], condition)
        for condition in CONDITION_IDS
        if condition != "BASELINE"
    }
    _write_json(ROUTE_RESULTS_PATH, _route_result_payload(rows_by_condition))
    _write_json(
        PAIRED_ROUTE_PATH,
        {
            condition: {policy: data["policies"][policy]["route_effect"] for policy in POLICY_NAMES}
            for condition, data in condition_analysis.items()
        },
    )
    _write_json(
        LOCAL_EFFECTS_PATH,
        {
            condition: {policy: data["policies"][policy]["local_effect"] for policy in POLICY_NAMES}
            for condition, data in condition_analysis.items()
        },
    )
    _write_json(
        HORIZON_PATH,
        {
            condition: {
                policy: data["policies"][policy]["horizon_alignment"] for policy in POLICY_NAMES
            }
            for condition, data in condition_analysis.items()
        },
    )
    mechanic_artifacts = {
        "MARK": MARK_PATH,
        "GRAB": GRAB_PATH,
        "GUARD": GUARD_PATH,
        "EFFORT_DRAIN": DRAIN_PATH,
    }
    for mechanic, path in mechanic_artifacts.items():
        _write_json(
            path,
            {
                condition: data
                for condition, data in condition_analysis.items()
                if data["mechanic"] == mechanic
            },
        )
    _write_json(
        OPPORTUNITY_PATH,
        {
            condition: {
                policy: data["policies"][policy]["opportunity_cost"] for policy in POLICY_NAMES
            }
            for condition, data in condition_analysis.items()
        },
    )
    _write_json(
        ROUTE_POSITION_PATH,
        {
            condition: {
                policy: data["policies"][policy]["perturbed_usage"]["by_route_position"]
                for policy in POLICY_NAMES
            }
            for condition, data in condition_analysis.items()
        },
    )
    _write_json(
        ENCOUNTER_HORIZON_PATH,
        {
            condition: {
                policy: data["policies"][policy]["encounter_horizon"] for policy in POLICY_NAMES
            }
            for condition, data in condition_analysis.items()
        },
    )
    _write_json(
        NICHE_PATH,
        {
            condition: {policy: data["policies"][policy]["state_niches"] for policy in POLICY_NAMES}
            for condition, data in condition_analysis.items()
        },
    )
    _write_json(
        POLICY_RESPONSE_PATH,
        {
            condition: {
                policy: {
                    "adoption_change": data["policies"][policy]["adoption_change"],
                    "route_effect": data["policies"][policy]["route_effect"]["enemy_victory"],
                    "local_effect": data["policies"][policy]["local_effect"]["enemy_victory"],
                }
                for policy in POLICY_NAMES
            }
            for condition, data in condition_analysis.items()
        },
    )
    _write_json(
        STRATEGY_PATH,
        {
            condition: {
                policy: data["policies"][policy]["strategy_space"] for policy in POLICY_NAMES
            }
            for condition, data in condition_analysis.items()
        },
    )
    _write_json(
        DOMINANCE_PATH,
        {
            condition: {policy: data["policies"][policy]["dominance"] for policy in POLICY_NAMES}
            for condition, data in condition_analysis.items()
        },
    )

    observability = _read_json(OBSERVABILITY_PATH)["conditions"]
    baseline_primary = _select_rows(
        baseline_rows, policy=PRIMARY_POLICY, hero_policy=PRIMARY_HERO_POLICY
    )
    baseline_pi0 = _select_rows(
        baseline_rows, policy="pi0_heuristic", hero_policy=PRIMARY_HERO_POLICY
    )
    classifications: dict[str, str] = {}
    for mechanic, conditions in MECHANIC_CONDITIONS.items():
        summaries = {
            condition: condition_analysis[condition]["policies"][PRIMARY_POLICY]
            for condition in conditions
        }
        classifications[mechanic] = causal_classification(
            mechanic,
            summaries,
            observability={
                condition: observability[condition]["classification"] for condition in conditions
            },
            baseline_pi0_usage=mechanic_usage(baseline_pi0, mechanic),
            baseline_pi2_usage=mechanic_usage(baseline_primary, mechanic),
        )
    _write_json(
        CAUSAL_PATH,
        {
            "classifications": classifications,
            "rules_frozen_before_evaluation": True,
            "condition_primary_evidence": {
                condition: condition_analysis[condition]["policies"][PRIMARY_POLICY]
                for condition in condition_analysis
            },
        },
    )

    any_collapse = any(
        bool(data["policies"][PRIMARY_POLICY]["dominance"]["potential_strategy_collapse"])
        for data in condition_analysis.values()
    )
    setup_understated = any(
        data["policies"][PRIMARY_POLICY]["horizon_alignment"]
        in {"LOCAL_NEUTRAL_ROUTE_POSITIVE", "LOCAL_NEGATIVE_ROUTE_POSITIVE"}
        for data in condition_analysis.values()
    )
    any_tradeoff = any(
        data["policies"][PRIMARY_POLICY]["horizon_alignment"]
        in {"LOCAL_POSITIVE_ROUTE_NEGATIVE", "LOCAL_NEGATIVE_ROUTE_POSITIVE"}
        for data in condition_analysis.values()
    )
    production_change_conditions = [
        condition
        for condition, data in condition_analysis.items()
        if classifications[data["mechanic"]] == "CURRENT_INCENTIVE_LIKELY_WEAK"
        and classify_effect(data["policies"][PRIMARY_POLICY]["route_effect"]["enemy_victory"])
        == "POSITIVE"
        and not data["policies"][PRIMARY_POLICY]["dominance"]["potential_strategy_collapse"]
    ]
    production_change = bool(production_change_conditions)
    direct_pressure = _direct_pressure_explanation(condition_analysis)
    orderings = {
        condition: policy_ordering(rows_by_condition[condition]) for condition in CONDITION_IDS
    }
    pi2_strongest_flags = []
    for ordering in orderings.values():
        pi2_rate = ordering[PRIMARY_POLICY]["win_rate"]
        peers = [ordering[name]["win_rate"] for name in POLICY_NAMES if name != PRIMARY_POLICY]
        pi2_strongest_flags.append(
            pi2_rate is not None and all(peer is not None and pi2_rate >= peer for peer in peers)
        )
    pi2_oracle = (
        "YES" if all(pi2_strongest_flags) else "MIXED" if any(pi2_strongest_flags) else "NO"
    )
    route_learning_case = (
        any_tradeoff
        or setup_understated
        or any(
            ordering != orderings["BASELINE"]
            for key, ordering in orderings.items()
            if key != "BASELINE"
        )
    )
    maze_case = any(
        any(
            bucket.get("opportunities", 0) >= 10
            and bucket.get("rate", 0.0)
            != data["policies"][PRIMARY_POLICY]["perturbed_usage"].get("conditional_selection_rate")
            for bucket in data["policies"][PRIMARY_POLICY]["perturbed_usage"][
                "by_route_position"
            ].values()
        )
        for data in condition_analysis.values()
    )
    if production_change:
        next_direction = "PRODUCTION_BALANCE_EXPERIMENT"
    elif route_learning_case:
        next_direction = "ROUTE_AWARE_POLICY_LEARNING"
    elif maze_case:
        next_direction = "ADVERSARIAL_MAZE_DIRECTOR"
    else:
        next_direction = "ANOTHER_CAUSAL_MECHANIC_EXPERIMENT"

    design = {
        "mechanic_classifications": classifications,
        "direct_pressure_dominance_explanation": direct_pressure,
        "setup_control_value_understated_by_single_combat": "YES" if setup_understated else "NO",
        "local_route_tradeoff_present": "YES" if any_tradeoff else "NO",
        "meaningful_niches_without_universal_dominance": "MIXED" if any_collapse else "YES",
        "potential_strategy_collapse": "YES" if any_collapse else "NO",
        "production_balance_change_justified": "YES" if production_change else "NO",
        "supporting_production_change_conditions": production_change_conditions,
        "pi2_still_strongest_route_oracle": pi2_oracle,
        "route_aware_policy_learning_case_strengthened": "YES" if route_learning_case else "NO",
        "maze_director_case_strengthened": "YES" if maze_case else "INCONCLUSIVE",
    }
    _write_json(DESIGN_PATH, design)
    _write_json(
        MAZE_PATH,
        {
            "strengthened": design["maze_director_case_strengthened"],
            "implications_only_no_implementation": True,
            "examples": [
                "route-positive resource pressure could motivate sequencing attrition before execution encounters",
                "duration-dependent setup value could inform encounter-composition horizon",
                "route-position niches could inform future director sequencing rather than per-fight buffs",
            ],
        },
    )
    next_payload = {
        "recommended_next_direction": next_direction,
        "implemented": False,
        "rule_frozen_before_evaluation": True,
    }
    _write_json(NEXT_PATH, next_payload)
    return {
        "condition_analysis": condition_analysis,
        "classifications": classifications,
        "design": design,
        "next": next_payload,
        "orderings": orderings,
    }


def _fmt_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def _fmt_pp(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:+.2f}pp"


def _write_report(analysis: Mapping[str, Any]) -> None:
    observability = _read_json(OBSERVABILITY_PATH)
    mechanics = _read_json(BASELINE_MECHANICS_PATH)
    lines = [
        "# THE CHARTER — CHECKPOINT 8 ROUTE-AWARE FROZEN-POLICY CAUSAL MECHANIC PERTURBATION",
        "",
        "## A. Provenance / CP7 preservation",
        f"- Immutable CP7 source commit: `{CP7_COMMIT}` (`Qualify production-faithful route termination`).",
        f"- Frozen CP8 protocol SHA-256: `{sha256_file(PROTOCOL_PATH)}`.",
        f"- π1 SHA-256: `{PI1_MODEL_SHA}`; π2 SHA-256: `{PI2_MODEL_SHA}`.",
        "- CP8 is research-only; production gameplay/content remained unchanged.",
        "",
        "## B. Exact baseline mechanics",
        f"- Mark lasts {mechanics['MARK']['duration_target_turns']} target turns; basic attacks gain +1 vs Marked, vulnerable-bonus actions gain +2 vs Marked or wounded, and production persistence clears Mark between combats.",
        "- Grab/drag is 1 Effort and baseline 1–2 base damage, swaps a backliner into the matching front lane when possible, then applies Mark.",
        f"- Guard costs 1 Effort in the relevant enemy kits, prevents up to {mechanics['GUARD']['damage_reduction']} damage on the next positive hit, is consumed by that hit, and has no round expiry.",
        "- Effort Drain costs 1 Effort, deals 1–2 base damage, removes 1 hero Effort on a successful eligible hit, heals the user up to 1 HP, and hero Effort persists between encounters.",
        "",
        "## C. Frozen perturbation protocol",
        "- Conditions: BASELINE, M1 Mark payoff, M2 Mark economics, G1 Grab immediate damage, G2 drag-Mark duration, D1 Guard mitigation, D2 Guard economics, E1 Effort-Drain economics.",
        "- No combined condition, no adaptive sweep, no direct-pressure damage perturbation, and no post-result value changes.",
        "",
        "## D. Policy observability",
    ]
    for condition_id in CONDITION_IDS:
        row = observability["conditions"][condition_id]
        lines.append(
            f"- {condition_id}: **{row['classification']}**; frozen feature changes={row['frozen_feature_changes']}."
        )
    lines += [
        "",
        "## E. Research perturbation architecture",
        "- Every condition executes in its own worker process. Data-driven variants operate on a deep-copied GameDefinitions; hard-coded mechanic variants are temporary process-local overrides restored before worker exit. Production modules do not depend on CP8 research code.",
        "",
        "## F. Baseline equivalence",
    ]
    equivalence = _read_json(BASELINE_EQUIVALENCE_PATH)
    lines.append(
        f"- Baseline research condition exactly reproduced CP7 across {equivalence['case_count']} previously consumed route/policy/hero-policy cases at seed {equivalence['consumed_cp7_seed_only']}: terminal status, persisted company state, final state/RNG, action sequences, and event digest all matched."
    )
    lines += [
        "",
        "## G. CP8 seed authority",
        f"- Fresh exploratory causal block: {CP8_SEEDS[0]}–{CP8_SEEDS[-1]} ({len(CP8_SEEDS)} seeds), shared identically by baseline, all variants, and π0/π1/π2.",
        "- This block is disjoint from CP3–CP7 research/test/confirmation ranges.",
        "",
        "## H. Route suite",
        "- Exact six-route CP7 primary structural suite: light→hard, light/medium→hard, control→hard, multiple-light→hard, fresh Hunt, and authored Shallow Cave→Cave Mini Boss.",
        "- Hero policies: company_survival primary, tactical and anti_mark secondary.",
        "",
        "## I. π2 primary route effects",
    ]
    for condition_id, data in analysis["condition_analysis"].items():
        primary = data["policies"][PRIMARY_POLICY]
        effect = primary["route_effect"]["enemy_victory"]
        ci = effect["paired_difference_ci_95"] or [None, None]
        lines.append(
            f"- {condition_id} ({data['mechanic']}/{data['hypothesis']}): "
            f"{effect['baseline_wins']}→{effect['perturbed_wins']} enemy route wins / {effect['case_count']}; "
            f"Δ {_fmt_pp(effect['difference'])}, 95% CI {_fmt_pp(ci[0])} to {_fmt_pp(ci[1])}, "
            f"McNemar p={effect['mcnemar_exact_two_sided_p']}; adoption {primary['baseline_usage']['conditional_selection_rate']}→{primary['perturbed_usage']['conditional_selection_rate']}."
        )
    lines += ["", "## J. π0/π1 secondary effects"]
    for condition_id, data in analysis["condition_analysis"].items():
        for policy in ("pi0_heuristic", "pi1_cp3"):
            effect = data["policies"][policy]["route_effect"]["enemy_victory"]
            lines.append(
                f"- {condition_id} {policy}: {effect['baseline_wins']}→{effect['perturbed_wins']}, Δ {_fmt_pp(effect['difference'])}."
            )
    lines += ["", "## K. Local-combat effects"]
    for condition_id, data in analysis["condition_analysis"].items():
        local = data["policies"][PRIMARY_POLICY]["local_effect"]
        enemy = local["enemy_victory"]
        lines.append(
            f"- {condition_id}: matched mechanic-relevant encounters={local['matched_relevant_encounters']}; local enemy-win Δ {_fmt_pp(enemy['difference'])}; HP-loss Δ {local['continuous']['hp_loss']['mean_difference']}; Effort-loss Δ {local['continuous']['effort_loss']['mean_difference']}; round Δ {local['continuous']['rounds']['mean_difference']}."
        )
    lines += ["", "## L. Local-vs-route horizon alignment"]
    for condition_id, data in analysis["condition_analysis"].items():
        lines.append(
            f"- {condition_id}: π2 **{data['policies'][PRIMARY_POLICY]['horizon_alignment']}**."
        )

    artifact_sections = [
        ("M. Mark causal audit", "MARK", MARK_PATH),
        ("N. Grab causal audit", "GRAB", GRAB_PATH),
        ("O. Guard causal audit", "GUARD", GUARD_PATH),
        ("P. Effort-drain audit", "EFFORT_DRAIN", DRAIN_PATH),
    ]
    for title, mechanic, _path in artifact_sections:
        lines += ["", f"## {title}"]
        for condition_id in MECHANIC_CONDITIONS[mechanic]:
            data = analysis["condition_analysis"][condition_id]["policies"][PRIMARY_POLICY]
            usage = data["perturbed_usage"]
            context = data["selected_context"]
            resolution = data["mechanic_resolution"]
            lines.append(
                f"- {condition_id}: {usage['selections']}/{usage['opportunities']} legal opportunities selected ({_fmt_pct(usage['conditional_selection_rate'])}); mean use round {context['mean_round']}; horizon={data['horizon_alignment']}; resolution diagnostics={resolution}."
            )
    lines += [
        "",
        "## Q. Opportunity-cost analysis",
        f"- Detailed deterministic direct-alternative diagnostics: `{OPPORTUNITY_PATH.name}`. Immediate pressure given up is reported only where a legal direct alternative can be reconstructed; later value remains separately measured at local and route horizons.",
        "",
        "## R. Route-position effects",
        f"- EARLY/MIDDLE/LATE/SINGLE conditional-selection effects are frozen in `{ROUTE_POSITION_PATH.name}`.",
        "",
        "## S. Encounter-horizon effects",
        f"- Baseline-duration SHORT/MEDIUM/LONG behavior is frozen in `{ENCOUNTER_HORIZON_PATH.name}`; duration bands use baseline rounds, not perturbed duration.",
        "",
        "## T. State niches",
        f"- Factual HP/Effort/Marked/route-position/encounter selection niches are frozen in `{NICHE_PATH.name}`; no designer desirability labels are injected.",
        "",
        "## U. Policy-response differences",
        f"- π0/π1/π2 adoption and local/route response contrasts are preserved in `{POLICY_RESPONSE_PATH.name}`.",
        "",
        "## V. Strategy-space effects",
        f"- Exact-common-state action switching and conditional binary action entropy are recorded in `{STRATEGY_PATH.name}`; higher entropy is not treated as inherently better.",
        "",
        "## W. Dominance/degeneracy analysis",
    ]
    collapse_rows: list[str] = []
    for condition_id, data in analysis["condition_analysis"].items():
        collapse = data["policies"][PRIMARY_POLICY]["dominance"]["potential_strategy_collapse"]
        lines.append(f"- {condition_id}: π2 potential strategy collapse = **{collapse}**.")
        if collapse:
            collapse_rows.append(condition_id)
    lines += ["", "## X. Causal classifications"]
    for mechanic, classification in analysis["classifications"].items():
        lines.append(f"- {mechanic}: **{classification}**")
    design = analysis["design"]
    lines += [
        "",
        "## Y. Player-facing design interpretation",
        f"- Direct-pressure dominance explanation: **{design['direct_pressure_dominance_explanation']['classification']}** with evidence dimensions {design['direct_pressure_dominance_explanation']['evidence_dimensions']}.",
        f"- Single-combat analysis understates at least one setup/control mechanic: **{design['setup_control_value_understated_by_single_combat']}**.",
        f"- At least one local-vs-route tradeoff: **{design['local_route_tradeoff_present']}**.",
        f"- Potential strategy collapse: **{design['potential_strategy_collapse']}** ({collapse_rows}).",
        f"- CP8 production balance change justified: **{design['production_balance_change_justified']}**; no change is implemented.",
        "",
        "## Z. Maze implications",
        f"- Case strengthened: **{design['maze_director_case_strengthened']}**. These are sequencing/scoring implications only; no Maze AI is implemented.",
        "",
        "## AA. Recommended next research direction",
        f"- **{analysis['next']['recommended_next_direction']}**. This recommendation is not implemented in CP8.",
        "",
        "## AB. Limitations",
        "- π1/π2 are frozen models trained under baseline mechanics; G2 and D1 are not visible prospectively to their frozen observation schema, so adoption non-response cannot establish mechanical weakness.",
        "- CP8 uses fresh exploratory causal seeds but is not a new pristine confirmation study; it estimates sensitivity for the frozen route/policy setup.",
        "- Local analyses include matched reached mechanic-relevant encounters; changed reachability is reported separately and the route endpoint remains authoritative for expedition value.",
        "- Opportunity-cost and follow-up diagnostics are observational consequences inside randomized paired perturbations, not a new learned counterfactual value model.",
        "",
        "## AC. Quality gates",
        "- Final current-tree quality gates are filled after sealed evaluation; scientific artifacts are not regenerated afterward.",
        "",
        "## AD. Artifact/checksum manifest",
        f"- Artifact root: `{ARTIFACT_ROOT}`. Manifest: `{MANIFEST_PATH.name}`. Raw per-condition workers are retained under `workers/` and remain ignored/untracked.",
        "",
        "## AE. Final verdict",
    ]
    lines.extend(
        [
            f"1. MARK: **{analysis['classifications']['MARK']}**",
            f"2. GRAB: **{analysis['classifications']['GRAB']}**",
            f"3. GUARD: **{analysis['classifications']['GUARD']}**",
            f"4. EFFORT DRAIN: **{analysis['classifications']['EFFORT_DRAIN']}**",
            f"5. Direct-pressure dominance: **{design['direct_pressure_dominance_explanation']['classification']}**",
            f"6. Setup/control value understated by single-combat analysis: **{design['setup_control_value_understated_by_single_combat']}**",
            f"7. Local-vs-route tradeoff present: **{design['local_route_tradeoff_present']}**",
            f"8. Meaningful strategic niches rather than universal dominance: **{design['meaningful_niches_without_universal_dominance']}**",
            f"9. Any potential strategy collapse: **{design['potential_strategy_collapse']}**",
            f"10. CP8 justifies a production balance change: **{design['production_balance_change_justified']}**",
            f"11. π2 remains strongest frozen route-level oracle across perturbations: **{design['pi2_still_strongest_route_oracle']}**",
            f"12. Route-aware future policy learning case strengthened: **{design['route_aware_policy_learning_case_strengthened']}**",
            f"13. Adversarial Maze-director case strengthened: **{design['maze_director_case_strengthened']}**",
            f"14. Recommended next direction: **{analysis['next']['recommended_next_direction']}**",
            "",
            "Checkpoint 8 stops here. No production balance change, winning perturbation promotion, retraining, π3, route-aware training, search, player optimization, or Maze AI was implemented.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest() -> dict[str, Any]:
    root_artifacts = (
        PROTOCOL_PATH,
        PROVENANCE_PATH,
        BASELINE_MECHANICS_PATH,
        MATRIX_PATH,
        OBSERVABILITY_PATH,
        BASELINE_EQUIVALENCE_PATH,
        ROUTE_RESULTS_PATH,
        PAIRED_ROUTE_PATH,
        LOCAL_EFFECTS_PATH,
        HORIZON_PATH,
        MARK_PATH,
        GRAB_PATH,
        GUARD_PATH,
        DRAIN_PATH,
        OPPORTUNITY_PATH,
        ROUTE_POSITION_PATH,
        ENCOUNTER_HORIZON_PATH,
        NICHE_PATH,
        POLICY_RESPONSE_PATH,
        STRATEGY_PATH,
        DOMINANCE_PATH,
        CAUSAL_PATH,
        DESIGN_PATH,
        MAZE_PATH,
        NEXT_PATH,
        REPORT_PATH,
    )
    worker_artifacts = tuple(
        path
        for condition in CONDITION_IDS
        for path in (
            WORKER_ROOT / f"{condition}.jsonl",
            WORKER_ROOT / f"{condition}.done.json",
        )
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "cp7_commit": CP7_COMMIT,
        "frozen_protocol_checksum": sha256_file(PROTOCOL_PATH),
        "pi1_model_checksum": PI1_MODEL_SHA,
        "pi2_model_checksum": PI2_MODEL_SHA,
        "artifacts": {
            str(path.relative_to(ARTIFACT_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in (*root_artifacts, *worker_artifacts)
        },
        "evaluation_accessed": True,
        "training_performed": False,
        "post_evaluation_tuning_performed": False,
        "production_gameplay_changed": False,
    }


def _reverify_manifest(manifest: Mapping[str, Any]) -> None:
    if sha256_file(PROTOCOL_PATH) != manifest["frozen_protocol_checksum"]:
        raise ValueError("CP8 frozen protocol changed after evaluation.")
    for relative, expected in manifest["artifacts"].items():
        path = ARTIFACT_ROOT / str(relative)
        if sha256_file(path) != expected:
            raise ValueError(f"CP8 manifest checksum mismatch: {relative}")
    provenance = _read_json(PROVENANCE_PATH)
    if provenance["implementation_hashes"] != _hash_paths(
        tuple(provenance["implementation_hashes"])
    ):
        raise ValueError("CP8 implementation changed after evaluation.")
    simulator, content = _current_production_hashes()
    if simulator != provenance["simulator_hashes"] or content != provenance["content_hashes"]:
        raise ValueError("Production simulator/content changed after CP8 evaluation.")


def run_evaluation() -> None:
    _validate_frozen_state()
    if MANIFEST_PATH.exists() or ROUTE_RESULTS_PATH.exists():
        raise ValueError("CP8 outputs already exist; refusing to rerun the sealed evaluation.")
    _run_workers()
    analysis = analyze_results()
    _write_report(analysis)
    manifest = _manifest()
    _write_json(MANIFEST_PATH, manifest)
    _reverify_manifest(manifest)
    print(f"CP8_PROTOCOL_SHA={sha256_file(PROTOCOL_PATH)}")
    print(f"CONDITIONS={len(CONDITION_IDS)}")
    print("TRAINING_PERFORMED=false")
    print("POST_EVALUATION_TUNING=false")
    print("CP8_COMPLETE=true")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CP8 route-aware mechanic perturbations.")
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="Freeze CP8 protocol before new-seed access.")
    freeze.add_argument("--pre-access-tests-passed", action="store_true")
    worker = sub.add_parser("worker", help="Run one isolated perturbation condition.")
    worker.add_argument("--condition", required=True, choices=CONDITION_IDS)
    sub.add_parser("evaluate", help="Run all frozen CP8 perturbations exactly once.")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze_protocol(pre_access_tests_passed=bool(args.pre_access_tests_passed))
        return 0
    if args.command == "worker":
        run_worker(str(args.condition))
        return 0
    if args.command == "evaluate":
        run_evaluation()
        return 0
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
