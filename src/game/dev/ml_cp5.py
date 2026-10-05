# ruff: noqa: E501
"""Checkpoint 5 untouched-seed confirmation and robustness audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from game.campaign.company import create_new_company
from game.combat.enemy_decision import EnemyDecisionPolicy, HeuristicEnemyDecisionPolicy
from game.combat.enemy_learning import create_hero_policy, run_enemy_learning_episode
from game.content.definitions import GameDefinitions
from game.core.rng import GameRng
from game.data.loaders import load_game_definitions
from game.expedition.cave import create_encounter_combat
from game.ml.artifacts import sha256_file
from game.ml.confirmation_audit import (
    AuditRecordingPolicy,
    aggregate_cases,
    attach_decision_events,
    classify_advantage,
    classify_direct_pressure,
    classify_equivalence,
    classify_reduced_behavior,
    disagreement_audit,
    paired_policy_comparison,
    stable_disagreement_sample,
    summarize_case,
)
from game.ml.counterfactual import combat_state_digest, compare_outcomes, stable_digest
from game.ml.outcome_features import CP3_FEATURE_NAMES, CP3_FEATURE_SCHEMA
from game.ml.outcome_ranking import load_outcome_normalization, load_outcome_ranker
from game.ml.policy_iteration import (
    FrozenOutcomeRankerPolicy,
    collect_policy_source_decisions,
    run_policy_counterfactual_branch,
    verify_policy_control_branch,
)
from game.ml.splits import DatasetSplit

EXPERIMENT_ID = "checkpoint5-untouched-confirmation-v1"
CP4_COMMIT = "356b72a8f078809dbaa7e01c20e832b34ddddef5"
CP3_ROOT = Path("artifacts/ml/checkpoint3-counterfactual-outcome-v1")
CP4_ROOT = Path("artifacts/ml/checkpoint4-policy-iteration-v1")
ARTIFACT_ROOT = Path("artifacts/ml") / EXPERIMENT_ID
PROTOCOL_PATH = ARTIFACT_ROOT / "frozen_protocol.json"
PROVENANCE_PATH = ARTIFACT_ROOT / "provenance.json"
RECURSIVE_RESULTS_PATH = ARTIFACT_ROOT / "recursive_results.json"
PAIRED_PATH = ARTIFACT_ROOT / "paired_comparisons.json"
CI_PATH = ARTIFACT_ROOT / "confidence_intervals.json"
ENCOUNTER_PATH = ARTIFACT_ROOT / "encounter_breakdown.json"
HERO_POLICY_PATH = ARTIFACT_ROOT / "hero_policy_breakdown.json"
BEHAVIOR_PATH = ARTIFACT_ROOT / "behavior_audit.json"
TARGETING_PATH = ARTIFACT_ROOT / "targeting_audit.json"
DISAGREEMENT_PATH = ARTIFACT_ROOT / "disagreement_audit.json"
SPOT_SAMPLE_PATH = ARTIFACT_ROOT / "counterfactual_spot_sample.json"
SPOT_PATH = ARTIFACT_ROOT / "counterfactual_spot_audit.json"
PLATEAU_PATH = ARTIFACT_ROOT / "plateau_assessment.json"
BALANCE_PATH = ARTIFACT_ROOT / "balance_evidence.json"
MANIFEST_PATH = ARTIFACT_ROOT / "manifest.json"
REPORT_PATH = ARTIFACT_ROOT / "REPORT.md"

PI1_NAME = "outcome_mlp_seed_20261004"
PI1_MODEL_SHA = "61930493025f286ef300a6229da5c266cf4637dbd7f9150aec24ad8f8f9e313c"
PI1_CONFIG_SHA = "bda1cad3bc8bd48742157430bc4aec0413a70156362ee6eea17604a47f0315d1"
PI1_NORM_SHA = "5b4b787b572014b767348862df06d0cbdfe0605c6687562b04f7abcb79e0391f"
PI2_NAME = "cp4_outcome_mlp_seed_20261009"
PI2_MODEL_SHA = "718fc3167a98a34aa5c17077c5d7ecea6e0c811c37c391e2ba558061a6576424"
PI2_CONFIG_SHA = "70f6cf92407e5808859dc83774fe32b0fa080877809d98b275d48bc7925d3e36"
PI2_NORM_SHA = "d149e8aead0f779dd180ec1ffa05622c45caa66fc89dc0a3b13a61d41b9a03b8"
CONFIRMATION_SEEDS = tuple(range(30001, 30129))
MAX_ROUNDS = 20
SPOT_PER_GROUP = 8
EQUIVALENCE_MARGIN = 0.10
ENCOUNTER_IDS = (
    "cave_mini_boss", "generated_maze_glass_pack", "generated_maze_hunt",
    "generated_maze_hunt_guarded", "generated_maze_pattern_cell", "generated_maze_probe",
    "generated_maze_stalker", "maze_depth_1", "road_bandits", "shallow_cave", "wolf_pack",
)
ID_HERO_POLICIES = ("naive", "damage_race", "survival", "anti_mark", "conservative")
TACTICAL_HERO_POLICIES = ("tactical",)
COMPANY_HERO_POLICIES = ("company_survival",)
GROUPS = {
    "in_distribution": ID_HERO_POLICIES,
    "tactical": TACTICAL_HERO_POLICIES,
    "company_survival": COMPANY_HERO_POLICIES,
}
POLICY_NAMES = ("pi0_heuristic", "pi1_cp3", "pi2_cp4")
PAIRS = (
    ("pi0_heuristic", "pi1_cp3"),
    ("pi0_heuristic", "pi2_cp4"),
    ("pi1_cp3", "pi2_cp4"),
)
IMPLEMENTATION_PATHS = (
    "src/game/dev/ml_cp5.py",
    "src/game/ml/confirmation_audit.py",
    "tests/test_ml_confirmation.py",
    "src/game/ml/policy_iteration.py",
    "src/game/ml/counterfactual.py",
    "src/game/ml/outcome_features.py",
    "src/game/ml/outcome_ranking.py",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CP5 untouched-seed confirmation.")
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="Freeze CP5 protocol without touching confirmation seeds.")
    freeze.add_argument("--pre-access-tests-passed", action="store_true")
    sub.add_parser("confirm", help="Open the frozen untouched range exactly once and evaluate.")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze_protocol(pre_access_tests_passed=bool(args.pre_access_tests_passed))
        return 0
    if args.command == "confirm":
        run_confirmation()
        return 0
    raise ValueError(args.command)


def freeze_protocol(*, pre_access_tests_passed: bool) -> None:
    if not pre_access_tests_passed:
        raise ValueError("CP5 freeze requires explicit evidence that pre-access tests passed.")
    if ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.iterdir()):
        raise ValueError("CP5 artifact root is not pristine; inspect before freezing.")
    if _git_head() != CP4_COMMIT:
        raise ValueError("CP5 must begin from the committed CP4 baseline.")
    definitions = load_game_definitions()
    policies, identities = _load_frozen_policies()
    _verify_schema()
    simulator_hashes, content_hashes = _current_simulator_content_hashes()
    old_seed_probe = _old_seed_integrity_probe(definitions, policies)
    protocol = _protocol_payload(identities)
    _write_json(PROTOCOL_PATH, protocol)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "cp4_commit": CP4_COMMIT,
        "current_head": _git_head(),
        "repository_dirty_for_cp5_implementation": _git_dirty(),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__},
        "policies": identities,
        "schema": {
            "id": CP3_FEATURE_SCHEMA.schema_id,
            "version": CP3_FEATURE_SCHEMA.version,
            "feature_count": len(CP3_FEATURE_NAMES),
            "feature_names_sha256": hashlib.sha256("\n".join(CP3_FEATURE_NAMES).encode()).hexdigest(),
        },
        "implementation_hashes": _hash_paths(IMPLEMENTATION_PATHS),
        "simulator_hashes": simulator_hashes,
        "content_hashes": content_hashes,
        "hero_policy_ids": list(ID_HERO_POLICIES + TACTICAL_HERO_POLICIES + COMPANY_HERO_POLICIES),
        "encounter_ids": list(ENCOUNTER_IDS),
        "rng_authority": "game.core.rng.GameRng; deterministic replay reuses exact seed and policy",
        "old_previously_used_seed_integrity_probe": old_seed_probe,
        "pre_access_tests_passed": True,
        "confirmation_accessed": False,
        "post_confirmation_tuning_performed": False,
        "frozen_protocol_checksum": protocol_sha,
    }
    _write_json(PROVENANCE_PATH, provenance)
    print(f"CP5_FROZEN_PROTOCOL={PROTOCOL_PATH}")
    print(f"CP5_FROZEN_PROTOCOL_SHA256={protocol_sha}")
    print("CONFIRMATION_ACCESSED=false")


def _protocol_payload(identities: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_before_confirmation",
        "confirmation_seeds": [30001, 30128],
        "confirmation_seed_count": len(CONFIRMATION_SEEDS),
        "confirmation_accessed": False,
        "post_confirmation_tuning_performed": False,
        "policies": identities,
        "encounter_ids": list(ENCOUNTER_IDS),
        "hero_policy_groups": {key: list(value) for key, value in GROUPS.items()},
        "observation_schema": {"id": CP3_FEATURE_SCHEMA.schema_id, "version": CP3_FEATURE_SCHEMA.version, "feature_count": len(CP3_FEATURE_NAMES)},
        "evaluation": {
            "recursive": "pi0/pi1/pi2 on identical encounter, hero-policy, seed, initial state and RNG",
            "max_rounds": MAX_ROUNDS,
            "enemy_wait_mode": "none",
            "enemy_movement_mode": "recovery_only",
            "primary_endpoint": "company_survival enemy win rate",
            "primary_metrics": ["enemy wins", "case count", "win rate", "Wilson 95% CI", "mean final hero HP", "median final hero HP", "hero deaths", "terminal downs", "mortal wounds", "enemy-caused Effort drain", "rounds", "terminal outcome counts"],
            "paired_metrics": ["shared wins", "baseline-only wins", "comparison-only wins", "win-rate difference", "paired-difference normal 95% CI", "exact two-sided McNemar p", "discordant count", "transition matrix"],
            "robustness": ["company_survival by encounter", "all established hero policies"],
            "behavior_diagnostics": ["Mark", "Grab", "Guard", "Bite", "direct pressure", "Effort drain", "target concentration", "setup-to-payoff", "movement", "waiting/delay"],
            "targeting_diagnostics": ["top-target share", "HHI", "repeat attacks", "lowest-HP legal target", "support-capable target", "target switches", "Downed attacks", "deaths", "mortal-wound conversions", "protection/formation", "target class"],
            "disagreement_pairs": [list(pair) for pair in PAIRS],
            "disagreement_dimensions": ["encounter", "round", "actor", "skill motif", "target change", "party HP band", "party Effort band", "marked state", "target life state", "formation/protection state"],
            "counterfactual_spot_audit": {"per_group": SPOT_PER_GROUP, "sampling": "lowest stable SHA256 over factual disagreement provenance only", "continuation": "the trajectory/source policy for the sampled state", "all_legal_candidates": True, "no_training": True},
        },
        "classification_rules": {
            "learned_vs_pi0": {
                "CONFIRMED": "positive paired difference, lower 95% paired CI > 0, exact McNemar p < 0.05",
                "DIRECTIONALLY_REPLICATED": "positive point estimate without the CONFIRMED rule",
                "CONTRADICTED": "negative paired difference, upper 95% paired CI < 0, p < 0.05",
                "NOT_REPLICATED": "negative point estimate without contradiction",
                "INCONCLUSIVE": "zero point estimate",
            },
            "pi2_vs_pi1": {
                "CLEARLY_STRONGER": "positive difference, lower CI > 0, p < 0.05",
                "CLEARLY_WEAKER": "negative difference, upper CI < 0, p < 0.05",
                "APPROXIMATELY_EQUIVALENT": "95% paired-difference CI entirely inside ±0.10",
                "INCONCLUSIVE": "otherwise",
            },
            "low_setup_behavior": {
                "CONFIRMED": "both learned policies use <=75% of heuristic rate",
                "DIRECTIONALLY_REPLICATED": "both learned policies are below heuristic but threshold not met",
                "INCONCLUSIVE": "only one learned policy is lower",
                "NOT_REPLICATED": "neither learned policy is lower",
            },
            "direct_pressure": {
                "CONFIRMED": "both learned direct-pressure rates are at least heuristic rate",
                "DIRECTIONALLY_REPLICATED": "mean learned rate is at least heuristic rate",
                "INCONCLUSIVE": "only one learned rate is at least heuristic",
                "NOT_REPLICATED": "both learned rates are lower",
            },
            "plateau": {
                "SUPPORTED": "pi1 and pi2 each replicate advantage vs pi0 and pi2≈pi1 by the predeclared ±10pp equivalence rule",
                "NOT_SUPPORTED": "pi2 is clearly stronger/weaker or either learned policy is contradicted vs pi0",
                "MIXED": "otherwise",
            },
        },
        "prior_company_survival_reference": {"pi0": 0.03409090909090909, "pi1": 0.2727272727272727, "pi2": 0.25},
        "forbidden_after_access": ["retraining", "model selection", "threshold changes", "feature/schema changes", "reward/objective changes", "encounter/hero-policy edits", "seed selection", "case dropping", "post-hoc metric substitution"],
    }


def _load_frozen_policies() -> tuple[dict[str, EnemyDecisionPolicy], dict[str, Any]]:
    _require_sha(CP3_ROOT / "frozen_config.json", PI1_CONFIG_SHA, "CP3 frozen config")
    _require_sha(CP4_ROOT / "frozen_config.json", PI2_CONFIG_SHA, "CP4 frozen config")
    cp3_manifest = _read_json(CP3_ROOT / "final_manifest.json")
    pi1_entries = [entry for entry in cp3_manifest["model_artifacts"] if entry["name"] == PI1_NAME]
    if len(pi1_entries) != 1:
        raise ValueError("CP3 pi1 manifest identity is not unique.")
    pi1_entry = pi1_entries[0]
    pi1_model_path = CP3_ROOT / str(pi1_entry["model_file"])
    _require_sha(pi1_model_path, PI1_MODEL_SHA, "pi1 model")
    _require_sha(CP3_ROOT / "normalization.json", PI1_NORM_SHA, "pi1 normalization")
    pi1_meta = CP3_ROOT / f"{PI1_NAME}.json"
    _require_sha(pi1_meta, str(pi1_entry["metadata_checksum"]), "pi1 metadata")
    pi1_loaded = load_outcome_ranker(pi1_meta)
    pi1_norm = load_outcome_normalization(CP3_ROOT / "normalization.json")

    cp4_manifest = _read_json(CP4_ROOT / "final_manifest.json")
    if cp4_manifest.get("pi2", {}).get("name") != PI2_NAME or cp4_manifest.get("pi2", {}).get("model_checksum") != PI2_MODEL_SHA:
        raise ValueError("CP4 manifest pi2 identity differs from reviewed result.")
    pi2_meta = CP4_ROOT / f"{PI2_NAME}.json"
    pi2_model_path = CP4_ROOT / f"{PI2_NAME}.pt"
    _require_sha(pi2_model_path, PI2_MODEL_SHA, "pi2 model")
    _require_sha(CP4_ROOT / "normalization.json", PI2_NORM_SHA, "pi2 normalization")
    pi2_meta_sha = next(entry["metadata_checksum"] for entry in cp4_manifest["model_artifacts"] if entry["name"] == PI2_NAME)
    _require_sha(pi2_meta, str(pi2_meta_sha), "pi2 metadata")
    pi2_loaded = load_outcome_ranker(pi2_meta)
    pi2_norm = load_outcome_normalization(CP4_ROOT / "normalization.json")
    policies: dict[str, EnemyDecisionPolicy] = {
        "pi0_heuristic": HeuristicEnemyDecisionPolicy(),
        "pi1_cp3": FrozenOutcomeRankerPolicy(pi1_loaded.model, pi1_norm, policy_id="cp3_pi1"),
        "pi2_cp4": FrozenOutcomeRankerPolicy(pi2_loaded.model, pi2_norm, policy_id="cp4_pi2"),
    }
    identities = {
        "pi0_heuristic": {"kind": "handcrafted heuristic baseline"},
        "pi1_cp3": {"name": PI1_NAME, "model_checksum": PI1_MODEL_SHA, "config_checksum": PI1_CONFIG_SHA, "normalization_checksum": PI1_NORM_SHA},
        "pi2_cp4": {"name": PI2_NAME, "model_checksum": PI2_MODEL_SHA, "config_checksum": PI2_CONFIG_SHA, "normalization_checksum": PI2_NORM_SHA},
    }
    return policies, identities


def _old_seed_integrity_probe(definitions: GameDefinitions, policies: Mapping[str, EnemyDecisionPolicy]) -> dict[str, Any]:
    encounter = "cave_mini_boss"
    hero_policy_id = "company_survival"
    seed = 20001
    result: dict[str, Any] = {"encounter_id": encounter, "hero_policy_id": hero_policy_id, "seed": seed, "policies": {}}
    for name, policy in policies.items():
        signatures = []
        for _ in range(2):
            state = create_encounter_combat(create_new_company(definitions), definitions, encounter)
            episode = run_enemy_learning_episode(
                state, definitions, GameRng(seed), policy=policy, max_rounds=MAX_ROUNDS,
                encounter_id=encounter, encounter_name="Cave Mini Boss",
                hero_policy=create_hero_policy(hero_policy_id, encounter_id=encounter, seed=seed),
                enemy_wait_mode="none", enemy_movement_mode="recovery_only",
            )
            signatures.append({"final_state": combat_state_digest(state), "episode": stable_digest(episode), "victor": episode.final_victor})
        if signatures[0] != signatures[1]:
            raise ValueError(f"Previously used deterministic case failed replay for {name}.")
        result["policies"][name] = signatures[0]
    result["deterministic_repeat_match"] = True
    return result


def run_confirmation() -> None:
    protocol, provenance = _validate_frozen_pre_access_state()
    if MANIFEST_PATH.exists() or RECURSIVE_RESULTS_PATH.exists():
        raise ValueError("CP5 confirmation outputs already exist; refusing to access untouched seeds again.")
    definitions = load_game_definitions()
    policies, _identities = _load_frozen_policies()
    case_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for group, hero_policy_ids in GROUPS.items():
        for encounter_id in ENCOUNTER_IDS:
            for hero_policy_id in hero_policy_ids:
                for seed in CONFIRMATION_SEEDS:
                    case_id = f"{encounter_id}|{hero_policy_id}|{seed}"
                    for policy_name in POLICY_NAMES:
                        state = create_encounter_combat(create_new_company(definitions), definitions, encounter_id)
                        audit_policy = AuditRecordingPolicy(executed_name=policy_name, policies=policies)
                        episode = run_enemy_learning_episode(
                            state,
                            definitions,
                            GameRng(seed),
                            policy=audit_policy,
                            max_rounds=MAX_ROUNDS,
                            encounter_id=encounter_id,
                            encounter_name=encounter_id.replace("_", " ").title(),
                            hero_policy=create_hero_policy(hero_policy_id, encounter_id=encounter_id, seed=seed),
                            enemy_wait_mode="none",
                            enemy_movement_mode="recovery_only",
                        )
                        attach_decision_events(audit_policy.decisions, episode)
                        for decision in audit_policy.decisions:
                            decision.update(
                                {
                                    "case_id": case_id,
                                    "group": group,
                                    "encounter_id": encounter_id,
                                    "hero_policy_id": hero_policy_id,
                                    "seed": seed,
                                    "trajectory_policy": policy_name,
                                }
                            )
                        decision_rows.extend(audit_policy.decisions)
                        case_rows.append(
                            summarize_case(
                                case_id=case_id,
                                group=group,
                                encounter_id=encounter_id,
                                hero_policy_id=hero_policy_id,
                                seed=seed,
                                policy_name=policy_name,
                                state=state,
                                episode=episode,
                                decisions=audit_policy.decisions,
                            )
                        )
    recursive = _recursive_payload(case_rows)
    _write_json(RECURSIVE_RESULTS_PATH, recursive)
    paired = _paired_payload(case_rows)
    _write_json(PAIRED_PATH, paired)
    _write_json(CI_PATH, _confidence_payload(recursive, paired))
    _write_json(ENCOUNTER_PATH, _encounter_breakdown(case_rows))
    _write_json(HERO_POLICY_PATH, _hero_policy_breakdown(case_rows))
    behavior = _behavior_audit(case_rows, recursive)
    _write_json(BEHAVIOR_PATH, behavior)
    _write_json(TARGETING_PATH, _targeting_audit(recursive))
    disagreement = disagreement_audit(decision_rows, PAIRS)
    _write_json(DISAGREEMENT_PATH, disagreement)

    sample = stable_disagreement_sample(decision_rows, per_group=SPOT_PER_GROUP)
    _write_json(
        SPOT_SAMPLE_PATH,
        {
            "method": "lowest-sha256-over-group-trajectory-policy-encounter-hero-seed-action-state-v1",
            "per_group": SPOT_PER_GROUP,
            "selected_before_counterfactual_outcomes": True,
            "sample": sample,
        },
    )
    spot = _counterfactual_spot_audit(definitions, policies, sample)
    _write_json(SPOT_PATH, spot)
    plateau = _plateau_assessment(recursive, paired, disagreement, spot)
    _write_json(PLATEAU_PATH, plateau)
    balance = _balance_evidence(recursive, behavior)
    _write_json(BALANCE_PATH, balance)
    _write_report(recursive, paired, behavior, disagreement, spot, plateau, balance)
    manifest = _manifest(protocol, provenance)
    _write_json(MANIFEST_PATH, manifest)
    _reverify_manifest(manifest)
    print(f"CP5_CONFIRMATION_CASES={len(case_rows)}")
    print(f"CP5_PROTOCOL_SHA256={sha256_file(PROTOCOL_PATH)}")
    print("POST_CONFIRMATION_TUNING_PERFORMED=false")
    print("CP5_CONFIRMATION_COMPLETE=true")


def _recursive_payload(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for group in GROUPS:
        summaries[group] = {}
        for policy in POLICY_NAMES:
            rows = [row for row in case_rows if row["group"] == group and row["policy"] == policy]
            summaries[group][policy] = aggregate_cases(rows)
    return {
        "experiment_id": EXPERIMENT_ID,
        "confirmation_seeds": [30001, 30128],
        "case_rows_are_paired_by_case_id": True,
        "summaries": summaries,
        "cases": case_rows,
    }


def _paired_payload(case_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in GROUPS:
        result[group] = {}
        for a, b in PAIRS:
            first = [row for row in case_rows if row["group"] == group and row["policy"] == a]
            second = [row for row in case_rows if row["group"] == group and row["policy"] == b]
            result[group][f"{b}_minus_{a}"] = paired_policy_comparison(first, second)
    return result


def _confidence_payload(recursive: Mapping[str, Any], paired: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "win_rate_wilson_95": {
            group: {policy: recursive["summaries"][group][policy]["enemy_win_rate_wilson_95"] for policy in POLICY_NAMES}
            for group in GROUPS
        },
        "paired_difference_ci_95": {
            group: {key: value["paired_difference_ci_95"] for key, value in paired[group].items()}
            for group in GROUPS
        },
    }


def _encounter_breakdown(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for encounter in ENCOUNTER_IDS:
        result[encounter] = {}
        for policy in POLICY_NAMES:
            selected = [row for row in rows if row["group"] == "company_survival" and row["encounter_id"] == encounter and row["policy"] == policy]
            result[encounter][policy] = aggregate_cases(selected)
    return result


def _hero_policy_breakdown(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hero_ids = ID_HERO_POLICIES + TACTICAL_HERO_POLICIES + COMPANY_HERO_POLICIES
    result: dict[str, Any] = {}
    for hero_id in hero_ids:
        result[hero_id] = {}
        for policy in POLICY_NAMES:
            selected = [row for row in rows if row["hero_policy_id"] == hero_id and row["policy"] == policy]
            result[hero_id][policy] = aggregate_cases(selected)
    return result


def _behavior_audit(rows: Sequence[Mapping[str, Any]], recursive: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"groups": {}, "diagnostic_only": True}
    for group in GROUPS:
        result["groups"][group] = {}
        for policy in POLICY_NAMES:
            selected = [row for row in rows if row["group"] == group and row["policy"] == policy]
            summary = recursive["summaries"][group][policy]
            wins = [row for row in selected if row["enemy_win"]]
            losses = [row for row in selected if not row["enemy_win"]]
            keys = (
                "marks_applied", "guard_uses", "grab_uses", "bite_uses", "effort_drain_uses",
                "direct_pressure_rate", "target_top_share", "repeat_target_transitions",
                "target_switches", "enemy_moves", "enemy_waits",
            )
            result["groups"][group][policy] = {
                "means": {key: summary[f"mean_{key}"] for key in keys},
                "win_loss_association": {
                    key: {
                        "mean_in_wins": mean(float(row[key]) for row in wins) if wins else None,
                        "mean_in_losses": mean(float(row[key]) for row in losses) if losses else None,
                    }
                    for key in keys
                },
            }
    company = recursive["summaries"]["company_survival"]
    result["untouched_classification"] = {
        "low_mark": classify_reduced_behavior(company["pi0_heuristic"]["mean_marks_applied"], company["pi1_cp3"]["mean_marks_applied"], company["pi2_cp4"]["mean_marks_applied"]),
        "low_grab": classify_reduced_behavior(company["pi0_heuristic"]["mean_grab_uses"], company["pi1_cp3"]["mean_grab_uses"], company["pi2_cp4"]["mean_grab_uses"]),
        "low_guard": classify_reduced_behavior(company["pi0_heuristic"]["mean_guard_uses"], company["pi1_cp3"]["mean_guard_uses"], company["pi2_cp4"]["mean_guard_uses"]),
        "direct_pressure": classify_direct_pressure(company["pi0_heuristic"]["mean_direct_pressure_rate"], company["pi1_cp3"]["mean_direct_pressure_rate"], company["pi2_cp4"]["mean_direct_pressure_rate"]),
    }
    return result


def _targeting_audit(recursive: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"diagnostic_only": True, "groups": {}}
    for group in GROUPS:
        output["groups"][group] = {}
        for policy in POLICY_NAMES:
            summary = recursive["summaries"][group][policy]
            output["groups"][group][policy] = {
                "mean_target_top_share": summary["mean_target_top_share"],
                "mean_target_hhi": summary["mean_target_hhi"],
                "mean_repeat_target_transitions": summary["mean_repeat_target_transitions"],
                "mean_target_switches": summary["mean_target_switches"],
                "targeting_rates": summary["targeting_rates"],
                "mean_downs_caused": summary["mean_downs_caused"],
                "mean_deaths_caused": summary["mean_deaths_caused"],
                "mean_mortal_wounds_caused": summary["mean_mortal_wounds_caused"],
                "target_class_distribution": summary["target_class_distribution"],
            }
    return output


def _counterfactual_spot_audit(
    definitions: GameDefinitions,
    policies: Mapping[str, EnemyDecisionPolicy],
    sample: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in sample:
        source_name = str(item["trajectory_policy"])
        source_policy = policies[source_name]
        decisions = collect_policy_source_decisions(
            definitions,
            source_policy=source_policy,
            source_policy_id=source_name,
            split=DatasetSplit.TEST,
            encounter_id=str(item["encounter_id"]),
            hero_policy_id=str(item["hero_policy_id"]),
            seed=int(item["seed"]),
            max_rounds=MAX_ROUNDS,
        )
        matching = [decision for decision in decisions if decision.action_index == int(item["action_index"])]
        if len(matching) != 1:
            raise ValueError("Spot-audit source decision could not be reconstructed uniquely.")
        source = matching[0]
        verify_policy_control_branch(
            definitions, source, continuation_policy=source_policy,
            continuation_policy_id=source_name, max_rounds=MAX_ROUNDS,
        )
        branches: dict[str, Any] = {}
        outcomes: dict[str, Any] = {}
        for candidate in source.candidates:
            result = run_policy_counterfactual_branch(
                definitions, source, candidate, continuation_policy=source_policy,
                continuation_policy_id=source_name, max_rounds=MAX_ROUNDS,
            )
            key = f"{candidate.skill_id}|{candidate.target_id}"
            branches[key] = asdict(result.outcome)
            outcomes[key] = result.outcome
        choices = dict(item["choices"])
        comparisons: dict[str, Any] = {}
        for a, b in PAIRS:
            ca, cb = choices.get(a), choices.get(b)
            if ca is None or cb is None or ca not in outcomes or cb not in outcomes:
                continue
            pref = compare_outcomes(outcomes[ca], outcomes[cb])
            comparisons[f"{a}_vs_{b}"] = {
                "a_choice": ca,
                "b_choice": cb,
                "preference": pref.value,
                "a_enemy_win": outcomes[ca].enemy_win,
                "b_enemy_win": outcomes[cb].enemy_win,
                "win_transition": (
                    "loss_to_win"
                    if not outcomes[ca].enemy_win and outcomes[cb].enemy_win
                    else "win_to_loss"
                    if outcomes[ca].enemy_win and not outcomes[cb].enemy_win
                    else "same_win_class"
                ),
            }
        rows.append(
            {
                "sample_hash": item["sample_hash"],
                "group": item["group"],
                "source_policy": source_name,
                "encounter_id": item["encounter_id"],
                "hero_policy_id": item["hero_policy_id"],
                "seed": item["seed"],
                "action_index": item["action_index"],
                "legal_actions": list(branches),
                "choices": choices,
                "branch_outcomes": branches,
                "comparisons": comparisons,
            }
        )
    preference_counts: Counter[str] = Counter()
    win_transitions: Counter[str] = Counter()
    for row in rows:
        for comparison in row["comparisons"].values():
            preference_counts[str(comparison["preference"])] += 1
            win_transitions[str(comparison["win_transition"])] += 1
    return {
        "sampling_locked_before_branch_outcomes": True,
        "continuation": "source trajectory policy",
        "sample_count": len(rows),
        "rows": rows,
        "preference_counts": dict(preference_counts),
        "win_transition_counts": dict(win_transitions),
        "training_or_tuning_performed": False,
    }


def _plateau_assessment(
    recursive: Mapping[str, Any],
    paired: Mapping[str, Any],
    disagreement: Mapping[str, Any],
    spot: Mapping[str, Any],
) -> dict[str, Any]:
    company = paired["company_survival"]
    pi1 = company["pi1_cp3_minus_pi0_heuristic"]
    pi2 = company["pi2_cp4_minus_pi0_heuristic"]
    pi21 = company["pi2_cp4_minus_pi1_cp3"]
    pi1_class = classify_advantage(pi1)
    pi2_class = classify_advantage(pi2)
    pi21_class = classify_equivalence(pi21, margin=EQUIVALENCE_MARGIN)
    if pi1_class in {"CONFIRMED", "DIRECTIONALLY_REPLICATED"} and pi2_class in {"CONFIRMED", "DIRECTIONALLY_REPLICATED"} and pi21_class == "APPROXIMATELY_EQUIVALENT":
        plateau = "SUPPORTED"
    elif pi21_class in {"CLEARLY_STRONGER", "CLEARLY_WEAKER"} or "CONTRADICTED" in {pi1_class, pi2_class}:
        plateau = "NOT_SUPPORTED"
    else:
        plateau = "MIXED"
    return {
        "pi1_vs_pi0": pi1_class,
        "pi2_vs_pi0": pi2_class,
        "pi2_vs_pi1": pi21_class,
        "early_stabilization": plateau,
        "another_identical_greedy_iteration": "JUSTIFIED" if pi21_class == "CLEARLY_STRONGER" else "NOT_YET_JUSTIFIED",
        "company_survival_rates": {policy: recursive["summaries"]["company_survival"][policy]["enemy_win_rate"] for policy in POLICY_NAMES},
        "disagreement_rates": {key: value["disagreement_rate"] for key, value in disagreement.items()},
        "spot_preference_counts": spot["preference_counts"],
        "rule": "predeclared advantage + ±10pp equivalence classification",
    }


def _balance_evidence(recursive: Mapping[str, Any], behavior: Mapping[str, Any]) -> dict[str, Any]:
    company = recursive["summaries"]["company_survival"]
    classifications = behavior["untouched_classification"]
    return {
        "diagnostic_only_no_balance_changes": True,
        "behaviors": {
            "Mark": _balance_row("setup/focus status application", "marks_applied", company, classifications["low_mark"]),
            "Grab": _balance_row("formation disruption/setup", "grab_uses", company, classifications["low_grab"]),
            "Guard": _balance_row("ally protection/setup", "guard_uses", company, classifications["low_guard"]),
            "Bite/direct pressure": _balance_row("direct incapacitation pressure", "bite_uses", company, classifications["direct_pressure"]),
        },
        "future_test_needed": "separate frozen-policy kit-perturbation experiment; no values should change from usage frequency alone",
    }


def _balance_row(role: str, key: str, company: Mapping[str, Any], classification: str) -> dict[str, Any]:
    return {
        "mechanical_role": role,
        "pi0_mean_uses": company["pi0_heuristic"][f"mean_{key}"],
        "pi1_mean_uses": company["pi1_cp3"][f"mean_{key}"],
        "pi2_mean_uses": company["pi2_cp4"][f"mean_{key}"],
        "untouched_persistence": classification,
        "hypothesis": "current action incentives may differ from handcrafted heuristic emphasis; frequency is evidence, not a buff/nerf instruction",
    }


def _manifest(protocol: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    scientific_paths = (
        PROTOCOL_PATH, PROVENANCE_PATH, RECURSIVE_RESULTS_PATH, PAIRED_PATH, CI_PATH,
        ENCOUNTER_PATH, HERO_POLICY_PATH, BEHAVIOR_PATH, TARGETING_PATH, DISAGREEMENT_PATH,
        SPOT_SAMPLE_PATH, SPOT_PATH, PLATEAU_PATH, BALANCE_PATH, REPORT_PATH,
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "cp4_commit": CP4_COMMIT,
        "frozen_protocol_checksum": sha256_file(PROTOCOL_PATH),
        "frozen_protocol_confirmation_accessed": protocol["confirmation_accessed"],
        "pre_access_provenance_checksum": sha256_file(PROVENANCE_PATH),
        "artifacts": {str(path.name): sha256_file(path) for path in scientific_paths},
        "policies": provenance["policies"],
        "confirmation_seeds": [30001, 30128],
        "confirmation_accessed": True,
        "post_confirmation_tuning_performed": False,
        "training_performed": False,
        "production_gameplay_changed": False,
    }


def _write_report(
    recursive: Mapping[str, Any],
    paired: Mapping[str, Any],
    behavior: Mapping[str, Any],
    disagreement: Mapping[str, Any],
    spot: Mapping[str, Any],
    plateau: Mapping[str, Any],
    balance: Mapping[str, Any],
) -> None:
    company = recursive["summaries"]["company_survival"]
    prior = {"pi0_heuristic": 0.03409090909090909, "pi1_cp3": 0.2727272727272727, "pi2_cp4": 0.25}
    classifications = behavior["untouched_classification"]
    encounter = _read_json(ENCOUNTER_PATH)
    hero = _read_json(HERO_POLICY_PATH)
    targeting = _read_json(TARGETING_PATH)
    next_direction = (
        "D. separate balance perturbation experiment"
        if plateau["early_stabilization"] == "SUPPORTED"
        else "A. improved policy-learning methodology"
    )
    lines = [
        "# The Charter — Checkpoint 5 Final Untouched-Seed Confirmation",
        "",
        "## A. Provenance",
        f"- CP4 immutable baseline: `{CP4_COMMIT}`.",
        f"- Frozen protocol SHA-256: `{sha256_file(PROTOCOL_PATH)}`.",
        "- Confirmation seeds: 30001–30128; policies π0/π1/π2 remained frozen.",
        "",
        "## B. Frozen CP5 protocol",
        "- Primary endpoint: company_survival enemy win rate. Paired comparisons, Wilson intervals, paired-difference 95% CIs, exact McNemar tests, robustness, behavior, targeting, disagreement, and a fixed 8-per-group counterfactual spot sample were predeclared before confirmation access.",
        "",
        "## C. Artifact integrity",
        f"- π1 `{PI1_NAME}` `{PI1_MODEL_SHA}`; π2 `{PI2_NAME}` `{PI2_MODEL_SHA}`; schema `{CP3_FEATURE_SCHEMA.schema_id}` v{CP3_FEATURE_SCHEMA.version} ({len(CP3_FEATURE_NAMES)} features).",
        "- Simulator/content/implementation hashes were frozen before confirmation and revalidated; no production code was modified.",
        "",
        "## D. Untouched confirmation dataset",
        f"- {len(CONFIRMATION_SEEDS)} untouched seeds × 11 encounters, paired identically across policies and the established hero-policy groups.",
        "- No confirmation seed entered training, model selection, tuning, threshold selection, or metric selection.",
        "",
        "## E. Recursive π0/π1/π2 results",
    ]
    for group in GROUPS:
        lines.append(f"### {group}")
        for policy in POLICY_NAMES:
            summary = recursive["summaries"][group][policy]
            lines.append(
                f"- {policy}: {summary['enemy_win_count']}/{summary['case_count']} wins = {100*summary['enemy_win_rate']:.2f}% "
                f"(Wilson 95% {100*summary['enemy_win_rate_wilson_95'][0]:.2f}%–{100*summary['enemy_win_rate_wilson_95'][1]:.2f}%); "
                f"mean/median hero HP {summary['mean_final_hero_hp']:.2f}/{summary['median_final_hero_hp']:.2f}; "
                f"deaths {summary['mean_hero_deaths']:.3f}; downs {summary['mean_terminal_hero_downs']:.3f}; "
                f"mortal wounds {summary['mean_mortal_wounds']:.3f}; Effort drain {summary['mean_enemy_caused_effort_drain']:.3f}; rounds {summary['mean_rounds']:.2f}."
            )
    lines += ["", "## F. Paired statistical comparison"]
    for key, comp in paired["company_survival"].items():
        lines.append(
            f"- {key}: Δ={100*comp['win_rate_difference']:+.2f}pp, 95% paired CI "
            f"{100*comp['paired_difference_ci_95'][0]:+.2f} to {100*comp['paired_difference_ci_95'][1]:+.2f}pp, "
            f"discordant={comp['discordant_pair_count']}, McNemar p={comp['mcnemar_exact_two_sided_p']:.6g}; matrix={comp['transition_matrix']}."
        )
    lines += ["", "## G. Prior-result replication"]
    for policy in POLICY_NAMES:
        current = company[policy]["enemy_win_rate"]
        lines.append(f"- {policy}: prior {100*prior[policy]:.2f}% → untouched {100*current:.2f}%.")
    lines += [
        f"- π1 vs π0: **{plateau['pi1_vs_pi0']}**.",
        f"- π2 vs π0: **{plateau['pi2_vs_pi0']}**.",
        f"- π2 vs π1: **{plateau['pi2_vs_pi1']}**.",
        "",
        "## H. Encounter robustness",
    ]
    for enc in ENCOUNTER_IDS:
        vals = encounter[enc]
        lines.append(
            f"- {enc}: π0 {100*vals['pi0_heuristic']['enemy_win_rate']:.1f}%, "
            f"π1 {100*vals['pi1_cp3']['enemy_win_rate']:.1f}%, π2 {100*vals['pi2_cp4']['enemy_win_rate']:.1f}% (128 cases/policy)."
        )
    lines += ["", "## I. Hero-policy robustness"]
    for hero_id, vals in hero.items():
        lines.append(
            f"- {hero_id}: π0 {100*vals['pi0_heuristic']['enemy_win_rate']:.2f}%, "
            f"π1 {100*vals['pi1_cp3']['enemy_win_rate']:.2f}%, π2 {100*vals['pi2_cp4']['enemy_win_rate']:.2f}%."
        )
    lines += ["", "## J. Emergent strategy"]
    for name, classification in classifications.items():
        lines.append(f"- {name}: **{classification}** on untouched seeds.")
    for policy in POLICY_NAMES:
        summary = company[policy]
        lines.append(
            f"- {policy}: Mark {summary['mean_marks_applied']:.2f}, Guard {summary['mean_guard_uses']:.2f}, "
            f"Grab {summary['mean_grab_uses']:.2f}, Bite {summary['mean_bite_uses']:.2f}, direct-pressure rate {100*summary['mean_direct_pressure_rate']:.1f}%."
        )
    lines += ["", "## K. Targeting/focus-fire behavior"]
    for policy, vals in targeting["groups"]["company_survival"].items():
        rates = vals["targeting_rates"]
        lines.append(
            f"- {policy}: top-target share {vals['mean_target_top_share']:.3f}; repeat transitions {vals['mean_repeat_target_transitions']:.2f}; "
            f"lowest-HP legal {100*rates['lowest_hp_legal']:.1f}%; support-capable {100*rates['support_capable']:.1f}%; "
            f"Downed {100*rates['downed']:.1f}%; deaths caused/case {vals['mean_deaths_caused']:.3f}; mortal-wound conversions/case {vals['mean_mortal_wounds_caused']:.3f}."
        )
    lines += ["", "## L. Policy disagreement"]
    for key, vals in disagreement.items():
        lines.append(
            f"- {key}: {vals['disagreements']}/{vals['comparable_states']} = {100*vals['disagreement_rate']:.1f}% disagreement; "
            f"leading skill motifs: {list(vals['skill_motifs'].items())[:5]}."
        )
    lines += [
        "",
        "## M. Counterfactual spot audit",
        f"- Stable provenance sample: {spot['sample_count']} states; selected before branch outcomes; source-policy continuation; no training/tuning.",
        f"- Preference counts across frozen-policy choice comparisons: {spot['preference_counts']}.",
        f"- Win-transition counts: {spot['win_transition_counts']}.",
        "",
        "## N. Plateau assessment",
        f"- Early stabilization: **{plateau['early_stabilization']}**.",
        f"- Another identical greedy policy-iteration cycle: **{plateau['another_identical_greedy_iteration']}**.",
        "",
        "## O. Model/representation diagnostic",
        "- Linear rankers remained competitive with the MLP through CP3/CP4, while policy-induced state coverage and victory supervision changed materially. Current evidence therefore points more strongly to sparse victory supervision, state-distribution coverage, and continuation-policy limitations than to raw model capacity. No architecture search was performed.",
        "",
        "## P. Balance evidence",
    ]
    for name, item in balance["behaviors"].items():
        lines.append(
            f"- {name}: role={item['mechanical_role']}; π0/π1/π2 mean uses "
            f"{item['pi0_mean_uses']:.2f}/{item['pi1_mean_uses']:.2f}/{item['pi2_mean_uses']:.2f}; "
            f"untouched persistence={item['untouched_persistence']}. This is evidence about current incentives, not a buff/nerf instruction."
        )
    lines += [
        "",
        "## Q. Production-readiness assessment",
        "- **NOT WARRANTED.** Even if strength confirms, production promotion remains a separate player-experience decision. Optimized focus-fire/Downed targeting, runtime model dependency, encounter-specific balance uncertainty, and the lack of a production-facing legibility/difficulty study remain unresolved. Production AI was not modified.",
        "",
        "## R. Recommended next research direction",
        f"- **{next_direction}.** This is the single recommended broad direction from CP5 evidence; it is not implemented here.",
        "- Another identical π2→π3 greedy policy iteration is not recommended unless untouched data actually classifies π2 as clearly stronger than π1.",
        "",
        "## S. Limitations",
        "- This is confirmation on one reserved seed block, not proof of global optimality. Exact paired CIs use a predeclared normal approximation for paired binary differences; McNemar exact tests are reported alongside them. Counterfactual spot auditing is intentionally bounded and uses source-policy continuation. Behavior telemetry is diagnostic, not causal balance proof.",
        "",
        "## T. Artifact/checksum manifest",
        f"- Manifest: `{MANIFEST_PATH}`; generated artifacts remain ignored.",
        "",
        "## U. Quality gates",
        "- Final repository quality gates are run after report generation and recorded in the final handoff; scientific hashes/checksums are independently reverified.",
        "",
        "## V. Final verdict",
        f"- π1 untouched strength vs π0: **{plateau['pi1_vs_pi0']}**",
        f"- π2 untouched strength vs π0: **{plateau['pi2_vs_pi0']}**",
        f"- π2 vs π1: **{plateau['pi2_vs_pi1']}**",
        f"- CP3/CP4 low-Mark finding: **{classifications['low_mark']}**",
        f"- CP3/CP4 low-Grab finding: **{classifications['low_grab']}**",
        f"- CP3/CP4 low-Guard finding: **{classifications['low_guard']}**",
        f"- Direct-pressure/Bite finding: **{classifications['direct_pressure']}**",
        f"- Early stabilization hypothesis: **{plateau['early_stabilization']}**",
        f"- Another identical greedy policy-iteration cycle: **{plateau['another_identical_greedy_iteration']}**",
        "- Production promotion: **NOT WARRANTED**",
        "",
        "Checkpoint 5 stops here. No π3 training, tuning, production promotion, balance change, search, Maze AI, or player optimization was performed.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _current_simulator_content_hashes() -> tuple[dict[str, str], dict[str, str]]:
    cp4 = _read_json(CP4_ROOT / "frozen_config.json")
    simulator_paths = tuple(str(path) for path in cp4["simulator_hashes"])
    content_paths = tuple(str(path) for path in cp4["content_hashes"])
    return _hash_paths(simulator_paths), _hash_paths(content_paths)


def _validate_frozen_pre_access_state() -> tuple[dict[str, Any], dict[str, Any]]:
    if not PROTOCOL_PATH.exists() or not PROVENANCE_PATH.exists():
        raise ValueError("CP5 must be frozen before confirmation access.")
    protocol = _read_json(PROTOCOL_PATH)
    provenance = _read_json(PROVENANCE_PATH)
    if protocol.get("confirmation_accessed") is not False or protocol.get("post_confirmation_tuning_performed") is not False:
        raise ValueError("Frozen CP5 protocol does not prove sealed confirmation state.")
    if provenance.get("pre_access_tests_passed") is not True or provenance.get("confirmation_accessed") is not False:
        raise ValueError("Pre-access verification marker is invalid.")
    if sha256_file(PROTOCOL_PATH) != provenance["frozen_protocol_checksum"]:
        raise ValueError("Frozen CP5 protocol checksum changed.")
    if _git_head() != CP4_COMMIT:
        raise ValueError("Repository HEAD moved after CP5 freeze.")
    if provenance["implementation_hashes"] != _hash_paths(tuple(provenance["implementation_hashes"])):
        raise ValueError("CP5 implementation changed after freeze.")
    simulator, content = _current_simulator_content_hashes()
    if simulator != provenance["simulator_hashes"] or content != provenance["content_hashes"]:
        raise ValueError("Simulator/content changed after CP5 freeze.")
    _load_frozen_policies()
    _verify_schema()
    return protocol, provenance


def _verify_schema() -> None:
    if CP3_FEATURE_SCHEMA.schema_id != "enemy-outcome-candidate-v1" or CP3_FEATURE_SCHEMA.version != 1 or len(CP3_FEATURE_NAMES) != 81:
        raise ValueError("Frozen 81-feature schema identity changed.")


def _reverify_manifest(manifest: Mapping[str, Any]) -> None:
    if sha256_file(PROTOCOL_PATH) != manifest["frozen_protocol_checksum"]:
        raise ValueError("Frozen protocol changed after confirmation.")
    for name, expected in manifest["artifacts"].items():
        path = ARTIFACT_ROOT / name
        if sha256_file(path) != expected:
            raise ValueError(f"Manifest checksum mismatch: {name}")
    provenance = _read_json(PROVENANCE_PATH)
    if provenance["implementation_hashes"] != _hash_paths(tuple(provenance["implementation_hashes"])):
        raise ValueError("Implementation changed after confirmation.")
    simulator, content = _current_simulator_content_hashes()
    if simulator != provenance["simulator_hashes"] or content != provenance["content_hashes"]:
        raise ValueError("Simulator/content changed after confirmation.")


def _require_sha(path: Path, expected: str, label: str) -> None:
    if not path.exists() or sha256_file(path) != expected:
        raise ValueError(f"{label} missing or checksum-mismatched: {path}")


def _hash_paths(paths: Sequence[str]) -> dict[str, str]:
    return {str(path): sha256_file(Path(path)) for path in paths}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_dirty() -> bool:
    return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
