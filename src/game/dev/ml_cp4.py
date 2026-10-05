# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections.abc import Sequence
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
from game.ml.counterfactual import OutcomePreference, compare_outcomes, outcome_from_episode
from game.ml.distribution_audit import (
    build_distribution_audit,
    reconstruct_cp3_distribution_records,
)
from game.ml.models import linear_scorer_spec, mlp_scorer_spec
from game.ml.outcome_dataset import (
    SAMPLING_METHOD_ID,
    SamplingSpec,
    load_counterfactual_dataset,
)
from game.ml.outcome_features import (
    CP2_FEATURE_AUDIT,
    CP3_DERIVED_FEATURE_NAMES,
    CP3_FEATURE_NAMES,
    CP3_FEATURE_SCHEMA,
    CP3_NEUTRAL_FACTUAL_FEATURE_NAMES,
)
from game.ml.outcome_ranking import (
    PairwiseTrainingConfig,
    choose_candidate_index,
    evaluate_pairwise,
    fit_outcome_normalization,
    grouped_pairwise_accuracy,
    load_outcome_normalization,
    load_outcome_ranker,
    save_outcome_normalization,
    save_outcome_ranker,
    train_pairwise_ranker,
)
from game.ml.policy_iteration import (
    FrozenOutcomeRankerPolicy,
    PolicyCounterfactualDataset,
    audit_policy_dataset,
    build_policy_counterfactual_dataset,
    collect_policy_source_decisions,
    run_policy_counterfactual_branch,
    save_policy_dataset,
    to_ranking_dataset,
)
from game.ml.splits import DEFAULT_SEED_SPLIT_SPEC, DatasetSplit

EXPERIMENT_ID = "checkpoint4-policy-iteration-v1"
CP3_COMMIT = "6cb009d8b0876f7950c2f7b68f5e096853c69e12"
CP3_EXPERIMENT_ID = "checkpoint3-counterfactual-outcome-v1"
CP3_PRIMARY_MODEL = "outcome_mlp_seed_20261004"
CP3_PRIMARY_MODEL_CHECKSUM = "61930493025f286ef300a6229da5c266cf4637dbd7f9150aec24ad8f8f9e313c"
CP3_ROOT = Path("artifacts/ml") / CP3_EXPERIMENT_ID
CP3_MANIFEST_PATH = CP3_ROOT / "final_manifest.json"
CP3_FROZEN_CONFIG_PATH = CP3_ROOT / "frozen_config.json"
CP3_TRAIN_DATASET_PATH = CP3_ROOT / "train_dataset.json"
CP3_PRIMARY_METADATA_PATH = CP3_ROOT / f"{CP3_PRIMARY_MODEL}.json"
CP3_NORMALIZATION_PATH = CP3_ROOT / "normalization.json"
CP3_FULL_POLICY_PATH = CP3_ROOT / "exploratory_full_policy.json"

ARTIFACT_ROOT = Path("artifacts/ml") / EXPERIMENT_ID
PROTOCOL_PATH = ARTIFACT_ROOT / "predeclared_protocol.json"
COST_PROBE_PATH = ARTIFACT_ROOT / "cost_probe.json"
FEATURE_AUDIT_PATH = ARTIFACT_ROOT / "feature_schema_audit.json"
DISTRIBUTION_AUDIT_PATH = ARTIFACT_ROOT / "cp3_to_cp4_distribution_audit.json"
TRAIN_DATASET_PATH = ARTIFACT_ROOT / "train_dataset.json"
VALIDATION_DATASET_PATH = ARTIFACT_ROOT / "validation_dataset.json"
TRAIN_AUDIT_PATH = ARTIFACT_ROOT / "train_outcome_audit.json"
VALIDATION_AUDIT_PATH = ARTIFACT_ROOT / "validation_outcome_audit.json"
NORMALIZATION_PATH = ARTIFACT_ROOT / "normalization.json"
VALIDATION_METRICS_PATH = ARTIFACT_ROOT / "validation_metrics.json"
FROZEN_CONFIG_PATH = ARTIFACT_ROOT / "frozen_config.json"
TEST_ID_PATH = ARTIFACT_ROOT / "test_in_distribution.json"
TEST_TACTICAL_PATH = ARTIFACT_ROOT / "test_tactical.json"
TEST_COMPANY_PATH = ARTIFACT_ROOT / "test_company_survival.json"
TEST_METRICS_PATH = ARTIFACT_ROOT / "test_metrics.json"
ONE_STEP_PATH = ARTIFACT_ROOT / "one_step_pi1_to_pi2.json"
FULL_POLICY_PATH = ARTIFACT_ROOT / "full_policy_pi0_pi1_pi2.json"
COMPANY_REPLICATION_PATH = ARTIFACT_ROOT / "company_survival_replication.json"
FINAL_MANIFEST_PATH = ARTIFACT_ROOT / "final_manifest.json"
REPORT_PATH = ARTIFACT_ROOT / "REPORT.md"

SOURCE_POLICY_ID = f"cp3_pi1:{CP3_PRIMARY_MODEL}"
CONTINUATION_POLICY_ID = SOURCE_POLICY_ID
MAX_ROUNDS = 20
ENCOUNTER_IDS = (
    "cave_mini_boss",
    "generated_maze_glass_pack",
    "generated_maze_hunt",
    "generated_maze_hunt_guarded",
    "generated_maze_pattern_cell",
    "generated_maze_probe",
    "generated_maze_stalker",
    "maze_depth_1",
    "road_bandits",
    "shallow_cave",
    "wolf_pack",
)
TRAIN_HERO_POLICY_IDS = (
    "naive",
    "damage_race",
    "survival",
    "anti_mark",
    "conservative",
)
TACTICAL_POLICY_IDS = ("tactical",)
COMPANY_POLICY_IDS = ("company_survival",)
TRAIN_SAMPLING = SamplingSpec(states_per_encounter_policy=16)
EVAL_SAMPLING = SamplingSpec(states_per_encounter_policy=8)
LINEAR_SEED = 20261007
MLP_SEEDS = (20261007, 20261008, 20261009)
TRAINING_CONFIG = PairwiseTrainingConfig(
    learning_rate=0.005,
    batch_size=1024,
    max_epochs=60,
    early_stopping_patience=8,
    min_delta=1e-5,
    optimizer="adam",
)
MIN_TRAIN_DECISIVE_PAIRS = 500
MIN_VALIDATION_DECISIVE_PAIRS = 100
FULL_POLICY_TEST_SEEDS = tuple(range(20001, 20009))
RESERVED_FUTURE_CONFIRMATION = (30001, 30128)
COST_PROBE_CASES = (
    ("shallow_cave", "naive", 1),
    ("road_bandits", "damage_race", 2),
    ("cave_mini_boss", "survival", 3),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Checkpoint 4 bounded policy iteration.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe", help="Run the TRAIN-only π1 compute probe.")
    subparsers.add_parser("train", help="Generate π1 TRAIN/VALIDATION, train π2, freeze TEST.")
    subparsers.add_parser("test", help="Open sealed TEST and evaluate π0/π1/π2.")
    args = parser.parse_args(argv)
    if args.command == "probe":
        definitions = load_game_definitions()
        pi1, _identity = _load_verified_pi1()
        _write_json(COST_PROBE_PATH, run_cost_probe(definitions, pi1))
        return 0
    if args.command == "train":
        run_train_and_freeze()
        return 0
    if args.command == "test":
        run_frozen_test()
        return 0
    raise ValueError(f"Unknown command: {args.command}")


def run_cost_probe(
    definitions: GameDefinitions,
    pi1: FrozenOutcomeRankerPolicy,
) -> dict[str, object]:
    started = time.perf_counter()
    source_seconds = 0.0
    branch_seconds = 0.0
    source_decisions = 0
    sampled_states = 0
    candidate_branches = 0
    for encounter_id, hero_policy_id, seed in COST_PROBE_CASES:
        source_started = time.perf_counter()
        decisions = collect_policy_source_decisions(
            definitions,
            source_policy=pi1,
            source_policy_id=SOURCE_POLICY_ID,
            split=DatasetSplit.TRAIN,
            encounter_id=encounter_id,
            hero_policy_id=hero_policy_id,
            seed=seed,
            max_rounds=MAX_ROUNDS,
        )
        source_seconds += time.perf_counter() - source_started
        source_decisions += len(decisions)
        for source in decisions[:2]:
            sampled_states += 1
            branch_started = time.perf_counter()
            for candidate in source.candidates:
                result = run_policy_counterfactual_branch(
                    definitions,
                    source,
                    candidate,
                    continuation_policy=pi1,
                    continuation_policy_id=CONTINUATION_POLICY_ID,
                    max_rounds=MAX_ROUNDS,
                )
                candidate_branches += 1
                if candidate == source.source_choice:
                    _assert_control(source, result)
            branch_seconds += time.perf_counter() - branch_started
    return {
        "probe_split": "train",
        "probe_cases": [list(item) for item in COST_PROBE_CASES],
        "source_policy": SOURCE_POLICY_ID,
        "continuation_policy": CONTINUATION_POLICY_ID,
        "source_decisions_seen": source_decisions,
        "sampled_source_states": sampled_states,
        "candidate_branches": candidate_branches,
        "average_candidates_per_state": candidate_branches / sampled_states,
        "source_states_per_second": sampled_states / source_seconds,
        "candidate_branches_per_second": candidate_branches / branch_seconds,
        "wall_seconds": time.perf_counter() - started,
        "sampling_decision_already_frozen": {
            "method_id": SAMPLING_METHOD_ID,
            "train_states_per_encounter_policy": TRAIN_SAMPLING.states_per_encounter_policy,
            "validation_test_states_per_encounter_policy": EVAL_SAMPLING.states_per_encounter_policy,
        },
    }


def run_train_and_freeze() -> None:
    _verify_cp3_head()
    if any(path.exists() for path in (TEST_ID_PATH, TEST_TACTICAL_PATH, TEST_COMPANY_PATH)):
        raise ValueError("CP4 TEST artifacts already exist; refusing to retrain over opened TEST.")
    definitions = load_game_definitions()
    _validate_encounters(definitions)
    pi1, pi1_identity = _load_verified_pi1()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(PROTOCOL_PATH, _predeclared_protocol(pi1_identity))
    protocol_checksum = sha256_file(PROTOCOL_PATH)
    _write_json(FEATURE_AUDIT_PATH, _feature_audit_payload())
    probe = run_cost_probe(definitions, pi1)
    _write_json(COST_PROBE_PATH, probe)

    train_dataset = build_policy_counterfactual_dataset(
        definitions,
        source_policy=pi1,
        source_policy_id=SOURCE_POLICY_ID,
        continuation_policy=pi1,
        continuation_policy_id=CONTINUATION_POLICY_ID,
        split=DatasetSplit.TRAIN,
        encounter_ids=ENCOUNTER_IDS,
        hero_policy_ids=TRAIN_HERO_POLICY_IDS,
        seeds=DEFAULT_SEED_SPLIT_SPEC.seeds_for(DatasetSplit.TRAIN),
        sampling_spec=TRAIN_SAMPLING,
        max_rounds=MAX_ROUNDS,
    )
    train_checksum = save_policy_dataset(train_dataset, TRAIN_DATASET_PATH)
    train_audit = _audit_payload(train_dataset)
    _write_json(TRAIN_AUDIT_PATH, train_audit)

    cp3_frozen = _read_json(CP3_FROZEN_CONFIG_PATH)
    cp3_train_expected = str(cp3_frozen["datasets"]["train"]["checksum"])
    if sha256_file(CP3_TRAIN_DATASET_PATH) != cp3_train_expected:
        raise ValueError("CP3 TRAIN dataset checksum mismatch before distribution audit.")
    cp3_train = load_counterfactual_dataset(CP3_TRAIN_DATASET_PATH)
    cp3_distribution_records = reconstruct_cp3_distribution_records(definitions, cp3_train)
    distribution_audit = build_distribution_audit(
        cp3_train,
        cp3_distribution_records,
        train_dataset,
    )
    distribution_audit["schema_decision"] = {
        "schema_id": CP3_FEATURE_SCHEMA.schema_id,
        "schema_version": CP3_FEATURE_SCHEMA.version,
        "decision": "reuse_v1",
        "reason": (
            "No concrete simulator evidence established that materially different legal "
            "decision situations are observationally indistinguishable; distribution shift "
            "alone is not a feature-change criterion."
        ),
    }
    _write_json(DISTRIBUTION_AUDIT_PATH, distribution_audit)

    validation_dataset = build_policy_counterfactual_dataset(
        definitions,
        source_policy=pi1,
        source_policy_id=SOURCE_POLICY_ID,
        continuation_policy=pi1,
        continuation_policy_id=CONTINUATION_POLICY_ID,
        split=DatasetSplit.VALIDATION,
        encounter_ids=ENCOUNTER_IDS,
        hero_policy_ids=TRAIN_HERO_POLICY_IDS,
        seeds=DEFAULT_SEED_SPLIT_SPEC.seeds_for(DatasetSplit.VALIDATION),
        sampling_spec=EVAL_SAMPLING,
        max_rounds=MAX_ROUNDS,
    )
    validation_checksum = save_policy_dataset(validation_dataset, VALIDATION_DATASET_PATH)
    validation_audit = _audit_payload(validation_dataset)
    _write_json(VALIDATION_AUDIT_PATH, validation_audit)

    train_decisive = int(train_audit["overall"]["decisive_pairs"])
    validation_decisive = int(validation_audit["overall"]["decisive_pairs"])
    if (
        train_decisive < MIN_TRAIN_DECISIVE_PAIRS
        or validation_decisive < MIN_VALIDATION_DECISIVE_PAIRS
    ):
        stop_payload = {
            "status": "stopped_after_outcome_signal_audit",
            "reason": "insufficient_decisive_preference_signal",
            "train_decisive_pairs": train_decisive,
            "validation_decisive_pairs": validation_decisive,
            "minimum_train": MIN_TRAIN_DECISIVE_PAIRS,
            "minimum_validation": MIN_VALIDATION_DECISIVE_PAIRS,
            "test_accessed": False,
        }
        _write_json(ARTIFACT_ROOT / "STOPPED.json", stop_payload)
        print(json.dumps(stop_payload, sort_keys=True))
        return

    ranking_train = to_ranking_dataset(train_dataset)
    ranking_validation = to_ranking_dataset(validation_dataset)
    normalization = fit_outcome_normalization(ranking_train)
    normalization_checksum = save_outcome_normalization(normalization, NORMALIZATION_PATH)
    model_entries: list[dict[str, object]] = []
    validation_metrics: dict[str, object] = {}

    linear = train_pairwise_ranker(
        linear_scorer_spec(CP3_FEATURE_NAMES),
        ranking_train,
        ranking_validation,
        normalization,
        training_seed=LINEAR_SEED,
        config=TRAINING_CONFIG,
    )
    model_entries.append(
        _save_and_evaluate_model(
            "cp4_outcome_linear",
            linear,
            ranking_validation,
            normalization,
            normalization_checksum,
            validation_metrics,
        )
    )
    for seed in MLP_SEEDS:
        trained = train_pairwise_ranker(
            mlp_scorer_spec(CP3_FEATURE_NAMES),
            ranking_train,
            ranking_validation,
            normalization,
            training_seed=seed,
            config=TRAINING_CONFIG,
        )
        model_entries.append(
            _save_and_evaluate_model(
                f"cp4_outcome_mlp_seed_{seed}",
                trained,
                ranking_validation,
                normalization,
                normalization_checksum,
                validation_metrics,
            )
        )
    primary_model = _select_primary_mlp(validation_metrics)
    validation_metrics["primary_model"] = primary_model
    _write_json(VALIDATION_METRICS_PATH, validation_metrics)

    frozen = {
        "status": "frozen_before_test",
        "test_accessed": False,
        "experiment_id": EXPERIMENT_ID,
        "cp3_commit": CP3_COMMIT,
        "repository_head": _git_head(),
        "repository_dirty": _git_dirty(),
        "runtime_versions": {"python": sys.version.split()[0], "torch": torch.__version__},
        "pi1": pi1_identity,
        "source_policy": SOURCE_POLICY_ID,
        "continuation_policy": CONTINUATION_POLICY_ID,
        "predeclared_protocol": {
            "path": str(PROTOCOL_PATH),
            "checksum": protocol_checksum,
        },
        "feature_audit": {
            "path": str(FEATURE_AUDIT_PATH),
            "checksum": sha256_file(FEATURE_AUDIT_PATH),
        },
        "distribution_audit": {
            "path": str(DISTRIBUTION_AUDIT_PATH),
            "checksum": sha256_file(DISTRIBUTION_AUDIT_PATH),
        },
        "cost_probe": {"path": str(COST_PROBE_PATH), "result": probe},
        "observation_schema": {
            "id": CP3_FEATURE_SCHEMA.schema_id,
            "version": CP3_FEATURE_SCHEMA.version,
            "feature_count": len(CP3_FEATURE_NAMES),
            "feature_names": list(CP3_FEATURE_NAMES),
        },
        "sampling": {
            "method_id": SAMPLING_METHOD_ID,
            "train_states_per_encounter_policy": TRAIN_SAMPLING.states_per_encounter_policy,
            "validation_test_states_per_encounter_policy": EVAL_SAMPLING.states_per_encounter_policy,
        },
        "branch_method": "deterministic-prefix-replay-v1",
        "outcome_schema": "enemy-counterfactual-outcome-v1",
        "preference_rule": "win-then-loss-pareto-v1",
        "datasets": {
            "train": {"path": str(TRAIN_DATASET_PATH), "checksum": train_checksum},
            "validation": {"path": str(VALIDATION_DATASET_PATH), "checksum": validation_checksum},
        },
        "outcome_signal_audit": {
            "train": train_audit["overall"],
            "validation": validation_audit["overall"],
        },
        "normalization": {
            "path": str(NORMALIZATION_PATH),
            "checksum": normalization_checksum,
            "fit_split": "cp4_train_only",
        },
        "training": {
            "fresh_initialization": True,
            "config": TRAINING_CONFIG.to_dict(),
            "linear_seed": LINEAR_SEED,
            "mlp_seeds": list(MLP_SEEDS),
            "primary_selection": (
                "highest validation pairwise accuracy, then lowest validation loss, then lowest seed"
            ),
        },
        "model_artifacts": model_entries,
        "primary_model": primary_model,
        "validation_metrics": {
            "path": str(VALIDATION_METRICS_PATH),
            "checksum": sha256_file(VALIDATION_METRICS_PATH),
        },
        "encounter_ids": list(ENCOUNTER_IDS),
        "train_validation_hero_policy_ids": list(TRAIN_HERO_POLICY_IDS),
        "test_policy_groups": {
            "in_distribution": list(TRAIN_HERO_POLICY_IDS),
            "tactical": list(TACTICAL_POLICY_IDS),
            "company_survival": list(COMPANY_POLICY_IDS),
        },
        "splits": {
            "train": [1, 256],
            "validation": [10001, 10064],
            "test": [20001, 20128],
            "reserved_future_confirmation": list(RESERVED_FUTURE_CONFIRMATION),
        },
        "cp3_test_knowledge_warning": (
            "TEST seed IDs were previously evaluated in CP3; CP4 labels are newly generated "
            "under pi1 source/continuation and were not used before this freeze."
        ),
        "full_policy_protocol": {
            "policies": ["pi0_heuristic", "pi1_cp3", "pi2_cp4"],
            "seeds": list(FULL_POLICY_TEST_SEEDS),
            "paired_uncertainty": "wilson-95%-per-policy plus exact paired McNemar discordance",
        },
        "tactical_telemetry": [
            "mark",
            "guard",
            "grab",
            "bite",
            "focus/repeat targeting",
            "support disruption",
            "target switching",
            "skill distribution",
            "effort drain",
            "kill/down conversion",
        ],
        "implementation_hashes": _implementation_hashes(),
        "simulator_hashes": _simulator_hashes(),
        "content_hashes": _content_hashes(),
    }
    _write_json(FROZEN_CONFIG_PATH, frozen)
    print(f"FROZEN_CONFIG={FROZEN_CONFIG_PATH}")
    print(f"FROZEN_CONFIG_SHA256={sha256_file(FROZEN_CONFIG_PATH)}")
    print(f"TRAIN_DECISIVE_PAIRS={train_decisive}")
    print(f"VALIDATION_DECISIVE_PAIRS={validation_decisive}")
    print(f"PRIMARY_MODEL={primary_model}")
    print("TEST_ACCESSED=false")


def run_frozen_test() -> None:
    if not FROZEN_CONFIG_PATH.exists():
        raise ValueError("CP4 frozen configuration is required before TEST.")
    frozen_checksum_before = sha256_file(FROZEN_CONFIG_PATH)
    frozen = _read_json(FROZEN_CONFIG_PATH)
    _validate_frozen_config(frozen)
    definitions = load_game_definitions()
    pi1, pi1_identity = _load_verified_pi1()
    if pi1_identity["model_checksum"] != frozen["pi1"]["model_checksum"]:
        raise ValueError("Source policy identity changed after CP4 freeze.")
    normalization = load_outcome_normalization(NORMALIZATION_PATH)
    models = _load_frozen_models(frozen)
    pi2_name = str(frozen["primary_model"])
    pi2_model = models[pi2_name]
    pi2 = FrozenOutcomeRankerPolicy(
        pi2_model,
        normalization,
        policy_id=f"cp4_pi2:{pi2_name}",
    )

    test_seeds = DEFAULT_SEED_SPLIT_SPEC.seeds_for(DatasetSplit.TEST)
    datasets = _build_test_datasets(definitions, pi1, test_seeds)
    dataset_paths = {
        "in_distribution": TEST_ID_PATH,
        "tactical": TEST_TACTICAL_PATH,
        "company_survival": TEST_COMPANY_PATH,
    }
    dataset_checksums = {
        name: save_policy_dataset(dataset, dataset_paths[name])
        for name, dataset in datasets.items()
    }

    test_metrics: dict[str, Any] = {
        "frozen_config_checksum": frozen_checksum_before,
        "dataset_checksums": dataset_checksums,
    }
    one_step: dict[str, Any] = {}
    for group_name, dataset in datasets.items():
        ranking_dataset = to_ranking_dataset(dataset)
        model_metrics: dict[str, Any] = {}
        for model_name, model in models.items():
            loss, accuracy = evaluate_pairwise(model, ranking_dataset, normalization)
            model_metrics[model_name] = {
                "decisive_pairs": _audit_payload(dataset)["overall"]["decisive_pairs"],
                "pairwise_accuracy": accuracy,
                "pairwise_loss": loss,
                "by_encounter": grouped_pairwise_accuracy(
                    model, ranking_dataset, normalization, group_by="encounter"
                ),
                "by_hero_policy": grouped_pairwise_accuracy(
                    model, ranking_dataset, normalization, group_by="hero_policy"
                ),
                "by_enemy_class": grouped_pairwise_accuracy(
                    model, ranking_dataset, normalization, group_by="enemy_class"
                ),
            }
        test_metrics[group_name] = {
            "audit": _audit_payload(dataset),
            "models": model_metrics,
        }
        one_step[group_name] = _one_step_pi1_to_pi2(dataset, pi2_model, normalization)
    _write_json(TEST_METRICS_PATH, test_metrics)
    _write_json(ONE_STEP_PATH, one_step)

    full_policy = _run_full_policy_comparison(definitions, pi1=pi1, pi2=pi2)
    _write_json(FULL_POLICY_PATH, full_policy)
    company_replication = _company_survival_replication(full_policy)
    _write_json(COMPANY_REPLICATION_PATH, company_replication)

    if sha256_file(FROZEN_CONFIG_PATH) != frozen_checksum_before:
        raise ValueError("Frozen CP4 configuration changed during TEST.")
    final_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "cp3_commit": CP3_COMMIT,
        "pi1": pi1_identity,
        "pi2": {
            "name": pi2_name,
            "model_checksum": _model_entry(frozen, pi2_name)["model_checksum"],
        },
        "frozen_config": {
            "path": str(FROZEN_CONFIG_PATH),
            "checksum": frozen_checksum_before,
            "unchanged_during_test": True,
        },
        "test_dataset_checksums": dataset_checksums,
        "test_metrics": {
            "path": str(TEST_METRICS_PATH),
            "checksum": sha256_file(TEST_METRICS_PATH),
        },
        "one_step": {"path": str(ONE_STEP_PATH), "checksum": sha256_file(ONE_STEP_PATH)},
        "full_policy": {
            "path": str(FULL_POLICY_PATH),
            "checksum": sha256_file(FULL_POLICY_PATH),
        },
        "company_survival_replication": {
            "path": str(COMPANY_REPLICATION_PATH),
            "checksum": sha256_file(COMPANY_REPLICATION_PATH),
        },
        "model_artifacts": frozen["model_artifacts"],
        "reserved_future_confirmation": list(RESERVED_FUTURE_CONFIRMATION),
        "reserved_future_confirmation_evaluated": False,
        "test_tuning_performed": False,
    }
    _write_json(FINAL_MANIFEST_PATH, final_manifest)
    REPORT_PATH.write_text(
        _build_report(
            frozen,
            test_metrics,
            one_step,
            full_policy,
            company_replication,
            final_manifest,
        ),
        encoding="utf-8",
    )
    print(f"FINAL_MANIFEST={FINAL_MANIFEST_PATH}")
    print(f"FROZEN_CONFIG_SHA256={frozen_checksum_before}")
    print(f"PRIMARY_PI2={pi2_name}")
    print("TEST_TUNING_PERFORMED=false")
    print("RESERVED_FUTURE_CONFIRMATION_EVALUATED=false")


def _build_test_datasets(
    definitions: GameDefinitions,
    pi1: FrozenOutcomeRankerPolicy,
    test_seeds: tuple[int, ...],
) -> dict[str, PolicyCounterfactualDataset]:
    groups = {
        "in_distribution": TRAIN_HERO_POLICY_IDS,
        "tactical": TACTICAL_POLICY_IDS,
        "company_survival": COMPANY_POLICY_IDS,
    }
    return {
        name: build_policy_counterfactual_dataset(
            definitions,
            source_policy=pi1,
            source_policy_id=SOURCE_POLICY_ID,
            continuation_policy=pi1,
            continuation_policy_id=CONTINUATION_POLICY_ID,
            split=DatasetSplit.TEST,
            encounter_ids=ENCOUNTER_IDS,
            hero_policy_ids=policy_ids,
            seeds=test_seeds,
            sampling_spec=EVAL_SAMPLING,
            max_rounds=MAX_ROUNDS,
        )
        for name, policy_ids in groups.items()
    }


def _one_step_pi1_to_pi2(
    dataset: PolicyCounterfactualDataset,
    pi2_model: torch.nn.Module,
    normalization: Any,
) -> dict[str, Any]:
    ranking = to_ranking_dataset(dataset)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(dataset.records):
        pi2_index = choose_candidate_index(pi2_model, index, ranking, normalization)
        pi1_outcome = record.candidates[record.source_policy_index].outcome
        pi2_outcome = record.candidates[pi2_index].outcome
        preference = compare_outcomes(pi2_outcome, pi1_outcome)
        rows.append(
            {
                "encounter_id": record.encounter_id,
                "hero_policy_id": record.hero_policy_id,
                "enemy_class_id": record.enemy_class_id,
                "changed_action": pi2_index != record.source_policy_index,
                "preference": preference.value,
                "pi1_win": pi1_outcome.enemy_win,
                "pi2_win": pi2_outcome.enemy_win,
                "hp_diff": pi2_outcome.final_hero_hp - pi1_outcome.final_hero_hp,
                "down_diff": (pi2_outcome.terminal_hero_downs - pi1_outcome.terminal_hero_downs),
                "death_diff": pi2_outcome.hero_deaths - pi1_outcome.hero_deaths,
                "wound_diff": pi2_outcome.mortal_wounds - pi1_outcome.mortal_wounds,
                "effort_drain_diff": (
                    pi2_outcome.enemy_caused_effort_drain - pi1_outcome.enemy_caused_effort_drain
                ),
            }
        )
    return {
        "overall": _summarize_one_step_rows(rows),
        "by_encounter": _group_one_step(rows, "encounter_id"),
        "by_hero_policy": _group_one_step(rows, "hero_policy_id"),
        "by_enemy_class": _group_one_step(rows, "enemy_class_id"),
    }


def _group_one_step(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    values: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        values.setdefault(str(row[key]), []).append(row)
    return {name: _summarize_one_step_rows(items) for name, items in sorted(values.items())}


def _summarize_one_step_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    preferences = {item.value: 0 for item in OutcomePreference}
    for row in rows:
        preferences[str(row["preference"])] += 1
    count = len(rows)
    return {
        "source_states": count,
        "changed_action_states": sum(bool(row["changed_action"]) for row in rows),
        "preference_counts_pi2_vs_pi1": preferences,
        "pi1_enemy_win_rate": sum(bool(row["pi1_win"]) for row in rows) / count if count else 0.0,
        "pi2_enemy_win_rate": sum(bool(row["pi2_win"]) for row in rows) / count if count else 0.0,
        "loss_to_win_flips": sum(
            (not bool(row["pi1_win"])) and bool(row["pi2_win"]) for row in rows
        ),
        "win_to_loss_regressions": sum(
            bool(row["pi1_win"]) and (not bool(row["pi2_win"])) for row in rows
        ),
        "mean_pi2_minus_pi1": {
            "final_hero_hp": mean(float(row["hp_diff"]) for row in rows) if rows else 0.0,
            "terminal_hero_downs": mean(float(row["down_diff"]) for row in rows) if rows else 0.0,
            "hero_deaths": mean(float(row["death_diff"]) for row in rows) if rows else 0.0,
            "mortal_wounds": mean(float(row["wound_diff"]) for row in rows) if rows else 0.0,
            "enemy_caused_effort_drain": mean(float(row["effort_drain_diff"]) for row in rows)
            if rows
            else 0.0,
        },
    }


def _run_full_policy_comparison(
    definitions: GameDefinitions,
    *,
    pi1: FrozenOutcomeRankerPolicy,
    pi2: FrozenOutcomeRankerPolicy,
) -> dict[str, Any]:
    policies: dict[str, EnemyDecisionPolicy] = {
        "pi0_heuristic": HeuristicEnemyDecisionPolicy(),
        "pi1_cp3": pi1,
        "pi2_cp4": pi2,
    }
    groups = {
        "in_distribution": TRAIN_HERO_POLICY_IDS,
        "tactical": TACTICAL_POLICY_IDS,
        "company_survival": COMPANY_POLICY_IDS,
    }
    result: dict[str, Any] = {
        "label": "BOUNDED HELD-OUT FULL-POLICY COMPARISON",
        "seeds": list(FULL_POLICY_TEST_SEEDS),
        "policies": list(policies),
        "reserved_future_confirmation_evaluated": False,
    }
    for group_name, hero_policy_ids in groups.items():
        rows: dict[str, list[dict[str, Any]]] = {name: [] for name in policies}
        for encounter_id in ENCOUNTER_IDS:
            for hero_policy_id in hero_policy_ids:
                for seed in FULL_POLICY_TEST_SEEDS:
                    case_id = f"{encounter_id}|{hero_policy_id}|{seed}"
                    for policy_name, policy in policies.items():
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
                                hero_policy_id,
                                encounter_id=encounter_id,
                                seed=seed,
                            ),
                            enemy_wait_mode="none",
                            enemy_movement_mode="recovery_only",
                        )
                        outcome = outcome_from_episode(state, episode)
                        switches, repeats = _target_transition_counts(episode.records)
                        rows[policy_name].append(
                            {
                                "case_id": case_id,
                                "encounter_id": encounter_id,
                                "hero_policy_id": hero_policy_id,
                                "seed": seed,
                                "enemy_win": int(outcome.enemy_win),
                                "final_hero_hp": outcome.final_hero_hp,
                                "final_hero_effort": outcome.final_hero_effort,
                                "terminal_hero_downs": outcome.terminal_hero_downs,
                                "hero_deaths": outcome.hero_deaths,
                                "mortal_wounds": outcome.mortal_wounds,
                                "enemy_caused_effort_drain": outcome.enemy_caused_effort_drain,
                                "rounds": outcome.rounds_elapsed,
                                "enemy_actions": episode.metrics.enemy_decisions,
                                "hero_damage": episode.metrics.total_hero_damage,
                                "marks_applied": episode.metrics.marks_applied,
                                "mark_multi_hit_focus": episode.metrics.mark_flow.multi_hit_focus,
                                "marked_downs": episode.metrics.mark_flow.marked_downs,
                                "guard_actions": episode.metrics.guard_actions,
                                "grab_uses": episode.metrics.boss_sequence.grab_uses,
                                "bite_uses": episode.metrics.boss_sequence.bite_uses,
                                "bite_downs": episode.metrics.boss_sequence.bite_downs,
                                "support_grabs": episode.metrics.boss_targeting.support_grabs,
                                "support_grab_downs": episode.metrics.boss_targeting.support_grab_downs,
                                "target_switches": switches,
                                "repeat_target_transitions": repeats,
                                "skill_uses": dict(episode.metrics.skill_uses),
                            }
                        )
        summaries = {name: _summarize_rollout_rows(items) for name, items in rows.items()}
        uncertainty = {
            "pi1_minus_pi0": _paired_win_uncertainty(rows["pi0_heuristic"], rows["pi1_cp3"]),
            "pi2_minus_pi1": _paired_win_uncertainty(rows["pi1_cp3"], rows["pi2_cp4"]),
            "pi2_minus_pi0": _paired_win_uncertainty(rows["pi0_heuristic"], rows["pi2_cp4"]),
        }
        result[group_name] = {
            "summaries": summaries,
            "paired_win_uncertainty": uncertainty,
        }
    return result


def _summarize_rollout_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"case_count": 0}
    scalar_keys = (
        "enemy_win",
        "final_hero_hp",
        "final_hero_effort",
        "terminal_hero_downs",
        "hero_deaths",
        "mortal_wounds",
        "enemy_caused_effort_drain",
        "rounds",
        "enemy_actions",
        "hero_damage",
        "marks_applied",
        "mark_multi_hit_focus",
        "marked_downs",
        "guard_actions",
        "grab_uses",
        "bite_uses",
        "bite_downs",
        "support_grabs",
        "support_grab_downs",
        "target_switches",
        "repeat_target_transitions",
    )
    skill_totals: dict[str, int] = {}
    for row in rows:
        for skill_id, count in row["skill_uses"].items():
            skill_totals[skill_id] = skill_totals.get(skill_id, 0) + int(count)
    wins = sum(int(row["enemy_win"]) for row in rows)
    return {
        "case_count": len(rows),
        **{key: mean(float(row[key]) for row in rows) for key in scalar_keys},
        "enemy_win_count": wins,
        "enemy_win_rate_wilson_95": _wilson_interval(wins, len(rows)),
        "mean_skill_uses_per_case": {
            skill_id: count / len(rows) for skill_id, count in sorted(skill_totals.items())
        },
    }


def _paired_win_uncertainty(
    baseline: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(baseline) != len(comparison):
        raise ValueError("Paired rollout comparison requires equal case counts.")
    baseline_by_case = {str(row["case_id"]): row for row in baseline}
    comparison_by_case = {str(row["case_id"]): row for row in comparison}
    if baseline_by_case.keys() != comparison_by_case.keys():
        raise ValueError("Paired rollout case identities differ.")
    baseline_only = 0
    comparison_only = 0
    both_win = 0
    both_loss = 0
    for case_id in sorted(baseline_by_case):
        first = bool(baseline_by_case[case_id]["enemy_win"])
        second = bool(comparison_by_case[case_id]["enemy_win"])
        if first and second:
            both_win += 1
        elif first:
            baseline_only += 1
        elif second:
            comparison_only += 1
        else:
            both_loss += 1
    count = len(baseline)
    difference = (comparison_only - baseline_only) / count if count else 0.0
    return {
        "case_count": count,
        "both_win": both_win,
        "both_loss": both_loss,
        "baseline_only_win": baseline_only,
        "comparison_only_win": comparison_only,
        "win_rate_difference": difference,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(baseline_only, comparison_only),
    }


def _mcnemar_exact(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    smaller = min(first_only, second_only)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _wilson_interval(successes: int, count: int) -> list[float]:
    if count == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count))
        / denominator
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def _target_transition_counts(records: Sequence[Any]) -> tuple[int, int]:
    last_target: dict[str, str] = {}
    switches = 0
    repeats = 0
    for record in records:
        enemy_id = str(record.enemy_id)
        target_id = str(record.chosen_target_id)
        if enemy_id in last_target:
            if last_target[enemy_id] == target_id:
                repeats += 1
            else:
                switches += 1
        last_target[enemy_id] = target_id
    return switches, repeats


def _company_survival_replication(full_policy: dict[str, Any]) -> dict[str, Any]:
    cp3 = _read_json(CP3_FULL_POLICY_PATH)["company_survival"]
    current = full_policy["company_survival"]
    summaries = current["summaries"]
    cp3_pi0 = float(cp3["heuristic"]["enemy_win"])
    cp3_pi1 = float(cp3["learned"]["enemy_win"])
    cp4_pi0 = float(summaries["pi0_heuristic"]["enemy_win"])
    cp4_pi1 = float(summaries["pi1_cp3"]["enemy_win"])
    cp4_pi2 = float(summaries["pi2_cp4"]["enemy_win"])
    return {
        "cp3_reference": {
            "pi0_heuristic_enemy_win_rate": cp3_pi0,
            "pi1_cp3_enemy_win_rate": cp3_pi1,
        },
        "cp4_clean_evaluation": {
            "pi0_heuristic_enemy_win_rate": cp4_pi0,
            "pi1_cp3_enemy_win_rate": cp4_pi1,
            "pi2_cp4_enemy_win_rate": cp4_pi2,
            "pi0_wilson_95": summaries["pi0_heuristic"]["enemy_win_rate_wilson_95"],
            "pi1_wilson_95": summaries["pi1_cp3"]["enemy_win_rate_wilson_95"],
            "pi2_wilson_95": summaries["pi2_cp4"]["enemy_win_rate_wilson_95"],
        },
        "deterministic_replication": {
            "pi0_exact_match": math.isclose(cp3_pi0, cp4_pi0, abs_tol=1e-15),
            "pi1_exact_match": math.isclose(cp3_pi1, cp4_pi1, abs_tol=1e-15),
        },
        "paired_uncertainty": current["paired_win_uncertainty"],
        "success_not_defined_as_exceeding_cp3_pi1": True,
    }


def _predeclared_protocol(pi1_identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "scientific_question": (
            "Does counterfactual policy improvement remain beneficial when source states "
            "and continuation are generated by frozen CP3 pi1?"
        ),
        "cp3_commit": CP3_COMMIT,
        "pi1": pi1_identity,
        "source_policy": SOURCE_POLICY_ID,
        "continuation_policy": CONTINUATION_POLICY_ID,
        "one_policy_iteration_only": True,
        "observation_schema": {
            "id": CP3_FEATURE_SCHEMA.schema_id,
            "version": CP3_FEATURE_SCHEMA.version,
        },
        "sampling": {
            "method_id": SAMPLING_METHOD_ID,
            "outcome_blind": True,
            "train_states_per_encounter_policy": TRAIN_SAMPLING.states_per_encounter_policy,
            "validation_test_states_per_encounter_policy": EVAL_SAMPLING.states_per_encounter_policy,
        },
        "splits": {
            "train": [1, 256],
            "validation": [10001, 10064],
            "test": [20001, 20128],
            "reserved_future_confirmation": list(RESERVED_FUTURE_CONFIRMATION),
        },
        "reserved_future_confirmation_evaluated": False,
        "cp3_test_seed_warning": (
            "TEST seed IDs have aggregate CP3 history; CP4 generates new pi1-source/pi1-continuation labels."
        ),
        "encounter_ids": list(ENCOUNTER_IDS),
        "train_validation_policies": list(TRAIN_HERO_POLICY_IDS),
        "test_groups": {
            "in_distribution": list(TRAIN_HERO_POLICY_IDS),
            "tactical": list(TACTICAL_POLICY_IDS),
            "company_survival": list(COMPANY_POLICY_IDS),
        },
        "outcome_rule": "win-then-loss-pareto-v1",
        "loss_dimensions": [
            "hero_deaths",
            "terminal_hero_downs",
            "mortal_wounds",
            "final_hero_hp",
            "enemy_caused_effort_drain",
        ],
        "two_enemy_wins": "tie",
        "crossing_loss_dimensions": "incomparable",
        "models": {
            "linear_seed": LINEAR_SEED,
            "mlp_seeds": list(MLP_SEEDS),
            "mlp_shape": [81, 64, 32, 1],
            "fresh_initialization": True,
        },
        "training": TRAINING_CONFIG.to_dict(),
        "primary_selection": ("validation pairwise accuracy desc, validation loss asc, seed asc"),
        "minimum_signal_gate": {
            "train_decisive_pairs": MIN_TRAIN_DECISIVE_PAIRS,
            "validation_decisive_pairs": MIN_VALIDATION_DECISIVE_PAIRS,
        },
        "full_policy": {
            "policies": ["pi0_heuristic", "pi1_cp3", "pi2_cp4"],
            "seeds": list(FULL_POLICY_TEST_SEEDS),
            "uncertainty": "wilson 95% plus exact paired McNemar discordance",
        },
        "test_accessed": False,
    }


def _feature_audit_payload() -> dict[str, Any]:
    return {
        "schema_decision_before_training": "start_from_cp3_v1",
        "schema_id": CP3_FEATURE_SCHEMA.schema_id,
        "schema_version": CP3_FEATURE_SCHEMA.version,
        "feature_count": len(CP3_FEATURE_NAMES),
        "neutral_factual": list(CP3_NEUTRAL_FACTUAL_FEATURE_NAMES),
        "derived_factual": list(CP3_DERIVED_FEATURE_NAMES),
        "prescriptive_included": [],
        "cp2_feature_audit": [asdict(item) for item in CP2_FEATURE_AUDIT],
        "version_change_rule": (
            "Only concrete simulator evidence of materially different legal decision situations "
            "remaining observationally indistinguishable may justify a schema revision."
        ),
        "performance_degradation_is_not_a_feature_change_rule": True,
    }


def _audit_payload(dataset: PolicyCounterfactualDataset) -> dict[str, Any]:
    return {
        "available_source_states": dataset.available_source_states,
        "sampled_source_states": len(dataset.records),
        "overall": asdict(audit_policy_dataset(dataset)),
        "by_encounter": _audits_by(dataset, "encounter"),
        "by_hero_policy": _audits_by(dataset, "hero_policy"),
        "by_enemy_class": _audits_by(dataset, "enemy_class"),
    }


def _audits_by(
    dataset: PolicyCounterfactualDataset,
    grouping: str,
) -> dict[str, dict[str, int]]:
    keys: dict[str, list[Any]] = {}
    for record in dataset.records:
        if grouping == "encounter":
            key = record.encounter_id
        elif grouping == "hero_policy":
            key = record.hero_policy_id
        elif grouping == "enemy_class":
            key = record.enemy_class_id
        else:
            raise ValueError(f"Unknown audit grouping: {grouping}")
        keys.setdefault(key, []).append(record)
    output: dict[str, dict[str, int]] = {}
    for key, records in sorted(keys.items()):
        subset = PolicyCounterfactualDataset(
            split=dataset.split,
            source_policy_id=dataset.source_policy_id,
            continuation_policy_id=dataset.continuation_policy_id,
            sampling_spec=dataset.sampling_spec,
            encounter_ids=dataset.encounter_ids,
            hero_policy_ids=dataset.hero_policy_ids,
            seed_count=dataset.seed_count,
            available_source_states=len(records),
            records=tuple(records),
        )
        output[key] = {
            name: int(value) for name, value in asdict(audit_policy_dataset(subset)).items()
        }
    return output


def _save_and_evaluate_model(
    name: str,
    trained: Any,
    validation_dataset: Any,
    normalization: Any,
    normalization_checksum: str,
    validation_metrics: dict[str, object],
) -> dict[str, object]:
    metadata_path = ARTIFACT_ROOT / f"{name}.json"
    metadata = save_outcome_ranker(
        trained,
        metadata_path,
        normalization_checksum=normalization_checksum,
    )
    loss, accuracy = evaluate_pairwise(trained.model, validation_dataset, normalization)
    validation_metrics[name] = {
        "decisive_pairs": len(
            [
                1
                for record in validation_dataset.records
                for first in range(len(record.candidates))
                for second in range(first + 1, len(record.candidates))
                if compare_outcomes(
                    record.candidates[first].outcome,
                    record.candidates[second].outcome,
                )
                in {OutcomePreference.FIRST, OutcomePreference.SECOND}
            ]
        ),
        "pairwise_accuracy": accuracy,
        "pairwise_loss": loss,
        "by_encounter": grouped_pairwise_accuracy(
            trained.model, validation_dataset, normalization, group_by="encounter"
        ),
        "by_hero_policy": grouped_pairwise_accuracy(
            trained.model, validation_dataset, normalization, group_by="hero_policy"
        ),
        "by_enemy_class": grouped_pairwise_accuracy(
            trained.model, validation_dataset, normalization, group_by="enemy_class"
        ),
    }
    return {
        "name": name,
        "training_seed": trained.result.training_seed,
        "selected_epoch": trained.result.selected_epoch,
        "metadata_path": str(metadata_path),
        "metadata_checksum": sha256_file(metadata_path),
        "model_file": str(metadata["model_file"]),
        "model_checksum": str(metadata["model_checksum"]),
    }


def _select_primary_mlp(validation_metrics: dict[str, object]) -> str:
    candidates: list[tuple[float, float, int, str]] = []
    for name, raw_metrics in validation_metrics.items():
        if not name.startswith("cp4_outcome_mlp_seed_"):
            continue
        metrics = raw_metrics
        if not isinstance(metrics, dict):
            continue
        seed = int(name.rsplit("_", 1)[1])
        candidates.append(
            (
                -float(metrics["pairwise_accuracy"]),
                float(metrics["pairwise_loss"]),
                seed,
                name,
            )
        )
    if not candidates:
        raise ValueError("No CP4 MLP validation candidates were available.")
    return min(candidates)[3]


def _require_sha256(path: Path, expected: str, label: str) -> None:
    if not path.exists() or sha256_file(path) != expected:
        raise ValueError(f"{label} is unavailable or checksum-mismatched.")


def _load_verified_pi1() -> tuple[FrozenOutcomeRankerPolicy, dict[str, Any]]:
    if not CP3_MANIFEST_PATH.exists():
        raise ValueError("CP3 final manifest is required for CP4 source-policy provenance.")
    manifest = _read_json(CP3_MANIFEST_PATH)
    if manifest.get("primary_model") != CP3_PRIMARY_MODEL:
        raise ValueError("CP3 manifest primary model identity does not match expected pi1.")
    entries = manifest.get("model_artifacts", [])
    matching = [item for item in entries if item.get("name") == CP3_PRIMARY_MODEL]
    if len(matching) != 1:
        raise ValueError("CP3 manifest does not contain exactly one primary model entry.")
    entry = matching[0]
    manifest_checksum = str(entry["model_checksum"])
    if manifest_checksum != CP3_PRIMARY_MODEL_CHECKSUM:
        raise ValueError("CP3 manifest primary model checksum differs from reviewed result.")
    model_path = CP3_ROOT / str(entry["model_file"])
    _require_sha256(model_path, manifest_checksum, "CP3 primary model artifact")
    _require_sha256(
        CP3_PRIMARY_METADATA_PATH,
        str(entry["metadata_checksum"]),
        "CP3 primary model metadata",
    )
    loaded = load_outcome_ranker(CP3_PRIMARY_METADATA_PATH)
    normalization_checksum = str(loaded.metadata["normalization_checksum"])
    if sha256_file(CP3_NORMALIZATION_PATH) != normalization_checksum:
        raise ValueError("CP3 normalization checksum mismatch before pi1 use.")
    normalization = load_outcome_normalization(CP3_NORMALIZATION_PATH)
    frozen = _read_json(CP3_FROZEN_CONFIG_PATH)
    if str(frozen["normalization"]["checksum"]) != normalization_checksum:
        raise ValueError("CP3 frozen config normalization identity differs from pi1 metadata.")
    identity = {
        "experiment_id": CP3_EXPERIMENT_ID,
        "model_name": CP3_PRIMARY_MODEL,
        "model_path": str(model_path),
        "model_checksum": manifest_checksum,
        "metadata_path": str(CP3_PRIMARY_METADATA_PATH),
        "metadata_checksum": str(entry["metadata_checksum"]),
        "normalization_path": str(CP3_NORMALIZATION_PATH),
        "normalization_checksum": normalization_checksum,
        "cp3_frozen_config_checksum": sha256_file(CP3_FROZEN_CONFIG_PATH),
    }
    return (
        FrozenOutcomeRankerPolicy(
            loaded.model,
            normalization,
            policy_id=SOURCE_POLICY_ID,
        ),
        identity,
    )


def _load_frozen_models(frozen: dict[str, Any]) -> dict[str, torch.nn.Module]:
    output: dict[str, torch.nn.Module] = {}
    expected_normalization = str(frozen["normalization"]["checksum"])
    for entry in frozen["model_artifacts"]:
        metadata_path = Path(str(entry["metadata_path"]))
        if sha256_file(metadata_path) != str(entry["metadata_checksum"]):
            raise ValueError(f"Model metadata checksum changed: {entry['name']}")
        loaded = load_outcome_ranker(metadata_path)
        if str(loaded.metadata["normalization_checksum"]) != expected_normalization:
            raise ValueError(f"Model normalization changed: {entry['name']}")
        output[str(entry["name"])] = loaded.model
    return output


def _validate_frozen_config(frozen: dict[str, Any]) -> None:
    if frozen.get("status") != "frozen_before_test" or frozen.get("test_accessed") is not False:
        raise ValueError("CP4 frozen config does not prove TEST was sealed before evaluation.")
    if frozen.get("cp3_commit") != CP3_COMMIT:
        raise ValueError("CP4 frozen config CP3 baseline changed.")
    if frozen.get("source_policy") != SOURCE_POLICY_ID:
        raise ValueError("CP4 source policy changed after freeze.")
    if frozen.get("continuation_policy") != CONTINUATION_POLICY_ID:
        raise ValueError("CP4 continuation policy changed after freeze.")
    if frozen.get("observation_schema", {}).get("id") != CP3_FEATURE_SCHEMA.schema_id:
        raise ValueError("CP4 observation schema changed after freeze.")
    pi1, identity = _load_verified_pi1()
    del pi1
    if identity["model_checksum"] != frozen["pi1"]["model_checksum"]:
        raise ValueError("CP4 pi1 checksum changed after freeze.")
    checks = (
        (frozen["predeclared_protocol"], "protocol"),
        (frozen["feature_audit"], "feature audit"),
        (frozen["distribution_audit"], "distribution audit"),
        (frozen["datasets"]["train"], "TRAIN dataset"),
        (frozen["datasets"]["validation"], "VALIDATION dataset"),
        (frozen["normalization"], "normalization"),
        (frozen["validation_metrics"], "validation metrics"),
    )
    for item, label in checks:
        path = Path(str(item["path"]))
        if not path.exists() or sha256_file(path) != str(item["checksum"]):
            raise ValueError(f"Frozen {label} checksum changed.")
    if frozen["implementation_hashes"] != _implementation_hashes():
        raise ValueError("CP4 implementation changed after TEST freeze.")
    if frozen["simulator_hashes"] != _simulator_hashes():
        raise ValueError("Simulator changed after CP4 TEST freeze.")
    if frozen["content_hashes"] != _content_hashes():
        raise ValueError("Content changed after CP4 TEST freeze.")
    _load_frozen_models(frozen)


def _model_entry(frozen: dict[str, Any], name: str) -> dict[str, Any]:
    matching = [item for item in frozen["model_artifacts"] if item["name"] == name]
    if len(matching) != 1:
        raise ValueError(f"Frozen model entry not unique: {name}")
    return matching[0]


def _assert_control(source: Any, result: Any) -> None:
    if result.outcome != source.baseline_outcome:
        raise ValueError("Source-policy control branch changed terminal outcome.")
    if result.final_state_digest != source.baseline_final_state_digest:
        raise ValueError("Source-policy control branch changed final state.")
    if result.episode_digest != source.baseline_episode_digest:
        raise ValueError("Source-policy control branch changed episode digest.")


def _verify_cp3_head() -> None:
    if _git_head() != CP3_COMMIT:
        raise ValueError(f"CP4 must begin from committed CP3 baseline {CP3_COMMIT}.")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_dirty() -> bool:
    return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())


def _implementation_hashes() -> dict[str, str]:
    return _hash_paths(
        (
            Path("src/game/dev/ml_cp4.py"),
            Path("src/game/ml/policy_iteration.py"),
            Path("src/game/ml/distribution_audit.py"),
            Path("src/game/ml/outcome_features.py"),
            Path("src/game/ml/outcome_dataset.py"),
            Path("src/game/ml/outcome_ranking.py"),
            Path("src/game/ml/models.py"),
            Path("pyproject.toml"),
        )
    )


def _simulator_hashes() -> dict[str, str]:
    return _hash_paths(
        (
            Path("src/game/combat/actions.py"),
            Path("src/game/combat/combat_state.py"),
            Path("src/game/combat/enemy_decision.py"),
            Path("src/game/combat/enemy_learning.py"),
            Path("src/game/core/rng.py"),
        )
    )


def _content_hashes() -> dict[str, str]:
    return _hash_paths(tuple(sorted(Path("data").glob("*.yaml"))))


def _hash_paths(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def _validate_encounters(definitions: GameDefinitions) -> None:
    missing = [
        encounter_id for encounter_id in ENCOUNTER_IDS if encounter_id not in definitions.encounters
    ]
    if missing:
        raise ValueError(f"Missing CP4 encounters: {missing}")


def _build_report(
    frozen: dict[str, Any],
    test_metrics: dict[str, Any],
    one_step: dict[str, Any],
    full_policy: dict[str, Any],
    company_replication: dict[str, Any],
    final_manifest: dict[str, Any],
) -> str:
    train_audit = _read_json(TRAIN_AUDIT_PATH)
    validation_audit = _read_json(VALIDATION_AUDIT_PATH)
    distribution = _read_json(DISTRIBUTION_AUDIT_PATH)
    validation = _read_json(VALIDATION_METRICS_PATH)
    lines = [
        "# The Charter — Checkpoint 4 Bounded Iterative Policy Improvement",
        "",
        "## A. Provenance",
        f"- CP3 commit: `{CP3_COMMIT}`",
        f"- Experiment: `{EXPERIMENT_ID}`",
        f"- π1: `{frozen['pi1']['model_name']}` / `{frozen['pi1']['model_checksum']}`",
        f"- Runtime: Python `{frozen['runtime_versions']['python']}`, Torch `{frozen['runtime_versions']['torch']}`.",
        "",
        "## B. CP3 → CP4 distribution shift",
        _distribution_report_line(distribution),
        "- Full deterministic feature diagnostics are persisted in the distribution-audit artifact; no single synthetic shift score is used.",
        "",
        "## C. Counterfactual method",
        "- Source trajectories are generated by frozen π1; after exactly one forced legal action, continuation returns to the same frozen π1.",
        "- Exact decision-state digest, RNG digest, legal-candidate collection, terminal control outcome, final state, and episode digest are checked.",
        "",
        "## D. CP4 dataset",
        f"- TRAIN: {train_audit['overall']['source_states']} states / {train_audit['overall']['candidate_branches']} branches.",
        f"- VALIDATION: {validation_audit['overall']['source_states']} states / {validation_audit['overall']['candidate_branches']} branches.",
        f"- Sampling: `{SAMPLING_METHOD_ID}`, {TRAIN_SAMPLING.states_per_encounter_policy} TRAIN and {EVAL_SAMPLING.states_per_encounter_policy} evaluation states per encounter/policy stratum.",
        "",
        "## E. Outcome-signal audit",
        _outcome_signal_line("TRAIN", train_audit["overall"]),
        _outcome_signal_line("VALIDATION", validation_audit["overall"]),
        "",
        "## F. Train / validation",
        f"- Selected π2: `{frozen['primary_model']}`.",
        *[
            f"- {name}: accuracy {float(metrics['pairwise_accuracy']):.2%}, loss {float(metrics['pairwise_loss']):.4f}."
            for name, metrics in validation.items()
            if isinstance(metrics, dict) and "pairwise_accuracy" in metrics
        ],
        "",
        "## G. Frozen TEST configuration",
        f"- Path: `{FROZEN_CONFIG_PATH}`",
        f"- Checksum: `{sha256_file(FROZEN_CONFIG_PATH)}`",
        "- `test_accessed=false` was frozen before CP4 TEST labels were generated; no post-TEST tuning was performed.",
        "",
        "## H. Held-out one-step π1 → π2",
        *[_one_step_report_line(name, payload["overall"]) for name, payload in one_step.items()],
        "",
        "## I. Full policy comparison",
        *[
            _full_policy_report_line(name, payload)
            for name, payload in full_policy.items()
            if name in {"in_distribution", "tactical", "company_survival"}
        ],
        "",
        "## J. Company-survival replication",
        _company_report_line(company_replication),
        "",
        "## K. Emergent strategy",
        _strategy_report_line(full_policy["company_survival"]),
        "",
        "## L. Balance hypotheses",
        "- Mark/Guard/Grab/Bite and related tactical telemetry remain diagnostics only. No kit values were changed.",
        "",
        "## M. Quality / regression",
        "- Scientific artifacts completed under the frozen implementation. Post-test repository quality gates are recorded by the execution task after this generated report.",
        "",
        "## N. Artifacts / checksums",
        f"- Final manifest: `{FINAL_MANIFEST_PATH}`",
        f"- TEST metrics: `{TEST_METRICS_PATH}` / `{final_manifest['test_metrics']['checksum']}`",
        f"- One-step: `{ONE_STEP_PATH}` / `{final_manifest['one_step']['checksum']}`",
        f"- Full policy: `{FULL_POLICY_PATH}` / `{final_manifest['full_policy']['checksum']}`",
        "",
        "## O. Limitations",
        "- CP4 is exactly one policy-iteration step; it does not establish convergence or global optimality.",
        "- CP4 retains previously used TEST seed IDs for direct continuity; the reserved future-confirmation range remains untouched.",
        "- Pairwise Pareto labels intentionally leave crossing outcome dimensions incomparable.",
        "",
        "## P. Recommended Checkpoint 5",
        _checkpoint5_recommendation(train_audit, validation_audit, one_step, full_policy),
        "",
        "Checkpoint 4 stops here. No further policy iteration, PPO, RL, self-play, kit rebalancing, or future-confirmation evaluation was performed.",
    ]
    return "\n".join(lines) + "\n"


def _distribution_report_line(distribution: dict[str, Any]) -> str:
    states = distribution["state_distribution"]
    before = states["cp3"]
    after = states["cp4"]
    top = distribution["feature_distribution"]["top_standardized_mean_shifts"][:5]
    shifts = ", ".join(
        f"{item['name']} {float(item['standardized_mean_shift']):+.2f}σ" for item in top
    )
    return (
        f"- Decision states: CP3 {before['decision_count']} vs CP4 {after['decision_count']}; "
        f"mean round {float(before['rounds']['mean']):.2f} → {float(after['rounds']['mean']):.2f}; "
        f"mean candidates {float(before['candidate_counts']['mean']):.2f} → "
        f"{float(after['candidate_counts']['mean']):.2f}. Top feature mean shifts: {shifts}."
    )


def _outcome_signal_line(label: str, audit: dict[str, Any]) -> str:
    return (
        f"- {label}: {audit['enemy_win_branches']} enemy-win branches; "
        f"{audit['states_with_win_loss_variation']} win/loss-varying states; "
        f"{audit['source_policy_loss_to_win_opportunities']} source-policy loss→win opportunities; "
        f"{audit['source_policy_win_to_loss_risks']} source-policy win→loss alternative risks; "
        f"{audit['decisive_pairs']} decisive pairs; {audit['pareto_loss_pairs']} Pareto-loss pairs; "
        f"{audit['ties']} ties; {audit['incomparable_pairs']} incomparables; "
        f"π1 dominated in {audit['source_policy_dominated_states']}/{audit['source_states']} states."
    )


def _one_step_report_line(label: str, payload: dict[str, Any]) -> str:
    preferences = payload["preference_counts_pi2_vs_pi1"]
    return (
        f"- {label}: π2 preferred {preferences['first']}, π1 preferred {preferences['second']}, "
        f"ties {preferences['tie']}, incomparable {preferences['incomparable']}; "
        f"win rate {float(payload['pi1_enemy_win_rate']):.2%} → "
        f"{float(payload['pi2_enemy_win_rate']):.2%}; loss→win {payload['loss_to_win_flips']}, "
        f"win→loss {payload['win_to_loss_regressions']}."
    )


def _full_policy_report_line(label: str, payload: dict[str, Any]) -> str:
    summaries = payload["summaries"]
    return (
        f"- {label}: π0 {float(summaries['pi0_heuristic']['enemy_win']):.2%}, "
        f"π1 {float(summaries['pi1_cp3']['enemy_win']):.2%}, "
        f"π2 {float(summaries['pi2_cp4']['enemy_win']):.2%} enemy win; "
        f"hero HP {float(summaries['pi0_heuristic']['final_hero_hp']):.2f} / "
        f"{float(summaries['pi1_cp3']['final_hero_hp']):.2f} / "
        f"{float(summaries['pi2_cp4']['final_hero_hp']):.2f}."
    )


def _company_report_line(payload: dict[str, Any]) -> str:
    current = payload["cp4_clean_evaluation"]
    matches = payload["deterministic_replication"]
    return (
        f"- CP4 clean held-out evaluation: π0 {float(current['pi0_heuristic_enemy_win_rate']):.2%}, "
        f"π1 {float(current['pi1_cp3_enemy_win_rate']):.2%}, "
        f"π2 {float(current['pi2_cp4_enemy_win_rate']):.2%}. "
        f"π0/π1 deterministic CP3 replication: {matches['pi0_exact_match']}/"
        f"{matches['pi1_exact_match']}. Wilson intervals and exact paired McNemar "
        "discordance are persisted in the replication artifact."
    )


def _strategy_report_line(payload: dict[str, Any]) -> str:
    summaries = payload["summaries"]
    parts: list[str] = []
    for name in ("pi0_heuristic", "pi1_cp3", "pi2_cp4"):
        row = summaries[name]
        parts.append(
            f"{name}: Mark {float(row['marks_applied']):.2f}, Guard {float(row['guard_actions']):.2f}, "
            f"Grab {float(row['grab_uses']):.2f}, Bite {float(row['bite_uses']):.2f}"
        )
    return "- Company-survival mean uses/case — " + "; ".join(parts) + "."


def _checkpoint5_recommendation(
    train_audit: dict[str, Any],
    validation_audit: dict[str, Any],
    one_step: dict[str, Any],
    full_policy: dict[str, Any],
) -> str:
    train = train_audit["overall"]
    validation = validation_audit["overall"]
    if (
        int(train["states_with_win_loss_variation"]) == 0
        and int(validation["states_with_win_loss_variation"]) == 0
    ):
        return (
            "Recommend exactly one encounter/state-distribution investigation before choosing a "
            "more complex learning algorithm, because π1-generated TRAIN/VALIDATION still failed "
            "to expose direct victory variation. Do not automatically escalate to PPO."
        )
    company = full_policy["company_survival"]["summaries"]
    pi1_rate = float(company["pi1_cp3"]["enemy_win"])
    pi2_rate = float(company["pi2_cp4"]["enemy_win"])
    one = one_step["company_survival"]["overall"]["preference_counts_pi2_vs_pi1"]
    if int(one["first"]) > int(one["second"]) and pi2_rate + 1e-12 < pi1_rate:
        return (
            "Recommend exactly one policy/value-iteration stability investigation, because π2 "
            "improved local Q^(π1) choices while recursive policy strength regressed."
        )
    return (
        "Recommend exactly one final confirmation on the reserved untouched seed range "
        "30001–30128 before considering broader optimization. Do not evaluate that range in CP4."
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
