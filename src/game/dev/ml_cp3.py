# ruff: noqa: E501

"""Checkpoint 3 counterfactual outcome-ranking experiment runner."""

from __future__ import annotations

import argparse
import json
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
from game.combat.enemy_decision import (
    EnemyDecisionCandidate,
    EnemyDecisionRuntimeContext,
    explain_enemy_decision,
)
from game.combat.enemy_learning import create_hero_policy, run_enemy_learning_episode
from game.content.definitions import GameDefinitions
from game.core.rng import GameRng
from game.data.loaders import load_game_definitions
from game.expedition.cave import create_encounter_combat
from game.ml.artifacts import sha256_file
from game.ml.counterfactual import (
    BRANCH_METHOD_ID,
    OUTCOME_SCHEMA_ID,
    OutcomePreference,
    SourceDecision,
    collect_heuristic_source_decisions,
    compare_outcomes,
    outcome_from_episode,
    run_counterfactual_branch,
)
from game.ml.models import linear_scorer_spec, mlp_scorer_spec
from game.ml.outcome_dataset import (
    PREFERENCE_RULE_ID,
    SAMPLING_METHOD_ID,
    CounterfactualDataset,
    SamplingSpec,
    audit_counterfactual_dataset,
    build_counterfactual_dataset,
    decisive_preference_pairs,
    save_counterfactual_dataset,
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
from game.ml.splits import DEFAULT_SEED_SPLIT_SPEC, DatasetSplit

CP2_COMMIT = "f519ebbb98d0009b9c013b79b1fca90c77c76437"
EXPERIMENT_ID = "checkpoint3-counterfactual-outcome-v1"
ARTIFACT_ROOT = Path("artifacts/ml") / EXPERIMENT_ID
PROTOCOL_PATH = ARTIFACT_ROOT / "predeclared_protocol.json"
COST_PROBE_PATH = ARTIFACT_ROOT / "cost_probe.json"
FEATURE_AUDIT_PATH = ARTIFACT_ROOT / "feature_audit.json"
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
ONE_STEP_PATH = ARTIFACT_ROOT / "one_step_improvement.json"
FULL_POLICY_PATH = ARTIFACT_ROOT / "exploratory_full_policy.json"
FINAL_MANIFEST_PATH = ARTIFACT_ROOT / "final_manifest.json"
REPORT_PATH = ARTIFACT_ROOT / "REPORT.md"

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
MAX_ROUNDS = 20
LINEAR_SEED = 20261004
MLP_SEEDS = (20261004, 20261005, 20261006)
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
_DEFAULT_RUNTIME_CONTEXT = EnemyDecisionRuntimeContext()

COST_PROBE_CASES = (
    ("shallow_cave", "naive", 1),
    ("road_bandits", "damage_race", 2),
    ("cave_mini_boss", "survival", 3),
)


class OutcomeRankerPolicy:
    """Research policy that ranks only the simulator's current legal candidates."""

    def __init__(self, model: torch.nn.Module, normalization: Any) -> None:
        self.model = model
        self.normalization = normalization

    def choose(
        self,
        state: Any,
        definitions: GameDefinitions,
        enemy_id: str,
        runtime_context: EnemyDecisionRuntimeContext = _DEFAULT_RUNTIME_CONTEXT,
    ) -> EnemyDecisionCandidate | None:
        trace = explain_enemy_decision(state, definitions, enemy_id, runtime_context)
        if trace is None or not trace.candidates:
            return None
        raw = torch.tensor(
            [
                CP3_FEATURE_SCHEMA.encode(state, definitions, candidate, runtime_context)
                for candidate in trace.candidates
            ],
            dtype=torch.float32,
        )
        features = self.normalization.normalize(raw)
        self.model.eval()
        with torch.no_grad():
            scores = self.model(features)
        return trace.candidates[int(torch.argmax(scores).item())]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Checkpoint 3 counterfactual ML research.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe", help="Run the frozen TRAIN-only compute probe.")
    subparsers.add_parser("train", help="Generate TRAIN/VALIDATION, train, and freeze TEST.")
    subparsers.add_parser("test", help="Generate sealed TEST and evaluate frozen models.")
    args = parser.parse_args(argv)
    if args.command == "probe":
        run_cost_probe(persist=True)
        return 0
    if args.command == "train":
        run_train_and_freeze()
        return 0
    if args.command == "test":
        run_frozen_test()
        return 0
    raise ValueError(f"Unknown command: {args.command}")


def run_cost_probe(*, persist: bool) -> dict[str, object]:
    definitions = load_game_definitions()
    start = time.perf_counter()
    source_seconds = 0.0
    branch_seconds = 0.0
    source_decisions = 0
    sampled_states = 0
    candidate_branches = 0
    storage_bytes = 0
    for encounter_id, hero_policy_id, seed in COST_PROBE_CASES:
        source_start = time.perf_counter()
        decisions = collect_heuristic_source_decisions(
            definitions,
            split=DatasetSplit.TRAIN,
            encounter_id=encounter_id,
            hero_policy_id=hero_policy_id,
            seed=seed,
            max_rounds=MAX_ROUNDS,
        )
        source_seconds += time.perf_counter() - source_start
        source_decisions += len(decisions)
        for source in decisions[:2]:
            sampled_states += 1
            storage_bytes += len(
                json.dumps(asdict(source), sort_keys=True, default=str).encode("utf-8")
            )
            branch_start = time.perf_counter()
            for candidate in source.candidates:
                branch_result = run_counterfactual_branch(
                    definitions,
                    source,
                    candidate,
                    max_rounds=MAX_ROUNDS,
                )
                candidate_branches += 1
                if candidate == source.heuristic_choice:
                    _assert_control_result(source, branch_result)
            branch_seconds += time.perf_counter() - branch_start
    probe_result: dict[str, object] = {
        "probe_split": "train",
        "probe_cases": [list(item) for item in COST_PROBE_CASES],
        "source_decisions_seen": source_decisions,
        "sampled_source_states": sampled_states,
        "candidate_branches": candidate_branches,
        "average_candidates_per_state": candidate_branches / sampled_states,
        "source_states_per_second": sampled_states / source_seconds,
        "candidate_branches_per_second": candidate_branches / branch_seconds,
        "approximate_storage_bytes_per_source_state": storage_bytes / sampled_states,
        "wall_seconds": time.perf_counter() - start,
        "sampling_decision_already_frozen": {
            "train_states_per_encounter_policy": TRAIN_SAMPLING.states_per_encounter_policy,
            "validation_test_states_per_encounter_policy": EVAL_SAMPLING.states_per_encounter_policy,
            "method_id": SAMPLING_METHOD_ID,
        },
    }
    if persist:
        _write_json(COST_PROBE_PATH, probe_result)
    return probe_result


def run_train_and_freeze() -> None:
    _verify_cp2_head()
    definitions = load_game_definitions()
    _validate_encounters(definitions)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(PROTOCOL_PATH, _predeclared_protocol())
    protocol_checksum = sha256_file(PROTOCOL_PATH)
    probe = run_cost_probe(persist=True)
    _write_json(FEATURE_AUDIT_PATH, _feature_audit_payload())

    train_dataset = build_counterfactual_dataset(
        definitions,
        split=DatasetSplit.TRAIN,
        encounter_ids=ENCOUNTER_IDS,
        hero_policy_ids=TRAIN_HERO_POLICY_IDS,
        seeds=DEFAULT_SEED_SPLIT_SPEC.seeds_for(DatasetSplit.TRAIN),
        sampling_spec=TRAIN_SAMPLING,
        max_rounds=MAX_ROUNDS,
    )
    train_checksum = save_counterfactual_dataset(train_dataset, TRAIN_DATASET_PATH)
    train_audit = _audit_payload(train_dataset)
    _write_json(TRAIN_AUDIT_PATH, train_audit)

    validation_dataset = build_counterfactual_dataset(
        definitions,
        split=DatasetSplit.VALIDATION,
        encounter_ids=ENCOUNTER_IDS,
        hero_policy_ids=TRAIN_HERO_POLICY_IDS,
        seeds=DEFAULT_SEED_SPLIT_SPEC.seeds_for(DatasetSplit.VALIDATION),
        sampling_spec=EVAL_SAMPLING,
        max_rounds=MAX_ROUNDS,
    )
    validation_checksum = save_counterfactual_dataset(
        validation_dataset,
        VALIDATION_DATASET_PATH,
    )
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

    normalization = fit_outcome_normalization(train_dataset)
    normalization_checksum = save_outcome_normalization(normalization, NORMALIZATION_PATH)
    model_entries: list[dict[str, object]] = []
    validation_metrics: dict[str, object] = {}

    linear = train_pairwise_ranker(
        linear_scorer_spec(CP3_FEATURE_NAMES),
        train_dataset,
        validation_dataset,
        normalization,
        training_seed=LINEAR_SEED,
        config=TRAINING_CONFIG,
    )
    model_entries.append(
        _save_and_evaluate_model(
            "outcome_linear",
            linear,
            validation_dataset,
            normalization,
            normalization_checksum,
            validation_metrics,
        )
    )
    for seed in MLP_SEEDS:
        trained = train_pairwise_ranker(
            mlp_scorer_spec(CP3_FEATURE_NAMES),
            train_dataset,
            validation_dataset,
            normalization,
            training_seed=seed,
            config=TRAINING_CONFIG,
        )
        model_entries.append(
            _save_and_evaluate_model(
                f"outcome_mlp_seed_{seed}",
                trained,
                validation_dataset,
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
        "cp2_commit": CP2_COMMIT,
        "repository_head": _git_head(),
        "repository_dirty": _git_dirty(),
        "runtime_versions": {"python": sys.version.split()[0], "torch": torch.__version__},
        "predeclared_protocol": {
            "path": str(PROTOCOL_PATH),
            "checksum": protocol_checksum,
        },
        "cost_probe": {"path": str(COST_PROBE_PATH), "result": probe},
        "feature_audit": {
            "path": str(FEATURE_AUDIT_PATH),
            "checksum": sha256_file(FEATURE_AUDIT_PATH),
        },
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
        "branch_method": BRANCH_METHOD_ID,
        "continuation_policy": "heuristic",
        "source_enemy_policy": "heuristic",
        "outcome_schema": OUTCOME_SCHEMA_ID,
        "preference_rule": PREFERENCE_RULE_ID,
        "datasets": {
            "train": {"path": str(TRAIN_DATASET_PATH), "checksum": train_checksum},
            "validation": {
                "path": str(VALIDATION_DATASET_PATH),
                "checksum": validation_checksum,
            },
        },
        "outcome_signal_audit": {
            "train": train_audit["overall"],
            "validation": validation_audit["overall"],
        },
        "normalization": {
            "path": str(NORMALIZATION_PATH),
            "checksum": normalization_checksum,
            "fit_split": "train",
        },
        "training": {
            "config": TRAINING_CONFIG.to_dict(),
            "linear_seed": LINEAR_SEED,
            "mlp_seeds": list(MLP_SEEDS),
            "primary_selection": "highest validation pairwise accuracy, then lowest validation loss, then lowest seed",
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
        },
        "test_metrics_predeclared": [
            "decisive pair count and pairwise accuracy",
            "accuracy by encounter/policy/enemy class",
            "heuristic dominated rate",
            "loss-to-win opportunity rate",
            "one-step paired preference and outcome differences",
        ],
        "exploratory_full_policy": {
            "enabled": True,
            "seeds": list(FULL_POLICY_TEST_SEEDS),
            "label": "EXPLORATORY FULL-POLICY ROLLOUT",
        },
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
        raise ValueError("CP3 frozen configuration is required before TEST.")
    frozen_checksum_before = sha256_file(FROZEN_CONFIG_PATH)
    frozen = _read_json(FROZEN_CONFIG_PATH)
    _validate_frozen_config(frozen)
    definitions = load_game_definitions()
    normalization = load_outcome_normalization(NORMALIZATION_PATH)
    models = _load_frozen_models(frozen)

    test_seeds = DEFAULT_SEED_SPLIT_SPEC.seeds_for(DatasetSplit.TEST)
    datasets = {
        "in_distribution": build_counterfactual_dataset(
            definitions,
            split=DatasetSplit.TEST,
            encounter_ids=ENCOUNTER_IDS,
            hero_policy_ids=TRAIN_HERO_POLICY_IDS,
            seeds=test_seeds,
            sampling_spec=EVAL_SAMPLING,
            max_rounds=MAX_ROUNDS,
        ),
        "tactical": build_counterfactual_dataset(
            definitions,
            split=DatasetSplit.TEST,
            encounter_ids=ENCOUNTER_IDS,
            hero_policy_ids=TACTICAL_POLICY_IDS,
            seeds=test_seeds,
            sampling_spec=EVAL_SAMPLING,
            max_rounds=MAX_ROUNDS,
        ),
        "company_survival": build_counterfactual_dataset(
            definitions,
            split=DatasetSplit.TEST,
            encounter_ids=ENCOUNTER_IDS,
            hero_policy_ids=COMPANY_POLICY_IDS,
            seeds=test_seeds,
            sampling_spec=EVAL_SAMPLING,
            max_rounds=MAX_ROUNDS,
        ),
    }
    paths = {
        "in_distribution": TEST_ID_PATH,
        "tactical": TEST_TACTICAL_PATH,
        "company_survival": TEST_COMPANY_PATH,
    }
    dataset_checksums = {
        name: save_counterfactual_dataset(dataset, paths[name])
        for name, dataset in datasets.items()
    }

    test_metrics: dict[str, object] = {
        "frozen_config_checksum": frozen_checksum_before,
        "dataset_checksums": dataset_checksums,
    }
    one_step: dict[str, object] = {}
    primary_name = str(frozen["primary_model"])
    for group_name, dataset in datasets.items():
        audit = _audit_payload(dataset)
        group_models: dict[str, object] = {}
        for model_name, loaded in models.items():
            loss, accuracy = evaluate_pairwise(loaded.model, dataset, normalization)
            group_models[model_name] = {
                "decisive_pairs": len(decisive_preference_pairs(dataset)),
                "pairwise_loss": loss,
                "pairwise_accuracy": accuracy,
                "by_encounter": grouped_pairwise_accuracy(
                    loaded.model, dataset, normalization, group_by="encounter"
                ),
                "by_hero_policy": grouped_pairwise_accuracy(
                    loaded.model, dataset, normalization, group_by="hero_policy"
                ),
                "by_enemy_class": grouped_pairwise_accuracy(
                    loaded.model, dataset, normalization, group_by="enemy_class"
                ),
            }
        test_metrics[group_name] = {
            "audit": audit,
            "models": group_models,
        }
        one_step[group_name] = _one_step_summary(
            dataset,
            models[primary_name].model,
            normalization,
        )
    _write_json(TEST_METRICS_PATH, test_metrics)
    _write_json(ONE_STEP_PATH, one_step)

    full_policy = _run_exploratory_full_policy(
        definitions,
        models[primary_name].model,
        normalization,
    )
    _write_json(FULL_POLICY_PATH, full_policy)

    frozen_checksum_after = sha256_file(FROZEN_CONFIG_PATH)
    if frozen_checksum_after != frozen_checksum_before:
        raise RuntimeError("Frozen CP3 configuration changed during TEST.")
    final_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "cp2_commit": CP2_COMMIT,
        "frozen_config": {
            "path": str(FROZEN_CONFIG_PATH),
            "checksum": frozen_checksum_after,
            "unchanged_during_test": True,
        },
        "primary_model": primary_name,
        "test_tuning_performed": False,
        "test_metrics": {
            "path": str(TEST_METRICS_PATH),
            "checksum": sha256_file(TEST_METRICS_PATH),
        },
        "one_step": {"path": str(ONE_STEP_PATH), "checksum": sha256_file(ONE_STEP_PATH)},
        "full_policy": {"path": str(FULL_POLICY_PATH), "checksum": sha256_file(FULL_POLICY_PATH)},
        "test_dataset_checksums": dataset_checksums,
        "model_artifacts": frozen["model_artifacts"],
    }
    _write_json(FINAL_MANIFEST_PATH, final_manifest)
    REPORT_PATH.write_text(
        _build_report(frozen, test_metrics, one_step, full_policy, final_manifest),
        encoding="utf-8",
    )
    print(f"TEST_METRICS={TEST_METRICS_PATH}")
    print(f"ONE_STEP={ONE_STEP_PATH}")
    print(f"FULL_POLICY={FULL_POLICY_PATH}")
    print(f"REPORT={REPORT_PATH}")
    print(f"FROZEN_CONFIG_UNCHANGED={frozen_checksum_after}")


def _predeclared_protocol() -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "cp2_commit": CP2_COMMIT,
        "scientific_question": "Can simulator-derived one-step counterfactual preferences improve over the heuristic action on unseen states?",
        "source_enemy_policy": "heuristic",
        "continuation_enemy_policy": "heuristic",
        "branch_method": BRANCH_METHOD_ID,
        "hero_policy_fixed_within_branches": True,
        "observation_schema": {
            "id": CP3_FEATURE_SCHEMA.schema_id,
            "version": CP3_FEATURE_SCHEMA.version,
        },
        "outcome_schema": OUTCOME_SCHEMA_ID,
        "preference_rule": PREFERENCE_RULE_ID,
        "loss_dimensions": [
            "hero_deaths (higher is worse for party)",
            "terminal_hero_downs (higher is worse)",
            "mortal_wounds (higher is worse)",
            "final_hero_hp (lower is worse)",
            "enemy_caused_effort_drain (higher is worse)",
        ],
        "two_enemy_wins": "tie",
        "crossing_loss_dimensions": "incomparable",
        "sampling": {
            "method_id": SAMPLING_METHOD_ID,
            "train_states_per_encounter_policy": 16,
            "validation_test_states_per_encounter_policy": 8,
            "outcome_blind": True,
        },
        "minimum_signal_gate": {
            "train_decisive_pairs": MIN_TRAIN_DECISIVE_PAIRS,
            "validation_decisive_pairs": MIN_VALIDATION_DECISIVE_PAIRS,
        },
        "models": {
            "linear": {"seed": LINEAR_SEED},
            "mlp": {"shape": [len(CP3_FEATURE_NAMES), 64, 32, 1], "seeds": list(MLP_SEEDS)},
        },
        "training": TRAINING_CONFIG.to_dict(),
        "primary_selection": "validation pairwise accuracy desc, validation loss asc, seed asc",
        "splits": {"train": [1, 256], "validation": [10001, 10064], "test": [20001, 20128]},
        "encounter_ids": list(ENCOUNTER_IDS),
        "train_validation_policies": list(TRAIN_HERO_POLICY_IDS),
        "test_groups": {
            "in_distribution": list(TRAIN_HERO_POLICY_IDS),
            "tactical": ["tactical"],
            "company_survival": ["company_survival"],
        },
        "exploratory_full_policy": {
            "enabled": True,
            "test_seeds": list(FULL_POLICY_TEST_SEEDS),
            "not_a_tuning_gate": True,
        },
        "test_accessed": False,
    }


def _feature_audit_payload() -> dict[str, object]:
    return {
        "cp2_feature_audit": [asdict(item) for item in CP2_FEATURE_AUDIT],
        "cp3_schema": {
            "id": CP3_FEATURE_SCHEMA.schema_id,
            "version": CP3_FEATURE_SCHEMA.version,
            "feature_count": len(CP3_FEATURE_NAMES),
            "neutral_factual": list(CP3_NEUTRAL_FACTUAL_FEATURE_NAMES),
            "derived_factual": list(CP3_DERIVED_FEATURE_NAMES),
            "prescriptive_included": [],
        },
        "principle": "represent what is true, not what the old heuristic says is good",
    }


def _audit_payload(dataset: CounterfactualDataset) -> dict[str, Any]:
    return {
        "overall": asdict(audit_counterfactual_dataset(dataset)),
        "by_encounter": _audits_by(dataset, "encounter"),
        "by_hero_policy": _audits_by(dataset, "hero_policy"),
        "by_enemy_class": _audits_by(dataset, "enemy_class"),
        "available_source_states": dataset.available_source_states,
        "sampled_source_states": len(dataset.records),
    }


def _audits_by(dataset: CounterfactualDataset, grouping: str) -> dict[str, dict[str, int]]:
    groups: dict[str, list[Any]] = {}
    for record in dataset.records:
        if grouping == "encounter":
            key = record.encounter_id
        elif grouping == "hero_policy":
            key = record.hero_policy_id
        elif grouping == "enemy_class":
            key = record.enemy_class_id
        else:
            raise ValueError(grouping)
        groups.setdefault(key, []).append(record)
    output: dict[str, dict[str, int]] = {}
    for key, records in sorted(groups.items()):
        subset = CounterfactualDataset(
            split=dataset.split,
            sampling_spec=dataset.sampling_spec,
            encounter_ids=dataset.encounter_ids,
            hero_policy_ids=dataset.hero_policy_ids,
            seed_count=dataset.seed_count,
            available_source_states=len(records),
            records=tuple(records),
        )
        output[key] = asdict(audit_counterfactual_dataset(subset))
    return output


def _save_and_evaluate_model(
    name: str,
    trained: Any,
    validation_dataset: CounterfactualDataset,
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
    loaded = load_outcome_ranker(metadata_path)
    loss, accuracy = evaluate_pairwise(loaded.model, validation_dataset, normalization)
    validation_metrics[name] = {
        "pairwise_loss": loss,
        "pairwise_accuracy": accuracy,
        "decisive_pairs": len(decisive_preference_pairs(validation_dataset)),
        "by_encounter": grouped_pairwise_accuracy(
            loaded.model, validation_dataset, normalization, group_by="encounter"
        ),
        "by_hero_policy": grouped_pairwise_accuracy(
            loaded.model, validation_dataset, normalization, group_by="hero_policy"
        ),
        "by_enemy_class": grouped_pairwise_accuracy(
            loaded.model, validation_dataset, normalization, group_by="enemy_class"
        ),
    }
    return {
        "name": name,
        "metadata_path": str(metadata_path),
        "metadata_checksum": sha256_file(metadata_path),
        "model_file": metadata["model_file"],
        "model_checksum": metadata["model_checksum"],
        "training_seed": metadata["training_result"]["training_seed"],
        "selected_epoch": metadata["training_result"]["selected_epoch"],
    }


def _select_primary_mlp(validation_metrics: dict[str, object]) -> str:
    candidates: list[tuple[float, float, int, str]] = []
    for seed in MLP_SEEDS:
        name = f"outcome_mlp_seed_{seed}"
        metrics = validation_metrics[name]
        assert isinstance(metrics, dict)
        candidates.append(
            (
                -float(metrics["pairwise_accuracy"]),
                float(metrics["pairwise_loss"]),
                seed,
                name,
            )
        )
    return min(candidates)[3]


def _one_step_summary(
    dataset: CounterfactualDataset,
    model: torch.nn.Module,
    normalization: Any,
) -> dict[str, object]:
    preference_counts = {item.value: 0 for item in OutcomePreference}
    heuristic_wins = 0
    learned_wins = 0
    loss_to_win = 0
    win_to_loss = 0
    hp_differences: list[int] = []
    down_differences: list[int] = []
    death_differences: list[int] = []
    wound_differences: list[int] = []
    effort_drain_differences: list[int] = []
    changed_action = 0
    for index, record in enumerate(dataset.records):
        learned_index = choose_candidate_index(model, index, dataset, normalization)
        heuristic = record.candidates[record.heuristic_index].outcome
        learned = record.candidates[learned_index].outcome
        changed_action += int(learned_index != record.heuristic_index)
        preference = compare_outcomes(learned, heuristic)
        preference_counts[preference.value] += 1
        heuristic_wins += int(heuristic.enemy_win)
        learned_wins += int(learned.enemy_win)
        loss_to_win += int(not heuristic.enemy_win and learned.enemy_win)
        win_to_loss += int(heuristic.enemy_win and not learned.enemy_win)
        hp_differences.append(learned.final_hero_hp - heuristic.final_hero_hp)
        down_differences.append(learned.terminal_hero_downs - heuristic.terminal_hero_downs)
        death_differences.append(learned.hero_deaths - heuristic.hero_deaths)
        wound_differences.append(learned.mortal_wounds - heuristic.mortal_wounds)
        effort_drain_differences.append(
            learned.enemy_caused_effort_drain - heuristic.enemy_caused_effort_drain
        )
    count = len(dataset.records)
    return {
        "source_states": count,
        "changed_action_states": changed_action,
        "preference_counts_learned_vs_heuristic": preference_counts,
        "heuristic_enemy_win_rate": heuristic_wins / count if count else 0.0,
        "learned_enemy_win_rate": learned_wins / count if count else 0.0,
        "loss_to_win_flips": loss_to_win,
        "win_to_loss_regressions": win_to_loss,
        "mean_learned_minus_heuristic": {
            "final_hero_hp": mean(hp_differences) if hp_differences else 0.0,
            "terminal_hero_downs": mean(down_differences) if down_differences else 0.0,
            "hero_deaths": mean(death_differences) if death_differences else 0.0,
            "mortal_wounds": mean(wound_differences) if wound_differences else 0.0,
            "enemy_caused_effort_drain": mean(effort_drain_differences)
            if effort_drain_differences
            else 0.0,
        },
    }


def _run_exploratory_full_policy(
    definitions: GameDefinitions,
    model: torch.nn.Module,
    normalization: Any,
) -> dict[str, object]:
    learned_policy = OutcomeRankerPolicy(model, normalization)
    policy_groups = {
        "in_distribution": TRAIN_HERO_POLICY_IDS,
        "tactical": TACTICAL_POLICY_IDS,
        "company_survival": COMPANY_POLICY_IDS,
    }
    result: dict[str, object] = {
        "label": "EXPLORATORY FULL-POLICY ROLLOUT",
        "training_distribution_warning": "model trained on heuristic-generated states",
        "seeds": list(FULL_POLICY_TEST_SEEDS),
    }
    for group_name, hero_policy_ids in policy_groups.items():
        rows: dict[str, list[dict[str, float | int]]] = {"heuristic": [], "learned": []}
        for encounter_id in ENCOUNTER_IDS:
            for hero_policy_id in hero_policy_ids:
                for seed in FULL_POLICY_TEST_SEEDS:
                    for policy_name, policy in (("heuristic", None), ("learned", learned_policy)):
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
                        rows[policy_name].append(
                            {
                                "enemy_win": int(outcome.enemy_win),
                                "final_hero_hp": outcome.final_hero_hp,
                                "hero_downs": outcome.terminal_hero_downs,
                                "hero_deaths": outcome.hero_deaths,
                                "mortal_wounds": outcome.mortal_wounds,
                                "rounds": outcome.rounds_elapsed,
                                "marks_applied": episode.metrics.marks_applied,
                                "guard_actions": episode.metrics.guard_actions,
                                "grab_uses": episode.metrics.boss_sequence.grab_uses,
                                "bite_uses": episode.metrics.boss_sequence.bite_uses,
                            }
                        )
        result[group_name] = {name: _mean_rollout_rows(items) for name, items in rows.items()}
    return result


def _mean_rollout_rows(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        return {"case_count": 0}
    keys = tuple(rows[0])
    return {
        "case_count": len(rows),
        **{key: mean(float(row[key]) for row in rows) for key in keys},
    }


def _build_report(
    frozen: dict[str, Any],
    test_metrics: dict[str, Any],
    one_step: dict[str, Any],
    full_policy: dict[str, Any],
    final_manifest: dict[str, Any],
) -> str:
    validation = _read_json(VALIDATION_METRICS_PATH)
    lines = [
        "# The Charter — Checkpoint 3 Counterfactual Outcome Ranking",
        "",
        "## A. Provenance",
        f"- CP2 commit: `{CP2_COMMIT}`",
        f"- Python: `{frozen['runtime_versions']['python']}`; Torch: `{frozen['runtime_versions']['torch']}`.",
        "",
        "## B. Counterfactual method",
        f"- Branch method: `{BRANCH_METHOD_ID}` deterministic prefix replay.",
        "- Every branch replays the same seed/prefix, verifies state/RNG/legal candidates, forces one legal action, then returns to heuristic continuation.",
        "- The heuristic-control branch is required to reproduce the original terminal state and episode digest exactly.",
        "",
        "## C. Feature audit",
        f"- CP3 observation schema: `{CP3_FEATURE_SCHEMA.schema_id}` v{CP3_FEATURE_SCHEMA.version}, {len(CP3_FEATURE_NAMES)} features.",
        f"- Neutral factual: {len(CP3_NEUTRAL_FACTUAL_FEATURE_NAMES)}; derived factual: {len(CP3_DERIVED_FEATURE_NAMES)}; prescriptive included: 0.",
        "- CP2 heuristic guidance is audited separately and excluded from the outcome optimizer.",
        "",
        "## D. Sampling / dataset",
        f"- Sampling: `{SAMPLING_METHOD_ID}`; TRAIN {TRAIN_SAMPLING.states_per_encounter_policy}/stratum, VAL/TEST {EVAL_SAMPLING.states_per_encounter_policy}/stratum.",
        f"- TRAIN sampled states: {frozen['outcome_signal_audit']['train']['source_states']}; VALIDATION: {frozen['outcome_signal_audit']['validation']['source_states']}.",
        "",
        "## E. Outcome-signal audit",
        _audit_report_line("TRAIN", frozen["outcome_signal_audit"]["train"]),
        _audit_report_line("VALIDATION", frozen["outcome_signal_audit"]["validation"]),
        "",
        "## F. Train / validation",
    ]
    for name in ("outcome_linear", *(f"outcome_mlp_seed_{seed}" for seed in MLP_SEEDS)):
        metrics = validation[name]
        lines.append(
            f"- {name}: pairwise accuracy {float(metrics['pairwise_accuracy']):.2%}, loss {float(metrics['pairwise_loss']):.4f}."
        )
    lines.extend(
        [
            f"- Validation-selected primary model: `{frozen['primary_model']}`.",
            "",
            "## G. Frozen TEST configuration",
            f"- Frozen config: `{FROZEN_CONFIG_PATH}`",
            f"- SHA-256: `{final_manifest['frozen_config']['checksum']}`",
            "- TEST counterfactual labels were not generated before this freeze; the train phase reported `TEST_ACCESSED=false`.",
            "",
            "## H. Held-out TEST",
        ]
    )
    for group in ("in_distribution", "tactical", "company_survival"):
        lines.append(f"### {group}")
        audit = test_metrics[group]["audit"]["overall"]
        lines.append(_audit_report_line("signal", audit))
        for name in ("outcome_linear", frozen["primary_model"]):
            metrics = test_metrics[group]["models"][name]
            lines.append(
                f"- {name}: pairwise accuracy {float(metrics['pairwise_accuracy']):.2%}, n={metrics['decisive_pairs']}."
            )
    lines.extend(["", "## I. One-step policy-improvement rollout"])
    for group in ("in_distribution", "tactical", "company_survival"):
        item = one_step[group]
        counts = item["preference_counts_learned_vs_heuristic"]
        lines.append(
            f"- {group}: learned preferred {counts['first']}, heuristic preferred {counts['second']}, ties {counts['tie']}, incomparable {counts['incomparable']}; loss→win {item['loss_to_win_flips']}, win→loss {item['win_to_loss_regressions']}."
        )
    lines.extend(
        [
            "",
            "## J. Exploratory full-policy rollout",
            "- This is distribution-shift exploration only: the ranker was trained on heuristic-generated states.",
        ]
    )
    for group in ("in_distribution", "tactical", "company_survival"):
        heuristic_summary = full_policy[group]["heuristic"]
        learned_summary = full_policy[group]["learned"]
        lines.append(
            f"- {group}: heuristic enemy win {float(heuristic_summary['enemy_win']):.2%}, learned {float(learned_summary['enemy_win']):.2%}; heuristic hero HP {float(heuristic_summary['final_hero_hp']):.2f}, learned {float(learned_summary['final_hero_hp']):.2f}."
        )
    lines.extend(
        [
            "",
            "## K. Emergent strategy / balance findings",
            "- Mark/Guard/Grab/Bite telemetry is diagnostic only. Divergence from authored behavior is balance evidence, not a reward failure.",
            "",
            "## L. Quality / regression results",
            "- Filled by the execution closeout after focused, AI/ML, full pytest, Ruff, mypy, diff, and normal-import gates.",
            "",
            "## M. Artifacts / checksums",
            f"- Final manifest: `{FINAL_MANIFEST_PATH}`",
            f"- TEST metrics: `{TEST_METRICS_PATH}`",
            f"- One-step results: `{ONE_STEP_PATH}`",
            f"- Exploratory full-policy: `{FULL_POLICY_PATH}`",
            "",
            "## N. Risks / limitations",
            "- CP3 estimates one-step improvement over heuristic-generated states and Q^(heuristic)(s,a), not a globally optimal Q function.",
            "- Pareto labeling intentionally leaves cross-dimensional loss outcomes incomparable instead of inventing exchange rates.",
            "- Full-policy rollout visits states outside the heuristic source distribution and is exploratory.",
            "",
            "## O. Recommended Checkpoint 4",
            "Run one bounded iterative policy-improvement cycle: regenerate source states under the CP3-improved policy, counterfactually evaluate them with a fixed continuation policy, retrain, and test whether the improvement survives the induced state-distribution shift.",
            "",
        ]
    )
    return "\n".join(lines)


def _audit_report_line(label: str, audit: dict[str, Any]) -> str:
    return (
        f"- {label}: states {audit['source_states']}, branches {audit['candidate_branches']}, "
        f"decisive pairs {audit['decisive_pairs']}, heuristic dominated {audit['heuristic_dominated_states']}, "
        f"loss→win opportunities {audit['heuristic_loss_to_win_opportunities']}, "
        f"ties {audit['ties']}, incomparable {audit['incomparable_pairs']}."
    )


def _assert_control_result(source: SourceDecision, result: Any) -> None:
    if result.outcome != source.baseline_outcome:
        raise RuntimeError("Cost-probe control branch changed outcome.")
    if result.final_state_digest != source.baseline_final_state_digest:
        raise RuntimeError("Cost-probe control branch changed final state.")
    if result.episode_digest != source.baseline_episode_digest:
        raise RuntimeError("Cost-probe control branch changed episode digest.")


def _load_frozen_models(frozen: dict[str, Any]) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for entry in frozen["model_artifacts"]:
        path = Path(entry["metadata_path"])
        if sha256_file(path) != entry["metadata_checksum"]:
            raise ValueError(f"Frozen model metadata changed: {entry['name']}")
        model = load_outcome_ranker(path)
        if model.metadata["model_checksum"] != entry["model_checksum"]:
            raise ValueError(f"Frozen model binary changed: {entry['name']}")
        loaded[str(entry["name"])] = model
    return loaded


def _validate_frozen_config(frozen: dict[str, Any]) -> None:
    if frozen.get("status") != "frozen_before_test" or frozen.get("test_accessed") is not False:
        raise ValueError("CP3 frozen configuration does not represent a sealed TEST state.")
    if frozen.get("cp2_commit") != CP2_COMMIT:
        raise ValueError("CP3 frozen configuration has the wrong CP2 commit.")
    if frozen.get("implementation_hashes") != _implementation_hashes():
        raise ValueError("CP3 implementation changed after freeze.")
    if frozen.get("simulator_hashes") != _simulator_hashes():
        raise ValueError("Simulator implementation changed after freeze.")
    if frozen.get("content_hashes") != _content_hashes():
        raise ValueError("Authored content changed after freeze.")
    if sha256_file(NORMALIZATION_PATH) != frozen["normalization"]["checksum"]:
        raise ValueError("Normalization changed after freeze.")


def _verify_cp2_head() -> None:
    if _git_head() != CP2_COMMIT:
        raise ValueError(f"Checkpoint 3 expected CP2 HEAD {CP2_COMMIT}, got {_git_head()}.")


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, encoding="utf-8"
        ).strip()
    )


def _implementation_hashes() -> dict[str, str]:
    return _hash_paths(
        (
            Path("src/game/dev/ml_cp3.py"),
            Path("src/game/ml/counterfactual.py"),
            Path("src/game/ml/outcome_features.py"),
            Path("src/game/ml/outcome_dataset.py"),
            Path("src/game/ml/outcome_ranking.py"),
            Path("src/game/ml/models.py"),
        )
    )


def _simulator_hashes() -> dict[str, str]:
    return _hash_paths(
        (
            Path("src/game/combat/enemy_learning.py"),
            Path("src/game/combat/enemy_decision.py"),
            Path("src/game/combat/actions.py"),
            Path("src/game/combat/combat_state.py"),
            Path("src/game/core/rng.py"),
        )
    )


def _content_hashes() -> dict[str, str]:
    return _hash_paths(tuple(sorted(Path("data").glob("*.yaml"))))


def _hash_paths(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def _validate_encounters(definitions: GameDefinitions) -> None:
    missing = tuple(item for item in ENCOUNTER_IDS if item not in definitions.encounters)
    if missing:
        raise ValueError(f"Missing CP3 encounters: {missing}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
