"""Checkpoint 2 supervised enemy candidate-ranking experiment runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from game.combat.enemy_decision import (
    production_enemy_decision_policy,
    production_enemy_movement_mode,
    production_enemy_wait_mode,
)
from game.content.definitions import GameDefinitions
from game.data.loaders import load_game_definitions
from game.ml.artifacts import sha256_file
from game.ml.dataset import (
    DATASET_FORMAT_ID,
    DATASET_FORMAT_VERSION,
    DatasetExtractionConfig,
    EnemyDecisionDataset,
    extract_enemy_decision_dataset,
)
from game.ml.evaluation import (
    ResearchScorerPolicy,
    RolloutPolicySpec,
    evaluate_imitation,
    run_rollout_comparison,
)
from game.ml.features import (
    ENEMY_DECISION_FEATURE_SCHEMA,
    HEURISTIC_VISIBLE_FEATURE_NAMES,
)
from game.ml.models import linear_scorer_spec, mlp_scorer_spec
from game.ml.splits import DEFAULT_SEED_SPLIT_SPEC, DatasetSplit
from game.ml.training import (
    TrainingConfig,
    fit_train_normalization,
    load_model_artifact,
    load_normalization,
    prepare_decision_groups,
    save_model_artifact,
    save_normalization,
    train_candidate_scorer,
)

FOUNDATION_COMMIT = "2a420ce7e1e2f7f57dde52a23526c2abac7c8b01"
EXPERIMENT_ID = "checkpoint2-supervised-imitation-v1"
ARTIFACT_ROOT = Path("artifacts") / "ml" / EXPERIMENT_ID
FROZEN_CONFIG_PATH = ARTIFACT_ROOT / "frozen_config.json"
NORMALIZATION_PATH = ARTIFACT_ROOT / "normalization.json"
VALIDATION_METRICS_PATH = ARTIFACT_ROOT / "validation_metrics.json"
TEST_METRICS_PATH = ARTIFACT_ROOT / "test_metrics.json"
ROLLOUT_METRICS_PATH = ARTIFACT_ROOT / "rollout_metrics.json"
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
POLICY_SHIFT_HERO_POLICY_IDS = ("tactical", "company_survival")
LINEAR_TRAINING_SEED = 20261004
MLP_TRAINING_SEEDS = (20261004, 20261005, 20261006)
ABLATION_TRAINING_SEED = 20261004
BOOTSTRAP_SEED = 20261004
BOOTSTRAP_SAMPLES = 1000
TOP1_MATERIALITY_THRESHOLD = 0.01
MAX_ROUNDS = 20
TRAINING_CONFIG = TrainingConfig(
    learning_rate=0.01,
    batch_size=512,
    max_epochs=50,
    early_stopping_patience=6,
    min_delta=1e-5,
    optimizer="adam",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Checkpoint 2 supervised ML research.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Train/validate and freeze without touching TEST.")
    subparsers.add_parser("test", help="Run sealed TEST and held-out simulator rollouts.")
    args = parser.parse_args(argv)
    if args.command == "train":
        run_train_and_freeze()
        return 0
    if args.command == "test":
        run_frozen_test()
        return 0
    raise ValueError(f"Unknown command: {args.command}")


def run_train_and_freeze() -> None:
    _verify_foundation_head()
    definitions = load_game_definitions()
    _validate_declared_encounters(definitions)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    train_dataset = _extract_dataset(DatasetSplit.TRAIN, TRAIN_HERO_POLICY_IDS)
    validation_dataset = _extract_dataset(DatasetSplit.VALIDATION, TRAIN_HERO_POLICY_IDS)
    normalization = fit_train_normalization(train_dataset)
    normalization_checksum = save_normalization(normalization, NORMALIZATION_PATH)

    full_feature_names = ENEMY_DECISION_FEATURE_SCHEMA.names
    train_full = prepare_decision_groups(train_dataset, normalization, full_feature_names)
    validation_full = prepare_decision_groups(
        validation_dataset,
        normalization,
        full_feature_names,
    )
    train_visible = prepare_decision_groups(
        train_dataset,
        normalization,
        HEURISTIC_VISIBLE_FEATURE_NAMES,
    )
    validation_visible = prepare_decision_groups(
        validation_dataset,
        normalization,
        HEURISTIC_VISIBLE_FEATURE_NAMES,
    )

    model_entries: list[dict[str, object]] = []
    validation_metrics: dict[str, object] = {}

    linear = train_candidate_scorer(
        linear_scorer_spec(full_feature_names),
        train_full,
        validation_full,
        training_seed=LINEAR_TRAINING_SEED,
        config=TRAINING_CONFIG,
    )
    linear_name = "research_imitation_linear"
    model_entries.append(
        _save_and_validate_model(
            linear_name,
            linear,
            validation_dataset,
            normalization,
            normalization_checksum,
            validation_metrics,
        )
    )

    for seed in MLP_TRAINING_SEEDS:
        trained = train_candidate_scorer(
            mlp_scorer_spec(full_feature_names),
            train_full,
            validation_full,
            training_seed=seed,
            config=TRAINING_CONFIG,
        )
        model_name = f"research_mlp_seed_{seed}"
        model_entries.append(
            _save_and_validate_model(
                model_name,
                trained,
                validation_dataset,
                normalization,
                normalization_checksum,
                validation_metrics,
            )
        )

    visible = train_candidate_scorer(
        mlp_scorer_spec(HEURISTIC_VISIBLE_FEATURE_NAMES),
        train_visible,
        validation_visible,
        training_seed=ABLATION_TRAINING_SEED,
        config=TRAINING_CONFIG,
    )
    visible_name = f"research_mlp_heuristic_visible_seed_{ABLATION_TRAINING_SEED}"
    model_entries.append(
        _save_and_validate_model(
            visible_name,
            visible,
            validation_dataset,
            normalization,
            normalization_checksum,
            validation_metrics,
        )
    )

    _write_json(VALIDATION_METRICS_PATH, validation_metrics)
    validation_checksum = sha256_file(VALIDATION_METRICS_PATH)
    frozen = _frozen_config(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        normalization_checksum=normalization_checksum,
        validation_checksum=validation_checksum,
        model_entries=tuple(model_entries),
    )
    _write_json(FROZEN_CONFIG_PATH, frozen)
    frozen_checksum = sha256_file(FROZEN_CONFIG_PATH)
    print(f"FROZEN_CONFIG={FROZEN_CONFIG_PATH}")
    print(f"FROZEN_CONFIG_SHA256={frozen_checksum}")
    print(f"TRAIN_DECISIONS={len(train_dataset.records)}")
    print(f"VALIDATION_DECISIONS={len(validation_dataset.records)}")
    print("TEST_ACCESSED=false")


def run_frozen_test() -> None:
    if not FROZEN_CONFIG_PATH.exists():
        raise ValueError("Frozen configuration is required before TEST evaluation.")
    frozen_checksum_before = sha256_file(FROZEN_CONFIG_PATH)
    frozen = _read_json(FROZEN_CONFIG_PATH)
    _validate_frozen_config(frozen)
    definitions = load_game_definitions()
    _validate_declared_encounters(definitions)
    normalization = load_normalization(NORMALIZATION_PATH)
    normalization_checksum = sha256_file(NORMALIZATION_PATH)
    if normalization_checksum != frozen["normalization"]["checksum"]:
        raise ValueError("Normalization checksum differs from frozen configuration.")

    loaded_models = _load_frozen_models(frozen)
    test_in_distribution = _extract_dataset(DatasetSplit.TEST, TRAIN_HERO_POLICY_IDS)
    test_policy_shift = _extract_dataset(DatasetSplit.TEST, POLICY_SHIFT_HERO_POLICY_IDS)

    test_metrics: dict[str, Any] = {
        "frozen_config_checksum": frozen_checksum_before,
        "in_distribution": {},
        "policy_shift": {},
    }
    for model_name, loaded in loaded_models.items():
        test_metrics["in_distribution"][model_name] = evaluate_imitation(
            loaded.model,
            test_in_distribution,
            normalization,
            loaded.spec.feature_names,
        )
        test_metrics["policy_shift"][model_name] = evaluate_imitation(
            loaded.model,
            test_policy_shift,
            normalization,
            loaded.spec.feature_names,
        )
    _write_json(TEST_METRICS_PATH, test_metrics)

    policies = _rollout_policy_specs(loaded_models, normalization)
    test_seeds = DEFAULT_SEED_SPLIT_SPEC.seeds_for(DatasetSplit.TEST)
    rollout_metrics = {
        "in_distribution": run_rollout_comparison(
            definitions,
            encounter_ids=ENCOUNTER_IDS,
            hero_policy_ids=TRAIN_HERO_POLICY_IDS,
            seeds=test_seeds,
            policies=policies,
            max_rounds=MAX_ROUNDS,
            bootstrap_seed=BOOTSTRAP_SEED,
            bootstrap_samples=BOOTSTRAP_SAMPLES,
        ),
        "policy_shift": run_rollout_comparison(
            definitions,
            encounter_ids=ENCOUNTER_IDS,
            hero_policy_ids=POLICY_SHIFT_HERO_POLICY_IDS,
            seeds=test_seeds,
            policies=policies,
            max_rounds=MAX_ROUNDS,
            bootstrap_seed=BOOTSTRAP_SEED,
            bootstrap_samples=BOOTSTRAP_SAMPLES,
        ),
    }
    _write_json(ROLLOUT_METRICS_PATH, rollout_metrics)

    frozen_checksum_after = sha256_file(FROZEN_CONFIG_PATH)
    if frozen_checksum_after != frozen_checksum_before:
        raise RuntimeError("Frozen configuration changed during TEST evaluation.")

    final_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "foundation_commit": FOUNDATION_COMMIT,
        "frozen_config": {
            "path": str(FROZEN_CONFIG_PATH),
            "checksum": frozen_checksum_after,
            "unchanged_during_test": True,
        },
        "normalization": {
            "path": str(NORMALIZATION_PATH),
            "checksum": normalization_checksum,
        },
        "test_metrics": {
            "path": str(TEST_METRICS_PATH),
            "checksum": sha256_file(TEST_METRICS_PATH),
        },
        "rollout_metrics": {
            "path": str(ROLLOUT_METRICS_PATH),
            "checksum": sha256_file(ROLLOUT_METRICS_PATH),
        },
        "model_artifacts": frozen["model_artifacts"],
        "test_dataset_counts": {
            "in_distribution": _dataset_counts(test_in_distribution),
            "policy_shift": _dataset_counts(test_policy_shift),
        },
        "test_tuning_performed": False,
    }
    _write_json(FINAL_MANIFEST_PATH, final_manifest)
    REPORT_PATH.write_text(
        _build_report(frozen, test_metrics, rollout_metrics, final_manifest),
        encoding="utf-8",
    )
    print(f"TEST_METRICS={TEST_METRICS_PATH}")
    print(f"ROLLOUT_METRICS={ROLLOUT_METRICS_PATH}")
    print(f"REPORT={REPORT_PATH}")
    print(f"FROZEN_CONFIG_UNCHANGED={frozen_checksum_after}")


def _extract_dataset(
    split: DatasetSplit,
    hero_policy_ids: tuple[str, ...],
) -> EnemyDecisionDataset:
    return extract_enemy_decision_dataset(
        DatasetExtractionConfig(
            split=split,
            encounter_ids=ENCOUNTER_IDS,
            hero_policy_ids=hero_policy_ids,
            max_rounds=MAX_ROUNDS,
            enemy_wait_mode="none",
            enemy_movement_mode="recovery_only",
        )
    )


def _save_and_validate_model(
    model_name: str,
    trained: Any,
    validation_dataset: EnemyDecisionDataset,
    normalization: Any,
    normalization_checksum: str,
    validation_metrics: dict[str, object],
) -> dict[str, object]:
    metadata_path = ARTIFACT_ROOT / f"{model_name}.json"
    metadata = save_model_artifact(
        trained,
        metadata_path,
        normalization_checksum=normalization_checksum,
    )
    loaded = load_model_artifact(metadata_path)
    validation_metrics[model_name] = evaluate_imitation(
        loaded.model,
        validation_dataset,
        normalization,
        loaded.spec.feature_names,
    )
    return {
        "name": model_name,
        "metadata_path": str(metadata_path),
        "metadata_checksum": sha256_file(metadata_path),
        "model_file": metadata["model_file"],
        "model_checksum": metadata["model_checksum"],
        "model_type": metadata["scorer_spec"]["model_type"],
        "feature_names": metadata["scorer_spec"]["feature_names"],
        "training_seed": metadata["training_result"]["training_seed"],
        "selected_epoch": metadata["training_result"]["selected_epoch"],
        "validation_cross_entropy": metadata["training_result"][
            "best_validation_cross_entropy"
        ],
    }


def _frozen_config(
    *,
    train_dataset: EnemyDecisionDataset,
    validation_dataset: EnemyDecisionDataset,
    normalization_checksum: str,
    validation_checksum: str,
    model_entries: tuple[dict[str, object], ...],
) -> dict[str, object]:
    test_partition = DEFAULT_SEED_SPLIT_SPEC.partition(DatasetSplit.TEST)
    train_partition = DEFAULT_SEED_SPLIT_SPEC.partition(DatasetSplit.TRAIN)
    validation_partition = DEFAULT_SEED_SPLIT_SPEC.partition(DatasetSplit.VALIDATION)
    return {
        "status": "frozen_before_test",
        "test_accessed": False,
        "experiment_id": EXPERIMENT_ID,
        "scientific_question": (
            "Can a small nonlinear candidate scorer imitate the heuristic enemy ranking "
            "and generalize to unseen seeds and hero-policy distributions?"
        ),
        "foundation_commit": FOUNDATION_COMMIT,
        "checkpoint2_start_commit": FOUNDATION_COMMIT,
        "repository_head": _git_head(),
        "repository_dirty": _git_dirty(),
        "implementation_hashes": _implementation_hashes(),
        "runtime_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "feature_schema": {
            "id": ENEMY_DECISION_FEATURE_SCHEMA.schema_id,
            "version": ENEMY_DECISION_FEATURE_SCHEMA.version,
            "feature_count": len(ENEMY_DECISION_FEATURE_SCHEMA.names),
            "full_feature_names": list(ENEMY_DECISION_FEATURE_SCHEMA.names),
            "heuristic_visible_feature_names": list(HEURISTIC_VISIBLE_FEATURE_NAMES),
        },
        "dataset_schema": {
            "id": DATASET_FORMAT_ID,
            "version": DATASET_FORMAT_VERSION,
        },
        "split_spec": {
            "id": DEFAULT_SEED_SPLIT_SPEC.spec_id,
            "version": DEFAULT_SEED_SPLIT_SPEC.version,
            "train": [train_partition.start, train_partition.end],
            "validation": [validation_partition.start, validation_partition.end],
            "test": [test_partition.start, test_partition.end],
        },
        "encounter_ids": list(ENCOUNTER_IDS),
        "training_hero_policy_ids": list(TRAIN_HERO_POLICY_IDS),
        "validation_hero_policy_ids": list(TRAIN_HERO_POLICY_IDS),
        "test_in_distribution_hero_policy_ids": list(TRAIN_HERO_POLICY_IDS),
        "test_policy_shift_hero_policy_ids": list(POLICY_SHIFT_HERO_POLICY_IDS),
        "extraction": {
            "max_rounds": MAX_ROUNDS,
            "enemy_wait_mode": "none",
            "enemy_movement_mode": "recovery_only",
            "fresh_isolated_encounters": True,
            "route_attrition": False,
        },
        "dataset_counts": {
            "train": _dataset_counts(train_dataset),
            "validation": _dataset_counts(validation_dataset),
        },
        "normalization": {
            "path": str(NORMALIZATION_PATH),
            "checksum": normalization_checksum,
            "fit_split": DatasetSplit.TRAIN.value,
            "zero_variance_scale": 1.0,
        },
        "training": {
            "config": TRAINING_CONFIG.to_dict(),
            "linear_training_seed": LINEAR_TRAINING_SEED,
            "mlp_training_seeds": list(MLP_TRAINING_SEEDS),
            "ablation_training_seed": ABLATION_TRAINING_SEED,
            "stopping_rule": (
                "minimum validation candidate-set cross-entropy; stop after six "
                "epochs without improvement greater than 1e-5"
            ),
        },
        "model_artifacts": list(model_entries),
        "validation_metrics": {
            "path": str(VALIDATION_METRICS_PATH),
            "checksum": validation_checksum,
        },
        "test_metrics_predeclared": [
            "candidate-set cross-entropy",
            "top-1 teacher agreement",
            "mean reciprocal rank",
            "agreement by encounter",
            "agreement by enemy class",
            "agreement by hero policy",
            "candidate-count distribution",
            "decision count",
        ],
        "rollout_metrics_predeclared": [
            "enemy/player victory",
            "remaining hero HP",
            "remaining hero effort",
            "hero downs",
            "hero deaths",
            "mortal wounds",
            "rounds",
            "legacy shaped reward",
            "mark/guard/boss package diagnostics",
        ],
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "samples": BOOTSTRAP_SAMPLES,
            "interval": 0.95,
        },
        "top1_materiality_threshold": TOP1_MATERIALITY_THRESHOLD,
        "ablation": {
            "count": 1,
            "comparison": "full-39 versus heuristic-visible feature subset",
        },
        "content_hashes": _content_hashes(),
    }


def _validate_frozen_config(frozen: dict[str, Any]) -> None:
    if frozen.get("status") != "frozen_before_test":
        raise ValueError("TEST requires a configuration explicitly frozen before TEST.")
    if frozen.get("test_accessed") is not False:
        raise ValueError("Frozen configuration does not prove an untouched TEST state.")
    if frozen.get("foundation_commit") != FOUNDATION_COMMIT:
        raise ValueError("Frozen experiment references the wrong foundation commit.")
    if tuple(frozen.get("encounter_ids", ())) != ENCOUNTER_IDS:
        raise ValueError("Frozen encounter set differs from the declared experiment.")
    split = frozen.get("split_spec", {})
    expected_test = DEFAULT_SEED_SPLIT_SPEC.partition(DatasetSplit.TEST)
    if split.get("test") != [expected_test.start, expected_test.end]:
        raise ValueError("Frozen TEST seed range differs from the declared split.")


def _load_frozen_models(frozen: dict[str, Any]) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for entry in frozen["model_artifacts"]:
        metadata_path = Path(entry["metadata_path"])
        if sha256_file(metadata_path) != entry["metadata_checksum"]:
            raise ValueError(f"Model metadata changed after freeze: {entry['name']}")
        model = load_model_artifact(metadata_path)
        if model.metadata["model_checksum"] != entry["model_checksum"]:
            raise ValueError(f"Model binary changed after freeze: {entry['name']}")
        loaded[str(entry["name"])] = model
    return loaded


def _rollout_policy_specs(
    loaded_models: dict[str, Any],
    normalization: Any,
) -> tuple[RolloutPolicySpec, ...]:
    specs: list[RolloutPolicySpec] = [
        RolloutPolicySpec("heuristic_teacher", None, "none", "recovery_only"),
        RolloutPolicySpec(
            "production_static_learned_linear",
            production_enemy_decision_policy("learned_static"),
            production_enemy_wait_mode("learned_static"),
            production_enemy_movement_mode("learned_static"),
        ),
    ]
    for name in ("research_imitation_linear", *tuple(
        f"research_mlp_seed_{seed}" for seed in MLP_TRAINING_SEEDS
    )):
        loaded = loaded_models[name]
        specs.append(
            RolloutPolicySpec(
                name,
                ResearchScorerPolicy(
                    model=loaded.model,
                    normalization=normalization,
                    feature_names=loaded.spec.feature_names,
                ),
                "none",
                "recovery_only",
            )
        )
    return tuple(specs)


def _dataset_counts(dataset: EnemyDecisionDataset) -> dict[str, int]:
    return {
        "decision_count": len(dataset.records),
        "candidate_count": sum(len(record.candidates) for record in dataset.records),
        "episode_case_count": (
            len(dataset.config.encounter_ids)
            * len(dataset.config.hero_policy_ids)
            * len(DEFAULT_SEED_SPLIT_SPEC.seeds_for(dataset.config.split))
        ),
    }


def _validate_declared_encounters(definitions: GameDefinitions) -> None:
    available = set(definitions.encounters)
    missing = tuple(encounter_id for encounter_id in ENCOUNTER_IDS if encounter_id not in available)
    if missing:
        raise ValueError(f"Declared Checkpoint 2 encounters are missing: {missing}")


def _verify_foundation_head() -> None:
    head = _git_head()
    if head != FOUNDATION_COMMIT:
        raise ValueError(
            f"Checkpoint 2 expected foundation HEAD {FOUNDATION_COMMIT}, got {head}."
        )


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()


def _git_dirty() -> bool:
    output = subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True,
        encoding="utf-8",
    )
    return bool(output.strip())


def _implementation_hashes() -> dict[str, str]:
    paths = (
        Path("src/game/ml/features.py"),
        Path("src/game/ml/models.py"),
        Path("src/game/ml/training.py"),
        Path("src/game/ml/evaluation.py"),
        Path("src/game/dev/ml_lab.py"),
        Path("pyproject.toml"),
    )
    return {str(path): sha256_file(path) for path in paths}


def _content_hashes() -> dict[str, str]:
    return {
        str(path): sha256_file(path)
        for path in sorted(Path("data").glob("*.yaml"))
    }


def _build_report(
    frozen: dict[str, Any],
    test_metrics: dict[str, Any],
    rollout_metrics: dict[str, Any],
    final_manifest: dict[str, Any],
) -> str:
    validation = _read_json(VALIDATION_METRICS_PATH)
    linear_name = "research_imitation_linear"
    mlp_names = tuple(f"research_mlp_seed_{seed}" for seed in MLP_TRAINING_SEEDS)
    visible_name = f"research_mlp_heuristic_visible_seed_{ABLATION_TRAINING_SEED}"

    validation_linear = validation[linear_name]["overall"]
    validation_mlp = [validation[name]["overall"] for name in mlp_names]
    id_linear = test_metrics["in_distribution"][linear_name]["overall"]
    id_mlp = [test_metrics["in_distribution"][name]["overall"] for name in mlp_names]
    shift_linear = test_metrics["policy_shift"][linear_name]["overall"]
    shift_mlp = [test_metrics["policy_shift"][name]["overall"] for name in mlp_names]
    id_visible = test_metrics["in_distribution"][visible_name]["overall"]
    shift_visible = test_metrics["policy_shift"][visible_name]["overall"]

    id_mlp_agreement = mean(float(item["top1_agreement"]) for item in id_mlp)
    shift_mlp_agreement = mean(float(item["top1_agreement"]) for item in shift_mlp)
    linear_id_agreement = float(id_linear["top1_agreement"])
    nonlinear_delta = id_mlp_agreement - linear_id_agreement
    material = abs(nonlinear_delta) >= TOP1_MATERIALITY_THRESHOLD
    nonlinear_answer = (
        f"MLP mean top-1 differs from linear by {nonlinear_delta:+.4f}; "
        + ("this crosses" if material else "this does not cross")
        + f" the predeclared {TOP1_MATERIALITY_THRESHOLD:.2%} materiality threshold."
    )
    shift_delta = shift_mlp_agreement - id_mlp_agreement
    learning_only_delta = id_mlp_agreement - float(id_visible["top1_agreement"])

    weakest_encounter = _weakest_group(
        test_metrics["in_distribution"][mlp_names[0]]["by_encounter"]
    )
    weakest_class = _weakest_group(
        test_metrics["in_distribution"][mlp_names[0]]["by_enemy_class"]
    )
    frozen_checksum = final_manifest["frozen_config"]["checksum"]

    lines = [
        "# The Charter — Checkpoint 2 Supervised Candidate-Ranking Report",
        "",
        "## A. Starting / provenance state",
        f"- Checkpoint 1 commit: `{FOUNDATION_COMMIT}`",
        f"- Checkpoint 2 starting commit: `{FOUNDATION_COMMIT}`",
        f"- Python: `{frozen['runtime_versions']['python']}`",
        f"- Torch: `{frozen['runtime_versions']['torch']}`",
        f"- Repository dirty during experiment: `{frozen['repository_dirty']}`",
        "",
        "## B. Experiment definition",
        "- Question: can a small nonlinear candidate scorer imitate the heuristic teacher "
        "and generalize across unseen seeds and hero-policy distributions?",
        f"- Encounters: {len(ENCOUNTER_IDS)} fixed isolated encounter IDs.",
        f"- Train decisions: {frozen['dataset_counts']['train']['decision_count']}",
        f"- Validation decisions: {frozen['dataset_counts']['validation']['decision_count']}",
        "- Features: schema-v1 39 features; no IDs, seeds, reward, outcome, future state, "
        "or heuristic candidate score.",
        "- Normalization: mean/std fitted only on TRAIN candidates; zero variance -> scale 1.",
        "- Linear: normalized features -> one candidate logit.",
        "- MLP: 39 -> 64 ReLU -> 32 ReLU -> 1 candidate logit.",
        f"- MLP training seeds: {', '.join(str(seed) for seed in MLP_TRAINING_SEEDS)}.",
        "- Stopping: best validation CE, patience 6, min delta 1e-5, max 50 epochs.",
        "",
        "## C. Validation results",
        _offline_line("Linear", validation_linear),
        *[
            _offline_line(f"MLP seed {seed}", metrics)
            for seed, metrics in zip(MLP_TRAINING_SEEDS, validation_mlp, strict=True)
        ],
        _offline_line("Heuristic-visible ablation", validation[visible_name]["overall"]),
        "",
        "## D. Frozen TEST configuration",
        f"- Frozen config: `{FROZEN_CONFIG_PATH}`",
        f"- Frozen config SHA-256: `{frozen_checksum}`",
        "- The train command exited after writing this file and reported "
        "`TEST_ACCESSED=false`.",
        "- The test command verified this checksum before and after TEST; it was unchanged.",
        "",
        "## E. Held-out TEST results",
        "### In-distribution hero policies",
        _offline_line("Linear", id_linear),
        *[
            _offline_line(f"MLP seed {seed}", metrics)
            for seed, metrics in zip(MLP_TRAINING_SEEDS, id_mlp, strict=True)
        ],
        _offline_line("Heuristic-visible ablation", id_visible),
        "### Policy-shift hero policies",
        _offline_line("Linear", shift_linear),
        *[
            _offline_line(f"MLP seed {seed}", metrics)
            for seed, metrics in zip(MLP_TRAINING_SEEDS, shift_mlp, strict=True)
        ],
        _offline_line("Heuristic-visible ablation", shift_visible),
        "",
        "## F. Simulator rollout results",
        _rollout_section(rollout_metrics),
        "",
        "## G. Interpretation",
        f"1. Nonlinearity: {nonlinear_answer}",
        "2. Simulator similarity: use the paired bootstrap intervals in rollout_metrics.json; "
        "imitation accuracy is not treated as strategic superiority.",
        f"3. First-seed MLP weakest encounter: {weakest_encounter}; weakest enemy class: "
        f"{weakest_class}.",
        f"4. Policy shift changed mean MLP top-1 by {shift_delta:+.4f} relative to held-out "
        "in-distribution TEST.",
        f"5. Full features changed in-distribution top-1 by {learning_only_delta:+.4f} versus "
        "the single heuristic-visible ablation.",
        "6. Reward optimization: no. This experiment validates supervised imitation "
        "infrastructure only; legacy shaped reward remains diagnostic, not an objective.",
        "",
        "## H. Quality / regression results",
        "- Filled by the execution report after repository-wide pytest/Ruff/mypy/diff checks.",
        "- Normal gameplay imports are tested without permitting a torch import.",
        "",
        "## I. Artifacts",
        f"- Frozen manifest: `{FROZEN_CONFIG_PATH}`",
        f"- Normalization: `{NORMALIZATION_PATH}`",
        f"- TEST metrics: `{TEST_METRICS_PATH}`",
        f"- Rollout metrics: `{ROLLOUT_METRICS_PATH}`",
        f"- Final manifest: `{FINAL_MANIFEST_PATH}`",
        "- Model binaries and generated experiment outputs live under ignored `artifacts/ml/`.",
        "",
        "## J. Risks / open questions",
        "- Fresh isolated encounters do not establish route/attrition generalization.",
        "- Production static learned-linear uses its native package timing in rollout, while "
        "research imitation models use teacher timing; interpret that baseline accordingly.",
        "- Checkpoint 2 implementation code was hashed in the frozen config while the worktree "
        "was dirty, so the hashes are part of experiment provenance.",
        "",
        "## K. Recommended Checkpoint 3",
        "Run one reward-definition / objective-design study before any reward-optimized neural "
        "training, using the held-out imitation baseline as the fixed reference.",
        "",
    ]
    return "\n".join(lines)


def _offline_line(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"- {label}: CE {float(metrics['cross_entropy']):.4f}, "
        f"top-1 {float(metrics['top1_agreement']):.2%}, "
        f"MRR {float(metrics['mean_reciprocal_rank']):.4f}, "
        f"n={int(metrics['decision_count'])}."
    )


def _rollout_section(rollout_metrics: dict[str, Any]) -> str:
    lines: list[str] = []
    for group_name in ("in_distribution", "policy_shift"):
        group = rollout_metrics[group_name]
        lines.append(f"### {group_name.replace('_', ' ').title()}")
        lines.append(f"Paired cases per policy: {group['case_count_per_policy']}")
        for policy_name, summary in group["by_policy"].items():
            lines.append(
                f"- {policy_name}: enemy win {summary['enemy_win']['mean']:.2%}, "
                f"hero HP {summary['final_hero_hp']['mean']:.2f}, downs "
                f"{summary['hero_downs']['mean']:.3f}, rounds "
                f"{summary['rounds']['mean']:.3f}."
            )
        lines.append(
            "Paired mean differences and bootstrap 95% CIs versus the heuristic teacher are "
            "recorded in rollout_metrics.json."
        )
    return "\n".join(lines)


def _weakest_group(groups: dict[str, Any]) -> str:
    if not groups:
        return "none"
    key = min(groups, key=lambda name: float(groups[name]["top1_agreement"]))
    return f"{key} ({float(groups[key]['top1_agreement']):.2%})"


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
