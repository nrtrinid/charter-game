# ruff: noqa: E402

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from game.campaign.company import create_new_company
from game.combat.enemy_decision import explain_enemy_decision
from game.data.loaders import load_game_definitions
from game.expedition.cave import create_encounter_combat
from game.ml.dataset import (
    DATASET_FORMAT_ID,
    DATASET_FORMAT_VERSION,
    DatasetExtractionConfig,
    EnemyDecisionCandidateRecord,
    EnemyDecisionDataset,
    EnemyDecisionDatasetRecord,
    EnemyDecisionOutcomeMetadata,
)
from game.ml.evaluation import ResearchScorerPolicy
from game.ml.features import (
    ENEMY_DECISION_FEATURE_SCHEMA,
    HEURISTIC_VISIBLE_FEATURE_NAMES,
    LEARNING_ONLY_FEATURE_NAMES,
)
from game.ml.models import (
    LinearCandidateScorer,
    build_candidate_scorer,
    linear_scorer_spec,
    mlp_scorer_spec,
)
from game.ml.splits import DEFAULT_SEED_SPLIT_SPEC, DatasetSplit
from game.ml.training import (
    EpochResult,
    FeatureNormalization,
    PreparedDecisionGroup,
    PreparedDecisionGroups,
    TrainedCandidateScorer,
    TrainingConfig,
    TrainingResult,
    candidate_set_cross_entropy,
    fit_train_normalization,
    load_model_artifact,
    prepared_cross_entropy,
    save_model_artifact,
)


def test_mlp_scorer_returns_one_scalar_per_candidate() -> None:
    spec = mlp_scorer_spec(ENEMY_DECISION_FEATURE_SCHEMA.names)
    model = build_candidate_scorer(spec)
    features = torch.zeros((7, len(spec.feature_names)), dtype=torch.float32)

    logits = model(features)

    assert logits.shape == (7,)


def test_candidate_set_loss_accepts_variable_candidate_counts() -> None:
    model = LinearCandidateScorer(1)
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        model.linear.bias.zero_()

    two_candidates = torch.tensor([[[0.0], [1.0]]])
    five_candidates = torch.tensor([[[0.0], [1.0], [2.0], [3.0], [4.0]]])

    assert torch.isfinite(candidate_set_cross_entropy(model, two_candidates, torch.tensor([1])))
    assert torch.isfinite(candidate_set_cross_entropy(model, five_candidates, torch.tensor([4])))


def test_candidate_set_loss_targets_chosen_index() -> None:
    model = LinearCandidateScorer(1)
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        model.linear.bias.zero_()
    features = torch.tensor([[[0.0], [1.0], [2.0]]])
    chosen = torch.tensor([2])

    actual = candidate_set_cross_entropy(model, features, chosen)
    expected = F.cross_entropy(torch.tensor([[0.0, 1.0, 2.0]]), chosen)

    assert torch.allclose(actual, expected)


def test_prepared_cross_entropy_weights_decisions_not_candidate_rows() -> None:
    model = LinearCandidateScorer(1)
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        model.linear.bias.zero_()
    first_features = torch.tensor([[[0.0], [2.0]]])
    second_features = torch.tensor([[[5.0], [0.0], [0.0], [0.0], [0.0]]])
    first_target = torch.tensor([0])
    second_target = torch.tensor([0])
    groups = PreparedDecisionGroups(
        feature_names=("formation",),
        groups=(
            PreparedDecisionGroup(2, (0,), first_features, first_target),
            PreparedDecisionGroup(5, (1,), second_features, second_target),
        ),
        decision_count=2,
    )

    actual = prepared_cross_entropy(model, groups, batch_size=16)
    expected = (
        candidate_set_cross_entropy(model, first_features, first_target).item()
        + candidate_set_cross_entropy(model, second_features, second_target).item()
    ) / 2

    assert actual == pytest.approx(expected)


def test_normalization_is_fit_from_train_only() -> None:
    train = _dataset(DatasetSplit.TRAIN, (2, 4))
    validation = _dataset(DatasetSplit.VALIDATION, (100, 200))

    normalization = fit_train_normalization(train)
    original_means = normalization.means
    _ = normalization.normalize_tensor(torch.tensor([[100.0] + [0.0] * 38]))

    assert normalization.means == original_means
    assert normalization.means[0] == pytest.approx(3.0)
    with pytest.raises(ValueError, match="TRAIN split"):
        fit_train_normalization(validation)


def test_zero_variance_normalization_uses_unit_scale() -> None:
    dataset = _dataset(DatasetSplit.TRAIN, (5, 5))

    normalization = fit_train_normalization(dataset)

    assert normalization.zero_variance[0] is True
    assert normalization.scales[0] == 1.0
    assert all(normalization.scales[index] == 1.0 for index in range(1, 39))


def test_validation_and_test_cannot_fit_normalization() -> None:
    for split in (DatasetSplit.VALIDATION, DatasetSplit.TEST):
        with pytest.raises(ValueError, match="TRAIN split"):
            fit_train_normalization(_dataset(split, (1, 2)))


def test_model_round_trip_preserves_deterministic_inference(tmp_path: Path) -> None:
    torch.manual_seed(9)
    spec = mlp_scorer_spec(ENEMY_DECISION_FEATURE_SCHEMA.names)
    model = build_candidate_scorer(spec)
    trained = _trained(spec, model)
    metadata_path = tmp_path / "model.json"
    features = torch.arange(78, dtype=torch.float32).reshape(2, 39)
    with torch.no_grad():
        before = model(features).clone()

    save_model_artifact(trained, metadata_path, normalization_checksum="normalization")
    loaded = load_model_artifact(metadata_path)
    with torch.no_grad():
        after = loaded.model(features)

    assert torch.equal(before, after)


def test_loaded_artifact_rejects_incompatible_feature_schema(tmp_path: Path) -> None:
    spec = linear_scorer_spec(ENEMY_DECISION_FEATURE_SCHEMA.names)
    model = build_candidate_scorer(spec)
    metadata_path = tmp_path / "model.json"
    save_model_artifact(_trained(spec, model), metadata_path, normalization_checksum="n")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["feature_schema_version"] = 999
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Incompatible feature schema version"):
        load_model_artifact(metadata_path)


def test_research_policy_can_only_return_simulator_candidate() -> None:
    definitions = load_game_definitions()
    company = create_new_company(definitions)
    state = create_encounter_combat(company, definitions, "road_bandits")
    enemy = next(candidate for candidate in state.enemies.values() if candidate.can_act())
    trace = explain_enemy_decision(state, definitions, enemy.actor_id)
    assert trace is not None and trace.candidates
    model = LinearCandidateScorer(39)
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()
    normalization = FeatureNormalization(
        means=(0.0,) * 39,
        scales=(1.0,) * 39,
        zero_variance=(False,) * 39,
    )
    policy = ResearchScorerPolicy(model, normalization, ENEMY_DECISION_FEATURE_SCHEMA.names)

    choice = policy.choose(state, definitions, enemy.actor_id)

    assert choice is not None
    assert choice in trace.candidates
    assert choice == trace.candidates[0]


def test_fixed_model_and_input_are_deterministic() -> None:
    torch.manual_seed(123)
    model = build_candidate_scorer(mlp_scorer_spec(ENEMY_DECISION_FEATURE_SCHEMA.names))
    features = torch.ones((4, 39), dtype=torch.float32)

    with torch.no_grad():
        first = model(features)
        second = model(features)

    assert torch.equal(first, second)


def test_normal_game_import_does_not_require_torch() -> None:
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise RuntimeError('normal gameplay imported torch')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import game.main
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path.cwd() / "src")

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_heuristic_visible_ablation_is_explicit_and_bounded() -> None:
    full = set(ENEMY_DECISION_FEATURE_SCHEMA.names)
    visible = set(HEURISTIC_VISIBLE_FEATURE_NAMES)
    learning_only = set(LEARNING_ONLY_FEATURE_NAMES)

    assert len(full) == 39
    assert len(visible) == 11
    assert visible.isdisjoint(learning_only)
    assert visible | learning_only == full


def _dataset(split: DatasetSplit, first_feature_values: tuple[int, ...]) -> EnemyDecisionDataset:
    seed = {
        DatasetSplit.TRAIN: 1,
        DatasetSplit.VALIDATION: 10001,
        DatasetSplit.TEST: 20001,
    }[split]
    candidates = tuple(
        EnemyDecisionCandidateRecord(
            skill_id=f"skill_{index}",
            target_id=f"target_{index}",
            skill_order=index,
            target_order=index,
            skill_tags=(),
            feature_values=(value,) + (0,) * 38,
        )
        for index, value in enumerate(first_feature_values)
    )
    record = EnemyDecisionDatasetRecord(
        feature_schema_id=ENEMY_DECISION_FEATURE_SCHEMA.schema_id,
        feature_schema_version=ENEMY_DECISION_FEATURE_SCHEMA.version,
        split=split,
        encounter_id="road_bandits",
        encounter_name="Road Bandits",
        simulation_seed=seed,
        hero_policy_id="naive",
        enemy_id="enemy",
        enemy_class_id="bandit",
        enemy_tags=("bandit",),
        round_number=1,
        action_index=0,
        candidates=candidates,
        chosen_index=0,
        outcome=EnemyDecisionOutcomeMetadata(0, "heroes", 0),
    )
    return EnemyDecisionDataset(
        format_id=DATASET_FORMAT_ID,
        format_version=DATASET_FORMAT_VERSION,
        feature_schema_id=ENEMY_DECISION_FEATURE_SCHEMA.schema_id,
        feature_schema_version=ENEMY_DECISION_FEATURE_SCHEMA.version,
        split_spec_id=DEFAULT_SEED_SPLIT_SPEC.spec_id,
        split_spec_version=DEFAULT_SEED_SPLIT_SPEC.version,
        config=DatasetExtractionConfig(
            split=split,
            encounter_ids=("road_bandits",),
            hero_policy_ids=("naive",),
            seeds=(seed,),
        ),
        records=(record,),
    )


def _trained(spec: object, model: object) -> TrainedCandidateScorer:
    return TrainedCandidateScorer(
        spec=spec,
        model=model,
        training_config=TrainingConfig(max_epochs=1, early_stopping_patience=1),
        result=TrainingResult(
            training_seed=1,
            selected_epoch=1,
            epochs_ran=1,
            best_validation_cross_entropy=0.0,
            history=(EpochResult(1, 0.0, 0.0),),
        ),
    )

