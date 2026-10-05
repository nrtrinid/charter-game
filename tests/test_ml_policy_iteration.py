# ruff: noqa: E402

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from game.campaign.company import create_new_company
from game.combat.enemy_decision import EnemyDecisionRuntimeContext, explain_enemy_decision
from game.data.loaders import load_game_definitions
from game.dev import ml_cp4
from game.expedition.cave import create_encounter_combat
from game.ml.counterfactual import CounterfactualOutcome, OutcomePreference, compare_outcomes
from game.ml.distribution_audit import (
    build_distribution_audit,
    reconstruct_cp3_distribution_records,
)
from game.ml.models import mlp_scorer_spec
from game.ml.outcome_dataset import (
    SamplingSpec,
    build_counterfactual_dataset,
    decisive_preference_pairs,
)
from game.ml.outcome_features import CP2_FEATURE_AUDIT, CP3_FEATURE_NAMES
from game.ml.outcome_ranking import (
    OutcomeNormalization,
    PairwiseTrainingConfig,
    choose_candidate_index,
    fit_outcome_normalization,
    load_outcome_ranker,
    save_outcome_normalization,
    save_outcome_ranker,
    train_pairwise_ranker,
)
from game.ml.policy_iteration import (
    FrozenOutcomeRankerPolicy,
    audit_policy_dataset,
    build_policy_counterfactual_dataset,
    collect_policy_source_decisions,
    load_policy_dataset,
    run_policy_counterfactual_branch,
    sample_policy_source_decisions,
    save_policy_dataset,
    to_ranking_dataset,
    verify_policy_control_branch,
)
from game.ml.splits import DatasetSplit

_DEFAULT_CONTEXT = EnemyDecisionRuntimeContext()


class LastLegalPolicy:
    policy_id = "test_pi1_last_legal"

    def __init__(self) -> None:
        self.calls = 0

    def choose(self, state, definitions, enemy_id, runtime_context=_DEFAULT_CONTEXT):
        self.calls += 1
        trace = explain_enemy_decision(state, definitions, enemy_id, runtime_context)
        if trace is None or not trace.candidates:
            return None
        return trace.candidates[-1]


@pytest.fixture(scope="module")
def definitions():
    return load_game_definitions()


@pytest.fixture(scope="module")
def pi1_sources(definitions):
    policy = LastLegalPolicy()
    decisions = collect_policy_source_decisions(
        definitions,
        source_policy=policy,
        source_policy_id=policy.policy_id,
        split=DatasetSplit.TRAIN,
        encounter_id="shallow_cave",
        hero_policy_id="naive",
        seed=1,
    )
    assert len(decisions) > 1
    return decisions


@pytest.fixture(scope="module")
def pi1_source(pi1_sources):
    return next(source for source in pi1_sources if len(source.candidates) > 1)


def _small_policy_dataset(definitions, split: DatasetSplit, seed: int, count: int = 2):
    policy = LastLegalPolicy()
    return build_policy_counterfactual_dataset(
        definitions,
        source_policy=policy,
        source_policy_id=policy.policy_id,
        continuation_policy=policy,
        continuation_policy_id=policy.policy_id,
        split=split,
        encounter_ids=("shallow_cave",),
        hero_policy_ids=("naive",),
        seeds=(seed,),
        sampling_spec=SamplingSpec(states_per_encounter_policy=count),
    )


def test_pi1_checksum_guard_is_used_before_artifact_loading(tmp_path: Path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"frozen-model")
    actual = ml_cp4.sha256_file(artifact)
    ml_cp4._require_sha256(artifact, actual, "test artifact")
    with pytest.raises(ValueError, match="checksum-mismatched"):
        ml_cp4._require_sha256(artifact, "0" * 64, "test artifact")


def test_real_pi1_artifact_matches_completed_cp3_manifest_when_present():
    if not ml_cp4.CP3_MANIFEST_PATH.exists():
        pytest.skip("Ignored CP3 research artifacts are not present in this checkout.")
    policy, identity = ml_cp4._load_verified_pi1()
    assert policy.policy_id == ml_cp4.SOURCE_POLICY_ID
    assert identity["model_checksum"] == ml_cp4.CP3_PRIMARY_MODEL_CHECKSUM


def test_source_trajectories_are_generated_by_supplied_pi1(pi1_sources):
    assert pi1_sources
    for source in pi1_sources:
        assert source.source_policy_id == LastLegalPolicy.policy_id
        assert source.source_choice == source.candidates[-1]


def test_forced_action_continuation_returns_to_pi1(definitions, pi1_sources):
    source = pi1_sources[0]
    continuation = LastLegalPolicy()
    alternate = source.candidates[0]
    result = run_policy_counterfactual_branch(
        definitions,
        source,
        alternate,
        continuation_policy=continuation,
        continuation_policy_id=continuation.policy_id,
    )
    assert result.continuation_policy_id == continuation.policy_id
    assert continuation.calls > source.action_index + 1


def test_pi1_control_branch_reproduces_original_trajectory(definitions, pi1_source):
    continuation = LastLegalPolicy()
    result = verify_policy_control_branch(
        definitions,
        pi1_source,
        continuation_policy=continuation,
        continuation_policy_id=continuation.policy_id,
    )
    assert result.outcome == pi1_source.baseline_outcome
    assert result.final_state_digest == pi1_source.baseline_final_state_digest
    assert result.episode_digest == pi1_source.baseline_episode_digest


def test_policy_replay_preserves_state_rng_and_candidate_identity(definitions, pi1_source):
    continuation = LastLegalPolicy()
    result = verify_policy_control_branch(
        definitions,
        pi1_source,
        continuation_policy=continuation,
        continuation_policy_id=continuation.policy_id,
    )
    assert result.replay_state_digest == pi1_source.state_digest
    assert result.replay_rng_digest == pi1_source.rng_digest
    assert result.replay_candidates == pi1_source.candidates


def test_policy_branch_mutations_are_isolated(definitions, pi1_source):
    alternate = next(
        candidate for candidate in pi1_source.candidates if candidate != pi1_source.source_choice
    )
    first = run_policy_counterfactual_branch(
        definitions,
        pi1_source,
        alternate,
        continuation_policy=LastLegalPolicy(),
        continuation_policy_id=LastLegalPolicy.policy_id,
    )
    second = run_policy_counterfactual_branch(
        definitions,
        pi1_source,
        alternate,
        continuation_policy=LastLegalPolicy(),
        continuation_policy_id=LastLegalPolicy.policy_id,
    )
    assert first.outcome == second.outcome
    assert first.final_state_digest == second.final_state_digest
    assert first.episode_digest == second.episode_digest


def test_sampling_is_deterministic_and_outcome_blind(definitions, pi1_source):
    first, available_first = sample_policy_source_decisions(
        definitions,
        source_policy=LastLegalPolicy(),
        source_policy_id=LastLegalPolicy.policy_id,
        split=DatasetSplit.TRAIN,
        encounter_ids=("shallow_cave",),
        hero_policy_ids=("naive",),
        seeds=(1, 2),
        sampling_spec=SamplingSpec(states_per_encounter_policy=2),
    )
    second, available_second = sample_policy_source_decisions(
        definitions,
        source_policy=LastLegalPolicy(),
        source_policy_id=LastLegalPolicy.policy_id,
        split=DatasetSplit.TRAIN,
        encounter_ids=("shallow_cave",),
        hero_policy_ids=("naive",),
        seeds=(1, 2),
        sampling_spec=SamplingSpec(states_per_encounter_policy=2),
    )
    assert available_first == available_second
    assert first == second
    changed_outcome = replace(
        pi1_source,
        baseline_outcome=CounterfactualOutcome(
            enemy_win=not pi1_source.baseline_outcome.enemy_win,
            final_hero_hp=999,
            final_hero_effort=999,
            terminal_hero_downs=0,
            hero_deaths=0,
            mortal_wounds=0,
            enemy_caused_effort_drain=0,
            rounds_elapsed=1,
        ),
    )
    from game.ml import policy_iteration

    assert policy_iteration._sample_key(changed_outcome) == policy_iteration._sample_key(pi1_source)


def test_policy_dataset_round_trips_with_source_policy_identity(definitions, tmp_path: Path):
    dataset = _small_policy_dataset(definitions, DatasetSplit.TRAIN, 1, count=1)
    path = tmp_path / "policy-dataset.json"
    save_policy_dataset(dataset, path)
    loaded = load_policy_dataset(path)
    assert loaded == dataset
    assert loaded.source_policy_id == LastLegalPolicy.policy_id
    assert loaded.continuation_policy_id == LastLegalPolicy.policy_id


def test_cp4_normalization_is_train_only(definitions):
    validation = _small_policy_dataset(definitions, DatasetSplit.VALIDATION, 10001, count=1)
    with pytest.raises(ValueError, match="TRAIN"):
        fit_outcome_normalization(to_ranking_dataset(validation))


def test_cp3_to_cp4_distribution_audit_is_deterministic(definitions):
    cp3 = build_counterfactual_dataset(
        definitions,
        split=DatasetSplit.TRAIN,
        encounter_ids=("shallow_cave",),
        hero_policy_ids=("naive",),
        seeds=(1,),
        sampling_spec=SamplingSpec(states_per_encounter_policy=1),
    )
    cp4 = _small_policy_dataset(definitions, DatasetSplit.TRAIN, 1, count=1)
    baseline = reconstruct_cp3_distribution_records(definitions, cp3)
    first = build_distribution_audit(cp3, baseline, cp4)
    second = build_distribution_audit(cp3, baseline, cp4)
    assert first == second
    assert first["state_distribution"]["cp3"]["decision_count"] == 1
    assert first["state_distribution"]["cp4"]["decision_count"] == 1


def test_cp4_observation_excludes_outcomes_ids_and_prescriptive_cp2_features():
    forbidden = (
        "enemy_win",
        "reward",
        "outcome",
        "seed",
        "actor_id",
        "target_id",
        "encounter_id",
        "hero_policy_id",
    )
    assert all(not any(token in name for token in forbidden) for name in CP3_FEATURE_NAMES)
    prescriptive = {
        item.name for item in CP2_FEATURE_AUDIT if item.classification == "prescriptive_excluded"
    }
    assert prescriptive.isdisjoint(CP3_FEATURE_NAMES)
    assert len(CP3_FEATURE_NAMES) == 81


def test_ties_and_incomparables_remain_excluded_from_ranking_labels(definitions):
    dataset = _small_policy_dataset(definitions, DatasetSplit.TRAIN, 1, count=2)
    ranking = to_ranking_dataset(dataset)
    pairs = decisive_preference_pairs(ranking)
    for pair in pairs:
        record = ranking.records[pair.record_index]
        assert (
            compare_outcomes(
                record.candidates[pair.preferred_index].outcome,
                record.candidates[pair.other_index].outcome,
            )
            is OutcomePreference.FIRST
        )


def test_win_still_strictly_dominates_loss():
    loss = CounterfactualOutcome(False, 0, 0, 4, 4, 8, 99, 20)
    win = CounterfactualOutcome(True, 100, 100, 0, 0, 0, 0, 1)
    assert compare_outcomes(win, loss) is OutcomePreference.FIRST
    assert compare_outcomes(loss, win) is OutcomePreference.SECOND


def test_source_seed_groups_remain_split_isolated(definitions):
    train = collect_policy_source_decisions(
        definitions,
        source_policy=LastLegalPolicy(),
        source_policy_id=LastLegalPolicy.policy_id,
        split=DatasetSplit.TRAIN,
        encounter_id="shallow_cave",
        hero_policy_id="naive",
        seed=1,
    )
    validation = collect_policy_source_decisions(
        definitions,
        source_policy=LastLegalPolicy(),
        source_policy_id=LastLegalPolicy.policy_id,
        split=DatasetSplit.VALIDATION,
        encounter_id="shallow_cave",
        hero_policy_id="naive",
        seed=10001,
    )
    assert {item.split for item in train} == {DatasetSplit.TRAIN}
    assert {item.split for item in validation} == {DatasetSplit.VALIDATION}
    assert {item.simulation_seed for item in train}.isdisjoint(
        {item.simulation_seed for item in validation}
    )


def test_source_policy_dominance_audit_uses_pi1_index(definitions):
    dataset = _small_policy_dataset(definitions, DatasetSplit.TRAIN, 1, count=2)
    audit = audit_policy_dataset(dataset)
    assert audit.source_states == len(dataset.records)
    assert 0 <= audit.source_policy_dominated_states <= audit.source_states
    assert 0 <= audit.source_policy_loss_to_win_opportunities <= audit.source_states
    assert 0 <= audit.source_policy_win_to_loss_risks <= audit.source_states


def test_pi2_model_reload_preserves_deterministic_inference(definitions, tmp_path: Path):
    train = _small_policy_dataset(definitions, DatasetSplit.TRAIN, 1, count=3)
    validation = _small_policy_dataset(definitions, DatasetSplit.VALIDATION, 10001, count=2)
    train_ranking = to_ranking_dataset(train)
    validation_ranking = to_ranking_dataset(validation)
    assert decisive_preference_pairs(train_ranking)
    assert decisive_preference_pairs(validation_ranking)
    normalization = fit_outcome_normalization(train_ranking)
    normalization_checksum = save_outcome_normalization(
        normalization,
        tmp_path / "normalization.json",
    )
    trained = train_pairwise_ranker(
        mlp_scorer_spec(CP3_FEATURE_NAMES),
        train_ranking,
        validation_ranking,
        normalization,
        training_seed=20261007,
        config=PairwiseTrainingConfig(
            max_epochs=3,
            early_stopping_patience=2,
            batch_size=64,
        ),
    )
    metadata_path = tmp_path / "pi2.json"
    save_outcome_ranker(
        trained,
        metadata_path,
        normalization_checksum=normalization_checksum,
    )
    loaded = load_outcome_ranker(metadata_path)
    before = [
        choose_candidate_index(trained.model, index, validation_ranking, normalization)
        for index in range(len(validation_ranking.records))
    ]
    after = [
        choose_candidate_index(loaded.model, index, validation_ranking, normalization)
        for index in range(len(validation_ranking.records))
    ]
    assert before == after


def test_frozen_learned_policy_can_only_return_simulator_legal_candidate(definitions):
    model = torch.nn.Linear(len(CP3_FEATURE_NAMES), 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    normalization = OutcomeNormalization(
        feature_names=CP3_FEATURE_NAMES,
        means=(0.0,) * len(CP3_FEATURE_NAMES),
        scales=(1.0,) * len(CP3_FEATURE_NAMES),
        zero_variance=(False,) * len(CP3_FEATURE_NAMES),
    )
    policy = FrozenOutcomeRankerPolicy(model, normalization, policy_id="legal-test")
    state = create_encounter_combat(create_new_company(definitions), definitions, "shallow_cave")
    enemy_id = next(iter(state.enemies))
    context = EnemyDecisionRuntimeContext(
        initiative_actor_ids=(enemy_id,),
        current_turn_index=0,
    )
    trace = explain_enemy_decision(state, definitions, enemy_id, context)
    chosen = policy.choose(state, definitions, enemy_id, context)
    assert trace is not None
    assert chosen in trace.candidates


def test_normal_gameplay_import_does_not_load_torch():
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import game.main, sys; print('torch' in sys.modules)",
        ],
        text=True,
    ).strip()
    assert output == "False"


def test_future_confirmation_range_is_reserved_not_evaluated():
    protocol = ml_cp4._predeclared_protocol(
        {
            "model_name": ml_cp4.CP3_PRIMARY_MODEL,
            "model_checksum": ml_cp4.CP3_PRIMARY_MODEL_CHECKSUM,
        }
    )
    assert protocol["splits"]["reserved_future_confirmation"] == [30001, 30128]
    assert protocol["reserved_future_confirmation_evaluated"] is False
    assert max(ml_cp4.FULL_POLICY_TEST_SEEDS) < 30001
