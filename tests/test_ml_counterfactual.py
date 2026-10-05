# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from game.data.loaders import load_game_definitions
from game.ml.counterfactual import (
    CandidateKey,
    CounterfactualOutcome,
    OutcomePreference,
    collect_heuristic_source_decisions,
    compare_outcomes,
    run_counterfactual_branch,
    verify_heuristic_control_branch,
)
from game.ml.models import linear_scorer_spec, mlp_scorer_spec
from game.ml.outcome_dataset import (
    SamplingSpec,
    audit_counterfactual_dataset,
    build_counterfactual_dataset,
    decisive_preference_pairs,
    load_counterfactual_dataset,
    save_counterfactual_dataset,
)
from game.ml.outcome_features import CP2_FEATURE_AUDIT, CP3_FEATURE_NAMES, CP3_FEATURE_SCHEMA
from game.ml.outcome_ranking import (
    PairwiseTrainingConfig,
    choose_candidate_index,
    fit_outcome_normalization,
    load_outcome_ranker,
    save_outcome_normalization,
    save_outcome_ranker,
    train_pairwise_ranker,
)
from game.ml.splits import DatasetSplit


@pytest.fixture(scope="module")
def definitions():
    return load_game_definitions()


@pytest.fixture(scope="module")
def source(definitions):
    decisions = collect_heuristic_source_decisions(
        definitions,
        split=DatasetSplit.TRAIN,
        encounter_id="shallow_cave",
        hero_policy_id="naive",
        seed=1,
    )
    return next(decision for decision in decisions if len(decision.candidates) > 1)


def test_control_branch_reproduces_state_rng_candidates_and_trajectory(definitions, source):
    result = verify_heuristic_control_branch(definitions, source)
    assert result.replay_state_digest == source.state_digest
    assert result.replay_rng_digest == source.rng_digest
    assert result.replay_candidates == source.candidates
    assert result.outcome == source.baseline_outcome
    assert result.final_state_digest == source.baseline_final_state_digest
    assert result.episode_digest == source.baseline_episode_digest
    assert result.continuation_policy_id == "heuristic"


def test_branch_is_legal_deterministic_and_isolated(definitions, source):
    alternate = next(
        candidate for candidate in source.candidates if candidate != source.heuristic_choice
    )
    first = run_counterfactual_branch(definitions, source, alternate)
    second = run_counterfactual_branch(definitions, source, alternate)
    assert first.outcome == second.outcome
    assert first.final_state_digest == second.final_state_digest
    assert first.episode_digest == second.episode_digest
    source_again = collect_heuristic_source_decisions(
        definitions,
        split=DatasetSplit.TRAIN,
        encounter_id=source.encounter_id,
        hero_policy_id=source.hero_policy_id,
        seed=source.simulation_seed,
    )[source.action_index]
    assert source_again.state_digest == source.state_digest
    assert source_again.rng_digest == source.rng_digest


def test_illegal_forced_candidate_is_rejected(definitions, source):
    bad = CandidateKey("not_a_skill", "not_a_target", 999, 999)
    with pytest.raises(ValueError, match="belong"):
        run_counterfactual_branch(definitions, source, bad)


def test_cp3_observation_excludes_outcomes_ids_and_prescriptive_features(source):
    forbidden_tokens = (
        "enemy_win",
        "reward",
        "outcome",
        "seed",
        "actor_id",
        "target_id",
        "encounter_id",
        "hero_policy_id",
    )
    assert all(not any(token in name for token in forbidden_tokens) for name in CP3_FEATURE_NAMES)
    prescriptive = {
        item.name for item in CP2_FEATURE_AUDIT if item.classification == "prescriptive_excluded"
    }
    assert prescriptive.isdisjoint(CP3_FEATURE_NAMES)
    assert "role_flavor" in prescriptive
    assert len(source.candidate_observations[0]) == len(CP3_FEATURE_SCHEMA.names)


def test_outcome_preference_win_always_dominates_loss():
    loss = _outcome(enemy_win=False, final_hero_hp=0, hero_deaths=4)
    win = _outcome(enemy_win=True, final_hero_hp=999)
    assert compare_outcomes(win, loss) is OutcomePreference.FIRST
    assert compare_outcomes(loss, win) is OutcomePreference.SECOND


def test_loss_pareto_preference_and_ambiguity():
    better = _outcome(final_hero_hp=8, hero_deaths=1, mortal_wounds=2)
    worse = _outcome(final_hero_hp=12, hero_deaths=0, mortal_wounds=1)
    assert compare_outcomes(better, worse) is OutcomePreference.FIRST
    death_but_hp = _outcome(final_hero_hp=30, hero_deaths=1)
    no_death_low_hp = _outcome(final_hero_hp=1, hero_deaths=0)
    assert compare_outcomes(death_but_hp, no_death_low_hp) is OutcomePreference.INCOMPARABLE


def test_ties_remain_ties_including_two_wins():
    loss = _outcome(final_hero_hp=10)
    assert compare_outcomes(loss, loss) is OutcomePreference.TIE
    assert (
        compare_outcomes(_outcome(enemy_win=True), _outcome(enemy_win=True))
        is OutcomePreference.TIE
    )


def test_dataset_keeps_whole_source_in_one_split_and_round_trips(definitions, tmp_path: Path):
    dataset = build_counterfactual_dataset(
        definitions,
        split=DatasetSplit.TRAIN,
        encounter_ids=("shallow_cave",),
        hero_policy_ids=("naive",),
        seeds=(1,),
        sampling_spec=SamplingSpec(states_per_encounter_policy=1),
    )
    assert dataset.records
    assert {record.split for record in dataset.records} == {DatasetSplit.TRAIN}
    path = tmp_path / "counterfactual.json"
    save_counterfactual_dataset(dataset, path)
    assert load_counterfactual_dataset(path) == dataset
    audit = audit_counterfactual_dataset(dataset)
    assert audit.candidate_branches == sum(len(record.candidates) for record in dataset.records)


def test_ambiguous_pairs_are_not_training_pairs(definitions):
    dataset = build_counterfactual_dataset(
        definitions,
        split=DatasetSplit.TRAIN,
        encounter_ids=("shallow_cave",),
        hero_policy_ids=("naive",),
        seeds=(1,),
        sampling_spec=SamplingSpec(states_per_encounter_policy=2),
    )
    pairs = decisive_preference_pairs(dataset)
    for pair in pairs:
        record = dataset.records[pair.record_index]
        preference = compare_outcomes(
            record.candidates[pair.preferred_index].outcome,
            record.candidates[pair.other_index].outcome,
        )
        assert preference is OutcomePreference.FIRST


def test_train_only_normalization_rejects_validation(definitions):
    validation = build_counterfactual_dataset(
        definitions,
        split=DatasetSplit.VALIDATION,
        encounter_ids=("shallow_cave",),
        hero_policy_ids=("naive",),
        seeds=(10001,),
        sampling_spec=SamplingSpec(states_per_encounter_policy=1),
    )
    with pytest.raises(ValueError, match="TRAIN"):
        fit_outcome_normalization(validation)


def test_linear_and_mlp_score_variable_candidate_sets_and_reload_deterministically(
    definitions,
    tmp_path: Path,
):
    train = build_counterfactual_dataset(
        definitions,
        split=DatasetSplit.TRAIN,
        encounter_ids=("shallow_cave", "road_bandits"),
        hero_policy_ids=("naive",),
        seeds=(1, 2, 3),
        sampling_spec=SamplingSpec(states_per_encounter_policy=2),
    )
    validation = build_counterfactual_dataset(
        definitions,
        split=DatasetSplit.VALIDATION,
        encounter_ids=("shallow_cave", "road_bandits"),
        hero_policy_ids=("naive",),
        seeds=(10001, 10002),
        sampling_spec=SamplingSpec(states_per_encounter_policy=1),
    )
    if not decisive_preference_pairs(train) or not decisive_preference_pairs(validation):
        pytest.skip("Tiny deterministic fixture produced no decisive pair.")
    normalization = fit_outcome_normalization(train)
    normalization_path = tmp_path / "normalization.json"
    checksum = save_outcome_normalization(normalization, normalization_path)
    config = PairwiseTrainingConfig(max_epochs=3, early_stopping_patience=2, batch_size=32)
    for spec in (linear_scorer_spec(CP3_FEATURE_NAMES), mlp_scorer_spec(CP3_FEATURE_NAMES)):
        trained = train_pairwise_ranker(
            spec,
            train,
            validation,
            normalization,
            training_seed=17,
            config=config,
        )
        path = tmp_path / f"{spec.model_type}.json"
        save_outcome_ranker(trained, path, normalization_checksum=checksum)
        loaded = load_outcome_ranker(path)
        first = [
            choose_candidate_index(loaded.model, index, validation, normalization)
            for index in range(len(validation.records))
        ]
        second = [
            choose_candidate_index(loaded.model, index, validation, normalization)
            for index in range(len(validation.records))
        ]
        assert first == second
        assert all(
            0 <= choice < len(validation.records[index].candidates)
            for index, choice in enumerate(first)
        )


def _outcome(
    *,
    enemy_win: bool = False,
    final_hero_hp: int = 20,
    final_hero_effort: int = 10,
    terminal_hero_downs: int = 0,
    hero_deaths: int = 0,
    mortal_wounds: int = 0,
    enemy_caused_effort_drain: int = 0,
    rounds_elapsed: int = 3,
) -> CounterfactualOutcome:
    return CounterfactualOutcome(
        enemy_win=enemy_win,
        final_hero_hp=final_hero_hp,
        final_hero_effort=final_hero_effort,
        terminal_hero_downs=terminal_hero_downs,
        hero_deaths=hero_deaths,
        mortal_wounds=mortal_wounds,
        enemy_caused_effort_drain=enemy_caused_effort_drain,
        rounds_elapsed=rounds_elapsed,
    )
