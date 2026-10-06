from __future__ import annotations

import torch

from game.data.loaders import load_game_definitions
from game.dev.ml_cp6 import PRIMARY_ROUTES
from game.dev.ml_cp7 import SAFETY_BUDGET
from game.dev.ml_cp10 import (
    DEV_SEEDS,
    RESERVED_CONFIRMATION,
    TEST_SEEDS,
    VALIDATION_SEEDS,
    _dataset_from_payload,
    _load_cp9_datasets,
    _load_models,
    _mixed_branch,
    _record_vectors,
)
from game.ml.conservative_route_policy import (
    EDGE_OF_SUPPORT,
    POORLY_SUPPORTED,
    WELL_SUPPORTED,
    ConservativeRoutePolicy,
    SupportProfile,
    SupportThresholds,
    fit_support_profile,
)
from game.ml.route_policy_learning import PRIMARY_FEATURE_NAMES, collect_route_source_decisions


def test_cp10_seed_ranges_are_disjoint_and_confirmation_stays_reserved() -> None:
    cp10 = set(DEV_SEEDS) | set(VALIDATION_SEEDS) | set(TEST_SEEDS)
    reserved = set(range(RESERVED_CONFIRMATION[0], RESERVED_CONFIRMATION[1] + 1))
    assert cp10.isdisjoint(reserved)
    assert set(DEV_SEEDS).isdisjoint(VALIDATION_SEEDS)
    assert set(DEV_SEEDS).isdisjoint(TEST_SEEDS)
    assert set(VALIDATION_SEEDS).isdisjoint(TEST_SEEDS)


def test_support_profile_classifies_well_edge_and_poor_transparently() -> None:
    size = len(PRIMARY_FEATURE_NAMES)
    profile = SupportProfile(
        feature_names=PRIMARY_FEATURE_NAMES,
        train_vectors=(tuple(0.0 for _ in range(size)),),
        normalized_min=tuple(-1.0 for _ in range(size)),
        normalized_max=tuple(1.0 for _ in range(size)),
        thresholds=SupportThresholds(well_distance_max=0.25, edge_distance_max=0.50),
    )
    well = profile.classify_normalized(torch.zeros(size))
    edge_vector = torch.zeros(size)
    edge_vector[0] = 2.0
    edge = profile.classify_normalized(edge_vector)
    poor = profile.classify_normalized(torch.full((size,), 2.0))
    assert well.category == WELL_SUPPORTED
    assert edge.category == EDGE_OF_SUPPORT
    assert edge.feature_range_exceedances == 1
    assert poor.category == POORLY_SUPPORTED
    assert poor.feature_range_exceedances == size


def test_support_profile_fit_is_deterministic_on_cp9_train() -> None:
    train, _validation, _test = _load_cp9_datasets()
    _policies, _pi3, normalization, _selected = _load_models()
    vectors = _record_vectors(train)
    first = fit_support_profile(vectors, normalization)
    second = fit_support_profile(vectors, normalization)
    assert first == second
    assert first.thresholds.well_distance_max < first.thresholds.edge_distance_max


def test_cp9_dataset_payload_roundtrip_preserves_records() -> None:
    raw = __import__("json").loads(
        open(
            "artifacts/ml/checkpoint9-route-aware-policy-learning-v1/route_counterfactual_branches.json",
            encoding="utf-8",
        ).read()
    )
    train = _dataset_from_payload(raw["train"])
    assert train.split == "train"
    assert len(train.records) == 120
    assert all(record.control_exact for record in train.records)


def test_pure_pi2_gate_executes_anchor_on_real_consumed_state() -> None:
    definitions = load_game_definitions()
    policies, pi3, normalization, _selected = _load_models()
    train, _validation, _test = _load_cp9_datasets()
    profile = fit_support_profile(_record_vectors(train), normalization)
    route = PRIMARY_ROUTES[0]
    sources = collect_route_source_decisions(
        definitions,
        route,
        root_seed=62001,
        source_policy=policies["pi2_cp4"],
        source_policy_name="pi2_cp4",
        hero_policy_id="company_survival",
        split="consumed_test",
        safety_budget=SAFETY_BUDGET,
    )
    assert sources
    from game.ml.route_policy_learning import run_route_trajectory

    gate = ConservativeRoutePolicy(
        anchor_policy=policies["pi2_cp4"],
        route_policy=pi3,
        support_profile=profile,
        gate_mode="PURE_PI2",
        margin_threshold=0.0,
        policy_id="test_pure_pi2",
    )
    outcome, _digest, _encounters, _local = run_route_trajectory(
        definitions,
        route,
        root_seed=62001,
        enemy_policy_name="test_pure_pi2",
        enemy_policy=gate,
        hero_policy_id="company_survival",
        safety_budget=SAFETY_BUDGET,
    )
    assert gate.audit
    assert all(not item["gate_accepted"] for item in gate.audit)
    assert outcome.valid_terminal
    assert all(item["executed_action"] == item["anchor_action"] for item in gate.audit)


def test_mixed_continuation_replays_consumed_pi3_prefix_exactly() -> None:
    definitions = load_game_definitions()
    policies, pi3, _normalization, _selected = _load_models()
    route = PRIMARY_ROUTES[0]
    sources = collect_route_source_decisions(
        definitions,
        route,
        root_seed=62001,
        source_policy=pi3,
        source_policy_name="pi3_route",
        hero_policy_id="company_survival",
        split="consumed_test",
        safety_budget=SAFETY_BUDGET,
    )
    assert sources
    source = sources[min(1, len(sources) - 1)]
    candidate = source.capture.source_choice
    branch = _mixed_branch(
        definitions,
        source,
        candidate,
        prefix_policy=pi3,
        continuation_policy=policies["pi2_cp4"],
        continuation_name="pi2_cp4",
    )
    assert branch["replay_state_digest"] == source.capture.state_digest
    assert branch["replay_rng_digest"] == source.capture.rng_digest
    assert tuple(branch["replay_candidates"]) == tuple(
        {
            "skill_id": item.skill_id,
            "target_id": item.target_id,
            "skill_order": item.skill_order,
            "target_order": item.target_order,
        }
        for item in source.capture.candidates
    )
