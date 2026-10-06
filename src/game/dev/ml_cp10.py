# ruff: noqa: E501
"""Checkpoint 10: conservative route-policy improvement / distribution-shift repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from game.data.loaders import load_game_definitions
from game.dev.ml_cp5 import _load_frozen_policies
from game.dev.ml_cp6 import PRIMARY_ROUTES
from game.dev.ml_cp7 import SAFETY_BUDGET
from game.dev.ml_cp9 import (
    ARTIFACT_ROOT as CP9_ROOT,
)
from game.dev.ml_cp9 import (
    PRIMARY_HERO_POLICY,
    _dataset_payload,
    _load_selected_pi3,
    _source_provenance,
)
from game.ml.conservative_route_policy import (
    EDGE_OF_SUPPORT,
    POORLY_SUPPORTED,
    WELL_SUPPORTED,
    BranchContinuationPolicy,
    ConservativeRoutePolicy,
    DiagnosticRoutePolicy,
    LimitedDeviationPolicy,
    SupportProfile,
    candidate_family,
    fit_support_profile,
    support_profile_payload,
)
from game.ml.counterfactual import CandidateKey, OutcomePreference
from game.ml.models import linear_scorer_spec, mlp_scorer_spec
from game.ml.route_policy_learning import (
    PRIMARY_FEATURE_NAMES,
    FrozenRouteRankerPolicy,
    LocalOutcome,
    RouteCandidateRecord,
    RouteDataset,
    RouteDecisionRecord,
    RouteNormalization,
    RouteOutcome,
    RouteSourceDecision,
    RouteTrainingConfig,
    collect_route_source_decisions,
    compare_route_outcomes,
    decisive_pairs,
    evaluate_route_ranker,
    fit_route_normalization,
    load_normalization,
    load_route_ranker,
    paired_binary_summary,
    run_route_trajectory,
    save_normalization,
    save_route_ranker,
    sha256_file,
    train_route_ranker,
)
from game.ml.route_termination import run_corrected_stateful_route

EXPERIMENT_ID = "checkpoint10-conservative-route-policy-repair-v1"
CP9_COMMIT = "88ddf45872c22044ac9661be165d24da8e017623"
ARTIFACT_ROOT = Path("artifacts/ml") / EXPERIMENT_ID
PROTOCOL_PATH = ARTIFACT_ROOT / "frozen_protocol.json"
PROVENANCE_PATH = ARTIFACT_ROOT / "provenance.json"
SEED_AUTHORITY_PATH = ARTIFACT_ROOT / "seed_authority.json"
SUPPORT_PATH = ARTIFACT_ROOT / "state_support_audit.json"
REPRO_PATH = ARTIFACT_ROOT / "cp9_reproduction.json"
TOP1_PATH = ARTIFACT_ROOT / "top1_vs_pairwise.json"
FIRST_DIV_PATH = ARTIFACT_ROOT / "first_divergence_audit.json"
COMPOUND_PATH = ARTIFACT_ROOT / "compounding_curve.json"
CONT_SAMPLE_PATH = ARTIFACT_ROOT / "continuation_sample.json"
CONT_PATH = ARTIFACT_ROOT / "continuation_mismatch.json"
MULTI_PATH = ARTIFACT_ROOT / "multi_deviation_audit.json"
MARK_PATH = ARTIFACT_ROOT / "mark_score_bias.json"
TARGET_PATH = ARTIFACT_ROOT / "targeting_bias.json"
PRESSURE_PATH = ARTIFACT_ROOT / "direct_pressure_audit.json"
GRAB_PATH = ARTIFACT_ROOT / "grab_audit.json"
DRAIN_PATH = ARTIFACT_ROOT / "effort_drain_audit.json"
SUPPORT_ERROR_PATH = ARTIFACT_ROOT / "support_error_relationship.json"
DIAGNOSIS_FREEZE_PATH = ARTIFACT_ROOT / "diagnosis_frozen.json"
REPAIR_CANDIDATES_PATH = ARTIFACT_ROOT / "repair_candidates.json"
REPAIRED_SUMMARY_PATH = ARTIFACT_ROOT / "repaired_dataset_summary.json"
CONT_LABEL_PATH = ARTIFACT_ROOT / "continuation_label_analysis.json"
NORMALIZATION_PATH = ARTIFACT_ROOT / "normalization.json"
VALIDATION_PATH = ARTIFACT_ROOT / "validation_results.json"
SELECTED_PATH = ARTIFACT_ROOT / "selected_pi4_conservative.json"
DEVIATION_PATH = ARTIFACT_ROOT / "deviation_budget.json"
TEST_ACTION_PATH = ARTIFACT_ROOT / "test_action_ranking.json"
RECURSIVE_PATH = ARTIFACT_ROOT / "recursive_route_results.json"
PAIRED_PATH = ARTIFACT_ROOT / "paired_route_comparison.json"
SHIFT_PATH = ARTIFACT_ROOT / "recursive_distribution_shift.json"
MARK_REPAIR_PATH = ARTIFACT_ROOT / "mark_repair.json"
DIRECT_RECOVERY_PATH = ARTIFACT_ROOT / "direct_pressure_recovery.json"
GRAB_DESIGN_PATH = ARTIFACT_ROOT / "grab_design_hypothesis.json"
DRAIN_DEGEN_PATH = ARTIFACT_ROOT / "effort_drain_degeneracy.json"
FAILURE_PATH = ARTIFACT_ROOT / "cp9_failure_classification.json"
PAIRWISE_LESSON_PATH = ARTIFACT_ROOT / "pairwise_metric_lesson.json"
POLICY_ITER_PATH = ARTIFACT_ROOT / "policy_iteration_assessment.json"
CONFIRM_PATH = ARTIFACT_ROOT / "confirmation_decision.json"
MAZE_PATH = ARTIFACT_ROOT / "maze_implications.json"
SUMMARY_MD_PATH = ARTIFACT_ROOT / "research_summary.md"
NEXT_PATH = ARTIFACT_ROOT / "next_research_recommendation.json"
MANIFEST_PATH = ARTIFACT_ROOT / "manifest.json"
REPORT_PATH = ARTIFACT_ROOT / "REPORT.md"

DEV_SEEDS = tuple(range(70001, 70033))
VALIDATION_SEEDS = tuple(range(71001, 71013))
TEST_SEEDS = tuple(range(72001, 72017))
RESERVED_CONFIRMATION = (63001, 63064)
PI3_DEV_SAMPLE = 72
VALIDATION_PI2_SAMPLE = 36
VALIDATION_PI3_SAMPLE = 36
TEST_PI2_SAMPLE = 72
TEST_PI3_SAMPLE = 36
CONTINUATION_SAMPLE_COUNT = 24
MULTI_DEVIATION_CASE_COUNT = 12
LINEAR_SEED = 20261013
MLP_SEED = 20261013
TRAINING_CONFIG = RouteTrainingConfig()

IMPLEMENTATION_PATHS = (
    "src/game/dev/ml_cp10.py",
    "src/game/ml/conservative_route_policy.py",
    "tests/test_ml_conservative_route_policy.py",
    "src/game/dev/ml_cp9.py",
    "src/game/ml/route_policy_learning.py",
    "src/game/ml/route_termination.py",
    "src/game/ml/stateful_route.py",
    "src/game/ml/outcome_features.py",
    "src/game/ml/outcome_ranking.py",
)
SIMULATOR_PATHS = (
    "src/game/combat/actions.py",
    "src/game/combat/combat_state.py",
    "src/game/combat/enemy_decision.py",
    "src/game/combat/enemy_learning.py",
    "src/game/core/rng.py",
)
CONTENT_PATHS = tuple(str(path).replace("/", "\\") for path in sorted(Path("data").glob("*.yaml")))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_dirty() -> bool:
    return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())


def _hash_paths(paths: Sequence[str]) -> dict[str, str]:
    return {path: sha256_file(Path(path)) for path in paths}


def _sha_payload(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _key(payload: Mapping[str, Any]) -> CandidateKey:
    return CandidateKey(**payload)


def _route_outcome(payload: Mapping[str, Any]) -> RouteOutcome:
    return RouteOutcome(**payload)


def _local_outcome(payload: Mapping[str, Any] | None) -> LocalOutcome | None:
    return None if payload is None else LocalOutcome(**payload)


def _dataset_from_payload(payload: Mapping[str, Any]) -> RouteDataset:
    records: list[RouteDecisionRecord] = []
    for raw in payload["records"]:
        candidates = tuple(
            RouteCandidateRecord(
                key=_key(item["key"]),
                primary_observation=tuple(float(x) for x in item["primary_observation"]),
                oracle_observation=tuple(float(x) for x in item["oracle_observation"]),
                route_outcome=_route_outcome(item["route_outcome"]),
                local_outcome=_local_outcome(item.get("local_outcome")),
            )
            for item in raw["candidates"]
        )
        records.append(
            RouteDecisionRecord(
                split=str(raw["split"]),
                route_id=str(raw["route_id"]),
                encounter_index=int(raw["encounter_index"]),
                encounter_id=str(raw["encounter_id"]),
                route_position=str(raw["route_position"]),
                hero_policy_id=str(raw["hero_policy_id"]),
                root_seed=int(raw["root_seed"]),
                enemy_class_id=str(raw["enemy_class_id"]),
                round_number=int(raw["round_number"]),
                state_digest=str(raw["state_digest"]),
                rng_digest=str(raw["rng_digest"]),
                candidates=candidates,
                source_policy_index=int(raw["source_policy_index"]),
                control_exact=bool(raw["control_exact"]),
            )
        )
    return RouteDataset(split=str(payload["split"]), records=tuple(records))


def _load_cp9_datasets() -> tuple[RouteDataset, RouteDataset, dict[str, RouteDataset]]:
    branch = _read_json(CP9_ROOT / "route_counterfactual_branches.json")
    test = _read_json(CP9_ROOT / "test_route_counterfactual_branches.json")
    train = _dataset_from_payload(branch["train"])
    validation = _dataset_from_payload(branch["validation"])
    test_sets = {name: _dataset_from_payload(value) for name, value in test.items()}
    return train, validation, test_sets


def _load_models() -> tuple[dict[str, Any], FrozenRouteRankerPolicy, RouteNormalization, dict[str, Any]]:
    policies, _ = _load_frozen_policies()
    pi3, normalization, selected = _load_selected_pi3()
    return policies, pi3, normalization, selected


def _record_vectors(dataset: RouteDataset) -> list[list[tuple[float, ...]]]:
    return [[candidate.primary_observation for candidate in record.candidates] for record in dataset.records]


def _stable_hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _seed_authority() -> dict[str, Any]:
    return {
        "prior_ranges": {
            "cp3_train": [1, 256],
            "cp3_validation": [10001, 10064],
            "cp3_test": [20001, 20128],
            "cp5_confirmation": [30001, 30128],
            "cp6_cp8_route_blocks": [[40001, 40064], [50001, 50064]],
            "cp9_train": [60001, 60032],
            "cp9_validation": [61001, 61012],
            "cp9_test": [62001, 62016],
            "reserved_confirmation": list(RESERVED_CONFIRMATION),
        },
        "cp10": {
            "development": [DEV_SEEDS[0], DEV_SEEDS[-1]],
            "validation": [VALIDATION_SEEDS[0], VALIDATION_SEEDS[-1]],
            "test": [TEST_SEEDS[0], TEST_SEEDS[-1]],
        },
        "reserved_confirmation_accessed": False,
    }


def _protocol_payload(profile: SupportProfile) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "purpose": "diagnose CP9 recursive failure and test exactly one conservative repair",
        "baseline_commit": CP9_COMMIT,
        "anchor_policy": "pi2_cp4",
        "failed_policy": "pi3_route",
        "production_changes_allowed": False,
        "reserved_confirmation": list(RESERVED_CONFIRMATION),
        "reserved_confirmation_accessed": False,
        "seed_authority": _seed_authority(),
        "routes": [route.route_id for route in PRIMARY_ROUTES],
        "primary_hero_policy": PRIMARY_HERO_POLICY,
        "support": support_profile_payload(profile),
        "support_categories": [WELL_SUPPORTED, EDGE_OF_SUPPORT, POORLY_SUPPORTED],
        "top1": {
            "preferred_set": "candidate is not strictly route-dominated by any legal candidate",
            "regret": "predicted top-1 is strictly route-dominated by at least one legal candidate",
        },
        "continuation_sample": {
            "count": CONTINUATION_SAMPLE_COUNT,
            "selection": "stable SHA-256 over pre-outcome disagreement provenance with quota priorities",
        },
        "multi_deviation_sample": {
            "count": MULTI_DEVIATION_CASE_COUNT,
            "selection": "stable SHA-256 from known pi2-win/pi3-loss CP9 cases",
            "limits": [1, 2, 3],
        },
        "repaired_data": {
            "cp9_pi2_source_records": 120,
            "cp10_pi3_induced_train_records": PI3_DEV_SAMPLE,
            "pi3_induced_sampling": "stable SHA-256 per-route strata before branch outcomes",
            "branch_continuation": "pi2_cp4",
            "pi3_prefix_replay": True,
            "deployment_aware_label_sample_count": 18,
            "deployment_aware_sampling": "stable SHA-256 before pi3-continuation outcomes",
        },
        "validation_sampling": {
            "pi2_source": VALIDATION_PI2_SAMPLE,
            "pi3_induced": VALIDATION_PI3_SAMPLE,
        },
        "test_sampling": {"pi2_source": TEST_PI2_SAMPLE, "pi3_induced": TEST_PI3_SAMPLE},
        "model_family": ["linear", "mlp_64_32"],
        "model_seeds": {"linear": LINEAR_SEED, "mlp": MLP_SEED},
        "candidate_family": [
            "PURE_PI2",
            "PURE_ROUTE_RANKER",
            "SUPPORT_GATED",
            "MARGIN_GATED",
            "SUPPORT_AND_MARGIN_GATED",
        ],
        "final_repair_eligibility": {
            "eligible": ["SUPPORT_GATED", "MARGIN_GATED", "SUPPORT_AND_MARGIN_GATED"],
            "pure_route_ranker_role": "diagnostic_control_only",
            "pure_pi2_fallback": "allowed only if every eligible conservative gate is >=10pp worse than pi2 on validation",
        },
        "margin_rule": (
            "TRAIN/VALIDATION only: q25 of positive route-vs-anchor margins among validation "
            "deviations where proposal is in the true preferred set and anchor is not; "
            "if none exist, q75 of positive disagreement margins"
        ),
        "ranker_selection": "validation pairwise accuracy desc, loss asc, linear before mlp on exact tie",
        "gate_selection": [
            "validation top1 preferred-set rate desc",
            "pi3-induced validation top1 preferred-set rate desc",
            "beneficial accepted-deviation precision desc",
            "recursive validation enemy-win rate desc",
            "deviation rate asc",
            "gate mode stable lexical",
        ],
        "test_selection_used": False,
        "test_accessed": False,
        "post_test_tuning_performed": False,
        "success_categories": [
            "CLEAR_IMPROVEMENT",
            "PRESERVED_BASELINE_WITH_USEFUL_ROUTE_DEVIATIONS",
            "APPROXIMATELY_EQUIVALENT",
            "CLEAR_REGRESSION",
            "INCONCLUSIVE",
        ],
        "success_classification_rules": {
            "preserved_baseline_floor": -0.05,
            "approximate_equivalence_ci_margin": 0.10,
            "clear_effect": "paired CI excludes zero and exact McNemar p < 0.05",
        },
        "behavior_classification_rules": {
            "mark_collapse_repaired": "pi4 Mark use <= midpoint between pi2 and pi3 use",
            "effort_drain_strategy_collapse_threshold": 0.95,
            "direct_pressure_recovered": "pi4 Bite use >= midpoint between pi3 and pi2 use",
            "grab_discovery": "pi4 Grab use >= 1.2x pi2 Grab use while pi4 recursive result is not a clear regression",
        },
        "diagnosis_verdict_rules": {
            "pairwise_misleading_yes": "top1 preferred-set rate trails pairwise accuracy by >=0.10",
            "pairwise_misleading_partial": "gap is >=0.05 but <0.10",
            "support_concentration_yes": ">=0.60 of pi3 disagreements are EDGE/POOR support",
            "support_concentration_no": "<=0.40 of pi3 disagreements are EDGE/POOR support",
            "continuation_material_yes": ">=0.20 reversal rate among ordered continuation pairs",
            "continuation_material_mixed": "reversal rate >=0.05 and <0.20",
            "family_major_contributor": "family is >=0.25 of first divergences plus corroborating continuation/outcome evidence",
            "family_partial_contributor": "family is >=0.10 of first divergences or has corroborating continuation/outcome evidence",
        },
        "implementation_hashes": _hash_paths(IMPLEMENTATION_PATHS),
        "simulator_hashes": _hash_paths(SIMULATOR_PATHS),
        "content_hashes": _hash_paths(CONTENT_PATHS),
    }


def _verify_cp9_integrity() -> dict[str, Any]:
    selected = _read_json(CP9_ROOT / "selected_pi3_route.json")
    manifest = _read_json(CP9_ROOT / "manifest.json")
    bad = []
    for name, expected in manifest["artifacts"].items():
        path = CP9_ROOT / name
        if not path.exists() or sha256_file(path) != expected:
            bad.append(name)
    if bad:
        raise ValueError(f"CP9 artifact mismatch: {bad}")
    if selected["pi3_route_model_checksum"] != "f01853dc9de45b215ab4309c826cb40956266d61aaf3fae2021b726d532392b7":
        raise ValueError("CP9 pi3 checksum mismatch")
    if selected.get("post_test_tuning_performed"):
        raise ValueError("CP9 indicates post-test tuning")
    return {"artifact_count": len(manifest["artifacts"]), "artifact_mismatches": bad, "selected": selected}


def freeze_protocol() -> None:
    if ARTIFACT_ROOT.exists():
        raise FileExistsError("CP10 artifact root already exists; refusing to overwrite frozen research state.")
    if _git_head() != CP9_COMMIT:
        raise ValueError(f"CP10 must begin from committed CP9 baseline {CP9_COMMIT}")
    cp9 = _verify_cp9_integrity()
    train, _validation, _test = _load_cp9_datasets()
    _policies, _pi3, normalization, selected = _load_models()
    profile = fit_support_profile(_record_vectors(train), normalization)
    ARTIFACT_ROOT.mkdir(parents=True)
    _write_json(SEED_AUTHORITY_PATH, _seed_authority())
    _write_json(SUPPORT_PATH, {"phase": "frozen_before_cp10_outcomes", **support_profile_payload(profile)})
    protocol = _protocol_payload(profile)
    _write_json(PROTOCOL_PATH, protocol)
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "git_head": _git_head(),
        "repository_dirty": _git_dirty(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cp9_integrity": cp9,
        "cp9_protocol_checksum": sha256_file(CP9_ROOT / "frozen_protocol.json"),
        "cp9_selected_checksum": sha256_file(CP9_ROOT / "selected_pi3_route.json"),
        "cp9_pi3_model_checksum": selected["pi3_route_model_checksum"],
        "protocol_checksum": sha256_file(PROTOCOL_PATH),
        "implementation_hashes": protocol["implementation_hashes"],
        "simulator_hashes": protocol["simulator_hashes"],
        "content_hashes": protocol["content_hashes"],
        "new_cp10_outcomes_accessed": False,
        "test_accessed": False,
        "reserved_confirmation_accessed": False,
    }
    _write_json(PROVENANCE_PATH, provenance)
    print(f"CP10_PROTOCOL_SHA={provenance['protocol_checksum']}")
    print(f"CP10_SUPPORT_Q75={profile.thresholds.well_distance_max}")
    print(f"CP10_SUPPORT_Q95={profile.thresholds.edge_distance_max}")
    print("CP10_NEW_OUTCOMES_ACCESSED=false")


def _preferred_set(record: RouteDecisionRecord) -> set[int]:
    preferred = set(range(len(record.candidates)))
    for i, first in enumerate(record.candidates):
        for j, second in enumerate(record.candidates):
            if i == j:
                continue
            pref = compare_route_outcomes(first.route_outcome, second.route_outcome)
            if pref == OutcomePreference.SECOND:
                preferred.discard(i)
                break
    return preferred


def _score_indices(model: Any, normalization: RouteNormalization, record: RouteDecisionRecord) -> tuple[list[float], list[int]]:
    raw = torch.tensor([item.primary_observation for item in record.candidates], dtype=torch.float32)
    values = normalization.normalize(raw)
    model.eval()
    with torch.no_grad():
        scores = model(values).reshape(-1)
    order = torch.argsort(scores, descending=True).tolist()
    return [float(x) for x in scores.tolist()], [int(x) for x in order]


def _top1_rows(dataset: RouteDataset, pi3: FrozenRouteRankerPolicy) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in dataset.records:
        scores, order = _score_indices(pi3.model, pi3.normalization, record)
        top = order[0]
        preferred = _preferred_set(record)
        source = record.source_policy_index
        families = [set(candidate_family(item.primary_observation)) for item in record.candidates]
        rows.append(
            {
                "route_id": record.route_id,
                "encounter_id": record.encounter_id,
                "route_position": record.route_position,
                "root_seed": record.root_seed,
                "state_digest": record.state_digest,
                "legal_actions": len(record.candidates),
                "pi3_top_index": top,
                "pi2_index": source,
                "agreement": top == source,
                "preferred_set": sorted(preferred),
                "top1_in_preferred_set": top in preferred,
                "top1_regret": top not in preferred,
                "pi2_in_preferred_set": source in preferred,
                "pi2_rank": order.index(source) + 1,
                "best_observed_ranks": sorted(order.index(index) + 1 for index in preferred),
                "top_vs_runner_margin": scores[top] - scores[order[1]] if len(order) > 1 else 0.0,
                "top_vs_pi2_margin": scores[top] - scores[source],
                "candidate_families_present": sorted(set().union(*families)),
                "pi3_top_families": sorted(families[top]),
                "pi2_families": sorted(families[source]),
            }
        )
    return rows


def _aggregate_top1(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "states": count,
        "top1_preferred_set_count": sum(bool(row["top1_in_preferred_set"]) for row in rows),
        "top1_preferred_set_rate": sum(bool(row["top1_in_preferred_set"]) for row in rows) / count if count else 0.0,
        "top1_regret_count": sum(bool(row["top1_regret"]) for row in rows),
        "pi2_preferred_set_rate": sum(bool(row["pi2_in_preferred_set"]) for row in rows) / count if count else 0.0,
        "agreement_rate": sum(bool(row["agreement"]) for row in rows) / count if count else 0.0,
        "mean_pi2_rank": mean(float(row["pi2_rank"]) for row in rows) if rows else 0.0,
        "mean_top_margin": mean(float(row["top_vs_runner_margin"]) for row in rows) if rows else 0.0,
    }


def top1_audit() -> dict[str, Any]:
    _train, _validation, test_sets = _load_cp9_datasets()
    _policies, pi3, _normalization, _selected = _load_models()
    dataset = test_sets[PRIMARY_HERO_POLICY]
    rows = _top1_rows(dataset, pi3)
    groups: dict[str, list[dict[str, Any]]] = {
        "all": rows,
        "agree_with_pi2": [row for row in rows if row["agreement"]],
        "disagree_with_pi2": [row for row in rows if not row["agreement"]],
    }
    for family in ("MARK", "GRAB", "GUARD", "EFFORT_DRAIN", "DIRECT_PRESSURE"):
        groups[family.lower()] = [row for row in rows if family in row["candidate_families_present"]]
    cp9_ranking = _read_json(CP9_ROOT / "test_ranking_results.json")
    payload = {
        "pairwise_accuracy": cp9_ranking["cp9_route_mlp_seed_20261010"][PRIMARY_HERO_POLICY]["accuracy"],
        "groups": {name: _aggregate_top1(items) for name, items in groups.items()},
        "rows": rows,
        "lesson": "pairwise accuracy and top-action control quality are separate metrics",
    }
    _write_json(TOP1_PATH, payload)
    return payload


def _route_map() -> dict[str, Any]:
    return {route.route_id: route for route in PRIMARY_ROUTES}


def _case_id(route_id: str, seed: int) -> str:
    return f"{route_id}|{PRIMARY_HERO_POLICY}|{seed}"


def _mixed_branch(
    definitions: Any,
    source: RouteSourceDecision,
    candidate: CandidateKey,
    *,
    prefix_policy: Any,
    continuation_policy: Any,
    continuation_name: str,
) -> dict[str, Any]:
    forcing = BranchContinuationPolicy(
        prefix_policy=prefix_policy,
        continuation_policy=continuation_policy,
        source=source,
        candidate_override=candidate,
    )
    outcome, digest, _encounters, local = run_route_trajectory(
        definitions,
        source.route,
        root_seed=source.root_seed,
        enemy_policy_name=continuation_name,
        enemy_policy=continuation_policy,
        hero_policy_id=source.hero_policy_id,
        safety_budget=SAFETY_BUDGET,
        forcing_policy=forcing,  # type: ignore[arg-type]
    )
    if not forcing.reached_target:
        raise RuntimeError("Mixed-continuation branch failed to reach target source state")
    return {
        "candidate": asdict(candidate),
        "route_outcome": asdict(outcome),
        "local_outcome": None if local is None else asdict(local),
        "route_digest": digest,
        "replay_state_digest": forcing.replay_state_digest,
        "replay_rng_digest": forcing.replay_rng_digest,
        "replay_candidates": [asdict(item) for item in forcing.replay_candidates],
    }


def _preference_from_payload(first: Mapping[str, Any], second: Mapping[str, Any]) -> str:
    return compare_route_outcomes(_route_outcome(first), _route_outcome(second)).value


def _known_cp9_rows() -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    raw = _read_json(CP9_ROOT / "recursive_route_results.json")
    pi2 = {
        str(row["case_id"]): row
        for row in raw
        if row["hero_policy_id"] == PRIMARY_HERO_POLICY and row["policy"] == "pi2_cp4"
    }
    pi3 = {
        str(row["case_id"]): row
        for row in raw
        if row["hero_policy_id"] == PRIMARY_HERO_POLICY and row["policy"] == "pi3_route"
    }
    return pi2, pi3


def _reproduce_cp9(definitions: Any, policies: Mapping[str, Any], pi3: FrozenRouteRankerPolicy) -> dict[str, Any]:
    pi2_known, pi3_known = _known_cp9_rows()
    selected_cases = sorted(pi2_known)[:2] + sorted(pi2_known)[-2:]
    checks: list[dict[str, Any]] = []
    for case_id in selected_cases:
        knowns = {"pi2_cp4": pi2_known[case_id], "pi3_route": pi3_known[case_id]}
        route = _route_map()[str(pi2_known[case_id]["route_id"])]
        seed = int(pi2_known[case_id]["root_seed"])
        for name, policy in (("pi2_cp4", policies["pi2_cp4"]), ("pi3_route", pi3)):
            result = run_corrected_stateful_route(
                definitions,
                route,
                root_seed=seed,
                enemy_policy_name=name,
                enemy_policy=policy,
                hero_policy_id=PRIMARY_HERO_POLICY,
                safety_budget=SAFETY_BUDGET,
            )
            known = knowns[name]
            encounter_equal = len(result["encounters"]) == len(known["encounters"])
            if encounter_equal:
                for actual, expected in zip(result["encounters"], known["encounters"], strict=True):
                    encounter_equal = encounter_equal and all(
                        actual[field] == expected[field]
                        for field in (
                            "termination_reason",
                            "final_state_digest",
                            "final_rng_digest",
                            "enemy_action_sequence",
                            "hero_action_sequence",
                            "enemy_event_digest",
                        )
                    )
            checks.append(
                {
                    "case_id": case_id,
                    "policy": name,
                    "terminal_equal": result["route_terminal_status"] == known["route_terminal_status"],
                    "final_company_equal": result["final_company_state"] == known["final_company_state"],
                    "encounter_trace_equal": encounter_equal,
                }
            )
    _train, _validation, tests = _load_cp9_datasets()
    sample = tests[PRIMARY_HERO_POLICY].records[:8]
    score_checks = []
    for record in sample:
        first = _score_indices(pi3.model, pi3.normalization, record)[0]
        second = _score_indices(pi3.model, pi3.normalization, record)[0]
        score_checks.append(first == second)
    payload = {
        "case_checks": checks,
        "all_route_reproduction_exact": all(
            item["terminal_equal"] and item["final_company_equal"] and item["encounter_trace_equal"]
            for item in checks
        ),
        "model_output_repeatability": all(score_checks),
        "cases": len(selected_cases),
    }
    _write_json(REPRO_PATH, payload)
    return payload


def _source_decision_rows(
    definitions: Any,
    policies: Mapping[str, Any],
    pi3: FrozenRouteRankerPolicy,
    profile: SupportProfile,
) -> tuple[dict[str, list[RouteSourceDecision]], dict[str, list[dict[str, Any]]]]:
    sources_by_case: dict[str, list[RouteSourceDecision]] = {}
    diagnostics_by_case: dict[str, list[dict[str, Any]]] = {}
    known_pi2, known_pi3 = _known_cp9_rows()
    for seed in range(62001, 62017):
        for route in PRIMARY_ROUTES:
            case = _case_id(route.route_id, seed)
            diagnostic = DiagnosticRoutePolicy(
                executed_name="pi3_route",
                executed_policy=pi3,
                anchor_policy=policies["pi2_cp4"],
                route_policy=pi3,
                support_profile=profile,
            )
            sources = collect_route_source_decisions(
                definitions,
                route,
                root_seed=seed,
                source_policy=diagnostic,
                source_policy_name="pi3_route",
                hero_policy_id=PRIMARY_HERO_POLICY,
                split="cp9_test_diagnosis",
                safety_budget=SAFETY_BUDGET,
            )
            if case not in known_pi3:
                raise RuntimeError(f"Missing known CP9 case {case}")
            if sources:
                baseline = sources[0].baseline_outcome
                if baseline.enemy_victory != bool(known_pi3[case]["route_failed_enemy_victory"]):
                    raise RuntimeError(f"CP9 pi3 reproduction mismatch for {case}")
            if len(sources) != len(diagnostic.decisions):
                raise RuntimeError("Diagnostic/source decision count mismatch")
            annotated: list[dict[str, Any]] = []
            for source, decision in zip(sources, diagnostic.decisions, strict=True):
                row = {**decision, **_source_provenance(source)}
                row["case_id"] = case
                row["pi2_case_win"] = bool(known_pi2[case]["route_failed_enemy_victory"])
                row["pi3_case_win"] = bool(known_pi3[case]["route_failed_enemy_victory"])
                annotated.append(row)
            sources_by_case[case] = list(sources)
            diagnostics_by_case[case] = annotated
        print(f"CP10_CP9_DIAGNOSTIC_SEED_DONE={seed}", flush=True)
    return sources_by_case, diagnostics_by_case


def _first_divergences(
    definitions: Any,
    policies: Mapping[str, Any],
    pi3: FrozenRouteRankerPolicy,
    sources_by_case: Mapping[str, Sequence[RouteSourceDecision]],
    diagnostics_by_case: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    pi2_known, pi3_known = _known_cp9_rows()
    failure_cases = sorted(
        case for case in pi2_known
        if bool(pi2_known[case]["route_failed_enemy_victory"])
        and not bool(pi3_known[case]["route_failed_enemy_victory"])
    )
    rows: list[dict[str, Any]] = []
    motifs: Counter[str] = Counter()
    for case in failure_cases:
        diagnostics = diagnostics_by_case[case]
        source_list = sources_by_case[case]
        index = next((i for i, row in enumerate(diagnostics) if not row["agreement"]), None)
        if index is None:
            rows.append({"case_id": case, "first_divergence": None})
            continue
        row = dict(diagnostics[index])
        source = source_list[index]
        pi2_key = _key(row["anchor_key"])
        pi3_key = _key(row["route_key"])
        pi2_branch = _mixed_branch(
            definitions, source, pi2_key,
            prefix_policy=pi3,
            continuation_policy=policies["pi2_cp4"],
            continuation_name="pi2_cp4",
        )
        pi3_branch = _mixed_branch(
            definitions, source, pi3_key,
            prefix_policy=pi3,
            continuation_policy=policies["pi2_cp4"],
            continuation_name="pi2_cp4",
        )
        preference = _preference_from_payload(pi2_branch["route_outcome"], pi3_branch["route_outcome"])
        route_families = set(row["route_choice_families"])
        anchor_families = set(row["anchor_choice_families"])
        if "MARK" in route_families and "MARK" not in anchor_families:
            motifs["MARK_OVER_SELECTION"] += 1
        if row["route_key"]["target_id"] != row["anchor_key"]["target_id"]:
            motifs["TARGET_CHANGE"] += 1
        if "GUARD" in route_families and "GUARD" not in anchor_families:
            motifs["GUARD"] += 1
        if "DIRECT_PRESSURE" in anchor_families and "DIRECT_PRESSURE" not in route_families:
            motifs["LOST_DIRECT_PRESSURE"] += 1
        if "GRAB" in anchor_families and "GRAB" not in route_families:
            motifs["FAILED_TO_GRAB"] += 1
        if row["route_target_is_lowest_hp"]:
            motifs["LOW_HP_FOCUS"] += 1
        rows.append(
            {
                **row,
                "first_divergence_index": index,
                "pi2_action_pi2_continuation": pi2_branch,
                "pi3_action_pi2_continuation": pi3_branch,
                "pi2_vs_pi3_action_preference_under_pi2_continuation": preference,
                "eventual_recursive_pi3_result": pi3_known[case]["route_terminal_status"],
            }
        )
    payload = {
        "failure_case_count": len(failure_cases),
        "first_divergence_found": sum(row.get("first_divergence_index") is not None for row in rows),
        "motifs": dict(motifs),
        "rows": rows,
    }
    _write_json(FIRST_DIV_PATH, payload)
    return payload


def _support_and_compounding(
    diagnostics_by_case: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    compounding: defaultdict[str, dict[str, float]] = defaultdict(lambda: {"decisions": 0, "failed": 0, "hp_divergence_sum": 0.0, "hp_matches": 0})
    pi2_known, _pi3_known = _known_cp9_rows()
    support_stats: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"decisions": 0, "disagreements": 0, "failed_route_decisions": 0})
    for case, rows in diagnostics_by_case.items():
        pi2_audit = list(pi2_known[case].get("decision_audit", []))
        deviations = 0
        for index, raw in enumerate(rows):
            row = dict(raw)
            if not row["agreement"]:
                deviations += 1
            bucket = str(deviations) if deviations <= 3 else "4+"
            failed = not bool(row["pi3_case_win"])
            compounding[bucket]["decisions"] += 1
            compounding[bucket]["failed"] += int(failed)
            if index < len(pi2_audit):
                diff = abs(float(row["party_hp_total"]) - float(pi2_audit[index]["party_hp"]))
                compounding[bucket]["hp_divergence_sum"] += diff
                compounding[bucket]["hp_matches"] += 1
                row["index_aligned_pi2_party_hp_difference"] = diff
            category = str(row["support"]["category"])
            support_stats[category]["decisions"] += 1
            support_stats[category]["disagreements"] += int(not row["agreement"])
            support_stats[category]["failed_route_decisions"] += int(failed)
            row["cumulative_deviations"] = deviations
            all_rows.append(row)
    support_payload = {
        "criteria_source": "frozen CP9 TRAIN support profile",
        "categories": {
            category: {
                **stats,
                "disagreement_rate": stats["disagreements"] / stats["decisions"] if stats["decisions"] else 0.0,
                "failed_route_decision_rate": stats["failed_route_decisions"] / stats["decisions"] if stats["decisions"] else 0.0,
            }
            for category, stats in support_stats.items()
        },
        "decision_rows": all_rows,
    }
    comp_payload = {
        "by_cumulative_deviation_count": {
            bucket: {
                **stats,
                "failure_rate": stats["failed"] / stats["decisions"] if stats["decisions"] else 0.0,
                "mean_index_aligned_party_hp_divergence": stats["hp_divergence_sum"] / stats["hp_matches"] if stats["hp_matches"] else None,
            }
            for bucket, stats in sorted(compounding.items())
        }
    }
    relationship = {"support_categories": support_payload["categories"]}
    _write_json(SUPPORT_PATH, support_payload)
    _write_json(COMPOUND_PATH, comp_payload)
    _write_json(SUPPORT_ERROR_PATH, relationship)
    return support_payload, comp_payload, relationship


def _sample_continuation_states(
    sources_by_case: Mapping[str, Sequence[RouteSourceDecision]],
    diagnostics_by_case: Mapping[str, Sequence[Mapping[str, Any]]],
    first_divergence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}
    source_lookup: dict[str, tuple[str, int]] = {}
    first_keys = {(row["case_id"], int(row["first_divergence_index"])) for row in first_divergence["rows"] if row.get("first_divergence_index") is not None}
    margins: list[float] = []
    for case, rows in diagnostics_by_case.items():
        for index, raw in enumerate(rows):
            if raw["agreement"]:
                continue
            provenance = {
                "case_id": case,
                "decision_index": index,
                "route_id": raw["route_id"],
                "root_seed": raw["root_seed"],
                "state_digest": raw["state_digest"],
                "encounter_id": raw["encounter_id"],
                "round_number": raw["round_number"],
                "support_category": raw["support"]["category"],
                "margin": raw["top_vs_anchor_margin"],
                "route_choice_families": raw["route_choice_families"],
                "is_first_divergence": (case, index) in first_keys,
            }
            token = _stable_hash(provenance)
            pool[token] = provenance
            source_lookup[token] = (case, index)
            margins.append(float(raw["top_vs_anchor_margin"]))
    strong_threshold = sorted(margins)[max(0, int(0.75 * (len(margins) - 1)))] if margins else 0.0
    selected: list[str] = []
    def add(items: Sequence[str], limit: int) -> None:
        for token in sorted(items):
            if token not in selected:
                selected.append(token)
            if len([x for x in selected if x in items]) >= limit:
                break
    add([t for t, p in pool.items() if p["is_first_divergence"]], 8)
    add([t for t, p in pool.items() if "MARK" in p["route_choice_families"]], 4)
    add([t for t, p in pool.items() if float(p["margin"]) >= strong_threshold], 4)
    add([t for t, p in pool.items() if p["support_category"] == POORLY_SUPPORTED], 4)
    add([t for t, p in pool.items() if p["support_category"] == WELL_SUPPORTED], 4)
    for token in sorted(pool):
        if len(selected) >= CONTINUATION_SAMPLE_COUNT:
            break
        if token not in selected:
            selected.append(token)
    selected = selected[:CONTINUATION_SAMPLE_COUNT]
    payload: list[dict[str, Any]] = []
    for token in selected:
        case, index = source_lookup[token]
        source = sources_by_case[case][index]
        payload.append({"selection_hash": token, **pool[token], "source": _source_provenance(source)})
    _write_json(CONT_SAMPLE_PATH, {"sample_count": len(payload), "strong_margin_threshold": strong_threshold, "sample": payload, "outcomes_opened": False})
    return payload


def _continuation_audit(
    definitions: Any,
    policies: Mapping[str, Any],
    pi3: FrozenRouteRankerPolicy,
    sample: Sequence[Mapping[str, Any]],
    sources_by_case: Mapping[str, Sequence[RouteSourceDecision]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    stable = reversals = indeterminate = 0
    for item in sample:
        case = str(item["case_id"])
        index = int(item["decision_index"])
        source = sources_by_case[case][index]
        pi2_branches: list[dict[str, Any]] = []
        pi3_branches: list[dict[str, Any]] = []
        for candidate in source.capture.candidates:
            pi2_branches.append(
                _mixed_branch(
                    definitions,
                    source,
                    candidate,
                    prefix_policy=pi3,
                    continuation_policy=policies["pi2_cp4"],
                    continuation_name="pi2_cp4",
                )
            )
            pi3_branches.append(
                _mixed_branch(
                    definitions,
                    source,
                    candidate,
                    prefix_policy=pi3,
                    continuation_policy=pi3,
                    continuation_name="pi3_route",
                )
            )
        pair_rows: list[dict[str, Any]] = []
        for first in range(len(source.capture.candidates)):
            for second in range(first + 1, len(source.capture.candidates)):
                p2 = _preference_from_payload(
                    pi2_branches[first]["route_outcome"], pi2_branches[second]["route_outcome"]
                )
                p3 = _preference_from_payload(
                    pi3_branches[first]["route_outcome"], pi3_branches[second]["route_outcome"]
                )
                ordered = {OutcomePreference.FIRST.value, OutcomePreference.SECOND.value}
                if p2 in ordered and p3 in ordered:
                    if p2 == p3:
                        status = "CONTINUATION_STABLE"
                        stable += 1
                    else:
                        status = "CONTINUATION_REVERSAL"
                        reversals += 1
                else:
                    status = "CONTINUATION_INDETERMINATE"
                    indeterminate += 1
                pair_rows.append(
                    {
                        "first_index": first,
                        "second_index": second,
                        "pi2_continuation_preference": p2,
                        "pi3_continuation_preference": p3,
                        "classification": status,
                    }
                )
        rows.append(
            {
                "sample": item,
                "pi2_continuation_branches": pi2_branches,
                "pi3_continuation_branches": pi3_branches,
                "pair_comparisons": pair_rows,
            }
        )
        print(f"CP10_CONTINUATION_DONE={case}|{index}", flush=True)
    payload = {
        "sample_count": len(rows),
        "pairwise_classification": {
            "CONTINUATION_STABLE": stable,
            "CONTINUATION_REVERSAL": reversals,
            "CONTINUATION_INDETERMINATE": indeterminate,
        },
        "rows": rows,
    }
    _write_json(CONT_PATH, payload)
    frozen_sample = _read_json(CONT_SAMPLE_PATH)
    frozen_sample["outcomes_opened"] = True
    frozen_sample["outcome_artifact_checksum"] = sha256_file(CONT_PATH)
    _write_json(CONT_SAMPLE_PATH, frozen_sample)
    return payload


def _multi_deviation_audit(
    definitions: Any,
    policies: Mapping[str, Any],
    pi3: FrozenRouteRankerPolicy,
) -> dict[str, Any]:
    pi2_known, pi3_known = _known_cp9_rows()
    failures = [
        case for case in pi2_known
        if bool(pi2_known[case]["route_failed_enemy_victory"])
        and not bool(pi3_known[case]["route_failed_enemy_victory"])
    ]
    failures.sort(key=lambda case: _stable_hash({"case": case, "purpose": "cp10-multi-deviation"}))
    selected = failures[:MULTI_DEVIATION_CASE_COUNT]
    rows: list[dict[str, Any]] = []
    for case in selected:
        known = pi2_known[case]
        route = _route_map()[str(known["route_id"])]
        seed = int(known["root_seed"])
        variants: dict[str, Any] = {}
        for label, policy in (
            ("BASE_PI2", policies["pi2_cp4"]),
            ("ONE_DEVIATION", LimitedDeviationPolicy(anchor_policy=policies["pi2_cp4"], route_policy=pi3, deviation_limit=1)),
            ("TWO_DEVIATION", LimitedDeviationPolicy(anchor_policy=policies["pi2_cp4"], route_policy=pi3, deviation_limit=2)),
            ("THREE_DEVIATION", LimitedDeviationPolicy(anchor_policy=policies["pi2_cp4"], route_policy=pi3, deviation_limit=3)),
            ("FULL_PI3", pi3),
        ):
            result = run_corrected_stateful_route(
                definitions,
                route,
                root_seed=seed,
                enemy_policy_name=label,
                enemy_policy=policy,
                hero_policy_id=PRIMARY_HERO_POLICY,
                safety_budget=SAFETY_BUDGET,
            )
            variants[label] = {
                "enemy_victory": bool(result["route_failed_enemy_victory"]),
                "terminal_status": result["route_terminal_status"],
                "encounters_completed": result["encounters_completed"],
                "final_company_state": result["final_company_state"],
                "deviations_used": getattr(policy, "deviations_used", None),
            }
        rows.append({"case_id": case, "route_id": route.route_id, "root_seed": seed, "variants": variants})
    payload = {
        "sample_count": len(rows),
        "rows": rows,
        "win_counts": {
            label: sum(bool(row["variants"][label]["enemy_victory"]) for row in rows)
            for label in ("BASE_PI2", "ONE_DEVIATION", "TWO_DEVIATION", "THREE_DEVIATION", "FULL_PI3")
        },
    }
    _write_json(MULTI_PATH, payload)
    return payload


def _family_audits(
    diagnostics_by_case: Mapping[str, Sequence[Mapping[str, Any]]],
    continuation: Mapping[str, Any],
) -> dict[str, Any]:
    decisions = [dict(row) for rows in diagnostics_by_case.values() for row in rows]
    disagreements = [row for row in decisions if not row["agreement"]]
    continuation_by_state = {
        str(row["sample"]["state_digest"]): row for row in continuation["rows"]
    }

    def family_summary(family: str) -> dict[str, Any]:
        chosen = [row for row in decisions if family in set(row["route_choice_families"])]
        disagree = [row for row in disagreements if family in set(row["route_choice_families"])]
        return {
            "decision_count": len(decisions),
            "chosen_count": len(chosen),
            "chosen_rate_all_decisions": len(chosen) / len(decisions) if decisions else 0.0,
            "route_choice_disagreement_count": len(disagree),
            "mean_route_margin_when_chosen": mean(float(row["top_vs_anchor_margin"]) for row in chosen) if chosen else None,
            "support_when_chosen": dict(Counter(str(row["support"]["category"]) for row in chosen)),
        }

    mark = family_summary("MARK")
    grab = family_summary("GRAB")
    drain = family_summary("EFFORT_DRAIN")
    direct_lost = [
        row for row in disagreements
        if "DIRECT_PRESSURE" in set(row["anchor_choice_families"])
        and "DIRECT_PRESSURE" not in set(row["route_choice_families"])
    ]
    targeting = {
        "decisions": len(decisions),
        "lowest_hp_route_choice": sum(bool(row["route_target_is_lowest_hp"]) for row in decisions),
        "downed_target_route_choice": sum(bool(row["route_target_is_downed"]) for row in decisions),
        "support_target_route_choice": sum(bool(row["route_target_support_capable"]) for row in decisions),
        "disagreement_target_changes": sum(
            row["route_key"]["target_id"] != row["anchor_key"]["target_id"] for row in disagreements
        ),
    }
    pressure = {
        "lost_direct_pressure_disagreements": len(direct_lost),
        "mean_projected_damage_delta_route_minus_anchor": mean(
            float(row["route_projected_damage_max"]) - float(row["anchor_projected_damage_max"])
            for row in disagreements
        ) if disagreements else 0.0,
        "examples": direct_lost[:32],
    }
    continuation_mark = [
        row for row in continuation["rows"]
        if "MARK" in set(row["sample"].get("route_choice_families", []))
    ]
    mark["continuation_sample_states"] = len(continuation_mark)
    mark["continuation_hypotheses"] = [
        "MARK_CORRELATION_BIAS",
        "MARK_EXTRAPOLATION",
        "CONTINUATION_DEPENDENCE",
        "ACTION_FAMILY_SCORE_SHIFT",
        "NO_SINGLE_MARK_FAILURE_MODE",
    ]
    drain_chosen = [row for row in decisions if "EFFORT_DRAIN" in set(row["route_choice_families"])]
    drain["selected_when_route_top_family_rate"] = len(drain_chosen) / max(1, len([row for row in decisions if "EFFORT_DRAIN" in set().union(*[set(x) for x in row.get("candidate_families", [])])]))
    _write_json(MARK_PATH, mark)
    _write_json(TARGET_PATH, targeting)
    _write_json(PRESSURE_PATH, pressure)
    _write_json(GRAB_PATH, grab)
    _write_json(DRAIN_PATH, drain)
    return {"mark": mark, "targeting": targeting, "pressure": pressure, "grab": grab, "drain": drain, "continuation_indexed_states": len(continuation_by_state)}


def diagnose() -> None:
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError("Freeze CP10 protocol before diagnosis")
    protocol = _read_json(PROTOCOL_PATH)
    if sha256_file(PROTOCOL_PATH) != _read_json(PROVENANCE_PATH)["protocol_checksum"]:
        raise ValueError("CP10 protocol changed after freeze")
    definitions = load_game_definitions()
    policies, pi3, normalization, _selected = _load_models()
    train, _validation, _tests = _load_cp9_datasets()
    profile = fit_support_profile(_record_vectors(train), normalization)
    reproduction = _reproduce_cp9(definitions, policies, pi3)
    top1 = top1_audit()
    sources_by_case, diagnostics_by_case = _source_decision_rows(definitions, policies, pi3, profile)
    first = _first_divergences(definitions, policies, pi3, sources_by_case, diagnostics_by_case)
    support, compounding, relationship = _support_and_compounding(diagnostics_by_case)
    sample = _sample_continuation_states(sources_by_case, diagnostics_by_case, first)
    continuation = _continuation_audit(definitions, policies, pi3, sample, sources_by_case)
    multi = _multi_deviation_audit(definitions, policies, pi3)
    family = _family_audits(diagnostics_by_case, continuation)
    diagnosis = {
        "reproduction": reproduction,
        "top1_summary": top1["groups"],
        "first_divergence_motifs": first["motifs"],
        "support_summary": support["categories"],
        "compounding": compounding,
        "continuation_pairwise": continuation["pairwise_classification"],
        "multi_deviation_win_counts": multi["win_counts"],
        "family_audits": family,
        "support_error_relationship": relationship,
        "diagnosis_artifact_checksums": {
            path.name: sha256_file(path)
            for path in (
                REPRO_PATH, TOP1_PATH, FIRST_DIV_PATH, SUPPORT_PATH, COMPOUND_PATH,
                CONT_SAMPLE_PATH, CONT_PATH, MULTI_PATH, MARK_PATH, TARGET_PATH,
                PRESSURE_PATH, GRAB_PATH, DRAIN_PATH, SUPPORT_ERROR_PATH,
            )
        },
        "new_cp10_outcomes_accessed": False,
    }
    _write_json(DIAGNOSIS_FREEZE_PATH, diagnosis)
    repair = {
        "status": "frozen_before_cp10_development_outcomes",
        "candidate_family": protocol["candidate_family"],
        "model_family": protocol["model_family"],
        "repaired_data": protocol["repaired_data"],
        "validation_sampling": protocol["validation_sampling"],
        "margin_rule": protocol["margin_rule"],
        "ranker_selection": protocol["ranker_selection"],
        "gate_selection": protocol["gate_selection"],
        "diagnosis_checksum": sha256_file(DIAGNOSIS_FREEZE_PATH),
        "test_accessed": False,
        "reserved_confirmation_accessed": False,
    }
    _write_json(REPAIR_CANDIDATES_PATH, repair)
    print("CP10_DIAGNOSIS_FROZEN=true")
    print(f"CP10_CONTINUATION_REVERSALS={continuation['pairwise_classification']['CONTINUATION_REVERSAL']}")
    print("CP10_NEW_OUTCOMES_ACCESSED=false")


def _source_identity(source: RouteSourceDecision) -> str:
    return _stable_hash(_source_provenance(source))


def _record_identity(record: RouteDecisionRecord) -> str:
    return _stable_hash(
        {
            "split": record.split,
            "route_id": record.route_id,
            "root_seed": record.root_seed,
            "encounter_id": record.encounter_id,
            "round_number": record.round_number,
            "state_digest": record.state_digest,
        }
    )


def _sample_source_pairs(
    pairs: Sequence[tuple[RouteSourceDecision, CandidateKey]],
    count: int,
    *,
    purpose: str,
) -> list[tuple[RouteSourceDecision, CandidateKey]]:
    if count <= 0:
        return []
    by_route: defaultdict[str, list[tuple[str, RouteSourceDecision, CandidateKey]]] = defaultdict(list)
    for source, anchor in pairs:
        token = _stable_hash({"purpose": purpose, **_source_provenance(source)})
        by_route[source.route.route_id].append((token, source, anchor))
    per_route = count // len(PRIMARY_ROUTES)
    selected: list[tuple[str, RouteSourceDecision, CandidateKey]] = []
    leftovers: list[tuple[str, RouteSourceDecision, CandidateKey]] = []
    for route in PRIMARY_ROUTES:
        rows = sorted(by_route[route.route_id], key=lambda item: item[0])
        selected.extend(rows[:per_route])
        leftovers.extend(rows[per_route:])
    for row in sorted(leftovers, key=lambda item: item[0]):
        if len(selected) >= count:
            break
        selected.append(row)
    if len(selected) < count:
        raise RuntimeError(f"Insufficient source states for {purpose}: {len(selected)} < {count}")
    return [(source, anchor) for _token, source, anchor in selected[:count]]


def _collect_source_pool(
    definitions: Any,
    *,
    seeds: Sequence[int],
    source_name: str,
    source_policy: Any,
    anchor_policy: Any,
    route_policy: FrozenRouteRankerPolicy,
    diagnosis_profile: SupportProfile,
    split: str,
) -> list[tuple[RouteSourceDecision, CandidateKey]]:
    output: list[tuple[RouteSourceDecision, CandidateKey]] = []
    for seed in seeds:
        for route in PRIMARY_ROUTES:
            if source_name == "pi2_cp4":
                sources = collect_route_source_decisions(
                    definitions,
                    route,
                    root_seed=seed,
                    source_policy=source_policy,
                    source_policy_name=source_name,
                    hero_policy_id=PRIMARY_HERO_POLICY,
                    split=split,
                    safety_budget=SAFETY_BUDGET,
                )
                output.extend((source, source.capture.source_choice) for source in sources)
            else:
                diagnostic = DiagnosticRoutePolicy(
                    executed_name=source_name,
                    executed_policy=source_policy,
                    anchor_policy=anchor_policy,
                    route_policy=route_policy,
                    support_profile=diagnosis_profile,
                )
                sources = collect_route_source_decisions(
                    definitions,
                    route,
                    root_seed=seed,
                    source_policy=diagnostic,
                    source_policy_name=source_name,
                    hero_policy_id=PRIMARY_HERO_POLICY,
                    split=split,
                    safety_budget=SAFETY_BUDGET,
                )
                if len(sources) != len(diagnostic.decisions):
                    raise RuntimeError("Source/anchor audit decision count mismatch")
                for source, decision in zip(sources, diagnostic.decisions, strict=True):
                    output.append((source, _key(decision["anchor_key"])))
        print(f"CP10_SOURCE_POOL_DONE={source_name}|{split}|{seed}", flush=True)
    return output


def _record_from_mixed_branches(
    source: RouteSourceDecision,
    branches: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> RouteDecisionRecord:
    by_key = {_key(branch["candidate"]): branch for branch in branches}
    candidates: list[RouteCandidateRecord] = []
    for key, primary, oracle in zip(
        source.capture.candidates,
        source.capture.primary_observations,
        source.capture.oracle_observations,
        strict=True,
    ):
        branch = by_key[key]
        candidates.append(
            RouteCandidateRecord(
                key=key,
                primary_observation=primary,
                oracle_observation=oracle,
                route_outcome=_route_outcome(branch["route_outcome"]),
                local_outcome=_local_outcome(branch.get("local_outcome")),
            )
        )
    source_index = source.capture.candidates.index(source.capture.source_choice)
    total = source.capture.total_encounters
    encounter_index = source.capture.encounter_index
    position = (
        "SINGLE"
        if total <= 1
        else "EARLY"
        if encounter_index == 0
        else "LATE"
        if encounter_index == total - 1
        else "MIDDLE"
    )
    return RouteDecisionRecord(
        split=split,
        route_id=source.route.route_id,
        encounter_index=encounter_index,
        encounter_id=source.capture.encounter_id,
        route_position=position,
        hero_policy_id=source.hero_policy_id,
        root_seed=source.root_seed,
        enemy_class_id=source.capture.enemy_class_id,
        round_number=source.capture.round_number,
        state_digest=source.capture.state_digest,
        rng_digest=source.capture.rng_digest,
        candidates=tuple(candidates),
        source_policy_index=source_index,
        control_exact=True,
    )


def _branch_sampled_sources(
    definitions: Any,
    sampled: Sequence[tuple[RouteSourceDecision, CandidateKey]],
    *,
    prefix_policy: Any,
    continuation_policy: Any,
    continuation_name: str,
    split: str,
) -> tuple[list[RouteDecisionRecord], dict[str, int], list[dict[str, Any]]]:
    records: list[RouteDecisionRecord] = []
    anchors: dict[str, int] = {}
    branch_audit: list[dict[str, Any]] = []
    for source, anchor in sampled:
        branches = [
            _mixed_branch(
                definitions,
                source,
                candidate,
                prefix_policy=prefix_policy,
                continuation_policy=continuation_policy,
                continuation_name=continuation_name,
            )
            for candidate in source.capture.candidates
        ]
        record = _record_from_mixed_branches(source, branches, split=split)
        record_key = _record_identity(record)
        anchors[record_key] = source.capture.candidates.index(anchor)
        records.append(record)
        branch_audit.append(
            {
                "record_id": record_key,
                "source": _source_provenance(source),
                "anchor_action": asdict(anchor),
                "candidate_count": len(branches),
                "continuation": continuation_name,
            }
        )
        print(
            f"CP10_BRANCH_DONE={split}|{source.route.route_id}|{source.root_seed}|{source.capture.global_action_index}",
            flush=True,
        )
    return records, anchors, branch_audit


def _merge_dataset(split: str, *datasets: RouteDataset | Sequence[RouteDecisionRecord]) -> RouteDataset:
    records: list[RouteDecisionRecord] = []
    for dataset in datasets:
        if isinstance(dataset, RouteDataset):
            records.extend(dataset.records)
        else:
            records.extend(dataset)
    return RouteDataset(split=split, records=tuple(records))


def _anchor_map_for_cp9(dataset: RouteDataset) -> dict[str, int]:
    return {_record_identity(record): record.source_policy_index for record in dataset.records}


def _candidate_policy_metrics(
    dataset: RouteDataset,
    model: Any,
    normalization: RouteNormalization,
    anchor_indices: Mapping[str, int],
    support_profile: SupportProfile,
    *,
    gate_mode: str,
    margin_threshold: float,
    source_labels: Mapping[str, str],
) -> dict[str, Any]:
    states = preferred = pi3_induced_preferred = deviations = beneficial = harmful = accepted = 0
    support_counts: Counter[str] = Counter()
    accepted_support: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for record in dataset.records:
        record_id = _record_identity(record)
        anchor_index = int(anchor_indices[record_id])
        scores, order = _score_indices(model, normalization, record)
        proposed_index = order[0]
        preferred_set = _preferred_set(record)
        raw = torch.tensor([item.primary_observation for item in record.candidates], dtype=torch.float32)
        normalized = normalization.normalize(raw)
        support = support_profile.classify_normalized(normalized[proposed_index])
        margin = scores[proposed_index] - scores[anchor_index]
        differs = proposed_index != anchor_index
        if gate_mode == "PURE_PI2":
            use = False
        elif gate_mode == "PURE_ROUTE_RANKER":
            use = differs
        elif gate_mode == "SUPPORT_GATED":
            use = differs and support.category == WELL_SUPPORTED
        elif gate_mode == "MARGIN_GATED":
            use = differs and margin >= margin_threshold
        else:
            use = differs and support.category == WELL_SUPPORTED and margin >= margin_threshold
        executed = proposed_index if use else anchor_index
        states += 1
        preferred += int(executed in preferred_set)
        if source_labels.get(record_id) == "pi3_induced":
            pi3_induced_preferred += int(executed in preferred_set)
        deviations += int(use)
        accepted += int(use)
        if use:
            beneficial += int(proposed_index in preferred_set and anchor_index not in preferred_set)
            harmful += int(proposed_index not in preferred_set and anchor_index in preferred_set)
            accepted_support[support.category] += 1
        support_counts[support.category] += 1
        rows.append(
            {
                "record_id": record_id,
                "source_type": source_labels.get(record_id),
                "anchor_index": anchor_index,
                "proposed_index": proposed_index,
                "executed_index": executed,
                "preferred_set": sorted(preferred_set),
                "support": asdict(support),
                "margin": margin,
                "deviated": use,
                "beneficial_deviation": use and proposed_index in preferred_set and anchor_index not in preferred_set,
                "harmful_deviation": use and proposed_index not in preferred_set and anchor_index in preferred_set,
            }
        )
    pi3_count = sum(source == "pi3_induced" for source in source_labels.values())
    return {
        "states": states,
        "top1_preferred_set_rate": preferred / states if states else 0.0,
        "pi3_induced_top1_preferred_set_rate": pi3_induced_preferred / pi3_count if pi3_count else 0.0,
        "deviation_rate": deviations / states if states else 0.0,
        "accepted_deviations": accepted,
        "beneficial_accepted": beneficial,
        "harmful_accepted": harmful,
        "beneficial_deviation_precision": beneficial / accepted if accepted else 1.0,
        "support_counts": dict(support_counts),
        "accepted_support": dict(accepted_support),
        "rows": rows,
    }


def _margin_threshold_from_validation(
    dataset: RouteDataset,
    model: Any,
    normalization: RouteNormalization,
    anchor_indices: Mapping[str, int],
) -> dict[str, Any]:
    beneficial: list[float] = []
    positives: list[float] = []
    for record in dataset.records:
        anchor = anchor_indices[_record_identity(record)]
        scores, order = _score_indices(model, normalization, record)
        proposed = order[0]
        if proposed == anchor:
            continue
        margin = scores[proposed] - scores[anchor]
        if margin > 0:
            positives.append(margin)
        preferred = _preferred_set(record)
        if proposed in preferred and anchor not in preferred and margin > 0:
            beneficial.append(margin)
    values = beneficial if beneficial else positives
    if not values:
        threshold = float("inf")
        source = "no_positive_disagreement_margin"
    else:
        probability = 0.25 if beneficial else 0.75
        ordered = sorted(values)
        point = (len(ordered) - 1) * probability
        low = int(point)
        high = min(len(ordered) - 1, low + 1)
        fraction = point - low
        threshold = ordered[low] * (1 - fraction) + ordered[high] * fraction
        source = "beneficial_q25" if beneficial else "positive_disagreement_q75"
    return {
        "threshold": threshold,
        "source": source,
        "beneficial_margin_count": len(beneficial),
        "positive_disagreement_margin_count": len(positives),
    }


def _recursive_validation_for_mode(
    definitions: Any,
    policies: Mapping[str, Any],
    route_policy: FrozenRouteRankerPolicy,
    support_profile: SupportProfile,
    *,
    mode: str,
    margin_threshold: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in VALIDATION_SEEDS:
        for route in PRIMARY_ROUTES:
            candidate = ConservativeRoutePolicy(
                anchor_policy=policies["pi2_cp4"],
                route_policy=route_policy,
                support_profile=support_profile,
                gate_mode=mode,
                margin_threshold=margin_threshold,
                policy_id=f"cp10_validation_{mode}",
            )
            result = run_corrected_stateful_route(
                definitions,
                route,
                root_seed=seed,
                enemy_policy_name=f"cp10_validation_{mode}",
                enemy_policy=candidate,
                hero_policy_id=PRIMARY_HERO_POLICY,
                safety_budget=SAFETY_BUDGET,
            )
            rows.append(
                {
                    "case_id": _case_id(route.route_id, seed),
                    "route_id": route.route_id,
                    "root_seed": seed,
                    "enemy_victory": bool(result["route_failed_enemy_victory"]),
                    "terminal_status": result["route_terminal_status"],
                    "decisions": len(candidate.audit),
                    "deviations": sum(bool(item["gate_accepted"]) for item in candidate.audit),
                    "audit": candidate.audit,
                }
            )
    return {
        "case_count": len(rows),
        "enemy_wins": sum(bool(row["enemy_victory"]) for row in rows),
        "enemy_win_rate": sum(bool(row["enemy_victory"]) for row in rows) / len(rows) if rows else 0.0,
        "deviations": sum(int(row["deviations"]) for row in rows),
        "decisions": sum(int(row["decisions"]) for row in rows),
        "rows": rows,
    }


def _deployment_aware_label_audit(
    definitions: Any,
    sampled: Sequence[tuple[RouteSourceDecision, CandidateKey]],
    pi3: FrozenRouteRankerPolicy,
    pi2_records: Mapping[str, RouteDecisionRecord],
) -> dict[str, Any]:
    ordered = sorted(
        sampled,
        key=lambda pair: _stable_hash({"purpose": "cp10-deployment-label", **_source_provenance(pair[0])}),
    )[:18]
    frozen_selection = [
        {"source": _source_provenance(source), "selection_hash": _stable_hash({"purpose": "cp10-deployment-label", **_source_provenance(source)})}
        for source, _anchor in ordered
    ]
    _write_json(CONT_LABEL_PATH, {"sample": frozen_selection, "outcomes_opened": False})
    rows: list[dict[str, Any]] = []
    stable = reversals = indeterminate = 0
    for source, _anchor in ordered:
        record_id = _source_identity(source)
        pi2_record = pi2_records[record_id]
        pi3_branches = [
            _mixed_branch(
                definitions,
                source,
                candidate,
                prefix_policy=pi3,
                continuation_policy=pi3,
                continuation_name="pi3_route",
            )
            for candidate in source.capture.candidates
        ]
        pair_rows: list[dict[str, Any]] = []
        for first in range(len(source.capture.candidates)):
            for second in range(first + 1, len(source.capture.candidates)):
                p2 = compare_route_outcomes(
                    pi2_record.candidates[first].route_outcome,
                    pi2_record.candidates[second].route_outcome,
                ).value
                p3 = _preference_from_payload(
                    pi3_branches[first]["route_outcome"], pi3_branches[second]["route_outcome"]
                )
                ordered_preferences = {OutcomePreference.FIRST.value, OutcomePreference.SECOND.value}
                if p2 in ordered_preferences and p3 in ordered_preferences:
                    if p2 == p3:
                        status = "CONTINUATION_STABLE"
                        stable += 1
                    else:
                        status = "CONTINUATION_REVERSAL"
                        reversals += 1
                else:
                    status = "CONTINUATION_INDETERMINATE"
                    indeterminate += 1
                pair_rows.append(
                    {
                        "first_index": first,
                        "second_index": second,
                        "pi2_continuation_preference": p2,
                        "pi3_continuation_preference": p3,
                        "classification": status,
                    }
                )
        rows.append({"source": _source_provenance(source), "pair_comparisons": pair_rows})
    payload = {
        "sample_count": len(rows),
        "continuation_policy_approximation": "pi3_route used as frozen deployment-aware comparison; no recursive fitting",
        "classification": {
            "CONTINUATION_STABLE": stable,
            "CONTINUATION_REVERSAL": reversals,
            "CONTINUATION_INDETERMINATE": indeterminate,
        },
        "rows": rows,
        "outcomes_opened": True,
    }
    _write_json(CONT_LABEL_PATH, payload)
    return payload


def develop() -> None:
    if not DIAGNOSIS_FREEZE_PATH.exists() or not REPAIR_CANDIDATES_PATH.exists():
        raise FileNotFoundError("Run and freeze CP10 diagnosis before development")
    provenance = _read_json(PROVENANCE_PATH)
    if sha256_file(PROTOCOL_PATH) != provenance["protocol_checksum"]:
        raise ValueError("CP10 protocol changed after freeze")
    definitions = load_game_definitions()
    policies, pi3, cp9_normalization, _selected = _load_models()
    cp9_train, _cp9_validation, _cp9_tests = _load_cp9_datasets()
    diagnosis_profile = fit_support_profile(_record_vectors(cp9_train), cp9_normalization)

    dev_pool = _collect_source_pool(
        definitions,
        seeds=DEV_SEEDS,
        source_name="pi3_route",
        source_policy=pi3,
        anchor_policy=policies["pi2_cp4"],
        route_policy=pi3,
        diagnosis_profile=diagnosis_profile,
        split="cp10_train_pi3_induced",
    )
    sampled_dev = _sample_source_pairs(dev_pool, PI3_DEV_SAMPLE, purpose="cp10-train-pi3-induced")
    _write_json(
        ARTIFACT_ROOT / "pi3_induced_train_sample.json",
        {"sample_count": len(sampled_dev), "sample": [{"source": _source_provenance(source), "anchor": asdict(anchor)} for source, anchor in sampled_dev], "branch_outcomes_opened": False},
    )
    dev_records, dev_anchors, dev_branch_audit = _branch_sampled_sources(
        definitions,
        sampled_dev,
        prefix_policy=pi3,
        continuation_policy=policies["pi2_cp4"],
        continuation_name="pi2_cp4",
        split="cp10_train_pi3_induced",
    )
    dev_record_by_source = {
        _source_identity(source): record
        for (source, _anchor), record in zip(sampled_dev, dev_records, strict=True)
    }
    _deployment_aware_label_audit(
        definitions, sampled_dev, pi3, dev_record_by_source
    )

    repaired_train = _merge_dataset("train", cp9_train, dev_records)
    train_anchors = _anchor_map_for_cp9(cp9_train)
    train_anchors.update(dev_anchors)
    train_source_labels = {_record_identity(record): "pi2_source" for record in cp9_train.records}
    train_source_labels.update({_record_identity(record): "pi3_induced" for record in dev_records})

    val_pi2_pool = _collect_source_pool(
        definitions,
        seeds=VALIDATION_SEEDS,
        source_name="pi2_cp4",
        source_policy=policies["pi2_cp4"],
        anchor_policy=policies["pi2_cp4"],
        route_policy=pi3,
        diagnosis_profile=diagnosis_profile,
        split="cp10_validation_pi2_source",
    )
    val_pi3_pool = _collect_source_pool(
        definitions,
        seeds=VALIDATION_SEEDS,
        source_name="pi3_route",
        source_policy=pi3,
        anchor_policy=policies["pi2_cp4"],
        route_policy=pi3,
        diagnosis_profile=diagnosis_profile,
        split="cp10_validation_pi3_induced",
    )
    sampled_val_pi2 = _sample_source_pairs(val_pi2_pool, VALIDATION_PI2_SAMPLE, purpose="cp10-validation-pi2")
    sampled_val_pi3 = _sample_source_pairs(val_pi3_pool, VALIDATION_PI3_SAMPLE, purpose="cp10-validation-pi3")
    _write_json(
        ARTIFACT_ROOT / "validation_source_sample.json",
        {
            "pi2": [{"source": _source_provenance(source), "anchor": asdict(anchor)} for source, anchor in sampled_val_pi2],
            "pi3": [{"source": _source_provenance(source), "anchor": asdict(anchor)} for source, anchor in sampled_val_pi3],
            "branch_outcomes_opened": False,
        },
    )
    val_pi2_records, val_pi2_anchors, val_pi2_audit = _branch_sampled_sources(
        definitions,
        sampled_val_pi2,
        prefix_policy=policies["pi2_cp4"],
        continuation_policy=policies["pi2_cp4"],
        continuation_name="pi2_cp4",
        split="cp10_validation_pi2_source",
    )
    val_pi3_records, val_pi3_anchors, val_pi3_audit = _branch_sampled_sources(
        definitions,
        sampled_val_pi3,
        prefix_policy=pi3,
        continuation_policy=policies["pi2_cp4"],
        continuation_name="pi2_cp4",
        split="cp10_validation_pi3_induced",
    )
    validation = _merge_dataset("validation", val_pi2_records, val_pi3_records)
    val_anchors = {**val_pi2_anchors, **val_pi3_anchors}
    val_source_labels = {_record_identity(record): "pi2_source" for record in val_pi2_records}
    val_source_labels.update({_record_identity(record): "pi3_induced" for record in val_pi3_records})

    _write_json(
        ARTIFACT_ROOT / "repaired_train_dataset.json",
        {"dataset": _dataset_payload(repaired_train), "anchor_indices": train_anchors, "source_labels": train_source_labels},
    )
    _write_json(
        ARTIFACT_ROOT / "repaired_validation_dataset.json",
        {"dataset": _dataset_payload(validation), "anchor_indices": val_anchors, "source_labels": val_source_labels},
    )

    normalization = fit_route_normalization(repaired_train)
    norm_sha = save_normalization(normalization, NORMALIZATION_PATH)
    support_profile = fit_support_profile(_record_vectors(repaired_train), normalization)
    _write_json(ARTIFACT_ROOT / "repaired_support_profile.json", support_profile_payload(support_profile))

    models: list[dict[str, Any]] = []
    for model_id, spec, seed in (
        ("cp10_route_linear", linear_scorer_spec(PRIMARY_FEATURE_NAMES), LINEAR_SEED),
        ("cp10_route_mlp_seed_20261013", mlp_scorer_spec(PRIMARY_FEATURE_NAMES), MLP_SEED),
    ):
        trained = train_route_ranker(
            spec,
            repaired_train,
            validation,
            normalization,
            seed=seed,
            config=TRAINING_CONFIG,
        )
        metadata_path = ARTIFACT_ROOT / f"{model_id}.json"
        metadata = save_route_ranker(
            trained,
            metadata_path,
            normalization_checksum=norm_sha,
            schema_id="enemy-route-outcome-candidate-v1",
            oracle=False,
        )
        metrics = evaluate_route_ranker(trained.model, validation, normalization)
        models.append(
            {
                "model_id": model_id,
                "model_checksum": metadata["model_checksum"],
                "metadata_path": str(metadata_path),
                "metadata_checksum": sha256_file(metadata_path),
                "validation": metrics,
            }
        )
    models.sort(
        key=lambda row: (
            -float(row["validation"]["accuracy"]),
            float(row["validation"]["loss"]),
            0 if row["model_id"] == "cp10_route_linear" else 1,
        )
    )
    selected_model = models[0]
    model, metadata = load_route_ranker(Path(str(selected_model["metadata_path"])))
    repaired_policy = FrozenRouteRankerPolicy(model, normalization, policy_id="cp10_repaired_route_ranker")
    margin_info = _margin_threshold_from_validation(validation, model, normalization, val_anchors)
    margin_threshold = float(margin_info["threshold"])

    candidate_metrics: dict[str, Any] = {}
    for mode in _read_json(PROTOCOL_PATH)["candidate_family"]:
        offline = _candidate_policy_metrics(
            validation,
            model,
            normalization,
            val_anchors,
            support_profile,
            gate_mode=mode,
            margin_threshold=margin_threshold,
            source_labels=val_source_labels,
        )
        recursive = _recursive_validation_for_mode(
            definitions,
            policies,
            repaired_policy,
            support_profile,
            mode=mode,
            margin_threshold=margin_threshold,
        )
        offline["recursive_validation"] = {
            "case_count": recursive["case_count"],
            "enemy_wins": recursive["enemy_wins"],
            "enemy_win_rate": recursive["enemy_win_rate"],
            "deviation_rate": recursive["deviations"] / recursive["decisions"] if recursive["decisions"] else 0.0,
        }
        candidate_metrics[mode] = offline
        _write_json(ARTIFACT_ROOT / f"validation_{mode.lower()}.json", recursive)

    pi2_rate = float(candidate_metrics["PURE_PI2"]["recursive_validation"]["enemy_win_rate"])
    eligible = ["SUPPORT_GATED", "MARGIN_GATED", "SUPPORT_AND_MARGIN_GATED"]
    if all(float(candidate_metrics[mode]["recursive_validation"]["enemy_win_rate"]) <= pi2_rate - 0.10 for mode in eligible):
        selected_mode = "PURE_PI2"
        selection_reason = "all conservative gates regressed >=10pp vs pi2 on validation"
    else:
        viable = [mode for mode in eligible if float(candidate_metrics[mode]["recursive_validation"]["enemy_win_rate"]) > pi2_rate - 0.10]
        viable.sort(
            key=lambda mode: (
                -float(candidate_metrics[mode]["top1_preferred_set_rate"]),
                -float(candidate_metrics[mode]["pi3_induced_top1_preferred_set_rate"]),
                -float(candidate_metrics[mode]["beneficial_deviation_precision"]),
                -float(candidate_metrics[mode]["recursive_validation"]["enemy_win_rate"]),
                float(candidate_metrics[mode]["deviation_rate"]),
                mode,
            )
        )
        selected_mode = viable[0]
        selection_reason = "predeclared conservative validation lexicographic rule"

    repaired_summary = {
        "original_pi2_source_records": len(cp9_train.records),
        "pi3_induced_added_records": len(dev_records),
        "train_records": len(repaired_train.records),
        "validation_records": len(validation.records),
        "train_decisive_pairs": len(decisive_pairs(repaired_train)),
        "validation_decisive_pairs": len(decisive_pairs(validation)),
        "source_labels_train": dict(Counter(train_source_labels.values())),
        "source_labels_validation": dict(Counter(val_source_labels.values())),
        "dev_branch_audit": dev_branch_audit,
        "validation_branch_audit": [*val_pi2_audit, *val_pi3_audit],
        "continuation_label_analysis_checksum": sha256_file(CONT_LABEL_PATH),
    }
    _write_json(REPAIRED_SUMMARY_PATH, repaired_summary)
    validation_payload = {
        "models": models,
        "selected_ranker": selected_model["model_id"],
        "margin_threshold": margin_info,
        "candidate_metrics": candidate_metrics,
        "selected_gate_mode": selected_mode,
        "selection_reason": selection_reason,
        "test_used": False,
    }
    _write_json(VALIDATION_PATH, validation_payload)
    deviation_payload = {
        mode: {
            "offline_deviation_rate": metrics["deviation_rate"],
            "accepted_support": metrics["accepted_support"],
            "beneficial_precision": metrics["beneficial_deviation_precision"],
            "recursive_validation_deviation_rate": metrics["recursive_validation"]["deviation_rate"],
        }
        for mode, metrics in candidate_metrics.items()
    }
    _write_json(DEVIATION_PATH, deviation_payload)

    selected_payload = {
        "status": "frozen_before_test",
        "test_accessed": False,
        "post_test_tuning_performed": False,
        "policy_id": "pi4_conservative",
        "ranker_model_id": selected_model["model_id"],
        "ranker_model_checksum": selected_model["model_checksum"],
        "ranker_metadata_path": selected_model["metadata_path"],
        "ranker_metadata_checksum": selected_model["metadata_checksum"],
        "normalization_path": str(NORMALIZATION_PATH),
        "normalization_checksum": norm_sha,
        "support_profile_path": str(ARTIFACT_ROOT / "repaired_support_profile.json"),
        "support_profile_checksum": sha256_file(ARTIFACT_ROOT / "repaired_support_profile.json"),
        "gate_mode": selected_mode,
        "margin_threshold": margin_threshold,
        "validation_results_checksum": sha256_file(VALIDATION_PATH),
        "repaired_train_checksum": sha256_file(ARTIFACT_ROOT / "repaired_train_dataset.json"),
        "repaired_validation_checksum": sha256_file(ARTIFACT_ROOT / "repaired_validation_dataset.json"),
        "protocol_checksum": sha256_file(PROTOCOL_PATH),
        "diagnosis_checksum": sha256_file(DIAGNOSIS_FREEZE_PATH),
        "repair_candidates_checksum": sha256_file(REPAIR_CANDIDATES_PATH),
        "reserved_confirmation_accessed": False,
    }
    _write_json(SELECTED_PATH, selected_payload)
    provenance = _read_json(PROVENANCE_PATH)
    provenance["new_cp10_outcomes_accessed"] = True
    provenance["development_validation_complete"] = True
    provenance["test_accessed"] = False
    _write_json(PROVENANCE_PATH, provenance)
    print(f"CP10_SELECTED_RANKER={selected_model['model_id']}")
    print(f"CP10_SELECTED_GATE={selected_mode}")
    print(f"CP10_MARGIN_THRESHOLD={margin_threshold}")
    print("CP10_TEST_ACCESSED=false")


def _load_selected_pi4() -> tuple[FrozenRouteRankerPolicy, RouteNormalization, SupportProfile, dict[str, Any]]:
    selected = _read_json(SELECTED_PATH)
    metadata_path = Path(str(selected["ranker_metadata_path"]))
    if sha256_file(metadata_path) != selected["ranker_metadata_checksum"]:
        raise ValueError("CP10 selected ranker metadata checksum mismatch")
    model, metadata = load_route_ranker(metadata_path)
    if metadata["model_checksum"] != selected["ranker_model_checksum"]:
        raise ValueError("CP10 selected ranker model checksum mismatch")
    normalization = load_normalization(Path(str(selected["normalization_path"])))
    if sha256_file(Path(str(selected["normalization_path"]))) != selected["normalization_checksum"]:
        raise ValueError("CP10 normalization checksum mismatch")
    train_payload = _read_json(ARTIFACT_ROOT / "repaired_train_dataset.json")
    train_dataset = _dataset_from_payload(train_payload["dataset"])
    support_profile = fit_support_profile(_record_vectors(train_dataset), normalization)
    policy = FrozenRouteRankerPolicy(model, normalization, policy_id="cp10_repaired_route_ranker")
    return policy, normalization, support_profile, selected


def _pi2_pair_accuracy(dataset: RouteDataset, anchors: Mapping[str, int]) -> dict[str, Any]:
    pairs = decisive_pairs(dataset)
    scored = correct = 0
    for pair in pairs:
        record = dataset.records[pair.record_index]
        anchor = anchors[_record_identity(record)]
        if anchor not in {pair.preferred_index, pair.other_index}:
            continue
        scored += 1
        correct += int(anchor == pair.preferred_index)
    return {"pair_count": scored, "correct": correct, "accuracy": correct / scored if scored else 0.0}


def _policy_action_metrics_from_records(
    dataset: RouteDataset,
    anchors: Mapping[str, int],
    pi3: FrozenRouteRankerPolicy,
    pi4: FrozenRouteRankerPolicy,
    support_profile: SupportProfile,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, dict[str, int]] = {
        "pi2": {"states": 0, "preferred": 0},
        "pi3": {"states": 0, "preferred": 0},
        "pi4": {"states": 0, "preferred": 0},
    }
    rows: list[dict[str, Any]] = []
    for record in dataset.records:
        record_id = _record_identity(record)
        anchor = anchors[record_id]
        preferred = _preferred_set(record)
        p3_scores, p3_order = _score_indices(pi3.model, pi3.normalization, record)
        p4_scores, p4_order = _score_indices(pi4.model, pi4.normalization, record)
        p3 = p3_order[0]
        proposed = p4_order[0]
        raw = torch.tensor([item.primary_observation for item in record.candidates], dtype=torch.float32)
        norm = pi4.normalization.normalize(raw)
        support = support_profile.classify_normalized(norm[proposed])
        margin = p4_scores[proposed] - p4_scores[anchor]
        mode = str(selected["gate_mode"])
        threshold = float(selected["margin_threshold"])
        differs = proposed != anchor
        if mode == "PURE_PI2":
            accept = False
        elif mode == "PURE_ROUTE_RANKER":
            accept = differs
        elif mode == "SUPPORT_GATED":
            accept = differs and support.category == WELL_SUPPORTED
        elif mode == "MARGIN_GATED":
            accept = differs and margin >= threshold
        else:
            accept = differs and support.category == WELL_SUPPORTED and margin >= threshold
        p4 = proposed if accept else anchor
        for name, index in (("pi2", anchor), ("pi3", p3), ("pi4", p4)):
            metrics[name]["states"] += 1
            metrics[name]["preferred"] += int(index in preferred)
        rows.append(
            {
                "record_id": record_id,
                "anchor_index": anchor,
                "pi3_index": p3,
                "pi4_proposed_index": proposed,
                "pi4_executed_index": p4,
                "preferred_set": sorted(preferred),
                "pi4_deviated": p4 != anchor,
                "support": asdict(support),
                "margin": margin,
                "pi3_margin_vs_anchor": p3_scores[p3] - p3_scores[anchor],
            }
        )
    summary = {
        name: {
            **value,
            "top1_preferred_set_rate": value["preferred"] / value["states"] if value["states"] else 0.0,
        }
        for name, value in metrics.items()
    }
    return {"summary": summary, "rows": rows}


def _test_continuation_sensitive(
    definitions: Any,
    sampled_pairs: Sequence[tuple[RouteSourceDecision, CandidateKey]],
    source_types: Mapping[str, str],
    offline_rows: Mapping[str, Mapping[str, Any]],
    policies: Mapping[str, Any],
    pi3: FrozenRouteRankerPolicy,
    pi4_route: FrozenRouteRankerPolicy,
    support_profile: SupportProfile,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(
        sampled_pairs,
        key=lambda pair: _stable_hash(
            {"purpose": "cp10-test-continuation", **_source_provenance(pair[0])}
        ),
    )[:18]
    rows: list[dict[str, Any]] = []
    preferred_pi4 = preferred_pi2 = ties = incomparable = 0
    for source, anchor_key in ordered:
        source_id = _source_identity(source)
        offline = offline_rows[source_id]
        executed_index = int(offline["pi4_executed_index"])
        pi4_key = source.capture.candidates[executed_index]
        prefix = policies["pi2_cp4"] if source_types[source_id] == "pi2_source" else pi3
        continuation = ConservativeRoutePolicy(
            anchor_policy=policies["pi2_cp4"],
            route_policy=pi4_route,
            support_profile=support_profile,
            gate_mode=str(selected["gate_mode"]),
            margin_threshold=float(selected["margin_threshold"]),
            policy_id="pi4_conservative_continuation",
        )
        anchor_branch = _mixed_branch(
            definitions,
            source,
            anchor_key,
            prefix_policy=prefix,
            continuation_policy=continuation,
            continuation_name="pi4_conservative",
        )
        continuation = ConservativeRoutePolicy(
            anchor_policy=policies["pi2_cp4"],
            route_policy=pi4_route,
            support_profile=support_profile,
            gate_mode=str(selected["gate_mode"]),
            margin_threshold=float(selected["margin_threshold"]),
            policy_id="pi4_conservative_continuation",
        )
        pi4_branch = _mixed_branch(
            definitions,
            source,
            pi4_key,
            prefix_policy=prefix,
            continuation_policy=continuation,
            continuation_name="pi4_conservative",
        )
        pref = _preference_from_payload(
            anchor_branch["route_outcome"], pi4_branch["route_outcome"]
        )
        if pref == OutcomePreference.FIRST.value:
            preferred_pi2 += 1
        elif pref == OutcomePreference.SECOND.value:
            preferred_pi4 += 1
        elif pref == OutcomePreference.TIE.value:
            ties += 1
        else:
            incomparable += 1
        rows.append(
            {
                "source": _source_provenance(source),
                "source_type": source_types[source_id],
                "pi2_anchor": anchor_branch,
                "pi4_action": pi4_branch,
                "preference_under_pi4_continuation": pref,
            }
        )
    return {
        "sample_count": len(rows),
        "pi2_preferred": preferred_pi2,
        "pi4_preferred": preferred_pi4,
        "ties": ties,
        "incomparable": incomparable,
        "rows": rows,
    }


def _behavior_from_recursive(
    rows: Sequence[Mapping[str, Any]], policy: str
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["hero_policy_id"] == PRIMARY_HERO_POLICY and row["policy"] == policy
    ]
    behavior: defaultdict[str, float] = defaultdict(float)
    for row in selected:
        for encounter in row["encounters"]:
            for key, value in encounter["behavior"].items():
                if isinstance(value, (int, float)) and key != "no_progress_round_rate":
                    behavior[key] += float(value)
    cases = len(selected)
    return {
        "case_count": cases,
        "totals": dict(behavior),
        "per_case": {key: value / cases for key, value in behavior.items()} if cases else {},
    }


def _run_recursive_test(
    definitions: Any,
    policies: Mapping[str, Any],
    pi3: FrozenRouteRankerPolicy,
    pi4_route: FrozenRouteRankerPolicy,
    support_profile: SupportProfile,
    selected: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hero_seed_sets = {
        PRIMARY_HERO_POLICY: TEST_SEEDS,
        "tactical": TEST_SEEDS[:8],
        "anti_mark": TEST_SEEDS[:8],
    }
    for hero_policy, seeds in hero_seed_sets.items():
        for seed in seeds:
            for route in PRIMARY_ROUTES:
                for name in (
                    "pi0_heuristic",
                    "pi1_cp3",
                    "pi2_cp4",
                    "pi3_route",
                    "pi4_conservative",
                ):
                    conservative: ConservativeRoutePolicy | None = None
                    if name == "pi3_route":
                        policy: Any = pi3
                    elif name == "pi4_conservative":
                        conservative = ConservativeRoutePolicy(
                            anchor_policy=policies["pi2_cp4"],
                            route_policy=pi4_route,
                            support_profile=support_profile,
                            gate_mode=str(selected["gate_mode"]),
                            margin_threshold=float(selected["margin_threshold"]),
                            policy_id="pi4_conservative",
                        )
                        policy = conservative
                    else:
                        policy = policies[name]
                    result = run_corrected_stateful_route(
                        definitions,
                        route,
                        root_seed=seed,
                        enemy_policy_name=name,
                        enemy_policy=policy,
                        hero_policy_id=hero_policy,
                        safety_budget=SAFETY_BUDGET,
                    )
                    rows.append(
                        {
                            "case_id": _case_id(route.route_id, seed),
                            "hero_policy_id": hero_policy,
                            "root_seed": seed,
                            "route_id": route.route_id,
                            "policy": name,
                            "route_failed_enemy_victory": bool(
                                result["route_failed_enemy_victory"]
                            ),
                            "route_completed": bool(result["route_completed"]),
                            "route_censored": bool(result["route_censored"]),
                            "route_invalid": bool(result["route_invalid"]),
                            "route_terminal_status": result["route_terminal_status"],
                            "encounters_completed": result["encounters_completed"],
                            "final_company_state": result["final_company_state"],
                            "encounters": result["encounters"],
                            "pi4_gate_audit": []
                            if conservative is None
                            else conservative.audit,
                        }
                    )
        print(f"CP10_TEST_RECURSIVE_DONE={hero_policy}", flush=True)
    return rows


def _paired_policy(
    rows: Sequence[Mapping[str, Any]],
    first: str,
    second: str,
    hero_policy: str,
) -> dict[str, Any]:
    first_rows = {
        str(row["case_id"]): row
        for row in rows
        if row["policy"] == first and row["hero_policy_id"] == hero_policy
    }
    second_rows = {
        str(row["case_id"]): row
        for row in rows
        if row["policy"] == second and row["hero_policy_id"] == hero_policy
    }
    keys = sorted(first_rows.keys() & second_rows.keys())
    overall = paired_binary_summary(
        [bool(first_rows[key]["route_failed_enemy_victory"]) for key in keys],
        [bool(second_rows[key]["route_failed_enemy_victory"]) for key in keys],
    )
    by_route = {}
    for route in PRIMARY_ROUTES:
        route_keys = [key for key in keys if first_rows[key]["route_id"] == route.route_id]
        by_route[route.route_id] = paired_binary_summary(
            [bool(first_rows[key]["route_failed_enemy_victory"]) for key in route_keys],
            [bool(second_rows[key]["route_failed_enemy_victory"]) for key in route_keys],
        )
    return {"overall": overall, "by_route": by_route}


def _pi4_gate_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audits = [
        item
        for row in rows
        if row["hero_policy_id"] == PRIMARY_HERO_POLICY
        and row["policy"] == "pi4_conservative"
        for item in row["pi4_gate_audit"]
    ]
    deviations = [item for item in audits if item["gate_accepted"]]
    return {
        "decisions": len(audits),
        "overrides": len(deviations),
        "override_rate": len(deviations) / len(audits) if audits else 0.0,
        "override_support": dict(
            Counter(str(item["support"]["category"]) for item in deviations)
        ),
        "override_families": dict(
            Counter(
                family
                for item in deviations
                for family in item["executed_action_families"]
            )
        ),
        "all_families": dict(
            Counter(
                family for item in audits for family in item["executed_action_families"]
            )
        ),
    }


def _success_category(summary: Mapping[str, Any], override_rate: float) -> str:
    diff = float(summary["win_rate_difference"])
    low, high = (float(x) for x in summary["paired_difference_ci_95"])
    p = float(summary["mcnemar_exact_two_sided_p"])
    if diff > 0 and low > 0 and p < 0.05:
        return "CLEAR_IMPROVEMENT"
    if diff < 0 and high < 0 and p < 0.05:
        return "CLEAR_REGRESSION"
    if diff >= -0.05 and override_rate > 0:
        return "PRESERVED_BASELINE_WITH_USEFUL_ROUTE_DEVIATIONS"
    if low >= -0.10 and high <= 0.10:
        return "APPROXIMATELY_EQUIVALENT"
    return "INCONCLUSIVE"


def _failure_classification() -> dict[str, Any]:
    top1 = _read_json(TOP1_PATH)
    support = _read_json(SUPPORT_ERROR_PATH)
    continuation = _read_json(CONT_PATH)
    multi = _read_json(MULTI_PATH)
    top1_regret_rate = 1.0 - float(
        top1["groups"]["all"]["top1_preferred_set_rate"]
    )
    cont = continuation["pairwise_classification"]
    ordered_total = int(cont["CONTINUATION_STABLE"]) + int(
        cont["CONTINUATION_REVERSAL"]
    )
    reversal_rate = (
        int(cont["CONTINUATION_REVERSAL"]) / ordered_total if ordered_total else 0.0
    )
    counts = multi["win_counts"]
    support_rows = support["support_categories"]
    weak_disagreements = sum(
        int(support_rows.get(cat, {}).get("disagreements", 0))
        for cat in (EDGE_OF_SUPPORT, POORLY_SUPPORTED)
    )
    all_disagreements = sum(
        int(value.get("disagreements", 0)) for value in support_rows.values()
    )
    weak_share = weak_disagreements / all_disagreements if all_disagreements else 0.0
    multi_harm = int(counts.get("FULL_PI3", 0)) < int(
        counts.get("ONE_DEVIATION", 0)
    )
    if reversal_rate >= 0.10 and multi_harm and top1_regret_rate >= 0.20:
        classification = "COMPOUNDING_MULTI_FACTOR_FAILURE"
    elif reversal_rate >= 0.20:
        classification = "PRIMARILY_CONTINUATION_MISMATCH"
    elif top1_regret_rate >= 0.30:
        classification = "PRIMARILY_TOP1_RANKING_FAILURE"
    elif weak_share >= 0.60:
        classification = "PRIMARILY_DISTRIBUTION_SHIFT"
    else:
        classification = "INCONCLUSIVE"
    payload = {
        "classification": classification,
        "top1_regret_rate": top1_regret_rate,
        "continuation_reversal_rate_among_ordered_pairs": reversal_rate,
        "weak_support_share_of_disagreements": weak_share,
        "multi_deviation_win_counts": counts,
        "multi_deviation_harm_signal": multi_harm,
    }
    _write_json(FAILURE_PATH, payload)
    return payload


def _final_analysis(
    rows: Sequence[Mapping[str, Any]],
    paired_pi4_pi2: Mapping[str, Any],
    gate_summary: Mapping[str, Any],
) -> dict[str, Any]:
    behavior = {
        name: _behavior_from_recursive(rows, name)
        for name in (
            "pi0_heuristic",
            "pi1_cp3",
            "pi2_cp4",
            "pi3_route",
            "pi4_conservative",
        )
    }
    p2_mark = float(behavior["pi2_cp4"]["totals"].get("mark_uses", 0))
    p3_mark = float(behavior["pi3_route"]["totals"].get("mark_uses", 0))
    p4_mark = float(behavior["pi4_conservative"]["totals"].get("mark_uses", 0))
    midpoint = p2_mark + 0.5 * (p3_mark - p2_mark)
    if p4_mark <= midpoint:
        mark_class = "MARK_COLLAPSE_REPAIRED"
    elif float(paired_pi4_pi2["overall"]["win_rate_difference"]) >= 0:
        mark_class = "MARK_REMAINS_ELEVATED_BUT_BENEFICIAL"
    else:
        mark_class = "MARK_REMAINS_ELEVATED_AND_HARMFUL"
    mark_payload = {
        "classification": mark_class,
        "pi2_mark_uses": p2_mark,
        "pi3_mark_uses": p3_mark,
        "pi4_mark_uses": p4_mark,
        "repair_midpoint": midpoint,
    }
    _write_json(MARK_REPAIR_PATH, mark_payload)

    pressure_payload = {
        "pi2_bite": behavior["pi2_cp4"]["totals"].get("bite_uses", 0),
        "pi3_bite": behavior["pi3_route"]["totals"].get("bite_uses", 0),
        "pi4_bite": behavior["pi4_conservative"]["totals"].get("bite_uses", 0),
        "pi2_direct": behavior["pi2_cp4"]["totals"].get("direct_pressure_uses", 0),
        "pi3_direct": behavior["pi3_route"]["totals"].get("direct_pressure_uses", 0),
        "pi4_direct": behavior["pi4_conservative"]["totals"].get(
            "direct_pressure_uses", 0
        ),
    }
    _write_json(DIRECT_RECOVERY_PATH, pressure_payload)
    grab_payload = {
        "pi2_grab": behavior["pi2_cp4"]["totals"].get("grab_uses", 0),
        "pi3_grab": behavior["pi3_route"]["totals"].get("grab_uses", 0),
        "pi4_grab": behavior["pi4_conservative"]["totals"].get("grab_uses", 0),
        "cp8_g1_not_implemented": True,
    }
    _write_json(GRAB_DESIGN_PATH, grab_payload)

    pi4_audits = [
        item
        for row in rows
        if row["hero_policy_id"] == PRIMARY_HERO_POLICY
        and row["policy"] == "pi4_conservative"
        for item in row["pi4_gate_audit"]
    ]
    drain_legal = sum(
        any(
            "EFFORT_DRAIN" in set(families)
            for families in item["candidate_families"]
        )
        for item in pi4_audits
    )
    drain_selected = sum(
        "EFFORT_DRAIN" in set(item["executed_action_families"])
        for item in pi4_audits
    )
    drain_rate = drain_selected / drain_legal if drain_legal else 0.0
    drain_payload = {
        "legal_opportunities": drain_legal,
        "selected": drain_selected,
        "selection_rate_when_legal": drain_rate,
        "potential_strategy_collapse": drain_legal > 0 and drain_rate >= 0.95,
    }
    _write_json(DRAIN_DEGEN_PATH, drain_payload)
    shift_payload = {
        "behavior": behavior,
        "pi4_gate": gate_summary,
        "mark": mark_payload,
        "direct_pressure": pressure_payload,
        "grab": grab_payload,
        "effort_drain": drain_payload,
    }
    _write_json(SHIFT_PATH, shift_payload)
    return shift_payload


def run_test() -> None:
    selected = _read_json(SELECTED_PATH)
    if selected.get("test_accessed"):
        raise RuntimeError("CP10 TEST already accessed; refusing to rerun")
    provenance = _read_json(PROVENANCE_PATH)
    protocol = _read_json(PROTOCOL_PATH)
    if sha256_file(PROTOCOL_PATH) != selected["protocol_checksum"]:
        raise ValueError("Frozen CP10 protocol changed")
    if sha256_file(DIAGNOSIS_FREEZE_PATH) != selected["diagnosis_checksum"]:
        raise ValueError("CP10 diagnosis changed after repair selection")
    if sha256_file(VALIDATION_PATH) != selected["validation_results_checksum"]:
        raise ValueError("CP10 validation results changed")
    if _hash_paths(IMPLEMENTATION_PATHS) != protocol["implementation_hashes"]:
        raise ValueError("CP10 implementation changed after protocol freeze")
    if _hash_paths(SIMULATOR_PATHS) != protocol["simulator_hashes"]:
        raise ValueError("Simulator changed after protocol freeze")
    if _hash_paths(CONTENT_PATHS) != protocol["content_hashes"]:
        raise ValueError("Content changed after protocol freeze")

    definitions = load_game_definitions()
    policies, pi3, cp9_norm, _cp9_selected = _load_models()
    pi4_route, pi4_norm, support_profile, selected = _load_selected_pi4()
    cp9_train, _cp9_val, _cp9_test = _load_cp9_datasets()
    diagnosis_profile = fit_support_profile(_record_vectors(cp9_train), cp9_norm)

    test_pi2_pool = _collect_source_pool(
        definitions,
        seeds=TEST_SEEDS,
        source_name="pi2_cp4",
        source_policy=policies["pi2_cp4"],
        anchor_policy=policies["pi2_cp4"],
        route_policy=pi3,
        diagnosis_profile=diagnosis_profile,
        split="cp10_test_pi2_source",
    )
    test_pi3_pool = _collect_source_pool(
        definitions,
        seeds=TEST_SEEDS,
        source_name="pi3_route",
        source_policy=pi3,
        anchor_policy=policies["pi2_cp4"],
        route_policy=pi3,
        diagnosis_profile=diagnosis_profile,
        split="cp10_test_pi3_induced",
    )
    sampled_pi2 = _sample_source_pairs(
        test_pi2_pool, TEST_PI2_SAMPLE, purpose="cp10-test-pi2"
    )
    sampled_pi3 = _sample_source_pairs(
        test_pi3_pool, TEST_PI3_SAMPLE, purpose="cp10-test-pi3"
    )
    _write_json(
        ARTIFACT_ROOT / "test_source_sample.json",
        {
            "pi2": [
                {"source": _source_provenance(source), "anchor": asdict(anchor)}
                for source, anchor in sampled_pi2
            ],
            "pi3": [
                {"source": _source_provenance(source), "anchor": asdict(anchor)}
                for source, anchor in sampled_pi3
            ],
            "branch_outcomes_opened": False,
        },
    )
    pi2_records, pi2_anchors, _ = _branch_sampled_sources(
        definitions,
        sampled_pi2,
        prefix_policy=policies["pi2_cp4"],
        continuation_policy=policies["pi2_cp4"],
        continuation_name="pi2_cp4",
        split="cp10_test_pi2_source",
    )
    pi3_records, pi3_anchors, _ = _branch_sampled_sources(
        definitions,
        sampled_pi3,
        prefix_policy=pi3,
        continuation_policy=policies["pi2_cp4"],
        continuation_name="pi2_cp4",
        split="cp10_test_pi3_induced",
    )
    dataset = _merge_dataset("cp10_test", pi2_records, pi3_records)
    anchors = {**pi2_anchors, **pi3_anchors}
    source_types = {
        _source_identity(source): "pi2_source" for source, _ in sampled_pi2
    }
    source_types.update(
        {_source_identity(source): "pi3_source" for source, _ in sampled_pi3}
    )
    _write_json(
        ARTIFACT_ROOT / "test_counterfactual_dataset.json",
        {"dataset": _dataset_payload(dataset), "anchor_indices": anchors},
    )

    action_metrics = _policy_action_metrics_from_records(
        dataset, anchors, pi3, pi4_route, support_profile, selected
    )
    row_by_record = {row["record_id"]: row for row in action_metrics["rows"]}
    row_by_source: dict[str, Mapping[str, Any]] = {}
    for source, _anchor in [*sampled_pi2, *sampled_pi3]:
        matching = next(
            record
            for record in dataset.records
            if record.route_id == source.route.route_id
            and record.root_seed == source.root_seed
            and record.state_digest == source.capture.state_digest
        )
        row_by_source[_source_identity(source)] = row_by_record[
            _record_identity(matching)
        ]
    continuation_sensitive = _test_continuation_sensitive(
        definitions,
        [*sampled_pi2, *sampled_pi3],
        source_types,
        row_by_source,
        policies,
        pi3,
        pi4_route,
        support_profile,
        selected,
    )
    ranking_payload = {
        "pi2_pairwise": _pi2_pair_accuracy(dataset, anchors),
        "pi3_pairwise": evaluate_route_ranker(
            pi3.model, dataset, pi3.normalization
        ),
        "pi4_ranker_pairwise": evaluate_route_ranker(
            pi4_route.model, dataset, pi4_norm
        ),
        "top_action": action_metrics,
        "continuation_sensitive": continuation_sensitive,
    }
    _write_json(TEST_ACTION_PATH, ranking_payload)

    recursive = _run_recursive_test(
        definitions, policies, pi3, pi4_route, support_profile, selected
    )
    _write_json(RECURSIVE_PATH, recursive)
    paired = {
        "pi4_vs_pi2": {
            hero: _paired_policy(recursive, "pi2_cp4", "pi4_conservative", hero)
            for hero in (PRIMARY_HERO_POLICY, "tactical", "anti_mark")
        },
        "pi4_vs_pi3": {
            hero: _paired_policy(recursive, "pi3_route", "pi4_conservative", hero)
            for hero in (PRIMARY_HERO_POLICY, "tactical", "anti_mark")
        },
    }
    _write_json(PAIRED_PATH, paired)
    gate_summary = _pi4_gate_summary(recursive)
    primary_pi4_pi2 = paired["pi4_vs_pi2"][PRIMARY_HERO_POLICY]
    success = _success_category(
        primary_pi4_pi2["overall"], float(gate_summary["override_rate"])
    )
    _final_analysis(recursive, primary_pi4_pi2, gate_summary)
    failure = _failure_classification()
    policy_iteration = (
        "CONSERVATIVE_IMPROVEMENT_PREFERRED"
        if success
        in {
            "CLEAR_IMPROVEMENT",
            "PRESERVED_BASELINE_WITH_USEFUL_ROUTE_DEVIATIONS",
            "APPROXIMATELY_EQUIVALENT",
        }
        else "INSUFFICIENT_EVIDENCE"
    )
    _write_json(
        POLICY_ITER_PATH,
        {
            "assessment": policy_iteration,
            "cp9_greedy_replacement_failed": True,
            "cp10_result": success,
        },
    )
    confirm = (
        "CONFIRM_PI4_ON_RESERVED_RANGE"
        if success
        in {"CLEAR_IMPROVEMENT", "PRESERVED_BASELINE_WITH_USEFUL_ROUTE_DEVIATIONS"}
        else "DO_NOT_OPEN_RESERVED_RANGE"
    )
    _write_json(
        CONFIRM_PATH,
        {
            "decision": confirm,
            "reserved_range": list(RESERVED_CONFIRMATION),
            "opened_in_cp10": False,
        },
    )
    horizon = (
        "BOTH"
        if policy_iteration == "CONSERVATIVE_IMPROVEMENT_PREFERRED"
        else "MAZE_DIRECTOR"
    )
    _write_json(
        MAZE_PATH,
        {"long_horizon_intelligence": horizon, "maze_ai_implemented": False},
    )
    _write_json(
        PAIRWISE_LESSON_PATH,
        {
            "cp9_pairwise_accuracy": 0.7272727272727273,
            "cp10_top1_regret_rate": failure["top1_regret_rate"],
            "continuation_reversal_rate": failure[
                "continuation_reversal_rate_among_ordered_pairs"
            ],
            "lesson": (
                "pairwise ranking does not guarantee safe recursive top-action control "
                "under induced distribution shift"
            ),
        },
    )
    _write_json(
        NEXT_PATH,
        {
            "recommended_next_broad_direction": (
                "UNTOUCHED_PI4_CONFIRMATION"
                if confirm == "CONFIRM_PI4_ON_RESERVED_RANGE"
                else "FURTHER_DISTRIBUTION_RESEARCH"
            )
        },
    )
    selected["test_accessed"] = True
    selected["post_test_tuning_performed"] = False
    selected["test_result_classification"] = success
    _write_json(SELECTED_PATH, selected)
    provenance["test_accessed"] = True
    provenance["post_test_tuning_performed"] = False
    provenance["reserved_confirmation_accessed"] = False
    provenance["test_outputs"] = {
        "ranking_checksum": sha256_file(TEST_ACTION_PATH),
        "recursive_checksum": sha256_file(RECURSIVE_PATH),
        "paired_checksum": sha256_file(PAIRED_PATH),
        "distribution_shift_checksum": sha256_file(SHIFT_PATH),
    }
    _write_json(PROVENANCE_PATH, provenance)
    print(f"CP10_TEST_CLASSIFICATION={success}")
    print(
        "CP10_PI2_WINS="
        f"{primary_pi4_pi2['overall']['first_wins']}"
    )
    print(
        "CP10_PI4_WINS="
        f"{primary_pi4_pi2['overall']['second_wins']}"
    )
    print(f"CP10_PI4_OVERRIDE_RATE={gate_summary['override_rate']}")
    print("CP10_RESERVED_CONFIRMATION_ACCESSED=false")


def _final_verdict_payload() -> dict[str, Any]:
    paired = _read_json(PAIRED_PATH)
    failure = _read_json(FAILURE_PATH)
    shift = _read_json(SHIFT_PATH)
    top1 = _read_json(TOP1_PATH)
    support = _read_json(SUPPORT_ERROR_PATH)
    continuation = _read_json(CONT_PATH)
    multi = _read_json(MULTI_PATH)
    selected = _read_json(SELECTED_PATH)
    confirmation = _read_json(CONFIRM_PATH)
    maze = _read_json(MAZE_PATH)
    primary = paired["pi4_vs_pi2"][PRIMARY_HERO_POLICY]["overall"]
    p43 = paired["pi4_vs_pi3"][PRIMARY_HERO_POLICY]["overall"]
    gate = shift["pi4_gate"]
    pairwise_accuracy = float(top1["pairwise_accuracy"])
    top1_rate = float(top1["groups"]["all"]["top1_preferred_set_rate"])
    pairwise_gap = pairwise_accuracy - top1_rate
    if pairwise_gap >= 0.10:
        pairwise_misleading = "YES"
    elif pairwise_gap >= 0.05:
        pairwise_misleading = "PARTIALLY"
    else:
        pairwise_misleading = "NO"
    support_rows = support["support_categories"]
    weak = sum(
        int(support_rows.get(cat, {}).get("disagreements", 0))
        for cat in (EDGE_OF_SUPPORT, POORLY_SUPPORTED)
    )
    total_disagree = sum(
        int(value.get("disagreements", 0)) for value in support_rows.values()
    )
    weak_share = weak / total_disagree if total_disagree else 0.0
    support_answer = "YES" if weak_share >= 0.60 else "NO" if weak_share <= 0.40 else "MIXED"
    cont = continuation["pairwise_classification"]
    ordered = int(cont["CONTINUATION_STABLE"]) + int(cont["CONTINUATION_REVERSAL"])
    reversal = int(cont["CONTINUATION_REVERSAL"]) / ordered if ordered else 0.0
    continuation_answer = "YES" if reversal >= 0.20 else "MIXED" if reversal >= 0.05 else "NO"
    counts = multi["win_counts"]
    multi_answer = (
        "YES"
        if int(counts.get("FULL_PI3", 0)) < int(counts.get("ONE_DEVIATION", 0))
        else "MIXED"
        if len({int(v) for v in counts.values()}) > 1
        else "NO"
    )
    first = _read_json(FIRST_DIV_PATH)
    first_count = max(1, int(first["first_divergence_found"]))
    mark_share = int(first["motifs"].get("MARK_OVER_SELECTION", 0)) / first_count
    direct_share = int(first["motifs"].get("LOST_DIRECT_PRESSURE", 0)) / first_count
    mark_answer = "YES" if mark_share >= 0.25 else "PARTIAL" if mark_share >= 0.10 else "NO"
    pressure_answer = "YES" if direct_share >= 0.25 else "PARTIAL" if direct_share >= 0.10 else "NO"
    def clear_compare(summary: Mapping[str, Any]) -> str:
        diff = float(summary["win_rate_difference"])
        low, high = (float(x) for x in summary["paired_difference_ci_95"])
        p = float(summary["mcnemar_exact_two_sided_p"])
        if diff > 0 and low > 0 and p < 0.05:
            return "YES"
        if diff < 0 and high < 0 and p < 0.05:
            return "NO"
        return "INCONCLUSIVE"
    pi4_vs_pi2 = (
        "YES"
        if selected["test_result_classification"] == "CLEAR_IMPROVEMENT"
        else "NO"
        if selected["test_result_classification"] == "CLEAR_REGRESSION"
        else "APPROXIMATELY_EQUIVALENT"
        if selected["test_result_classification"]
        in {"PRESERVED_BASELINE_WITH_USEFUL_ROUTE_DEVIATIONS", "APPROXIMATELY_EQUIVALENT"}
        else "INCONCLUSIVE"
    )
    mark_repair = _read_json(MARK_REPAIR_PATH)
    pressure_recovery = _read_json(DIRECT_RECOVERY_PATH)
    grab = _read_json(GRAB_DESIGN_PATH)
    drain = _read_json(DRAIN_DEGEN_PATH)
    p2_bite = float(pressure_recovery["pi2_bite"])
    p3_bite = float(pressure_recovery["pi3_bite"])
    p4_bite = float(pressure_recovery["pi4_bite"])
    bite_mid = p3_bite + 0.5 * (p2_bite - p3_bite)
    pressure_recovered = "YES" if p4_bite >= bite_mid else "PARTIAL" if p4_bite > p3_bite else "NO"
    p2_grab = float(grab["pi2_grab"])
    p4_grab = float(grab["pi4_grab"])
    grab_answer = (
        "YES"
        if p2_grab > 0 and p4_grab >= 1.2 * p2_grab and selected["test_result_classification"] != "CLEAR_REGRESSION"
        else "NO"
        if p4_grab <= p2_grab
        else "MIXED"
    )
    return {
        "1_cp9_failure": failure["classification"],
        "2_pairwise_misleading": pairwise_misleading,
        "3_errors_near_or_outside_support": support_answer,
        "4_continuation_preferences_change": continuation_answer,
        "5_multi_deviation_joint_harm": multi_answer,
        "6_mark_major_contributor": mark_answer,
        "7_direct_pressure_disappearance_major": pressure_answer,
        "8_pi4_outperforms_pi3": clear_compare(p43),
        "9_pi4_outperforms_pi2": pi4_vs_pi2,
        "10_pi2_test_route_win_rate": float(primary["first_rate"]),
        "11_pi4_test_route_win_rate": float(primary["second_rate"]),
        "12_pi4_minus_pi2": {
            "difference": float(primary["win_rate_difference"]),
            "ci95": primary["paired_difference_ci_95"],
            "mcnemar_p": float(primary["mcnemar_exact_two_sided_p"]),
        },
        "13_pi4_override_rate": float(gate["override_rate"]),
        "14_overrides_well_supported": (
            "YES"
            if int(gate["override_support"].get(WELL_SUPPORTED, 0)) == int(gate["overrides"])
            else "NO"
            if int(gate["override_support"].get(WELL_SUPPORTED, 0)) == 0
            else "MIXED"
        ),
        "15_mark_collapse_avoided": (
            "YES" if mark_repair["classification"] == "MARK_COLLAPSE_REPAIRED" else "PARTIAL"
        ),
        "16_direct_pressure_recovered": pressure_recovered,
        "17_stronger_baseline_grab_without_g1": grab_answer,
        "18_effort_drain_strategy_collapse": (
            "YES" if drain["potential_strategy_collapse"] else "NO"
        ),
        "19_supported_policy_iteration": _read_json(POLICY_ITER_PATH)["assessment"],
        "20_reserved_confirmation_decision": confirmation["decision"],
        "21_immediate_production_ai_promotion": "NO",
        "22_immediate_production_balance_changes": "NO",
        "23_long_horizon_intelligence": maze["long_horizon_intelligence"],
        "24_recommended_next_direction": _read_json(NEXT_PATH)["recommended_next_broad_direction"],
        "diagnostic_rates": {
            "pairwise_minus_top1_gap": pairwise_gap,
            "weak_support_disagreement_share": weak_share,
            "continuation_reversal_rate": reversal,
        },
    }


def _write_research_summary() -> None:
    failure = _read_json(FAILURE_PATH)
    paired = _read_json(PAIRED_PATH)["pi4_vs_pi2"][PRIMARY_HERO_POLICY]["overall"]
    selected = _read_json(SELECTED_PATH)
    text = f"""# Charter ML research summary

- **CP3 — counterfactual policy learning:** learned candidate ranking from deterministic forced-action branches; exploratory recursive company-survival enemy wins rose from 3.41% heuristic to 27.27% learned policy.
- **CP5 — untouched confirmation:** on 1,408 untouched company-survival cases, π1 reached 23.44% and π2 25.36% enemy wins versus 4.26% for the heuristic, confirming the learned-policy advantage without claiming optimality.
- **CP7 — horizon-ordering correction:** separated research safety stops from game outcomes and showed all former primary round-20 withdrawals eventually resolved as enemy victories; corrected route ordering made π2 the strongest frozen route policy.
- **CP8 — causal perturbation:** isolated mechanic interventions with frozen policies; only the modest immediate Grab-damage variant showed a clear π2 route improvement (+4.17pp), treated as sensitivity evidence rather than a production balance instruction.
- **CP9 — offline ranking vs recursive control:** a route-aware ranker reached 72.73% held-out pairwise accuracy but collapsed recursively from π2's 84.38% to 33.33% route wins, exposing sequential distribution/continuation risk.
- **CP10 — conservative repair:** diagnosed CP9 as `{failure['classification']}` and selected `{selected['gate_mode']}` anchoring to π2. Held-out π4-conservative route wins were {100*float(paired['second_rate']):.2f}% versus π2 {100*float(paired['first_rate']):.2f}% (paired difference {100*float(paired['win_rate_difference']):+.2f}pp).

Themes: deterministic simulation, counterfactual policy learning, offline sequential decision making, distribution shift, continuation mismatch, conservative policy improvement, sealed evaluation, and reproducible experimental design. This work uses supervised/offline counterfactual ranking rather than an online reinforcement-learning algorithm.
"""
    SUMMARY_MD_PATH.write_text(text, encoding="utf-8")


def _fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _write_report() -> None:
    verdict = _final_verdict_payload()
    paired = _read_json(PAIRED_PATH)
    p42 = paired["pi4_vs_pi2"][PRIMARY_HERO_POLICY]["overall"]
    p43 = paired["pi4_vs_pi3"][PRIMARY_HERO_POLICY]["overall"]
    selected = _read_json(SELECTED_PATH)
    validation = _read_json(VALIDATION_PATH)
    top1 = _read_json(TOP1_PATH)
    first = _read_json(FIRST_DIV_PATH)
    support = _read_json(SUPPORT_ERROR_PATH)
    compound = _read_json(COMPOUND_PATH)
    continuation = _read_json(CONT_PATH)
    multi = _read_json(MULTI_PATH)
    shift = _read_json(SHIFT_PATH)
    quality = (
        _read_json(ARTIFACT_ROOT / "quality_gates.json")
        if (ARTIFACT_ROOT / "quality_gates.json").exists()
        else {"status": "PENDING_FINAL_CLOSEOUT"}
    )
    sections = [
        ("A. Provenance / CP9 preservation", f"CP9 committed baseline: `{CP9_COMMIT}`. CP9 sealed artifact integrity was reverified before CP10. Production gameplay/content remained unchanged."),
        ("B. Frozen CP10 protocol", f"Protocol SHA-256: `{sha256_file(PROTOCOL_PATH)}`. π2 is the anchor; π3 is diagnostic; exactly one final π4-conservative candidate is frozen before TEST."),
        ("C. Seed authority", "Development 70001–70032; validation 71001–71012; TEST 72001–72016. Reserved 63001–63064 remained unopened."),
        ("D. CP9 reproduction", json.dumps(_read_json(REPRO_PATH), indent=2)),
        ("E. Pairwise-vs-top1 audit", f"CP9 pairwise accuracy: {_fmt_pct(float(top1['pairwise_accuracy']))}. Held-out top-1 preferred-set rate: {_fmt_pct(float(top1['groups']['all']['top1_preferred_set_rate']))}. Pairwise-vs-top1 verdict: **{verdict['2_pairwise_misleading']}**."),
        ("F. First-divergence analysis", f"Known π2-win/π3-loss cases: {first['failure_case_count']}; first divergences found: {first['first_divergence_found']}. Motifs: `{first['motifs']}`."),
        ("G. Distribution-support analysis", json.dumps(support, indent=2)[:12000]),
        ("H. Error-compounding curve", json.dumps(compound, indent=2)),
        ("I. Continuation-policy mismatch", f"Classification counts: `{continuation['pairwise_classification']}`. Material continuation change verdict: **{verdict['4_continuation_preferences_change']}**."),
        ("J. Multi-deviation interaction", f"Win counts across bounded π3 deviations: `{multi['win_counts']}`. Joint-harm verdict: **{verdict['5_multi_deviation_joint_harm']}**."),
        ("K. Mark score-bias analysis", json.dumps(_read_json(MARK_PATH), indent=2)),
        ("L. Targeting audit", json.dumps(_read_json(TARGET_PATH), indent=2)),
        ("M. Direct-pressure disappearance", json.dumps(_read_json(PRESSURE_PATH), indent=2)[:12000]),
        ("N. Grab audit", json.dumps(_read_json(GRAB_PATH), indent=2)),
        ("O. Effort Drain audit", json.dumps(_read_json(DRAIN_PATH), indent=2)),
        ("P. Support-vs-error relationship", f"Weak-support share of π3 disagreements: {_fmt_pct(float(verdict['diagnostic_rates']['weak_support_disagreement_share']))}; verdict **{verdict['3_errors_near_or_outside_support']}**."),
        ("Q. Conservative repair design", f"π2 anchor with final gate `{selected['gate_mode']}`. Pure route ranker remained diagnostic-only. Margin threshold: `{selected['margin_threshold']}`."),
        ("R. Repaired dataset", json.dumps(_read_json(REPAIRED_SUMMARY_PATH), indent=2)[:12000]),
        ("S. Continuation-label analysis", json.dumps(_read_json(CONT_LABEL_PATH), indent=2)[:12000]),
        ("T. Model training/selection", f"Selected repaired ranker: `{selected['ranker_model_id']}`. Validation candidates: `{[(m['model_id'], m['validation']['accuracy']) for m in validation['models']]}`."),
        ("U. π4-conservative identity/checksums", f"Policy ID `{selected['policy_id']}`; model SHA `{selected['ranker_model_checksum']}`; normalization SHA `{selected['normalization_checksum']}`; support profile SHA `{selected['support_profile_checksum']}`."),
        ("V. Validation deviation budget", json.dumps(_read_json(DEVIATION_PATH), indent=2)),
        ("W. Held-out top-action performance", json.dumps(_read_json(TEST_ACTION_PATH)['top_action']['summary'], indent=2)),
        ("X. Recursive route evaluation", f"Primary company_survival: π2 {_fmt_pct(float(p42['first_rate']))} ({p42['first_wins']}/{p42['case_count']}), π4 {_fmt_pct(float(p42['second_rate']))} ({p42['second_wins']}/{p42['case_count']}), π3 {_fmt_pct(float(p43['first_rate']))} ({p43['first_wins']}/{p43['case_count']})."),
        ("Y. π4 vs π2 statistics", f"Difference {100*float(p42['win_rate_difference']):+.2f}pp; paired 95% CI [{100*float(p42['paired_difference_ci_95'][0]):+.2f}, {100*float(p42['paired_difference_ci_95'][1]):+.2f}]pp; exact McNemar p={float(p42['mcnemar_exact_two_sided_p']):.6g}. Classification: **{selected['test_result_classification']}**."),
        ("Z. π4 vs π3 statistics", f"Difference {100*float(p43['win_rate_difference']):+.2f}pp; paired 95% CI [{100*float(p43['paired_difference_ci_95'][0]):+.2f}, {100*float(p43['paired_difference_ci_95'][1]):+.2f}]pp; exact McNemar p={float(p43['mcnemar_exact_two_sided_p']):.6g}."),
        ("AA. Recursive distribution shift", json.dumps(shift['pi4_gate'], indent=2)),
        ("AB. Mark-collapse repair", json.dumps(_read_json(MARK_REPAIR_PATH), indent=2)),
        ("AC. Direct-pressure recovery", json.dumps(_read_json(DIRECT_RECOVERY_PATH), indent=2)),
        ("AD. Grab/G1 interpretation", json.dumps(_read_json(GRAB_DESIGN_PATH), indent=2)),
        ("AE. Effort-drain interpretation", json.dumps(_read_json(DRAIN_DEGEN_PATH), indent=2)),
        ("AF. CP9 failure classification", json.dumps(_read_json(FAILURE_PATH), indent=2)),
        ("AG. Pairwise-metric lesson", json.dumps(_read_json(PAIRWISE_LESSON_PATH), indent=2)),
        ("AH. Policy-iteration assessment", json.dumps(_read_json(POLICY_ITER_PATH), indent=2)),
        ("AI. Untouched-confirmation decision", json.dumps(_read_json(CONFIRM_PATH), indent=2)),
        ("AJ. Maze/director implications", json.dumps(_read_json(MAZE_PATH), indent=2)),
        ("AK. Resume/research-summary artifact", "See `research_summary.md`. It describes counterfactual/offline sequential policy research and does not mislabel this work as online reinforcement learning."),
        ("AL. Limitations", "CP10 tests one bounded repair family, uses deterministic simulator routes and frozen hero policies, retains Pareto/incomparable outcomes rather than a scalar reward, and does not establish optimality or production readiness. The untouched confirmation range remains reserved."),
        ("AM. Quality gates", json.dumps(quality, indent=2)),
        ("AN. Artifact/checksum manifest", "See `manifest.json`; every listed artifact is independently hash-verified during closeout."),
        ("AO. Final verdict", "\n".join(f"{key}: **{value}**" for key, value in verdict.items() if not key.startswith('diagnostic_'))),
    ]
    body = [
        "# THE CHARTER — ML RESEARCH PLATFORM V0.6",
        "## CHECKPOINT 10 — CONSERVATIVE ROUTE-POLICY IMPROVEMENT / DISTRIBUTION-SHIFT REPAIR",
        "",
        "**Status:** COMPLETED  ",
        f"**π4-conservative TEST classification:** `{selected['test_result_classification']}`  ",
        "**Production gameplay:** UNCHANGED  ",
        "**Reserved confirmation 63001–63064:** UNOPENED",
        "",
    ]
    for title, content in sections:
        body.extend([f"## {title}", "", content, ""])
    REPORT_PATH.write_text("\n".join(body) + "\n", encoding="utf-8")
    _write_json(ARTIFACT_ROOT / "final_verdict.json", verdict)


def _write_manifest() -> dict[str, Any]:
    artifacts: dict[str, str] = {}
    for path in sorted(ARTIFACT_ROOT.rglob("*")):
        if not path.is_file() or path == MANIFEST_PATH:
            continue
        artifacts[path.relative_to(ARTIFACT_ROOT).as_posix()] = sha256_file(path)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "protocol_checksum": sha256_file(PROTOCOL_PATH),
        "post_test_tuning_performed": bool(_read_json(PROVENANCE_PATH).get("post_test_tuning_performed", False)),
        "reserved_confirmation_accessed": bool(_read_json(PROVENANCE_PATH).get("reserved_confirmation_accessed", False)),
        "production_gameplay_changed": False,
    }
    _write_json(MANIFEST_PATH, payload)
    return payload


def _verify_manifest() -> dict[str, Any]:
    manifest = _read_json(MANIFEST_PATH)
    bad = []
    for name, expected in manifest["artifacts"].items():
        path = ARTIFACT_ROOT / name
        if not path.exists() or sha256_file(path) != expected:
            bad.append(name)
    if bad:
        raise ValueError(f"CP10 manifest mismatch: {bad}")
    return {"artifact_count": manifest["artifact_count"], "bad": bad}


def close_report() -> None:
    _write_research_summary()
    _write_report()
    manifest = _write_manifest()
    verified = _verify_manifest()
    print(f"CP10_MANIFEST_ARTIFACTS={manifest['artifact_count']}")
    print(f"CP10_MANIFEST_BAD={verified['bad']}")
    print(f"CP10_REPORT_SHA={sha256_file(REPORT_PATH)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("freeze", "diagnose", "develop", "test", "report", "verify"),
    )
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_protocol()
    elif args.command == "diagnose":
        diagnose()
    elif args.command == "develop":
        develop()
    elif args.command == "test":
        run_test()
    elif args.command == "report":
        close_report()
    else:
        result = _verify_manifest()
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
