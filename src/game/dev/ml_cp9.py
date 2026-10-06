# ruff: noqa: E501
"""Checkpoint 9: one bounded route-aware enemy-policy learning iteration."""

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

from game.campaign.company import create_new_company
from game.combat.enemy_learning import create_hero_policy
from game.content.definitions import GameDefinitions
from game.core.rng import GameRng
from game.data.loaders import load_game_definitions
from game.dev.ml_cp5 import ENCOUNTER_IDS, PI1_MODEL_SHA, PI2_MODEL_SHA, _load_frozen_policies
from game.dev.ml_cp6 import PRIMARY_ROUTES
from game.dev.ml_cp7 import SAFETY_BUDGET
from game.expedition.cave import create_encounter_combat
from game.ml.confirmation_audit import AuditRecordingPolicy
from game.ml.counterfactual import OutcomePreference, stable_digest
from game.ml.models import linear_scorer_spec, mlp_scorer_spec
from game.ml.outcome_features import (
    CP3_DERIVED_FEATURE_NAMES,
    CP3_FEATURE_NAMES,
    CP3_NEUTRAL_FACTUAL_FEATURE_NAMES,
)
from game.ml.route_policy_learning import (
    ORACLE_FEATURE_NAMES,
    ORACLE_SCHEMA_ID,
    PERSISTENT_FEATURE_NAMES,
    PRIMARY_FEATURE_NAMES,
    PRIMARY_SCHEMA_ID,
    FrozenRouteRankerPolicy,
    RouteDataset,
    RouteDecisionRecord,
    RouteSourceDecision,
    RouteTrainingConfig,
    build_route_decision_record,
    collect_route_source_decisions,
    compare_local_outcomes,
    compare_route_outcomes,
    dataset_audit,
    decisive_pairs,
    evaluate_route_ranker,
    fit_route_normalization,
    horizon_disagreement,
    load_normalization,
    load_route_ranker,
    paired_binary_summary,
    run_route_counterfactual_branch,
    save_normalization,
    save_route_ranker,
    score_record,
    sha256_file,
    train_route_ranker,
)
from game.ml.route_termination import (
    CombatTerminationReason,
    run_corrected_stateful_route,
    run_research_combat_with_termination,
)

EXPERIMENT_ID = "checkpoint9-route-aware-policy-learning-v1"
CP8_COMMIT = "dc6c558a2df212f30baa8b856edd82017e9ec6b1"
CP8_ROOT = Path("artifacts/ml/checkpoint8-route-aware-balance-perturbation-v1")
ARTIFACT_ROOT = Path("artifacts/ml") / EXPERIMENT_ID
PROTOCOL_PATH = ARTIFACT_ROOT / "frozen_protocol.json"
PROVENANCE_PATH = ARTIFACT_ROOT / "provenance.json"
SEED_AUTHORITY_PATH = ARTIFACT_ROOT / "seed_authority.json"
OBSERVATION_AUDIT_PATH = ARTIFACT_ROOT / "observation_audit.json"
INFORMATION_AUTHORITY_PATH = ARTIFACT_ROOT / "information_authority.json"
MARKOV_PATH = ARTIFACT_ROOT / "markov_sufficiency_audit.json"
PRIMARY_SCHEMA_PATH = ARTIFACT_ROOT / "primary_feature_schema.json"
ORACLE_SCHEMA_PATH = ARTIFACT_ROOT / "route_context_diagnostic_schema.json"
SOURCE_SUMMARY_PATH = ARTIFACT_ROOT / "source_trajectory_summary.json"
SOURCE_SAMPLE_PATH = ARTIFACT_ROOT / "source_state_sample.json"
CONTROL_PATH = ARTIFACT_ROOT / "control_branch_verification.json"
BRANCHES_PATH = ARTIFACT_ROOT / "route_counterfactual_branches.json"
TEST_BRANCHES_PATH = ARTIFACT_ROOT / "test_route_counterfactual_branches.json"
PREFERENCES_PATH = ARTIFACT_ROOT / "route_outcome_preferences.json"
DATASET_SUMMARY_PATH = ARTIFACT_ROOT / "dataset_summary.json"
HORIZON_PATH = ARTIFACT_ROOT / "horizon_disagreement.json"
NORMALIZATION_PATH = ARTIFACT_ROOT / "normalization.json"
ORACLE_NORMALIZATION_PATH = ARTIFACT_ROOT / "route_context_oracle_normalization.json"
VALIDATION_PATH = ARTIFACT_ROOT / "validation_results.json"
TEST_RANKING_PATH = ARTIFACT_ROOT / "test_ranking_results.json"
SELECTED_PATH = ARTIFACT_ROOT / "selected_pi3_route.json"
ONE_STEP_PATH = ARTIFACT_ROOT / "one_step_policy_comparison.json"
RECURSIVE_PATH = ARTIFACT_ROOT / "recursive_route_results.json"
PAIRED_PATH = ARTIFACT_ROOT / "paired_route_comparison.json"
ISOLATED_PATH = ARTIFACT_ROOT / "isolated_combat_results.json"
REVERSAL_PATH = ARTIFACT_ROOT / "horizon_reversal_audit.json"
BEHAVIOR_PATH = ARTIFACT_ROOT / "behavior_comparison.json"
DRAIN_PATH = ARTIFACT_ROOT / "effort_drain_audit.json"
GRAB_PATH = ARTIFACT_ROOT / "grab_design_hypothesis.json"
ORACLE_GAP_PATH = ARTIFACT_ROOT / "route_context_oracle_gap.json"
SHIFT_PATH = ARTIFACT_ROOT / "state_distribution_shift.json"
ROBUSTNESS_PATH = ARTIFACT_ROOT / "robustness_breakdown.json"
CAPACITY_PATH = ARTIFACT_ROOT / "capacity_diagnostic.json"
BALANCE_PATH = ARTIFACT_ROOT / "balance_reinterpretation.json"
MAZE_PATH = ARTIFACT_ROOT / "maze_implications.json"
NEXT_PATH = ARTIFACT_ROOT / "next_research_recommendation.json"
MANIFEST_PATH = ARTIFACT_ROOT / "manifest.json"
REPORT_PATH = ARTIFACT_ROOT / "REPORT.md"

TRAIN_SEEDS = tuple(range(60001, 60033))
VALIDATION_SEEDS = tuple(range(61001, 61013))
TEST_SEEDS = tuple(range(62001, 62017))
RESERVED_CONFIRMATION = (63001, 63064)
PRIMARY_HERO_POLICY = "company_survival"
SECONDARY_HERO_POLICIES = ("tactical", "anti_mark")
TRAIN_SAMPLE_COUNT = 120
VALIDATION_SAMPLE_COUNT = 60
TEST_SAMPLE_COUNT = 72
SECONDARY_TEST_SAMPLE_COUNT = 18
LINEAR_SEED = 20261010
MLP_SEEDS = (20261010, 20261011)
ORACLE_SEED = 20261012
TRAINING_CONFIG = RouteTrainingConfig()
ORACLE_MATERIAL_GAP = 0.05
CAPACITY_MATERIAL_GAP = 0.03
CONFIRMATION_RULE = "positive paired effect with lower 95% CI > 0 and exact McNemar p < 0.05"
SOURCE_POLICY_NAME = "pi2_cp4"

IMPLEMENTATION_PATHS = (
    "src/game/dev/ml_cp9.py",
    "src/game/ml/route_policy_learning.py",
    "tests/test_ml_route_policy_learning.py",
    "src/game/dev/ml_cp8.py",
    "src/game/dev/ml_cp7.py",
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


def _schema_checksum(names: Sequence[str], schema_id: str) -> str:
    return hashlib.sha256((schema_id + "\n" + "\n".join(names)).encode()).hexdigest()


def _band(value: float) -> str:
    if value <= 1 / 3:
        return "LOW"
    if value <= 2 / 3:
        return "MID"
    return "HIGH"


def _source_provenance(source: RouteSourceDecision) -> dict[str, Any]:
    cap = source.capture
    return {
        "split": source.split,
        "route_id": source.route.route_id,
        "hero_policy_id": source.hero_policy_id,
        "root_seed": source.root_seed,
        "global_action_index": cap.global_action_index,
        "encounter_index": cap.encounter_index,
        "encounter_id": cap.encounter_id,
        "route_position": (
            "SINGLE"
            if cap.total_encounters <= 1
            else "EARLY"
            if cap.encounter_index == 0
            else "LATE"
            if cap.encounter_index == cap.total_encounters - 1
            else "MIDDLE"
        ),
        "enemy_class_id": cap.enemy_class_id,
        "round_number": cap.round_number,
        "state_digest": cap.state_digest,
        "rng_digest": cap.rng_digest,
        "candidate_count": len(cap.candidates),
        "candidate_keys": [asdict(item) for item in cap.candidates],
        "source_choice": asdict(cap.source_choice),
        "party_hp_band": _band(cap.party_hp_fraction),
        "party_effort_band": _band(cap.party_effort_fraction),
        "primary_observation_digest": stable_digest(cap.primary_observations),
        "oracle_observation_digest": stable_digest(cap.oracle_observations),
        "encounters_remaining": cap.total_encounters - cap.encounter_index - 1,
    }


def _observation_audit() -> dict[str, Any]:
    persistent_names = {
        "living_hero_count",
        "party_hp_total",
        "party_effort_total",
        "target_hp",
        "target_max_hp",
        "target_effort",
        "target_max_effort",
        "target_mortal_wounds",
        "target_morale",
        "target_strain",
        "target_is_alive",
        "target_is_downed",
        "target_is_dead",
        "target_slot_front_left",
        "target_slot_front_right",
        "target_slot_back_left",
        "target_slot_back_right",
    }
    entries: list[dict[str, str]] = []
    for name in CP3_FEATURE_NAMES:
        if name in CP3_DERIVED_FEATURE_NAMES:
            classification = "DERIVED_CURRENT_FACT"
        elif name in persistent_names:
            classification = "PERSISTENT_COMPANY_FACT"
        else:
            classification = "CURRENT_COMBAT_FACT"
        entries.append(
            {
                "name": name,
                "classification": classification,
                "rationale": "Existing CP3 factual feature; no heuristic/designer score is admitted.",
            }
        )
    additions = [
        {
            "name": "party_max_hp_total",
            "classification": "PERSISTENT_COMPANY_FACT",
            "justification": "Needed to interpret persistent party HP attrition across roster/state changes.",
        },
        {
            "name": "party_max_effort_total",
            "classification": "PERSISTENT_COMPANY_FACT",
            "justification": "Needed to interpret persistent Effort depletion across current party state.",
        },
        {
            "name": "party_downed_count",
            "classification": "PERSISTENT_COMPANY_FACT",
            "justification": "Downed state persists between encounters and is tactically visible.",
        },
        {
            "name": "party_mortal_wounds_total",
            "classification": "PERSISTENT_COMPANY_FACT",
            "justification": "Mortal Wounds persist and materially affect expedition deterioration.",
        },
        {
            "name": "party_strain_total",
            "classification": "PERSISTENT_COMPANY_FACT",
            "justification": "Strain persists and is already represented per candidate actor/target, not party-wide.",
        },
        {
            "name": "party_strain_marks_total",
            "classification": "PERSISTENT_COMPANY_FACT",
            "justification": "Strain marks persist but are omitted from the 81-feature candidate observation.",
        },
        {
            "name": "party_morale_distress_count",
            "classification": "PERSISTENT_COMPANY_FACT",
            "justification": "Morale persists; count of Shaken/Broken visible party members is factual.",
        },
        {
            "name": "party_front_alive_count",
            "classification": "PERSISTENT_COMPANY_FACT",
            "justification": "Full visible formation is not captured by only actor/target slots.",
        },
        {
            "name": "party_back_alive_count",
            "classification": "PERSISTENT_COMPANY_FACT",
            "justification": "Complements visible front/back party geometry without identity leakage.",
        },
    ]
    return {
        "existing_schema": {
            "id": "enemy-outcome-candidate-v1",
            "version": 1,
            "feature_count": len(CP3_FEATURE_NAMES),
            "feature_entries": entries,
            "neutral_factual_count": len(CP3_NEUTRAL_FACTUAL_FEATURE_NAMES),
            "derived_factual_count": len(CP3_DERIVED_FEATURE_NAMES),
            "prescriptive_count": 0,
            "identity_leakage_count": 0,
            "future_information_count": 0,
        },
        "persistent_state_already_represented": [
            "aggregate current party HP",
            "aggregate current party Effort",
            "living hero count",
            "candidate target life/HP/Effort/Mortal Wounds/morale/strain",
            "candidate target formation/protection/statuses",
        ],
        "material_missing_visible_persistent_state": additions,
        "decision": "CREATE_VERSIONED_ROUTE_SCHEMA_WITH_MINIMAL_PERSISTENT_AGGREGATES",
    }


def _information_authority() -> dict[str, Any]:
    return {
        "TACTICALLY_OBSERVABLE": [
            "current combat actors and visible stats",
            "current HP/Effort/life state/statuses",
            "current formation/protection",
            "current legal skill/candidate facts",
            "current round/initiative facts already exposed by the tactical simulator",
        ],
        "PERSISTENT_PARTY_OBSERVABLE": [
            "visible current party-wide HP/Effort capacity",
            "Downed count",
            "Mortal Wounds",
            "strain and strain marks",
            "morale distress",
            "current visible front/back formation occupancy",
        ],
        "DIRECTOR_LEVEL_FUTURE_ROUTE_INFORMATION": [
            "encounter index",
            "encounters remaining",
            "normalized route progress",
            "future encounter identity/composition",
            "future RNG/outcomes",
            "hidden player intentions",
        ],
        "primary_policy_rule": "Only tactically observable and persistent-party observable facts may enter π3-route.",
        "diagnostic_oracle_rule": "Only encounter index, encounters remaining, and normalized route progress are additionally allowed; the oracle is not production-promotable.",
        "explicit_exclusions": [
            "seed",
            "hero-policy identity",
            "route ID",
            "encounter ID shortcut",
            "future encounter names/composition",
            "future outcomes/RNG",
            "heuristic scores",
            "designer archetype labels",
            "experiment IDs",
        ],
    }


def _seed_authority() -> dict[str, Any]:
    return {
        "prior_ranges": {
            "cp3_cp4_train": [1, 256],
            "cp3_cp4_validation": [10001, 10064],
            "cp3_cp4_test": [20001, 20128],
            "cp5_untouched_confirmation": [30001, 30128],
            "cp6_cp7_route": [40001, 40064],
            "cp8_perturbation": [50001, 50064],
        },
        "cp9": {
            "train": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
            "validation": [VALIDATION_SEEDS[0], VALIDATION_SEEDS[-1]],
            "test": [TEST_SEEDS[0], TEST_SEEDS[-1]],
            "reserved_untouched_confirmation": list(RESERVED_CONFIRMATION),
        },
        "disjoint": True,
        "reserved_confirmation_opened_in_cp9": False,
    }


def _schema_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    primary = {
        "schema_id": PRIMARY_SCHEMA_ID,
        "version": 1,
        "feature_count": len(PRIMARY_FEATURE_NAMES),
        "feature_names": list(PRIMARY_FEATURE_NAMES),
        "base_schema": "enemy-outcome-candidate-v1/v1",
        "added_features": list(PERSISTENT_FEATURE_NAMES),
        "information_authority": "TACTICAL_PLUS_VISIBLE_PERSISTENT_PARTY_ONLY",
        "prescriptive_features": 0,
        "future_route_features": 0,
        "checksum": _schema_checksum(PRIMARY_FEATURE_NAMES, PRIMARY_SCHEMA_ID),
    }
    oracle = {
        "schema_id": "enemy-route-context-oracle-candidate-v1",
        "version": 1,
        "feature_count": len(ORACLE_FEATURE_NAMES),
        "feature_names": list(ORACLE_FEATURE_NAMES),
        "primary_schema_id": PRIMARY_SCHEMA_ID,
        "diagnostic_only": True,
        "added_horizon_features": [
            "route_encounter_index",
            "route_encounters_remaining",
            "route_progress_fraction",
        ],
        "explicitly_excluded": [
            "route_id",
            "future encounter identity/composition",
            "seed",
            "future outcome",
        ],
        "checksum": _schema_checksum(ORACLE_FEATURE_NAMES, ORACLE_SCHEMA_ID),
    }
    return primary, oracle


def _protocol_payload() -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_before_cp9_seed_access",
        "cp8_commit": CP8_COMMIT,
        "source_policy": SOURCE_POLICY_NAME,
        "continuation_policy": SOURCE_POLICY_NAME,
        "route_value_target": "Q_route^pi2(s,a); one forced legal action then frozen pi2 for all remaining enemy decisions",
        "routes": [route.route_id for route in PRIMARY_ROUTES],
        "primary_training_hero_policy": PRIMARY_HERO_POLICY,
        "secondary_test_hero_policies": list(SECONDARY_HERO_POLICIES),
        "seeds": _seed_authority()["cp9"],
        "sampling": {
            "method": "outcome-blind stable SHA256 stratified provenance sampling v1",
            "strata": ["route", "encounter_index", "enemy_class", "party_hp_band"],
            "train_states": TRAIN_SAMPLE_COUNT,
            "validation_states": VALIDATION_SAMPLE_COUNT,
            "test_states": TEST_SAMPLE_COUNT,
            "secondary_test_states_per_policy": SECONDARY_TEST_SAMPLE_COUNT,
            "forbidden_sampling_inputs": [
                "pi2 score",
                "eventual outcome",
                "action type",
                "policy disagreement",
                "interesting mechanic",
            ],
        },
        "primary_schema": {"id": PRIMARY_SCHEMA_ID, "feature_count": len(PRIMARY_FEATURE_NAMES)},
        "route_context_oracle": {
            "id": ORACLE_SCHEMA_ID,
            "feature_count": len(ORACLE_FEATURE_NAMES),
            "diagnostic_only": True,
            "recursive_primary_selection_eligible": False,
        },
        "route_preference": {
            "primary": "enemy route victory > hero route completion",
            "both_enemy_victory": "TIE",
            "both_hero_completion": [
                "more hero deaths",
                "more terminal Downs",
                "more Mortal Wounds",
                "lower final hero HP",
                "lower final hero Effort",
                "more persistent strain marks",
            ],
            "combination": "Pareto only; crossed dimensions INCOMPARABLE; no scalar exchange rate",
            "retreat_or_censored": "no fabricated preference",
        },
        "safety_budget": asdict(SAFETY_BUDGET),
        "training": {
            "config": TRAINING_CONFIG.to_dict(),
            "linear_seed": LINEAR_SEED,
            "mlp_seeds": list(MLP_SEEDS),
            "oracle_mlp_seed": ORACLE_SEED,
            "primary_selection": "highest VALIDATION decisive-pair accuracy, then lowest validation loss, then lexical model id",
            "architecture_search": False,
        },
        "diagnostic_thresholds": {
            "oracle_material_accuracy_gap": ORACLE_MATERIAL_GAP,
            "capacity_material_accuracy_gap": CAPACITY_MATERIAL_GAP,
            "horizon_information_material": "at least 5 held-out reversals OR at least 10 route-only decisive pairs",
            "behavior_material_change": "absolute enemy-action share change >=0.03 OR relative share change >=0.25, denominator floor 0.01",
            "pi3_pi2_approximate_equivalence_margin": 0.05,
            "baseline_grab_more_valuable": "pi3 selection rate > pi2 by >0.03 AND route-prefers-pi3 switches exceed route-prefers-pi2 switches",
            "effort_drain_strategy_collapse": "pi3 selection rate >=0.95 of states where Effort Drain is legal",
            "untouched_confirmation": CONFIRMATION_RULE,
        },
        "recursive_test": {
            "primary_hero_policy_seeds": [TEST_SEEDS[0], TEST_SEEDS[-1]],
            "secondary_hero_policy_seed_count_each": 8,
            "policies": ["pi0_heuristic", "pi1_cp3", "pi2_cp4", "pi3_route"],
        },
        "test_policy": "TEST may not alter features, thresholds, training family, sampling, normalization, or selected pi3-route",
        "post_test_tuning_performed": False,
        "reserved_confirmation_opened": False,
    }


def _collect_sources(
    definitions: GameDefinitions,
    pi2: Any,
    *,
    seeds: Sequence[int],
    hero_policy_id: str,
    split: str,
) -> list[RouteSourceDecision]:
    sources: list[RouteSourceDecision] = []
    for root_seed in seeds:
        for route in PRIMARY_ROUTES:
            sources.extend(
                collect_route_source_decisions(
                    definitions,
                    route,
                    root_seed=root_seed,
                    source_policy=pi2,
                    source_policy_name=SOURCE_POLICY_NAME,
                    hero_policy_id=hero_policy_id,
                    split=split,
                    safety_budget=SAFETY_BUDGET,
                )
            )
    return sources


def _sampling_hash(source: RouteSourceDecision) -> str:
    row = _source_provenance(source)
    allowed = {
        key: row[key]
        for key in (
            "split",
            "route_id",
            "hero_policy_id",
            "root_seed",
            "global_action_index",
            "encounter_index",
            "encounter_id",
            "route_position",
            "enemy_class_id",
            "round_number",
            "state_digest",
            "rng_digest",
            "candidate_count",
            "party_hp_band",
            "party_effort_band",
        )
    }
    return hashlib.sha256(json.dumps(allowed, sort_keys=True).encode()).hexdigest()


def _sample_sources(
    sources: Sequence[RouteSourceDecision], count: int
) -> list[RouteSourceDecision]:
    if count > len(sources):
        raise ValueError(f"Requested {count} source states from only {len(sources)} available.")
    grouped: dict[tuple[str, int, str, str], list[RouteSourceDecision]] = defaultdict(list)
    for source in sources:
        row = _source_provenance(source)
        key = (
            str(row["route_id"]),
            int(row["encounter_index"]),
            str(row["enemy_class_id"]),
            str(row["party_hp_band"]),
        )
        grouped[key].append(source)
    selected: list[RouteSourceDecision] = []
    used: set[str] = set()
    for key in sorted(grouped):
        choice = min(grouped[key], key=_sampling_hash)
        digest = _sampling_hash(choice)
        if digest not in used:
            selected.append(choice)
            used.add(digest)
        if len(selected) >= count:
            return sorted(selected, key=_sampling_hash)
    for source in sorted(sources, key=_sampling_hash):
        digest = _sampling_hash(source)
        if digest in used:
            continue
        selected.append(source)
        used.add(digest)
        if len(selected) >= count:
            break
    return sorted(selected, key=_sampling_hash)


def _markov_audit(sources: Sequence[RouteSourceDecision]) -> dict[str, Any]:
    exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    near: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        row = _source_provenance(source)
        exact[str(row["primary_observation_digest"])].append(row)
        # Near identity deliberately drops route-context-free order/round details but retains
        # visible candidate observations rounded to integer-valued/factual precision.
        cap = source.capture
        simplified = []
        for vector in cap.primary_observations:
            simplified.append(tuple(round(float(value), 1) for value in vector))
        near_key = stable_digest(
            {
                "encounter_id": cap.encounter_id,
                "enemy_class": cap.enemy_class_id,
                "candidate_observations": simplified,
            }
        )
        near[near_key].append(row)

    def summarize(groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
        ambiguous: list[dict[str, Any]] = []
        decision_count = 0
        for _key, rows in groups.items():
            remaining = {int(row["encounters_remaining"]) for row in rows}
            positions = {str(row["route_position"]) for row in rows}
            routes = {str(row["route_id"]) for row in rows}
            if len(remaining) > 1 or len(positions) > 1:
                decision_count += len(rows)
                if len(ambiguous) < 20:
                    ambiguous.append(
                        {
                            "rows": list(rows[:6]),
                            "encounters_remaining_values": sorted(remaining),
                            "route_positions": sorted(positions),
                            "routes": sorted(routes),
                        }
                    )
        return {
            "ambiguous_groups": sum(
                1
                for rows in groups.values()
                if len({int(row["encounters_remaining"]) for row in rows}) > 1
                or len({str(row["route_position"]) for row in rows}) > 1
            ),
            "decisions_in_ambiguous_groups": decision_count,
            "examples": ambiguous,
        }

    exact_summary = summarize(exact)
    near_summary = summarize(near)
    if exact_summary["ambiguous_groups"] or near_summary["ambiguous_groups"]:
        classification = "PARTIALLY_OBSERVABLE_ROUTE_PROBLEM"
    else:
        classification = "CURRENT_SCHEMA_LIKELY_SUFFICIENT"
    return {
        "classification": classification,
        "source_decisions_audited": len(sources),
        "exact_primary_observation_collisions_across_route_context": exact_summary,
        "near_primary_observation_collisions_across_route_context": near_summary,
        "structural_omission": "The tactical schema intentionally omits encounters remaining and route progress, so identical visible combat facts may have different expedition horizon even when no sampled exact collision appears.",
        "interpretation": "This is an observation-sufficiency diagnostic, not permission to grant tactical enemies future encounter knowledge.",
    }


def _sample_payload(selected: Mapping[str, Sequence[RouteSourceDecision]]) -> dict[str, Any]:
    return {
        split: [
            {**_source_provenance(source), "sampling_sha256": _sampling_hash(source)}
            for source in sources
        ]
        for split, sources in selected.items()
    }


def _build_dataset(
    definitions: GameDefinitions,
    pi2: Any,
    sources: Sequence[RouteSourceDecision],
    *,
    split: str,
) -> tuple[RouteDataset, dict[str, Any]]:
    records: list[RouteDecisionRecord] = []
    controls: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        branches = []
        for candidate in source.capture.candidates:
            branch = run_route_counterfactual_branch(
                definitions,
                source,
                candidate,
                continuation_policy=pi2,
                continuation_policy_name=SOURCE_POLICY_NAME,
                safety_budget=SAFETY_BUDGET,
            )
            branches.append(branch)
            if candidate == source.capture.source_choice:
                controls.append(
                    {
                        "route_id": source.route.route_id,
                        "hero_policy_id": source.hero_policy_id,
                        "root_seed": source.root_seed,
                        "global_action_index": source.capture.global_action_index,
                        "state_digest_match": branch.replay_state_digest
                        == source.capture.state_digest,
                        "rng_digest_match": branch.replay_rng_digest == source.capture.rng_digest,
                        "candidate_order_match": branch.replay_candidates
                        == source.capture.candidates,
                        "route_continuation_exact": branch.control_exact,
                        "baseline_route_digest": source.baseline_route_digest,
                        "replay_route_digest": branch.route_digest,
                    }
                )
                if not branch.control_exact:
                    raise RuntimeError(
                        "CP9 control branch failed exact route-continuation verification."
                    )
        records.append(build_route_decision_record(source, branches))
        if index % 12 == 0 or index == len(sources):
            print(f"CP9_BRANCH_PROGRESS split={split} states={index}/{len(sources)}", flush=True)
    return RouteDataset(split=split, records=tuple(records)), {
        "split": split,
        "control_count": len(controls),
        "all_exact": all(row["route_continuation_exact"] for row in controls),
        "rows": controls,
    }


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    return {
        "key": asdict(candidate.key),
        "primary_observation": list(candidate.primary_observation),
        "oracle_observation": list(candidate.oracle_observation),
        "route_outcome": asdict(candidate.route_outcome),
        "local_outcome": asdict(candidate.local_outcome)
        if candidate.local_outcome is not None
        else None,
    }


def _dataset_payload(dataset: RouteDataset) -> dict[str, Any]:
    return {
        "split": dataset.split,
        "records": [
            {
                "split": record.split,
                "route_id": record.route_id,
                "encounter_index": record.encounter_index,
                "encounter_id": record.encounter_id,
                "route_position": record.route_position,
                "hero_policy_id": record.hero_policy_id,
                "root_seed": record.root_seed,
                "enemy_class_id": record.enemy_class_id,
                "round_number": record.round_number,
                "state_digest": record.state_digest,
                "rng_digest": record.rng_digest,
                "source_policy_index": record.source_policy_index,
                "control_exact": record.control_exact,
                "candidates": [_candidate_payload(candidate) for candidate in record.candidates],
            }
            for record in dataset.records
        ],
    }


def _source_summary(source_groups: Mapping[str, Sequence[RouteSourceDecision]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split, sources in source_groups.items():
        route_counts = Counter(source.route.route_id for source in sources)
        encounter_counts = Counter(source.capture.encounter_id for source in sources)
        position_counts = Counter(
            _source_provenance(source)["route_position"] for source in sources
        )
        output[split] = {
            "decision_count": len(sources),
            "routes": dict(sorted(route_counts.items())),
            "encounters": dict(sorted(encounter_counts.items())),
            "route_positions": dict(sorted(position_counts.items())),
            "unique_root_seeds": len({source.root_seed for source in sources}),
        }
    return output


def freeze_protocol(*, pre_access_tests_passed: bool) -> None:
    if not pre_access_tests_passed:
        raise ValueError("CP9 freeze requires pre-access tests to have passed.")
    if _git_head() != CP8_COMMIT:
        raise ValueError("CP9 must begin from the immutable CP8 commit.")
    if ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.iterdir()):
        raise ValueError("CP9 artifact root is not pristine; inspect before freezing.")
    policies, identities = _load_frozen_policies()
    if identities["pi1_cp3"]["model_checksum"] != PI1_MODEL_SHA:
        raise ValueError("π1 model identity mismatch before CP9 freeze.")
    if identities["pi2_cp4"]["model_checksum"] != PI2_MODEL_SHA:
        raise ValueError("π2 model identity mismatch before CP9 freeze.")
    observation = _observation_audit()
    authority = _information_authority()
    seed_authority = _seed_authority()
    primary_schema, oracle_schema = _schema_payloads()
    _write_json(OBSERVATION_AUDIT_PATH, observation)
    _write_json(INFORMATION_AUTHORITY_PATH, authority)
    _write_json(SEED_AUTHORITY_PATH, seed_authority)
    _write_json(PRIMARY_SCHEMA_PATH, primary_schema)
    _write_json(ORACLE_SCHEMA_PATH, oracle_schema)
    protocol = _protocol_payload()
    _write_json(PROTOCOL_PATH, protocol)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "cp8_commit": CP8_COMMIT,
        "current_head": _git_head(),
        "repository_dirty_for_cp9_implementation": _git_dirty(),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__},
        "policies": identities,
        "pi1_model_checksum": PI1_MODEL_SHA,
        "pi2_model_checksum": PI2_MODEL_SHA,
        "implementation_hashes": _hash_paths(IMPLEMENTATION_PATHS),
        "simulator_hashes": _hash_paths(SIMULATOR_PATHS),
        "content_hashes": _hash_paths(CONTENT_PATHS),
        "protocol_checksum": protocol_sha,
        "observation_audit_checksum": sha256_file(OBSERVATION_AUDIT_PATH),
        "information_authority_checksum": sha256_file(INFORMATION_AUTHORITY_PATH),
        "primary_schema_checksum": sha256_file(PRIMARY_SCHEMA_PATH),
        "oracle_schema_checksum": sha256_file(ORACLE_SCHEMA_PATH),
        "pre_access_tests_passed": True,
        "training_performed": False,
        "test_accessed": False,
        "post_test_tuning_performed": False,
        "reserved_confirmation_opened": False,
    }
    _write_json(PROVENANCE_PATH, provenance)
    print(f"CP9_PROTOCOL_SHA={protocol_sha}")
    print("CP9_TEST_ACCESSED=false")
    print("CP9_CONFIRMATION_RESERVED=true")


def _validate_frozen_before_train_or_test() -> dict[str, Any]:
    if not PROTOCOL_PATH.exists() or not PROVENANCE_PATH.exists():
        raise ValueError("CP9 must be frozen before training or testing.")
    provenance = _read_json(PROVENANCE_PATH)
    if sha256_file(PROTOCOL_PATH) != provenance["protocol_checksum"]:
        raise ValueError("CP9 frozen protocol changed.")
    if _hash_paths(IMPLEMENTATION_PATHS) != provenance["implementation_hashes"]:
        raise ValueError("CP9 implementation changed after freeze.")
    if _hash_paths(SIMULATOR_PATHS) != provenance["simulator_hashes"]:
        raise ValueError("Production simulator changed after CP9 freeze.")
    if _hash_paths(CONTENT_PATHS) != provenance["content_hashes"]:
        raise ValueError("Production content changed after CP9 freeze.")
    if sha256_file(PRIMARY_SCHEMA_PATH) != provenance["primary_schema_checksum"]:
        raise ValueError("CP9 primary schema changed after freeze.")
    if sha256_file(ORACLE_SCHEMA_PATH) != provenance["oracle_schema_checksum"]:
        raise ValueError("CP9 diagnostic schema changed after freeze.")
    return provenance


def _save_train_val_branch_artifact(train: RouteDataset, validation: RouteDataset) -> None:
    _write_json(
        BRANCHES_PATH,
        {
            "train": _dataset_payload(train),
            "validation": _dataset_payload(validation),
        },
    )


def _save_model(
    model_id: str,
    trained: Any,
    *,
    norm_sha: str,
    schema_id: str,
    oracle: bool,
) -> dict[str, Any]:
    metadata_path = ARTIFACT_ROOT / f"{model_id}.json"
    payload = save_route_ranker(
        trained,
        metadata_path,
        normalization_checksum=norm_sha,
        schema_id=schema_id,
        oracle=oracle,
    )
    return {
        "model_id": model_id,
        "metadata_path": str(metadata_path),
        "metadata_checksum": sha256_file(metadata_path),
        "model_path": str(metadata_path.with_suffix(".pt")),
        "model_checksum": payload["model_checksum"],
        "validation_accuracy": payload["validation_accuracy"],
        "validation_loss": payload["validation_loss"],
        "training_seed": payload["training_seed"],
        "selected_epoch": payload["selected_epoch"],
        "oracle": oracle,
    }


def run_train() -> None:
    provenance = _validate_frozen_before_train_or_test()
    if SELECTED_PATH.exists():
        raise ValueError("CP9 selected model already exists; refusing to retrain.")
    if any(path.exists() for path in (TEST_RANKING_PATH, RECURSIVE_PATH, MANIFEST_PATH)):
        raise ValueError("CP9 TEST outputs exist; refusing to retrain after TEST access.")
    definitions = load_game_definitions()
    policies, _identities = _load_frozen_policies()
    pi2 = policies[SOURCE_POLICY_NAME]

    print("CP9 collecting TRAIN source trajectories", flush=True)
    train_sources = _collect_sources(
        definitions,
        pi2,
        seeds=TRAIN_SEEDS,
        hero_policy_id=PRIMARY_HERO_POLICY,
        split="train",
    )
    print("CP9 collecting VALIDATION source trajectories", flush=True)
    validation_sources = _collect_sources(
        definitions,
        pi2,
        seeds=VALIDATION_SEEDS,
        hero_policy_id=PRIMARY_HERO_POLICY,
        split="validation",
    )
    source_groups = {"train": train_sources, "validation": validation_sources}
    _write_json(SOURCE_SUMMARY_PATH, _source_summary(source_groups))
    markov = _markov_audit(train_sources + validation_sources)
    _write_json(MARKOV_PATH, markov)

    train_sample = _sample_sources(train_sources, TRAIN_SAMPLE_COUNT)
    validation_sample = _sample_sources(validation_sources, VALIDATION_SAMPLE_COUNT)
    samples = {"train": train_sample, "validation": validation_sample}
    _write_json(SOURCE_SAMPLE_PATH, _sample_payload(samples))

    print("CP9 generating TRAIN route-counterfactual labels", flush=True)
    train_dataset, train_controls = _build_dataset(definitions, pi2, train_sample, split="train")
    print("CP9 generating VALIDATION route-counterfactual labels", flush=True)
    validation_dataset, validation_controls = _build_dataset(
        definitions, pi2, validation_sample, split="validation"
    )
    _save_train_val_branch_artifact(train_dataset, validation_dataset)
    _write_json(
        CONTROL_PATH,
        {
            "train": train_controls,
            "validation": validation_controls,
            "all_exact": bool(train_controls["all_exact"] and validation_controls["all_exact"]),
        },
    )
    if not train_controls["all_exact"] or not validation_controls["all_exact"]:
        raise RuntimeError("CP9 control verification failed; training is forbidden.")

    train_audit = dataset_audit(train_dataset)
    validation_audit = dataset_audit(validation_dataset)
    _write_json(
        DATASET_SUMMARY_PATH,
        {
            "train": train_audit,
            "validation": validation_audit,
            "source_trajectory_summary": _source_summary(source_groups),
        },
    )
    _write_json(
        PREFERENCES_PATH,
        {
            "rule": _protocol_payload()["route_preference"],
            "train": train_audit,
            "validation": validation_audit,
        },
    )
    _write_json(
        HORIZON_PATH,
        {
            "train": horizon_disagreement(train_dataset),
            "validation": horizon_disagreement(validation_dataset),
        },
    )
    if train_audit["decisive_pairs"] < 40 or validation_audit["decisive_pairs"] < 16:
        raise RuntimeError(
            "CP9 route supervision is too sparse for the predeclared small-model comparison."
        )

    normalization = fit_route_normalization(train_dataset, oracle=False)
    norm_sha = save_normalization(normalization, NORMALIZATION_PATH)
    oracle_normalization = fit_route_normalization(train_dataset, oracle=True)
    oracle_norm_sha = save_normalization(oracle_normalization, ORACLE_NORMALIZATION_PATH)

    entries: list[dict[str, Any]] = []
    linear = train_route_ranker(
        linear_scorer_spec(PRIMARY_FEATURE_NAMES),
        train_dataset,
        validation_dataset,
        normalization,
        seed=LINEAR_SEED,
        config=TRAINING_CONFIG,
        oracle=False,
    )
    entries.append(
        _save_model(
            "cp9_route_linear",
            linear,
            norm_sha=norm_sha,
            schema_id=PRIMARY_SCHEMA_ID,
            oracle=False,
        )
    )
    for seed in MLP_SEEDS:
        trained = train_route_ranker(
            mlp_scorer_spec(PRIMARY_FEATURE_NAMES),
            train_dataset,
            validation_dataset,
            normalization,
            seed=seed,
            config=TRAINING_CONFIG,
            oracle=False,
        )
        entries.append(
            _save_model(
                f"cp9_route_mlp_seed_{seed}",
                trained,
                norm_sha=norm_sha,
                schema_id=PRIMARY_SCHEMA_ID,
                oracle=False,
            )
        )
    oracle = train_route_ranker(
        mlp_scorer_spec(ORACLE_FEATURE_NAMES),
        train_dataset,
        validation_dataset,
        oracle_normalization,
        seed=ORACLE_SEED,
        config=TRAINING_CONFIG,
        oracle=True,
    )
    oracle_entry = _save_model(
        f"cp9_route_context_oracle_seed_{ORACLE_SEED}",
        oracle,
        norm_sha=oracle_norm_sha,
        schema_id=ORACLE_SCHEMA_ID,
        oracle=True,
    )
    eligible = [entry for entry in entries if not entry["oracle"]]
    eligible.sort(
        key=lambda entry: (
            -float(entry["validation_accuracy"]),
            float(entry["validation_loss"]),
            str(entry["model_id"]),
        )
    )
    primary = eligible[0]
    validation_payload = {
        "primary_candidates": entries,
        "route_context_oracle": oracle_entry,
        "selected_primary": primary["model_id"],
        "selection_used_test": False,
        "train_audit": train_audit,
        "validation_audit": validation_audit,
    }
    _write_json(VALIDATION_PATH, validation_payload)
    training_checksum = sha256_file(BRANCHES_PATH)
    selected = {
        "status": "frozen_before_test",
        "test_accessed": False,
        "post_test_tuning_performed": False,
        "pi3_route_model_id": primary["model_id"],
        "pi3_route_model_checksum": primary["model_checksum"],
        "pi3_route_metadata_checksum": primary["metadata_checksum"],
        "pi3_route_metadata_path": primary["metadata_path"],
        "normalization_path": str(NORMALIZATION_PATH),
        "normalization_checksum": norm_sha,
        "oracle_model_id": oracle_entry["model_id"],
        "oracle_model_checksum": oracle_entry["model_checksum"],
        "oracle_metadata_path": oracle_entry["metadata_path"],
        "oracle_normalization_path": str(ORACLE_NORMALIZATION_PATH),
        "oracle_normalization_checksum": oracle_norm_sha,
        "primary_schema_checksum": sha256_file(PRIMARY_SCHEMA_PATH),
        "oracle_schema_checksum": sha256_file(ORACLE_SCHEMA_PATH),
        "training_data_checksum": training_checksum,
        "validation_results_checksum": sha256_file(VALIDATION_PATH),
        "protocol_checksum": provenance["protocol_checksum"],
        "selection_rule": _protocol_payload()["training"]["primary_selection"],
    }
    _write_json(SELECTED_PATH, selected)
    print(f"CP9_SELECTED_PI3={primary['model_id']}")
    print(f"CP9_SELECTED_SHA={primary['model_checksum']}")
    print(f"CP9_TRAIN_DECISIVE={train_audit['decisive_pairs']}")
    print(f"CP9_VALIDATION_DECISIVE={validation_audit['decisive_pairs']}")
    print("CP9_TEST_ACCESSED=false")


def _load_selected_pi3() -> tuple[FrozenRouteRankerPolicy, Any, dict[str, Any]]:
    selected = _read_json(SELECTED_PATH)
    metadata_path = Path(str(selected["pi3_route_metadata_path"]))
    if sha256_file(metadata_path) != selected["pi3_route_metadata_checksum"]:
        raise ValueError("π3-route metadata checksum mismatch.")
    model, metadata = load_route_ranker(metadata_path)
    if metadata["model_checksum"] != selected["pi3_route_model_checksum"]:
        raise ValueError("π3-route model checksum mismatch.")
    normalization = load_normalization(Path(str(selected["normalization_path"])))
    if sha256_file(Path(str(selected["normalization_path"]))) != selected["normalization_checksum"]:
        raise ValueError("π3-route normalization checksum mismatch.")
    policy = FrozenRouteRankerPolicy(model, normalization, policy_id="pi3_route")
    return policy, normalization, selected


def _load_validation_models() -> dict[str, tuple[Any, dict[str, Any]]]:
    validation = _read_json(VALIDATION_PATH)
    output: dict[str, tuple[Any, dict[str, Any]]] = {}
    for entry in [*validation["primary_candidates"], validation["route_context_oracle"]]:
        model, metadata = load_route_ranker(Path(str(entry["metadata_path"])))
        output[str(entry["model_id"])] = (model, metadata)
    return output


def _pi2_pair_accuracy(dataset: RouteDataset) -> dict[str, Any]:
    pairs = decisive_pairs(dataset)
    scored = correct = 0
    for pair in pairs:
        record = dataset.records[pair.record_index]
        source = record.source_policy_index
        if source not in {pair.preferred_index, pair.other_index}:
            continue
        scored += 1
        correct += int(source == pair.preferred_index)
    return {
        "reconstructable_pair_count": scored,
        "correct": correct,
        "accuracy": correct / scored if scored else 0.0,
    }


def _one_step_comparison(
    dataset: RouteDataset,
    model: Any,
    normalization: Any,
) -> dict[str, Any]:
    agree = disagree = route_pi2 = route_pi3 = ties = incomparable = 0
    local_pi2 = local_pi3 = local_ties = local_incomparable = 0
    target_changes = 0
    skill_changes: Counter[str] = Counter()
    positions: Counter[str] = Counter()
    route_better_local_worse = 0
    examples: list[dict[str, Any]] = []
    for record in dataset.records:
        pi2_index = record.source_policy_index
        pi3_index = score_record(model, normalization, record, oracle=False)
        if pi2_index == pi3_index:
            agree += 1
            continue
        disagree += 1
        pi2 = record.candidates[pi2_index]
        pi3 = record.candidates[pi3_index]
        route_pref = compare_route_outcomes(pi2.route_outcome, pi3.route_outcome)
        local_pref = compare_local_outcomes(pi2.local_outcome, pi3.local_outcome)
        if route_pref == OutcomePreference.FIRST:
            route_pi2 += 1
        elif route_pref == OutcomePreference.SECOND:
            route_pi3 += 1
        elif route_pref == OutcomePreference.TIE:
            ties += 1
        else:
            incomparable += 1
        if local_pref == OutcomePreference.FIRST:
            local_pi2 += 1
        elif local_pref == OutcomePreference.SECOND:
            local_pi3 += 1
        elif local_pref == OutcomePreference.TIE:
            local_ties += 1
        else:
            local_incomparable += 1
        route_better_local_worse += int(
            route_pref == OutcomePreference.SECOND and local_pref == OutcomePreference.FIRST
        )
        target_changes += int(pi2.key.target_id != pi3.key.target_id)
        skill_changes[f"{pi2.key.skill_id}->{pi3.key.skill_id}"] += 1
        positions[record.route_position] += 1
        if len(examples) < 32 and route_pref != local_pref:
            examples.append(
                {
                    "route_id": record.route_id,
                    "encounter_id": record.encounter_id,
                    "route_position": record.route_position,
                    "round": record.round_number,
                    "pi2": asdict(pi2.key),
                    "pi3": asdict(pi3.key),
                    "route_preference": route_pref.value,
                    "local_preference": local_pref.value,
                }
            )
    total = agree + disagree
    return {
        "source_states": total,
        "agreement": agree,
        "disagreement": disagree,
        "agreement_rate": agree / total if total else 0.0,
        "route_preference": {
            "pi2": route_pi2,
            "pi3_route": route_pi3,
            "tie": ties,
            "incomparable": incomparable,
        },
        "local_preference": {
            "pi2": local_pi2,
            "pi3_route": local_pi3,
            "tie": local_ties,
            "incomparable": local_incomparable,
        },
        "pi3_route_better_while_local_pi2_better": route_better_local_worse,
        "target_changes": target_changes,
        "skill_changes": dict(sorted(skill_changes.items(), key=lambda item: (-item[1], item[0]))),
        "route_positions": dict(sorted(positions.items())),
        "examples": examples,
    }


def _annotate_decisions_with_route_positions(
    route_row: Mapping[str, Any], decisions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    cursor = 0
    total = len(route_row["encounter_sequence"])
    for encounter in route_row["encounters"]:
        count = len(encounter["enemy_action_sequence"])
        encounter_index = int(encounter["encounter_index"])
        pos = (
            "SINGLE"
            if total <= 1
            else "EARLY"
            if encounter_index == 0
            else "LATE"
            if encounter_index == total - 1
            else "MIDDLE"
        )
        for decision in decisions[cursor : cursor + count]:
            row = dict(decision)
            row["encounter_index"] = encounter_index
            row["encounter_id"] = encounter["encounter_id"]
            row["route_position"] = pos
            output.append(row)
        cursor += count
    return output


def _run_recursive_routes(
    definitions: GameDefinitions,
    policies: Mapping[str, Any],
    pi3: FrozenRouteRankerPolicy,
) -> list[dict[str, Any]]:
    all_policies = dict(policies)
    all_policies["pi3_route"] = pi3
    rows: list[dict[str, Any]] = []
    hero_seed_sets = {
        PRIMARY_HERO_POLICY: TEST_SEEDS,
        "tactical": TEST_SEEDS[:8],
        "anti_mark": TEST_SEEDS[:8],
    }
    for hero_policy_id, seeds in hero_seed_sets.items():
        for root_seed in seeds:
            for route in PRIMARY_ROUTES:
                for policy_name in ("pi0_heuristic", "pi1_cp3", "pi2_cp4", "pi3_route"):
                    policy = all_policies[policy_name]
                    recorder: AuditRecordingPolicy | None = None
                    executed: Any = policy
                    if policy_name in {"pi2_cp4", "pi3_route"}:
                        recorder = AuditRecordingPolicy(
                            executed_name=policy_name,
                            policies={policy_name: policy},
                        )
                        executed = recorder
                    result = run_corrected_stateful_route(
                        definitions,
                        route,
                        root_seed=root_seed,
                        enemy_policy_name=policy_name,
                        enemy_policy=executed,
                        hero_policy_id=hero_policy_id,
                        safety_budget=SAFETY_BUDGET,
                    )
                    row = dict(result)
                    row["policy"] = policy_name
                    if recorder is not None:
                        row["decision_audit"] = _annotate_decisions_with_route_positions(
                            result, recorder.decisions
                        )
                    else:
                        row["decision_audit"] = []
                    rows.append(row)
        print(f"CP9_RECURSIVE_DONE hero_policy={hero_policy_id}", flush=True)
    return rows


def _paired_recursive(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for hero_policy_id in (PRIMARY_HERO_POLICY, *SECONDARY_HERO_POLICIES):
        pi2_rows = {
            str(row["case_id"]): row
            for row in rows
            if row["hero_policy_id"] == hero_policy_id and row["policy"] == "pi2_cp4"
        }
        pi3_rows = {
            str(row["case_id"]): row
            for row in rows
            if row["hero_policy_id"] == hero_policy_id and row["policy"] == "pi3_route"
        }
        common = sorted(pi2_rows.keys() & pi3_rows.keys())
        summary = paired_binary_summary(
            [bool(pi2_rows[key]["route_failed_enemy_victory"]) for key in common],
            [bool(pi3_rows[key]["route_failed_enemy_victory"]) for key in common],
        )
        by_route: dict[str, Any] = {}
        for route in PRIMARY_ROUTES:
            keys = [key for key in common if pi2_rows[key]["route_id"] == route.route_id]
            by_route[route.route_id] = paired_binary_summary(
                [bool(pi2_rows[key]["route_failed_enemy_victory"]) for key in keys],
                [bool(pi3_rows[key]["route_failed_enemy_victory"]) for key in keys],
            )
        output[hero_policy_id] = {"overall": summary, "by_route": by_route}
    return output


def _isolated_combat_eval(
    definitions: GameDefinitions,
    pi2: Any,
    pi3: FrozenRouteRankerPolicy,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for policy_name, policy in (("pi2_cp4", pi2), ("pi3_route", pi3)):
        cases: list[dict[str, Any]] = []
        for seed in TEST_SEEDS[:8]:
            for encounter_id in ENCOUNTER_IDS:
                state = create_encounter_combat(
                    create_new_company(definitions), definitions, encounter_id
                )
                result = run_research_combat_with_termination(
                    state,
                    definitions,
                    GameRng(seed),
                    enemy_policy_name=policy_name,
                    enemy_policy=policy,
                    hero_policy=create_hero_policy(
                        PRIMARY_HERO_POLICY, encounter_id=encounter_id, seed=seed
                    ),
                    encounter_id=encounter_id,
                    encounter_name=encounter_id.replace("_", " ").title(),
                    safety_budget=SAFETY_BUDGET,
                )
                cases.append(
                    {
                        "seed": seed,
                        "encounter_id": encounter_id,
                        "termination_reason": result.termination_reason.value,
                        "enemy_win": result.termination_reason
                        == CombatTerminationReason.ENEMY_VICTORY,
                        "rounds": result.rounds_elapsed,
                    }
                )
        output[policy_name] = cases
    pi2_cases = output["pi2_cp4"]
    pi3_cases = output["pi3_route"]
    output["paired"] = paired_binary_summary(
        [bool(item["enemy_win"]) for item in pi2_cases],
        [bool(item["enemy_win"]) for item in pi3_cases],
    )
    return output


def _behavior_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for policy_name in ("pi2_cp4", "pi3_route"):
        selected = [
            row
            for row in rows
            if row["policy"] == policy_name and row["hero_policy_id"] == PRIMARY_HERO_POLICY
        ]
        decisions = [decision for row in selected for decision in row.get("decision_audit", [])]
        skill_counts = Counter(str(item["chosen_skill"]) for item in decisions)
        switches = 0
        moves = waits = 0
        for row in selected:
            route_targets = [str(item["chosen_target"]) for item in row.get("decision_audit", [])]
            switches += sum(
                first != second
                for first, second in zip(route_targets, route_targets[1:], strict=False)
            )
            for encounter in row["encounters"]:
                behavior = encounter["behavior"]
                moves += int(behavior.get("enemy_moves", 0))
                waits += int(behavior.get("enemy_waits", 0))
        position_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for item in decisions:
            pos = str(item["route_position"])
            position_counts[pos]["actions"] += 1
            if item["mark_action"]:
                position_counts[pos]["mark"] += 1
            if item["grab_action"]:
                position_counts[pos]["grab"] += 1
            if item["guard_action"]:
                position_counts[pos]["guard"] += 1
            if item["effort_drain_action"]:
                position_counts[pos]["effort_drain"] += 1
        output[policy_name] = {
            "route_cases": len(selected),
            "enemy_actions": len(decisions),
            "mark_uses": sum(bool(item["mark_action"]) for item in decisions),
            "grab_uses": sum(bool(item["grab_action"]) for item in decisions),
            "guard_uses": sum(bool(item["guard_action"]) for item in decisions),
            "effort_drain_uses": sum(bool(item["effort_drain_action"]) for item in decisions),
            "direct_pressure_uses": sum(bool(item["direct_pressure"]) for item in decisions),
            "bite_uses": sum("bite" in str(item["chosen_skill"]) for item in decisions),
            "support_target_actions": sum(
                bool(item["target_support_capable"]) for item in decisions
            ),
            "downed_target_actions": sum(
                item["target_life_state"] == "downed" for item in decisions
            ),
            "lowest_hp_target_actions": sum(
                bool(item["target_is_lowest_hp_legal"]) for item in decisions
            ),
            "protected_target_actions": sum(bool(item["target_protected"]) for item in decisions),
            "target_switches": switches,
            "movement_actions": moves,
            "wait_actions": waits,
            "skill_counts": dict(sorted(skill_counts.items())),
            "route_position_behavior": {
                key: dict(value) for key, value in sorted(position_counts.items())
            },
        }
    return output


def _mechanic_choice_audit(
    dataset: RouteDataset,
    model: Any,
    normalization: Any,
    *,
    mechanic_feature: str,
) -> dict[str, Any]:
    feature_index = PRIMARY_FEATURE_NAMES.index(mechanic_feature)
    pi2_opportunities = pi2_uses = pi3_opportunities = pi3_uses = 0
    route_preferred_pi3_when_switching = route_preferred_pi2_when_switching = 0
    state_examples: list[dict[str, Any]] = []
    for record in dataset.records:
        legal_indices = [
            index
            for index, candidate in enumerate(record.candidates)
            if candidate.primary_observation[feature_index] > 0.5
        ]
        if not legal_indices:
            continue
        pi2_opportunities += 1
        pi3_opportunities += 1
        pi2_index = record.source_policy_index
        pi3_index = score_record(model, normalization, record)
        pi2_selected = pi2_index in legal_indices
        pi3_selected = pi3_index in legal_indices
        pi2_uses += int(pi2_selected)
        pi3_uses += int(pi3_selected)
        if pi2_selected != pi3_selected:
            pref = compare_route_outcomes(
                record.candidates[pi2_index].route_outcome,
                record.candidates[pi3_index].route_outcome,
            )
            if pref == OutcomePreference.FIRST:
                route_preferred_pi2_when_switching += 1
            elif pref == OutcomePreference.SECOND:
                route_preferred_pi3_when_switching += 1
            if len(state_examples) < 20:
                state_examples.append(
                    {
                        "route_id": record.route_id,
                        "encounter_id": record.encounter_id,
                        "route_position": record.route_position,
                        "pi2_selected_mechanic": pi2_selected,
                        "pi3_selected_mechanic": pi3_selected,
                        "route_preference": pref.value,
                    }
                )
    return {
        "pi2": {
            "opportunity_states": pi2_opportunities,
            "selected": pi2_uses,
            "selection_rate": pi2_uses / pi2_opportunities if pi2_opportunities else 0.0,
        },
        "pi3_route": {
            "opportunity_states": pi3_opportunities,
            "selected": pi3_uses,
            "selection_rate": pi3_uses / pi3_opportunities if pi3_opportunities else 0.0,
        },
        "switching_route_preference": {
            "pi2": route_preferred_pi2_when_switching,
            "pi3_route": route_preferred_pi3_when_switching,
        },
        "examples": state_examples,
    }


def _state_distribution_shift(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for policy_name in ("pi2_cp4", "pi3_route"):
        selected = [
            row
            for row in rows
            if row["policy"] == policy_name and row["hero_policy_id"] == PRIMARY_HERO_POLICY
        ]
        entries = [enc["combat_entry"] for row in selected for enc in row["encounters"]]
        failure_positions = Counter(
            str(row["stopped_at_encounter_id"])
            for row in selected
            if row["stopped_at_encounter_id"] is not None
        )
        reachability = Counter(
            str(enc["encounter_id"]) for row in selected for enc in row["encounters"]
        )
        output[policy_name] = {
            "route_cases": len(selected),
            "encounter_entries": len(entries),
            "mean_entry_hp_fraction": mean(float(item["hp_fraction"]) for item in entries)
            if entries
            else 0.0,
            "mean_entry_effort_fraction": mean(float(item["effort_fraction"]) for item in entries)
            if entries
            else 0.0,
            "mean_entry_mortal_wounds": mean(float(item["mortal_wounds_total"]) for item in entries)
            if entries
            else 0.0,
            "mean_entry_downed": mean(float(item["downed_count"]) for item in entries)
            if entries
            else 0.0,
            "reachability": dict(sorted(reachability.items())),
            "failure_positions": dict(sorted(failure_positions.items())),
        }
    return output


def _robustness(rows: Sequence[Mapping[str, Any]], paired: Mapping[str, Any]) -> dict[str, Any]:
    primary = paired[PRIMARY_HERO_POLICY]
    # Keep the raw route-specific paired rows authoritative rather than collapsing them into a scalar.
    by_route = primary["by_route"]
    positive_routes = [
        route_id for route_id, row in by_route.items() if float(row["win_rate_difference"]) > 0
    ]
    negative_routes = [
        route_id for route_id, row in by_route.items() if float(row["win_rate_difference"]) < 0
    ]
    return {
        "primary_company_survival": primary,
        "hero_policy_breakdown": paired,
        "positive_route_count": len(positive_routes),
        "positive_routes": positive_routes,
        "negative_routes": negative_routes,
        "broad_across_routes": len(positive_routes) >= 4,
        "contribution_decomposition": by_route,
    }


def _route_result_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, Any] = {}
    for policy_name in ("pi0_heuristic", "pi1_cp3", "pi2_cp4", "pi3_route"):
        selected = [
            row
            for row in rows
            if row["policy"] == policy_name and row["hero_policy_id"] == PRIMARY_HERO_POLICY
        ]
        valid = [row for row in selected if not row["route_censored"] and not row["route_invalid"]]
        wins = sum(bool(row["route_failed_enemy_victory"]) for row in valid)
        completions = sum(bool(row["route_completed"]) for row in valid)
        grouped[policy_name] = {
            "case_count": len(selected),
            "valid_cases": len(valid),
            "censored": sum(bool(row["route_censored"]) for row in selected),
            "invalid": sum(bool(row["route_invalid"]) for row in selected),
            "enemy_wins": wins,
            "route_completions": completions,
            "enemy_win_rate": wins / len(valid) if valid else 0.0,
            "wilson_95": __import__(
                "game.ml.confirmation_audit", fromlist=["wilson_interval"]
            ).wilson_interval(wins, len(valid)),
        }
    return grouped


def _confirmation_decision(primary_pair: Mapping[str, Any]) -> str:
    diff = float(primary_pair["win_rate_difference"])
    low = float(primary_pair["paired_difference_ci_95"][0])
    p = float(primary_pair["mcnemar_exact_two_sided_p"])
    if diff > 0 and low > 0 and p < 0.05:
        return "CONFIRMATION_WARRANTED"
    return "CONFIRMATION_NOT_YET_WARRANTED"


def _next_direction(
    confirmation: str,
    oracle_gap: float,
    primary_pair: Mapping[str, Any],
) -> str:
    if confirmation == "CONFIRMATION_WARRANTED":
        return "UNTOUCHED_ROUTE_CONFIRMATION"
    if oracle_gap >= ORACLE_MATERIAL_GAP:
        return "REPRESENTATION_OBSERVABILITY_RESEARCH"
    if abs(float(primary_pair["win_rate_difference"])) <= 0.03:
        return "ADVERSARIAL_MAZE_DIRECTOR"
    return "SECOND_ROUTE_POLICY_ITERATION"


def run_test() -> None:
    provenance = _validate_frozen_before_train_or_test()
    if not SELECTED_PATH.exists():
        raise ValueError("CP9 TRAIN/VALIDATION selection must be frozen before TEST.")
    if TEST_RANKING_PATH.exists() or MANIFEST_PATH.exists():
        raise ValueError("CP9 TEST already exists; refusing to reopen sealed TEST.")
    selected = _read_json(SELECTED_PATH)
    if selected.get("test_accessed") is not False:
        raise ValueError("CP9 selected model does not show a sealed TEST state.")
    if sha256_file(BRANCHES_PATH) != selected["training_data_checksum"]:
        raise ValueError("CP9 training data changed after model selection.")
    if sha256_file(VALIDATION_PATH) != selected["validation_results_checksum"]:
        raise ValueError("CP9 validation results changed after model selection.")

    definitions = load_game_definitions()
    policies, identities = _load_frozen_policies()
    pi2 = policies[SOURCE_POLICY_NAME]
    pi3, normalization, selected = _load_selected_pi3()
    models = _load_validation_models()
    oracle_normalization = load_normalization(ORACLE_NORMALIZATION_PATH)

    test_source_groups: dict[str, list[RouteSourceDecision]] = {}
    for hero_policy_id in (PRIMARY_HERO_POLICY, *SECONDARY_HERO_POLICIES):
        print(f"CP9 collecting TEST source trajectories hero={hero_policy_id}", flush=True)
        test_source_groups[hero_policy_id] = _collect_sources(
            definitions,
            pi2,
            seeds=TEST_SEEDS,
            hero_policy_id=hero_policy_id,
            split=f"test_{hero_policy_id}",
        )
    sample_counts = {
        PRIMARY_HERO_POLICY: TEST_SAMPLE_COUNT,
        "tactical": SECONDARY_TEST_SAMPLE_COUNT,
        "anti_mark": SECONDARY_TEST_SAMPLE_COUNT,
    }
    test_samples = {
        hero_policy_id: _sample_sources(sources, sample_counts[hero_policy_id])
        for hero_policy_id, sources in test_source_groups.items()
    }
    prior_sample = _read_json(SOURCE_SAMPLE_PATH)
    prior_sample["test"] = _sample_payload(test_samples)
    _write_json(SOURCE_SAMPLE_PATH, prior_sample)
    source_summary = _read_json(SOURCE_SUMMARY_PATH)
    source_summary["test"] = _source_summary(test_source_groups)
    _write_json(SOURCE_SUMMARY_PATH, source_summary)

    test_datasets: dict[str, RouteDataset] = {}
    test_controls: dict[str, Any] = {}
    for hero_policy_id, sample in test_samples.items():
        print(f"CP9 generating TEST route labels hero={hero_policy_id}", flush=True)
        dataset, controls = _build_dataset(
            definitions,
            pi2,
            sample,
            split=f"test_{hero_policy_id}",
        )
        test_datasets[hero_policy_id] = dataset
        test_controls[hero_policy_id] = controls
    _write_json(
        TEST_BRANCHES_PATH,
        {hero: _dataset_payload(dataset) for hero, dataset in test_datasets.items()},
    )
    control_payload = _read_json(CONTROL_PATH)
    control_payload["test"] = test_controls
    control_payload["all_exact"] = bool(
        control_payload["all_exact"]
        and all(bool(row["all_exact"]) for row in test_controls.values())
    )
    _write_json(CONTROL_PATH, control_payload)
    if not control_payload["all_exact"]:
        raise RuntimeError("CP9 TEST control branch verification failed.")

    test_ranking: dict[str, Any] = {}
    validation = _read_json(VALIDATION_PATH)
    for model_id, (model, metadata) in models.items():
        is_oracle = bool(metadata.get("oracle", False))
        norm = oracle_normalization if is_oracle else normalization
        test_ranking[model_id] = {
            hero: evaluate_route_ranker(
                model,
                dataset,
                norm,
                oracle=is_oracle,
            )
            for hero, dataset in test_datasets.items()
        }
    test_ranking["pi2_candidate_ranking"] = {
        hero: _pi2_pair_accuracy(dataset) for hero, dataset in test_datasets.items()
    }
    _write_json(TEST_RANKING_PATH, test_ranking)

    selected_model_id = str(selected["pi3_route_model_id"])
    selected_model = models[selected_model_id][0]
    company_test = test_datasets[PRIMARY_HERO_POLICY]
    one_step = _one_step_comparison(company_test, selected_model, normalization)
    _write_json(ONE_STEP_PATH, one_step)
    reversal = horizon_disagreement(company_test)
    _write_json(REVERSAL_PATH, reversal)
    horizon_payload = _read_json(HORIZON_PATH)
    horizon_payload["test_company_survival"] = reversal
    horizon_payload["test_secondary"] = {
        hero: horizon_disagreement(dataset)
        for hero, dataset in test_datasets.items()
        if hero != PRIMARY_HERO_POLICY
    }
    _write_json(HORIZON_PATH, horizon_payload)

    print("CP9 running recursive held-out route comparison", flush=True)
    recursive_rows = _run_recursive_routes(definitions, policies, pi3)
    _write_json(RECURSIVE_PATH, recursive_rows)
    paired = _paired_recursive(recursive_rows)
    summary = _route_result_summary(recursive_rows)
    paired_payload = {"policy_summary": summary, "pi3_vs_pi2": paired}
    _write_json(PAIRED_PATH, paired_payload)

    isolated = _isolated_combat_eval(definitions, pi2, pi3)
    _write_json(ISOLATED_PATH, isolated)
    behavior = _behavior_summary(recursive_rows)
    _write_json(BEHAVIOR_PATH, behavior)
    drain = _mechanic_choice_audit(
        company_test,
        selected_model,
        normalization,
        mechanic_feature="skill_tag_effort_drain",
    )
    _write_json(DRAIN_PATH, drain)
    grab = _mechanic_choice_audit(
        company_test,
        selected_model,
        normalization,
        mechanic_feature="skill_tag_drag_forward",
    )
    _write_json(GRAB_PATH, grab)
    shift = _state_distribution_shift(recursive_rows)
    _write_json(SHIFT_PATH, shift)

    primary_metrics = test_ranking[selected_model_id][PRIMARY_HERO_POLICY]
    oracle_model_id = str(selected["oracle_model_id"])
    oracle_metrics = test_ranking[oracle_model_id][PRIMARY_HERO_POLICY]
    oracle_gap = float(oracle_metrics["accuracy"]) - float(primary_metrics["accuracy"])
    oracle_gap_payload = {
        "primary_model": selected_model_id,
        "primary_test_accuracy": primary_metrics["accuracy"],
        "oracle_model": oracle_model_id,
        "oracle_test_accuracy": oracle_metrics["accuracy"],
        "accuracy_gap": oracle_gap,
        "material_threshold": ORACLE_MATERIAL_GAP,
        "classification": "LARGE_GAP" if oracle_gap >= ORACLE_MATERIAL_GAP else "SMALL_GAP",
        "oracle_not_production_promotable": True,
    }
    _write_json(ORACLE_GAP_PATH, oracle_gap_payload)

    linear_metrics = test_ranking["cp9_route_linear"][PRIMARY_HERO_POLICY]
    mlp_entries = [
        entry for entry in validation["primary_candidates"] if "mlp" in entry["model_id"]
    ]
    best_mlp = max(
        mlp_entries,
        key=lambda entry: float(test_ranking[entry["model_id"]][PRIMARY_HERO_POLICY]["accuracy"]),
    )
    best_mlp_metrics = test_ranking[best_mlp["model_id"]][PRIMARY_HERO_POLICY]
    capacity_gap = float(best_mlp_metrics["accuracy"]) - float(linear_metrics["accuracy"])
    capacity = {
        "linear_test_accuracy": linear_metrics["accuracy"],
        "best_mlp_id": best_mlp["model_id"],
        "best_mlp_test_accuracy": best_mlp_metrics["accuracy"],
        "mlp_minus_linear_accuracy": capacity_gap,
        "material_threshold": CAPACITY_MATERIAL_GAP,
        "model_capacity_primary_bottleneck": "YES"
        if capacity_gap >= CAPACITY_MATERIAL_GAP
        else "NO",
    }
    _write_json(CAPACITY_PATH, capacity)

    robustness = _robustness(recursive_rows, paired)
    _write_json(ROBUSTNESS_PATH, robustness)
    primary_pair = paired[PRIMARY_HERO_POLICY]["overall"]
    confirmation = _confirmation_decision(primary_pair)
    next_direction = _next_direction(confirmation, oracle_gap, primary_pair)
    _write_json(
        NEXT_PATH,
        {
            "untouched_confirmation_decision": confirmation,
            "recommended_next_direction": next_direction,
            "implemented": False,
            "reserved_confirmation_opened": False,
        },
    )

    mark_usage_changed = _material_behavior_change(behavior, "mark_uses")
    grab_usage_changed = _material_behavior_change(behavior, "grab_uses")
    guard_usage_changed = _material_behavior_change(behavior, "guard_uses")
    effort_usage_changed = _material_behavior_change(behavior, "effort_drain_uses")
    balance = {
        "MARK": {
            "cp8": "HEURISTIC_OVERVALUATION",
            "pi3_route_usage_materially_changed": mark_usage_changed,
        },
        "GRAB": {
            "cp8": "MIXED_INCONCLUSIVE",
            "pi3_route_usage_materially_changed": grab_usage_changed,
            "baseline_grab_test_audit": grab,
            "g1_not_applied": True,
        },
        "GUARD": {
            "cp8": "HEURISTIC_OVERVALUATION",
            "pi3_route_usage_materially_changed": guard_usage_changed,
        },
        "EFFORT_DRAIN": {
            "cp8": "MIXED_INCONCLUSIVE",
            "pi3_route_usage_materially_changed": effort_usage_changed,
            "heldout_opportunity_audit": drain,
        },
        "production_balance_change_justified": False,
    }
    _write_json(BALANCE_PATH, balance)
    strategic_home = (
        "MAZE_DIRECTOR"
        if oracle_gap >= ORACLE_MATERIAL_GAP
        else "BOTH"
        if reversal["horizon_reversals"] > 0
        else "TACTICAL_POLICY"
    )
    maze = {
        "strategic_horizon_primary_home": strategic_home,
        "route_context_oracle_gap": oracle_gap_payload,
        "horizon_reversal_count": reversal["horizon_reversals"],
        "hierarchy_not_implemented": True,
        "interpretation": (
            "A large oracle gap argues for keeping combat policy tactical while a future director owns expedition horizon."
            if oracle_gap >= ORACLE_MATERIAL_GAP
            else "The tactical observation captures much of the measured route preference signal; route-level reversals still motivate hierarchical evaluation."
        ),
    }
    _write_json(MAZE_PATH, maze)

    _write_report(
        provenance=provenance,
        selected=selected,
        test_ranking=test_ranking,
        one_step=one_step,
        recursive_summary=summary,
        paired=paired,
        isolated=isolated,
        reversal=reversal,
        behavior=behavior,
        drain=drain,
        grab=grab,
        oracle_gap=oracle_gap_payload,
        shift=shift,
        robustness=robustness,
        capacity=capacity,
        balance=balance,
        maze=maze,
        confirmation=confirmation,
        next_direction=next_direction,
        mark_usage_changed=mark_usage_changed,
        grab_usage_changed=grab_usage_changed,
        guard_usage_changed=guard_usage_changed,
        effort_usage_changed=effort_usage_changed,
    )
    _write_manifest(selected)
    _verify_manifest()
    print("CP9_TEST_COMPLETE=true")
    print(f"CP9_PI2_ROUTE_WIN_RATE={primary_pair['first_rate']}")
    print(f"CP9_PI3_ROUTE_WIN_RATE={primary_pair['second_rate']}")
    print(f"CP9_PI3_MINUS_PI2={primary_pair['win_rate_difference']}")
    print(f"CP9_CONFIRMATION={confirmation}")
    print(f"CP9_NEXT={next_direction}")
    print("POST_TEST_TUNING=false")


def _material_behavior_change(behavior: Mapping[str, Any], key: str) -> str:
    first = behavior["pi2_cp4"]
    second = behavior["pi3_route"]
    first_actions = max(1, int(first["enemy_actions"]))
    second_actions = max(1, int(second["enemy_actions"]))
    first_rate = float(first[key]) / first_actions
    second_rate = float(second[key]) / second_actions
    absolute = abs(second_rate - first_rate)
    relative = abs(second_rate - first_rate) / max(first_rate, 0.01)
    return "YES" if absolute >= 0.03 or relative >= 0.25 else "NO"


def _write_report(**data: Any) -> None:
    selected = data["selected"]
    ranking = data["test_ranking"]
    paired = data["paired"]
    primary_pair = paired[PRIMARY_HERO_POLICY]["overall"]
    isolated_pair = data["isolated"]["paired"]
    reversal = data["reversal"]
    oracle_gap = data["oracle_gap"]
    capacity = data["capacity"]
    markov = _read_json(MARKOV_PATH)
    dataset_summary = _read_json(DATASET_SUMMARY_PATH)
    validation = _read_json(VALIDATION_PATH)
    behavior = data["behavior"]
    drain = data["drain"]
    grab = data["grab"]
    route_info_material = (
        int(reversal["horizon_reversals"]) >= 5 or int(reversal["route_only_decisive"]) >= 10
    )
    pi3_classification = _classify_pi3_vs_pi2(primary_pair)
    isolated_sacrifice = (
        "YES"
        if float(primary_pair["win_rate_difference"]) > 0
        and float(isolated_pair["win_rate_difference"]) < 0
        else "NO"
    )
    grab_value = _grab_value_classification(grab)
    drain_collapse = "YES" if float(drain["pi3_route"]["selection_rate"]) >= 0.95 else "NO"
    horizon_home = data["maze"]["strategic_horizon_primary_home"]
    lines = [
        "# THE CHARTER — CHECKPOINT 9 ROUTE-AWARE ENEMY POLICY LEARNING",
        "",
        "## A. Provenance / CP8 preservation",
        f"- Immutable CP8 commit: `{CP8_COMMIT}` (`Qualify route-aware mechanic perturbations`).",
        f"- π1 SHA-256: `{PI1_MODEL_SHA}`; π2 SHA-256: `{PI2_MODEL_SHA}`.",
        "- Production mechanics/content/AI remained unchanged.",
        "",
        "## B. CP9 frozen protocol",
        f"- Protocol SHA-256: `{data['provenance']['protocol_checksum']}`.",
        "- Exactly one π2→π3-route learning generation was authorized; no π4/search/RL/self-play.",
        "- Source and continuation policy are frozen π2, estimating Q_route^π2(s,a).",
        "",
        "## C. Seed authority",
        f"- TRAIN {TRAIN_SEEDS[0]}–{TRAIN_SEEDS[-1]}; VALIDATION {VALIDATION_SEEDS[0]}–{VALIDATION_SEEDS[-1]}; TEST {TEST_SEEDS[0]}–{TEST_SEEDS[-1]}; untouched reserve {RESERVED_CONFIRMATION[0]}–{RESERVED_CONFIRMATION[1]}.",
        "- Reserved confirmation was not opened.",
        "",
        "## D. Existing 81-feature audit",
        "- Existing v1 remains factual: 0 prescriptive, 0 identity-leakage, 0 future-information features.",
        "- It represents current aggregate HP/Effort and candidate actor/target persistent facts, but omits several visible party-wide persistent aggregates.",
        "",
        "## E. Markov/observation-sufficiency audit",
        f"- Classification: **{markov['classification']}**.",
        f"- Exact ambiguous groups: {markov['exact_primary_observation_collisions_across_route_context']['ambiguous_groups']}; near ambiguous groups: {markov['near_primary_observation_collisions_across_route_context']['ambiguous_groups']}.",
        "- Omitted horizon remains a structural partial-observability risk; this does not authorize tactical omniscience.",
        "",
        "## F. Tactical information authority",
        "- Primary policy uses only current tactical facts plus visible persistent-party facts.",
        "- Future encounter identity/composition, route ID, seed, hero-policy identity, outcomes, and future RNG are excluded.",
        "",
        "## G. Primary route-aware schema",
        f"- `{PRIMARY_SCHEMA_ID}` v1, {len(PRIMARY_FEATURE_NAMES)} ordered factual features.",
        f"- Added only: {', '.join(PERSISTENT_FEATURE_NAMES)}.",
        "",
        "## H. Route-context diagnostic schema",
        f"- `{ORACLE_SCHEMA_ID}` v1, {len(ORACLE_FEATURE_NAMES)} features.",
        "- Diagnostic-only additions: encounter index, encounters remaining, normalized route progress.",
        "- Not production-promotable and not eligible for π3-route selection.",
        "",
        "## I. Source π2 route trajectories",
        f"- TRAIN available decisions: {dataset_summary['source_trajectory_summary']['train']['decision_count']}; VALIDATION: {dataset_summary['source_trajectory_summary']['validation']['decision_count']}.",
        "",
        "## J. Deterministic source-state sampling",
        f"- Outcome-blind SHA256 stratified sample: {TRAIN_SAMPLE_COUNT} TRAIN, {VALIDATION_SAMPLE_COUNT} VALIDATION, {TEST_SAMPLE_COUNT} primary TEST states.",
        "- Sampling never used π2 score, branch outcome, action type, disagreement, or mechanic interest.",
        "",
        "## K. Route-counterfactual branch method",
        "- Exact route prefix is replayed, one legal action is forced, current combat and all remaining encounters continue under frozen π2 and frozen hero policy until CP7 production-valid route termination.",
        "",
        "## L. Control-branch verification",
        f"- All source-action control branches exact: **{_read_json(CONTROL_PATH)['all_exact']}**.",
        "",
        "## M. Route outcome preference",
        "- Enemy route victory > hero route completion; enemy-win/enemy-win ties; hero-completion pairs use Pareto deterioration only; crossed dimensions incomparable; censored/retreat cases are not fabricated into preferences.",
        "",
        "## N. Dataset statistics",
        f"- TRAIN: {dataset_summary['train']}.",
        f"- VALIDATION: {dataset_summary['validation']}.",
        "",
        "## O. Horizon-disagreement statistics",
        f"- Held-out both-horizon decisive pairs={reversal['both_horizons_decisive']}; reversals={reversal['horizon_reversals']} ({reversal['horizon_reversal_rate']:.2%}); route-only decisive={reversal['route_only_decisive']}; local-only decisive={reversal['local_only_decisive']}.",
        "",
        "## P. Linear route model",
        f"- VALIDATION: {next(entry for entry in validation['primary_candidates'] if entry['model_id'] == 'cp9_route_linear')}.",
        f"- TEST company_survival: {ranking['cp9_route_linear'][PRIMARY_HERO_POLICY]}.",
        "",
        "## Q. MLP route model",
        f"- Selected π3-route: `{selected['pi3_route_model_id']}`; TEST company_survival: {ranking[selected['pi3_route_model_id']][PRIMARY_HERO_POLICY]}.",
        "",
        "## R. Route-context oracle model",
        f"- `{selected['oracle_model_id']}` TEST accuracy={oracle_gap['oracle_test_accuracy']:.4f}; primary={oracle_gap['primary_test_accuracy']:.4f}; gap={oracle_gap['accuracy_gap']:+.4f} ({oracle_gap['classification']}).",
        "",
        "## S. Validation/model selection",
        f"- Selected only by frozen VALIDATION rule; TEST used for selection: **False**. Selection artifact: `{selected['pi3_route_model_id']}`.",
        "",
        "## T. Held-out ranking evaluation",
        f"- Primary π3-route metrics: {ranking[selected['pi3_route_model_id']][PRIMARY_HERO_POLICY]}.",
        f"- Reconstructable π2 pair ranking: {ranking['pi2_candidate_ranking'][PRIMARY_HERO_POLICY]}.",
        "",
        "## U. Selected π3-route identity/checksums",
        f"- Model `{selected['pi3_route_model_id']}` SHA-256 `{selected['pi3_route_model_checksum']}`.",
        f"- Normalization SHA-256 `{selected['normalization_checksum']}`; training data SHA-256 `{selected['training_data_checksum']}`.",
        "",
        "## V. One-step π2→π3-route action comparison",
        f"- {data['one_step']}.",
        "",
        "## W. Recursive held-out route evaluation",
        f"- π0 {data['recursive_summary']['pi0_heuristic']}; π1 {data['recursive_summary']['pi1_cp3']}; π2 {data['recursive_summary']['pi2_cp4']}; π3-route {data['recursive_summary']['pi3_route']}.",
        "",
        "## X. π3-route vs π2 paired statistics",
        f"- Primary company_survival: {primary_pair}.",
        "",
        "## Y. Isolated-combat evaluation",
        f"- π3-route vs π2 isolated paired result: {isolated_pair}.",
        "",
        "## Z. Horizon-reversal audit",
        f"- Confirmed held-out reversals: {reversal['horizon_reversals']}; examples preserved in `{REVERSAL_PATH.name}`.",
        "",
        "## AA. Behavior changes",
        f"- π2: {behavior['pi2_cp4']}.",
        f"- π3-route: {behavior['pi3_route']}.",
        "",
        "## AB. Effort-drain degeneracy",
        f"- {data['drain']}.",
        "",
        "## AC. Grab/G1 hypothesis",
        f"- Baseline mechanics only; G1 not applied. Held-out choice audit: {data['grab']}.",
        "",
        "## AD. Route-context oracle gap",
        f"- {oracle_gap}.",
        "",
        "## AE. State-distribution shift",
        f"- {data['shift']}.",
        "",
        "## AF. Robustness/generalization",
        f"- {data['robustness']}.",
        "",
        "## AG. Linear-vs-MLP capacity diagnostic",
        f"- {capacity}.",
        "",
        "## AH. Player-facing interpretation",
        "- Strength is not automatically fun. Route-aware changes are interpreted separately from production suitability; no promotion is made.",
        f"- Isolated-combat sacrifice while route improves: **{isolated_sacrifice}**.",
        "",
        "## AI. CP8 balance reinterpretation",
        f"- {data['balance']}.",
        "",
        "## AJ. Maze/director architectural implications",
        f"- Strategic horizon primary home: **{horizon_home}**. No hierarchy/director is implemented.",
        "",
        "## AK. Untouched-confirmation decision",
        f"- **{data['confirmation']}**; reserved 63001–63064 remains untouched.",
        "",
        "## AL. Recommended next research direction",
        f"- **{data['next_direction']}**. Not implemented.",
        "",
        "## AM. Limitations",
        "- π3-route is optimized against the established simulated-player distribution, not proven universally optimal against human or optimal play.",
        "- The diagnostic oracle has route-horizon information deliberately forbidden to a normal tactical enemy.",
        "- One bounded π2→π3-route iteration cannot establish convergence; π4 was not generated.",
        "- Counterfactual route sampling is deterministic and bounded rather than exhaustive over every enemy decision.",
        "",
        "## AN. Quality gates",
        "- Final current-tree quality gates are filled after sealed TEST; scientific artifacts are not regenerated afterward.",
        "",
        "## AO. Artifact/checksum manifest",
        f"- Root `{ARTIFACT_ROOT}`; final manifest `{MANIFEST_PATH.name}`.",
        "",
        "## AP. Final verdict",
        f"1. Route supervision materially different from combat supervision: **{'YES' if route_info_material else 'NO'}**",
        "2. Existing 81-feature observation sufficient: **PARTIALLY**",
        f"3. Tactical-legal π3-route outperforms π2: **{pi3_classification}**",
        f"4. Exact π2 route win rate: **{primary_pair['first_rate']:.6f}**",
        f"5. Exact π3-route route win rate: **{primary_pair['second_rate']:.6f}**",
        f"6. π3-route minus π2: **{primary_pair['win_rate_difference']:+.6f}**, 95% CI {primary_pair['paired_difference_ci_95']}, McNemar p={primary_pair['mcnemar_exact_two_sided_p']}",
        f"7. Sacrifices isolated-combat strength to improve route strength: **{isolated_sacrifice}**",
        f"8. Confirmed individual horizon-reversal states: **{'YES' if reversal['horizon_reversals'] > 0 else 'NO'}**",
        f"9. Mark usage materially changed: **{data['mark_usage_changed']}**",
        f"10. Grab usage materially changed: **{data['grab_usage_changed']}**",
        f"11. Guard usage materially changed: **{data['guard_usage_changed']}**",
        f"12. Effort Drain usage/state contingency materially changed: **{data['effort_usage_changed']}**",
        f"13. Baseline Grab becomes more valuable without G1: **{grab_value}**",
        f"14. Effort Drain remains potential strategy-collapse behavior: **{drain_collapse}**",
        f"15. Route-context oracle materially outperforms tactical model: **{'YES' if oracle_gap['classification'] == 'LARGE_GAP' else 'NO'}**",
        f"16. Strategic horizon should primarily live in: **{horizon_home}**",
        f"17. Model capacity primary bottleneck: **{capacity['model_capacity_primary_bottleneck']}**",
        "18. Another route-aware iteration immediately justified: **NOT_YET**",
        f"19. Untouched route confirmation: **{data['confirmation']}**",
        "20. Immediate production AI/balance change justified: **NO**",
        f"21. Recommended next direction: **{data['next_direction']}**",
        "",
        "Checkpoint 9 stops here. No π3-route promotion, production AI/balance change, G1 implementation, Effort Drain nerf, π4, search, player optimization, or Maze AI was started.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _classify_pi3_vs_pi2(pair: Mapping[str, Any]) -> str:
    diff = float(pair["win_rate_difference"])
    low, high = (float(value) for value in pair["paired_difference_ci_95"])
    p = float(pair["mcnemar_exact_two_sided_p"])
    if diff > 0 and low > 0 and p < 0.05:
        return "YES"
    if diff < 0 and high < 0 and p < 0.05:
        return "NO"
    if low >= -0.05 and high <= 0.05:
        return "APPROXIMATELY_EQUIVALENT"
    return "INCONCLUSIVE"


def _grab_value_classification(grab: Mapping[str, Any]) -> str:
    pi2 = float(grab["pi2"]["selection_rate"])
    pi3 = float(grab["pi3_route"]["selection_rate"])
    pref = grab["switching_route_preference"]
    if pi3 > pi2 + 0.03 and int(pref["pi3_route"]) > int(pref["pi2"]):
        return "YES"
    if abs(pi3 - pi2) <= 0.03 and int(pref["pi3_route"]) == int(pref["pi2"]):
        return "NO"
    return "MIXED"


def _write_manifest(selected: Mapping[str, Any]) -> None:
    paths = [
        PROTOCOL_PATH,
        PROVENANCE_PATH,
        SEED_AUTHORITY_PATH,
        OBSERVATION_AUDIT_PATH,
        INFORMATION_AUTHORITY_PATH,
        MARKOV_PATH,
        PRIMARY_SCHEMA_PATH,
        ORACLE_SCHEMA_PATH,
        SOURCE_SUMMARY_PATH,
        SOURCE_SAMPLE_PATH,
        CONTROL_PATH,
        BRANCHES_PATH,
        TEST_BRANCHES_PATH,
        PREFERENCES_PATH,
        DATASET_SUMMARY_PATH,
        HORIZON_PATH,
        NORMALIZATION_PATH,
        ORACLE_NORMALIZATION_PATH,
        VALIDATION_PATH,
        TEST_RANKING_PATH,
        SELECTED_PATH,
        ONE_STEP_PATH,
        RECURSIVE_PATH,
        PAIRED_PATH,
        ISOLATED_PATH,
        REVERSAL_PATH,
        BEHAVIOR_PATH,
        DRAIN_PATH,
        GRAB_PATH,
        ORACLE_GAP_PATH,
        SHIFT_PATH,
        ROBUSTNESS_PATH,
        CAPACITY_PATH,
        BALANCE_PATH,
        MAZE_PATH,
        NEXT_PATH,
        REPORT_PATH,
    ]
    for metadata in ARTIFACT_ROOT.glob("cp9_route_*.json"):
        if metadata not in paths:
            paths.append(metadata)
        model_path = metadata.with_suffix(".pt")
        if model_path.exists():
            paths.append(model_path)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "cp8_commit": CP8_COMMIT,
        "protocol_checksum": sha256_file(PROTOCOL_PATH),
        "pi1_model_checksum": PI1_MODEL_SHA,
        "pi2_model_checksum": PI2_MODEL_SHA,
        "pi3_route_model_checksum": selected["pi3_route_model_checksum"],
        "artifacts": {
            str(path.relative_to(ARTIFACT_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in sorted(set(paths))
        },
        "implementation_hashes": _hash_paths(IMPLEMENTATION_PATHS),
        "simulator_hashes": _hash_paths(SIMULATOR_PATHS),
        "content_hashes": _hash_paths(CONTENT_PATHS),
        "training_performed": True,
        "test_accessed": True,
        "post_test_tuning_performed": False,
        "reserved_confirmation_opened": False,
        "production_gameplay_changed": False,
    }
    _write_json(MANIFEST_PATH, manifest)


def _verify_manifest() -> None:
    manifest = _read_json(MANIFEST_PATH)
    if sha256_file(PROTOCOL_PATH) != manifest["protocol_checksum"]:
        raise ValueError("CP9 protocol checksum changed after TEST.")
    bad = [
        relative
        for relative, expected in manifest["artifacts"].items()
        if sha256_file(ARTIFACT_ROOT / relative) != expected
    ]
    if bad:
        raise ValueError(f"CP9 artifact checksum mismatch: {bad}")
    if _hash_paths(IMPLEMENTATION_PATHS) != manifest["implementation_hashes"]:
        raise ValueError("CP9 implementation changed after TEST.")
    if _hash_paths(SIMULATOR_PATHS) != manifest["simulator_hashes"]:
        raise ValueError("Production simulator changed after CP9 TEST.")
    if _hash_paths(CONTENT_PATHS) != manifest["content_hashes"]:
        raise ValueError("Production content changed after CP9 TEST.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Checkpoint 9 route-aware policy learning.")
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--pre-access-tests-passed", action="store_true")
    sub.add_parser("train")
    sub.add_parser("test")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze_protocol(pre_access_tests_passed=bool(args.pre_access_tests_passed))
    elif args.command == "train":
        run_train()
    elif args.command == "test":
        run_test()
    else:
        raise ValueError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
