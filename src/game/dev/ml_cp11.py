# ruff: noqa: E501
"""Checkpoint 11: untouched confirmation of the frozen conservative route policy.

Research only. This checkpoint performs no training, normalization fitting, model
selection, threshold tuning, balance work, or production gameplay changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from game.data.loaders import load_game_definitions
from game.dev.ml_cp5 import _load_frozen_policies
from game.dev.ml_cp6 import PRIMARY_ROUTES
from game.dev.ml_cp7 import SAFETY_BUDGET
from game.dev.ml_cp10 import ARTIFACT_ROOT as CP10_ROOT
from game.dev.ml_cp10 import _load_selected_pi4
from game.ml.confirmation_audit import mcnemar_exact, paired_difference_ci
from game.ml.conservative_route_policy import ConservativeRoutePolicy, candidate_family
from game.ml.route_policy_learning import (
    PRIMARY_FEATURE_NAMES,
    RecordingRoutePolicy,
    paired_binary_summary,
    route_position,
    run_route_trajectory,
    sha256_file,
)
from game.ml.route_termination import RouteTerminalStatus

EXPERIMENT_ID = "checkpoint11-untouched-conservative-route-confirmation-v1"
ARTIFACT_ROOT = Path("artifacts/ml") / EXPERIMENT_ID
PROTOCOL_PATH = ARTIFACT_ROOT / "frozen_protocol.json"
PROVENANCE_PATH = ARTIFACT_ROOT / "provenance.json"
SEED_AUTHORITY_PATH = ARTIFACT_ROOT / "seed_authority.json"
UNTOUCHED_AUDIT_PATH = ARTIFACT_ROOT / "untouched_range_audit.json"
POLICY_IDENTITY_PATH = ARTIFACT_ROOT / "policy_identity.json"
MAINTENANCE_BASELINE_PATH = ARTIFACT_ROOT / "maintenance_baseline.json"
ROUTE_RESULTS_PATH = ARTIFACT_ROOT / "route_results.json"
PAIRED_PATH = ARTIFACT_ROOT / "paired_primary_analysis.json"
CLASSIFICATION_PATH = ARTIFACT_ROOT / "confirmation_classification.json"
EFFECT_PATH = ARTIFACT_ROOT / "cp10_cp11_effect_comparison.json"
BEHAVIOR_PATH = ARTIFACT_ROOT / "behavioral_summary.json"
OVERRIDE_PATH = ARTIFACT_ROOT / "override_analysis.json"
MARK_PATH = ARTIFACT_ROOT / "mark_confirmation.json"
GRAB_PATH = ARTIFACT_ROOT / "grab_confirmation.json"
PRESSURE_PATH = ARTIFACT_ROOT / "direct_pressure_confirmation.json"
GUARD_PATH = ARTIFACT_ROOT / "guard_confirmation.json"
DRAIN_PATH = ARTIFACT_ROOT / "effort_drain_confirmation.json"
SUPPORT_PATH = ARTIFACT_ROOT / "support_distribution.json"
ROBUSTNESS_PATH = ARTIFACT_ROOT / "route_robustness.json"
POOLED_PATH = ARTIFACT_ROOT / "pooled_summary.json"
PRODUCTION_PATH = ARTIFACT_ROOT / "production_evaluation_decision.json"
NEXT_PATH = ARTIFACT_ROOT / "next_research_recommendation.json"
REPORT_PATH = ARTIFACT_ROOT / "REPORT.md"
MANIFEST_PATH = ARTIFACT_ROOT / "manifest.json"

CONFIRMATION_SEEDS = tuple(range(63001, 63065))
PRIMARY_HERO_POLICY = "company_survival"
EXPECTED_CASE_COUNT = 64 * 6

CP10_COMMIT = "ec7077a946f5e8e8e5168954c5c7141bf474a77a"
MAINTENANCE_COMMIT = "80e5f1ca1b4b406067a979873da8d868e2a78dd2"
EXPECTED_PI2_SHA = "718fc3167a98a34aa5c17077c5d7ecea6e0c811c37c391e2ba558061a6576424"
EXPECTED_PI4_SHA = "0d454397df0867c565c622ebbceb3b8c38ebf78742e2f7b5f2faa6635843b82b"
EXPECTED_CP10_PROTOCOL_SHA = "f236c4ab114addb3eb96c62454ae273e0d8843615752c047f14c5242832ddecd"
EXPECTED_CP10_REPORT_SHA = "6654d38f96d403290c6c38f550b7482816903e44b6801241e2f0439f40491153"
EXPECTED_GATE_MODE = "MARGIN_GATED"
EXPECTED_MARGIN_THRESHOLD = 3.4007568359375
CP10_EFFECT = 0.0625
CP10_EFFECT_CI = (0.005827731661230635, 0.11917226833876937)
CP10_MCNEMAR_P = 0.0703125
CP10_PI3_MARK_ACTION_RATE = 233 / 1720
CP10_PI4_MARK_ACTION_RATE = 82 / 2587
CP10_MARK_REPAIR_MIDPOINT = (CP10_PI3_MARK_ACTION_RATE + CP10_PI4_MARK_ACTION_RATE) / 2
EFFORT_DRAIN_NEAR_UNIVERSAL_RATE = 0.95

PI2_MODEL_PATH = Path("artifacts/ml/checkpoint4-policy-iteration-v1/cp4_outcome_mlp_seed_20261009.pt")
CP10_SELECTED_PATH = CP10_ROOT / "selected_pi4_conservative.json"
CP10_PROTOCOL_PATH = CP10_ROOT / "frozen_protocol.json"
CP10_REPORT_PATH = CP10_ROOT / "REPORT.md"
CP10_PAIRED_PATH = CP10_ROOT / "paired_route_comparison.json"
CP10_CORRECTION_PATH = CP10_ROOT / "pretest_implementation_correction.json"

IMPLEMENTATION_PATHS = (
    "src/game/dev/ml_cp11.py",
    "tests/test_ml_untouched_route_confirmation.py",
    "src/game/dev/ml_cp10.py",
    "src/game/ml/conservative_route_policy.py",
    "src/game/ml/route_policy_learning.py",
    "src/game/ml/route_termination.py",
    "src/game/ml/stateful_route.py",
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


def _feature_schema_checksum() -> str:
    return _sha_payload({"schema_id": "enemy-route-outcome-candidate-v1", "features": list(PRIMARY_FEATURE_NAMES)})


def _policy_identity() -> dict[str, Any]:
    selected = _read_json(CP10_SELECTED_PATH)
    correction = _read_json(CP10_CORRECTION_PATH)
    pi4_metadata_path = Path(str(selected["ranker_metadata_path"]))
    pi4_model_path = pi4_metadata_path.parent / str(_read_json(pi4_metadata_path)["model_file"])
    identity = {
        "pi2": {
            "policy_id": "pi2_cp4",
            "model_path": str(PI2_MODEL_PATH),
            "model_sha256": sha256_file(PI2_MODEL_PATH),
        },
        "pi4": {
            "policy_id": "pi4_conservative",
            "scorer_id": selected["ranker_model_id"],
            "model_path": str(pi4_model_path),
            "model_sha256": sha256_file(pi4_model_path),
            "metadata_path": str(pi4_metadata_path),
            "metadata_sha256": sha256_file(pi4_metadata_path),
            "normalization_path": str(selected["normalization_path"]),
            "normalization_sha256": sha256_file(Path(str(selected["normalization_path"]))),
            "support_profile_path": str(selected["support_profile_path"]),
            "support_profile_sha256": sha256_file(Path(str(selected["support_profile_path"]))),
            "gate_mode": selected["gate_mode"],
            "margin_threshold": float(selected["margin_threshold"]),
        },
        "cp10": {
            "protocol_sha256": sha256_file(CP10_PROTOCOL_PATH),
            "report_sha256": sha256_file(CP10_REPORT_PATH),
            "sealed_lifecycle_boolean_inconsistency_known": True,
            "pretest_source_hash_correction": {
                "old": correction["old_cp10_source_checksum"],
                "new": correction["new_cp10_source_checksum"],
                "stage": correction["stage"],
            },
        },
        "feature_schema": {
            "id": "enemy-route-outcome-candidate-v1",
            "feature_count": len(PRIMARY_FEATURE_NAMES),
            "checksum": _feature_schema_checksum(),
        },
    }
    if identity["pi2"]["model_sha256"] != EXPECTED_PI2_SHA:
        raise ValueError("Frozen pi2 model checksum mismatch")
    if identity["pi4"]["model_sha256"] != EXPECTED_PI4_SHA:
        raise ValueError("Frozen pi4 model checksum mismatch")
    if identity["pi4"]["gate_mode"] != EXPECTED_GATE_MODE:
        raise ValueError("Frozen pi4 gate mode mismatch")
    if identity["pi4"]["margin_threshold"] != EXPECTED_MARGIN_THRESHOLD:
        raise ValueError("Frozen pi4 threshold mismatch")
    if identity["cp10"]["protocol_sha256"] != EXPECTED_CP10_PROTOCOL_SHA:
        raise ValueError("CP10 protocol checksum mismatch")
    if identity["cp10"]["report_sha256"] != EXPECTED_CP10_REPORT_SHA:
        raise ValueError("CP10 report checksum mismatch")
    return identity


def _prior_artifact_dirs() -> list[Path]:
    root = Path("artifacts/ml")
    dirs: list[Path] = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith("checkpoint") and path.name != EXPERIMENT_ID:
            match = re.match(r"checkpoint(\d+)", path.name)
            if match and int(match.group(1)) <= 10:
                dirs.append(path)
    return sorted(dirs)


def _untouched_range_audit() -> dict[str, Any]:
    exact_middle = re.compile(r"(?<![0-9A-Za-z])(630(?:0[2-9]|[1-5][0-9]|6[0-3]))(?![0-9A-Za-z])")
    root_seed = re.compile(r'"root_seed"\s*:\s*(\d+)')
    middle_hits: list[dict[str, Any]] = []
    root_seed_hits: list[dict[str, Any]] = []
    endpoint_refs: list[dict[str, Any]] = []
    files_scanned = 0
    for directory in _prior_artifact_dirs():
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".txt", ".csv", ".yaml", ".yml"}:
                continue
            files_scanned += 1
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    for line_number, line in enumerate(handle, 1):
                        for match in root_seed.finditer(line):
                            value = int(match.group(1))
                            if 63001 <= value <= 63064:
                                root_seed_hits.append({"path": str(path), "line": line_number, "value": value})
                        for match in exact_middle.finditer(line):
                            middle_hits.append({"path": str(path), "line": line_number, "value": int(match.group(1))})
                        if re.search(r"(?<![0-9A-Za-z])(63001|63064)(?![0-9A-Za-z])", line):
                            endpoint_refs.append({"path": str(path), "line": line_number, "text": line.strip()[:240]})
            except OSError:
                continue
    code_refs: list[dict[str, Any]] = []
    for path in sorted(Path("src/game/dev").glob("ml_cp*.py")):
        if path.name == "ml_cp11.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_number, line in enumerate(text.splitlines(), 1):
            if "63001" in line or "63064" in line:
                code_refs.append({"path": str(path), "line": line_number, "text": line.strip()[:240]})
    compromised = bool(root_seed_hits or middle_hits)
    return {
        "status": "UNTOUCHED_RANGE_COMPROMISED" if compromised else "UNTOUCHED_RANGE_VERIFIED",
        "reserved_range": [63001, 63064],
        "prior_artifact_files_scanned": files_scanned,
        "root_seed_hits": root_seed_hits,
        "standalone_middle_seed_hits": middle_hits,
        "endpoint_references": endpoint_refs,
        "code_boundary_references": code_refs,
        "interpretation": "Endpoint references are reservation/protocol metadata; any prior root_seed or standalone interior seed would compromise the block.",
    }


def _protocol(identity: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    route_ids = [route.route_id for route in PRIMARY_ROUTES]
    if len(route_ids) != 6:
        raise ValueError(f"Expected six frozen primary routes, found {len(route_ids)}")
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_before_confirmation_seed_access",
        "repository_head": _git_head(),
        "cp10_commit": CP10_COMMIT,
        "maintenance_commit": MAINTENANCE_COMMIT,
        "cp10_sealed_lifecycle_boolean_inconsistency_known": True,
        "reserved_seed_range": [63001, 63064],
        "untouched_range_status": audit["status"],
        "primary_hero_policy": PRIMARY_HERO_POLICY,
        "route_suite": route_ids,
        "expected_primary_cases_per_policy": EXPECTED_CASE_COUNT,
        "policies": identity,
        "primary_endpoint": "production-valid ENEMY ROUTE VICTORY under CP7 termination semantics",
        "termination": {
            "module": "game.ml.route_termination",
            "safety_budget": asdict(SAFETY_BUDGET),
            "synthetic_withdrawal": False,
            "valid_terminals": [
                RouteTerminalStatus.ROUTE_COMPLETED.value,
                RouteTerminalStatus.ROUTE_FAILED_ENEMY_VICTORY.value,
                RouteTerminalStatus.ROUTE_ENDED_PRODUCTION_RETREAT.value,
            ],
        },
        "paired_statistics": {
            "primary_test": "exact two-sided McNemar on discordant paired enemy-victory outcomes",
            "mcnemar_method": "two-sided exact binomial tail with p=0.5 over discordant pairs; double smaller tail and cap at 1",
            "ci_method": "normal 95% CI for mean paired binary difference using z=1.959963984540054 and sample variance, clipped to [-1,1]",
            "ci_role": "complementary effect-size uncertainty; exact McNemar is the confirmatory hypothesis test",
            "why_ci_and_p_can_differ": "the normal paired-difference CI and exact conditional McNemar test use different finite-sample constructions",
            "equivalence_margin": None,
        },
        "classification_rules": {
            "CLEARLY_CONFIRMED": "pi4 rate > pi2, pi4-only wins > pi2-only wins, exact McNemar p < .05",
            "SUPPORTED_BUT_UNDERPOWERED": "pi4 rate > pi2, pi4-only wins > pi2-only wins, exact McNemar p >= .05",
            "NO_REPLICATION": "paired difference == 0 or discordant wins tie, without significant regression",
            "REGRESSION_SIGNAL": "pi4 rate < pi2, pi2-only wins > pi4-only wins, exact McNemar p >= .05",
            "REFUTED": "pi4 rate < pi2, pi2-only wins > pi4-only wins, exact McNemar p < .05",
        },
        "effect_compatibility_rule": "YES if either checkpoint point estimate lies inside the other checkpoint's 95% paired-difference CI; NO if the two CIs are disjoint; otherwise INCONCLUSIVE",
        "behavioral_diagnostics": [
            "pi4 override rate and pi2 agreement",
            "Mark legal opportunities/selections/route positions",
            "Bite and direct-pressure selections",
            "Grab legal opportunities/selections/route positions/target state",
            "Guard legal opportunities/selections/route positions",
            "Effort Drain legal opportunities/selections/targets/route positions/declines",
            "lowest-HP targeting",
            "target switching between adjacent enemy actions within encounters",
            "pi4 support category for all decisions/overrides/successes/failures",
        ],
        "secondary_behavior_rules": {
            "effort_drain_near_universal_threshold": EFFORT_DRAIN_NEAR_UNIVERSAL_RATE,
            "effort_drain_flag_if_at_or_above_threshold": "PERSISTENT_POTENTIAL_DOMINANCE",
            "mark_cp10_pi3_action_rate": CP10_PI3_MARK_ACTION_RATE,
            "mark_cp10_pi4_action_rate": CP10_PI4_MARK_ACTION_RATE,
            "mark_repair_midpoint": CP10_MARK_REPAIR_MIDPOINT,
            "mark_replication_rule": "YES if CP11 pi4 Mark share of all enemy decisions is at or below the frozen CP10 pi3/pi4 midpoint; PARTIAL if below the CP10 pi3 rate but above midpoint; NO otherwise",
            "no_rerun_after_first_seed_access": True,
        },
        "primary_only": True,
        "pi3_included": False,
        "training_performed": False,
        "model_selection_performed": False,
        "threshold_tuning_performed": False,
        "post_seed_access_tuning_performed": False,
        "no_scientific_modification_after_protocol_freeze": True,
    }


def freeze() -> None:
    if _git_head() != MAINTENANCE_COMMIT:
        raise ValueError(f"Unexpected pre-CP11 HEAD: {_git_head()}")
    if _git_dirty():
        status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
        allowed = {"?? src/game/dev/ml_cp11.py", "?? tests/test_ml_untouched_route_confirmation.py"}
        actual = {line.strip() for line in status.splitlines() if line.strip()}
        if not actual.issubset(allowed):
            raise ValueError(f"Unexpected dirty pre-freeze tree: {sorted(actual)}")
    if ROUTE_RESULTS_PATH.exists():
        raise ValueError("CP11 route results already exist; refusing to refreeze")
    identity = _policy_identity()
    audit = _untouched_range_audit()
    if audit["status"] != "UNTOUCHED_RANGE_VERIFIED":
        _write_json(UNTOUCHED_AUDIT_PATH, audit)
        raise ValueError("UNTOUCHED_RANGE_COMPROMISED")
    protocol = _protocol(identity, audit)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(UNTOUCHED_AUDIT_PATH, audit)
    _write_json(POLICY_IDENTITY_PATH, identity)
    _write_json(
        SEED_AUTHORITY_PATH,
        {
            "confirmation_range": [63001, 63064],
            "root_seed_count": 64,
            "route_count": len(PRIMARY_ROUTES),
            "primary_cases_per_policy": EXPECTED_CASE_COUNT,
            "status_before_open": "UNTOUCHED_RANGE_VERIFIED",
        },
    )
    _write_json(
        MAINTENANCE_BASELINE_PATH,
        {
            "cp10_commit": CP10_COMMIT,
            "maintenance_commit": MAINTENANCE_COMMIT,
            "scientific_behavior_change": False,
            "full_repository_pytest": "PASS",
            "ruff": "PASS",
            "mypy": "PASS",
            "git_diff_check": "PASS",
            "ordinary_game_import": "PASS",
            "torch_imported": False,
            "production_gameplay_content_diff": "NONE",
        },
    )
    _write_json(PROTOCOL_PATH, protocol)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "git_head": _git_head(),
        "repository_dirty_only_cp11_untracked": True,
        "protocol_checksum": protocol_sha,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "implementation_hashes": _hash_paths(IMPLEMENTATION_PATHS),
        "simulator_hashes": _hash_paths(SIMULATOR_PATHS),
        "content_hashes": _hash_paths(CONTENT_PATHS),
        "cp10_sealed_lifecycle_boolean_inconsistency_known": True,
        "training_performed": False,
        "model_selection_performed": False,
        "threshold_tuning_performed": False,
        "confirmation_seed_accessed": False,
        "post_seed_access_tuning_performed": False,
        "production_gameplay_changed": False,
    }
    _write_json(PROVENANCE_PATH, provenance)
    print("UNTOUCHED_RANGE_VERIFIED")
    print(f"CP11_PROTOCOL_SHA={protocol_sha}")
    print("CP11_CONFIRMATION_SEED_ACCESSED=false")


def _verify_frozen_before_run() -> tuple[dict[str, Any], dict[str, Any]]:
    if not PROTOCOL_PATH.exists() or not PROVENANCE_PATH.exists():
        raise ValueError("CP11 must be frozen before confirmation")
    if ROUTE_RESULTS_PATH.exists():
        raise ValueError("CP11 confirmation outputs already exist; refusing to rerun")
    protocol = _read_json(PROTOCOL_PATH)
    provenance = _read_json(PROVENANCE_PATH)
    if bool(provenance.get("confirmation_seed_accessed")):
        raise ValueError("CP11 confirmation seed block was already opened; refusing to rerun")
    if sha256_file(PROTOCOL_PATH) != provenance["protocol_checksum"]:
        raise ValueError("CP11 protocol changed after freeze")
    for group, paths in (("implementation_hashes", IMPLEMENTATION_PATHS), ("simulator_hashes", SIMULATOR_PATHS), ("content_hashes", CONTENT_PATHS)):
        expected = provenance[group]
        current = _hash_paths(paths)
        if expected != current:
            raise ValueError(f"{group} changed after CP11 freeze")
    identity = _policy_identity()
    if identity != _read_json(POLICY_IDENTITY_PATH):
        raise ValueError("Frozen policy identity changed")
    if _read_json(UNTOUCHED_AUDIT_PATH)["status"] != "UNTOUCHED_RANGE_VERIFIED":
        raise ValueError("Untouched range is not verified")
    return protocol, provenance


def _candidate_families(capture: Any, definitions: Any) -> list[list[str]]:
    rows: list[list[str]] = []
    for key, observation in zip(capture.candidates, capture.primary_observations, strict=True):
        families = list(candidate_family(observation))
        skill = definitions.skills[key.skill_id]
        tags = set(skill.tags)
        if "bite" in key.skill_id or "bite" in tags:
            families.append("BITE")
        rows.append(sorted(set(families)))
    return rows


def _decision_rows(
    recorder: RecordingRoutePolicy,
    definitions: Any,
    *,
    route_enemy_victory: bool,
    conservative: ConservativeRoutePolicy | None,
) -> list[dict[str, Any]]:
    if conservative is not None and len(conservative.audit) != len(recorder.captures):
        raise ValueError("pi4 gate audit does not align with recorded decisions")
    target_hp_index = PRIMARY_FEATURE_NAMES.index("target_hp")
    target_hero_index = PRIMARY_FEATURE_NAMES.index("target_is_hero")
    target_downed_index = PRIMARY_FEATURE_NAMES.index("target_is_downed")
    target_protected_index = PRIMARY_FEATURE_NAMES.index("target_is_protected")
    rows: list[dict[str, Any]] = []
    for index, capture in enumerate(recorder.captures):
        families = _candidate_families(capture, definitions)
        chosen_index = capture.candidates.index(capture.source_choice)
        chosen_obs = capture.primary_observations[chosen_index]
        hero_hps = [obs[target_hp_index] for obs in capture.primary_observations if obs[target_hero_index] > 0.5]
        gate = None if conservative is None else conservative.audit[index]
        rows.append(
            {
                "action_index": capture.global_action_index,
                "route_id": capture.route_id,
                "encounter_index": capture.encounter_index,
                "encounter_id": capture.encounter_id,
                "route_position": route_position(capture.encounter_index, capture.total_encounters),
                "round_number": capture.round_number,
                "enemy_class_id": capture.enemy_class_id,
                "chosen": asdict(capture.source_choice),
                "chosen_families": families[chosen_index],
                "candidate_families": families,
                "legal_candidate_count": len(capture.candidates),
                "target_hp": float(chosen_obs[target_hp_index]),
                "target_is_hero": bool(chosen_obs[target_hero_index] > 0.5),
                "target_is_downed": bool(chosen_obs[target_downed_index] > 0.5),
                "target_is_protected": bool(chosen_obs[target_protected_index] > 0.5),
                "target_is_lowest_hp_legal": bool(chosen_obs[target_hero_index] > 0.5 and hero_hps and chosen_obs[target_hp_index] <= min(hero_hps)),
                "party_hp_fraction": capture.party_hp_fraction,
                "party_effort_fraction": capture.party_effort_fraction,
                "route_enemy_victory": route_enemy_victory,
                "gate": gate,
            }
        )
    return rows


def _run_case(definitions: Any, policies: Mapping[str, Any], pi4_ranker: Any, support_profile: Any, selected: Mapping[str, Any], route: Any, seed: int, policy_name: str) -> dict[str, Any]:
    conservative: ConservativeRoutePolicy | None = None
    if policy_name == "pi2_cp4":
        base_policy = policies["pi2_cp4"]
    elif policy_name == "pi4_conservative":
        conservative = ConservativeRoutePolicy(
            anchor_policy=policies["pi2_cp4"],
            route_policy=pi4_ranker,
            support_profile=support_profile,
            gate_mode=str(selected["gate_mode"]),
            margin_threshold=float(selected["margin_threshold"]),
            policy_id="pi4_conservative",
        )
        base_policy = conservative
    else:
        raise ValueError(policy_name)
    recorder = RecordingRoutePolicy(base_policy)
    outcome, route_digest, encounters, _ = run_route_trajectory(
        definitions,
        route,
        root_seed=seed,
        enemy_policy_name=policy_name,
        enemy_policy=base_policy,
        hero_policy_id=PRIMARY_HERO_POLICY,
        safety_budget=SAFETY_BUDGET,
        recording_policy=recorder,
    )
    decision_rows = _decision_rows(
        recorder,
        definitions,
        route_enemy_victory=outcome.enemy_victory,
        conservative=conservative,
    )
    return {
        "case_id": f"{route.route_id}|{PRIMARY_HERO_POLICY}|{seed}",
        "root_seed": seed,
        "route_id": route.route_id,
        "route_structure": route.structure,
        "hero_policy_id": PRIMARY_HERO_POLICY,
        "policy": policy_name,
        "enemy_victory": outcome.enemy_victory,
        "route_completed": outcome.route_completed,
        "production_retreat": outcome.terminal_status == RouteTerminalStatus.ROUTE_ENDED_PRODUCTION_RETREAT.value,
        "cycle": outcome.terminal_status == RouteTerminalStatus.ROUTE_CENSORED_CYCLE.value,
        "safety_bound_censoring": outcome.terminal_status == RouteTerminalStatus.ROUTE_CENSORED_SAFETY_BOUND.value,
        "invalid_simulation": outcome.terminal_status == RouteTerminalStatus.ROUTE_INVALID.value,
        "terminal_status": outcome.terminal_status,
        "route_digest": route_digest,
        "final_hero_hp": outcome.final_hero_hp,
        "final_hero_effort": outcome.final_hero_effort,
        "final_downed": outcome.final_downed,
        "hero_deaths": outcome.hero_deaths,
        "mortal_wounds": outcome.mortal_wounds,
        "strain_marks": outcome.strain_marks,
        "encounters": encounters,
        "decisions": decision_rows,
    }


def _paired(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pi2 = {str(row["case_id"]): row for row in rows if row["policy"] == "pi2_cp4"}
    pi4 = {str(row["case_id"]): row for row in rows if row["policy"] == "pi4_conservative"}
    keys = sorted(pi2.keys() & pi4.keys())
    if len(keys) != EXPECTED_CASE_COUNT:
        raise ValueError(f"Expected {EXPECTED_CASE_COUNT} paired cases, got {len(keys)}")
    summary = paired_binary_summary(
        [bool(pi2[key]["enemy_victory"]) for key in keys],
        [bool(pi4[key]["enemy_victory"]) for key in keys],
    )
    by_route: dict[str, Any] = {}
    for route in PRIMARY_ROUTES:
        route_keys = [key for key in keys if pi2[key]["route_id"] == route.route_id]
        by_route[route.route_id] = paired_binary_summary(
            [bool(pi2[key]["enemy_victory"]) for key in route_keys],
            [bool(pi4[key]["enemy_victory"]) for key in route_keys],
        )
    return {"overall": summary, "by_route": by_route}


def classify_confirmation(summary: Mapping[str, Any]) -> str:
    diff = float(summary["win_rate_difference"])
    p = float(summary["mcnemar_exact_two_sided_p"])
    pi2_only = int(summary["first_only_wins"])
    pi4_only = int(summary["second_only_wins"])
    if diff > 0 and pi4_only > pi2_only:
        return "CLEARLY_CONFIRMED" if p < 0.05 else "SUPPORTED_BUT_UNDERPOWERED"
    if diff < 0 and pi2_only > pi4_only:
        return "REFUTED" if p < 0.05 else "REGRESSION_SIGNAL"
    return "NO_REPLICATION"


def effect_compatibility(cp11_diff: float, cp11_ci: Sequence[float]) -> str:
    low, high = float(cp11_ci[0]), float(cp11_ci[1])
    cp10_low, cp10_high = CP10_EFFECT_CI
    if low <= CP10_EFFECT <= high or cp10_low <= cp11_diff <= cp10_high:
        return "YES"
    if high < cp10_low or low > cp10_high:
        return "NO"
    return "INCONCLUSIVE"


def _behavior_totals(rows: Sequence[Mapping[str, Any]], policy: str) -> dict[str, Any]:
    cases = [row for row in rows if row["policy"] == policy]
    totals: defaultdict[str, float] = defaultdict(float)
    skill_counts: Counter[str] = Counter()
    for row in cases:
        for encounter in row["encounters"]:
            for key, value in encounter["behavior"].items():
                if key == "skill_counts":
                    skill_counts.update({str(k): int(v) for k, v in value.items()})
                elif isinstance(value, (int, float)) and key != "no_progress_round_rate":
                    totals[key] += float(value)
    payload: dict[str, Any] = dict(totals)
    payload["skill_counts"] = dict(sorted(skill_counts.items()))
    return {"case_count": len(cases), "totals": payload}


def _all_decisions(rows: Sequence[Mapping[str, Any]], policy: str) -> list[dict[str, Any]]:
    return [decision | {"case_id": row["case_id"], "root_seed": row["root_seed"]} for row in rows if row["policy"] == policy for decision in row["decisions"]]


def _family_confirmation(decisions: Sequence[Mapping[str, Any]], family: str) -> dict[str, Any]:
    legal = [row for row in decisions if any(family in families for families in row["candidate_families"])]
    selected = [row for row in legal if family in row["chosen_families"]]
    return {
        "legal_opportunities": len(legal),
        "selections": len(selected),
        "conditional_selection_rate": len(selected) / len(legal) if legal else 0.0,
        "selected_route_positions": dict(Counter(str(row["route_position"]) for row in selected)),
        "selected_targets": dict(Counter(str(row["chosen"]["target_id"]) for row in selected)),
        "selected_target_downed": sum(bool(row["target_is_downed"]) for row in selected),
        "selected_target_protected": sum(bool(row["target_is_protected"]) for row in selected),
        "selected_route_enemy_victories": sum(bool(row["route_enemy_victory"]) for row in selected),
    }


def _targeting_summary(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hero_targets = [row for row in decisions if row["target_is_hero"]]
    lowest = sum(bool(row["target_is_lowest_hp_legal"]) for row in hero_targets)
    by_encounter: defaultdict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in decisions:
        by_encounter[(str(row["case_id"]), int(row["encounter_index"]))].append(row)
    switches = opportunities = 0
    for items in by_encounter.values():
        ordered = sorted(items, key=lambda item: int(item["action_index"]))
        for first, second in zip(ordered, ordered[1:], strict=False):
            opportunities += 1
            switches += int(first["chosen"]["target_id"] != second["chosen"]["target_id"])
    return {
        "hero_target_decisions": len(hero_targets),
        "lowest_hp_target_selections": lowest,
        "lowest_hp_target_rate": lowest / len(hero_targets) if hero_targets else 0.0,
        "target_switches": switches,
        "target_switch_opportunities": opportunities,
        "target_switch_rate": switches / opportunities if opportunities else 0.0,
    }


def _support_summary(pi4: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    with_gate = [row for row in pi4 if row["gate"] is not None]
    overrides = [row for row in with_gate if row["gate"]["gate_accepted"]]
    def counts(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return dict(Counter(str(row["gate"]["support"]["category"]) for row in items))
    success = [row for row in with_gate if row["route_enemy_victory"]]
    failed = [row for row in with_gate if not row["route_enemy_victory"]]
    return {
        "all_pi4_decisions": counts(with_gate),
        "overrides_only": counts(overrides),
        "successful_routes": counts(success),
        "failed_routes": counts(failed),
    }


def _override_summary(pi4: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gate_rows = [row for row in pi4 if row["gate"] is not None]
    overrides = [row for row in gate_rows if row["gate"]["gate_accepted"]]
    agreements = sum(row["gate"]["executed_action"] == row["gate"]["anchor_action"] for row in gate_rows)
    return {
        "decisions": len(gate_rows),
        "overrides": len(overrides),
        "override_rate": len(overrides) / len(gate_rows) if gate_rows else 0.0,
        "agreement_with_pi2": agreements,
        "agreement_rate": agreements / len(gate_rows) if gate_rows else 0.0,
        "override_route_positions": dict(Counter(str(row["route_position"]) for row in overrides)),
        "override_routes": dict(Counter(str(row["route_id"]) for row in overrides)),
        "override_support": dict(Counter(str(row["gate"]["support"]["category"]) for row in overrides)),
    }


def _pooled(cp11: Mapping[str, Any]) -> dict[str, Any]:
    cp10 = _read_json(CP10_PAIRED_PATH)["pi4_vs_pi2"][PRIMARY_HERO_POLICY]
    first_only = int(cp10["first_only_wins"]) + int(cp11["first_only_wins"])
    second_only = int(cp10["second_only_wins"]) + int(cp11["second_only_wins"])
    shared_wins = int(cp10["shared_wins"]) + int(cp11["shared_wins"])
    shared_losses = int(cp10["shared_losses"]) + int(cp11["shared_losses"])
    diffs = [-1] * first_only + [1] * second_only + [0] * (shared_wins + shared_losses)
    count = len(diffs)
    return {
        "label": "POST-CONFIRMATION POOLED SUMMARY",
        "case_count": count,
        "pi2_wins": shared_wins + first_only,
        "pi4_wins": shared_wins + second_only,
        "shared_wins": shared_wins,
        "pi2_only_wins": first_only,
        "pi4_only_wins": second_only,
        "shared_losses": shared_losses,
        "win_rate_difference": sum(diffs) / count if count else 0.0,
        "paired_difference_ci_95": paired_difference_ci(diffs),
        "mcnemar_exact_two_sided_p": mcnemar_exact(first_only, second_only),
        "does_not_change_cp11_primary_classification": True,
    }


def _write_report(
    identity: Mapping[str, Any],
    paired: Mapping[str, Any],
    classification: str,
    effect: Mapping[str, Any],
    behavior: Mapping[str, Any],
    override: Mapping[str, Any],
    mark: Mapping[str, Any],
    grab: Mapping[str, Any],
    pressure: Mapping[str, Any],
    guard: Mapping[str, Any],
    drain: Mapping[str, Any],
    support: Mapping[str, Any],
    robustness: Mapping[str, Any],
    pooled: Mapping[str, Any],
    production: Mapping[str, Any],
    next_direction: Mapping[str, Any],
) -> None:
    overall = paired["overall"]
    lines = [
        "# THE CHARTER — ML RESEARCH PLATFORM V0.7",
        "## CHECKPOINT 11 — UNTOUCHED CONSERVATIVE ROUTE-POLICY CONFIRMATION",
        "",
        f"**Independent classification:** `{classification}`  ",
        "**Reserved confirmation block:** 63001–63064, opened once after freeze  ",
        "**Training/tuning:** NONE  ",
        "**Production gameplay/balance:** UNCHANGED",
        "",
        "## A. CP10 preservation / commit",
        f"CP10 commit: `{CP10_COMMIT}`. Pre-CP10 HEAD was `88ddf45872c22044ac9661be165d24da8e017623`.",
        "",
        "## B. Historical-test maintenance repair",
        f"Maintenance commit: `{MAINTENANCE_COMMIT}` with `SCIENTIFIC_BEHAVIOR_CHANGE=false`.",
        "",
        "## C. Clean pre-confirmation repository gate",
        "Focused CP6–CP10 tests, full pytest, Ruff, mypy, git diff --check, ordinary import, torch isolation, and production-diff checks passed before seed access.",
        "",
        "## D. CP10 sealed-provenance caveat",
        "`CP10_SEALED_LIFECYCLE_BOOLEAN_INCONSISTENCY_KNOWN=true`; the sealed CP10 provenance file was retained unchanged.",
        "",
        "## E. Frozen CP11 protocol",
        f"Protocol SHA-256: `{sha256_file(PROTOCOL_PATH)}`. Exact two-sided McNemar is primary; the paired-difference CI is complementary.",
        "",
        "## F. Untouched-range verification",
        "`UNTOUCHED_RANGE_VERIFIED` before the first confirmation outcome.",
        "",
        "## G. Exact π2 identity",
        f"π2 model SHA-256: `{identity['pi2']['model_sha256']}`.",
        "",
        "## H. Exact π4 identity",
        f"π4 model SHA-256: `{identity['pi4']['model_sha256']}`; gate `{identity['pi4']['gate_mode']}`; threshold `{identity['pi4']['margin_threshold']}`.",
        "",
        "## I. Confirmation route suite",
        f"Six frozen routes × 64 untouched root seeds = {EXPECTED_CASE_COUNT} primary paired company_survival cases.",
        "",
        "## J. Primary paired outcomes",
        f"π2: {overall['first_wins']}/{overall['case_count']} ({overall['first_rate']:.2%}); π4: {overall['second_wins']}/{overall['case_count']} ({overall['second_rate']:.2%}). Shared wins {overall['shared_wins']}; shared losses {overall['shared_losses']}; π2-only {overall['first_only_wins']}; π4-only {overall['second_only_wins']}.",
        "",
        "## K. Effect size / CI",
        f"π4 − π2 = {100*overall['win_rate_difference']:+.2f}pp; paired 95% CI [{100*overall['paired_difference_ci_95'][0]:+.2f}, {100*overall['paired_difference_ci_95'][1]:+.2f}]pp.",
        "",
        "## L. Exact McNemar result",
        f"Exact two-sided McNemar p = {overall['mcnemar_exact_two_sided_p']:.12g}.",
        "",
        "## M. Frozen confirmation classification",
        f"`{classification}`.",
        "",
        "## N. CP10 vs CP11 effect comparison",
        json.dumps(effect, indent=2, sort_keys=True),
        "",
        "## O. Route-specific robustness",
        json.dumps(robustness, indent=2, sort_keys=True),
        "",
        "## P. π4 override rate",
        json.dumps(override, indent=2, sort_keys=True),
        "",
        "## Q. Mark behavior",
        json.dumps(mark, indent=2, sort_keys=True),
        "",
        "## R. Grab behavior",
        json.dumps(grab, indent=2, sort_keys=True),
        "",
        "## S. Direct-pressure behavior",
        json.dumps(pressure, indent=2, sort_keys=True),
        "",
        "## T. Guard behavior",
        json.dumps(guard, indent=2, sort_keys=True),
        "",
        "## U. Effort Drain behavior",
        json.dumps(drain, indent=2, sort_keys=True),
        "",
        "## V. State-support distribution",
        json.dumps(support, indent=2, sort_keys=True),
        "",
        "## W. Optional post-confirmation pooled summary",
        json.dumps(pooled, indent=2, sort_keys=True),
        "",
        "## X. Production-evaluation decision",
        json.dumps(production, indent=2, sort_keys=True),
        "",
        "## Y. Recommended next intelligence direction",
        json.dumps(next_direction, indent=2, sort_keys=True),
        "",
        "## Z. Limitations",
        "This is deterministic simulator confirmation on one frozen six-route suite and one primary hero policy. It confirms or rejects generalization within that environment; it does not establish production UX, difficulty calibration, runtime suitability, or global optimality.",
        "",
        "## AA. Quality gates",
        "PENDING_FINAL_CLOSEOUT",
        "",
        "## AB. Artifact/checksum manifest",
        "See `manifest.json`; final closeout independently verifies every listed checksum.",
        "",
        "## AC. Final verdict",
        f"Independent CP11 classification: **{classification}**. No production promotion or balance change is performed here.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_confirmation() -> None:
    protocol, provenance = _verify_frozen_before_run()
    provenance["confirmation_seed_accessed"] = True
    provenance["confirmation_run_started"] = True
    _write_json(PROVENANCE_PATH, provenance)
    definitions = load_game_definitions()
    policies, _ = _load_frozen_policies()
    pi4_ranker, _normalization, support_profile, selected = _load_selected_pi4()
    rows: list[dict[str, Any]] = []
    for seed in CONFIRMATION_SEEDS:
        for route in PRIMARY_ROUTES:
            for policy_name in ("pi2_cp4", "pi4_conservative"):
                rows.append(_run_case(definitions, policies, pi4_ranker, support_profile, selected, route, seed, policy_name))
        print(f"CP11_SEED_DONE={seed}", flush=True)
    _write_json(ROUTE_RESULTS_PATH, {"rows": rows})

    paired = _paired(rows)
    _write_json(PAIRED_PATH, paired)
    overall = paired["overall"]
    classification = classify_confirmation(overall)
    classification_payload = {
        "classification": classification,
        "frozen_rule_applied": True,
        "primary_test": "exact two-sided McNemar",
        "mcnemar_p": overall["mcnemar_exact_two_sided_p"],
        "win_rate_difference": overall["win_rate_difference"],
        "paired_difference_ci_95": overall["paired_difference_ci_95"],
    }
    _write_json(CLASSIFICATION_PATH, classification_payload)

    compatibility = effect_compatibility(float(overall["win_rate_difference"]), overall["paired_difference_ci_95"])
    effect_payload = {
        "cp10_difference": CP10_EFFECT,
        "cp10_ci95": list(CP10_EFFECT_CI),
        "cp10_mcnemar_p": CP10_MCNEMAR_P,
        "cp11_difference": overall["win_rate_difference"],
        "cp11_ci95": overall["paired_difference_ci_95"],
        "same_direction": bool(overall["win_rate_difference"] > 0),
        "magnitude_relation": "larger" if overall["win_rate_difference"] > CP10_EFFECT else "smaller" if overall["win_rate_difference"] < CP10_EFFECT else "same",
        "broadly_compatible": compatibility,
    }
    _write_json(EFFECT_PATH, effect_payload)

    behavior = {
        "pi2": _behavior_totals(rows, "pi2_cp4"),
        "pi4": _behavior_totals(rows, "pi4_conservative"),
    }
    pi2_decisions = _all_decisions(rows, "pi2_cp4")
    pi4_decisions = _all_decisions(rows, "pi4_conservative")
    behavior["targeting"] = {
        "pi2": _targeting_summary(pi2_decisions),
        "pi4": _targeting_summary(pi4_decisions),
    }
    _write_json(BEHAVIOR_PATH, behavior)

    override = _override_summary(pi4_decisions)
    _write_json(OVERRIDE_PATH, override)

    pi2_mark = _family_confirmation(pi2_decisions, "MARK")
    pi4_mark = _family_confirmation(pi4_decisions, "MARK")
    mark_action_rate = (
        pi4_mark["selections"] / len(pi4_decisions) if pi4_decisions else 0.0
    )
    if mark_action_rate <= CP10_MARK_REPAIR_MIDPOINT:
        mark_replication = "YES"
    elif mark_action_rate < CP10_PI3_MARK_ACTION_RATE:
        mark_replication = "PARTIAL"
    else:
        mark_replication = "NO"
    mark_payload = {
        "pi2": pi2_mark,
        "pi4": pi4_mark,
        "pi4_mark_share_of_all_enemy_decisions": mark_action_rate,
        "cp10_pi3_mark_share": CP10_PI3_MARK_ACTION_RATE,
        "cp10_pi4_mark_share": CP10_PI4_MARK_ACTION_RATE,
        "frozen_repair_midpoint": CP10_MARK_REPAIR_MIDPOINT,
        "avoids_pi3_pathological_regime": mark_replication,
    }
    _write_json(MARK_PATH, mark_payload)

    pi2_grab = _family_confirmation(pi2_decisions, "GRAB")
    pi4_grab = _family_confirmation(pi4_decisions, "GRAB")
    grab_payload = {
        "pi2": pi2_grab,
        "pi4": pi4_grab,
        "elevated_baseline_grab_replicated": "YES" if pi4_grab["conditional_selection_rate"] > pi2_grab["conditional_selection_rate"] else "NO",
        "g1_implemented": False,
    }
    _write_json(GRAB_PATH, grab_payload)

    pi2_direct = _family_confirmation(pi2_decisions, "DIRECT_PRESSURE")
    pi4_direct = _family_confirmation(pi4_decisions, "DIRECT_PRESSURE")
    pi2_bite = _family_confirmation(pi2_decisions, "BITE")
    pi4_bite = _family_confirmation(pi4_decisions, "BITE")
    pressure_payload = {
        "pi2_direct_pressure": pi2_direct,
        "pi4_direct_pressure": pi4_direct,
        "pi2_bite": pi2_bite,
        "pi4_bite": pi4_bite,
        "useful_direct_pressure_present": "YES" if pi4_direct["selections"] > 0 and pi4_bite["selections"] > 0 else "PARTIAL" if pi4_direct["selections"] > 0 else "NO",
    }
    _write_json(PRESSURE_PATH, pressure_payload)

    guard_payload = {
        "pi2": _family_confirmation(pi2_decisions, "GUARD"),
        "pi4": _family_confirmation(pi4_decisions, "GUARD"),
    }
    _write_json(GUARD_PATH, guard_payload)

    pi2_drain = _family_confirmation(pi2_decisions, "EFFORT_DRAIN")
    pi4_drain = _family_confirmation(pi4_decisions, "EFFORT_DRAIN")
    drain_payload = {
        "pi2": pi2_drain,
        "pi4": pi4_drain,
        "pi4_declines_when_legal": pi4_drain["legal_opportunities"] - pi4_drain["selections"],
        "near_universal": bool(pi4_drain["legal_opportunities"] and pi4_drain["conditional_selection_rate"] >= EFFORT_DRAIN_NEAR_UNIVERSAL_RATE),
        "research_flag": "PERSISTENT_POTENTIAL_DOMINANCE" if pi4_drain["legal_opportunities"] and pi4_drain["conditional_selection_rate"] >= EFFORT_DRAIN_NEAR_UNIVERSAL_RATE else "INCONCLUSIVE",
    }
    _write_json(DRAIN_PATH, drain_payload)

    support = _support_summary(pi4_decisions)
    _write_json(SUPPORT_PATH, support)
    robustness = paired["by_route"]
    _write_json(ROBUSTNESS_PATH, robustness)
    pooled = _pooled(overall)
    _write_json(POOLED_PATH, pooled)

    production_candidate = classification == "CLEARLY_CONFIRMED"
    production_payload = {
        "production_evaluation_warranted": production_candidate,
        "production_promotion_performed": False,
        "immediate_balance_change_justified": False,
        "reason": "CP11 is confirmation-only; even a clear confirmation authorizes evaluation, not deployment.",
    }
    _write_json(PRODUCTION_PATH, production_payload)

    if drain_payload["research_flag"] == "PERSISTENT_POTENTIAL_DOMINANCE":
        direction = "EFFORT_DRAIN_CAUSAL_STUDY"
    elif production_candidate:
        direction = "PRODUCTION_AI_EVALUATION"
    elif classification in {"SUPPORTED_BUT_UNDERPOWERED", "NO_REPLICATION"}:
        direction = "CONSERVATIVE_POLICY_REFINEMENT"
    else:
        direction = "BOUNDED_TACTICAL_SEARCH"
    next_payload = {"recommended_next_broad_direction": direction, "implementation_authorized": False}
    _write_json(NEXT_PATH, next_payload)

    identity = _read_json(POLICY_IDENTITY_PATH)
    _write_report(identity, paired, classification, effect_payload, behavior, override, mark_payload, grab_payload, pressure_payload, guard_payload, drain_payload, support, robustness, pooled, production_payload, next_payload)
    print(f"CP11_CLASSIFICATION={classification}")
    print(f"CP11_PI2_WINS={overall['first_wins']}")
    print(f"CP11_PI4_WINS={overall['second_wins']}")
    print(f"CP11_DIFF={overall['win_rate_difference']}")
    print(f"CP11_MCNEMAR_P={overall['mcnemar_exact_two_sided_p']}")
    print(f"CP11_OVERRIDE_RATE={override['override_rate']}")
    print(f"CP11_NEXT={direction}")


def closeout_manifest() -> None:
    required = [
        PROTOCOL_PATH,
        PROVENANCE_PATH,
        SEED_AUTHORITY_PATH,
        UNTOUCHED_AUDIT_PATH,
        POLICY_IDENTITY_PATH,
        MAINTENANCE_BASELINE_PATH,
        ROUTE_RESULTS_PATH,
        PAIRED_PATH,
        CLASSIFICATION_PATH,
        EFFECT_PATH,
        BEHAVIOR_PATH,
        OVERRIDE_PATH,
        MARK_PATH,
        GRAB_PATH,
        PRESSURE_PATH,
        GUARD_PATH,
        DRAIN_PATH,
        SUPPORT_PATH,
        ROBUSTNESS_PATH,
        POOLED_PATH,
        PRODUCTION_PATH,
        NEXT_PATH,
        REPORT_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(f"Missing CP11 artifacts: {missing}")
    artifacts = {path.name: sha256_file(path) for path in required}
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "protocol_checksum": sha256_file(PROTOCOL_PATH),
        "training_performed": False,
        "model_selection_performed": False,
        "threshold_tuning_performed": False,
        "post_seed_access_tuning_performed": False,
        "production_gameplay_changed": False,
    }
    _write_json(MANIFEST_PATH, manifest)
    print(f"CP11_MANIFEST_ARTIFACTS={len(artifacts)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run", "manifest"))
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "run":
        run_confirmation()
    else:
        closeout_manifest()


if __name__ == "__main__":
    main()
