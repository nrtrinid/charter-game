"""CP8 analysis helpers for paired route-aware mechanic perturbations."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean, median
from typing import Any

from game.ml.confirmation_audit import mcnemar_exact, paired_difference_ci, wilson_interval

_Z95 = 1.959963984540054
EQUIVALENCE_MARGIN = 0.03
ADOPTION_ABSOLUTE_DELTA = 0.10
ADOPTION_RELATIVE_GAIN = 0.20
DOMINANCE_SELECTION_RATE = 0.90
DOMINANCE_STRATUM_RATE = 0.80
DOMINANCE_MIN_STRATA = 3


def route_case_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["case_id"]), str(row["enemy_policy"])


def route_position(index: int, length: int) -> str:
    if length <= 1:
        return "SINGLE"
    if index == 0:
        return "EARLY"
    if index == length - 1:
        return "LATE"
    return "MIDDLE"


def duration_band(rounds: int) -> str:
    if rounds <= 5:
        return "SHORT"
    if rounds <= 15:
        return "MEDIUM"
    return "LONG"


def factual_band(value: float) -> str:
    if value <= 0.50:
        return "LOW"
    if value <= 0.75:
        return "MID"
    return "HIGH"


def _mean_ci(values: Sequence[float]) -> list[float]:
    samples = [float(value) for value in values]
    if not samples:
        return [0.0, 0.0]
    center = mean(samples)
    if len(samples) == 1:
        return [center, center]
    variance = sum((value - center) ** 2 for value in samples) / (len(samples) - 1)
    half = _Z95 * math.sqrt(variance / len(samples))
    return [center - half, center + half]


def _binary_effect(first: Sequence[int], second: Sequence[int]) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError("Paired binary arrays differ in length.")
    differences = [int(b) - int(a) for a, b in zip(first, second, strict=True)]
    a_only = sum(bool(a) and not bool(b) for a, b in zip(first, second, strict=True))
    b_only = sum(not bool(a) and bool(b) for a, b in zip(first, second, strict=True))
    both = sum(bool(a) and bool(b) for a, b in zip(first, second, strict=True))
    neither = len(first) - a_only - b_only - both
    first_wins = sum(first)
    second_wins = sum(second)
    return {
        "case_count": len(first),
        "baseline_wins": first_wins,
        "perturbed_wins": second_wins,
        "baseline_rate": first_wins / len(first) if first else None,
        "perturbed_rate": second_wins / len(second) if second else None,
        "baseline_wilson_95": wilson_interval(first_wins, len(first)) if first else None,
        "perturbed_wilson_95": wilson_interval(second_wins, len(second)) if second else None,
        "difference": mean(differences) if differences else None,
        "paired_difference_ci_95": paired_difference_ci(differences) if differences else None,
        "mcnemar_exact_two_sided_p": mcnemar_exact(a_only, b_only) if differences else None,
        "transition_matrix": {
            "both_non_enemy": neither,
            "baseline_only_enemy": a_only,
            "perturbed_only_enemy": b_only,
            "both_enemy": both,
        },
    }


def _continuous_effect(first: Sequence[float], second: Sequence[float]) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError("Paired continuous arrays differ in length.")
    differences = [float(b) - float(a) for a, b in zip(first, second, strict=True)]
    return {
        "case_count": len(differences),
        "baseline_mean": mean(first) if first else None,
        "perturbed_mean": mean(second) if second else None,
        "mean_difference": mean(differences) if differences else None,
        "difference_ci_95": _mean_ci(differences) if differences else None,
    }


def paired_route_effect(
    baseline_rows: Sequence[Mapping[str, Any]],
    condition_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    first = {route_case_key(row): row for row in baseline_rows}
    second = {route_case_key(row): row for row in condition_rows}
    if first.keys() != second.keys():
        raise ValueError("Paired route perturbation coverage differs from baseline.")
    keys = sorted(first)
    valid_keys = [
        key
        for key in keys
        if not bool(first[key]["route_censored"])
        and not bool(second[key]["route_censored"])
        and not bool(first[key]["route_invalid"])
        and not bool(second[key]["route_invalid"])
    ]
    enemy = _binary_effect(
        [int(first[key]["route_failed_enemy_victory"]) for key in valid_keys],
        [int(second[key]["route_failed_enemy_victory"]) for key in valid_keys],
    )
    completed = _binary_effect(
        [int(first[key]["route_completed"]) for key in valid_keys],
        [int(second[key]["route_completed"]) for key in valid_keys],
    )
    continuous: dict[str, Any] = {}
    scalar_paths = {
        "final_hp": ("hp_total",),
        "final_hp_fraction": ("hp_fraction",),
        "final_effort": ("effort_total",),
        "final_mortal_wounds": ("mortal_wounds_total",),
        "final_downed": ("downed_count",),
        "final_living": ("active_hero_count",),
    }
    for name, path in scalar_paths.items():
        continuous[name] = _continuous_effect(
            [float(first[key]["final_company_state"][path[0]]) for key in valid_keys],
            [float(second[key]["final_company_state"][path[0]]) for key in valid_keys],
        )
    continuous["encounters_completed"] = _continuous_effect(
        [float(first[key]["encounters_completed"]) for key in valid_keys],
        [float(second[key]["encounters_completed"]) for key in valid_keys],
    )
    continuous["total_rounds"] = _continuous_effect(
        [float(first[key]["total_rounds"]) for key in valid_keys],
        [float(second[key]["total_rounds"]) for key in valid_keys],
    )
    stopped_first = Counter(str(first[key]["stopped_at_encounter_id"]) for key in valid_keys)
    stopped_second = Counter(str(second[key]["stopped_at_encounter_id"]) for key in valid_keys)
    return {
        "paired_case_count": len(keys),
        "paired_valid_count": len(valid_keys),
        "excluded_for_censoring_or_invalid": len(keys) - len(valid_keys),
        "enemy_victory": enemy,
        "route_completion": completed,
        "continuous": continuous,
        "baseline_failure_encounters": dict(sorted(stopped_first.items())),
        "perturbed_failure_encounters": dict(sorted(stopped_second.items())),
    }


def _encounter_key(route: Mapping[str, Any], encounter: Mapping[str, Any]) -> tuple[str, str, int]:
    return route_case_key(route)[0], route_case_key(route)[1], int(encounter["encounter_index"])


def _mechanic_fields(mechanic: str) -> tuple[str, str]:
    return {
        "MARK": ("legal_mark", "selected_mark"),
        "GRAB": ("legal_grab", "selected_grab"),
        "GUARD": ("legal_guard", "selected_guard"),
        "EFFORT_DRAIN": ("legal_effort_drain", "selected_effort_drain"),
    }[mechanic]


def _encounter_has_mechanic_opportunity(encounter: Mapping[str, Any], mechanic: str) -> bool:
    legal, _selected = _mechanic_fields(mechanic)
    return any(bool(row[legal]) for row in encounter["decision_audit"])


def paired_local_effect(
    baseline_rows: Sequence[Mapping[str, Any]],
    condition_rows: Sequence[Mapping[str, Any]],
    *,
    mechanic: str,
) -> dict[str, Any]:
    first_routes = {route_case_key(row): row for row in baseline_rows}
    second_routes = {route_case_key(row): row for row in condition_rows}
    if first_routes.keys() != second_routes.keys():
        raise ValueError("Local perturbation route coverage differs.")
    matched: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    reachability_changed = 0
    for key in sorted(first_routes):
        first_enc = {int(row["encounter_index"]): row for row in first_routes[key]["encounters"]}
        second_enc = {int(row["encounter_index"]): row for row in second_routes[key]["encounters"]}
        for index in sorted(set(first_enc) | set(second_enc)):
            if index not in first_enc or index not in second_enc:
                reachability_changed += 1
                continue
            a, b = first_enc[index], second_enc[index]
            if not (
                _encounter_has_mechanic_opportunity(a, mechanic)
                or _encounter_has_mechanic_opportunity(b, mechanic)
            ):
                continue
            if not bool(a["production_terminal"]) or not bool(b["production_terminal"]):
                continue
            matched.append((a, b))
    first_enemy = [int(a["termination_reason"] == "ENEMY_VICTORY") for a, _ in matched]
    second_enemy = [int(b["termination_reason"] == "ENEMY_VICTORY") for _, b in matched]
    metrics: dict[str, Any] = {}
    for name, getter in {
        "hp_loss": lambda row: float((row.get("deterioration") or {}).get("hp_loss", 0)),
        "effort_loss": lambda row: float((row.get("deterioration") or {}).get("effort_loss", 0)),
        "downs": lambda row: float(row.get("downs_caused", 0)),
        "mortal_wounds": lambda row: float(row.get("mortal_wounds_caused", 0)),
        "rounds": lambda row: float(row.get("rounds", 0)),
    }.items():
        metrics[name] = _continuous_effect(
            [getter(a) for a, _ in matched], [getter(b) for _, b in matched]
        )
    return {
        "matched_relevant_encounters": len(matched),
        "reachability_changed_encounter_slots": reachability_changed,
        "enemy_victory": _binary_effect(first_enemy, second_enemy),
        "continuous": metrics,
    }


def classify_effect(effect: Mapping[str, Any]) -> str:
    difference = effect.get("difference")
    ci = effect.get("paired_difference_ci_95")
    if difference is None or ci is None:
        return "INCONCLUSIVE"
    low, high = float(ci[0]), float(ci[1])
    if float(difference) > 0 and low > 0:
        return "POSITIVE"
    if float(difference) < 0 and high < 0:
        return "NEGATIVE"
    if low >= -EQUIVALENCE_MARGIN and high <= EQUIVALENCE_MARGIN:
        return "NEUTRAL"
    return "INCONCLUSIVE"


def horizon_alignment(local: Mapping[str, Any], route: Mapping[str, Any]) -> str:
    local_sign = classify_effect(local["enemy_victory"])
    route_sign = classify_effect(route["enemy_victory"])
    if "INCONCLUSIVE" in {local_sign, route_sign}:
        return "INCONCLUSIVE"
    return f"LOCAL_{local_sign}_ROUTE_{route_sign}"


def _decision_rows(routes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for route in routes:
        for encounter in route["encounters"]:
            position = route_position(int(encounter["encounter_index"]), int(route["route_length"]))
            for decision in encounter["decision_audit"]:
                row = dict(decision)
                row.update(
                    route_id=route["route_id"],
                    case_id=route["case_id"],
                    enemy_policy=route["enemy_policy"],
                    hero_policy_id=route["hero_policy_id"],
                    root_seed=route["root_seed"],
                    encounter_id=encounter["encounter_id"],
                    encounter_index=encounter["encounter_index"],
                    route_position=position,
                    encounter_rounds=encounter["rounds"],
                )
                result.append(row)
    return result


def mechanic_usage(routes: Sequence[Mapping[str, Any]], mechanic: str) -> dict[str, Any]:
    legal_field, selected_field = _mechanic_fields(mechanic)
    rows = _decision_rows(routes)
    legal = [row for row in rows if bool(row[legal_field])]
    selected = [row for row in legal if bool(row[selected_field])]
    by_encounter: dict[str, dict[str, int | float]] = {}
    for encounter in sorted({str(row["encounter_id"]) for row in legal}):
        subset = [row for row in legal if row["encounter_id"] == encounter]
        choices = sum(bool(row[selected_field]) for row in subset)
        by_encounter[encounter] = {
            "opportunities": len(subset),
            "selections": choices,
            "rate": choices / len(subset) if subset else 0.0,
        }
    by_position: dict[str, dict[str, int | float]] = {}
    for position in ("EARLY", "MIDDLE", "LATE", "SINGLE"):
        subset = [row for row in legal if row["route_position"] == position]
        choices = sum(bool(row[selected_field]) for row in subset)
        by_position[position] = {
            "opportunities": len(subset),
            "selections": choices,
            "rate": choices / len(subset) if subset else 0.0,
        }
    return {
        "opportunities": len(legal),
        "selections": len(selected),
        "conditional_selection_rate": len(selected) / len(legal) if legal else None,
        "selection_round_mean": mean(float(row["round"]) for row in selected) if selected else None,
        "by_encounter": by_encounter,
        "by_route_position": by_position,
    }


def adoption_change(baseline: Mapping[str, Any], perturbed: Mapping[str, Any]) -> dict[str, Any]:
    first = baseline.get("conditional_selection_rate")
    second = perturbed.get("conditional_selection_rate")
    if first is None or second is None:
        return {"interpretable": False}
    absolute = float(second) - float(first)
    relative = absolute / float(first) if float(first) > 0 else (math.inf if absolute > 0 else 0.0)
    material = absolute >= ADOPTION_ABSOLUTE_DELTA or relative >= ADOPTION_RELATIVE_GAIN
    return {
        "interpretable": True,
        "baseline_rate": first,
        "perturbed_rate": second,
        "absolute_change": absolute,
        "relative_change": relative,
        "material_increase": material,
    }


def binary_entropy(rate: float | None) -> float | None:
    if rate is None:
        return None
    if rate <= 0.0 or rate >= 1.0:
        return 0.0
    return -(rate * math.log2(rate) + (1.0 - rate) * math.log2(1.0 - rate))


def strategy_space(
    baseline_routes: Sequence[Mapping[str, Any]],
    condition_routes: Sequence[Mapping[str, Any]],
    mechanic: str,
) -> dict[str, Any]:
    legal_field, selected_field = _mechanic_fields(mechanic)
    first = _decision_rows(baseline_routes)
    second = _decision_rows(condition_routes)
    first_legal = [row for row in first if bool(row[legal_field])]
    second_legal = [row for row in second if bool(row[legal_field])]
    first_rate = (
        sum(bool(row[selected_field]) for row in first_legal) / len(first_legal)
        if first_legal
        else None
    )
    second_rate = (
        sum(bool(row[selected_field]) for row in second_legal) / len(second_legal)
        if second_legal
        else None
    )
    first_by_state = {
        (str(row["state_digest"]), str(row["enemy_id"])): bool(row[selected_field])
        for row in first_legal
    }
    second_by_state = {
        (str(row["state_digest"]), str(row["enemy_id"])): bool(row[selected_field])
        for row in second_legal
    }
    common = sorted(first_by_state.keys() & second_by_state.keys())
    toggled_to_mechanic = sum(not first_by_state[key] and second_by_state[key] for key in common)
    toggled_from_mechanic = sum(first_by_state[key] and not second_by_state[key] for key in common)
    return {
        "baseline_binary_entropy": binary_entropy(first_rate),
        "perturbed_binary_entropy": binary_entropy(second_rate),
        "exact_common_decision_states": len(common),
        "states_toggled_to_mechanic": toggled_to_mechanic,
        "states_toggled_from_mechanic": toggled_from_mechanic,
        "warning": "higher entropy is diagnostic only, not intrinsically better",
    }


def dominance_audit(routes: Sequence[Mapping[str, Any]], mechanic: str) -> dict[str, Any]:
    legal_field, selected_field = _mechanic_fields(mechanic)
    rows = [row for row in _decision_rows(routes) if bool(row[legal_field])]
    selected = sum(bool(row[selected_field]) for row in rows)
    rate = selected / len(rows) if rows else 0.0
    strata: dict[str, Any] = {}
    dominant_strata = 0
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['encounter_id']}|{row['route_position']}"].append(row)
    for name, subset in sorted(grouped.items()):
        selections = sum(bool(row[selected_field]) for row in subset)
        stratum_rate = selections / len(subset)
        strata[name] = {"opportunities": len(subset), "rate": stratum_rate}
        dominant_strata += int(len(subset) >= 10 and stratum_rate >= DOMINANCE_STRATUM_RATE)
    collapse = (
        len(rows) >= 20
        and rate >= DOMINANCE_SELECTION_RATE
        and dominant_strata >= DOMINANCE_MIN_STRATA
    )
    return {
        "opportunities": len(rows),
        "selections": selected,
        "rate": rate,
        "dominant_strata": dominant_strata,
        "strata": strata,
        "potential_strategy_collapse": collapse,
    }


def opportunity_cost(routes: Sequence[Mapping[str, Any]], mechanic: str) -> dict[str, Any]:
    _legal, selected_field = _mechanic_fields(mechanic)
    rows = [row for row in _decision_rows(routes) if bool(row[selected_field])]
    reconstructable = [row for row in rows if row.get("best_direct_alternative") is not None]
    deltas = [
        float(row["best_direct_alternative"]["projected_damage_max"])
        - float(row["chosen_projected_damage_max"])
        for row in reconstructable
    ]
    alternatives = Counter(
        str(row["best_direct_alternative"]["skill_id"]) for row in reconstructable
    )
    return {
        "selected_mechanic_actions": len(rows),
        "reconstructable_direct_alternative_count": len(reconstructable),
        "mean_immediate_projected_damage_given_up": mean(deltas) if deltas else None,
        "median_immediate_projected_damage_given_up": median(deltas) if deltas else None,
        "alternative_skill_counts": dict(alternatives.most_common()),
    }


def state_niches(routes: Sequence[Mapping[str, Any]], mechanic: str) -> dict[str, Any]:
    legal_field, selected_field = _mechanic_fields(mechanic)
    legal = [row for row in _decision_rows(routes) if bool(row[legal_field])]
    grouped: dict[str, dict[str, Any]] = {}
    for dimension, value_fn in {
        "party_hp_fraction": lambda row: factual_band(
            float(row["party_hp_total"]) / max(1.0, float(row["party_max_hp_total"]))
        ),
        "party_effort_fraction": lambda row: factual_band(
            float(row["party_effort_total"]) / max(1.0, float(row["party_max_effort_total"]))
        ),
        "marked_state": lambda row: (
            "MARKED_PRESENT" if int(row["marked_hero_count"]) > 0 else "NO_MARK"
        ),
        "route_position": lambda row: str(row["route_position"]),
        "encounter": lambda row: str(row["encounter_id"]),
    }.items():
        buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in legal:
            buckets[value_fn(row)].append(row)
        grouped[dimension] = {
            key: {
                "opportunities": len(items),
                "selections": sum(bool(item[selected_field]) for item in items),
                "rate": sum(bool(item[selected_field]) for item in items) / len(items),
            }
            for key, items in sorted(buckets.items())
        }
    return grouped


def encounter_horizon_analysis(
    baseline_routes: Sequence[Mapping[str, Any]],
    condition_routes: Sequence[Mapping[str, Any]],
    mechanic: str,
) -> dict[str, Any]:
    baseline_duration: dict[tuple[str, str, int], str] = {}
    for route in baseline_routes:
        for encounter in route["encounters"]:
            baseline_duration[_encounter_key(route, encounter)] = duration_band(
                int(encounter["rounds"])
            )
    legal_field, selected_field = _mechanic_fields(mechanic)
    result: dict[str, Any] = {}
    for label, routes in (("baseline", baseline_routes), ("perturbed", condition_routes)):
        by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for route in routes:
            for encounter in route["encounters"]:
                key = _encounter_key(route, encounter)
                if key not in baseline_duration:
                    continue
                for decision in encounter["decision_audit"]:
                    if bool(decision[legal_field]):
                        by_band[baseline_duration[key]].append(decision)
        result[label] = {
            band: {
                "opportunities": len(rows),
                "selections": sum(bool(row[selected_field]) for row in rows),
                "rate": sum(bool(row[selected_field]) for row in rows) / len(rows),
            }
            for band, rows in sorted(by_band.items())
        }
    return result


def policy_ordering(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for policy in ("pi0_heuristic", "pi1_cp3", "pi2_cp4"):
        subset = [
            row
            for row in rows
            if row["enemy_policy"] == policy and row["hero_policy_id"] == "company_survival"
        ]
        valid = [row for row in subset if not row["route_censored"] and not row["route_invalid"]]
        wins = sum(bool(row["route_failed_enemy_victory"]) for row in valid)
        result[policy] = {
            "valid": len(valid),
            "wins": wins,
            "win_rate": wins / len(valid) if valid else None,
        }
    return result


def causal_classification(
    mechanic: str,
    condition_summaries: Mapping[str, Mapping[str, Any]],
    *,
    observability: Mapping[str, str],
    baseline_pi0_usage: Mapping[str, Any],
    baseline_pi2_usage: Mapping[str, Any],
) -> str:
    alignments = {str(row["horizon_alignment"]) for row in condition_summaries.values()}
    if any(
        value in {"LOCAL_POSITIVE_ROUTE_NEGATIVE", "LOCAL_NEGATIVE_ROUTE_POSITIVE"}
        for value in alignments
    ):
        return "LOCAL_ROUTE_TRADEOFF"
    observable_positive = False
    route_only_positive = False
    local_only_positive = False
    nonobservable_positive = False
    for condition_id, row in condition_summaries.items():
        route_sign = classify_effect(row["route_effect"]["enemy_victory"])
        local_sign = classify_effect(row["local_effect"]["enemy_victory"])
        adoption = row["adoption_change"]
        if route_sign == "POSITIVE" and local_sign == "NEUTRAL":
            route_only_positive = True
        if route_sign == "NEUTRAL" and local_sign == "POSITIVE":
            local_only_positive = True
        if route_sign == "POSITIVE" and observability[condition_id] == "NOT_OBSERVABLE":
            nonobservable_positive = True
        if (
            route_sign == "POSITIVE"
            and observability[condition_id] == "FULLY_OBSERVABLE_TO_FROZEN_POLICY"
            and bool(adoption.get("material_increase"))
        ):
            observable_positive = True
    if observable_positive:
        return "CURRENT_INCENTIVE_LIKELY_WEAK"
    if route_only_positive:
        return "CONTEXTUAL_ROUTE_VALUE"
    if local_only_positive:
        return "CONTEXTUAL_LOCAL_VALUE"
    first = baseline_pi0_usage.get("conditional_selection_rate")
    second = baseline_pi2_usage.get("conditional_selection_rate")
    if first is not None and second is not None and float(first) > 0:
        if float(second) <= 0.5 * float(first):
            if not any(
                classify_effect(row["route_effect"]["enemy_victory"]) == "POSITIVE"
                and observability[condition_id] == "FULLY_OBSERVABLE_TO_FROZEN_POLICY"
                for condition_id, row in condition_summaries.items()
            ):
                return "HEURISTIC_OVERVALUATION"
    if nonobservable_positive:
        return "FROZEN_POLICY_OBSERVABILITY_LIMITATION"
    return "MIXED_INCONCLUSIVE"
