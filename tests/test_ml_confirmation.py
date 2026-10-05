from __future__ import annotations

import json
import subprocess
import sys

from game.campaign.company import create_new_company
from game.combat.enemy_learning import create_hero_policy, run_enemy_learning_episode
from game.core.rng import GameRng
from game.data.loaders import load_game_definitions
from game.dev.ml_cp5 import (
    CONFIRMATION_SEEDS,
    CP4_ROOT,
    PI1_MODEL_SHA,
    PI2_MODEL_SHA,
    _load_frozen_policies,
    _old_seed_integrity_probe,
    _protocol_payload,
)
from game.expedition.cave import create_encounter_combat
from game.ml.confirmation_audit import (
    AuditRecordingPolicy,
    aggregate_cases,
    attach_decision_events,
    classify_advantage,
    classify_equivalence,
    classify_reduced_behavior,
    paired_policy_comparison,
    stable_disagreement_sample,
    summarize_case,
)


def _row(case_id: str, win: int) -> dict[str, object]:
    return {"case_id": case_id, "enemy_win": win}


def test_paired_comparison_preserves_transition_matrix_and_difference() -> None:
    first = [_row("a", 0), _row("b", 1), _row("c", 0), _row("d", 1)]
    second = [_row("a", 1), _row("b", 1), _row("c", 0), _row("d", 0)]
    result = paired_policy_comparison(first, second)
    assert result["both_loss"] == 1
    assert result["both_win"] == 1
    assert result["baseline_only_win"] == 1
    assert result["comparison_only_win"] == 1
    assert result["discordant_pair_count"] == 2
    assert result["win_rate_difference"] == 0.0
    assert result["mcnemar_exact_two_sided_p"] == 1.0


def test_predeclared_classification_rules_are_directional_and_equivalence_bounded() -> None:
    strong = {
        "win_rate_difference": 0.2,
        "paired_difference_ci_95": [0.1, 0.3],
        "mcnemar_exact_two_sided_p": 0.01,
    }
    equivalent = {
        "win_rate_difference": 0.01,
        "paired_difference_ci_95": [-0.04, 0.06],
        "mcnemar_exact_two_sided_p": 0.8,
    }
    assert classify_advantage(strong) == "CONFIRMED"
    assert classify_equivalence(equivalent, margin=0.10) == "APPROXIMATELY_EQUIVALENT"


def test_reduced_behavior_rule_requires_both_learned_policies() -> None:
    assert classify_reduced_behavior(2.0, 1.0, 1.4) == "CONFIRMED"
    assert classify_reduced_behavior(2.0, 1.0, 1.8) == "DIRECTIONALLY_REPLICATED"
    assert classify_reduced_behavior(2.0, 1.0, 2.2) == "INCONCLUSIVE"
    assert classify_reduced_behavior(2.0, 2.1, 2.2) == "NOT_REPLICATED"


def test_disagreement_sampling_is_stable_and_outcome_blind() -> None:
    rows = []
    for group in ("in_distribution", "tactical", "company_survival"):
        for index in range(5):
            rows.append(
                {
                    "group": group,
                    "trajectory_policy": "pi0_heuristic",
                    "encounter_id": "shallow_cave",
                    "hero_policy_id": "naive",
                    "seed": 20001 + index,
                    "action_index": index,
                    "state_digest": f"state-{group}-{index}",
                    "choices": {
                        "pi0_heuristic": "a|h1",
                        "pi1_cp3": "b|h1",
                        "pi2_cp4": "a|h1",
                    },
                }
            )
    first = stable_disagreement_sample(rows, per_group=2)
    second = stable_disagreement_sample(list(reversed(rows)), per_group=2)
    assert [(row["sample_hash"], row["group"]) for row in first] == [
        (row["sample_hash"], row["group"]) for row in second
    ]
    assert len(first) == 6


def test_frozen_policy_artifacts_verify_without_confirmation_access() -> None:
    policies, identities = _load_frozen_policies()
    assert tuple(policies) == ("pi0_heuristic", "pi1_cp3", "pi2_cp4")
    assert identities["pi1_cp3"]["model_checksum"] == PI1_MODEL_SHA
    assert identities["pi2_cp4"]["model_checksum"] == PI2_MODEL_SHA
    assert CONFIRMATION_SEEDS[0] == 30001
    assert CONFIRMATION_SEEDS[-1] == 30128
    cp4_manifest = json.loads((CP4_ROOT / "final_manifest.json").read_text(encoding="utf-8"))
    assert cp4_manifest["reserved_future_confirmation_evaluated"] is False


def test_previously_used_seed_replay_is_deterministic() -> None:
    definitions = load_game_definitions()
    policies, _identities = _load_frozen_policies()
    result = _old_seed_integrity_probe(definitions, policies)
    assert result["seed"] == 20001
    assert result["deterministic_repeat_match"] is True


def test_audit_wrapper_observes_legal_frozen_choices_on_old_seed() -> None:
    definitions = load_game_definitions()
    policies, _identities = _load_frozen_policies()
    state = create_encounter_combat(create_new_company(definitions), definitions, "shallow_cave")
    recorder = AuditRecordingPolicy(executed_name="pi1_cp3", policies=policies)
    episode = run_enemy_learning_episode(
        state,
        definitions,
        GameRng(20001),
        policy=recorder,
        max_rounds=20,
        encounter_id="shallow_cave",
        encounter_name="Shallow Cave",
        hero_policy=create_hero_policy("naive", encounter_id="shallow_cave", seed=20001),
        enemy_wait_mode="none",
        enemy_movement_mode="recovery_only",
    )
    attach_decision_events(recorder.decisions, episode)
    assert recorder.decisions
    assert all(decision["choices"]["pi1_cp3"] for decision in recorder.decisions)
    assert all(decision["state_digest"] for decision in recorder.decisions)
    case = summarize_case(
        case_id="old",
        group="in_distribution",
        encounter_id="shallow_cave",
        hero_policy_id="naive",
        seed=20001,
        policy_name="pi1_cp3",
        state=state,
        episode=episode,
        decisions=recorder.decisions,
    )
    summary = aggregate_cases([case])
    assert summary["case_count"] == 1
    assert summary["median_final_hero_hp"] == case["final_hero_hp"]


def test_protocol_declares_confirmation_without_opening_it() -> None:
    _policies, identities = _load_frozen_policies()
    protocol = _protocol_payload(identities)
    assert protocol["confirmation_accessed"] is False
    assert protocol["post_confirmation_tuning_performed"] is False
    assert protocol["confirmation_seeds"] == [30001, 30128]
    assert protocol["evaluation"]["primary_endpoint"] == "company_survival enemy win rate"
    assert protocol["evaluation"]["counterfactual_spot_audit"]["per_group"] == 8


def test_ordinary_game_import_does_not_eagerly_import_torch() -> None:
    probe = subprocess.run(
        [sys.executable, "-c", "import sys; import game.main; print('torch' in sys.modules)"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "False"
