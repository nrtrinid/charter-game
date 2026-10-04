"""Versioned observation schemas for enemy-decision research."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

ENEMY_DECISION_FEATURE_SCHEMA_ID = "enemy-decision-candidate-v1"
ENEMY_DECISION_FEATURE_SCHEMA_VERSION = 1

ENEMY_DECISION_FEATURE_NAMES: tuple[str, ...] = (
    "formation",
    "marked_focus",
    "damage_pressure",
    "boss_pressure",
    "guard",
    "vulnerable_payoff",
    "maw_grab_setup",
    "maw_bite_payoff",
    "target_has_heal_skill",
    "target_has_recovery_skill",
    "target_has_support_skill",
    "target_has_effort_for_support",
    "target_is_support_actor",
    "target_is_backline_support",
    "grab_disrupts_support_actor",
    "grab_action_tax_value",
    "grab_target_has_not_acted",
    "grab_creates_support_denial",
    "maw_grab_high_value_support",
    "maw_grab_down_threat",
    "maw_grab_bite_expected_collapse",
    "boss_guard_package",
    "bandit_mark_collapse",
    "bandit_mark_kill_lane",
    "mark_ally_reach_count",
    "mark_expected_followup_damage",
    "mark_payoff_attacker_count",
    "mark_acts_before_target_count",
    "mark_followup_can_down",
    "mark_target_low_hp",
    "mark_target_guarded",
    "mark_target_protected",
    "bandit_marked_attack",
    "bandit_marked_payoff",
    "bandit_ignored_marked_legal",
    "effort_drain",
    "mark",
    "drag_forward",
    "role_flavor",
)


@dataclass(frozen=True)
class EnemyDecisionFeatureSchema:
    schema_id: str
    version: int
    names: tuple[str, ...]

    def encode(self, features: Mapping[str, int]) -> tuple[int, ...]:
        """Validate a candidate mapping and encode it in the contract order."""
        actual = set(features)
        expected = set(self.names)
        missing = tuple(name for name in self.names if name not in actual)
        unexpected = tuple(sorted(actual - expected))
        if missing or unexpected:
            parts: list[str] = []
            if missing:
                parts.append(f"missing={missing}")
            if unexpected:
                parts.append(f"unexpected={unexpected}")
            raise ValueError(
                f"Feature schema {self.schema_id} mismatch: " + ", ".join(parts)
            )
        return tuple(int(features[name]) for name in self.names)

    def decode(self, values: tuple[int, ...]) -> dict[str, int]:
        if len(values) != len(self.names):
            raise ValueError(
                f"Feature schema {self.schema_id} expected {len(self.names)} values, "
                f"got {len(values)}"
            )
        return dict(zip(self.names, values, strict=True))


ENEMY_DECISION_FEATURE_SCHEMA = EnemyDecisionFeatureSchema(
    schema_id=ENEMY_DECISION_FEATURE_SCHEMA_ID,
    version=ENEMY_DECISION_FEATURE_SCHEMA_VERSION,
    names=ENEMY_DECISION_FEATURE_NAMES,
)
