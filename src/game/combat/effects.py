"""Typed combat effect event helpers."""

from __future__ import annotations

from game.core.events import CombatEffectEvent


def effort_delta_event(
    *,
    actor_id: str,
    actor_name: str,
    delta: int,
    before: int,
    after: int,
    source_kind: str,
    source_id: str,
    source_label: str = "",
) -> CombatEffectEvent:
    if delta > 0:
        amount = delta
        prefix = f"{source_label} " if source_label else ""
        message = f"{prefix}restores {amount} Effort to {actor_name}: {before} -> {after}."
    else:
        amount = abs(delta)
        message = f"{actor_name} loses {amount} Effort: {before} -> {after}."
    return CombatEffectEvent(
        message=message,
        actor_id=actor_id,
        target_id=actor_id,
        effect_type="resource",
        resource="effort",
        label=f"EF {delta:+d}",
        delta=delta,
        before=before,
        after=after,
        source_kind=source_kind,
        source_id=source_id,
        emphasis="good" if delta > 0 else "bad",
    )


def tag_effect_event(
    *,
    actor_id: str,
    actor_name: str,
    tag: str,
    added: bool,
    source_kind: str,
    source_id: str,
    label: str = "",
) -> CombatEffectEvent:
    display = label or tag.replace("_", " ").title()
    verb = "gains" if added else "loses"
    return CombatEffectEvent(
        message=f"{actor_name} {verb} {display}.",
        actor_id=actor_id,
        target_id=actor_id,
        effect_type="tag",
        tag=tag,
        label=display if added else f"-{display}",
        delta=1 if added else -1,
        source_kind=source_kind,
        source_id=source_id,
        emphasis="bad" if added else "good",
    )


def mitigation_effect_event(
    *,
    actor_id: str,
    actor_name: str,
    amount: int,
    source_kind: str,
    source_id: str,
    label: str = "Guard",
) -> CombatEffectEvent:
    return CombatEffectEvent(
        message=f"{actor_name}'s {label} absorbs {amount} damage.",
        actor_id=actor_id,
        target_id=actor_id,
        effect_type="mitigation",
        label=f"{label} -{amount}",
        delta=-amount,
        source_kind=source_kind,
        source_id=source_id,
        emphasis="good",
    )
