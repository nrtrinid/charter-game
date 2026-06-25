"""Save migrations for backwards-compatible campaign loading."""

from __future__ import annotations

from typing import Any

from game.campaign.company import SAVE_VERSION, STARTING_COIN


def upgrade_raw_save(raw: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a raw save payload to the current expected shape.

    This function should be safe to run on already-upgraded payloads.
    """

    # Work on a shallow copy so callers can pass shared dicts safely.
    upgraded: dict[str, Any] = dict(raw)

    save_version = upgraded.get("save_version")
    try:
        version = int(save_version) if save_version is not None else 0
    except (TypeError, ValueError):
        version = 0

    # Normalize top-level fields that older saves/tests expect.
    if upgraded.get("coin") is None:
        upgraded["coin"] = STARTING_COIN

    # Reports / accounting fields: keep defaults consistent with tests.
    reports = upgraded.get("expedition_reports")
    if isinstance(reports, list):
        normalized_reports: list[Any] = []
        for report in reports:
            if not isinstance(report, dict):
                continue
            report_up = dict(report)
            for key in ("start_coin", "end_coin", "coin_gained"):
                if report_up.get(key) is None:
                    report_up[key] = 0
            normalized_reports.append(report_up)
        upgraded["expedition_reports"] = normalized_reports

    # Version discipline: we keep output shape stable unless a migration step changes it.
    upgraded["save_version"] = SAVE_VERSION if version > SAVE_VERSION else SAVE_VERSION
    return upgraded

