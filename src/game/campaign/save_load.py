"""JSON save/load for company state."""

from __future__ import annotations

import json
from pathlib import Path

from game.campaign.company import CompanyState
from game.core.events import LoadEvent, SaveEvent


def save_company(company: CompanyState, path: Path) -> SaveEvent:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(company.to_dict(), file, indent=2, sort_keys=True)
    return SaveEvent(message=f"Saved {company.name}.", path=str(path))


def load_company(path: Path) -> tuple[CompanyState, LoadEvent]:
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError("Save file must contain a JSON object.")
    company = CompanyState.from_dict(raw)
    return company, LoadEvent(message=f"Loaded {company.name}.", path=str(path))
