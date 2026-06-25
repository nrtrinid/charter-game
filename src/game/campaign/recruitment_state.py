"""Recruitment state for campaign saves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecruitmentOfferState:
    name: str
    class_id: str
    background: str = ""
    motive: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class_id": self.class_id,
            "background": self.background,
            "motive": self.motive,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecruitmentOfferState:
        return cls(
            name=str(data["name"]),
            class_id=str(data["class_id"]),
            background=str(data.get("background", "")),
            motive=str(data.get("motive", "")),
        )


@dataclass
class RecruitmentState:
    current_offers: list[RecruitmentOfferState] = field(default_factory=list)
    refresh_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_offers": [offer.to_dict() for offer in self.current_offers],
            "refresh_count": self.refresh_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecruitmentState:
        return cls(
            current_offers=[
                RecruitmentOfferState.from_dict(offer)
                for offer in data.get("current_offers", [])
                if isinstance(offer, dict)
            ],
            refresh_count=int(data.get("refresh_count", 0)),
        )

