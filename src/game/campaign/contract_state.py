"""Contract record state for campaign saves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ContractRecordState:
    contract_id: str
    state: str = "available"
    accepted_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    last_run_id: str = ""
    rooms_scouted: int = 0
    hunt_cleared: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "state": self.state,
            "accepted_count": self.accepted_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "last_run_id": self.last_run_id,
            "rooms_scouted": self.rooms_scouted,
            "hunt_cleared": self.hunt_cleared,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        contract_id: str | None = None,
    ) -> ContractRecordState:
        return cls(
            contract_id=str(data.get("contract_id") or contract_id or ""),
            state=str(data.get("state", "available")),
            accepted_count=int(data.get("accepted_count", 0)),
            completed_count=int(data.get("completed_count", 0)),
            failed_count=int(data.get("failed_count", 0)),
            last_run_id=str(data.get("last_run_id", "")),
            rooms_scouted=int(data.get("rooms_scouted", 0)),
            hunt_cleared=bool(data.get("hunt_cleared", False)),
        )

