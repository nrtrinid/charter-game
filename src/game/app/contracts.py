"""Shared contract board and acceptance rules."""

from __future__ import annotations

from game.campaign.company import CompanyState
from game.content.definitions import GameDefinitions
from game.data.schemas import ContractDefinition
from game.expedition.generated_maze import (
    GENERATED_MAZE_CONTRACT_ID,
    GENERATED_MAZE_HUNT_CONTRACT_ID,
    GENERATED_MAZE_REPEATABLE_HUNT_CONTRACT_ID,
    GENERATED_MAZE_REPEATABLE_SCOUT_CONTRACT_ID,
)

GENERATED_MAZE_EXCLUSIVE_CONTRACT_IDS = frozenset(
    {
        GENERATED_MAZE_CONTRACT_ID,
        GENERATED_MAZE_HUNT_CONTRACT_ID,
        GENERATED_MAZE_REPEATABLE_SCOUT_CONTRACT_ID,
        GENERATED_MAZE_REPEATABLE_HUNT_CONTRACT_ID,
    }
)


def contract_board_ids(
    company: CompanyState,
    definitions: GameDefinitions,
) -> tuple[str, ...]:
    return tuple(
        contract.id
        for contract in sorted(
            definitions.contracts.values(),
            key=lambda contract: (
                contract.board_order is None,
                contract.board_order or 0,
                contract.name,
            ),
        )
        if contract.board_order is not None
    )


def contract_is_repeatable(contract: ContractDefinition) -> bool:
    return "repeatable" in contract.tags


def contract_is_posted(
    company: CompanyState,
    definitions: GameDefinitions,
    contract: ContractDefinition,
) -> bool:
    record = company.contract_records.get(contract.id)
    record_state = record.state if record is not None else ""
    if (
        record_state in {"active", "completed", "repeatable_completed"}
        or contract.id in company.active_contract_ids
        or contract.id in company.completed_contract_ids
    ):
        return True
    if contract.board_order is None:
        return False
    return not _missing_posting_reason(company, definitions, contract)


def contract_unavailable_reason(
    company: CompanyState,
    definitions: GameDefinitions,
    contract_id: str,
) -> str:
    contract = definitions.contracts[contract_id]
    record = company.contract_records.get(contract_id)
    record_state = record.state if record is not None else ""
    repeatable = contract_is_repeatable(contract)
    if not repeatable and (
        record_state == "completed" or contract_id in company.completed_contract_ids
    ):
        return "Contract already completed."
    if record_state == "active" or contract_id in company.active_contract_ids:
        return "Contract already active."
    if not contract_is_posted(company, definitions, contract):
        posting_reason = _missing_posting_reason(company, definitions, contract)
        if posting_reason:
            return posting_reason
        return "That contract is not posted on the Haven board yet."
    missing_reason = _missing_requirements_reason(company, definitions, contract)
    if missing_reason:
        return missing_reason
    exclusive_reason = _exclusive_contract_reason(company, definitions, contract)
    if exclusive_reason:
        return exclusive_reason
    return ""


def contract_board_state(
    company: CompanyState,
    definitions: GameDefinitions,
    contract_id: str,
) -> tuple[str, str]:
    contract = definitions.contracts[contract_id]
    record = company.contract_records.get(contract_id)
    record_state = record.state if record is not None else ""
    repeatable = contract_is_repeatable(contract)
    if not repeatable and (
        record_state == "completed" or contract_id in company.completed_contract_ids
    ):
        return "completed", "Contract already completed."
    if record_state == "active" or contract_id in company.active_contract_ids:
        return "active", "Contract already active."
    posting_reason = _missing_posting_reason(company, definitions, contract)
    if posting_reason:
        return "locked", posting_reason
    missing_reason = _missing_requirements_reason(company, definitions, contract)
    if missing_reason:
        return "locked", missing_reason
    exclusive_reason = _exclusive_contract_reason(company, definitions, contract)
    if exclusive_reason:
        return "locked", exclusive_reason
    return "available", ""


def _missing_posting_reason(
    company: CompanyState,
    definitions: GameDefinitions,
    contract: ContractDefinition,
) -> str:
    for required_contract_id in contract.posted_after_completed_contracts:
        if required_contract_id not in company.completed_contract_ids:
            return contract.locked_reason or _complete_contract_reason(
                definitions,
                required_contract_id,
            )
    for breach_id in contract.posted_after_known_breaches:
        if breach_id not in company.known_breaches:
            return contract.locked_reason or _find_breach_reason(definitions, breach_id)
    for flag_id in contract.posted_after_flags:
        if not company.flags.get(flag_id):
            return contract.locked_reason or _file_proof_reason(flag_id)
    return ""


def _missing_requirements_reason(
    company: CompanyState,
    definitions: GameDefinitions,
    contract: ContractDefinition,
) -> str:
    for required_contract_id in contract.requires_completed_contracts:
        if required_contract_id not in company.completed_contract_ids:
            return contract.locked_reason or _complete_contract_reason(
                definitions,
                required_contract_id,
            )
    for breach_id in contract.requires_known_breaches:
        if breach_id not in company.known_breaches:
            return contract.locked_reason or _find_breach_reason(definitions, breach_id)
    return ""


def _exclusive_contract_reason(
    company: CompanyState,
    definitions: GameDefinitions,
    contract: ContractDefinition,
) -> str:
    if contract.id not in GENERATED_MAZE_EXCLUSIVE_CONTRACT_IDS:
        return ""
    active_exclusive_ids = (
        GENERATED_MAZE_EXCLUSIVE_CONTRACT_IDS & company.active_contract_ids
    )
    conflicting_contract_ids = active_exclusive_ids - {contract.id}
    if not conflicting_contract_ids:
        return ""
    conflicting_id = sorted(conflicting_contract_ids)[0]
    conflicting_contract = definitions.contracts.get(conflicting_id)
    if conflicting_contract is None:
        return "Finish the active breach contract first."
    return f"Finish {conflicting_contract.name} first."


def _complete_contract_reason(
    definitions: GameDefinitions,
    contract_id: str,
) -> str:
    contract = definitions.contracts.get(contract_id)
    if contract is None:
        return "Complete the required contract first."
    return f"Complete {contract.name} first."


def _find_breach_reason(definitions: GameDefinitions, breach_id: str) -> str:
    location = definitions.locations.get(breach_id)
    if location is None:
        return "Find the required breach first."
    return f"Find {location.name} first."


def _file_proof_reason(flag_id: str) -> str:
    if flag_id == "cave_relic_filed":
        return "File the cave relic with Haven's relic clerk first."
    return "File the required proof with Haven first."
