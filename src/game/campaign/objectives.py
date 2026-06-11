"""Derived campaign objective spine for the v0.1 charter loop."""

from __future__ import annotations

from dataclasses import dataclass

from game.campaign.company import CompanyState, ContractRecordState

BLACKWOOD_ROAD_CHARTER_ID = "blackwood_road_charter"
SHALLOW_CAVE_BREACH_SCOUT_ID = "shallow_cave_breach_scout"
BREACH_STALKER_HUNT_ID = "shallow_cave_breach_hunt"
SHALLOW_CAVE_BREACH_ID = "shallow_cave_breach"


@dataclass(frozen=True)
class CampaignObjectiveStepView:
    contract_id: str
    name: str
    state: str


@dataclass(frozen=True)
class CampaignObjectiveView:
    title: str
    summary: str
    status: str
    next_step: str
    chapter_status: str
    progress: str = ""
    current_contract_id: str = ""
    steps: tuple[CampaignObjectiveStepView, ...] = ()


def build_campaign_objective(company: CompanyState) -> CampaignObjectiveView:
    """Derive the current Act 1 objective from existing campaign state."""

    steps = (
        CampaignObjectiveStepView(
            BLACKWOOD_ROAD_CHARTER_ID,
            "Blackwood Road Charter",
            _contract_step_state(company, BLACKWOOD_ROAD_CHARTER_ID, unlocked=True),
        ),
        CampaignObjectiveStepView(
            SHALLOW_CAVE_BREACH_SCOUT_ID,
            "Shallow Cave Breach Scout",
            _contract_step_state(
                company,
                SHALLOW_CAVE_BREACH_SCOUT_ID,
                unlocked=(
                    BLACKWOOD_ROAD_CHARTER_ID in company.completed_contract_ids
                    and SHALLOW_CAVE_BREACH_ID in company.known_breaches
                ),
            ),
        ),
        CampaignObjectiveStepView(
            BREACH_STALKER_HUNT_ID,
            "Breach Stalker Hunt",
            _contract_step_state(
                company,
                BREACH_STALKER_HUNT_ID,
                unlocked=SHALLOW_CAVE_BREACH_SCOUT_ID in company.completed_contract_ids,
            ),
        ),
    )

    if BREACH_STALKER_HUNT_ID in company.completed_contract_ids:
        return CampaignObjectiveView(
            title="Charter Review Complete",
            summary="The company has proved the breach can be scouted, hunted, and reported.",
            status="complete",
            next_step="Spend Coin, review reports, or take repeatable breach postings.",
            chapter_status="Act 1 Charter Review Complete",
            progress="3 / 3 charter objectives complete.",
            steps=steps,
        )
    if BREACH_STALKER_HUNT_ID in company.active_contract_ids:
        return CampaignObjectiveView(
            title="Complete Breach Stalker Hunt",
            summary="A marked route waits beyond the Shallow Cave Breach.",
            status="active",
            next_step="Enter the breach, clear the marked lair, and return with proof.",
            chapter_status="Act 1 Frontier Charter",
            progress=_hunt_progress(company),
            current_contract_id=BREACH_STALKER_HUNT_ID,
            steps=steps,
        )
    if SHALLOW_CAVE_BREACH_SCOUT_ID in company.completed_contract_ids:
        return CampaignObjectiveView(
            title="Accept Breach Stalker Hunt",
            summary="The first scout report has made a harder breach contract credible.",
            status="available",
            next_step="Accept Breach Stalker Hunt from the Haven contract board.",
            chapter_status="Act 1 Frontier Charter",
            progress="2 / 3 charter objectives complete.",
            current_contract_id=BREACH_STALKER_HUNT_ID,
            steps=steps,
        )
    if SHALLOW_CAVE_BREACH_SCOUT_ID in company.active_contract_ids:
        return CampaignObjectiveView(
            title="Scout Shallow Cave Breach",
            summary="The company has a paid reason to step through the breach and map it.",
            status="active",
            next_step=(
                "Chart four Maze rooms, complete one survey action, then return to the breach."
            ),
            chapter_status="Act 1 Frontier Charter",
            progress=_scout_progress(company),
            current_contract_id=SHALLOW_CAVE_BREACH_SCOUT_ID,
            steps=steps,
        )
    if BLACKWOOD_ROAD_CHARTER_ID in company.completed_contract_ids:
        if SHALLOW_CAVE_BREACH_ID in company.known_breaches:
            return CampaignObjectiveView(
                title="Accept Shallow Cave Breach Scout",
                summary="Haven now has proof that the cave hazard is a breach.",
                status="available",
                next_step="Accept Shallow Cave Breach Scout from the contract board.",
                chapter_status="Act 1 Frontier Charter",
                progress="1 / 3 charter objectives complete.",
                current_contract_id=SHALLOW_CAVE_BREACH_SCOUT_ID,
                steps=steps,
            )
        return CampaignObjectiveView(
            title="Find Shallow Cave Breach",
            summary="The road charter is done, but the breach is not yet recorded.",
            status="locked",
            next_step="Return to Shallow Cave and bring back proof of the breach.",
            chapter_status="Act 1 Frontier Charter",
            progress="1 / 3 charter objectives complete; breach proof missing.",
            current_contract_id=SHALLOW_CAVE_BREACH_SCOUT_ID,
            steps=steps,
        )
    if BLACKWOOD_ROAD_CHARTER_ID in company.active_contract_ids:
        return CampaignObjectiveView(
            title="Complete Blackwood Road Charter",
            summary="The company's first paid frontier route still needs a report.",
            status="active",
            next_step="Leave Haven, clear Shallow Cave, and return with proof.",
            chapter_status="Act 1 Frontier Charter",
            progress=_blackwood_progress(company),
            current_contract_id=BLACKWOOD_ROAD_CHARTER_ID,
            steps=steps,
        )
    return CampaignObjectiveView(
        title="Accept Blackwood Road Charter",
        summary="The company needs a first paid route before the breach work opens.",
        status="available",
        next_step="Accept Blackwood Road Charter from the contract board.",
        chapter_status="Act 1 Frontier Charter",
        progress="0 / 3 charter objectives complete.",
        current_contract_id=BLACKWOOD_ROAD_CHARTER_ID,
        steps=steps,
    )


def _contract_step_state(
    company: CompanyState,
    contract_id: str,
    *,
    unlocked: bool,
) -> str:
    if contract_id in company.completed_contract_ids:
        return "completed"
    if contract_id in company.active_contract_ids:
        return "active"
    if unlocked:
        return "available"
    return "locked"


def _blackwood_progress(company: CompanyState) -> str:
    cave_memory = company.dungeon_memory.get("shallow_cave")
    cleared_count = len(cave_memory.cleared_node_ids) if cave_memory else 0
    discovered_count = len(cave_memory.visited_node_ids) if cave_memory else 0
    if SHALLOW_CAVE_BREACH_ID in company.known_breaches:
        return "Breach proof recorded; return to Haven to close the charter."
    if cleared_count:
        return (
            f"{cleared_count} threats cleared; continue until the cave breach is recorded."
        )
    if discovered_count:
        return f"{discovered_count} cave rooms charted; find the breach proof."
    return "Route accepted; no cave progress recorded yet."


def _scout_progress(company: CompanyState) -> str:
    target_rooms = 4
    rooms = max(
        _active_generated_room_count(company),
        _contract_rooms_scouted(company.contract_records.get(SHALLOW_CAVE_BREACH_SCOUT_ID)),
    )
    rooms = min(rooms, target_rooms)
    survey_actions = _active_generated_action_count(company)
    if rooms >= target_rooms:
        if survey_actions:
            return (
                f"{rooms} / {target_rooms} Maze rooms charted; "
                "survey action complete; return to the breach."
            )
        return (
            f"{rooms} / {target_rooms} Maze rooms charted; "
            "complete one survey action."
        )
    return f"{rooms} / {target_rooms} Maze rooms charted."


def _hunt_progress(company: CompanyState) -> str:
    record = company.contract_records.get(BREACH_STALKER_HUNT_ID)
    if record and record.hunt_cleared:
        return "Marked lair cleared; return with proof."
    if _active_hunt_lair_cleared(company):
        return "Marked lair cleared; return with proof."
    rooms = _active_generated_room_count(company)
    if rooms:
        return f"{rooms} Maze rooms charted; marked lair not cleared."
    return "Marked lair not found yet."


def _contract_rooms_scouted(record: ContractRecordState | None) -> int:
    if record is None:
        return 0
    return max(0, record.rooms_scouted)


def _active_generated_room_count(company: CompanyState) -> int:
    session = company.active_expedition
    if session is None or session.generated_dungeon is None:
        return 0
    entry_id = session.generated_dungeon.entry_node_id
    return sum(
        1
        for node_id in session.generated_dungeon.visited_node_ids
        if node_id.startswith("maze_run_") and node_id != entry_id
    )


def _active_generated_action_count(company: CompanyState) -> int:
    session = company.active_expedition
    if session is None or session.generated_dungeon is None:
        return 0
    generated_node_ids = {node.id for node in session.generated_dungeon.nodes}
    return sum(
        1
        for action_key in session.completed_action_ids
        if action_key.split(":", 1)[0] in generated_node_ids
    )


def _active_hunt_lair_cleared(company: CompanyState) -> bool:
    session = company.active_expedition
    if session is None or session.generated_dungeon is None:
        return False
    return any(
        node_id.endswith("_hunt_lair")
        for node_id in session.generated_dungeon.cleared_node_ids
    )
