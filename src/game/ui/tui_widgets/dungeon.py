"""Dungeon map, room, and report Textual widgets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.text import Text
from textual.widgets import Static

from game.app.views import (
    DungeonView,
    ExpeditionReportView,
    ScreenAction,
)
from game.ui.tui_widgets.animation import _compact_art_lines
from game.ui.tui_widgets.combat import _lines_or_none
from game.ui.tui_widgets.constants import (
    MAP_NODE_CENTER,
    MAP_NODE_WIDTH,
    MAP_VIEWPORT_HEIGHT,
    MAP_VIEWPORT_WIDTH,
    MAP_X_STRIDE,
    MAP_Y_STRIDE,
)
from game.ui.tui_widgets.shell import format_meta_line


class DungeonMapPanel(Static):
    """Known dungeon node graph summary."""

    def update_map(self, view: DungeonView) -> None:
        self.update(self.render_minimap(view))

    @staticmethod
    def render_minimap(
        view: DungeonView,
        *,
        highlighted_node_id: str = "",
        actions: Sequence[ScreenAction] | None = None,
    ) -> Text:
        text = Text(
            DungeonMapPanel.render_minimap_text(
                view,
                actions=actions,
                highlighted_node_id=highlighted_node_id,
            )
        )
        _highlight_minimap_nodes(
            text,
            view,
            highlighted_node_id=highlighted_node_id,
            actions=actions,
        )
        return text

    @staticmethod
    def render_minimap_text(
        view: DungeonView,
        *,
        actions: Sequence[ScreenAction] | None = None,
        highlighted_node_id: str = "",
    ) -> str:
        lines = [
            "Mini Map",
            _short_map_name(view.current_room.name),
            "",
        ]
        lines.extend(
            _minimap_lines(
                view,
                actions=actions,
                highlighted_node_id=highlighted_node_id,
            )
        )
        lines.extend(("", "Legend: @ you   o known   ? unknown   ! quest"))
        return "\n".join(lines)

    @staticmethod
    def render_text(
        view: DungeonView,
        *,
        actions: Sequence[ScreenAction] | None = None,
        title: str = "Company Map",
        survey_label: str = "Survey",
        legend_line: str = "@ current  |  o visited  |  ? known  |  number = route command",
    ) -> str:
        lines = [
            title,
            view.current_map_id.replace("_", " ").title(),
            "",
            survey_label,
        ]
        map_lines = _full_map_lines(view)
        lines.extend(map_lines if map_lines else ["No explored map yet."])
        lines.extend(
            (
                "",
                "Legend",
                legend_line,
                "",
                "Inventory",
                _map_inventory_line(view),
                "",
                "Known Places",
            )
        )
        for node in view.map_nodes:
            lines.extend(_map_node_detail_lines(view, node, actions=actions))
        if view.exits:
            lines.extend(("", "Current Routes"))
            for exit_node in view.exits:
                state = "visited" if exit_node.visited else "unexplored"
                if exit_node.map_id != view.current_map_id:
                    state = f"{state}, enters {exit_node.map_id.replace('_', ' ')}"
                number = _action_number_for_value(
                    view,
                    exit_node.node_id,
                    actions=actions,
                )
                prefix = f"{number}. " if number else "- "
                route_detail = format_meta_line(
                    exit_node.name,
                    state,
                    _node_inventory_brief(exit_node),
                )
                lines.append(f"{prefix}{route_detail}")
        return "\n".join(lines)

    @staticmethod
    def render_legacy_text(view: DungeonView) -> str:
        lines = [
            "Company Map",
            view.current_map_id.replace("_", " ").title(),
            "",
            "Known Places",
        ]
        for node in view.map_nodes:
            safe = "safe return" if node.safe_return else ""
            lines.append(f"{_map_status_label(node)}: {node.name}")
            node_detail = format_meta_line(
                node.node_type.replace("_", " "),
                node.status,
                safe,
            )
            lines.append(f"  {node_detail}")
        connection_lines = _full_map_lines(view)
        if connection_lines:
            lines.extend(("", "Connections", *connection_lines))
        if view.exits:
            lines.extend(("", "Current Routes"))
            for exit_node in view.exits:
                state = "visited" if exit_node.visited else "unexplored"
                if exit_node.map_id != view.current_map_id:
                    state = f"{state}, enters {exit_node.map_id.replace('_', ' ')}"
                number = _action_number_for_value(view, exit_node.node_id)
                prefix = f"{number}. " if number else "- "
                lines.append(f"{prefix}{format_meta_line(exit_node.name, state)}")
        return "\n".join(lines)


class DungeonRoomPanel(Static):
    """Current dungeon room panel."""

    def update_room(self, view: DungeonView) -> None:
        self.update(self.render_text(view))

    @staticmethod
    def render_text(view: DungeonView) -> str:
        room = view.current_room
        art_lines = _compact_art_lines(room.art_lines, max_lines=10, max_width=72)
        lines = [
            *art_lines,
            "" if art_lines else None,
            room.name,
            "",
            room.text,
        ]
        impact_lines = [
            line for line in (room.scene_state, room.route_hint, room.party_hint) if line.strip()
        ][:3]
        if impact_lines:
            lines.extend(("", *impact_lines))
        if view.maze_off_spine_hint.strip():
            lines.extend(("", view.maze_off_spine_hint))
        interactable_hint = _interactable_hint(view.room_actions)
        if interactable_hint:
            lines.extend(("", interactable_hint))
        return "\n".join(line for line in lines if line is not None)


def _interactable_hint(room_actions: Sequence[Any]) -> str:
    targets = tuple(
        dict.fromkeys(
            _interactable_target(str(action.label))
            for action in room_actions
            if str(getattr(action, "state", "")) != "completed"
        )
    )
    targets = tuple(target for target in targets if target)
    if not targets:
        return ""
    if len(targets) == 1:
        return "\n".join(
            (
                f"Focus: {targets[0]}",
                "Action: Interact to handle it.",
            )
        )
    return "\n".join(
        (
            "Notable: " + ", ".join(targets[:3]),
            "Action: Interact to choose one.",
        )
    )


def _interactable_target(label: str) -> str:
    words = label.split()
    while words and words[0].lower() in {
        "brace",
        "clear",
        "force",
        "recover",
        "search",
        "unlock",
        "use",
    }:
        words = words[1:]
    return " ".join(words).strip()


class ExpeditionReportPanel(Static):
    """Filed company record summary."""

    def update_report(self, view: ExpeditionReportView) -> None:
        self.update(self.render_text(view))

    @staticmethod
    def render_text(view: ExpeditionReportView) -> str:
        return "\n".join(
            (
                "Filed Company Record",
                f"Outcome: {view.outcome.replace('_', ' ').title()}",
                f"Route: {view.expedition_id}",
                f"Dungeon: {view.dungeon_id or 'none'}",
                "",
                "Record Brief",
                _lines_or_none(_report_brief_lines(view)),
                "",
                "What Changed",
                _lines_or_none(view.what_changed),
                "",
                "Next Objective",
                _lines_or_none(_objective_lines(view.objective)),
                "",
                "Rooms Entered",
                _lines_or_none(view.rooms_entered),
                "",
                "Encounters Resolved",
                _lines_or_none(view.encounters_resolved),
                "",
                "Room Actions",
                _lines_or_none(view.room_actions),
                "",
                "Rewards",
                _reward_lines(view),
                "",
                "Operational Delta",
                f"Reputation: {view.reputation_start}->{view.reputation_end} "
                f"({_signed(view.reputation_delta)})",
                f"Coin: {view.coin_start}->{view.coin_end} ({_signed(view.coin_delta)})",
                "Supplies",
                _delta_lines(view.supply_deltas),
                "Inventory",
                _delta_lines(view.inventory_deltas),
                "Gear",
                _delta_lines(view.gear_deltas),
                "",
                "Hero Outcomes",
                _lines_or_none(view.hero_outcomes),
                "",
                "Notable Moments",
                _lines_or_none(view.notable_moments),
                "",
                "Company Condition",
                f"Wounded: {view.wounded_count}",
                f"Downed: {view.downed_count}",
                f"Memorial: {view.deceased_count}",
            )
        )


def _objective_lines(objective: Any) -> tuple[str, ...]:
    lines = [
        f"{objective.title} [{objective.status}]",
        objective.summary,
    ]
    if getattr(objective, "progress", ""):
        lines.append(f"Progress: {objective.progress}")
    lines.extend(
        [
            f"Next: {objective.next_step}",
            f"Chapter: {objective.chapter_status}",
        ]
    )
    lines.extend(f"{step.state.title()}: {step.name}" for step in objective.steps)
    return tuple(line for line in lines if line)


def _contract_summary_line(entry: Any) -> str:
    reward = _contract_reward_summary(entry)
    line = f"- {entry.name}: {entry.state.title()} ({reward})"
    if entry.unavailable_reason:
        line += f" - {entry.unavailable_reason}"
    return line


def _contract_reward_summary(entry: Any) -> str:
    pieces: list[str] = []
    if getattr(entry, "reward_reputation", 0):
        pieces.append(f"+{entry.reward_reputation} rep")
    if getattr(entry, "coin_reward", 0):
        pieces.append(f"+{entry.coin_reward} Coin")
    return ", ".join(pieces) or "no payout"


def _upgrade_summary_line(entry: Any) -> str:
    line = f"- {entry.name}: {entry.state.title()} (cost {entry.cost})"
    if entry.effect_summary:
        line += f" - {entry.effect_summary}"
    if entry.unavailable_reason and entry.state != "installed":
        line += f" - {entry.unavailable_reason}"
    return line


def _report_brief_lines(view: ExpeditionReportView) -> tuple[str, ...]:
    lines = [
        f"Outcome: {view.outcome.replace('_', ' ').title()}.",
        (
            f"Reputation: {view.reputation_start}->{view.reputation_end} "
            f"({_signed(view.reputation_delta)})."
        ),
        (f"Coin: {view.coin_start}->{view.coin_end} ({_signed(view.coin_delta)})."),
    ]
    if view.wounded_count or view.downed_count or view.deceased_count:
        lines.append(
            "Condition: "
            f"{view.wounded_count} wounded, "
            f"{view.downed_count} downed, "
            f"{view.deceased_count} memorialized."
        )
    if view.supplies_spent:
        spent = ", ".join(f"{supply_id} x{quantity}" for supply_id, quantity in view.supplies_spent)
        lines.append(f"Supplies spent: {spent}.")
    if view.loot:
        loot = ", ".join(f"{item_id} x{quantity}" for item_id, quantity in view.loot)
        lines.append(f"Loot secured: {loot}.")
    if getattr(view, "gear", ()):
        gear = ", ".join(f"{gear_id} x{quantity}" for gear_id, quantity in view.gear)
        lines.append(f"Gear secured: {gear}.")
    if view.breaches_discovered:
        breaches = ", ".join(view.breaches_discovered)
        lines.append(f"Breach discovered: {breaches}.")
    lines.extend(view.hero_outcomes[:2])
    lines.extend(view.notable_moments[:3])
    return tuple(lines)


def _reward_lines(view: ExpeditionReportView) -> str:
    lines = [f"Reputation: {view.reputation_gained}", f"Coin: {view.coin_gained}"]
    lines.extend(f"Loot: {item_id} x{quantity}" for item_id, quantity in view.loot)
    lines.extend(f"Supplies: {supply_id} x{quantity}" for supply_id, quantity in view.supplies)
    lines.extend(f"Gear: {gear_id} x{quantity}" for gear_id, quantity in view.gear)
    lines.extend(f"Spent: {supply_id} x{quantity}" for supply_id, quantity in view.supplies_spent)
    lines.extend(f"Breach: {breach_id}" for breach_id in view.breaches_discovered)
    return "\n".join(lines) if lines else "none"


def _delta_lines(values: Sequence[tuple[str, int, int, int]]) -> str:
    if not values:
        return "none"
    return "\n".join(
        f"- {item_id}: {start}->{end} ({_signed(delta)})" for item_id, start, end, delta in values
    )


def _signed(value: int) -> str:
    return f"+{value}" if value > 0 else str(value)


def _action_number_for_value(
    view: DungeonView,
    value: str,
    *,
    actions: Sequence[ScreenAction] | None = None,
) -> str:
    available_actions = view.actions if actions is None else actions
    action = next(
        (candidate for candidate in available_actions if candidate.value == value),
        None,
    )
    return action.number if action is not None else ""


def _map_inventory_line(view: DungeonView) -> str:
    inventory = _quantity_line(view.inventory)
    supplies = _quantity_line(view.supplies)
    return format_meta_line(
        f"inventory: {inventory}" if inventory else "inventory: none",
        f"supplies: {supplies}" if supplies else "",
    )


def _map_node_detail_lines(
    view: DungeonView,
    node: Any,
    *,
    actions: Sequence[ScreenAction] | None = None,
) -> list[str]:
    marker = _minimap_node_label(view, node, actions=actions)
    safe = "safe return" if node.safe_return else ""
    heading = f"{marker} {node.name}"
    detail = format_meta_line(
        _map_status_label(node),
        node.node_type.replace("_", " "),
        node.direction,
        safe,
    )
    lines = [heading, f"  {detail}"]
    if node.memory_summary:
        lines.append(f"  memory: {node.memory_summary}")
    inventory = _node_inventory_brief(node)
    if inventory:
        lines.append(f"  inventory: {inventory}")
    for action_line in node.action_summaries[:3]:
        lines.append(f"  action: {action_line}")
    if len(node.action_summaries) > 3:
        lines.append(f"  action: +{len(node.action_summaries) - 3} more")
    for note in node.memory_notes[:2]:
        lines.append(f"  note: {note}")
    return lines


def _node_inventory_brief(node: Any) -> str:
    pieces = [
        f"loot {_quantity_line(node.inventory_rewards)}" if node.inventory_rewards else "",
        f"supplies {_quantity_line(node.supply_rewards)}" if node.supply_rewards else "",
        f"reputation +{node.reputation_reward}" if node.reputation_reward else "",
        f"Coin +{node.coin_reward}" if node.coin_reward else "",
        f"needs {_quantity_line(node.inventory_requirements)}"
        if node.inventory_requirements
        else "",
        f"costs {_quantity_line(node.supply_costs)}" if node.supply_costs else "",
    ]
    return format_meta_line(*pieces)


def _quantity_line(values: Sequence[tuple[str, int]]) -> str:
    return ", ".join(
        f"{_label_identifier(item_id)} x{quantity}" for item_id, quantity in values if quantity
    )


def _label_identifier(value: str) -> str:
    return value.replace("_", " ")


def _minimap_lines(
    view: DungeonView,
    *,
    actions: Sequence[ScreenAction] | None = None,
    highlighted_node_id: str = "",
) -> list[str]:
    return _coordinate_minimap_lines(
        view,
        actions=actions,
        highlighted_node_id=highlighted_node_id,
    ) or _minimap_branch_lines(
        view,
        actions=actions,
        highlighted_node_id=highlighted_node_id,
    )


def _highlight_minimap_nodes(
    text: Text,
    view: DungeonView,
    *,
    highlighted_node_id: str,
    actions: Sequence[ScreenAction] | None = None,
) -> None:
    map_start = text.plain.find("\n\n")
    if map_start == -1:
        return
    map_start += 2
    for node in view.map_nodes:
        if node.current:
            style = "bold cyan"
            labels = _minimap_current_node_labels(view, node, actions=actions)
        elif node.node_id == highlighted_node_id:
            style = "bold yellow"
            labels = _minimap_highlighted_node_labels(
                view,
                node,
                actions=actions,
                highlighted_node_id=highlighted_node_id,
            )
        else:
            continue
        for label in labels:
            if not label:
                continue
            start = map_start
            while True:
                index = text.plain.find(label, start)
                if index == -1:
                    break
                text.stylize(style, index, index + len(label))
                start = index + len(label)


def _minimap_current_node_labels(
    view: DungeonView,
    node: Any,
    *,
    actions: Sequence[ScreenAction] | None = None,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                _minimap_node_label(view, node, actions=actions),
                _minimap_node_label(
                    view,
                    node,
                    include_number=False,
                    actions=actions,
                ),
            )
        )
    )


def _minimap_highlighted_node_labels(
    view: DungeonView,
    node: Any,
    *,
    actions: Sequence[ScreenAction] | None = None,
    highlighted_node_id: str = "",
) -> tuple[str, ...]:
    numbered_label = _minimap_node_label(
        view,
        node,
        actions=actions,
        highlighted_node_id=highlighted_node_id,
    )
    if _action_number_for_value(view, node.node_id, actions=actions):
        return (numbered_label,)
    return (
        _minimap_node_label(
            view,
            node,
            include_number=False,
            actions=actions,
            highlighted_node_id=highlighted_node_id,
        ),
    )


def _coordinate_minimap_lines(
    view: DungeonView,
    *,
    actions: Sequence[ScreenAction] | None = None,
    highlighted_node_id: str = "",
) -> list[str]:
    spatial_nodes = _spatial_map_nodes(view)
    if not spatial_nodes:
        return []
    current = next((node for node in spatial_nodes if node.current), None)
    if current is None:
        return _coordinate_full_map_lines(view)

    center_row = MAP_VIEWPORT_HEIGHT // 2
    center_col = (MAP_VIEWPORT_WIDTH - MAP_NODE_WIDTH) // 2
    canvas = [[" " for _ in range(MAP_VIEWPORT_WIDTH)] for _ in range(MAP_VIEWPORT_HEIGHT)]
    nodes_by_id = {node.node_id: node for node in spatial_nodes}
    drawn_edges: set[tuple[str, str]] = set()
    anchor_x, anchor_y = _minimap_viewport_anchor(current, nodes_by_id)

    def position_for(node: Any) -> tuple[int, int]:
        return (
            center_row + round((node.map_y - anchor_y) * MAP_Y_STRIDE),
            center_col + round((node.map_x - anchor_x) * MAP_X_STRIDE),
        )

    for node in spatial_nodes:
        for exit_node_id in node.exit_node_ids:
            exit_node = nodes_by_id.get(exit_node_id)
            if exit_node is None or not _map_nodes_share_drawable_edge(node, exit_node):
                continue
            edge_key = _map_edge_key(node, exit_node)
            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)
            from_row, from_col = position_for(node)
            to_row, to_col = position_for(exit_node)
            _draw_map_edge_between(canvas, from_row, from_col, to_row, to_col)

    overflow_top = overflow_bottom = overflow_left = overflow_right = False
    for node in spatial_nodes:
        row, col = position_for(node)
        overflow_top = overflow_top or row < 0
        overflow_bottom = overflow_bottom or row >= MAP_VIEWPORT_HEIGHT
        overflow_left = overflow_left or col < 0
        overflow_right = overflow_right or col + MAP_NODE_WIDTH > MAP_VIEWPORT_WIDTH
        _draw_map_label_at(
            canvas,
            view,
            node,
            row=row,
            col=col,
            actions=actions,
            highlighted_node_id=highlighted_node_id,
        )

    _draw_map_overflow_markers(
        canvas,
        top=overflow_top,
        bottom=overflow_bottom,
        left=overflow_left,
        right=overflow_right,
    )
    return [line.rstrip() for line in ("".join(row) for row in canvas)]


def _coordinate_full_map_lines(view: DungeonView) -> list[str]:
    spatial_nodes = _spatial_map_nodes(view)
    if not spatial_nodes:
        return []

    min_x = min(node.map_x for node in spatial_nodes if node.map_x is not None)
    max_x = max(node.map_x for node in spatial_nodes if node.map_x is not None)
    min_y = min(node.map_y for node in spatial_nodes if node.map_y is not None)
    max_y = max(node.map_y for node in spatial_nodes if node.map_y is not None)
    width = (max_x - min_x) * MAP_X_STRIDE + MAP_NODE_WIDTH
    height = (max_y - min_y) * MAP_Y_STRIDE + 1
    canvas = [[" " for _ in range(width)] for _ in range(height)]
    nodes_by_id = {node.node_id: node for node in spatial_nodes}
    drawn_edges: set[tuple[str, str]] = set()

    for node in spatial_nodes:
        for exit_node_id in node.exit_node_ids:
            exit_node = nodes_by_id.get(exit_node_id)
            if exit_node is None or not _map_nodes_share_drawable_edge(node, exit_node):
                continue
            edge_key = _map_edge_key(node, exit_node)
            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)
            _draw_map_edge(canvas, node, exit_node, min_x=min_x, min_y=min_y)

    for node in spatial_nodes:
        _draw_map_label(canvas, view, node, min_x=min_x, min_y=min_y)

    return [line.rstrip() for line in ("".join(row) for row in canvas)]


def _minimap_viewport_anchor(
    current: Any,
    nodes_by_id: dict[str, Any],
) -> tuple[float, float]:
    focus_nodes = [current]
    for exit_node_id in current.exit_node_ids:
        exit_node = nodes_by_id.get(exit_node_id)
        if exit_node is not None and exit_node.map_x is not None and exit_node.map_y is not None:
            focus_nodes.append(exit_node)
    xs = [node.map_x for node in focus_nodes if node.map_x is not None]
    ys = [node.map_y for node in focus_nodes if node.map_y is not None]
    if not xs or not ys:
        return float(current.map_x or 0), float(current.map_y or 0)
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def _map_nodes_share_drawable_edge(from_node: Any, to_node: Any) -> bool:
    if (
        from_node.map_x is None
        or from_node.map_y is None
        or to_node.map_x is None
        or to_node.map_y is None
    ):
        return False
    return not (from_node.map_x == to_node.map_x and from_node.map_y == to_node.map_y)


def _full_map_lines(view: DungeonView) -> list[str]:
    return _coordinate_full_map_lines(view) or _minimap_branch_lines(view)


def _spatial_map_nodes(view: DungeonView) -> list[Any]:
    return [node for node in view.map_nodes if node.map_x is not None and node.map_y is not None]


def _draw_map_edge(
    canvas: list[list[str]],
    from_node: Any,
    to_node: Any,
    *,
    min_x: int,
    min_y: int,
) -> None:
    from_row, from_col = _map_canvas_position(from_node, min_x=min_x, min_y=min_y)
    to_row, to_col = _map_canvas_position(to_node, min_x=min_x, min_y=min_y)
    _draw_map_edge_between(canvas, from_row, from_col, to_row, to_col)


def _draw_map_edge_between(
    canvas: list[list[str]],
    from_row: int,
    from_col: int,
    to_row: int,
    to_col: int,
) -> None:
    from_center_col = from_col + MAP_NODE_CENTER
    to_center_col = to_col + MAP_NODE_CENTER
    if from_row == to_row:
        _draw_map_horizontal_connector(canvas, from_row, from_center_col, to_center_col)
        return
    if from_center_col == to_center_col:
        _draw_map_vertical_connector(canvas, from_center_col, from_row, to_row)
        return
    _draw_map_horizontal_connector(canvas, from_row, from_center_col, to_center_col)
    _draw_map_vertical_connector(canvas, to_center_col, from_row, to_row)


def _draw_map_horizontal_connector(
    canvas: list[list[str]],
    row: int,
    from_center_col: int,
    to_center_col: int,
) -> None:
    if from_center_col == to_center_col:
        return
    left_center = min(from_center_col, to_center_col)
    right_center = max(from_center_col, to_center_col)
    start = left_center - MAP_NODE_CENTER + MAP_NODE_WIDTH
    end = right_center - MAP_NODE_CENTER
    for col in range(start, end):
        _put_map_char(canvas, row, col, "-")


def _draw_map_vertical_connector(
    canvas: list[list[str]],
    center_col: int,
    from_row: int,
    to_row: int,
) -> None:
    start = min(from_row, to_row) + 1
    end = max(from_row, to_row)
    for row in range(start, end):
        _put_map_char(canvas, row, center_col, "|")


def _draw_map_label(
    canvas: list[list[str]],
    view: DungeonView,
    node: Any,
    *,
    min_x: int,
    min_y: int,
) -> None:
    row, col = _map_canvas_position(node, min_x=min_x, min_y=min_y)
    _draw_map_label_at(canvas, view, node, row=row, col=col)


def _draw_map_label_at(
    canvas: list[list[str]],
    view: DungeonView,
    node: Any,
    *,
    row: int,
    col: int,
    actions: Sequence[ScreenAction] | None = None,
    highlighted_node_id: str = "",
) -> None:
    node_label = _minimap_node_label(
        view,
        node,
        actions=actions,
        highlighted_node_id=highlighted_node_id,
    )
    label = f"{node_label:^{MAP_NODE_WIDTH}}"[:MAP_NODE_WIDTH]
    for offset, character in enumerate(label):
        _put_map_char(canvas, row, col + offset, character)


def _draw_map_overflow_markers(
    canvas: list[list[str]],
    *,
    top: bool,
    bottom: bool,
    left: bool,
    right: bool,
) -> None:
    if not canvas:
        return
    middle_row = len(canvas) // 2
    if top:
        _put_map_char(canvas, 0, len(canvas[0]) - 1, "^")
    if bottom:
        _put_map_char(canvas, len(canvas) - 1, len(canvas[0]) - 1, "v")
    if left:
        _put_map_char(canvas, middle_row, 0, "<")
    if right:
        _put_map_char(canvas, middle_row, len(canvas[0]) - 1, ">")


def _map_edge_key(from_node: Any, to_node: Any) -> tuple[str, str]:
    return (
        min(from_node.node_id, to_node.node_id),
        max(from_node.node_id, to_node.node_id),
    )


def _map_canvas_position(node: Any, *, min_x: int, min_y: int) -> tuple[int, int]:
    return (
        (node.map_y - min_y) * MAP_Y_STRIDE,
        (node.map_x - min_x) * MAP_X_STRIDE,
    )


def _put_map_char(
    canvas: list[list[str]],
    row: int,
    col: int,
    character: str,
) -> None:
    if row < 0 or col < 0 or row >= len(canvas) or col >= len(canvas[row]):
        return
    existing = canvas[row][col]
    if existing in {"-", "|"} and character in {"-", "|"} and existing != character:
        canvas[row][col] = "+"
    else:
        canvas[row][col] = character


def _minimap_branch_lines(
    view: DungeonView,
    *,
    actions: Sequence[ScreenAction] | None = None,
    highlighted_node_id: str = "",
) -> list[str]:
    nodes_by_id = {node.node_id: node for node in view.map_nodes}
    current = next((node for node in view.map_nodes if node.current), None)
    if current is None:
        return ["No explored map yet."]

    exits = [nodes_by_id[node_id] for node_id in current.exit_node_ids if node_id in nodes_by_id]
    north, west, east, south, extra = _compass_exits(exits)
    current_label = _minimap_node_label(
        view,
        current,
        include_number=False,
        actions=actions,
        highlighted_node_id=highlighted_node_id,
    )

    lines: list[str] = []
    if north is not None:
        north_label = _minimap_node_label(
            view,
            north,
            actions=actions,
            highlighted_node_id=highlighted_node_id,
        )
        lines.extend(
            (
                f"        {north_label}",
                "         |",
            )
        )

    left_label = (
        _minimap_node_label(
            view,
            west,
            actions=actions,
            highlighted_node_id=highlighted_node_id,
        )
        if west is not None
        else ""
    )
    right_label = (
        _minimap_node_label(
            view,
            east,
            actions=actions,
            highlighted_node_id=highlighted_node_id,
        )
        if east is not None
        else ""
    )
    if west is not None or east is not None:
        lines.append(f"{left_label:<3} --  {current_label}  --  {right_label}")
    else:
        lines.append(f"         {current_label}")

    if south is not None:
        south_label = _minimap_node_label(
            view,
            south,
            actions=actions,
            highlighted_node_id=highlighted_node_id,
        )
        lines.extend(
            (
                "         |",
                f"        {south_label}",
            )
        )

    visited_neighbors = [
        exit_node for exit_node in exits if exit_node.visited and not exit_node.current
    ]
    for neighbor in visited_neighbors:
        neighbor_exits = [
            nodes_by_id[node_id]
            for node_id in neighbor.exit_node_ids
            if node_id in nodes_by_id and node_id != current.node_id
        ]
        if neighbor_exits:
            lines.extend(
                (
                    "",
                    *_visited_branch_lines(
                        view,
                        neighbor,
                        neighbor_exits,
                        actions=actions,
                        highlighted_node_id=highlighted_node_id,
                    ),
                )
            )

    if extra:
        lines.extend(("", "More paths"))
        for node in extra:
            label = _minimap_node_label(
                view,
                node,
                actions=actions,
                highlighted_node_id=highlighted_node_id,
            )
            lines.append(f"  {label}")
    return lines


def _compass_exits(
    nodes: list[Any],
) -> tuple[Any | None, Any | None, Any | None, Any | None, list[Any]]:
    slots: list[Any | None] = [None, None, None, None]
    extra: list[Any] = []
    for index, node in enumerate(nodes):
        if index < len(slots):
            slots[index] = node
        else:
            extra.append(node)
    return slots[0], slots[1], slots[2], slots[3], extra


def _visited_branch_lines(
    view: DungeonView,
    node: Any,
    exits: list[Any],
    *,
    actions: Sequence[ScreenAction] | None = None,
    highlighted_node_id: str = "",
) -> list[str]:
    node_label = _minimap_node_label(
        view,
        node,
        include_number=False,
        actions=actions,
        highlighted_node_id=highlighted_node_id,
    )
    lines = [node_label]
    for index, exit_node in enumerate(exits[:4]):
        branch = "`-" if index == min(len(exits), 4) - 1 else "|-"
        exit_label = _minimap_node_label(
            view,
            exit_node,
            actions=actions,
            highlighted_node_id=highlighted_node_id,
        )
        lines.append(f"  {branch} {exit_label}")
    if len(exits) > 4:
        lines.append(f"  `- +{len(exits) - 4} more")
    return lines


def _minimap_node_label(
    view: DungeonView,
    node: Any,
    *,
    include_number: bool = True,
    actions: Sequence[ScreenAction] | None = None,
    highlighted_node_id: str = "",
) -> str:
    if node.current:
        marker = "@"
    elif node.visited:
        marker = "o"
    elif _is_quest_marker_node(node) and not node.cleared:
        marker = "!"
    else:
        marker = "?"
    number = _action_number_for_value(view, node.node_id, actions=actions) if include_number else ""
    focus = ">" if node.node_id == highlighted_node_id and number else ""
    return f"{focus}{number}{marker}"


def _is_quest_marker_node(node: Any) -> bool:
    return bool(getattr(node, "quest_marker", False))


def _map_status_label(node: Any) -> str:
    if node.current:
        return "Current"
    if node.cleared:
        return "Cleared"
    if node.visited:
        return "Visited"
    return "Unexplored"


def _short_map_name(name: str) -> str:
    replacements = {
        "Haven East Gate": "Gate",
        "Old Road": "Road",
        "Abandoned Toll Post": "Toll Post",
        "Hunter's Trail": "Trail",
        "Bramble Shrine": "Shrine",
        "Shallow Cave Entrance": "Cave Entrance",
        "Cave Mouth": "Mouth",
        "Forked Descent": "Fork",
        "Fungus-Lit Gallery": "Gallery",
        "Old Works Cache": "Cache",
        "Black Stone Sinkhole": "Sinkhole",
        "Dry Creek Bed": "Dry Creek",
        "Black Stone Gate": "Gate",
        "Maze-Touched Lair": "Lair",
    }
    return replacements.get(name, name)
