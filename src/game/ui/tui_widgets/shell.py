"""Shell chrome widgets (header, panes, command dock)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from textual.widgets import Static

from game.app.views import (
    ScreenAction,
    ShellStatusView,
)
from game.ui.tui_widgets.constants import (
    COMMAND_LABEL_WIDTH,
    DOCK_COMMAND_COLUMN_WIDTH,
    DOCK_GAP,
    DOCK_WIDE_MIN_WIDTH,
    META_SEPARATOR,
)


def format_meta_line(*parts: object) -> str:
    """Join compact metadata with consistent spacing for monospaced panes."""
    return META_SEPARATOR.join(text for part in parts if (text := str(part).strip()))


def primary_hotkey(action: ScreenAction) -> str:
    return next((alias for alias in action.aliases if len(alias) == 1), "")


def _dock_label(action: ScreenAction) -> str:
    risk = str(action.risk)
    is_combat = str(action.kind) == "combat"
    if not action.enabled:
        return f"x {action.label}"
    prefix = "!" if _show_dock_warning(action, risk=risk, is_combat=is_combat) else " "
    return f"{prefix}{action.label}"


def _show_dock_warning(action: ScreenAction, *, risk: str, is_combat: bool) -> bool:
    if is_combat:
        return False
    if str(action.kind) == "travel":
        return bool(action.route_warning)
    return risk in {"risky", "irreversible"} or bool(action.confirm)


def _is_boss_route_action(action: ScreenAction) -> bool:
    detail = f"{action.description} {action.preview} {action.result_hint}".lower()
    return "boss" in detail.split()


class StatusHeader(Static):
    """Top status strip for company and save-slot state."""

    def update_status(self, status: ShellStatusView) -> None:
        self.update(self.render_text(status))

    @staticmethod
    def render_text(status: ShellStatusView) -> str:
        company = status.company_name or "No company"
        return "\n".join(
            (
                format_meta_line("The Charter", company, status.location),
                format_meta_line(
                    f"Rep: {status.reputation}",
                    f"Coin: {status.coin}",
                    f"Party: {status.active_count} active / {status.reserve_count} reserve",
                    f"Condition: {status.wounded_count} wounded / {status.downed_count} downed",
                ),
                format_meta_line(
                    f"Save: {status.save_status}",
                    f"Breaches: {status.breaches}",
                ),
            )
        )


class BodyPane(Static):
    """Main screen body without repeated pane chrome."""

    def update_screen(self, _title: str, body: str) -> None:
        self.update(body)


class DetailPane(Static):
    """Focused command or contextual detail pane."""

    def update_detail(self, text: str) -> None:
        self.update(text)


class LogPane(Static):
    """Compact recent log pane."""

    def update_log(self, text: Any) -> None:
        self.update(text or "No recent log.")


class CommandDock(Static):
    """Bottom command dock with focus, numbers, and hotkeys."""

    def update_actions(
        self,
        actions: Sequence[ScreenAction],
        focused_index: int,
        help_text: str = "",
        shortcut_text: str = "",
    ) -> None:
        self.update(
            self.render_text(
                actions,
                focused_index,
                help_text=help_text,
                shortcut_text=shortcut_text,
                width=self.size.width,
            )
        )

    @staticmethod
    def render_text(
        actions: Sequence[ScreenAction],
        focused_index: int,
        *,
        help_text: str = "",
        shortcut_text: str = "",
        width: int = 0,
    ) -> str:
        shortcut_line = _shortcut_line(shortcut_text)
        if not actions:
            lines = ["No commands available."]
            if shortcut_line:
                lines.extend(("", shortcut_line))
            return "\n".join(lines)
        command_lines = _flat_command_lines(actions, focused_index)
        help_lines = _dock_help_lines(help_text)
        if help_lines and width >= DOCK_WIDE_MIN_WIDTH:
            dock_text = _wide_dock_text(command_lines, help_lines, width=width)
            if shortcut_line:
                return f"{dock_text}\n\n{shortcut_line}"
            return dock_text
        lines = list(command_lines)
        if shortcut_line:
            lines.extend(("", shortcut_line))
        if help_lines:
            lines.extend(("", "Focus", *help_lines))
        return "\n".join(lines)


def _flat_command_lines(actions: Sequence[ScreenAction], focused_index: int) -> list[str]:
    command_lines = ["Commands"]
    for index, action in enumerate(actions):
        command_lines.append(_dock_command_line(action, focused=index == focused_index))
    return command_lines


def _dock_command_line(action: ScreenAction, *, focused: bool) -> str:
    focus = ">" if focused else " "
    label = _dock_label(action)
    return f"{focus} {action.number:>2}  {label[:COMMAND_LABEL_WIDTH]:<{COMMAND_LABEL_WIDTH}}"


def _shortcut_line(shortcut_text: str) -> str:
    if not shortcut_text:
        return ""
    return shortcut_text.replace("[", "\\[")


def _dock_help_lines(text: str) -> list[str]:
    return [line[:88].rstrip() for line in text.splitlines() if line.strip()][:4]


def _wide_dock_text(command_lines: list[str], help_lines: list[str], *, width: int) -> str:
    left_width = min(DOCK_COMMAND_COLUMN_WIDTH, max(34, width // 2 - len(DOCK_GAP)))
    right_width = max(24, width - left_width - len(DOCK_GAP))
    left = [_fit_dock_line(line, left_width) for line in command_lines]
    right = ["Hint", *(_fit_dock_line(line, right_width) for line in help_lines)]
    height = max(len(left), len(right))
    left.extend([" " * left_width] * (height - len(left)))
    right.extend([""] * (height - len(right)))
    return "\n".join(
        f"{left_line}{DOCK_GAP}{right_line}"
        for left_line, right_line in zip(left, right, strict=True)
    )


def _fit_dock_line(line: str, width: int) -> str:
    return line[:width].ljust(width)
