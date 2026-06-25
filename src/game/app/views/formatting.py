"""Shared string formatting helpers for view builders."""

from __future__ import annotations

from collections.abc import Sequence


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped

def _signed(value: int) -> str:
    return f"+{value}" if value >= 0 else str(value)

def _dedupe_lines(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped

def _join_detail(*pieces: str) -> str:
    return "  |  ".join(piece.strip() for piece in pieces if piece.strip())
