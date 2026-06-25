"""Shared helpers for expedition memory modules."""

from __future__ import annotations


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
def _label(value: str) -> str:
    return value.replace("_", " ").title()
