"""Document Slice 9 memory.py line ranges (modules already extracted)."""

from __future__ import annotations

# 1-based inclusive ranges from pre-Slice-9 memory.py
MODULE_RANGES: dict[str, list[tuple[int, int]]] = {
    "memory_util.py": [(1231, 1239), (1328, 1329)],
    "memory_capture.py": [(74, 93), (850, 884), (1307, 1309)],
    "memory_signals.py": [(52, 71), (887, 1152)],
    "memory_event_signals.py": [(96, 351), (1155, 1228)],
    "memory_manifestation.py": [(427, 507), (583, 641)],
    "memory_post_expedition.py": [(383, 424), (510, 580), (811, 847)],
    "memory_archive.py": [(644, 731), (734, 808), (1242, 1304), (1312, 1325)],
}


def main() -> None:
    for name, ranges in MODULE_RANGES.items():
        print(name, ranges)


if __name__ == "__main__":
    main()