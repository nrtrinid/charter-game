"""Deterministic top-level seed partitions for enemy-decision research."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

SEED_SPLIT_SPEC_ID = "enemy-decision-seeds-v1"
SEED_SPLIT_SPEC_VERSION = 1


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class SeedPartition:
    split: DatasetSplit
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"Invalid seed range for {self.split}: {self.start}..{self.end}")

    def contains(self, seed: int) -> bool:
        return self.start <= seed <= self.end

    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.end + 1))


@dataclass(frozen=True)
class SeedSplitSpec:
    spec_id: str
    version: int
    partitions: tuple[SeedPartition, ...]

    def __post_init__(self) -> None:
        split_names = tuple(partition.split for partition in self.partitions)
        if len(set(split_names)) != len(split_names):
            raise ValueError("Seed split specification contains duplicate split names.")
        for index, left in enumerate(self.partitions):
            for right in self.partitions[index + 1 :]:
                if left.start <= right.end and right.start <= left.end:
                    raise ValueError(
                        f"Seed split overlap: {left.split} {left.start}..{left.end} and "
                        f"{right.split} {right.start}..{right.end}"
                    )

    def partition(self, split: DatasetSplit) -> SeedPartition:
        for partition in self.partitions:
            if partition.split == split:
                return partition
        raise ValueError(f"Unknown dataset split: {split}")

    def split_for_seed(self, seed: int) -> DatasetSplit:
        matches = tuple(
            partition.split for partition in self.partitions if partition.contains(seed)
        )
        if len(matches) != 1:
            raise ValueError(
                f"Seed {seed} belongs to {len(matches)} configured research splits; "
                "expected exactly one."
            )
        return matches[0]

    def seeds_for(self, split: DatasetSplit) -> tuple[int, ...]:
        return self.partition(split).seeds()

    def validate_seeds(self, split: DatasetSplit, seeds: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(seeds)) != len(seeds):
            raise ValueError("Dataset extraction seeds must be unique.")
        wrong_split = tuple(seed for seed in seeds if self.split_for_seed(seed) != split)
        if wrong_split:
            raise ValueError(
                f"Seeds {wrong_split} do not belong to requested split {split.value}."
            )
        return seeds


DEFAULT_SEED_SPLIT_SPEC = SeedSplitSpec(
    spec_id=SEED_SPLIT_SPEC_ID,
    version=SEED_SPLIT_SPEC_VERSION,
    partitions=(
        SeedPartition(DatasetSplit.TRAIN, 1, 256),
        SeedPartition(DatasetSplit.VALIDATION, 10001, 10064),
        SeedPartition(DatasetSplit.TEST, 20001, 20128),
    ),
)
