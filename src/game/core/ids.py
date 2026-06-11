"""Typed aliases for runtime identifiers."""

from typing import NewType

ActorId = NewType("ActorId", str)
CompanyId = NewType("CompanyId", str)
HeroId = NewType("HeroId", str)
EnemyId = NewType("EnemyId", str)
SkillId = NewType("SkillId", str)
