"""Dev-only umbrella command for enemy and route research."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from game.combat.enemy_learning import (
    SUPPORTED_ENEMY_MOVEMENT_MODES,
    SUPPORTED_ENEMY_WAIT_MODES,
    SUPPORTED_HERO_POLICY_IDS,
)
from game.dev.ai_counterfactuals import (
    CounterfactualVariant,
    DecisionWeightOverride,
    SkillOverride,
    format_counterfactual_sweep,
    run_counterfactual_sweep,
)
from game.dev.ai_packages import evaluate_enemy_packages, format_package_report
from game.dev.ai_tactics import (
    TacticDiscoveryConfig,
    format_tactic_discovery_report,
    run_tactic_discovery,
)
from game.dev.breach_balance_lab import (
    BreachFightBalanceConfig,
    format_breach_fight_balance_result,
    run_breach_fight_balance,
)
from game.dev.policy_band_report import (
    SCENARIO_GENERATED_HUNT,
    SCENARIO_GENERATED_SCOUT,
    format_policy_band_pair_report,
    format_policy_band_report,
    run_authored_route_policy_band,
    run_breach_policy_band_pair,
    run_generated_route_policy_band,
)
from game.dev.route_lab import (
    SUPPORTED_GENERATED_ROUTE_STRATEGIES,
    SUPPORTED_ROUTE_ENVELOPE_IDS,
    GeneratedRouteLabConfig,
    RouteLabConfig,
    format_generated_route_lab_summary,
    format_route_lab_summary,
    run_generated_route_lab,
    run_route_lab,
)
from game.dev.train_enemy_ai import (
    SUPPORTED_POLICY_SCOPE_IDS,
    SUPPORTED_PRESET_IDS,
    SUPPORTED_ROUTE_IDS,
    TrainingRunConfig,
    format_training_summary,
    run_training_harness,
)
from game.expedition.maze_director import MAZE_PRESSURE_PROFILES

_BUILT_IN_ENEMY_VARIANTS: dict[str, CounterfactualVariant] = {
    "maw_slam_plus_1": CounterfactualVariant(
        variant_id="maw_slam_plus_1",
        description="Raise Maw Slam range by 1 for in-memory comparison.",
        skill_overrides=(
            SkillOverride(
                "maw_slam",
                {"damage": 4, "damage_min": 4, "damage_max": 5},
            ),
        ),
    ),
    "drag_no_mark": CounterfactualVariant(
        variant_id="drag_no_mark",
        description="Remove Mark payoff setup from Drag Forward tags for comparison.",
        skill_overrides=(
            SkillOverride(
                "drag_forward",
                {"tags": ["enemy", "boss", "formation", "drag_forward"]},
            ),
        ),
    ),
    "bandit_payoff_plus_1": CounterfactualVariant(
        variant_id="bandit_payoff_plus_1",
        description="Raise bandit payoff attack floor/range by 1 for comparison.",
        skill_overrides=(
            SkillOverride(
                "dirty_finish",
                {"damage": 4, "damage_min": 4, "damage_max": 5},
            ),
        ),
    ),
    "damage_pressure_zero": CounterfactualVariant(
        variant_id="damage_pressure_zero",
        description="Set learned damage_pressure weight to zero for comparison.",
        decision_weight_overrides=(DecisionWeightOverride("damage_pressure", 0),),
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dev-only AI and route evaluation lab.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_enemy_packages_parser(subparsers)
    _add_route_parser(subparsers)
    _add_generated_route_parser(subparsers)
    _add_enemy_sweep_parser(subparsers)
    _add_route_sweep_parser(subparsers)
    _add_discover_tactics_parser(subparsers)
    _add_balance_breach_fights_parser(subparsers)
    _add_policy_band_parser(subparsers)
    args = parser.parse_args(argv)

    if args.command == "enemy-packages":
        print(_run_enemy_packages_command(args))
    elif args.command == "route":
        print(_run_route_command(args))
    elif args.command == "generated-route":
        print(_run_generated_route_command(args))
    elif args.command == "enemy-sweep":
        print(_run_enemy_sweep_command(args))
    elif args.command == "route-sweep":
        print(_run_route_sweep_command(args))
    elif args.command == "discover-tactics":
        print(_run_discover_tactics_command(args))
    elif args.command == "balance-breach-fights":
        print(_run_balance_breach_fights_command(args))
    elif args.command == "policy-band":
        print(_run_policy_band_command(args))
    return 0


def _add_common_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--preset", choices=SUPPORTED_PRESET_IDS, default="fresh")
    parser.add_argument("--hero-policy", choices=SUPPORTED_HERO_POLICY_IDS, default="mixed")
    parser.add_argument("--enemy-wait-mode", choices=SUPPORTED_ENEMY_WAIT_MODES, default="none")
    parser.add_argument(
        "--enemy-move-mode",
        choices=SUPPORTED_ENEMY_MOVEMENT_MODES,
        default="recovery_only",
    )


def _add_enemy_packages_parser(subparsers) -> None:
    parser = subparsers.add_parser("enemy-packages", help="report enemy package health")
    _add_common_training_args(parser)
    parser.add_argument("--route", choices=SUPPORTED_ROUTE_IDS, default="opening_pressure_path")
    parser.add_argument("--encounter", action="append", default=[])
    parser.add_argument("--eval-hero-policy", action="append", default=[])
    parser.add_argument("--cross-policy", action="store_true")
    parser.add_argument(
        "--policy-scope",
        action="append",
        default=[],
        choices=SUPPORTED_POLICY_SCOPE_IDS,
    )


def _add_route_parser(subparsers) -> None:
    parser = subparsers.add_parser("route", help="evaluate an authored route envelope")
    _add_common_training_args(parser)
    parser.add_argument("--route", choices=SUPPORTED_ROUTE_IDS, default="opening_critical_path")
    parser.add_argument("--envelope", choices=SUPPORTED_ROUTE_ENVELOPE_IDS, default="")


def _add_generated_route_parser(subparsers) -> None:
    parser = subparsers.add_parser("generated-route", help="evaluate generated Maze routes")
    _add_common_training_args(parser)
    parser.add_argument("--breach", default="shallow_cave_breach")
    parser.add_argument(
        "--strategy",
        choices=SUPPORTED_GENERATED_ROUTE_STRATEGIES,
        default="mainline",
    )
    parser.add_argument("--profile", choices=_pressure_profile_ids(), default="")
    parser.add_argument("--envelope", choices=SUPPORTED_ROUTE_ENVELOPE_IDS, default="")


def _add_enemy_sweep_parser(subparsers) -> None:
    parser = subparsers.add_parser("enemy-sweep", help="run built-in enemy counterfactuals")
    _add_common_training_args(parser)
    parser.add_argument("--route", choices=SUPPORTED_ROUTE_IDS, default="opening_pressure_path")
    parser.add_argument(
        "--variant",
        action="append",
        choices=tuple(_BUILT_IN_ENEMY_VARIANTS),
        default=[],
    )


def _add_route_sweep_parser(subparsers) -> None:
    parser = subparsers.add_parser("route-sweep", help="compare generated route profiles")
    _add_common_training_args(parser)
    parser.add_argument("--breach", default="shallow_cave_breach")
    parser.add_argument(
        "--strategy",
        choices=SUPPORTED_GENERATED_ROUTE_STRATEGIES,
        default="take_all_optional",
    )
    parser.add_argument("--profile", action="append", choices=_pressure_profile_ids(), default=[])
    parser.add_argument("--envelope", choices=SUPPORTED_ROUTE_ENVELOPE_IDS, default="")


def _add_discover_tactics_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "discover-tactics",
        help="discover learned tactic candidates for an enemy package",
    )
    _add_common_training_args(parser)
    parser.add_argument("--package", required=True, dest="package_id")
    parser.add_argument("--route", choices=SUPPORTED_ROUTE_IDS, default="opening_pressure_path")
    parser.add_argument("--eval-hero-policy", action="append", default=[])
    parser.add_argument(
        "--policy-scope",
        action="append",
        default=[],
        choices=SUPPORTED_POLICY_SCOPE_IDS,
    )
    parser.add_argument(
        "--emphasis-scale",
        action="append",
        type=float,
        default=[],
    )


def _add_policy_band_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "policy-band",
        help="report cross-policy robustness gates for route/breach scenarios",
    )
    _add_common_training_args(parser)
    parser.add_argument("--route", choices=SUPPORTED_ROUTE_IDS, default="")
    parser.add_argument("--breach", default="shallow_cave_breach")
    parser.add_argument(
        "--strategy",
        choices=SUPPORTED_GENERATED_ROUTE_STRATEGIES,
        default="take_all_optional",
    )
    parser.add_argument("--profile", choices=_pressure_profile_ids(), default="")
    parser.add_argument(
        "--breach-pair",
        action="store_true",
        help="evaluate scout (breach_probe) and hunt (marked_hunt) generated routes",
    )


def _add_balance_breach_fights_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "balance-breach-fights",
        help="search and optionally apply generated breach fight balance candidates",
    )
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--preset", choices=SUPPORTED_PRESET_IDS, default="attrition")
    parser.add_argument("--hero-policy", choices=SUPPORTED_HERO_POLICY_IDS, default="mixed")
    parser.add_argument(
        "--strategy",
        choices=SUPPORTED_GENERATED_ROUTE_STRATEGIES,
        default="take_all_optional",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="rank candidates without writing YAML",
    )


def _run_enemy_packages_command(args) -> str:
    evaluation_hero_policy_ids = (
        SUPPORTED_HERO_POLICY_IDS if args.cross_policy else tuple(args.eval_hero_policy)
    )
    summary = run_training_harness(
        TrainingRunConfig(
            encounter_ids=tuple(args.encounter),
            seeds=args.seeds,
            max_rounds=args.max_rounds,
            hero_policy_id=args.hero_policy,
            evaluation_hero_policy_ids=evaluation_hero_policy_ids,
            policy_scope_ids=tuple(args.policy_scope) or ("global",),
            preset_id=args.preset,
            route_id=args.route if not args.encounter else "",
            enemy_wait_mode=args.enemy_wait_mode,
            enemy_movement_mode=args.enemy_move_mode,
        )
    )
    return "\n\n".join(
        (
            format_training_summary(summary),
            format_package_report(evaluate_enemy_packages(summary)),
        )
    )


def _run_route_command(args) -> str:
    return format_route_lab_summary(
        run_route_lab(
            RouteLabConfig(
                route_id=args.route,
                seeds=args.seeds,
                max_rounds=args.max_rounds,
                hero_policy_id=args.hero_policy,
                preset_id=args.preset,
                envelope_id=args.envelope,
                enemy_wait_mode=args.enemy_wait_mode,
                enemy_movement_mode=args.enemy_move_mode,
            )
        )
    )


def _run_generated_route_command(args) -> str:
    return format_generated_route_lab_summary(
        run_generated_route_lab(
            GeneratedRouteLabConfig(
                breach_id=args.breach,
                seeds=args.seeds,
                max_rounds=args.max_rounds,
                hero_policy_id=args.hero_policy,
                preset_id=args.preset,
                strategy_id=args.strategy,
                pressure_profile_id=args.profile,
                envelope_id=args.envelope,
            )
        )
    )


def _run_enemy_sweep_command(args) -> str:
    selected = tuple(args.variant) or ("maw_slam_plus_1", "drag_no_mark")
    variants = tuple(_BUILT_IN_ENEMY_VARIANTS[variant_id] for variant_id in selected)
    summary = run_counterfactual_sweep(
        TrainingRunConfig(
            seeds=args.seeds,
            max_rounds=args.max_rounds,
            hero_policy_id=args.hero_policy,
            preset_id=args.preset,
            route_id=args.route,
            enemy_wait_mode=args.enemy_wait_mode,
            enemy_movement_mode=args.enemy_move_mode,
        ),
        variants,
    )
    return format_counterfactual_sweep(summary)


def _run_route_sweep_command(args) -> str:
    profile_ids = tuple(args.profile) or _pressure_profile_ids()
    summaries = tuple(
        run_generated_route_lab(
            GeneratedRouteLabConfig(
                breach_id=args.breach,
                seeds=args.seeds,
                max_rounds=args.max_rounds,
                hero_policy_id=args.hero_policy,
                preset_id=args.preset,
                strategy_id=args.strategy,
                pressure_profile_id=profile_id,
                envelope_id=args.envelope,
            )
        )
        for profile_id in profile_ids
    )
    ranked = sorted(
        summaries,
        key=lambda summary: (-summary.envelope_score.score, summary.pressure_profile_id),
    )
    lines = ["Route Sweep", "Ranked generated profiles:"]
    for summary in ranked:
        completed = sum(1 for run in summary.runs if run.completed)
        lines.append(
            "  "
            f"{summary.pressure_profile_id or 'director'}: "
            f"{summary.envelope_score.status} score={summary.envelope_score.score}; "
            f"completed {completed}/{len(summary.runs)}"
        )
    return "\n".join(lines)


def _run_discover_tactics_command(args) -> str:
    return format_tactic_discovery_report(
        run_tactic_discovery(
            TacticDiscoveryConfig(
                package_id=args.package_id,
                route_id=args.route,
                seeds=args.seeds,
                max_rounds=args.max_rounds,
                hero_policy_id=args.hero_policy,
                evaluation_hero_policy_ids=tuple(args.eval_hero_policy)
                or SUPPORTED_HERO_POLICY_IDS,
                preset_id=args.preset,
                policy_scope_ids=tuple(args.policy_scope)
                or ("global", "boss", "per_role"),
                emphasis_scales=tuple(args.emphasis_scale) or (1.0, 1.25, 1.5),
                enemy_wait_mode=args.enemy_wait_mode,
                enemy_movement_mode=args.enemy_move_mode,
            )
        )
    )


def _run_balance_breach_fights_command(args) -> str:
    return format_breach_fight_balance_result(
        run_breach_fight_balance(
            BreachFightBalanceConfig(
                seeds=args.seeds,
                max_rounds=args.max_rounds,
                hero_policy_id=args.hero_policy,
                preset_id=args.preset,
                strategy_id=args.strategy,
                dry_run=args.dry_run,
            )
        )
    )


def _run_policy_band_command(args) -> str:
    if args.breach_pair:
        return format_policy_band_pair_report(
            run_breach_policy_band_pair(
                BreachFightBalanceConfig(
                    seeds=args.seeds,
                    max_rounds=args.max_rounds,
                    preset_id=args.preset,
                    strategy_id=args.strategy,
                )
            )
        )
    if args.route:
        return format_policy_band_report(
            run_authored_route_policy_band(
                RouteLabConfig(
                    route_id=args.route,
                    seeds=args.seeds,
                    max_rounds=args.max_rounds,
                    preset_id=args.preset,
                    enemy_wait_mode=args.enemy_wait_mode,
                    enemy_movement_mode=args.enemy_move_mode,
                )
            )
        )
    profile_id = args.profile or "breach_probe"
    scenario_kind = (
        SCENARIO_GENERATED_HUNT
        if profile_id == "marked_hunt"
        else SCENARIO_GENERATED_SCOUT
    )
    return format_policy_band_report(
        run_generated_route_policy_band(
            GeneratedRouteLabConfig(
                breach_id=args.breach,
                seeds=args.seeds,
                max_rounds=args.max_rounds,
                preset_id=args.preset,
                strategy_id=args.strategy,
                pressure_profile_id=profile_id,
            ),
            scenario_kind=scenario_kind,
        )
    )


def _pressure_profile_ids() -> tuple[str, ...]:
    return tuple(profile.pressure_id for profile in MAZE_PRESSURE_PROFILES)


if __name__ == "__main__":
    raise SystemExit(main())
