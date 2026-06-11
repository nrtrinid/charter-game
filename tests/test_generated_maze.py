from __future__ import annotations

from game.campaign.company import GeneratedDungeonState, MazeRecipe, create_new_company
from game.core.rng import GameRng
from game.expedition.generated_maze import (
    _forward_spine_cell,
    _open_adjacent_position,
    derive_maze_extension_seed,
    extend_maze_main_spine,
    extend_maze_spur,
    frontier_exit_previews,
    frontier_node_id,
    frontier_preview_map_position,
    frontier_preview_map_positions,
    generate_maze_breach_route,
    generated_nodes_by_id,
    resolve_generated_maze_travel,
)
from game.expedition.maze_director import (
    RandomMazeDirectorPolicy,
    ScriptedMazeDirectorPolicy,
    build_maze_observation,
    choose_maze_recipe,
    legal_maze_pressure_profiles,
)
from tests.conftest import get_definitions


def test_maze_recipe_round_trips() -> None:
    recipe = MazeRecipe(
        pressure_id="test_pressure",
        route_length=4,
        combat_budget=1,
        hazard_budget=0,
        reward_lure=False,
        include_hunt=True,
        enemy_policy_id="basic",
        pressure_tags=("test", "hunt"),
        layout_style="forked",
        branch_budget=3,
        room_palette="market",
        encounter_style="light",
    )

    loaded = MazeRecipe.from_dict(recipe.to_dict())

    assert loaded == recipe


def test_generated_maze_route_is_deterministic_and_connected() -> None:
    first = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=101,
    )
    second = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(999),
        seed=101,
    )

    assert [node.model_dump(mode="json") for node in first.nodes] == [
        node.model_dump(mode="json") for node in second.nodes
    ]
    assert 7 <= len(first.nodes) <= 9
    node_ids = {node.id for node in first.nodes}
    assert len(node_ids) == len(first.nodes)
    assert first.entry_node_id in node_ids
    for node in first.nodes:
        assert all(exit_id in node_ids for exit_id in node.exits)
    assert any(node.encounter for node in first.nodes)
    assert any(action.loot for node in first.nodes for action in node.actions)
    assert first.recipe is not None
    assert first.recipe.pressure_id == "legacy_breach_route"
    main_positions = [
        node.position
        for node in first.nodes
        if node.id.startswith(f"{first.run_id}_room_")
    ]
    assert len({position[0] for position in main_positions if position is not None}) > 1


def test_generated_maze_hunt_contract_adds_marked_lair() -> None:
    route = generate_maze_breach_route(
        run_number=2,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=202,
        include_hunt=True,
    )

    hunt_nodes = [node for node in route.nodes if node.id.endswith("_hunt_lair")]
    assert len(hunt_nodes) == 1
    assert hunt_nodes[0].name in {"Marked Lair", "Maw-Marked Den"}
    assert hunt_nodes[0].encounter == "generated_maze_hunt"
    assert hunt_nodes[0].exits[0] in {node.id for node in route.nodes}
    assert route.recipe is not None
    assert route.recipe.include_hunt

    definitions = get_definitions()
    encounter = definitions.encounters["generated_maze_hunt"]
    assert [enemy.enemy_id for enemy in encounter.enemies] == [
        "cave_maw_brute",
        "pattern_ward",
    ]


def test_generated_maze_recipe_controls_route_shape() -> None:
    recipe = MazeRecipe(
        pressure_id="lean_pressure",
        route_length=4,
        combat_budget=1,
        hazard_budget=0,
        reward_lure=False,
        include_hunt=False,
        enemy_policy_id="future_policy",
        pressure_tags=("lean",),
        layout_style="winding",
        branch_budget=0,
        room_palette="stone",
        encounter_style="light",
    )

    route = generate_maze_breach_route(
        run_number=3,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=303,
        recipe=recipe,
    )

    assert route.recipe == recipe
    assert len(_main_room_ids(route)) == 4
    assert not any(node.id.endswith("_reward") for node in route.nodes)
    assert not any(node.id.endswith("_hard_room") for node in route.nodes)
    assert not any(node.id.endswith("_echo") for node in route.nodes)
    assert any(node.encounter == "generated_maze_probe" for node in route.nodes)
    assert route.recipe.enemy_policy_id == "future_policy"


def test_generated_maze_recipe_can_add_branches_palette_and_hunt() -> None:
    recipe = MazeRecipe(
        pressure_id="full_pressure",
        route_length=5,
        combat_budget=2,
        hazard_budget=0,
        reward_lure=True,
        include_hunt=True,
        enemy_policy_id="basic",
        pressure_tags=("reward", "hunt"),
        layout_style="forked",
        branch_budget=3,
        room_palette="market",
        encounter_style="light",
    )

    route = generate_maze_breach_route(
        run_number=4,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=404,
        recipe=recipe,
    )

    assert len(_main_room_ids(route)) == 5
    assert any(node.id.endswith("_reward") for node in route.nodes)
    assert any(node.id.endswith("_hard_room") for node in route.nodes)
    assert any(node.id.endswith("_hunt_lair") for node in route.nodes)
    assert any(node.id.endswith("_echo") for node in route.nodes)
    assert any(
        node.name in {"Auction-Chalk Niche", "Folded Corridor"}
        for node in route.nodes
    )
    hard_room = next(node for node in route.nodes if node.id.endswith("_hard_room"))
    assert hard_room.encounter == "generated_maze_pattern_cell"


def test_generated_maze_standard_routes_use_multiple_maze_fights() -> None:
    recipe = MazeRecipe(
        pressure_id="standard_pressure",
        route_length=4,
        combat_budget=1,
        hazard_budget=0,
        reward_lure=False,
        include_hunt=False,
        enemy_policy_id="basic",
        pressure_tags=("standard",),
        encounter_style="standard",
    )

    encounter_ids = {
        next(
            node.encounter
            for node in generate_maze_breach_route(
                run_number=seed,
                source_node_id="maze_breach",
                return_node_id="maze_breach",
                rng=GameRng(7),
                seed=seed,
                recipe=recipe,
            ).nodes
            if node.encounter
        )
        for seed in range(410, 430)
    }

    assert len(encounter_ids) > 1
    assert encounter_ids <= {
        "generated_maze_probe",
        "generated_maze_pattern_cell",
        "maze_depth_1",
    }


def test_generated_maze_old_payload_without_recipe_loads() -> None:
    route = generate_maze_breach_route(
        run_number=5,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=505,
    )
    payload = route.to_dict()
    payload.pop("recipe")

    loaded = GeneratedDungeonState.from_dict(payload)

    assert loaded.recipe is None
    assert loaded.seed == 505
    assert loaded.entry_node_id == route.entry_node_id


def test_maze_recipe_old_payload_defaults_new_director_tools() -> None:
    loaded = MazeRecipe.from_dict(
        {
            "pressure_id": "old_pressure",
            "route_length": 4,
            "combat_budget": 2,
            "hazard_budget": 0,
            "reward_lure": True,
            "include_hunt": False,
            "enemy_policy_id": "basic",
            "pressure_tags": ["old"],
        }
    )

    assert loaded.layout_style == "winding"
    assert loaded.branch_budget == 2
    assert loaded.room_palette == "stone"
    assert loaded.encounter_style == "standard"


def test_legal_maze_pressure_profiles_gate_hunt_pressure() -> None:
    company = create_new_company(get_definitions())

    observation = build_maze_observation(
        company,
        source_node_id="maze_breach",
        run_number=1,
    )
    pressure_ids = {
        profile.pressure_id for profile in legal_maze_pressure_profiles(observation)
    }
    assert "marked_hunt" not in pressure_ids

    company.active_contract_ids.add("shallow_cave_breach_hunt")
    hunt_observation = build_maze_observation(
        company,
        source_node_id="maze_breach",
        run_number=2,
    )
    hunt_pressure_ids = {
        profile.pressure_id for profile in legal_maze_pressure_profiles(hunt_observation)
    }
    assert "marked_hunt" in hunt_pressure_ids


def test_scripted_maze_director_prefers_hunt_when_legal() -> None:
    company = create_new_company(get_definitions())

    default_recipe = choose_maze_recipe(
        company,
        source_node_id="maze_breach",
        run_number=1,
        rng=GameRng(1),
        policy=ScriptedMazeDirectorPolicy(),
    )
    assert default_recipe.pressure_id == "breach_probe"
    assert not default_recipe.include_hunt
    assert default_recipe.layout_style == "winding"
    assert default_recipe.branch_budget == 2
    assert default_recipe.room_palette == "glass"

    company.active_contract_ids.add("shallow_cave_breach_hunt")
    hunt_recipe = choose_maze_recipe(
        company,
        source_node_id="maze_breach",
        run_number=2,
        rng=GameRng(1),
        policy=ScriptedMazeDirectorPolicy(),
    )
    assert hunt_recipe.pressure_id == "marked_hunt"
    assert hunt_recipe.include_hunt
    assert hunt_recipe.layout_style == "dead_end_heavy"
    assert hunt_recipe.branch_budget == 3
    assert hunt_recipe.room_palette == "maw"
    assert hunt_recipe.encounter_style == "brute"


def test_active_contract_can_request_maze_pressure_profile() -> None:
    definitions = get_definitions()
    company = create_new_company(definitions)
    company.active_contract_ids.add("shallow_cave_breach_scout")

    recipe = choose_maze_recipe(
        company,
        source_node_id="maze_breach",
        run_number=1,
        rng=GameRng(1),
        definitions=definitions,
        policy=ScriptedMazeDirectorPolicy(),
    )

    assert recipe.pressure_id == "long_pressure"
    assert recipe.route_length == 5
    assert recipe.layout_style == "forked"
    assert recipe.branch_budget == 3
    assert recipe.room_palette == "market"


def test_random_maze_director_is_seed_deterministic() -> None:
    company = create_new_company(get_definitions())
    policy = RandomMazeDirectorPolicy()

    first = choose_maze_recipe(
        company,
        source_node_id="maze_breach",
        run_number=1,
        rng=GameRng(33),
        policy=policy,
    )
    second = choose_maze_recipe(
        company,
        source_node_id="maze_breach",
        run_number=1,
        rng=GameRng(33),
        policy=policy,
    )

    assert second == first


def test_frontier_exit_previews_forward_always_at_spine_tip() -> None:
    recipe = MazeRecipe(
        pressure_id="lean_pressure",
        route_length=3,
        combat_budget=1,
        hazard_budget=0,
        reward_lure=False,
        include_hunt=False,
        enemy_policy_id="basic",
        pressure_tags=("lean",),
        branch_budget=0,
    )
    route = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=1515,
        recipe=recipe,
    )
    previews = frontier_exit_previews(route, frontier_node_id(route))
    assert len(previews) == 1
    assert previews[0].kind == "forward"
    assert previews[0].exit_id == f"{route.run_id}_room_4"


def test_frontier_exit_previews_includes_spur_when_recipe_allows() -> None:
    recipe = MazeRecipe(
        pressure_id="full_pressure",
        route_length=3,
        combat_budget=2,
        hazard_budget=0,
        reward_lure=True,
        include_hunt=False,
        enemy_policy_id="basic",
        pressure_tags=("reward",),
        layout_style="forked",
        branch_budget=3,
    )
    route = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=1616,
        recipe=recipe,
    )
    previews = frontier_exit_previews(route, frontier_node_id(route))
    assert len(previews) == 2
    assert previews[0].kind == "forward"
    assert previews[1].kind == "spur"
    assert previews[1].exit_id == f"{route.run_id}_spur_4"


def test_frontier_exit_previews_empty_when_not_at_frontier() -> None:
    route = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=1717,
    )
    inner_room = f"{route.run_id}_room_1"
    assert frontier_exit_previews(route, inner_room) == ()


def _reward_spur_recipe() -> MazeRecipe:
    return MazeRecipe(
        pressure_id="full_pressure",
        route_length=3,
        combat_budget=2,
        hazard_budget=0,
        reward_lure=True,
        include_hunt=False,
        enemy_policy_id="basic",
        pressure_tags=("reward",),
        layout_style="forked",
        branch_budget=3,
    )


def _full_hunt_recipe() -> MazeRecipe:
    return MazeRecipe(
        pressure_id="full_pressure",
        route_length=3,
        combat_budget=2,
        hazard_budget=0,
        reward_lure=True,
        include_hunt=True,
        enemy_policy_id="basic",
        pressure_tags=("reward",),
        layout_style="forked",
        branch_budget=3,
    )


def _long_pressure_recipe() -> MazeRecipe:
    return MazeRecipe(
        pressure_id="long_pressure",
        route_length=5,
        combat_budget=2,
        hazard_budget=0,
        reward_lure=True,
        include_hunt=False,
        enemy_policy_id="basic",
        pressure_tags=("long_route", "combat", "overreach"),
        layout_style="forked",
        branch_budget=3,
        room_palette="market",
        encounter_style="standard",
    )


def _marked_hunt_recipe() -> MazeRecipe:
    return MazeRecipe(
        pressure_id="marked_hunt",
        route_length=5,
        combat_budget=2,
        hazard_budget=0,
        reward_lure=True,
        include_hunt=True,
        enemy_policy_id="basic",
        pressure_tags=("hunt", "boss", "contract"),
        layout_style="dead_end_heavy",
        branch_budget=3,
        room_palette="maw",
        encounter_style="brute",
    )


def _manhattan_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _assert_generated_links_are_cardinally_adjacent(route: GeneratedDungeonState) -> None:
    nodes = {node.id: node for node in route.nodes}
    for node in route.nodes:
        if node.position is None:
            continue
        for exit_id in node.exits:
            exit_node = nodes.get(exit_id)
            if exit_node is None or exit_node.position is None:
                continue
            assert _manhattan_distance(node.position, exit_node.position) == 1, (
                node.id,
                exit_id,
                node.position,
                exit_node.position,
            )


def test_generated_route_links_stay_cardinally_adjacent_across_seed_sweep() -> None:
    recipes = (
        _reward_spur_recipe(),
        _full_hunt_recipe(),
        _marked_hunt_recipe(),
        _long_pressure_recipe(),
    )
    for recipe in recipes:
        for seed in range(1, 151):
            route = generate_maze_breach_route(
                run_number=1,
                source_node_id="maze_breach",
                return_node_id="maze_breach",
                rng=GameRng(7),
                seed=seed,
                recipe=recipe,
            )
            _assert_generated_links_are_cardinally_adjacent(route)


def test_marked_hunt_keeps_hunt_lair_adjacent_when_optional_branches_crowd() -> None:
    route = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=10,
        recipe=_marked_hunt_recipe(),
    )

    hunt = next(node for node in route.nodes if node.id.endswith("_hunt_lair"))
    anchor = next(node for node in route.nodes if node.id == hunt.exits[0])

    assert hunt.position is not None
    assert anchor.position is not None
    assert _manhattan_distance(anchor.position, hunt.position) == 1
    _assert_generated_links_are_cardinally_adjacent(route)


def test_seamless_generation_previews_and_extensions_stay_adjacent() -> None:
    for seed in range(1, 51):
        route = generate_maze_breach_route(
            run_number=1,
            source_node_id="maze_breach",
            return_node_id="maze_breach",
            rng=GameRng(7),
            seed=seed,
            recipe=_long_pressure_recipe(),
        )
        for _step in range(8):
            frontier = next(node for node in route.nodes if node.id == frontier_node_id(route))
            assert frontier.position is not None
            previews = frontier_exit_previews(route, frontier.id)
            preview_positions = frontier_preview_map_positions(route, previews)
            assert any(preview.kind == "forward" for preview in previews)
            for preview in previews:
                position = preview_positions[preview.exit_id]
                assert _manhattan_distance(frontier.position, position) == 1

            spur = next((preview for preview in previews if preview.kind == "spur"), None)
            if spur is not None:
                spur_result = extend_maze_spur(route)
                assert spur_result.node.position is not None
                assert _manhattan_distance(frontier.position, spur_result.node.position) == 1

            forward_result = extend_maze_main_spine(route)
            assert forward_result.node.position is not None
            assert _manhattan_distance(frontier.position, forward_result.node.position) == 1
            _assert_generated_links_are_cardinally_adjacent(route)


def test_spur_extension_avoids_forward_spine_cell() -> None:
    for seed in range(1, 101):
        route = generate_maze_breach_route(
            run_number=1,
            source_node_id="maze_breach",
            return_node_id="maze_breach",
            rng=GameRng(7),
            seed=seed,
            recipe=_reward_spur_recipe(),
        )
        previews = frontier_exit_previews(route, frontier_node_id(route))
        if not any(preview.kind == "spur" for preview in previews):
            continue
        recipe = route.recipe or _reward_spur_recipe()
        nodes_by_id = generated_nodes_by_id(route)
        frontier_id = frontier_node_id(route)
        next_depth = route.main_spine_length + 1
        reserved = _forward_spine_cell(
            nodes_by_id,
            route.run_id,
            frontier_id,
            next_depth,
            recipe,
        )
        if reserved is None:
            continue
        result = extend_maze_spur(route)
        assert result.node.position is not None
        assert result.node.position != reserved


def test_forward_extension_prefers_spine_direction_when_free() -> None:
    route = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=2020,
        recipe=_reward_spur_recipe(),
    )
    recipe = route.recipe or _reward_spur_recipe()
    nodes_by_id = generated_nodes_by_id(route)
    frontier_id = frontier_node_id(route)
    next_depth = route.main_spine_length + 1
    reserved = _forward_spine_cell(
        nodes_by_id,
        route.run_id,
        frontier_id,
        next_depth,
        recipe,
    )
    assert reserved is not None
    result = extend_maze_main_spine(route)
    assert result.node.position == reserved


def test_forward_extension_survives_surrounding_occupancy() -> None:
    route = generate_maze_breach_route(
        run_number=2,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=633031,
        recipe=_long_pressure_recipe(),
    )
    extend_maze_spur(route)
    previews = frontier_exit_previews(route, frontier_node_id(route))
    forward = next(preview for preview in previews if preview.kind == "forward")
    positions = frontier_preview_map_positions(route, previews)
    assert forward.exit_id in positions
    result = extend_maze_main_spine(route)
    assert result.node.position is not None
    assert positions[forward.exit_id] == result.node.position


def test_first_spine_extension_stays_cardinally_adjacent() -> None:
    for seed in range(1, 201):
        route = generate_maze_breach_route(
            run_number=1,
            source_node_id="maze_breach",
            return_node_id="maze_breach",
            rng=GameRng(7),
            seed=seed,
            recipe=_full_hunt_recipe(),
        )
        frontier = next(
            node for node in route.nodes if node.id == frontier_node_id(route)
        )
        assert frontier.position is not None
        result = extend_maze_main_spine(route)
        assert result.node.position is not None
        assert _manhattan_distance(frontier.position, result.node.position) == 1


def test_forward_preview_label_matches_generated_room_name() -> None:
    for seed in range(1, 51):
        route = generate_maze_breach_route(
            run_number=1,
            source_node_id="maze_breach",
            return_node_id="maze_breach",
            rng=GameRng(7),
            seed=seed,
            recipe=_full_hunt_recipe(),
        )
        forward = next(
            preview
            for preview in frontier_exit_previews(route, frontier_node_id(route))
            if preview.kind == "forward"
        )
        generated = extend_maze_main_spine(route)
        assert forward.label == generated.node.name


def test_spur_preview_label_matches_generated_room_name() -> None:
    for seed in range(1, 51):
        route = generate_maze_breach_route(
            run_number=1,
            source_node_id="maze_breach",
            return_node_id="maze_breach",
            rng=GameRng(7),
            seed=seed,
            recipe=_reward_spur_recipe(),
        )
        previews = frontier_exit_previews(route, frontier_node_id(route))
        spur = next((preview for preview in previews if preview.kind == "spur"), None)
        if spur is None:
            continue
        generated = extend_maze_spur(route)
        assert spur.label == generated.node.name


def test_frontier_preview_map_position_matches_forward_extension() -> None:
    route = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=2020,
        recipe=_reward_spur_recipe(),
    )
    forward = next(
        preview
        for preview in frontier_exit_previews(route, frontier_node_id(route))
        if preview.kind == "forward"
    )
    assert frontier_preview_map_position(route, forward) == (
        extend_maze_main_spine(route).node.position
    )


def test_frontier_preview_map_positions_reserve_forward_before_spur() -> None:
    route = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=2020,
        recipe=_reward_spur_recipe(),
    )
    previews = frontier_exit_previews(route, frontier_node_id(route))
    positions = frontier_preview_map_positions(route, previews)
    assert len(positions) == len(previews)
    assert len(set(positions.values())) == len(positions)
    frontier = next(node for node in route.nodes if node.id == frontier_node_id(route))
    assert frontier.position is not None
    for position in positions.values():
        assert _manhattan_distance(frontier.position, position) == 1


def test_frontier_preview_map_position_matches_spur_extension() -> None:
    route = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=2020,
        recipe=_reward_spur_recipe(),
    )
    spur = next(
        preview
        for preview in frontier_exit_previews(route, frontier_node_id(route))
        if preview.kind == "spur"
    )
    assert frontier_preview_map_position(route, spur) == (
        extend_maze_spur(route).node.position
    )


def test_resolve_generated_maze_travel_forward_extends_spine() -> None:
    recipe = MazeRecipe(
        pressure_id="lean_pressure",
        route_length=3,
        combat_budget=1,
        hazard_budget=0,
        reward_lure=False,
        include_hunt=False,
        enemy_policy_id="basic",
        pressure_tags=("lean",),
        branch_budget=0,
    )
    route = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=1818,
        recipe=recipe,
    )
    forward_id = f"{route.run_id}_room_4"
    result = resolve_generated_maze_travel(
        route,
        current_node_id=frontier_node_id(route),
        target_node_id=forward_id,
    )
    assert result is not None
    assert route.main_spine_length == 4
    assert result.node.id == forward_id


def test_resolve_generated_maze_travel_spur_does_not_increment_spine() -> None:
    recipe = MazeRecipe(
        pressure_id="full_pressure",
        route_length=3,
        combat_budget=2,
        hazard_budget=0,
        reward_lure=True,
        include_hunt=False,
        enemy_policy_id="basic",
        pressure_tags=("reward",),
        layout_style="forked",
        branch_budget=3,
    )
    route = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=1919,
        recipe=recipe,
    )
    spur_id = f"{route.run_id}_spur_4"
    result = resolve_generated_maze_travel(
        route,
        current_node_id=frontier_node_id(route),
        target_node_id=spur_id,
    )
    assert result is not None
    assert route.main_spine_length == 3
    assert result.node.id == spur_id


def test_extend_maze_spur_has_no_loot() -> None:
    recipe = MazeRecipe(
        pressure_id="full_pressure",
        route_length=3,
        combat_budget=2,
        hazard_budget=0,
        reward_lure=True,
        include_hunt=False,
        enemy_policy_id="basic",
        pressure_tags=("reward",),
        layout_style="forked",
        branch_budget=3,
        encounter_style="light",
    )
    route = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=2020,
        recipe=recipe,
    )
    result = extend_maze_spur(route)
    assert not result.node.loot
    assert result.node.coin_reward == 0
    for action in result.node.actions:
        assert not action.loot
        assert action.coin_reward == 0


def test_initial_spine_frontier_has_no_forward_exit() -> None:
    recipe = MazeRecipe(
        pressure_id="lean_pressure",
        route_length=4,
        combat_budget=1,
        hazard_budget=0,
        reward_lure=False,
        include_hunt=False,
        enemy_policy_id="basic",
        pressure_tags=("lean",),
        branch_budget=0,
    )
    route = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=808,
        recipe=recipe,
    )

    assert route.main_spine_length == 4
    frontier = next(node for node in route.nodes if node.id == frontier_node_id(route))
    assert f"{route.run_id}_room_5" not in frontier.exits
    assert f"{route.run_id}_room_3" in frontier.exits


def test_derive_maze_extension_seed_is_stable() -> None:
    seed = derive_maze_extension_seed(101, "maze_run_0001", 4)
    assert seed == derive_maze_extension_seed(101, "maze_run_0001", 4)
    assert seed != derive_maze_extension_seed(101, "maze_run_0001", 5)
    assert seed > 0


def test_extend_maze_main_spine_appends_connected_room() -> None:
    recipe = MazeRecipe(
        pressure_id="lean_pressure",
        route_length=3,
        combat_budget=1,
        hazard_budget=0,
        reward_lure=False,
        include_hunt=False,
        enemy_policy_id="basic",
        pressure_tags=("lean",),
        branch_budget=0,
    )
    route = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=909,
        recipe=recipe,
    )
    assert route.main_spine_length == 3

    result = extend_maze_main_spine(route)

    assert route.main_spine_length == 4
    assert result.new_depth == 4
    assert result.node.id == f"{route.run_id}_room_4"
    node_ids = {node.id for node in route.nodes}
    assert result.node.id in node_ids
    frontier = next(node for node in route.nodes if node.id == result.previous_frontier_id)
    assert result.node.id in frontier.exits
    assert result.previous_frontier_id in result.node.exits
    assert f"{route.run_id}_room_5" not in result.node.exits


def test_extend_maze_main_spine_is_deterministic() -> None:
    recipe = MazeRecipe(
        pressure_id="lean_pressure",
        route_length=3,
        combat_budget=2,
        hazard_budget=0,
        reward_lure=False,
        include_hunt=False,
        enemy_policy_id="basic",
        pressure_tags=("lean",),
        branch_budget=0,
    )

    def extend_once(seed: int) -> GeneratedDungeonState:
        route = generate_maze_breach_route(
            run_number=1,
            source_node_id="maze_breach",
            return_node_id="maze_breach",
            rng=GameRng(7),
            seed=seed,
            recipe=recipe,
        )
        extend_maze_main_spine(route)
        return route

    first = extend_once(111)
    second = extend_once(111)
    fourth_room_first = next(
        node for node in first.nodes if node.id == f"{first.run_id}_room_4"
    )
    fourth_room_second = next(
        node for node in second.nodes if node.id == f"{second.run_id}_room_4"
    )
    assert fourth_room_first.model_dump(mode="json") == fourth_room_second.model_dump(
        mode="json"
    )


def test_extend_maze_main_spine_scales_combat_and_curio_by_depth() -> None:
    recipe = MazeRecipe(
        pressure_id="lean_pressure",
        route_length=3,
        combat_budget=2,
        hazard_budget=0,
        reward_lure=False,
        include_hunt=False,
        enemy_policy_id="basic",
        pressure_tags=("lean",),
        branch_budget=0,
        encounter_style="light",
    )
    route = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=1212,
        recipe=recipe,
    )
    extend_maze_main_spine(route)
    room_4 = next(node for node in route.nodes if node.id.endswith("_room_4"))
    assert room_4.node_type.value == "maze"

    extend_maze_main_spine(route)
    room_5 = next(node for node in route.nodes if node.id.endswith("_room_5"))
    assert room_5.node_type.value == "maze"

    extend_maze_main_spine(route)
    room_6 = next(node for node in route.nodes if node.id.endswith("_room_6"))
    assert room_6.node_type.value == "curio"

    extend_maze_main_spine(route)
    room_7 = next(node for node in route.nodes if node.id.endswith("_room_7"))
    assert room_7.node_type.value == "maze"

    extend_maze_main_spine(route)
    room_8 = next(node for node in route.nodes if node.id.endswith("_room_8"))
    assert room_8.node_type.value == "maze"

    extend_maze_main_spine(route)
    room_9 = next(node for node in route.nodes if node.id.endswith("_room_9"))
    assert room_9.node_type.value == "combat"
    assert room_9.encounter == "generated_maze_probe"


def test_pushed_main_spine_rooms_do_not_grant_free_coin_or_loot() -> None:
    recipe = MazeRecipe(
        pressure_id="lean_pressure",
        route_length=3,
        combat_budget=2,
        hazard_budget=0,
        reward_lure=True,
        include_hunt=False,
        enemy_policy_id="basic",
        pressure_tags=("lean",),
        branch_budget=2,
    )
    route = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=1313,
        recipe=recipe,
    )
    initial_main_ids = set(_main_room_ids(route))

    for _ in range(5):
        extend_maze_main_spine(route)

    pushed_main_rooms = [
        node
        for node in route.nodes
        if node.id.startswith(f"{route.run_id}_room_")
        and node.id not in initial_main_ids
    ]
    assert pushed_main_rooms
    assert not any(node.id.endswith("_reward") for node in pushed_main_rooms)
    for node in pushed_main_rooms:
        assert not node.loot
        assert node.coin_reward == 0
        for action in node.actions:
            assert not action.loot
            assert action.coin_reward == 0


def test_generated_dungeon_state_migrates_main_spine_length() -> None:
    route = generate_maze_breach_route(
        run_number=1,
        source_node_id="maze_breach",
        return_node_id="maze_breach",
        rng=GameRng(7),
        seed=1414,
    )
    payload = route.to_dict()
    payload.pop("main_spine_length")

    loaded = GeneratedDungeonState.from_dict(payload)

    assert loaded.main_spine_length == route.main_spine_length


def _main_room_ids(route: GeneratedDungeonState) -> list[str]:
    return [
        node.id
        for node in route.nodes
        if node.id.startswith(f"{route.run_id}_room_")
    ]


def test_forward_preview_node_type_matches_generated_room() -> None:
    for seed in range(1, 51):
        route = generate_maze_breach_route(
            run_number=1,
            source_node_id="maze_breach",
            return_node_id="maze_breach",
            rng=GameRng(7),
            seed=seed,
            recipe=_full_hunt_recipe(),
        )
        forward = next(
            preview
            for preview in frontier_exit_previews(route, frontier_node_id(route))
            if preview.kind == "forward"
        )
        generated = extend_maze_main_spine(route)
        assert forward.node_type == generated.node.node_type.value


def test_spur_preview_node_type_matches_generated_room() -> None:
    for seed in range(1, 101):
        route = generate_maze_breach_route(
            run_number=1,
            source_node_id="maze_breach",
            return_node_id="maze_breach",
            rng=GameRng(7),
            seed=seed,
            recipe=_reward_spur_recipe(),
        )
        spur = next(
            (
                preview
                for preview in frontier_exit_previews(route, frontier_node_id(route))
                if preview.kind == "spur"
            ),
            None,
        )
        if spur is None:
            continue
        generated = extend_maze_spur(route)
        assert spur.node_type == generated.node.node_type.value


def test_generated_route_avoids_duplicate_room_names() -> None:
    for seed in range(1, 101):
        route = generate_maze_breach_route(
            run_number=1,
            source_node_id="maze_breach",
            return_node_id="maze_breach",
            rng=GameRng(7),
            seed=seed,
            recipe=_full_hunt_recipe(),
        )
        names = [node.name for node in route.nodes]
        assert len(names) == len(set(names))


def test_open_adjacent_position_finds_cell_beyond_cardinal_ring() -> None:
    anchor = (0, 0)
    occupied: set[tuple[int, int]] = set()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        occupied.add((anchor[0] + dx, anchor[1] + dy))
    for dx in range(-8, 9):
        for dy in range(-8, 9):
            if abs(dx) + abs(dy) == 8:
                occupied.add((anchor[0] + dx, anchor[1] + dy))
    position = _open_adjacent_position(
        anchor,
        occupied,
        GameRng(1),
        allow_distant=True,
    )
    assert position not in occupied
