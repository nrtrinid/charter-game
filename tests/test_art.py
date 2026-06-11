from __future__ import annotations

import shutil

import pytest

from game.data.loaders import default_data_dir, load_game_definitions
from tests.conftest import get_definitions


def test_art_data_loads_and_missing_file_falls_back_to_empty(tmp_path) -> None:
    definitions = get_definitions()

    watchman = definitions.art.hero_classes["watchman"]
    assert watchman.display_name == "Watchman"
    assert watchman.glyph == "#"
    assert watchman.mini is not None
    assert watchman.mini.lines == ["[#]", "[#\\", "/ \\"]
    assert watchman.mini.frames["idle"][1].lines == ["[#]", "[#|", "/ \\"]
    assert definitions.art.hero_classes["cutpurse"].mini is not None
    assert definitions.art.hero_classes["cutpurse"].mini.lines[0] == " o "
    assert definitions.art.hero_classes["field_surgeon"].mini is not None
    assert definitions.art.hero_classes["field_surgeon"].mini.lines[0] == " o "
    assert definitions.art.hero_classes["scribe"].mini is not None
    assert definitions.art.hero_classes["scribe"].mini.lines[0] == " o "
    assert definitions.art.enemies["bandit_cutthroat"].mini is not None
    assert definitions.art.enemies["bandit_cutthroat"].mini.lines[0] == " o "
    assert definitions.art.enemies["bandit_slinger"].mini is not None
    assert definitions.art.enemies["bandit_slinger"].mini.lines[0] == " o "
    lookout = definitions.art.enemies["bandit_lookout"]
    assert lookout.mini is not None
    assert lookout.mini.lines[0] == "?o "
    assert lookout.mini.frames["idle"][1].lines[0] == "!o "
    assert watchman.lines
    assert watchman.frames["idle"][1].lines
    assert watchman.frame_metadata["attack"].impact_frame == 1
    assert definitions.art.enemies["bone_soldier"].frames["hurt"][0].lines
    assert definitions.art.enemies["bone_soldier"].lines
    assert definitions.art.dungeon_nodes["town_gate"].lines
    assert all(
        len(asset.lines) >= 7
        for asset in definitions.art.dungeon_nodes.values()
    )
    assert all(
        max(len(line) for line in asset.lines) >= 48
        for asset in definitions.art.dungeon_nodes.values()
    )
    assert all(
        max(len(line) for line in asset.lines) <= 72
        for asset in definitions.art.dungeon_nodes.values()
    )
    label_words = {
        "black stones",
        "bone",
        "cave",
        "cold air",
        "daylight",
        "deer",
        "fungus",
        "glass",
        "lock",
        "marks",
        "smoke",
        "warning",
        "web",
        "white stones",
    }
    dungeon_art_text = "\n".join(
        line.lower()
        for asset in definitions.art.dungeon_nodes.values()
        for line in asset.lines
    )
    assert not any(word in dungeon_art_text for word in label_words)

    data_copy = tmp_path / "data"
    shutil.copytree(default_data_dir(), data_copy)
    (data_copy / "art.yaml").unlink()

    fallback_definitions = load_game_definitions(data_copy)

    assert fallback_definitions.art.hero_classes == {}
    assert fallback_definitions.art.enemies == {}
    assert fallback_definitions.art.dungeon_nodes == {}


def test_art_data_allows_assets_without_optional_frames(tmp_path) -> None:
    data_copy = tmp_path / "data"
    shutil.copytree(default_data_dir(), data_copy)
    (data_copy / "art.yaml").write_text(
        "hero_classes:\n"
        "  watchman:\n"
        "    lines:\n"
        "      - '<#>'\n",
        encoding="utf-8",
    )

    definitions = load_game_definitions(data_copy)

    assert definitions.art.hero_classes["watchman"].lines == ["<#>"]
    assert definitions.art.hero_classes["watchman"].frames == {}
    assert definitions.art.hero_classes["watchman"].mini is None


def test_art_data_loads_list_frames_and_object_impact_frames(tmp_path) -> None:
    data_copy = tmp_path / "data"
    shutil.copytree(default_data_dir(), data_copy)
    (data_copy / "art.yaml").write_text(
        "hero_classes:\n"
        "  watchman:\n"
        "    display_name: Watchman\n"
        "    glyph: '#'\n"
        "    mini:\n"
        "      lines:\n"
        "        - '[#]'\n"
        "        - '/#>'\n"
        "        - '_/\\'\n"
        "      frames:\n"
        "        idle:\n"
        "          - lines:\n"
        "              - '[#]'\n"
        "              - '/#>'\n"
        "              - '_/\\'\n"
        "          - lines:\n"
        "              - '[#]'\n"
        "              - '/#\\'\n"
        "              - '_/>'\n"
        "    lines:\n"
        "      - '<#>'\n"
        "    frames:\n"
        "      idle:\n"
        "        - lines:\n"
        "            - '<#>'\n"
        "        - lines:\n"
        "            - '<# '\n"
        "      attack:\n"
        "        impact_frame: 2\n"
        "        frames:\n"
        "          - lines:\n"
        "              - 'wind'\n"
        "            hold: 3\n"
        "          - lines:\n"
        "              - 'swing'\n"
        "          - lines:\n"
        "              - 'impact'\n"
        "          - lines:\n"
        "              - 'recover'\n",
        encoding="utf-8",
    )

    definitions = load_game_definitions(data_copy)
    guard = definitions.art.hero_classes["watchman"]

    assert guard.display_name == "Watchman"
    assert guard.glyph == "#"
    assert guard.mini is not None
    assert guard.mini.lines == ["[#]", "/#>", "_/\\"]
    assert guard.mini.frames["idle"][1].lines == ["[#]", "/#\\", "_/>"]
    assert [frame.lines for frame in guard.frames["idle"]] == [["<#>"], ["<# "]]
    assert [frame.lines for frame in guard.frames["attack"]] == [
        ["wind"],
        ["swing"],
        ["impact"],
        ["recover"],
    ]
    assert [frame.hold for frame in guard.frames["attack"]] == [3, 2, 2, 2]
    assert guard.frame_metadata["attack"].impact_frame == 2


def test_art_data_rejects_invalid_frame_hold(tmp_path) -> None:
    data_copy = tmp_path / "data"
    shutil.copytree(default_data_dir(), data_copy)
    (data_copy / "art.yaml").write_text(
        "hero_classes:\n"
        "  watchman:\n"
        "    lines:\n"
        "      - '<#>'\n"
        "    frames:\n"
        "      attack:\n"
        "        - hold: 0\n"
        "          lines:\n"
        "            - 'wind'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hold"):
        load_game_definitions(data_copy)


def test_art_data_rejects_malformed_mini_frames(tmp_path) -> None:
    data_copy = tmp_path / "data"
    shutil.copytree(default_data_dir(), data_copy)
    (data_copy / "art.yaml").write_text(
        "hero_classes:\n"
        "  watchman:\n"
        "    mini:\n"
        "      lines:\n"
        "        - '[#]'\n"
        "        - '/#>'\n"
        "    lines:\n"
        "      - '<#>'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mini.lines must have exactly 3 lines"):
        load_game_definitions(data_copy)


def test_art_data_rejects_out_of_range_impact_frame(tmp_path) -> None:
    data_copy = tmp_path / "data"
    shutil.copytree(default_data_dir(), data_copy)
    (data_copy / "art.yaml").write_text(
        "hero_classes:\n"
        "  watchman:\n"
        "    lines:\n"
        "      - '<#>'\n"
        "    frames:\n"
        "      attack:\n"
        "        impact_frame: 3\n"
        "        frames:\n"
        "          - lines:\n"
        "              - 'wind'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="impact_frame is outside"):
        load_game_definitions(data_copy)


def test_art_data_rejects_unknown_references(tmp_path) -> None:
    data_copy = tmp_path / "data"
    shutil.copytree(default_data_dir(), data_copy)
    (data_copy / "art.yaml").write_text(
        "hero_classes:\n"
        "  unknown:\n"
        "    lines:\n"
        "      - '???'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown hero class art: unknown"):
        load_game_definitions(data_copy)
