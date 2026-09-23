"""Reading a coach's corrections back off a Pitchboard board (D104)."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from football_tracks import learn


def _board() -> dict[str, Any]:
    """Two players from video, one scene, and what the importer said about them."""
    scene = {
        "id": "scene-1",
        "carrier": "home-1",
        "positions": {"home-1": {"x": 30.0, "y": 20.0}, "away-1": {"x": 40.0, "y": 30.0}},
    }
    return {
        "teams": [
            {"players": [{"id": "home-1", "number": 1}]},
            {"players": [{"id": "away-1", "number": 1}]},
        ],
        "scenes": [scene],
        "origin": {
            "clip": "Untitled_1",
            "fps": 25,
            "scenes": {
                "scene-1": {
                    "frame": 120,
                    "carrier": "home-1",
                    "positions": {"home-1": [30.0, 20.0], "away-1": [40.0, 30.0]},
                }
            },
            "players": {
                "home-1": {"track": 7, "from": 1, "to": 300, "side": "home", "number": 1},
                "away-1": {"track": 12002, "from": 150, "to": 400, "side": "away", "number": 1},
            },
        },
    }


KNOWN = {7, 12}


def test_an_untouched_board_teaches_nothing() -> None:
    """Only a change says someone looked; a scene left alone may merely be unexamined."""
    assert learn.corrections(_board(), KNOWN).count() == 0


def test_a_changed_carrier_is_the_ball_at_that_frame() -> None:
    board = _board()
    board["scenes"][0]["carrier"] = "away-1"
    got = learn.corrections(board, KNOWN).carriers
    assert got == [
        {
            "frame": 120,
            "was": {"track": 7, "from": 1, "to": 300},
            "now": {"track": 12, "from": 150, "to": 400},
            "handPlaced": False,
            "at": [40.0, 30.0],
        }
    ]


def test_a_split_fragment_is_mapped_back_to_its_track() -> None:
    """Pitchboard numbers the n-th piece of a split track id * 1000 + n (its splitImpossible)."""
    assert learn.resolve(12002, KNOWN) == 12
    assert learn.resolve(7, KNOWN) == 7
    assert learn.resolve(99, KNOWN) is None


def test_a_number_set_labels_the_whole_track() -> None:
    board = _board()
    board["teams"][1]["players"][0]["number"] = 44
    got = learn.corrections(board, KNOWN).numbers
    assert got == [{"track": 12, "from": 150, "to": 400, "number": 44, "was": 1}]


def test_a_drag_is_a_position_and_a_nudge_is_not() -> None:
    board = _board()
    board["scenes"][0]["positions"]["home-1"] = {"x": 30.4, "y": 20.0}
    assert learn.corrections(board, KNOWN).positions == []
    board["scenes"][0]["positions"]["home-1"] = {"x": 36.0, "y": 22.0}
    got = learn.corrections(board, KNOWN).positions
    assert got == [
        {"frame": 120, "track": 7, "from": 1, "to": 300, "was": [30.0, 20.0], "now": [36.0, 22.0]}
    ]


def test_a_player_moved_across_is_a_side_and_one_deleted_is_removed() -> None:
    board = _board()
    moved = copy.deepcopy(board)
    both = [{"id": "away-1", "number": 1}, {"id": "home-1", "number": 1}]
    moved["teams"] = [{"players": []}, {"players": both}]
    got = learn.corrections(moved, KNOWN)
    assert got.sides == [{"track": 7, "from": 1, "to": 300, "side": "away", "was": "home"}]

    gone = copy.deepcopy(board)
    gone["teams"][1]["players"] = []
    assert learn.corrections(gone, KNOWN).removed == [{"track": 12, "from": 150, "to": 400}]


def test_a_scene_the_coach_added_is_not_a_frame() -> None:
    board = _board()
    board["scenes"].append(dict(board["scenes"][0], id="scene-9", carrier=None))
    assert learn.corrections(board, KNOWN).carriers == []


def test_a_track_the_file_no_longer_has_is_reported_not_guessed() -> None:
    board = _board()
    board["teams"][0]["players"][0]["number"] = 10
    got = learn.corrections(board, {12})
    assert got.numbers == []
    assert got.unresolved == [7]


def test_a_board_drawn_by_hand_has_nothing_to_learn_from() -> None:
    board = _board()
    del board["origin"]
    with pytest.raises(ValueError):
        learn.corrections(board, KNOWN)
