"""SoccerNet ground truth -> Tracks.

The conversion is where two coordinate conventions and two vocabularies meet, which
is exactly where a silent error lives. These tests are cheap because the input is a
dict; none of them need the dataset.
"""

from __future__ import annotations

from typing import Any

from football_tracks.config import PITCH_LENGTH, PITCH_WIDTH
from football_tracks.soccernet import to_tracks


def ann(
    track_id: int,
    role: str,
    team: str | None,
    *,
    x: float = 0.0,
    y: float = 0.0,
    jersey: str | None = None,
    image_id: str = "img1",
) -> dict[str, Any]:
    return {
        "id": f"a{track_id}",
        "image_id": image_id,
        "track_id": track_id,
        "category_id": 1,
        "attributes": {"role": role, "team": team, "jersey": jersey},
        "bbox_pitch": {"x_bottom_middle": x, "y_bottom_middle": y},
    }


def labels(*annotations: dict[str, Any]) -> dict[str, Any]:
    ids = sorted({a["image_id"] for a in annotations})
    return {
        "info": {"frame_rate": 25},
        "images": [
            {"image_id": i, "file_name": f"{n + 1:06d}.jpg", "width": 1920, "height": 1080}
            for n, i in enumerate(ids)
        ],
        "annotations": list(annotations),
    }


def test_centre_origin_becomes_top_left_origin() -> None:
    # SoccerNet's origin is the centre spot; Pitchboard's is the top-left corner.
    built = to_tracks(labels(ann(1, "player", "left", x=0.0, y=0.0)))
    assert built[0].samples[0].x == PITCH_LENGTH / 2
    assert built[0].samples[0].y == PITCH_WIDTH / 2


def test_corner_maps_to_the_origin() -> None:
    built = to_tracks(labels(ann(1, "player", "left", x=-52.5, y=-34.0)))
    assert built[0].samples[0].x == 0.0
    assert built[0].samples[0].y == 0.0


def test_roles_and_sides_become_team_labels() -> None:
    built = to_tracks(
        labels(
            ann(1, "player", "left"),
            ann(2, "player", "right"),
            ann(3, "goalkeeper", "left"),
            ann(4, "goalkeeper", "right"),
        )
    )
    assert [t.team for t in built] == ["home", "away", "gkHome", "gkAway"]


def test_the_ball_is_not_a_track() -> None:
    # The ball carries a pitch position and attributes exactly as a player does, so it
    # survives every guard that is not about the ball specifically (D4). It arrived as
    # two `unknown` tracks the first time this ran.
    built = to_tracks(labels(ann(1, "player", "left"), ann(2, "ball", None)))
    assert [t.id for t in built] == [1]


def test_referees_are_dropped_by_default_and_kept_on_request() -> None:
    both = labels(ann(1, "player", "left"), ann(2, "referee", None))
    assert [t.id for t in to_tracks(both)] == [1]
    kept = to_tracks(both, keep_referees=True)
    assert [(t.id, t.team) for t in kept] == [(1, "home"), (2, "referee")]


def test_positions_far_off_the_pitch_are_dropped_not_clamped() -> None:
    # A homography extrapolates without bound near the horizon; SoccerNet's own labels
    # carry positions 230 m out. Clamping would launder a failure into a coordinate the
    # reduction then fits a curve through.
    built = to_tracks(
        labels(
            ann(1, "player", "left", x=0.0, y=0.0, image_id="a"),
            ann(1, "player", "left", x=-230.0, y=-430.0, image_id="b"),
        )
    )
    assert len(built[0].samples) == 1


def test_a_track_with_no_credible_position_does_not_appear() -> None:
    assert to_tracks(labels(ann(1, "player", "left", x=-230.0, y=-430.0))) == []


def test_jersey_is_read_as_a_vote() -> None:
    # Ground truth should be unanimous, but it is read the same way stage 5's OCR vote
    # will be read, so the two paths cannot disagree about a track's number.
    built = to_tracks(
        labels(
            ann(1, "player", "left", jersey="9", image_id="a"),
            ann(1, "player", "left", jersey="9", image_id="b"),
            ann(1, "player", "left", jersey=None, image_id="c"),
        )
    )
    assert built[0].number == 9
    assert built[0].number_confidence == 1.0


def test_an_unread_number_is_none_and_never_a_guess() -> None:
    built = to_tracks(labels(ann(1, "player", "left", jersey=None)))
    assert built[0].number is None
    assert built[0].number_confidence is None


def test_samples_are_ordered_by_frame() -> None:
    built = to_tracks(
        labels(
            ann(1, "player", "left", image_id="c"),
            ann(1, "player", "left", image_id="a"),
            ann(1, "player", "left", image_id="b"),
        )
    )
    frames = [s["f"] for s in built[0].to_json()["samples"]]
    assert frames == sorted(frames)
