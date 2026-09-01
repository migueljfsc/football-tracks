"""The human seed: clicked landmarks -> a camera model."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from football_tracks import calibration, seed
from football_tracks.config import PITCH_LENGTH, PITCH_WIDTH

PITCH_TO_IMAGE = np.asarray(
    cv2.getPerspectiveTransform(
        np.array(
            [[0, 0], [PITCH_LENGTH, 0], [PITCH_LENGTH, PITCH_WIDTH], [0, PITCH_WIDTH]], np.float32
        ),
        np.array([[620, 300], [1480, 315], [1880, 1010], [80, 960]], np.float32),
    ),
    dtype=np.float64,
)


def clicked(names: list[str], *, jitter: float = 0.0) -> seed.Seed:
    rng = np.random.default_rng(0)
    pts = []
    for n in names:
        pitch = seed.LANDMARKS[n]
        img = calibration.apply(PITCH_TO_IMAGE, np.array([pitch]))[0]
        if jitter:
            img = img + rng.normal(0, jitter, 2)
        pts.append(((float(img[0]), float(img[1])), pitch))
    return seed.Seed(frame=1, points=pts)


# Spread across the pitch on purpose. The obvious four - both posts and both corners -
# all sit on x = 0 and are degenerate; `test_landmarks_along_one_line_are_refused`
# pins that.
SIX = [
    "goal post far",
    "goal post near",
    "penalty box front far",
    "penalty box front near",
    "6yd front far",
    "penalty spot",
]


def test_clicked_landmarks_recover_the_camera() -> None:
    h = seed.homography(clicked(SIX))
    assert h is not None
    got = calibration.apply(h, calibration.apply(PITCH_TO_IMAGE, np.array([[52.5, 34.0]])))
    assert got[0] == pytest.approx([52.5, 34.0], abs=0.01)


def test_four_points_is_the_minimum() -> None:
    assert seed.homography(clicked(SIX[:3])) is None
    assert seed.homography(clicked(SIX[:4])) is not None


def test_a_misclick_is_absorbed_when_there_are_spare_points() -> None:
    # Four points fit whatever was misclicked and cannot be checked - the same trap as
    # D17 one level up. With six, RANSAC has something to disagree with.
    pts = clicked(SIX).points
    (ix, iy), pitch = pts[2]
    pts[2] = ((ix + 60.0, iy - 40.0), pitch)
    h = seed.homography(seed.Seed(frame=1, points=pts))
    assert h is not None
    got = calibration.apply(h, calibration.apply(PITCH_TO_IMAGE, np.array([[52.5, 34.0]])))
    assert got[0] == pytest.approx([52.5, 34.0], abs=1.5)


def test_landmarks_along_one_line_are_refused() -> None:
    # Both posts and both corners of a goal are the four most natural things to click,
    # and all four sit on x = 0. That fits perfectly and describes nothing, so it is
    # refused rather than returned - the same call as D17.
    on_the_goal_line = ["goal post far", "goal post near", "corner far", "corner near"]
    assert seed.homography(clicked(on_the_goal_line)) is None


def test_the_far_goal_is_the_same_landmark_mirrored() -> None:
    # So a coach never has to think about which end the pitch model calls zero.
    assert seed.LANDMARKS["goal post far"][0] == 0.0
    assert seed.mirrored("goal post far")[0] == PITCH_LENGTH
    assert seed.mirrored("goal post far")[1] == seed.LANDMARKS["goal post far"][1]


def test_a_seed_survives_a_round_trip(tmp_path: Path) -> None:
    original = clicked(SIX)
    seed.write(tmp_path / "seed.json", original)
    back = seed.read(tmp_path / "seed.json")
    assert back.frame == original.frame
    assert len(back.points) == len(original.points)
    assert back.points[0][1] == original.points[0][1]


def flipped(s: seed.Seed) -> seed.Seed:
    return seed.Seed(
        frame=s.frame, points=[(i, (px, PITCH_WIDTH - py)) for i, (px, py) in s.points]
    )


def test_a_correctly_seeded_clip_agrees_with_the_camera() -> None:
    # The synthetic camera has the near touchline at the bottom of the frame, which is
    # what a broadcast camera on a touchline always gives.
    assert seed.orientation(clicked(SIX)) > seed.ORIENTATION_CONFIDENT


def test_swapped_far_and_near_is_detected() -> None:
    # The failure the reprojection overlay is blind to: a pitch is symmetric about the
    # halfway line, so a y-mirrored model lands on the real markings perfectly and only
    # the arithmetic can tell.
    assert seed.orientation(flipped(clicked(SIX))) < -seed.ORIENTATION_CONFIDENT


def test_flipping_restores_the_orientation() -> None:
    assert seed.orientation(seed.flip_y(flipped(clicked(SIX)))) > seed.ORIENTATION_CONFIDENT


def test_flipping_keeps_the_clicks_and_moves_only_the_pitch_side() -> None:
    original = clicked(SIX)
    turned = seed.flip_y(original)
    assert [p[0] for p in turned.points] == [p[0] for p in original.points]
    assert [p[1][0] for p in turned.points] == [p[1][0] for p in original.points]


def traced(name: str, n: int = 6) -> list[tuple[tuple[float, float], tuple[float, float, float]]]:
    """Points along a named marking, projected through the synthetic camera."""
    a, b, c = seed.TRACEABLE[name]
    out = []
    for t in np.linspace(0.15, 0.85, n):
        p = (-c / a, t * PITCH_WIDTH) if abs(a) > abs(b) else (t * PITCH_LENGTH, -c / b)
        img = calibration.apply(PITCH_TO_IMAGE, np.array([p]))[0]
        out.append(((float(img[0]), float(img[1])), (a, b, c)))
    return out


def test_tracing_two_crossing_markings_recovers_the_camera() -> None:
    # What a tight goalmouth shot actually offers: long clear lines whose corners are
    # off screen. Clicking anywhere along them is enough.
    s = seed.Seed(
        frame=1,
        points=[],
        lines=traced("goal line")
        + traced("penalty box front")
        + traced("far touchline")
        + traced("near touchline"),
    )
    h = seed.homography(s)
    assert h is not None
    got = calibration.apply(h, calibration.apply(PITCH_TO_IMAGE, np.array([[52.5, 34.0]])))
    assert got[0] == pytest.approx([52.5, 34.0], abs=0.05)


def test_tracing_only_parallel_markings_is_refused() -> None:
    # Three lines all parallel to the goal line leave the camera free to slide along
    # the pitch. The fit would come back looking like any other matrix.
    s = seed.Seed(
        frame=1,
        points=[],
        lines=traced("goal line") + traced("6yd box front") + traced("penalty box front"),
    )
    assert seed.homography(s) is None


def test_landmarks_and_traces_combine() -> None:
    s = seed.Seed(
        frame=1,
        points=[clicked(["goal post far"]).points[0], clicked(["goal post near"]).points[0]],
        lines=traced("penalty box front")
        + traced("penalty box near side")
        + traced("far touchline"),
    )
    h = seed.homography(s)
    assert h is not None
    got = calibration.apply(h, calibration.apply(PITCH_TO_IMAGE, np.array([[30.0, 40.0]])))
    assert got[0] == pytest.approx([30.0, 40.0], abs=0.5)


def test_two_traced_lines_alone_are_refused() -> None:
    # Two lines always cross, and a homography sending the whole image to that crossing
    # satisfies every point-on-line constraint exactly. It fits with zero residuals,
    # which is the most convincing way to be wrong.
    s = seed.Seed(
        frame=1,
        points=[],
        lines=traced("penalty box front") + traced("penalty box near side"),
    )
    assert seed.homography(s) is None


def test_a_traced_seed_can_be_flipped_and_round_tripped(tmp_path: Path) -> None:
    s = seed.Seed(frame=1, points=[], lines=traced("goal line") + traced("far touchline"))
    seed.write(tmp_path / "s.json", s)
    back = seed.read(tmp_path / "s.json")
    assert len(back.lines) == len(s.lines)
    # Flipping y must move the marking to the mirrored side, not leave it be.
    flipped_line = seed.flip_y(s).lines[-1][1]
    assert flipped_line != s.lines[-1][1]
