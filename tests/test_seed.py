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
