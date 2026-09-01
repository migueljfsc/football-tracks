"""Stage 1 geometry, against a synthetic camera.

A homography fitted from real annotations can only be judged by looking at it, which is
what `ft calibrate --frame` is for. What CAN be tested is the algebra: invent a camera,
project the markings through it, and check the fit recovers what was invented.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from football_tracks import calibration
from football_tracks.config import PITCH_LENGTH, PITCH_WIDTH

WIDTH, HEIGHT = 1920, 1080

# An oblique view, of the kind that broke the intersection-based fit: the pitch fills a
# trapezoid with a far edge much shorter than the near one.
PITCH_TO_IMAGE: npt.NDArray[np.float64] = np.asarray(
    cv2.getPerspectiveTransform(
        np.array(
            [[0, 0], [PITCH_LENGTH, 0], [PITCH_LENGTH, PITCH_WIDTH], [0, PITCH_WIDTH]], np.float32
        ),
        np.array([[620, 300], [1480, 315], [1880, 1010], [80, 960]], np.float32),
    ),
    dtype=np.float64,
)

# Pitch extents of the markings the tests draw on, in metres.
EXTENTS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "Side line left": ((0, 0), (0, PITCH_WIDTH)),
    "Side line right": ((PITCH_LENGTH, 0), (PITCH_LENGTH, PITCH_WIDTH)),
    "Side line top": ((0, 0), (PITCH_LENGTH, 0)),
    "Side line bottom": ((0, PITCH_WIDTH), (PITCH_LENGTH, PITCH_WIDTH)),
    "Middle line": ((PITCH_LENGTH / 2, 0), (PITCH_LENGTH / 2, PITCH_WIDTH)),
    "Big rect. left main": ((16.5, 13.84), (16.5, 54.16)),
    "Big rect. left top": ((0, 13.84), (16.5, 13.84)),
    "Big rect. left bottom": ((0, 54.16), (16.5, 54.16)),
    "Small rect. left main": ((5.5, 24.84), (5.5, 43.16)),
    "Small rect. left top": ((0, 24.84), (5.5, 24.84)),
}


def annotate(
    names: list[str], *, n: int = 6, noise: float = 0.0
) -> dict[str, list[dict[str, float]]]:
    """Project pitch markings through the synthetic camera into normalised image points."""
    rng = np.random.default_rng(0)
    out: dict[str, list[dict[str, float]]] = {}
    for name in names:
        (x0, y0), (x1, y1) = EXTENTS[name]
        t = np.linspace(0, 1, n)
        pts = np.stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)], axis=1)
        img = calibration.apply(PITCH_TO_IMAGE, pts)
        if noise:
            img = img + rng.normal(0, noise, img.shape)
        out[name] = [{"x": float(p[0] / WIDTH), "y": float(p[1] / HEIGHT)} for p in img]
    return out


def max_error(h: npt.NDArray[np.float64]) -> float:
    """Worst position error, in metres, over a grid covering the pitch."""
    gx, gy = np.meshgrid(np.linspace(2, PITCH_LENGTH - 2, 8), np.linspace(2, PITCH_WIDTH - 2, 6))
    want = np.stack([gx.ravel(), gy.ravel()], axis=1)
    got = calibration.apply(h, calibration.apply(PITCH_TO_IMAGE, want))
    return float(np.max(np.linalg.norm(got - want, axis=1)))


ALL = list(EXTENTS)


def test_recovers_the_camera_it_was_given() -> None:
    h = calibration.homography(annotate(ALL), WIDTH, HEIGHT)
    assert h is not None
    assert max_error(h) < 0.01


def test_a_realistic_subset_of_visible_lines_is_enough() -> None:
    # What a frame actually shows: one end of the pitch and no halfway line.
    visible = [
        "Side line left",
        "Side line top",
        "Side line bottom",
        "Big rect. left main",
        "Big rect. left top",
        "Big rect. left bottom",
    ]
    h = calibration.homography(annotate(visible), WIDTH, HEIGHT)
    assert h is not None
    assert max_error(h) < 0.01


def test_four_lines_are_refused_even_though_they_determine_a_homography() -> None:
    # Eight constraints for eight degrees of freedom fits perfectly and cannot be
    # checked. Measured on real footage those frames land 21.7 m out, so the fit is
    # refused rather than returned - see MIN_LINES.
    four = ["Side line left", "Middle line", "Side line top", "Side line bottom"]
    assert calibration.homography(annotate(four), WIDTH, HEIGHT) is None


def test_lines_all_running_the_same_way_are_refused() -> None:
    # Parallel markings pin down only half the transform, however many there are.
    parallel = ["Side line left", "Middle line", "Side line right", "Big rect. left main"]
    assert calibration.homography(annotate(parallel), WIDTH, HEIGHT) is None


def test_unknown_line_names_are_ignored_rather_than_guessed_at() -> None:
    lines = annotate(ALL)
    lines["Circle central"] = [{"x": 0.5, "y": 0.5}, {"x": 0.6, "y": 0.5}]
    lines["Goal left crossbar"] = [{"x": 0.3, "y": 0.2}, {"x": 0.4, "y": 0.2}]
    h = calibration.homography(lines, WIDTH, HEIGHT)
    assert h is not None
    # A crossbar is metres above the grass; treating it as a ground line would drag the
    # whole fit. The result must be unchanged from the ground lines alone.
    assert max_error(h) < 0.01


def test_a_frame_showing_almost_nothing_gets_no_homography() -> None:
    assert calibration.homography(annotate(["Side line left"], n=3), WIDTH, HEIGHT) is None
    assert calibration.homography({}, WIDTH, HEIGHT) is None


@pytest.mark.parametrize("noise", [0.5, 2.0])
def test_pixel_noise_stays_proportionate(noise: float) -> None:
    # Over-determined by design, so noise averages out instead of passing through.
    h = calibration.homography(annotate(ALL, n=12, noise=noise), WIDTH, HEIGHT)
    assert h is not None
    assert max_error(h) < 1.0 * noise


def test_apply_round_trips_through_the_inverse() -> None:
    h = calibration.homography(annotate(ALL), WIDTH, HEIGHT)
    assert h is not None
    pts = np.array([[10.0, 10.0], [52.5, 34.0], [95.0, 60.0]])
    there = calibration.apply(np.linalg.inv(h), pts)
    back = calibration.apply(h, there)
    assert np.allclose(back, pts, atol=1e-6)
